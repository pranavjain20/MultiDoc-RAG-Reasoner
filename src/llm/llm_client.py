"""
LLM Client with Groq (smart) + HuggingFace API + local transformers fallback.

Priority:
1) Groq chat completion (if GROQ_API_KEY is set)
2) HuggingFace Inference API (if HF_TOKEN is set and endpoint works)
3) Local transformers (always works)
"""

import os
import time
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner

# Smart external model (your own stronger setup)
from llm.llm_api_groq import generate_llm_response


class LLMClient:
    """Client for multi-backend text generation."""

    def __init__(
        self,
        model_id: str = "google/flan-t5-large",
        local_model_id: str = "google/flan-t5-base",
        groq_model_name: Optional[str] = None,
    ) -> None:
        load_dotenv()

        # Groq (smart backend)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model_name = groq_model_name  # optional override

        # HuggingFace API (secondary)
        self.hf_token = os.getenv("HF_TOKEN")
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.hf_headers = (
            {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        )

        # Defaults
        self.default_temperature = 0.2
        self.default_max_tokens = 512

        # Local fallback
        self.local_model_id = local_model_id
        self._local_pipeline = None
        self._local_tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 2,
    ) -> str:
        """
        Generate text from a single prompt string.

        Note: In our project, `prompt` is often already a structured prompt
        produced by `MultiDocReasoner.build_prompt(...)`.
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # Construct full prompt for API-style generation
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"Question: {prompt}\n\nAnswer:"

        # 1) Try Groq (smart backend) if key exists
        if self.groq_api_key:
            try:
                # Groq API is chat-based; we feed the whole prompt as "question"
                # with empty context (since `prompt` already contains context).
                return generate_llm_response(
                    question=full_prompt,
                    context=None,
                    model_name=self.groq_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception:
                # fall through to HF/local
                pass

        # 2) Try HuggingFace Inference API
        api_result = self._generate_via_hf_api(
            full_prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        if api_result is not None:
            return api_result

        # 3) Local fallback (short prompt to avoid instruction echo)
        local_prompt = f"Question: {prompt}\n\nAnswer:"
        return self._generate_via_local_model(local_prompt, max_tokens)

    def generate_with_reasoning(
        self,
        question: str,
        chunks: List[Union[Dict[str, str], str]],
        reasoner: MultiDocReasoner,
    ) -> Dict[str, Any]:
        """
        High-level wrapper used by UI and evaluation.

        Strategy:
        - Keep reasoner for query_type (and fallback prompt building).
        - Preferred: Groq (question + grouped context).
        - Fallback: existing HF/local using structured prompt.
        """
        prompt, query_type = reasoner.build_prompt(question, chunks)

        # Build grouped context for Groq (better for chat models)
        chunks_by_doc = reasoner.organize_chunks_by_doc(chunks)

        context_parts: List[str] = []
        for doc_name, doc_chunks in chunks_by_doc.items():
            context_parts.append(f"--- {doc_name} ---")

            for c in doc_chunks:
                # IMPORTANT: doc_chunks might be a list of dicts OR a list of strings
                if isinstance(c, dict):
                    txt = (c.get("text") or "").strip()
                else:
                    txt = str(c).strip()

                if txt:
                    context_parts.append(txt)

        context = "\n\n".join(context_parts).strip()

        # 1) Primary: Groq if available
        if self.groq_api_key:
            try:
                response_text = generate_llm_response(
                    question=question,
                    context=context,
                    model_name=self.groq_model_name,
                    temperature=self.default_temperature,
                    max_tokens=self.default_max_tokens,
                    system_prompt=SYSTEM_PROMPT,
                )
                return {"response": response_text, "query_type": query_type}
            except Exception:
                # fall through to HF/local
                pass

        # 2) Fallback: existing HF/local path using structured prompt
        response_text = self.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return {"response": response_text, "query_type": query_type}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_via_hf_api(
        self,
        full_prompt: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
    ) -> Optional[str]:
        """Try HuggingFace Inference API. Return None if unavailable."""
        if not self.hf_token:
            return None

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }

        for _ in range(max_retries):
            try:
                r = requests.post(
                    self.api_url,
                    headers=self.hf_headers,
                    json=payload,
                    timeout=60,
                )

                if r.status_code == 200:
                    out = r.json()
                    if isinstance(out, list) and out:
                        return out[0].get("generated_text", "").strip()
                    return str(out).strip()

                # Known transient / policy / routing errors → fallback
                if r.status_code in {404, 410, 429, 503}:
                    return None

                return None

            except requests.RequestException:
                time.sleep(2)
                continue

        return None

    def _generate_via_local_model(self, prompt: str, max_tokens: int) -> str:
        """
        Local transformers fallback.

        Key fix:
        - Truncate inputs to avoid exceeding model max length.
        """
        if self._local_pipeline is None or self._local_tokenizer is None:
            from transformers import AutoTokenizer, pipeline

            self._local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id)
            self._local_pipeline = pipeline(
                "text2text-generation",
                model=self.local_model_id,
            )

        # Keep buffer under typical max length (works well for flan-t5-base too)
        inputs = self._local_tokenizer(
            prompt,
            truncation=True,
            max_length=768,  # base can handle more than small; still keep safe
            return_tensors="pt",
        )
        truncated_prompt = self._local_tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=True,
        )

        out = self._local_pipeline(
            truncated_prompt,
            max_new_tokens=min(max_tokens, 256),
            do_sample=False,
        )

        if isinstance(out, list) and out:
            return out[0].get("generated_text", "").strip()
        return str(out)



