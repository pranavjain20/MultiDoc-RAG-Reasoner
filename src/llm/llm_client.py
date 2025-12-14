"""
LLM Client with Groq (smart) + (optional) HuggingFace + local transformers fallback.

Priority:
1) Groq chat completion (if GROQ_API_KEY is set)
2) HuggingFace (optional; if you later wire a working endpoint)
3) Local transformers fallback (always works)
"""

import os
import time
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner
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

        # HF (optional / best-effort only)
        self.hf_token = os.getenv("HF_TOKEN")
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.hf_headers = (
            {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        )

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
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # Reasoner often already gives a structured prompt; keep it.
        full_prompt = (
            f"System: {system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
            if system_prompt
            else f"Question: {prompt}\n\nAnswer:"
        )

        # 1) Groq (preferred)
        if self.groq_api_key:
            try:
                return generate_llm_response(
                    question=prompt,          # pass the reasoner prompt as the "question"
                    context=None,             # prompt already includes context
                    model_name=self.groq_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt or SYSTEM_PROMPT,
                )
            except Exception:
                pass  # fall through

        # 2) HF (best-effort only; your curl shows this may be dead / 410)
        api_result = self._generate_via_hf_api(
            full_prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        if api_result is not None:
            return api_result

        # 3) Local fallback
        local_prompt = f"Question: {prompt}\n\nAnswer:"
        return self._generate_via_local_model(local_prompt, max_tokens)

    def generate_with_reasoning(
        self,
        question: str,
        chunks: List[Union[Dict[str, str], str]],
        reasoner: MultiDocReasoner,
    ) -> Dict[str, Any]:
        prompt, query_type = reasoner.build_prompt(question, chunks)

        # Better “smart” path: give Groq a clean context string
        chunks_by_doc = reasoner.organize_chunks_by_doc(chunks)
        context_parts: List[str] = []
        for doc_name, doc_chunks in chunks_by_doc.items():
            context_parts.append(f"--- {doc_name} ---")
            for c in doc_chunks:
                if isinstance(c, dict):
                    txt = (c.get("text") or "").strip()
                else:
                    txt = str(c).strip()
                if txt:
                    context_parts.append(txt)
        context = "\n\n".join(context_parts).strip()

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
                pass

        # fallback: use whatever generate() can do
        response_text = self.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return {"response": response_text, "query_type": query_type}

    # ------------------------------------------------------------------
    # HF (best-effort)
    # ------------------------------------------------------------------
    def _generate_via_hf_api(
        self,
        full_prompt: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
    ) -> Optional[str]:
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

                # 410/404/503/429 -> give up and fallback
                if r.status_code in {404, 410, 429, 503}:
                    return None

                if r.status_code == 200:
                    out = r.json()
                    if isinstance(out, list) and out and isinstance(out[0], dict):
                        return (out[0].get("generated_text") or "").strip()
                    if isinstance(out, dict) and "generated_text" in out:
                        return (out.get("generated_text") or "").strip()
                    if isinstance(out, str):
                        return out.strip()
                    return None

                return None

            except requests.RequestException:
                time.sleep(1.5)
                continue

        return None

    # ------------------------------------------------------------------
    # Local fallback
    # ------------------------------------------------------------------
    def _generate_via_local_model(self, prompt: str, max_tokens: int) -> str:
        if self._local_pipeline is None or self._local_tokenizer is None:
            from transformers import AutoTokenizer, pipeline

            self._local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id)
            self._local_pipeline = pipeline(
                "text2text-generation",
                model=self.local_model_id,
            )

        inputs = self._local_tokenizer(
            prompt,
            truncation=True,
            max_length=768,
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

        if isinstance(out, list) and out and isinstance(out[0], dict):
            return (out[0].get("generated_text") or "").strip()
        return str(out)




