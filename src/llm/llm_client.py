"""
LLM Client with Groq (smart) + HuggingFace Hub Inference + local transformers fallback.

Priority:
1) Groq chat completion (if GROQ_API_KEY is set)
2) HuggingFace Inference (if HF_TOKEN is set and works)
3) Local transformers (always works)
"""

import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner
from llm.llm_api_groq import generate_llm_response


class LLMClient:
    """Client for multi-backend text generation."""

    def __init__(
        self,
        hf_model_id: str = "google/flan-t5-large",
        local_model_id: str = "google/flan-t5-base",
        groq_model_name: Optional[str] = None,
    ) -> None:
        load_dotenv()

        # Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model_name = groq_model_name  # optional override

        # HF (optional)
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_model_id = hf_model_id
        self._hf_client = None  # lazy

        # Local
        self.local_model_id = local_model_id
        self._local_pipeline = None
        self._local_tokenizer = None

        self.default_temperature = 0.2
        self.default_max_tokens = 512

    # ------------------------------------------------------
    # Public API
    # ------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # Note: for your project, `prompt` often already contains context/instructions
        sys = system_prompt or ""
        full_prompt = (
            f"{sys}\n\n{prompt}".strip() if sys else prompt.strip()
        )

        # 1) Groq (smart)
        if self.groq_api_key:
            try:
                # Here we treat full_prompt as "question" with empty context,
                # because prompt may already include retrieved chunks.
                return generate_llm_response(
                    question=full_prompt,
                    context=None,
                    model_name=self.groq_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=SYSTEM_PROMPT,
                )
            except Exception:
                pass  # fallback

        # 2) HF hub inference (optional)
        hf_out = self._generate_via_hf(full_prompt, max_tokens)
        if hf_out is not None and hf_out.strip():
            return hf_out.strip()

        # 3) Local
        return self._generate_via_local_model(full_prompt, max_tokens)

    def generate_with_reasoning(
        self,
        question: str,
        chunks: List[Dict[str, str]],
        reasoner: MultiDocReasoner,
    ) -> Dict[str, Any]:
        """
        - Keep reasoner for query_type + prompt building.
        - For Groq: pass (question + grouped context) which chat models like.
        - For HF/local: use reasoner-built prompt.
        """
        prompt, query_type = reasoner.build_prompt(question, chunks)

        # Build grouped context for Groq
        context_parts: List[str] = []
        chunks_by_doc = reasoner.organize_chunks_by_doc(chunks)  # doc -> list[str]
        for doc_name, texts in chunks_by_doc.items():
            context_parts.append(f"--- {doc_name} ---")
            for t in texts:
                t = (t or "").strip()
                if t:
                    context_parts.append(t)
        context = "\n\n".join(context_parts).strip()

        # 1) Groq: best answer quality
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
                pass  # fallback

        # 2) HF/local path: use reasoner prompt
        response_text = self.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return {"response": response_text, "query_type": query_type}

    # ------------------------------------------------------
    # HF hub inference
    # ------------------------------------------------------
    def _get_hf_client(self):
        if self._hf_client is None:
            from huggingface_hub import InferenceClient

            # Let huggingface_hub handle backend routing/protocol changes.
            self._hf_client = InferenceClient(
                model=self.hf_model_id,
                token=self.hf_token,
                timeout=60,
            )
        return self._hf_client

    def _generate_via_hf(self, prompt: str, max_tokens: int) -> Optional[str]:
        if not self.hf_token:
            return None

        try:
            client = self._get_hf_client()

            # Some HF backends may raise / return weird structures;
            # normalize to string.
            out = client.text_generation(
                prompt=prompt,
                max_new_tokens=min(max_tokens, 256),
                temperature=0.2,
                do_sample=False,
                return_full_text=False,
            )
            if out is None:
                return None
            return str(out)

        except Exception:
            return None

    # ------------------------------------------------------
    # Local transformers
    # ------------------------------------------------------
    def _generate_via_local_model(self, prompt: str, max_tokens: int) -> str:
        if self._local_pipeline is None or self._local_tokenizer is None:
            from transformers import AutoTokenizer, pipeline

            self._local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id)
            self._local_pipeline = pipeline(
                "text2text-generation",
                model=self.local_model_id,
            )

        # keep inputs bounded (avoid flan max length issues)
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



