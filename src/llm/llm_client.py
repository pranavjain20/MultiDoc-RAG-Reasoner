"""
LLM Client with Groq (smart) + local transformers fallback.

Priority:
1) Groq chat completion (if GROQ_API_KEY is set)
2) Local transformers fallback (always works)
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner
from llm.llm_api_groq import generate_llm_response


class LLMClient:
    """Client for multi-backend text generation (Groq → Local)."""

    def __init__(
        self,
        local_model_id: str = "google/flan-t5-base",
        groq_model_name: Optional[str] = None,
    ) -> None:
        load_dotenv()

        # Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model_name = groq_model_name  # optional override

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
    ) -> str:
        """
        Generate text from a single prompt string.

        In this project, `prompt` is often already a structured prompt
        produced by MultiDocReasoner.build_prompt(...).
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # 1) Groq (preferred)
        if self.groq_api_key:
            try:
                # We treat `prompt` as the question body.
                # If you pass a structured prompt, context=None is fine.
                return generate_llm_response(
                    question=prompt,
                    context=None,
                    model_name=self.groq_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=(system_prompt or SYSTEM_PROMPT),
                )
            except Exception:
                # fall through to local
                pass

        # 2) Local fallback
        local_prompt = self._wrap_as_t5_prompt(prompt, system_prompt=system_prompt)
        return self._generate_via_local_model(local_prompt, max_tokens=max_tokens)

    def generate_with_reasoning(
        self,
        question: str,
        chunks: List[Dict[str, str]],
        reasoner: MultiDocReasoner,
    ) -> Dict[str, Any]:
        """
        High-level wrapper used by UI and evaluation.
        """
        # reasoner still useful for query_type
        _structured_prompt, query_type = reasoner.build_prompt(question, chunks)

        # Build grouped context for Groq (chat models do much better with clean context)
        # NOTE: organize_chunks_by_doc returns Dict[str, List[str]]  (list of TEXT STRINGS)
        chunks_by_doc = reasoner.organize_chunks_by_doc(chunks)

        context_parts: List[str] = []
        for doc_name, doc_text_list in chunks_by_doc.items():
            context_parts.append(f"--- {doc_name} ---")
            for txt in doc_text_list:
                t = (txt or "").strip()
                if t:
                    context_parts.append(t)

        context = "\n\n".join(context_parts).strip()

        # 1) Groq path: question + context
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
                # fall through to local
                pass

        # 2) Local fallback: keep it short and safe
        local_prompt = self._wrap_as_t5_prompt(
            f"Context:\n{context}\n\nQuestion: {question}",
            system_prompt=None,  # avoid long rule blocks for small T5
        )
        response_text = self._generate_via_local_model(local_prompt, max_tokens=256)
        return {"response": response_text, "query_type": query_type}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _wrap_as_t5_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        # For T5-like local models, keep formatting simple
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}\n\nAnswer:"
        return f"{prompt}\n\nAnswer:"

    def _generate_via_local_model(self, prompt: str, max_tokens: int) -> str:
        """
        Local transformers fallback.

        Truncation is important to avoid exceeding model max length.
        """
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


