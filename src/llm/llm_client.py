"""
LLM Client with HuggingFace Hub InferenceClient + local transformers fallback.

Priority:
1) Hugging Face Inference (via huggingface_hub.InferenceClient) if HF_TOKEN is set and call succeeds
2) Local transformers fallback (always works)
"""

import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner


class LLMClient:
    """Client for multi-backend text generation (HF -> Local)."""

    def __init__(
        self,
        model_id: str = "google/flan-t5-large",
        local_model_id: str = "google/flan-t5-base",
    ) -> None:
        load_dotenv()

        # HF Inference
        self.hf_token = os.getenv("HF_TOKEN")  # optional
        self.model_id = model_id
        self._hf_client = None  # lazy init

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

        Notes:
        - In this repo, `prompt` is often already a structured prompt produced by MultiDocReasoner.
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # HF call: keep system prompt in-text (works with T5-style text2text)
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"Question: {prompt}\n\nAnswer:"

        # 1) Try HF (only if token exists)
        hf_out = self._generate_via_hf(full_prompt, temperature=temperature, max_tokens=max_tokens)
        if hf_out is not None and hf_out.strip():
            return hf_out.strip()

        # 2) Local fallback: shorter prompt reduces “instruction echo”
        local_prompt = f"Question: {prompt}\n\nAnswer:"
        return self._generate_via_local_model(local_prompt, max_tokens=max_tokens)

    def generate_with_reasoning(
        self,
        question: str,
        chunks: List[Union[Dict[str, str], str]],
        reasoner: MultiDocReasoner,
    ) -> Dict[str, Any]:
        """High-level wrapper used by UI and evaluation."""
        prompt, query_type = reasoner.build_prompt(question, chunks)
        response_text = self.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return {"response": response_text, "query_type": query_type}

    # ------------------------------------------------------------------
    # HF inference (huggingface_hub)
    # ------------------------------------------------------------------
    def _get_hf_client(self):
        if self._hf_client is None:
            from huggingface_hub import InferenceClient

            # Force provider to "hf-inference" so it uses the new Inference Providers / router logic.
            # (This avoids you hand-crafting router URLs that may 404/410.)
            self._hf_client = InferenceClient(
                model=self.model_id,
                provider="hf-inference",
                token=self.hf_token,
                timeout=60,
            )
        return self._hf_client

    def _generate_via_hf(self, full_prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        if not self.hf_token:
            return None

        try:
            client = self._get_hf_client()

            # flan-t5-* is text2text (seq2seq). However, not all providers expose a dedicated
            # text2text endpoint consistently; text_generation is the most broadly supported.
            # It accepts `truncate` which helps with long prompts.
            out = client.text_generation(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
                truncate=2048,  # prevents provider-side errors on long prompts
            )

            # huggingface_hub returns either `str` or a structured output depending on backend.
            if isinstance(out, str):
                return out
            # Sometimes it's a dataclass-like object with .generated_text
            if hasattr(out, "generated_text"):
                return getattr(out, "generated_text") or ""
            return str(out)

        except Exception:
            # Any HF error -> fallback to local
            return None

    # ------------------------------------------------------------------
    # Local transformers fallback
    # ------------------------------------------------------------------
    def _generate_via_local_model(self, prompt: str, max_tokens: int) -> str:
        """
        Local transformers fallback.

        Key fix:
        - Truncate inputs to avoid exceeding model max length (T5 family tends to be strict).
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

        return str(out).strip()


