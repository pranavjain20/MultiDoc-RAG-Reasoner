"""
LLM Client with HuggingFace API + local transformers fallback.
"""

import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner


class LLMClient:
    """
    Client for text generation.

    Priority:
    1. HuggingFace Inference API (if available)
    2. Local transformers fallback (always available)
    """

    def __init__(
        self,
        model_id: str = "google/flan-t5-large",
        local_model_id: str = "google/flan-t5-small",
    ) -> None:
        load_dotenv()

        self.token = os.getenv("HF_TOKEN")
        self.model_id = model_id
        self.local_model_id = local_model_id

        # ---- HuggingFace public inference API (correct endpoint) ----
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = (
            {"Authorization": f"Bearer {self.token}"} if self.token else {}
        )

        self.default_temperature = 0.3
        self.default_max_tokens = 512

        # ---- Lazy-loaded local model ----
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
        Generate text. Try HF API first; fallback to local model if it fails.
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # Full prompt used for HF API (if it works)
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"Question: {prompt}\n\nAnswer:"

        # 1) Try HuggingFace API
        api_result = self._generate_via_hf_api(
            full_prompt, temperature, max_tokens, max_retries
        )
        if api_result is not None:
            return api_result

        # 2) Local fallback:
        # Avoid huge system prompt (prevents instruction-echo),
        # keep it short so flan-t5-small answers instead of repeating rules.
        local_prompt = f"Question: {prompt}\n\nAnswer:"
        return self._generate_via_local_model(local_prompt, max_tokens)

    def generate_with_reasoning(
        self,
        question: str,
        chunks: List[Dict[str, str]],
        reasoner: MultiDocReasoner,
    ) -> Dict[str, Any]:
        """
        High-level wrapper used by UI and evaluation.
        """
        prompt, query_type = reasoner.build_prompt(question, chunks)
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
        """
        Try HuggingFace Inference API.
        Returns None if unavailable.
        """
        if not self.token:
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
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )

                if r.status_code == 200:
                    out = r.json()
                    if isinstance(out, list) and out:
                        return out[0].get("generated_text", "").strip()
                    return str(out).strip()

                # Known transient / policy errors → fallback
                if r.status_code in {404, 410, 429, 503}:
                    return None

                return None

            except requests.RequestException:
                time.sleep(2)
                continue

        return None

    def _generate_via_local_model(self, prompt: str, max_tokens: int) -> str:
        """
        Local transformers fallback (always works).

        Key fix:
        - Truncate inputs to avoid exceeding flan-t5-small max length (512).
        """
        if self._local_pipeline is None or self._local_tokenizer is None:
            from transformers import AutoTokenizer, pipeline

            self._local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id)
            self._local_pipeline = pipeline(
                "text2text-generation",
                model=self.local_model_id,
            )

        # flan-t5-small typically supports 512 input tokens; keep buffer
        inputs = self._local_tokenizer(
            prompt,
            truncation=True,
            max_length=480,
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


