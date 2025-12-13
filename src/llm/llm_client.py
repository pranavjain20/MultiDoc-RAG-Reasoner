"""
LLM Client with HuggingFace Router API + local transformers fallback.

Priority:
1) HuggingFace Router (if HF_TOKEN is set and endpoint works)
2) Local transformers fallback (always works)
"""

import os
import time
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner


class LLMClient:
    """Client for multi-backend text generation (HF Router → Local)."""

    def __init__(
        self,
        model_id: str = "google/flan-t5-large",
        local_model_id: str = "google/flan-t5-base",
    ) -> None:
        load_dotenv()

        self.hf_token = os.getenv("HF_TOKEN")
        self.model_id = model_id

        # ✅ HuggingFace Router endpoint (HF now requires this)
        self.api_url = f"https://router.huggingface.co/models/{model_id}"
        self.hf_headers = (
            {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        )

        self.default_temperature = 0.2
        self.default_max_tokens = 512

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

        # For HF, keep the system prompt in a single string (works ok for T5)
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"Question: {prompt}\n\nAnswer:"

        # 1) Try HF Router
        api_result = self._generate_via_hf_router(
            full_prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        if api_result is not None:
            return api_result

        # 2) Local fallback (short prompt to reduce instruction echo)
        local_prompt = f"Question: {prompt}\n\nAnswer:"
        return self._generate_via_local_model(local_prompt, max_tokens)

    def generate_with_reasoning(
        self,
        question: str,
        chunks: List[Union[Dict[str, str], str]],
        reasoner: MultiDocReasoner,
    ) -> Dict[str, Any]:
        prompt, query_type = reasoner.build_prompt(question, chunks)
        response_text = self.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return {"response": response_text, "query_type": query_type}

    # ------------------------------------------------------------------
    # HF Router
    # ------------------------------------------------------------------
    def _generate_via_hf_router(
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

        for attempt in range(max_retries):
            try:
                r = requests.post(
                    self.api_url,
                    headers=self.hf_headers,
                    json=payload,
                    timeout=60,
                )

                # HF Router can return JSON errors even with 200 sometimes; parse safely
                try:
                    out = r.json()
                except Exception:
                    return None

                if r.status_code == 200:
                    # Most common: list of dicts
                    if isinstance(out, list) and out:
                        item = out[0]
                        if isinstance(item, dict):
                            return (item.get("generated_text") or "").strip()
                        # sometimes item is string
                        if isinstance(item, str):
                            return item.strip()
                        return str(item).strip()

                    # Sometimes: dict
                    if isinstance(out, dict):
                        # If HF returns an error in dict form, fallback
                        if "error" in out:
                            return None
                        if "generated_text" in out:
                            return (out.get("generated_text") or "").strip()
                        # Some backends use "summary_text"
                        if "summary_text" in out:
                            return (out.get("summary_text") or "").strip()
                        return None

                    # Fallback to string
                    if isinstance(out, str):
                        return out.strip()

                    return None

                # transient/rate limit/loading → fallback
                if r.status_code in {429, 503}:
                    time.sleep(2 + attempt * 2)
                    continue

                # other status codes -> fallback to local
                return None

            except requests.RequestException:
                time.sleep(2 + attempt * 2)
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

        # keep inputs bounded
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


