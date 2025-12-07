"""LLM Client for HuggingFace Inference API."""

import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner


class LLMClient:
    """Client for HuggingFace Inference API."""

    def __init__(self, model_id: str = "google/flan-t5-large") -> None:
        """Initialize with model_id and load HF_TOKEN from environment."""
        load_dotenv()
        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file.")

        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.default_temperature = 0.3
        self.default_max_tokens = 512

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> str:
        """Generate text using HuggingFace API with retry logic."""
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"Question: {prompt}\n\nAnswer:"

        payload = {
            "inputs": full_prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list):
                        return result[0].get("generated_text", "").strip()
                    return str(result).strip()

                elif response.status_code == 503:
                    print(f"Model loading. Waiting 20 seconds...")
                    time.sleep(20)
                    continue
                elif response.status_code == 429:
                    print(f"Rate limited. Waiting 20 seconds...")
                    time.sleep(20)
                    continue
                else:
                    raise RuntimeError(f"API error {response.status_code}: {response.text}")

            except requests.Timeout:
                if attempt < max_retries - 1:
                    print(f"Request timed out. Retrying...")
                    time.sleep(5)
                    continue
                raise TimeoutError("Request timed out. The model may be taking too long to respond.")

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request failed: {e}. Retrying...")
                    time.sleep(5)
                    continue
                raise RuntimeError(f"Request failed after {max_retries} attempts: {e}")

        raise RuntimeError("Max retries exceeded")

    def generate_with_reasoning(
        self,
        question: str,
        chunks: List[Dict[str, str]],
        reasoner: MultiDocReasoner,
    ) -> Dict[str, Any]:
        """Generate response using reasoner to build prompt."""
        prompt, query_type = reasoner.build_prompt(question, chunks)
        response_text = self.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return {'response': response_text, 'query_type': query_type}
