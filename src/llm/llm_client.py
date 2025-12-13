"""
LLM Client with Groq (smart) + local transformers fallback.

Priority:
1) Groq chat completion (if GROQ_API_KEY is set)
2) Local transformers fallback (always works)

NOTE:
- HuggingFace legacy Inference API `api-inference.huggingface.co` is deprecated (410).
- HF Router exists, but it is OpenAI-compatible chat-first and not a drop-in
  replacement for the old `POST /models/{repo_id}` pattern for all models.
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT
from llm.reasoning import MultiDocReasoner

# Groq wrapper (separate file)
from llm.llm_api_groq import generate_llm_response


class LLMClient:
    """Client for multi-backend text generation (Groq → Local)."""

    def __init__(
        self,
        local_model_id: str = "google/flan-t5-base",
        groq_model_name: Optional[str] = None,
    ) -> None:
        load_dotenv()

        # Groq (smart)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model_name = groq_model_name  # optional override

        # Defaults
        self.default_temperature = 0.2
        self.default_max_tokens = 512

        # Local fallback (always available)
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
        In this repo, `prompt` is often already a structured prompt produced by
        `MultiDocReasoner.build_prompt(...)`.
        """
        temperature = self.default_temperature if temperature is None else temperature
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens

        # 1) Groq (preferred)
        if self.groq_api_key:
            # For chat models, it's cleaner to pass `prompt` as "question" and keep context empty,
            # because the prompt already contains the retrieved text.
            try:
                full_prompt = (
                    f"{system_prompt}\n\n{prompt}"
                    if system_prompt
                    else prompt
                )
                return generate_llm_response(
                    question=full_prompt,
                    context=None,
                    model_name=self.groq_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt or SYSTEM_PROMPT,
                )
            except Exception:
                # fall through to local
                pass

        # 2) Local fallback
        # Keep local prompt short to avoid instruction-echo.
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

        - Keep your existing reasoner (query_type classification + prompt structuring).
        - If Groq exists: send (question + grouped context) to Groq (best quality).
        - Else: fallback to local using your structured prompt.
        """
        prompt, query_type = reasoner.build_prompt(question, chunks)

        # If Groq is available, build a cleaner context string (better than stuffing everything into one mega prompt)
        if self.groq_api_key:
            try:
                chunks_by_doc = reasoner.organize_chunks_by_doc(chunks)
                context_parts: List[str] = []
                for doc_name, doc_chunks in chunks_by_doc.items():
                    context_parts.append(f"--- {doc_name} ---")
                    for c in doc_chunks:
                        txt = (c.get("text") or "").strip()
                        if txt:
                            context_parts.append(txt)
                context = "\n\n".join(context_parts).strip()

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
                # fallback to local prompt path
                pass

        response_text = self.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return {"response": response_text, "query_type": query_type}

    # ------------------------------------------------------------------
    # Local fallback
    # ------------------------------------------------------------------
    def _generate_via_local_model(self, prompt: str, max_tokens: int) -> str:
        """
        Local transformers fallback.

        - Truncates inputs to avoid exceeding model limits.
        - Uses text2text-generation pipeline for FLAN-T5.
        """
        if self._local_pipeline is None or self._local_tokenizer is None:
            from transformers import AutoTokenizer, pipeline

            self._local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id)
            self._local_pipeline = pipeline(
                "text2text-generation",
                model=self.local_model_id,
            )

        # Keep input bounded (flan-t5-base max input ~512 tokens typical; we keep safe)
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

        if isinstance(out, list) and out and isinstance(out[0], dict):
            return (out[0].get("generated_text") or "").strip()

        return str(out).strip()


