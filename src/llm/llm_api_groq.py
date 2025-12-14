"""
Groq LLM API wrapper for the Multi-Document RAG system.
"""

from __future__ import annotations

import os
from typing import Optional

from groq import Groq


# Human-readable names exposed in the UI (if you add dropdown later)
AVAILABLE_MODELS = {
    "LLaMA 3.1 8B (fast)": "llama-3.1-8b-instant",
    # ✅ 70B 3.1 已下线；用 3.3
    "LLaMA 3.3 70B (better)": "llama-3.3-70b-versatile",
    "LLaMA 3.3 70B (specdec)": "llama-3.3-70b-specdec",
    # Mixtral 8x7b-32768 也在 deprecations 里，尽量别当默认
}

DEFAULT_MODEL = AVAILABLE_MODELS["LLaMA 3.3 70B (better)"]

DEFAULT_SYSTEM_PROMPT = """
You are a careful teaching assistant for a data science / NLP course project.

- You must answer only based on the provided context from the documents.
- If the context is not enough to answer confidently, say:
  "I don't know based on the documents."
- Prefer short, precise answers unless asked for details.
- When comparing documents, explicitly mention which document supports each claim.
""".strip()


def _build_user_prompt(question: str, context: Optional[str]) -> str:
    return (
        "You are given context chunks retrieved from a PDF corpus.\n"
        "Use ONLY these chunks to answer the question.\n\n"
        f"Context:\n{context or ''}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_llm_response(
    question: str,
    context: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")

    client = Groq(api_key=api_key)
    model_name = model_name or DEFAULT_MODEL

    user_content = _build_user_prompt(question, context)

    completion = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    message = completion.choices[0].message
    return (message.content or "").strip()


