"""
Groq LLM API wrapper for the Multi-Document RAG system.
"""

from __future__ import annotations

import os
from typing import Optional, List

from groq import Groq
from groq import BadRequestError


# Human-readable names exposed in UI (optional)
AVAILABLE_MODELS = {
    "LLaMA 3.3 70B (best)": "llama-3.3-70b-versatile",
    "LLaMA 3.1 8B (fast)": "llama-3.1-8b-instant",
    "Mixtral 8x7B (long context)": "mixtral-8x7b-32768",
}

# ✅ Default to the strongest available
DEFAULT_MODEL = AVAILABLE_MODELS["LLaMA 3.3 70B (best)"]

# If a model is decommissioned, try these in order
FALLBACK_MODELS: List[str] = [
    AVAILABLE_MODELS["LLaMA 3.3 70B (best)"],
    AVAILABLE_MODELS["Mixtral 8x7B (long context)"],
    AVAILABLE_MODELS["LLaMA 3.1 8B (fast)"],
]

DEFAULT_SYSTEM_PROMPT = """
You are a careful teaching assistant for a data science / NLP course project.

- You must answer only based on the provided context from the documents.
- If the context is not enough to answer confidently, say:
  "I don't know based on the documents."
- Prefer short, precise answers unless asked for details.
- Cite sources using the format [DocumentName.pdf] when referencing evidence.
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
    user_content = _build_user_prompt(question, context)

    # Try requested model first, then fallbacks
    tried = []
    candidates = []
    if model_name:
        candidates.append(model_name)
    candidates += [m for m in FALLBACK_MODELS if m not in candidates]

    last_err: Optional[Exception] = None

    for m in candidates:
        tried.append(m)
        try:
            completion = client.chat.completions.create(
                model=m,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            msg = completion.choices[0].message
            return (msg.content or "").strip()

        except BadRequestError as e:
            # Model decommissioned / invalid model id -> try next
            last_err = e
            continue

        except Exception as e:
            # transient network etc -> try next
            last_err = e
            continue

    raise RuntimeError(f"Groq failed for models={tried}. Last error: {last_err}")

