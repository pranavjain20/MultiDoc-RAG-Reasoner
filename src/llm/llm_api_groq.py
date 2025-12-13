"""
Groq LLM API wrapper for the Multi-Document RAG system.
"""

from __future__ import annotations

import os
from typing import Optional

from groq import Groq


AVAILABLE_MODELS = {
    "LLaMA 3.1 8B (fast)": "llama-3.1-8b-instant",
    "LLaMA 3.1 70B (better)": "llama-3.1-70b-versatile",
    "Mixtral 8x7B (long context)": "mixtral-8x7b-32768",
}
DEFAULT_MODEL = AVAILABLE_MODELS["LLaMA 3.1 70B (better)"]

DEFAULT_SYSTEM_PROMPT = """
You are a careful teaching assistant for a data science / NLP course project.

Core rules:
- You MUST answer only using the provided context excerpts.
- If the context is insufficient, say exactly: "I don't know based on the documents."
- Do NOT copy/paste long passages from the context. Summarize in your own words.
- Prefer short, structured answers (bullets or short paragraphs).
- If multiple documents disagree, explicitly state the disagreement.
- Cite sources using [DocumentName.pdf] when referencing a specific claim.
""".strip()


def _build_user_prompt(question: str, context: Optional[str]) -> str:
    # Strong “task framing” to prevent the model from echoing chunks.
    # IMPORTANT: context should already be grouped by document upstream (in llm_client.py),
    # but we still defensively instruct here.
    ctx = (context or "").strip()

    return f"""
You will be given excerpts from multiple PDF documents.

Your task:
1) Answer the question directly.
2) Use the excerpts as evidence, but DO NOT quote long text.
3) Provide a concise, high-level explanation (2–8 bullet points is ideal).
4) When you use evidence, cite it like [doc1.pdf]. If multiple docs support the same point, cite both.
5) If the excerpts do not contain enough information, reply exactly:
   I don't know based on the documents.

Excerpts (grouped by document):
{ctx}

Question:
{question}

Answer (concise, evidence-based, with citations):
""".strip()


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
