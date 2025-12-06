"""LLM module for multi-document RAG system."""

from llm.prompts import (
    SYSTEM_PROMPT,
    build_comparison_prompt,
    build_extraction_prompt,
    build_synthesis_prompt,
    select_prompt_builder,
)
from llm.reasoning import MultiDocReasoner

try:
    from llm.llm_client import LLMClient
except ImportError:
    LLMClient = None

__all__ = [
    "SYSTEM_PROMPT",
    "build_synthesis_prompt",
    "build_comparison_prompt",
    "build_extraction_prompt",
    "select_prompt_builder",
    "MultiDocReasoner",
    "LLMClient",
]
