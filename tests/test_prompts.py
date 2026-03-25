"""Tests for prompt builders."""

import pytest

from llm.prompts import (
    SYSTEM_PROMPT,
    build_comparison_prompt,
    build_extraction_prompt,
    build_synthesis_prompt,
    select_prompt_builder,
)


class TestSystemPrompt:
    def test_is_nonempty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_contains_citation_format(self):
        assert "[DocumentName.pdf]" in SYSTEM_PROMPT

    def test_mentions_multi_document(self):
        assert "multi-document" in SYSTEM_PROMPT.lower()


class TestBuildSynthesisPrompt:
    def test_includes_document_names(self, sample_chunks):
        prompt = build_synthesis_prompt("Summarize main ideas", sample_chunks)
        assert "doc1.pdf" in prompt
        assert "doc2.pdf" in prompt

    def test_includes_question(self, sample_chunks):
        question = "What are the key themes?"
        prompt = build_synthesis_prompt(question, sample_chunks)
        assert question in prompt

    def test_includes_chunk_text(self, sample_chunks):
        prompt = build_synthesis_prompt("Summarize", sample_chunks)
        assert "Content from document 1" in prompt
        assert "Content from document 2" in prompt

    def test_empty_chunks(self):
        prompt = build_synthesis_prompt("Test question", [])
        assert "No documents" in prompt

    def test_single_doc(self):
        chunks = [{"doc_name": "only.pdf", "text": "Solo content"}]
        prompt = build_synthesis_prompt("Summarize", chunks)
        assert "only.pdf" in prompt
        assert "Solo content" in prompt

    def test_missing_doc_name_defaults(self):
        chunks = [{"text": "No doc name here"}]
        prompt = build_synthesis_prompt("Summarize", chunks)
        assert "Unknown.pdf" in prompt


class TestBuildComparisonPrompt:
    def test_includes_document_names(self, sample_chunks_by_doc):
        prompt = build_comparison_prompt("Compare approaches", sample_chunks_by_doc)
        assert "doc1.pdf" in prompt
        assert "doc2.pdf" in prompt

    def test_includes_question(self, sample_chunks_by_doc):
        question = "How do they differ?"
        prompt = build_comparison_prompt(question, sample_chunks_by_doc)
        assert question in prompt

    def test_empty_chunks(self):
        prompt = build_comparison_prompt("Compare", {})
        assert "No documents" in prompt

    def test_single_doc_notes_limitation(self):
        chunks_by_doc = {"only.pdf": ["Some content"]}
        prompt = build_comparison_prompt("Compare", chunks_by_doc)
        assert "only one document" in prompt.lower()

    def test_uses_equals_delimiters(self, sample_chunks_by_doc):
        # Comparison prompts use === to visually separate docs being compared
        prompt = build_comparison_prompt("Compare", sample_chunks_by_doc)
        assert "===" in prompt


class TestBuildExtractionPrompt:
    def test_includes_document_names(self):
        chunks = [{"doc_name": "doc1.pdf", "text": "We assume X"}]
        prompt = build_extraction_prompt("What are the assumptions?", chunks)
        assert "doc1.pdf" in prompt

    def test_includes_question(self):
        chunks = [{"doc_name": "doc.pdf", "text": "content"}]
        question = "What are the limitations?"
        prompt = build_extraction_prompt(question, chunks)
        assert question in prompt

    def test_empty_chunks(self):
        prompt = build_extraction_prompt("Extract info", [])
        assert "No documents" in prompt

    def test_mentions_extraction_terms(self):
        chunks = [{"doc_name": "doc.pdf", "text": "content"}]
        prompt = build_extraction_prompt("What are the assumptions?", chunks)
        assert "assumptions" in prompt.lower()


class TestSelectPromptBuilder:
    def test_synthesis(self):
        assert select_prompt_builder("synthesis") == build_synthesis_prompt

    def test_comparison(self):
        assert select_prompt_builder("comparison") == build_comparison_prompt

    def test_extraction(self):
        assert select_prompt_builder("extraction") == build_extraction_prompt

    def test_case_insensitive(self):
        assert select_prompt_builder("Synthesis") == build_synthesis_prompt
        assert select_prompt_builder("COMPARISON") == build_comparison_prompt

    def test_strips_whitespace(self):
        assert select_prompt_builder("  extraction  ") == build_extraction_prompt

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown query_type"):
            select_prompt_builder("invalid_type")
