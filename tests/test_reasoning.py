"""Tests for the reasoning module."""

import pytest

from llm.reasoning import MultiDocReasoner


class TestClassifyQuery:
    def test_extraction(self, reasoner):
        assert reasoner.classify_query("What are the assumptions?") == "extraction"

    def test_comparison(self, reasoner):
        assert reasoner.classify_query("Compare methodologies") == "comparison"

    def test_synthesis(self, reasoner):
        assert reasoner.classify_query("Summarize main ideas") == "synthesis"

    def test_default_is_synthesis(self, reasoner):
        assert reasoner.classify_query("Tell me about documents") == "synthesis"
        assert reasoner.classify_query("Hello world") == "synthesis"

    def test_case_insensitive(self, reasoner):
        assert reasoner.classify_query("COMPARE these docs") == "comparison"
        assert reasoner.classify_query("What are the ASSUMPTIONS?") == "extraction"

    @pytest.mark.parametrize(
        "question,expected",
        [
            ("Summarize the main ideas discussed across these documents", "synthesis"),
            ("What are the main sources of risk mentioned across the documents", "synthesis"),
            ("Compare how different documents describe the same concept", "comparison"),
            ("What are the key assumptions and limitations highlighted", "extraction"),
            ("How do the documents differ in their conclusions", "comparison"),
        ],
    )
    def test_evaluation_questions(self, reasoner, question, expected):
        assert reasoner.classify_query(question) == expected


class TestOrganizeChunksByDoc:
    def test_groups_by_document(self, reasoner, sample_chunks_three):
        result = reasoner.organize_chunks_by_doc(sample_chunks_three)
        assert len(result["doc1.pdf"]) == 2
        assert len(result["doc2.pdf"]) == 1

    def test_empty_list(self, reasoner):
        assert reasoner.organize_chunks_by_doc([]) == {}

    def test_missing_doc_name(self, reasoner):
        chunks = [{"text": "no doc name"}]
        result = reasoner.organize_chunks_by_doc(chunks)
        assert "Unknown.pdf" in result

    def test_preserves_text_content(self, reasoner):
        chunks = [{"doc_name": "a.pdf", "text": "hello"}]
        result = reasoner.organize_chunks_by_doc(chunks)
        assert result["a.pdf"] == ["hello"]


class TestGetUniqueDocuments:
    def test_returns_sorted(self, reasoner):
        chunks = [
            {"doc_name": "doc2.pdf", "text": "..."},
            {"doc_name": "doc1.pdf", "text": "..."},
        ]
        assert reasoner.get_unique_documents(chunks) == ["doc1.pdf", "doc2.pdf"]

    def test_deduplicates(self, reasoner, sample_chunks_three):
        result = reasoner.get_unique_documents(sample_chunks_three)
        assert result == ["doc1.pdf", "doc2.pdf"]

    def test_empty_list(self, reasoner):
        assert reasoner.get_unique_documents([]) == []


class TestMitigateLostInMiddle:
    def test_reorders_three_chunks(self, reasoner):
        chunks = [
            {"doc_name": "doc.pdf", "text": "best", "score": 0.9},
            {"doc_name": "doc.pdf", "text": "2nd", "score": 0.8},
            {"doc_name": "doc.pdf", "text": "3rd", "score": 0.7},
        ]
        reordered = reasoner.mitigate_lost_in_middle(chunks)
        assert len(reordered) == 3
        # Even indices [0,2] reversed = [2,0], then odd [1] appended
        assert reordered[0] == chunks[2]
        assert reordered[1] == chunks[0]
        assert reordered[2] == chunks[1]

    def test_single_chunk(self, reasoner):
        chunks = [{"doc_name": "doc.pdf", "text": "only"}]
        assert reasoner.mitigate_lost_in_middle(chunks) == chunks

    def test_empty_list(self, reasoner):
        assert reasoner.mitigate_lost_in_middle([]) == []

    def test_two_chunks(self, reasoner):
        chunks = [
            {"doc_name": "doc.pdf", "text": "first"},
            {"doc_name": "doc.pdf", "text": "second"},
        ]
        reordered = reasoner.mitigate_lost_in_middle(chunks)
        assert len(reordered) == 2

    def test_preserves_all_chunks(self, reasoner):
        chunks = [{"doc_name": "d.pdf", "text": str(i)} for i in range(6)]
        reordered = reasoner.mitigate_lost_in_middle(chunks)
        assert len(reordered) == 6
        assert set(c["text"] for c in reordered) == set(c["text"] for c in chunks)


class TestBuildPrompt:
    def test_synthesis_prompt(self, reasoner, sample_chunks):
        prompt, query_type = reasoner.build_prompt("Summarize main ideas", sample_chunks)
        assert query_type == "synthesis"
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_comparison_prompt(self, reasoner, sample_chunks):
        prompt, query_type = reasoner.build_prompt("Compare these docs", sample_chunks)
        assert query_type == "comparison"

    def test_extraction_prompt(self, reasoner, sample_chunks):
        prompt, query_type = reasoner.build_prompt("What are the assumptions?", sample_chunks)
        assert query_type == "extraction"

    def test_empty_chunks(self, reasoner):
        prompt, query_type = reasoner.build_prompt("Summarize", [])
        assert "No documents" in prompt
        assert query_type == "synthesis"

    def test_no_mitigation(self, reasoner, sample_chunks):
        prompt, _ = reasoner.build_prompt("Summarize", sample_chunks, apply_lost_in_middle=False)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
