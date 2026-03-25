"""Shared fixtures for the test suite."""

import pytest

from llm.reasoning import MultiDocReasoner


@pytest.fixture
def reasoner():
    return MultiDocReasoner()


@pytest.fixture
def sample_chunks():
    return [
        {"doc_name": "doc1.pdf", "text": "Content from document 1"},
        {"doc_name": "doc2.pdf", "text": "Content from document 2"},
    ]


@pytest.fixture
def sample_chunks_three():
    return [
        {"doc_name": "doc1.pdf", "text": "First chunk", "score": 0.9},
        {"doc_name": "doc2.pdf", "text": "Second chunk", "score": 0.8},
        {"doc_name": "doc1.pdf", "text": "Third chunk", "score": 0.7},
    ]


@pytest.fixture
def sample_chunks_by_doc():
    return {
        "doc1.pdf": ["Content 1a", "Content 1b"],
        "doc2.pdf": ["Content 2a"],
    }
