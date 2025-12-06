"""Test reasoning module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.reasoning import MultiDocReasoner


def test_classify_query():
    reasoner = MultiDocReasoner()
    assert reasoner.classify_query("What are the assumptions?") == 'extraction'
    assert reasoner.classify_query("Compare methodologies") == 'comparison'
    assert reasoner.classify_query("Summarize main ideas") == 'synthesis'
    assert reasoner.classify_query("Tell me about documents") == 'synthesis'
    print("✓ classify_query works")


def test_organize_chunks():
    reasoner = MultiDocReasoner()
    chunks = [
        {'doc_name': 'doc1.pdf', 'text': 'chunk1'},
        {'doc_name': 'doc2.pdf', 'text': 'chunk2'},
        {'doc_name': 'doc1.pdf', 'text': 'chunk3'},
    ]
    result = reasoner.organize_chunks_by_doc(chunks)
    assert len(result['doc1.pdf']) == 2
    assert len(result['doc2.pdf']) == 1
    print("✓ organize_chunks_by_doc works")


def test_get_unique_documents():
    reasoner = MultiDocReasoner()
    chunks = [
        {'doc_name': 'doc2.pdf', 'text': '...'},
        {'doc_name': 'doc1.pdf', 'text': '...'},
    ]
    result = reasoner.get_unique_documents(chunks)
    assert result == ['doc1.pdf', 'doc2.pdf']
    print("✓ get_unique_documents works")


def test_mitigate_lost_in_middle():
    reasoner = MultiDocReasoner()
    chunks = [
        {'doc_name': 'doc.pdf', 'text': 'best', 'score': 0.9},
        {'doc_name': 'doc.pdf', 'text': '2nd', 'score': 0.8},
        {'doc_name': 'doc.pdf', 'text': '3rd', 'score': 0.7},
    ]
    reordered = reasoner.mitigate_lost_in_middle(chunks)
    assert len(reordered) == 3
    assert reordered[0] == chunks[2]
    assert reordered[-1] == chunks[1]
    print("✓ mitigate_lost_in_middle works")


def test_build_prompt():
    reasoner = MultiDocReasoner()
    chunks = [
        {'doc_name': 'doc1.pdf', 'text': 'Content 1'},
        {'doc_name': 'doc2.pdf', 'text': 'Content 2'},
    ]
    prompt, query_type = reasoner.build_prompt("Summarize main ideas", chunks)
    assert query_type == 'synthesis'
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    print("✓ build_prompt works")


def test_all_5_evaluation_questions():
    reasoner = MultiDocReasoner()
    chunks = [{'doc_name': 'doc.pdf', 'text': 'content'}]
    
    questions = [
        ("Summarize the main ideas discussed across these documents", 'synthesis'),
        ("What are the main sources of risk mentioned across the documents", 'synthesis'),
        ("Compare how different documents describe the same concept", 'comparison'),
        ("What are the key assumptions and limitations highlighted", 'extraction'),
        ("How do the documents differ in their conclusions", 'comparison'),
    ]
    
    for question, expected_type in questions:
        _, query_type = reasoner.build_prompt(question, chunks)
        assert query_type == expected_type
    print("✓ All 5 evaluation questions classified correctly")


def main():
    print("Testing reasoning.py\n")
    test_classify_query()
    test_organize_chunks()
    test_get_unique_documents()
    test_mitigate_lost_in_middle()
    test_build_prompt()
    test_all_5_evaluation_questions()
    print("\n✓ All reasoning tests passed!")


if __name__ == "__main__":
    main()
