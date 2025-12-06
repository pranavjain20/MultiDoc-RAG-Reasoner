"""Test prompt builders."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.prompts import (
    SYSTEM_PROMPT,
    build_synthesis_prompt,
    build_comparison_prompt,
    build_extraction_prompt,
    select_prompt_builder,
)


def test_system_prompt():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
    assert "[DocumentName.pdf]" in SYSTEM_PROMPT
    print("✓ SYSTEM_PROMPT works")


def test_build_synthesis_prompt():
    chunks = [
        {"doc_name": "doc1.pdf", "text": "Content from document 1"},
        {"doc_name": "doc2.pdf", "text": "Content from document 2"},
    ]
    prompt = build_synthesis_prompt("Summarize main ideas", chunks)
    assert isinstance(prompt, str)
    assert "doc1.pdf" in prompt
    assert "doc2.pdf" in prompt
    print("✓ build_synthesis_prompt works")

    empty_prompt = build_synthesis_prompt("Test", [])
    assert "No documents" in empty_prompt
    print("✓ Empty chunks handled")


def test_build_comparison_prompt():
    chunks_by_doc = {
        "doc1.pdf": ["Content 1"],
        "doc2.pdf": ["Content 2"],
    }
    prompt = build_comparison_prompt("Compare approaches", chunks_by_doc)
    assert isinstance(prompt, str)
    assert "doc1.pdf" in prompt
    assert "doc2.pdf" in prompt
    print("✓ build_comparison_prompt works")


def test_build_extraction_prompt():
    chunks = [
        {"doc_name": "doc1.pdf", "text": "We assume X"},
    ]
    prompt = build_extraction_prompt("What are the assumptions?", chunks)
    assert isinstance(prompt, str)
    assert "doc1.pdf" in prompt
    print("✓ build_extraction_prompt works")


def test_select_prompt_builder():
    builder = select_prompt_builder('synthesis')
    assert builder == build_synthesis_prompt
    builder = select_prompt_builder('comparison')
    assert builder == build_comparison_prompt
    print("✓ select_prompt_builder works")


def main():
    print("Testing prompts.py\n")
    test_system_prompt()
    test_build_synthesis_prompt()
    test_build_comparison_prompt()
    test_build_extraction_prompt()
    test_select_prompt_builder()
    print("\n✓ All prompt tests passed!")


if __name__ == "__main__":
    main()

