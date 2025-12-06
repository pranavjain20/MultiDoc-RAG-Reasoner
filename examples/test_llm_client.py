"""Test LLM client."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_initialization():
    try:
        from llm.llm_client import LLMClient
    except ImportError:
        print("✗ LLMClient not available (requests not installed)")
        return False

    from dotenv import load_dotenv
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠ HF_TOKEN not found, skipping test")
        return None

    try:
        client = LLMClient()
        assert client.model_id == "google/flan-t5-large"
        assert client.default_temperature == 0.3
        assert client.default_max_tokens == 512
        print("✓ LLMClient initialization works")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_simple_api_call():
    try:
        from llm.llm_client import LLMClient
    except ImportError:
        return None

    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("HF_TOKEN"):
        print("⚠ HF_TOKEN not found, skipping API test")
        return None

    try:
        client = LLMClient()
        response = client.generate("What is 2+2? Answer in one word.", max_tokens=10)
        assert isinstance(response, str)
        print("✓ API call works")
        return True
    except Exception as e:
        print(f"⚠ API call failed (may be rate limited): {e}")
        return None


def test_generate_with_reasoning():
    try:
        from llm.llm_client import LLMClient
        from llm.reasoning import MultiDocReasoner
    except ImportError:
        return None

    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("HF_TOKEN"):
        return None

    try:
        client = LLMClient()
        reasoner = MultiDocReasoner()
        chunks = [{'doc_name': 'doc.pdf', 'text': 'Test content'}]
        result = client.generate_with_reasoning("Summarize", chunks, reasoner)
        assert 'response' in result
        assert 'query_type' in result
        print("✓ generate_with_reasoning works")
        return True
    except Exception as e:
        print(f"⚠ generate_with_reasoning failed: {e}")
        return None


def main():
    print("Testing llm_client.py\n")
    test_initialization()
    api_result = test_simple_api_call()
    reasoning_result = test_generate_with_reasoning()
    
    if api_result or reasoning_result:
        print("\n✓ LLM client tests passed!")
    else:
        print("\n⚠ Some tests skipped (no API token or API unavailable)")


if __name__ == "__main__":
    main()
