"""Tests for the LLM client."""

from unittest.mock import MagicMock, patch

import pytest

from llm.llm_client import LLMClient
from llm.reasoning import MultiDocReasoner


class TestInitialization:
    def test_default_model(self):
        with patch("llm.llm_client.load_dotenv"):
            client = LLMClient()
        assert client.model_id == "google/flan-t5-large"

    def test_default_temperature(self):
        with patch("llm.llm_client.load_dotenv"):
            client = LLMClient()
        assert client.default_temperature == 0.2

    def test_default_max_tokens(self):
        with patch("llm.llm_client.load_dotenv"):
            client = LLMClient()
        assert client.default_max_tokens == 512

    def test_custom_model(self):
        with patch("llm.llm_client.load_dotenv"):
            client = LLMClient(model_id="custom/model")
        assert client.model_id == "custom/model"

    def test_local_model_default(self):
        with patch("llm.llm_client.load_dotenv"):
            client = LLMClient()
        assert client.local_model_id == "google/flan-t5-base"


class TestGenerate:
    @patch("llm.llm_client.load_dotenv")
    @patch("llm.llm_client.generate_llm_response")
    def test_groq_called_first_when_key_set(self, mock_groq, mock_dotenv):
        mock_groq.return_value = "Groq response"
        client = LLMClient()
        client.groq_api_key = "test-key"

        result = client.generate("test prompt")

        assert result == "Groq response"
        mock_groq.assert_called_once()

    @patch("llm.llm_client.load_dotenv")
    @patch("llm.llm_client.generate_llm_response")
    def test_groq_fallback_on_error(self, mock_groq, mock_dotenv):
        mock_groq.side_effect = Exception("Groq down")
        client = LLMClient()
        client.groq_api_key = "test-key"
        client.hf_token = None  # skip HF too

        # Should fall through to local model
        with patch.object(client, "_generate_via_local_model", return_value="local response"):
            result = client.generate("test prompt")

        assert result == "local response"

    @patch("llm.llm_client.load_dotenv")
    def test_hf_api_skipped_without_token(self, mock_dotenv):
        client = LLMClient()
        client.groq_api_key = None
        client.hf_token = None

        with patch.object(client, "_generate_via_local_model", return_value="local") as mock_local:
            result = client.generate("test")

        assert result == "local"
        mock_local.assert_called_once()


class TestHfApi:
    @patch("llm.llm_client.load_dotenv")
    def test_returns_none_without_token(self, mock_dotenv):
        client = LLMClient()
        client.hf_token = None
        result = client._generate_via_hf_api("prompt", 0.2, 512, 2)
        assert result is None

    @patch("llm.llm_client.load_dotenv")
    @patch("llm.llm_client.requests.post")
    def test_returns_none_on_404(self, mock_post, mock_dotenv):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        client = LLMClient()
        client.hf_token = "token"
        result = client._generate_via_hf_api("prompt", 0.2, 512, 2)
        assert result is None

    @patch("llm.llm_client.load_dotenv")
    @patch("llm.llm_client.requests.post")
    def test_parses_list_response(self, mock_post, mock_dotenv):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"generated_text": "  hello  "}]
        mock_post.return_value = mock_response

        client = LLMClient()
        client.hf_token = "token"
        result = client._generate_via_hf_api("prompt", 0.2, 512, 2)
        assert result == "hello"

    @patch("llm.llm_client.load_dotenv")
    @patch("llm.llm_client.requests.post")
    def test_parses_dict_response(self, mock_post, mock_dotenv):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"generated_text": "world"}
        mock_post.return_value = mock_response

        client = LLMClient()
        client.hf_token = "token"
        result = client._generate_via_hf_api("prompt", 0.2, 512, 2)
        assert result == "world"


class TestGenerateWithReasoning:
    @patch("llm.llm_client.load_dotenv")
    @patch("llm.llm_client.generate_llm_response")
    def test_returns_expected_keys(self, mock_groq, mock_dotenv):
        mock_groq.return_value = "Test response"
        client = LLMClient()
        client.groq_api_key = "test-key"
        reasoner = MultiDocReasoner()

        chunks = [{"doc_name": "doc.pdf", "text": "Test content"}]
        result = client.generate_with_reasoning("Summarize", chunks, reasoner)

        assert "response" in result
        assert "query_type" in result
        assert result["response"] == "Test response"
        assert result["query_type"] == "synthesis"

    @patch("llm.llm_client.load_dotenv")
    @patch("llm.llm_client.generate_llm_response")
    def test_classifies_comparison(self, mock_groq, mock_dotenv):
        mock_groq.return_value = "Comparison response"
        client = LLMClient()
        client.groq_api_key = "test-key"
        reasoner = MultiDocReasoner()

        chunks = [{"doc_name": "doc.pdf", "text": "content"}]
        result = client.generate_with_reasoning("Compare these documents", chunks, reasoner)

        assert result["query_type"] == "comparison"
