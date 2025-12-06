# LLM Reasoning Layer

LLM reasoning component for multi-document RAG system. Handles prompt construction and LLM API calls.

## Installation

```bash
pip install -r requirements.txt
```

Create `.env` file with your HuggingFace token:
```
HF_TOKEN=your_token_here
```

## Usage

```python
from llm import LLMClient, MultiDocReasoner

reasoner = MultiDocReasoner()
client = LLMClient()

chunks = [
    {'doc_name': 'doc1.pdf', 'text': 'Content here'},
    {'doc_name': 'doc2.pdf', 'text': 'More content'}
]

result = client.generate_with_reasoning(
    question="Summarize main ideas",
    chunks=chunks,
    reasoner=reasoner
)

print(result['response'])  # Generated answer
print(result['query_type'])  # 'synthesis', 'comparison', or 'extraction'
```

## Input Format

Chunks should be a list of dictionaries:
```python
chunks = [
    {
        'doc_name': 'document.pdf',  # Required
        'text': 'chunk text',         # Required
        'score': 0.95                 # Optional (for optimization)
    }
]
```

## Query Types

The system automatically classifies questions:
- **Synthesis**: "Summarize main ideas", "What are the sources of risk"
- **Comparison**: "Compare approaches", "How do documents differ"
- **Extraction**: "What are the assumptions", "List limitations"

## Files

- `prompts.py`: Prompt builders for different query types
- `reasoning.py`: Query classification and chunk organization
- `llm_client.py`: HuggingFace API client

## Testing

```bash
python3 examples/test_prompts.py
python3 examples/test_reasoning.py
python3 examples/test_llm_client.py
```
