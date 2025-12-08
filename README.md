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




# Document Ingestion & FAISS Index Builder

This component handles **PDF loading, text chunking, embedding generation, and FAISS index construction** for the multi-document RAG system.
All other system components (reasoning layer, API UI, evaluation) depend on this index.

## Usage (Google Colab Notebook)

The ingestion pipeline is implemented in:

```
AML_Project.ipynb
```

### Steps

1. Upload one or more PDFs in Colab
2. Run the notebook (all cells)
3. It will:

   * Load pages using `PyPDFLoader`
   * Split text into overlapping chunks
   * Encode chunks using the `all-MiniLM-L6-v2` SentenceTransformer
   * Build a FAISS vector index
   * Save the index into:

```
index_store/
    faiss_store.index
    index.pkl
    index.json
```

### Loading the index (for other components)

```python
from langchain_community.vectorstores import FAISS

db = FAISS.load_local("index_store", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
```

This retriever is what the **LLM Reasoning Layer** uses to fetch relevant chunks.

---

## Pipeline Details

### 1. PDF Loading

```python
from langchain_community.document_loaders import PyPDFLoader
```

Extracts text per PDF page.

### 2. Text Chunking

Uses:

* `RecursiveCharacterTextSplitter`
* `chunk_size = 800`
* `chunk_overlap = 150`

These can be adjusted depending on model context window size.

### 3. Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

Lightweight and fast for Colab while producing solid semantic embeddings.

### 4. FAISS Index Building

The notebook creates and stores a FAISS index with metadata pointing to:

* Original PDF filename
* Page number
* Chunk text

### 5. Index Storage

The index is saved in a standard LangChain FAISS format so it can be used by:

* Reasoning Layer
* UI / Gradio Demo
* Evaluation notebooks

---

## Example Retrieval Call

```python
docs = retriever.get_relevant_documents("What is the main idea?")
for d in docs:
    print(d.metadata["source"], d.page_content[:200])
```

---

## Notes for Group Members

* You **do not** need to re-run the notebook unless new PDFs are added.
* The `index_store/` folder should be committed (already git-safe).
* The notebook allows new team members to rebuild the index in minutes.



```bash
python3 examples/test_prompts.py
python3 examples/test_reasoning.py
python3 examples/test_llm_client.py
```
