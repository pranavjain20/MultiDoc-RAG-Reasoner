# Multi-Document RAG Reasoner

A retrieval-augmented generation system that performs **document-aware reasoning** across multiple PDFs. Unlike flat RAG pipelines that treat all retrieved chunks as undifferentiated context, this system classifies queries by type, builds structured prompts with explicit document boundaries, and enforces source citations.

> Built as a final project for **Columbia's COMS 4995 (Applied Machine Learning)** with Cheng Wu, Jaewon Cho, and Winston Li. This fork contains my post-class improvements focused on code quality, testability, and reproducibility.

---

## My Contributions

### Original Project
- Designed and implemented the **LLM reasoning layer** — the core of the system:
  - `MultiDocReasoner`: query classification (synthesis / comparison / extraction), chunk organization, lost-in-the-middle mitigation
  - Structured prompt builders with document-level grouping and `[DocumentName.pdf]` citation enforcement
  - `LLMClient` for HuggingFace Inference API integration
- Wrote the initial test suite and project documentation

### Fork Improvements
- Migrated to proper Python packaging (`pyproject.toml`) — eliminates manual `PYTHONPATH` hacks
- Replaced print-statement pseudo-tests with a real **pytest** suite with assertions, mocking, and edge case coverage
- Fixed silent exception swallowing in the LLM client fallback chain — errors are now logged instead of silently discarded
- Translated non-English comments, removed dead code, normalized prompt formatting
- Pinned all dependencies for reproducible installs
- Removed 17MB of tracked binary artifacts from the repo

---

## Architecture

```
PDF Documents
     |
     v
[PyPDF Loader]  -->  [Text Chunker]  -->  [SBERT Embeddings]  -->  [FAISS Index]
                      800 chars              all-MiniLM-L6-v2
                      150 overlap

User Query
     |
     v
[FAISS Retrieval]  -->  [Query Classifier]  -->  [Prompt Builder]  -->  [LLM]  -->  Response
     top-6 chunks        synthesis |               document-grouped       Groq /
                         comparison |              sections with           HuggingFace /
                         extraction                citation rules         local FLAN-T5
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/pranavjain20/MultiDoc-RAG-Reasoner.git
cd MultiDoc-RAG-Reasoner
pip install -e ".[dev]"

# Set up API keys
echo "HF_TOKEN=your_hf_token_here" > .env
# Optional: echo "GROQ_API_KEY=your_groq_key" >> .env

# Place PDFs in evaluation_files/
cp your_documents/*.pdf evaluation_files/

# Build the FAISS index
python examples/build_index.py

# Launch the web UI
python examples/build_UI.py
```

Or use the Makefile:

```bash
make install    # pip install -e ".[dev]"
make test       # pytest tests/ -v
make index      # build FAISS index
make ui         # launch Gradio UI
make eval       # run E1-E4 evaluation suite
```

---

## Repository Structure

```
src/llm/                          Core reasoning module
  reasoning.py                    MultiDocReasoner: classify, organize, mitigate
  prompts.py                      Prompt builders (synthesis / comparison / extraction)
  llm_client.py                   Multi-backend LLM client (Groq -> HF -> local)
  llm_api_groq.py                 Groq API wrapper

examples/                         Runnable scripts
  build_index.py                  PDF ingestion + FAISS index builder
  build_UI.py                     Gradio web interface
  evaluation.py                   E1-E4 evaluation driver
  export_evaluation_chunks.py     Retrieval -> export top-k chunks

tests/                            pytest suite
  test_prompts.py                 Prompt builder tests
  test_reasoning.py               Reasoning module tests
  test_llm_client.py              LLM client tests (mocked API calls)
  conftest.py                     Shared fixtures
```

---

## How It Works

### Query Classification

`MultiDocReasoner.classify_query` routes each question to a specialized prompt strategy:

| Query Type | Triggers | Prompt Strategy |
|---|---|---|
| **Synthesis** | "summarize", "main ideas", "overview" | Groups findings by theme across all documents |
| **Comparison** | "compare", "differ", "vs" | Analyzes common ground and key differences |
| **Extraction** | "assumptions", "limitations", "variables" | Targeted extraction with source citations |

### Document-Aware Prompting

Retrieved chunks are grouped by source document before prompt construction:

```
--- doc1.pdf ---
[chunk text from doc1]

--- doc2.pdf ---
[chunk text from doc2]
```

Each prompt includes explicit instructions to cite sources as `[DocumentName.pdf]`, forcing the LLM to attribute claims to specific documents rather than generating unsourced summaries.

### Lost-in-the-Middle Mitigation

Transformers tend to underweight tokens in the middle of long contexts. The optional `mitigate_lost_in_middle` flag reorders chunks so that the highest-relevance content appears at the beginning and end of the prompt.

### LLM Fallback Chain

The client tries backends in order: **Groq** (fast, high-quality) -> **HuggingFace Inference API** -> **local FLAN-T5-base** (offline fallback). Failures at each level are logged and the next backend is tried automatically.

---

## Web UI

An interactive Gradio interface for the full pipeline is available in `examples/build_UI.py`.

- **Setup Tab**: Upload PDFs and build the FAISS index
- **Query Tab**: Ask questions and view answers with detected query type, supporting evidence chunks, and citation-aware responses

```bash
python examples/build_UI.py
# Opens at http://localhost:7860
```

---

## Evaluation Suite

Four experiments test different aspects of the system:

| Experiment | What It Tests | Output |
|---|---|---|
| **E1** | Query type classification accuracy (5 gold questions) | `e1_query_type.json` |
| **E2** | Prompt structure sanity (document names, grouping, keywords) | `e2_prompt_sanity.json` |
| **E3** | Baseline (flat) vs structured reasoning prompts | `e3_baseline_vs_reasoning.json` |
| **E4** | Lost-in-the-middle mitigation ablation | `e4_lost_in_middle.json` |

```bash
# Export top-k chunks, then run all experiments
python examples/export_evaluation_chunks.py
python examples/evaluation.py
```

Results are saved as JSON in `evaluation_outputs/` for inspection.

---

## Configuration

| Setting | Default | Location |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | `build_index.py`, `export_evaluation_chunks.py` |
| Chunk size / overlap | 800 / 150 | `build_index.py` |
| Retrieval top-k | 6 | `export_evaluation_chunks.py` |
| LLM temperature | 0.2 | `llm_client.py` |
| Max tokens | 512 | `llm_client.py` |
| Groq model | `llama-3.3-70b-versatile` | `llm_api_groq.py` |
| HF model | `google/flan-t5-large` | `llm_client.py` |

---

## Testing

```bash
pytest tests/ -v                        # all unit tests
pytest tests/ -v -m "not integration"   # skip tests that call external APIs
```
