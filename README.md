# Multi-Document RAG Reasoning Module

LLM reasoning, document ingestion, and evaluation components for a multi-document
Retrieval-Augmented Generation (RAG) system.

This repo contains the core backend used in our COMS 4995 final project
(multi-document assistant with structured reasoning and evaluation).

---

## 1. Repository Structure

```text
├── evaluation_files/           # PDFs used for evaluation (doc1.pdf, doc2.pdf, ...)
├── evaluation_outputs/         # JSON outputs from evaluation pipeline (E1–E4)
│   ├── e1_query_type.json
│   ├── e2_prompt_sanity.json
│   ├── e3_baseline_vs_reasoning.json
│   ├── e4_lost_in_middle.json
│   └── evaluation_chunks.json
├── examples/
│   ├── build_index.py          # Offline ingestion + FAISS index builder
│   ├── export_evaluation_chunks.py   # Retrieval → export top-k chunks
│   ├── evaluation.py           # E1–E4 evaluation driver
│   ├── test_llm_client.py
│   ├── test_prompts.py
│   └── test_reasoning.py
├── src/llm/
│   ├── __init__.py
│   ├── llm_client.py           # HuggingFace API client
│   ├── prompts.py              # Prompt builders (synthesis/comparison/extraction)
│   └── reasoning.py            # MultiDocReasoner (routing + mitigation)
├── AML_Project.ipynb           # Original Colab prototype (single-notebook version)
├── requirements.txt
└── README.md
````

---

## 2. Setup

### 2.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your HuggingFace token:

```bash
HF_TOKEN=your_hf_token_here
```

> The code uses the public HuggingFace Inference API with
> `google/flan-t5-large`. If the endpoint is temporarily unavailable or
> rate-limited, evaluation scripts will still run and record the error
> message in the JSON outputs.

### 2.2 Python path

All example scripts assume `src/` is on `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

---

## 3. Document Ingestion & FAISS Index Builder

This component handles **PDF loading, text chunking, SBERT embeddings, and FAISS
index construction**. All downstream components (reasoning layer, evaluation,
future UI/memory modules) depend on this index.

We now provide a **scripted, reproducible** ingestion pipeline in
`examples/build_index.py` (no longer only in a Colab notebook).

### 3.1 Input PDFs

Place one or more PDF files into

```text
evaluation_files/
```

For our evaluation, we use:

```text
evaluation_files/doc1.pdf
evaluation_files/doc2.pdf
```

### 3.2 Build the FAISS Index

```bash
export PYTHONPATH=src
python examples/build_index.py
```

This script:

1. **Loads PDFs** using `PyPDFLoader`.
2. **Chunks text** with `RecursiveCharacterTextSplitter`:

   * `chunk_size = 800`
   * `chunk_overlap = 150`
3. **Embeds chunks** using SBERT:

   * `sentence-transformers/all-MiniLM-L6-v2`
4. **Builds a FAISS index** and saves it under:

```text
index_store/
    faiss_store.index
    index.pkl
    index.json
```

The index is stored in LangChain FAISS format so it can be reused by
the evaluation pipeline and any future UI / memory module.

### 3.3 Loading the Index in Code

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

db = FAISS.load_local(
    "index_store",
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = db.as_retriever(search_kwargs={"k": 6})
```

This `retriever` is what we use in `export_evaluation_chunks.py` and can
also be used in a UI to answer arbitrary user queries.

---

## 4. LLM Reasoning Layer

The reasoning layer is responsible for:

1. Classifying the user query into a **query type**
   (`synthesis`, `comparison`, or `extraction`).
2. Building a **structured prompt** that:

   * groups text by document,
   * enforces citation rules `"[DocumentName.pdf]"`,
   * optionally applies a **lost-in-the-middle mitigation** strategy.
3. Calling the LLM via the HuggingFace Inference API.

### 4.1 Core Classes

All reasoning components live in `src/llm/`:

* `prompts.py`

  * `SYSTEM_PROMPT`
  * `build_synthesis_prompt(question, chunks)`
  * `build_comparison_prompt(question, chunks_by_doc)`
  * `build_extraction_prompt(question, chunks)`
  * `select_prompt_builder(query_type)`
* `reasoning.py`

  * `MultiDocReasoner`

    * `classify_query(question)` → `"synthesis" | "comparison" | "extraction"`
    * `organize_chunks_by_doc(chunks)`
    * `mitigate_lost_in_middle(chunks)` (optional reordering)
    * `build_prompt(question, chunks, apply_lost_in_middle=False)`
* `llm_client.py`

  * `LLMClient`

    * `generate(...)` – low-level HF API call with retry logic.
    * `generate_with_reasoning(question, chunks, reasoner)` – high-level wrapper.

### 4.2 Chunk Input Format

The reasoning layer expects a list of chunk dictionaries with at least
`doc_name` and `text`:

```python
chunks = [
    {
        "doc_name": "doc1.pdf",     # name of source document
        "text": "chunk text here",  # text content
        "score": 0.95,              # optional: retrieval score (for future use)
    },
    ...
]
```

These structures are exactly what `export_evaluation_chunks.py` produces
from FAISS retrieval.

### 4.3 Basic Usage Example

```python
from llm import LLMClient, MultiDocReasoner

reasoner = MultiDocReasoner()
client = LLMClient()  # uses HF_TOKEN from .env

chunks = [
    {"doc_name": "doc1.pdf", "text": "First document content ..."},
    {"doc_name": "doc2.pdf", "text": "Second document content ..."},
]

result = client.generate_with_reasoning(
    question="Summarize the main ideas across these documents.",
    chunks=chunks,
    reasoner=reasoner,
)

print("Answer:\n", result["response"])
print("Query type:", result["query_type"])   # "synthesis" / "comparison" / "extraction"
```

### 4.4 Query Type Heuristics

`MultiDocReasoner.classify_query` uses simple rule-based heuristics over
the question text:

* **Synthesis**

  * e.g., “Summarize…”, “What are the main sources…”, “Give an overview…”
* **Comparison**

  * uses keywords like “compare”, “difference(s)”, “vs.”, “relative to…”
* **Extraction**

  * focuses on “assumptions”, “limitations”, “variables”, “definitions”.

The mapping is tested in `examples/test_reasoning.py` and achieves 100%
accuracy on our five evaluation questions.

---

## 5. Evaluation Suite (E1–E4)

We provide a small but fully scripted evaluation suite in
`examples/evaluation.py`. It operates on the real system components and
records results as JSON artefacts.

### 5.1 Step 1 – Export Top-k Chunks for Evaluation

After building the FAISS index, run:

```bash
export PYTHONPATH=src
python examples/export_evaluation_chunks.py
```

This script:

* loads `index_store/` and builds a retriever,
* uses five canonical evaluation questions (Q1–Q5),
* retrieves `k = 6` chunks per question,
* saves them as:

```text
evaluation_outputs/evaluation_chunks.json
```

The format is:

```json
{
  "q1": [
    {"doc_name": "doc1.pdf", "text": "..."},
    {"doc_name": "doc2.pdf", "text": "..."}
  ],
  "q2": [...],
  ...
}
```

### 5.2 Step 2 – Run Full Evaluation

```bash
export PYTHONPATH=src
python examples/evaluation.py
```

This runs four experiments:

#### E1 — Query Type Classification Accuracy

* Uses `MultiDocReasoner.classify_query`.

* Compares predicted type with gold labels for Q1–Q5.

* Saves:

  ```text
  evaluation_outputs/e1_query_type.json
  ```

* On our evaluation set, accuracy is **1.00 (5/5)**.

#### E2 — Prompt-Structure Sanity Checks

* Builds toy prompts via `build_synthesis_prompt`, `build_comparison_prompt`,
  and `build_extraction_prompt`.
* Checks for:

  * presence of `doc1.pdf` and `doc2.pdf` in prompts,
  * proper document grouping,
  * mention of “assumptions” in extraction instructions.
* Saves:

  ```text
  evaluation_outputs/e2_prompt_sanity.json
  ```

All checks are `True`, verifying that the templates expose document-level
structure and citation instructions.

#### E3 — Baseline vs Structured Reasoning Prompts

For each evaluation question:

1. Construct a **baseline prompt**:

   * flat concatenation of all retrieved chunk texts,
   * minimal instructions.

2. Construct a **structured reasoning prompt**:

   * built via `MultiDocReasoner.build_prompt`,
   * document-grouped sections (`--- doc1.pdf ---`, `--- doc2.pdf ---`),
   * explicit cross-document reasoning instructions,
   * citation rules `[DocumentName.pdf]`.

Both prompts are sent to `LLMClient.generate` (when the HF endpoint is
available). The script records:

* question text and gold query type,
* full baseline + reasoning prompts,
* corresponding model answers (or error messages if the API fails).

Output:

```text
evaluation_outputs/e3_baseline_vs_reasoning.json
```

This file is designed for **qualitative inspection**: instructors can
directly compare baseline vs structured prompts and their answers.

#### E4 — Lost-in-the-Middle Ablation

Long contexts often under-weight tokens in the middle. We implement a simple
mitigation flag `apply_lost_in_middle` in `MultiDocReasoner.build_prompt`:

* If `False`: chunks are concatenated in original order.
* If `True`:

  * chunks are grouped by document,
  * high-score chunks are prioritized near the beginning and end,
  * chunks are interleaved across documents.

For Q1, Q3, and Q5, E4 builds:

* a **no-mitigation prompt**, and
* a **mitigated prompt**,

then records both prompts and answers in:

```text
evaluation_outputs/e4_lost_in_middle.json
```

Again, this is intended for qualitative analysis of how prompt structure
affects the model’s use of evidence.

---

## 6. End-to-End Recipe

To reproduce our full pipeline from scratch:

```bash
# 0. Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
echo "HF_TOKEN=your_hf_token_here" > .env

# 1. (Optional) Place your own PDFs into evaluation_files/
# 2. Build FAISS index
python examples/build_index.py

# 3. Export top-k chunks for evaluation
python examples/export_evaluation_chunks.py

# 4. Run evaluation (E1–E4)
python examples/evaluation.py
```

All results will appear in `evaluation_outputs/`.

---

## 7. Web UI (Gradio Demo)

We provide an interactive web interface for the multi-document RAG system in `examples/demo_app.py`.

### 7.1 Features

- **Setup Tab**: Upload PDFs and build FAISS index
- **Query Tab**: Ask questions and view AI-generated answers with:
  - Detected query type (synthesis/comparison/extraction)
  - Supporting evidence from retrieved chunks
  - Citation-aware responses

### 7.2 Running the UI
```bash
export PYTHONPATH=src
python examples/demo_app.py
```

The interface will launch at `http://localhost:7860`.

### 7.3 Usage

1. **Upload Documents** (Setup tab):
   - Select one or more PDF files
   - Click "Build Index" to process documents
   
2. **Ask Questions** (Query tab):
   - Enter your question
   - View the answer with query type classification
   - Inspect supporting evidence chunks

### 7.4 Architecture

The UI directly uses the existing backend components:
- `MultiDocReasoner` for query classification and prompt building
- `LLMClient` for HuggingFace API calls (google/flan-t5-large)
- FAISS retriever (k=6) from `index_store/`
- Same chunking config as evaluation pipeline (chunk_size=800, overlap=150)

### 7.5 Requirements

Make sure to install the additional UI dependency:
```bash
pip install gradio langchain-huggingface
```

And set your HuggingFace token in `.env`:
```
HF_TOKEN=your_hf_token_here
```

### 7.6 Design

The UI features a clean, professional design with:
- Gradient header with project branding
- Two-tab workflow (Setup → Query)
- Team credits footer (COMS 4995 Final Project)
---

## 8. Testing

We include lightweight tests for each major component.

```bash
export PYTHONPATH=src

# Prompt builders (SYSTEM_PROMPT, synthesis/comparison/extraction templates)
python examples/test_prompts.py

# Reasoning module (query classification, chunk grouping, mitigation logic)
python examples/test_reasoning.py

# LLM client (initialization + basic API call, with graceful error handling)
python examples/test_llm_client.py
```

If the HuggingFace endpoint is down or rate-limited, `test_llm_client.py`
will still confirm that `LLMClient` initializes correctly and will surface the
API error without crashing other scripts.

---

## 9. Notes and Future Work

* The repository currently focuses on **multi-document reasoning** and a
  reproducible evaluation framework.
* A conversational **memory module** and **UI (e.g., Streamlit / Gradio)**
  can be built on top of:

  * the FAISS retriever (`index_store/`),
  * the `MultiDocReasoner`,
  * and `LLMClient`.

These components are intentionally decoupled so they can be easily reused in
other projects or extended for more advanced evaluations.

