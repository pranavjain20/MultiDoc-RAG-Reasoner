"""
Export top-k chunks per evaluation question into evaluation_outputs/evaluation_chunks.json.

This script:
- loads the FAISS index from index_store/
- builds a retriever using the same embedding model as in AML_Project.ipynb
- runs retrieval for each evaluation question
- saves results in the format expected by examples/evaluation.py:

{
  "q1": [
    {"doc_name": "doc1.pdf", "text": "..."},
    ...
  ],
  "q2": [...],
  ...
}
"""

import os
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Where outputs will be written
OUTPUT_DIR = "evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FAISS index directory (already created by examples/build_index.py or AML_Project.ipynb)
INDEX_DIR = "index_store"

# Embedding model name — must match the one used to build the index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Same questions as in evaluation.py
EVAL_QUESTIONS = {
    "q1": "Summarize the main ideas discussed across these documents.",
    "q2": "What are the main sources of risk mentioned across the documents?",
    "q3": "Compare how different documents describe the same concept or methodology?",
    "q4": "What are the key assumptions and limitations highlighted in these documents?",
    "q5": "How do the documents differ in their conclusions or policy implications?",
}

TOP_K = 6  # number of chunks per question


def build_retriever():
    """
    Load FAISS index and return a retriever.

    We use HuggingFaceEmbeddings with model_name so that the embedding
    dimension matches the one used when building the index.
    """
    if not os.path.isdir(INDEX_DIR):
        raise FileNotFoundError(
            f"{INDEX_DIR} not found. Make sure you ran examples/build_index.py "
            "or copied the 'index_store' folder into the project root."
        )

    # New style: pass model_name instead of a SentenceTransformer instance
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Load FAISS index
    db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return db.as_retriever(search_kwargs={"k": TOP_K})


def main():
    retriever = build_retriever()
    all_chunks = {}

    for qid, question in EVAL_QUESTIONS.items():
        docs = retriever.get_relevant_documents(question)

        # Map LangChain docs -> simple dicts with doc_name + text
        chunks = []
        for d in docs:
            # Common LangChain convention: metadata["source"] = original PDF path
            doc_name = d.metadata.get("source", "Unknown.pdf")
            text = d.page_content
            chunks.append({"doc_name": doc_name, "text": text})

        all_chunks[qid] = chunks
        print(f"{qid}: retrieved {len(chunks)} chunks")

    out_path = os.path.join(OUTPUT_DIR, "evaluation_chunks.json")
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nSaved evaluation chunks to {out_path}")


if __name__ == "__main__":
    main()








