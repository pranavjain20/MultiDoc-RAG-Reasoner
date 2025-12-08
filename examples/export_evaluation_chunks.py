"""
Export top-k chunks per evaluation question into evaluation_chunks.json.

This script:
- loads the FAISS index from index_store/
- builds a retriever
- runs retrieval for each evaluation question
- saves results in the format expected by examples/evaluation.py.
"""

import os
import json

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


OUTPUT_DIR = "evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_DIR = "index_store"          # where AML_Project.ipynb saved the FAISS index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Same questions as in evaluation.py
EVAL_QUESTIONS = {
    "q1": "Summarize the main ideas discussed across these documents.",
    "q2": "What are the main sources of risk mentioned across the documents?",
    "q3": "Compare how different documents describe the same concept or methodology.",
    "q4": "What are the key assumptions and limitations highlighted in these documents?",
    "q5": "How do the documents differ in their conclusions or policy implications?",
}

TOP_K = 6  # number of chunks per question


def build_retriever():
    """
    Load FAISS index and return a retriever.
    This matches the pattern described in README.md.
    """
    # Build embedding model (must match what was used in AML_Project.ipynb)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model=encoder)

    # Load FAISS index
    db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    return retriever


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
