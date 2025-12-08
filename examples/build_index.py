"""
Build a FAISS index from one or more PDFs and save it to `index_store/`.

This script mirrors the logic from AML_Project.ipynb:
- load PDF(s)
- split into chunks (size 800, overlap 150)
- embed with SBert (all-MiniLM-L6-v2)
- build a FAISS index and save it locally
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from sentence_transformers import SentenceTransformer


# TODO: set paths to the PDFs you want to index.
# For now I put a placeholder; change it to your actual file name(s).
PDF_PATHS: List[str] = [
    "evaluation_files/doc1.pdf",
    "evaluation_files/doc2.pdf",
]


INDEX_DIR = "index_store"


class SBertEmbeddings(Embeddings):
    """SBERT embeddings using all-MiniLM-L6-v2, same as in AML_Project.ipynb."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


def build_index():
    all_docs = []

    # 1) Load all PDFs
    for pdf_path in PDF_PATHS:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        all_docs.extend(raw_docs)

    print(f"Loaded {len(all_docs)} pages from {len(PDF_PATHS)} PDF(s).")

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    docs = splitter.split_documents(all_docs)
    print(f"Number of chunks: {len(docs)}")

    # 3) Build FAISS index
    embedding_function = SBertEmbeddings()
    db = FAISS.from_documents(docs, embedding_function)
    print("FAISS index built.")

    # 4) Save to index_store/
    db.save_local(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/")


def main():
    build_index()


if __name__ == "__main__":
    main()
