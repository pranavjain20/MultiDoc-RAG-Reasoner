# UI code

import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple, Dict

import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm.reasoning import MultiDocReasoner
from llm.llm_client import LLMClient
from llm.llm_api_groq import DEFAULT_MODEL as GROQ_MODEL

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 6
RETRIEVAL_FETCH_K = max(RETRIEVAL_K * 3, 12)

INDEX_STORE_PATH = "index_store"
UPLOAD_DIR = "ui_uploads"

# =========================================================
# Chunk filtering (CRITICAL FIX)
# =========================================================

_BAD_PATTERNS = [
    r"\breferences\b",
    r"\bbibliography\b",
    r"\bworks cited\b",
    r"\bappendix\b",
    r"\backnowledg(e)?ments?\b",
    r"\bproceedings\b",
    r"\bassociation for computational linguistics\b",
    r"\bet al\.\b",
    r"\bdoi\b",
    r"\barxiv\b",
    r"\bjournal\b",
    r"\bvol\.\b",
    r"\bno\.\b",
    r"\bpages?\b",
]

def _looks_like_references(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 60:
        return True

    low = t.lower()
    if any(re.search(p, low) for p in _BAD_PATTERNS):
        return True

    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", t))
    if year_hits >= 3:
        return True

    bracket_hits = len(re.findall(r"\[\d+\]", t))
    if bracket_hits >= 3:
        return True

    digits = sum(ch.isdigit() for ch in t)
    punct = sum(ch in ".,;:()[]{}" for ch in t)
    ratio = (digits + punct) / max(len(t), 1)
    if ratio > 0.22:
        return True

    return False

def filter_chunks(chunks: List[Dict], keep: int = RETRIEVAL_K) -> List[Dict]:
    clean = []
    for c in chunks:
        txt = (c.get("text") or "").strip()
        if not _looks_like_references(txt):
            clean.append(c)
        if len(clean) >= keep:
            break
    return clean


# =========================================================
# Uploading File
# =========================================================

def save_uploads(files) -> List[str]:
    if not files:
        return []

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_paths = []

    for file in files:
        if file is None:
            continue
        filename = os.path.basename(file.name)
        dest_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copy(file.name, dest_path)
        saved_paths.append(dest_path)

    return saved_paths


def build_index_from_pdfs(pdf_paths: List[str]) -> str:
    try:
        if not pdf_paths:
            return "No files uploaded"

        all_docs = []
        doc_count = 0

        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_path)
            all_docs.extend(docs)
            doc_count += 1

        if not all_docs:
            return "No pages loaded from PDFs"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        os.makedirs(INDEX_STORE_PATH, exist_ok=True)
        vectorstore.save_local(INDEX_STORE_PATH)

        return f"Index created\n{len(all_docs)} pages \u2192 {len(chunks)} chunks from {doc_count} documents"

    except Exception as e:
        return f"Build failed: {str(e)}"


# =========================================================
# Retriever
# =========================================================

def load_retriever():
    try:
        if not os.path.exists(INDEX_STORE_PATH):
            return None

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            INDEX_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        return vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_FETCH_K}
        )

    except Exception as e:
        print(f"Error loading retriever: {e}")
        return None


def retrieve_chunks(question: str, retriever) -> List[Dict]:
    if not retriever:
        return []

    docs = retriever.invoke(question)

    chunks = []
    for doc in docs:
        chunks.append(
            {
                "doc_name": doc.metadata.get("source", "Unknown.pdf"),
                "text": doc.page_content,
            }
        )

    return filter_chunks(chunks, keep=RETRIEVAL_K)


# =========================================================
# QA Pipeline
# =========================================================

def answer_question(question: str) -> Tuple[str, str, str]:
    if not os.path.exists(INDEX_STORE_PATH):
        return ("Build an index first (Setup tab)", "", "")

    if not question or not question.strip():
        return ("Enter a question", "", "")

    retriever = load_retriever()
    if not retriever:
        return ("Failed to load index", "", "")

    chunks = retrieve_chunks(question, retriever)
    if not chunks:
        return ("No relevant information found", "", "")

    if not (os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN")):
        return (
            "Missing GROQ_API_KEY or HF_TOKEN in .env (need at least one)",
            "",
            format_evidence(chunks),
        )

    try:
        client = LLMClient()
        reasoner = MultiDocReasoner()

        result = client.generate_with_reasoning(
            question=question,
            chunks=chunks,
            reasoner=reasoner,
        )

        return (
            result.get("response", "No answer generated"),
            result.get("query_type", "unknown"),
            format_evidence(chunks),
        )

    except Exception as e:
        return (
            f"API Error: {str(e)}",
            "",
            format_evidence(chunks),
        )


def format_evidence(chunks: List[Dict]) -> str:
    parts = [f"Retrieved {len(chunks)} chunks:\n"]
    for i, c in enumerate(chunks, 1):
        preview = c["text"][:250] + ("..." if len(c["text"]) > 250 else "")
        parts.append(f"\n[{i}] {c['doc_name']}\n{preview}\n")
    return "\n".join(parts)


# =========================================================
# UI Design System
# =========================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300..800&family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&family=JetBrains+Mono:wght@400;500&display=swap');

:root, .dark {
    --primary: #818cf8;
    --primary-hover: #a5b4fc;
    --primary-dim: #6366f1;
    --primary-glow: rgba(129, 140, 248, 0.15);
    --primary-ghost: rgba(129, 140, 248, 0.08);
    --surface: #161b22;
    --surface-raised: #1c2129;
    --page-bg: #0d1117;
    --text: #e6edf3;
    --text-secondary: #b1bac4;
    --text-muted: #7d8590;
    --border: #30363d;
    --border-light: #21262d;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
    --radius: 8px;
    --radius-lg: 12px;
    --font-display: 'Newsreader', Georgia, serif;
    --font-body: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --ease: cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Force Dark Foundation ── */
body, .gradio-container, .main, .contain {
    background: var(--page-bg) !important;
    color: var(--text) !important;
}
.gradio-container {
    font-family: var(--font-body) !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 2rem !important;
}
.main { max-width: 100% !important; }
footer { display: none !important; }

/* Override ALL Gradio dark backgrounds */
.block, .wrap, .panel, .form, .container {
    background: transparent !important;
}
.block.padded {
    background: var(--surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
}

/* ── Typography ── */
h1, h2, h3, h4, p, span, label {
    color: var(--text) !important;
}
h1, h2, h3 {
    font-family: var(--font-display) !important;
    letter-spacing: -0.02em !important;
    font-weight: 500 !important;
}

/* ── Tabs ── */
.tab-nav {
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
    gap: 0 !important;
    padding: 0 !important;
}
.tab-nav button {
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    color: var(--text-muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.5rem !important;
    margin: 0 !important;
    background: transparent !important;
    transition: all 0.2s var(--ease) !important;
    border-radius: 0 !important;
}
.tab-nav button.selected {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
    background: transparent !important;
}
.tab-nav button:hover:not(.selected) {
    color: var(--text-secondary) !important;
}

/* ── Primary Button ── */
button.primary {
    background: var(--primary-dim) !important;
    color: #fff !important;
    border: none !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 2rem !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s var(--ease) !important;
    letter-spacing: 0.01em !important;
    min-height: 46px !important;
}
button.primary:hover {
    background: var(--primary) !important;
    box-shadow: var(--shadow-md), 0 0 0 3px var(--primary-glow) !important;
    transform: translateY(-1px) !important;
}
button.primary:active {
    transform: translateY(0) !important;
}

/* ── Example Pill Buttons ── */
.example-pill button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 1rem !important;
    border-radius: 20px !important;
    transition: all 0.2s var(--ease) !important;
    min-height: unset !important;
}
.example-pill button:hover {
    background: var(--primary-ghost) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* ── Inputs ── */
textarea, input[type="text"], input[type="search"] {
    font-family: var(--font-body) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-size: 0.92rem !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease) !important;
}
textarea:focus, input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
    outline: none !important;
}

/* ── Labels ── */
label > span, .label-wrap > span {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Upload Zone ── */
.upload-area {
    border-radius: var(--radius-lg) !important;
    background: var(--surface) !important;
    border-color: var(--border) !important;
}

/* ── Answer Card ── */
.answer-card textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--primary-dim) !important;
    border-radius: var(--radius) !important;
    line-height: 1.75 !important;
    color: var(--text) !important;
}

/* ── Query Type Badge ── */
.query-badge input {
    font-family: var(--font-body) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--primary) !important;
    background: var(--primary-ghost) !important;
    border: 1px solid var(--primary-glow) !important;
    border-radius: 20px !important;
    text-align: center !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Evidence Display ── */
.evidence-box textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    background: var(--surface-raised) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
    line-height: 1.65 !important;
    color: var(--text-secondary) !important;
}

/* ── Build Status ── */
.build-status textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    background: var(--surface-raised) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
    color: var(--text-secondary) !important;
}

/* ── Accordion ── */
.gradio-accordion {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
}
.gradio-accordion > .label-wrap {
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    padding: 0.75rem 1rem !important;
}
/* Fix accordion arrow — flip so it points right when closed, down when open */
.gradio-accordion > .label-wrap .icon {
    transform: rotate(90deg) !important;
    transition: transform 0.2s var(--ease) !important;
}
.gradio-accordion > .label-wrap.open .icon {
    transform: rotate(180deg) !important;
}

/* ── Markdown ── */
.prose, .markdown-text, .md, .markdown p, .markdown li {
    font-family: var(--font-body) !important;
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}
.markdown strong, .markdown b {
    color: var(--text) !important;
}

/* ── Section Labels ── */
.section-label h3 {
    font-family: var(--font-display) !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}

/* ── Config Table ── */
.gradio-accordion table {
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
    border-color: var(--border-light) !important;
}
.gradio-accordion table th {
    font-weight: 600 !important;
    color: var(--text) !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    border-color: var(--border) !important;
}
.gradio-accordion table td {
    border-color: var(--border-light) !important;
}

/* ── Load Animation ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.app-header { animation: fadeIn 0.5s ease-out both; }
.tabitem { animation: fadeIn 0.3s ease-out 0.1s both; }

/* ── Scrollbar ── */
textarea::-webkit-scrollbar { width: 5px; }
textarea::-webkit-scrollbar-track { background: transparent; }
textarea::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
textarea::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Row Spacing ── */
.row { gap: 1.5rem !important; }

/* ── Action Buttons (not full-width per Stitch design) ── */
.action-btn {
    max-width: 200px !important;
}
.action-btn button {
    width: 100% !important;
}

/* ── Config Fields ── */
.config-field input, .config-field textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    background: var(--surface-raised) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-secondary) !important;
}
"""


# =========================================================
# UI Layout
# =========================================================

FORCE_DARK_JS = """
() => {
    document.body.classList.add('dark');
    document.documentElement.style.setProperty('color-scheme', 'dark');
}
"""

def create_demo():
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
    )
    with gr.Blocks(title="Multi-Doc RAG Assistant", theme=theme, js=FORCE_DARK_JS) as demo:

        gr.HTML(f"<style>{custom_css}</style>")

        # ── Header ──
        gr.HTML("""
        <div class="app-header" style="text-align: center; padding: 2.5rem 0 1.5rem;">
            <div style="width: 36px; height: 3px; background: #818cf8; margin: 0 auto 1.25rem; border-radius: 2px;"></div>
            <h1 style="
                font-family: 'DM Sans', sans-serif;
                font-size: 2.2rem;
                font-weight: 600;
                color: #e6edf3;
                letter-spacing: -0.03em;
                margin: 0 0 0.4rem;
                line-height: 1.2;
            ">Multi-Document RAG Assistant</h1>
            <p style="
                font-family: 'DM Sans', sans-serif;
                color: #7d8590;
                font-size: 0.95rem;
                font-weight: 400;
                margin: 0;
            ">AI-powered question answering across multiple PDF documents</p>
        </div>
        """)

        with gr.Tabs():

            # ── Setup Tab ──
            with gr.Tab("Setup"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Documents", elem_classes=["section-label"])
                        file_upload = gr.File(
                            label="PDF Documents",
                            file_count="multiple",
                            file_types=[".pdf"],
                            type="filepath",
                            elem_classes=["upload-area"],
                            height=220,
                        )
                        upload_status = gr.Textbox(
                            label="Status",
                            value="No files uploaded",
                            interactive=False,
                            lines=1,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Build Index", elem_classes=["section-label"])
                        gr.Markdown(
                            "Process uploaded PDFs into a searchable vector index "
                            "for retrieval-augmented generation."
                        )
                        build_btn = gr.Button(
                            "Build Index",
                            variant="primary",
                            elem_classes=["action-btn"],
                        )
                        build_output = gr.Textbox(
                            label="Build Log",
                            interactive=False,
                            lines=6,
                            elem_classes=["build-status"],
                        )

                with gr.Accordion("Advanced Configuration", open=False):
                    with gr.Row():
                        gr.Textbox(
                            label="Chunk Size",
                            value=f"{CHUNK_SIZE} chars",
                            interactive=False,
                            elem_classes=["config-field"],
                        )
                        gr.Textbox(
                            label="Overlap",
                            value=f"{CHUNK_OVERLAP} chars",
                            interactive=False,
                            elem_classes=["config-field"],
                        )
                        gr.Textbox(
                            label="Retrieval",
                            value=f"Top-{RETRIEVAL_K} (fetch {RETRIEVAL_FETCH_K})",
                            interactive=False,
                            elem_classes=["config-field"],
                        )
                    with gr.Row():
                        gr.Textbox(
                            label="Embedding Model",
                            value=EMBEDDING_MODEL.split("/")[-1],
                            interactive=False,
                            elem_classes=["config-field"],
                        )
                        gr.Textbox(
                            label="LLM Backend",
                            value=f"Groq ({GROQ_MODEL}) \u2192 HF (flan-t5-large) \u2192 Local (flan-t5-base)",
                            interactive=False,
                            elem_classes=["config-field"],
                        )

                # Event handlers
                def update_upload_status(files):
                    if not files:
                        return "No files uploaded"
                    count = len([f for f in files if f is not None])
                    return f"{count} file(s) ready"

                file_upload.change(
                    fn=update_upload_status,
                    inputs=[file_upload],
                    outputs=[upload_status],
                )

                def build_index_ui(files):
                    saved_paths = save_uploads(files)
                    return build_index_from_pdfs(saved_paths)

                build_btn.click(
                    fn=build_index_ui,
                    inputs=[file_upload],
                    outputs=[build_output],
                )

            # ── Query Tab ──
            with gr.Tab("Query"):

                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What are the main findings across all documents?",
                    lines=2,
                )

                ask_btn = gr.Button(
                    "Get Answer",
                    variant="primary",
                    elem_classes=["action-btn"],
                )

                gr.Markdown("**Try an example:**")
                with gr.Row(elem_classes=["example-pill"]):
                    example_btn1 = gr.Button(
                        "Main findings across documents", size="sm",
                    )
                    example_btn2 = gr.Button(
                        "Compare methodologies", size="sm",
                    )
                    example_btn3 = gr.Button(
                        "Key assumptions & limitations", size="sm",
                    )

                # Answer + query type
                with gr.Row():
                    with gr.Column(scale=5):
                        answer_output = gr.Textbox(
                            label="Answer",
                            interactive=False,
                            lines=12,
                            elem_classes=["answer-card"],
                        )
                    with gr.Column(scale=1, min_width=120):
                        query_type_output = gr.Textbox(
                            label="Query Type",
                            interactive=False,
                            lines=1,
                            elem_classes=["query-badge"],
                        )

                # Evidence
                with gr.Accordion("Supporting Evidence", open=False):
                    evidence_output = gr.Textbox(
                        label="Retrieved Chunks",
                        interactive=False,
                        lines=15,
                        elem_classes=["evidence-box"],
                    )

                # Event handlers
                ask_btn.click(
                    fn=answer_question,
                    inputs=[question_input],
                    outputs=[answer_output, query_type_output, evidence_output],
                )

                example_btn1.click(
                    lambda: "What are the main sources of risk mentioned across all documents?",
                    outputs=[question_input],
                )
                example_btn2.click(
                    lambda: "How do the documents differ in their treatment of the same concept or methodology?",
                    outputs=[question_input],
                )
                example_btn3.click(
                    lambda: "What assumptions and limitations are highlighted in the documents?",
                    outputs=[question_input],
                )

        # ── Footer ──
        gr.HTML("""
        <div style="
            margin-top: 3rem;
            padding: 1.25rem 0;
            border-top: 1px solid #30363d;
            text-align: center;
        ">
            <p style="
                font-family: 'DM Sans', sans-serif;
                font-size: 0.82rem;
                color: #7d8590;
                margin: 0 0 0.5rem;
            ">
                Built at <strong style="color: #b1bac4;">Columbia University</strong>
                &middot; COMS 4995 Applied Machine Learning
            </p>
            <div style="
                display: flex;
                justify-content: center;
                gap: 0.5rem;
                align-items: center;
            ">
                <span style="font-family: 'DM Sans'; font-size: 0.78rem; color: #7d8590;">Cheng Wu</span>
                <span style="color: #30363d;">&middot;</span>
                <span style="font-family: 'DM Sans'; font-size: 0.78rem; color: #7d8590;">Pranav Jain</span>
                <span style="color: #30363d;">&middot;</span>
                <span style="font-family: 'DM Sans'; font-size: 0.78rem; color: #7d8590;">Jaewon Cho</span>
                <span style="color: #30363d;">&middot;</span>
                <span style="font-family: 'DM Sans'; font-size: 0.78rem; color: #7d8590;">Winston Li</span>
            </div>
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False
    )
