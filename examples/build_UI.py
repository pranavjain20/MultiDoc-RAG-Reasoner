#UI code

import os
import shutil
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

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 6
INDEX_STORE_PATH = "index_store"
UPLOAD_DIR = "ui_uploads"

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
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = os.path.basename(pdf_path)
                all_docs.extend(docs)
                doc_count += 1
            except Exception as e:
                return f"Error loading {os.path.basename(pdf_path)}: {str(e)}"
        
        if not all_docs:
            return "No pages loaded from PDFs"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_documents(all_docs)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Build and save index
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(INDEX_STORE_PATH, exist_ok=True)
        vectorstore.save_local(INDEX_STORE_PATH)
        
        return f"Index created\n{len(all_docs)} pages → {len(chunks)} chunks from {doc_count} documents"
        
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
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
    except Exception as e:
        print(f"Error loading retriever: {e}")
        return None


def retrieve_chunks(question: str, retriever) -> List[Dict]:
    if not retriever:
        return []
    
    try:
        docs = retriever.invoke(question)
        chunks = []
        for doc in docs:
            chunk = {
                "doc_name": doc.metadata.get('source', 'Unknown.pdf'),
                "text": doc.page_content
            }
            chunks.append(chunk)
        return chunks
        
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []


def answer_question(question: str) -> Tuple[str, str, str]:
    try:
        # Validation
        if not os.path.exists(INDEX_STORE_PATH):
            return (
                "Build an index first (Setup tab)",
                "",
                ""
            )
        
        if not question or not question.strip():
            return ("Enter a question", "", "")
        
        # Load retriever
        retriever = load_retriever()
        if not retriever:
            return ("Failed to load index", "", "")
        
        # Retrieve chunks
        chunks = retrieve_chunks(question, retriever)
        if not chunks:
            return ("No relevant information found", "", "")
        
        if not os.getenv("HF_TOKEN"):
            return (
                "Missing HF_TOKEN in .env file",
                "",
                format_evidence(chunks)
            )
        
        # Generate answer
        try:
            client = LLMClient()
            reasoner = MultiDocReasoner()
            
            result = client.generate_with_reasoning(
                question=question,
                chunks=chunks,
                reasoner=reasoner
            )
            
            answer = result.get('response', 'No answer generated')
            query_type = result.get('query_type', 'unknown')
            evidence = format_evidence(chunks)
            
            return (answer, query_type, evidence)
            
        except Exception as e:
            error_msg = f"API Error: {str(e)}\n\nPossible causes:\n• Invalid HF_TOKEN\n• Rate limits\n• Model loading (wait 20s)"
            return (error_msg, "", format_evidence(chunks))
        
    except Exception as e:
        return (f"Error: {str(e)}", "", "")


def format_evidence(chunks: List[Dict]) -> str:
    if not chunks:
        return "No evidence"
    
    parts = [f"Retrieved {len(chunks)} chunks:\n"]
    
    for i, chunk in enumerate(chunks, 1):
        doc_name = chunk.get('doc_name', 'Unknown')
        text = chunk.get('text', '')
        preview = text[:250] + "..." if len(text) > 250 else text
        parts.append(f"\n[{i}] {doc_name}\n{preview}\n")
    
    return "\n".join(parts)


# =========================================================
# Design with CSS
# =========================================================
custom_css = """
/* Import clean fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global overrides - Center everything */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 95% !important;
    width: 95% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* Main content centering */
.main {
    max-width: 100% !important;
    margin: 0 auto !important;
}

/* Headers */
h1, h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

h1 {
    font-size: 2rem !important;
    margin-bottom: 0.5rem !important;
}

.subtitle {
    color: #666 !important;
    font-size: 0.95rem !important;
    margin-bottom: 2rem !important;
}

/* Tabs */
.tab-nav button {
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 500 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.05rem !important;
    min-height: 50px !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

button.secondary {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    color: #495057 !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
}

button.secondary:hover {
    background: #e9ecef !important;
}

/* Textboxes */
textarea, input {
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
    font-size: 1rem !important;
}

textarea {
    min-height: 60px !important;
}

textarea:focus, input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Code/Evidence blocks */
.monospace {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    background: #f8f9fa !important;
    border-radius: 6px !important;
}

/* File upload */
.file-upload {
    border: 2px dashed #d0d0d0 !important;
    border-radius: 8px !important;
    transition: border-color 0.3s !important;
    min-height: 220px !important;
    padding: 1.5rem !important;
}

.file-upload:hover {
    border-color: #667eea !important;
}

/* File upload container - target Gradio's internal classes */
.file-preview {
    min-height: 400px !important;
}

label:has(> .file-upload) {
    min-height: 400px !important;
}

/* Force consistent file upload height */
.upload-container, .file-container {
    min-height: 400px !important;
    max-height: 400px !important;
}

/* Gradio file component wrapper */
div[data-testid="file"] {
    min-height: 400px !important;
}

.wrap.svelte-1ipelgc {
    min-height: 400px !important;
}

/* Status messages */
.success {
    color: #28a745 !important;
}

.error {
    color: #dc3545 !important;
}

/* Spacing */
.gap-sm {
    margin-bottom: 1rem !important;
}

.gap-md {
    margin-bottom: 1.5rem !important;
}

.gap-lg {
    margin-bottom: 2rem !important;
}

/* Info boxes */
.info-box {
    background: #f8f9fa;
    border-left: 3px solid #667eea;
    padding: 1rem;
    border-radius: 4px;
    margin: 1rem 0;
}

.info-box code {
    background: #e9ecef;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
}

/* Row alignment fix */
.row {
    display: flex !important;
    align-items: stretch !important;
    gap: 2.5rem !important;
}

/* Column balance */
.column {
    flex: 1 !important;
    min-width: 0 !important;
}

/* Tab content spacing */
.tab-content {
    padding: 1.5rem 0 !important;
}
"""

# =========================================================
# Footer & Query
# =========================================================
def create_demo():
    with gr.Blocks(title="Multi-Doc RAG", theme=gr.themes.Soft()) as demo:
        

        gr.HTML(f"<style>{custom_css}</style>")
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 style="
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 0.5rem;
                letter-spacing: -0.03em;
            ">Multi-Document RAG Assistant</h1>
            <p style="
                color: #6b7280;
                font-size: 1rem;
                font-weight: 400;
                margin: 0;
            ">AI-powered question answering across multiple PDF documents</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("Setup"):
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Upload Documents")
                        file_upload = gr.File(
                            label="Select PDF files",
                            file_count="multiple",
                            file_types=[".pdf"],
                            type="filepath",
                            elem_classes=["file-upload"],
                            height=250
                        )
                        
                        upload_status = gr.Textbox(
                            label="Status",
                            value="No files uploaded",
                            interactive=False,
                            lines=2
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 2. Build Index")
                        gr.Markdown("Process PDFs and create searchable index.")
                        
                        build_btn = gr.Button(
                            "Build Index",
                            variant="primary",
                            size="lg"
                        )
                        
                        build_output = gr.Textbox(
                            label="Build Status",
                            interactive=False,
                            lines=8
                        )
                
                # Configuration info
                with gr.Accordion("Configuration Details", open=False):
                    gr.Markdown("""
                    **Index Settings**
                    - Chunk Size: 800 characters
                    - Overlap: 150 characters
                    - Embeddings: all-MiniLM-L6-v2
                    - Retrieval: Top-6 chunks
                    
                    **LLM Settings**
                    - Model: google/flan-t5-large
                    - Temperature: 0.3
                    - Max Tokens: 512
                    """)
                
                # Event handlers
                def update_upload_status(files):
                    if not files:
                        return "No files uploaded"
                    count = len([f for f in files if f is not None])
                    return f"✓ {count} file(s) ready"
                
                file_upload.change(
                    fn=update_upload_status,
                    inputs=[file_upload],
                    outputs=[upload_status]
                )
                
                def build_index_ui(files):
                    saved_paths = save_uploads(files)
                    return build_index_from_pdfs(saved_paths)
                
                build_btn.click(
                    fn=build_index_ui,
                    inputs=[file_upload],
                    outputs=[build_output]
                )
            
            with gr.Tab("Query"):
                
                gr.Markdown("### Ask a Question")
                
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="What are the main findings across all documents?",
                    lines=2
                )
                
                ask_btn = gr.Button(
                    "Get Answer",
                    variant="primary",
                    size="lg"
                )
                
                # Example questions as buttons
                gr.Markdown("**Quick Examples:**")
                with gr.Row():
                    example_btn1 = gr.Button(
                        "Main findings across documents",
                        size="sm",
                        variant="secondary"
                    )
                    example_btn2 = gr.Button(
                        "Compare methodologies",
                        size="sm",
                        variant="secondary"
                    )
                    example_btn3 = gr.Button(
                        "Key assumptions and limitations",
                        size="sm",
                        variant="secondary"
                    )
                
                # Results
                answer_output = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    lines=10
                )
                
                with gr.Row():
                    query_type_output = gr.Textbox(
                        label="Query Type",
                        interactive=False,
                        lines=1
                    )
                    
                    with gr.Column():
                        gr.Markdown("""
                        **Query Types:**
                        - **synthesis**: Combining info
                        - **comparison**: Contrasting sources
                        - **extraction**: Finding specifics
                        """)
                
                evidence_output = gr.Textbox(
                    label="Supporting Evidence",
                    interactive=False,
                    lines=12,
                    elem_classes=["monospace"]
                )
                
                # Event handlers
                ask_btn.click(
                    fn=answer_question,
                    inputs=[question_input],
                    outputs=[answer_output, query_type_output, evidence_output]
                )
                
                # Example button handlers
                example_btn1.click(
                    lambda: "What are the main sources of risk mentioned across all documents?",
                    outputs=[question_input]
                )
                
                example_btn2.click(
                    lambda: "How do the documents differ in their treatment of the same concept or methodology?",
                    outputs=[question_input]
                )
                
                example_btn3.click(
                    lambda: "What assumptions and limitations are highlighted in the documents?",
                    outputs=[question_input]
                )
        

        gr.HTML("""
        <div style="
            margin-top: 2rem;
            padding: 1.5rem 0 1rem 0;
            border-top: 1px solid #e5e7eb;
            text-align: center;
        ">
            <div style="margin-bottom: 1rem;">
                <p style="
                    font-size: 0.9rem;
                    color: #9ca3af;
                    margin-bottom: 0.5rem;
                ">
                    <strong style="color: #667eea;">COMS 4995:</strong> Applied Machine Learning — Final Project
                </p>
            </div>
            
            <div style="
                display: flex;
                justify-content: center;
                gap: 2rem;
                flex-wrap: wrap;
                margin-top: 1rem;
            ">
                <span style="color: #6b7280; font-size: 0.9rem;">
                    <strong>Cheng Wu</strong> <span style="color: #9ca3af;">(cw3729)</span>
                </span>
                <span style="color: #6b7280; font-size: 0.9rem;">
                    <strong>Pranav Jain</strong> <span style="color: #9ca3af;">(pj2459)</span>
                </span>
                <span style="color: #6b7280; font-size: 0.9rem;">
                    <strong>Jaewon Cho</strong> <span style="color: #9ca3af;">(jc6618)</span>
                </span>
                <span style="color: #6b7280; font-size: 0.9rem;">
                    <strong>Winston Li</strong> <span style="color: #9ca3af;">(wl3062)</span>
                </span>
            </div>
            
            <p style="
                margin-top: 1.5rem;
                font-size: 0.85rem;
                color: #9ca3af;
            ">
                <strong>Requirements:</strong> Set <code style="
                    background: #f3f4f6;
                    padding: 0.2rem 0.4rem;
                    border-radius: 3px;
                    font-family: 'JetBrains Mono', monospace;
                ">HF_TOKEN</code> in <code style="
                    background: #f3f4f6;
                    padding: 0.2rem 0.4rem;
                    border-radius: 3px;
                    font-family: 'JetBrains Mono', monospace;
                ">.env</code> file 
                (<a href="https://huggingface.co/settings/tokens" target="_blank" style="color: #667eea; text-decoration: none;">Get token</a>)
            </p>
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
