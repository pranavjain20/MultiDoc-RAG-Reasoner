"""Prompt builders for multi-document RAG queries."""

from typing import Callable, Dict, List

SYSTEM_PROMPT = """You are a research document analyst specializing in multi-document analysis.

Your responsibilities:
- Analyze information across multiple documents comprehensively
- Always cite sources using the format [DocumentName.pdf] when referencing specific information
- Be comprehensive in your analysis but concise in your presentation
- Acknowledge explicitly when information is missing or unavailable
- Handle contradictions between documents by explicitly stating the conflicting viewpoints
- Maintain objectivity and present findings clearly

Citation rules:
- Every claim or finding must include a source citation [DocumentName.pdf]
- If information appears in multiple documents, cite all relevant sources
- Use [DocumentName.pdf] format exactly as provided in the document names"""


def build_synthesis_prompt(question: str, chunks: List[Dict[str, str]]) -> str:
    if not chunks:
        return f"Question: {question}\n\nNo documents provided for analysis."

    doc_chunks: Dict[str, List[str]] = {}
    for chunk in chunks:
        doc_name = chunk.get("doc_name", "Unknown.pdf")
        doc_chunks.setdefault(doc_name, []).append(chunk.get("text", ""))

    doc_sections = []
    for doc_name, texts in doc_chunks.items():
        joined = "\n\n".join(texts)
        doc_sections.append(f"--- {doc_name} ---\n" + joined)

    documents_text = "\n\n".join(doc_sections)
    is_single = len(doc_chunks) == 1

    if is_single:
        doc_name = list(doc_chunks.keys())[0]
        return f"""Analyze the following question across the provided document:

Question: {question}

Document:
{documents_text}

Instructions:
1. Identify the key themes, ideas, or elements that address the question
2. Organize findings in a clear, structured manner
3. For each finding or theme, cite the source as [{doc_name}]
4. If the question asks about risks, sources, or specific elements, list them systematically
5. If information is missing or unclear, explicitly state what is not available
6. Provide a comprehensive but concise response that integrates findings from across the document

Please provide your analysis:"""

    return f"""Analyze the following question across multiple documents:

Question: {question}

Documents:
{documents_text}

Instructions:
1. Identify themes, ideas, or elements that address the question across all documents
2. Organize findings by theme or category, not by document
3. For each finding or theme, cite all relevant sources using [DocumentName.pdf] format
4. Integrate findings across documents - look for common themes, unique contributions, and patterns
5. If the question asks about risks, sources, or specific elements, list them systematically with citations
6. If information is missing or unclear, explicitly state what is not available in any document
7. Provide a comprehensive but concise response that synthesizes information from all documents

Please provide your analysis:"""


def build_comparison_prompt(question: str, chunks_by_doc: Dict[str, List[str]]) -> str:
    if not chunks_by_doc:
        return f"Question: {question}\n\nNo documents provided for analysis."

    if len(chunks_by_doc) == 1:
        doc_name = list(chunks_by_doc.keys())[0]
        chunks_text = "\n\n".join(chunks_by_doc[doc_name])
        return f"""Analyze the following question for a single document:

Question: {question}

Document: {doc_name}
{chunks_text}

Note: This question asks for a comparison, but only one document is provided.
Please analyze how the document addresses the concepts in the question, and note
what aspects might benefit from comparison with additional documents.

Please provide your analysis with citations as [{doc_name}]:"""

    doc_sections = []
    for doc_name, chunks in chunks_by_doc.items():
        joined = "\n\n".join(chunks)
        doc_sections.append(f"=== {doc_name} ===\n" + joined + "\n")

    return f"""Compare information across the following documents to answer the question:

Question: {question}

Documents:
{''.join(doc_sections)}

Instructions:
1. Analyze each document's treatment of the concept, methodology, or conclusion in question
2. Identify COMMON GROUND: shared perspectives, definitions, or approaches (cite sources)
3. Identify KEY DIFFERENCES: varying approaches, methodologies, perspectives
4. Highlight UNIQUE CONTRIBUTIONS: what each document adds that others do not
5. Address CONTRADICTIONS OR TENSIONS explicitly
6. Structure your response: overview, common ground, key differences, unique contributions, contradictions

Please provide your comparative analysis:"""


def build_extraction_prompt(question: str, chunks: List[Dict[str, str]]) -> str:
    if not chunks:
        return f"Question: {question}\n\nNo documents provided for analysis."

    doc_chunks: Dict[str, List[str]] = {}
    for chunk in chunks:
        doc_name = chunk.get("doc_name", "Unknown.pdf")
        doc_chunks.setdefault(doc_name, []).append(chunk.get("text", ""))

    doc_sections = []
    for doc_name, texts in doc_chunks.items:
        joined = "\n\n".join(texts)
        doc_sections.append(f"--- {doc_name} ---\n" + joined)

    return f"""Extract specific information from the following documents to answer the question:

Question: {question}

Documents:
{"\n\n".join(doc_sections)}

Instructions:
1. List all relevant elements (assumptions, limitations, etc.)
2. For each element: quote or paraphrase & cite the source
3. Indicate whether each element is explicit or implicit
4. Group by category if applicable
5. Extract ALL relevant instances across documents
6. If something is missing, explicitly note what is absent

Please provide your extraction:"""


def select_prompt_builder(query_type: str) -> Callable:
    mapping: Dict[str, Callable] = {
        "synthesis": build_synthesis_prompt,
        "comparison": build_comparison_prompt,
        "extraction": build_extraction_prompt,
    }
    query_type_lower = query_type.lower().strip()
    if query_type_lower not in mapping:
        raise ValueError(
            f"Unknown query_type '{query_type}'. Must be one of: {', '.join(mapping.keys())}"
        )
    return mapping[query_type_lower]

