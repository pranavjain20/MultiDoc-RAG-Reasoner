"""Multi-document reasoning and query classification."""

from typing import Dict, List, Tuple

from llm.prompts import (
    build_comparison_prompt,
    build_extraction_prompt,
    build_synthesis_prompt,
)


class MultiDocReasoner:
    """Handles multi-document reasoning for RAG systems."""

    def __init__(self) -> None:
        self.query_patterns: Dict[str, List[str]] = {
            'extraction': [
                'assumptions and limitations', 'assumptions', 'limitations',
                'key assumptions', 'key limitations', 'highlighted', 'explicitly stated', 'explicitly',
            ],
            'comparison': [
                'compare', 'contrast', 'differ', 'difference', 'versus', 'vs',
                'same concept', 'same methodology', 'different documents', 'how do the documents', 'how do',
            ],
            'synthesis': [
                'summarize', 'main ideas', 'key points', 'across', 'overall',
                'sources of', 'mentioned', 'discussed',
            ],
        }

    def classify_query(self, question: str) -> str:
        """Classify question as 'synthesis', 'comparison', or 'extraction'."""
        question_lower = question.lower()
        for query_type in ['extraction', 'comparison', 'synthesis']:
            for keyword in sorted(self.query_patterns[query_type], key=len, reverse=True):
                if keyword in question_lower:
                    return query_type
        return 'synthesis'

    def organize_chunks_by_doc(self, chunks: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Group chunks by document name."""
        if not chunks:
            return {}
        organized: Dict[str, List[str]] = {}
        for chunk in chunks:
            doc_name = chunk.get('doc_name', 'Unknown.pdf')
            if doc_name not in organized:
                organized[doc_name] = []
            organized[doc_name].append(chunk.get('text', ''))
        return organized

    def get_unique_documents(self, chunks: List[Dict[str, str]]) -> List[str]:
        """Return sorted list of unique document names."""
        if not chunks:
            return []
        return sorted(set(chunk.get('doc_name', 'Unknown.pdf') for chunk in chunks))

    def mitigate_lost_in_middle(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Reorder chunks to put highest relevance items at start and end."""
        if len(chunks) <= 1:
            return chunks
        even_chunks = [chunks[i] for i in range(0, len(chunks), 2)]
        odd_chunks = [chunks[i] for i in range(1, len(chunks), 2)]
        return list(reversed(even_chunks)) + odd_chunks

    def build_prompt(
        self,
        question: str,
        chunks: List[Dict[str, str]],
        apply_lost_in_middle: bool = True
    ) -> Tuple[str, str]:
        """Build prompt for question and chunks, return (prompt, query_type)."""
        if not chunks:
            return (f"Question: {question}\n\nNo documents provided for analysis.", 'synthesis')

        query_type = self.classify_query(question)
        processed_chunks = self.mitigate_lost_in_middle(chunks) if apply_lost_in_middle else chunks

        if query_type == 'comparison':
            chunks_by_doc = self.organize_chunks_by_doc(processed_chunks)
            prompt = build_comparison_prompt(question, chunks_by_doc)
        elif query_type == 'extraction':
            prompt = build_extraction_prompt(question, processed_chunks)
        else:
            prompt = build_synthesis_prompt(question, processed_chunks)

        return (prompt, query_type)
