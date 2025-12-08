"""
Evaluation script for Multi-Document RAG system.

This script runs four evaluation components:
E1 — Query type classification accuracy
E2 — Prompt structure sanity checks
E3 — Baseline vs Reasoning answer quality
E4 — Lost-in-the-middle ablation

NOTE:
You MUST provide a file:
evaluation_outputs/evaluation_chunks.json

Format:
{
  "q1": [
    {"doc_name": "doc1.pdf", "text": "..."},
    {"doc_name": "doc2.pdf", "text": "..."}
  ],
  "q2": [...],
  ...
}
"""

import os
import json
from typing import Dict, List, Any, Tuple

from llm.llm_client import LLMClient
from llm.reasoning import MultiDocReasoner
from llm.prompts import (
    build_synthesis_prompt,
    build_comparison_prompt,
    build_extraction_prompt,
)

OUTPUT_DIR = "evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Evaluation questions

EVAL_QUESTIONS = [
    ("q1", "Summarize the main ideas discussed across these documents.", "synthesis"),
    ("q2", "What are the main sources of risk mentioned across the documents?", "synthesis"),
    ("q3", "Compare how different documents describe the same concept or methodology.", "comparison"),
    ("q4", "What are the key assumptions and limitations highlighted in these documents?", "extraction"),
    ("q5", "How do the documents differ in their conclusions or policy implications?", "comparison"),
]


# Load chunks

def load_chunks_for_question(qid: str) -> List[Dict[str, Any]]:
    """
    Loads the top-k retrieved chunks for a question.
    The user must create evaluation_chunks.json beforehand using a retriever.
    """
    path = os.path.join(OUTPUT_DIR, "evaluation_chunks.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. Please generate it in AML_Project.ipynb."
        )
    with open(path, "r") as f:
        data = json.load(f)
    return data.get(qid, [])


# E1 — Query type classification

def run_e1(reasoner: MultiDocReasoner):
    results = []
    correct = 0

    for qid, question, gold_type in EVAL_QUESTIONS:
        pred = reasoner.classify_query(question)
        is_correct = pred == gold_type
        correct += int(is_correct)

        results.append({
            "qid": qid,
            "question": question,
            "gold_type": gold_type,
            "predicted": pred,
            "correct": is_correct
        })

    accuracy = correct / len(EVAL_QUESTIONS)

    out_path = os.path.join(OUTPUT_DIR, "e1_query_type.json")
    with open(out_path, "w") as f:
        json.dump({"accuracy": accuracy, "records": results}, f, indent=2)

    print("\n=== E1: Query Type Classification ===")
    print(f"Accuracy: {accuracy:.3f}")
    return results


# E2 — Prompt structure sanity checks

def run_e2():
    chunks = [
        {"doc_name": "doc1.pdf", "text": "Content A"},
        {"doc_name": "doc2.pdf", "text": "Content B"}
    ]

    synth = build_synthesis_prompt("Test synthesis", chunks)
    comp = build_comparison_prompt(
        "Compare X",
        {"doc1.pdf": ["Content A"], "doc2.pdf": ["Content B"]},
    )
    extr = build_extraction_prompt("List assumptions", chunks)

    results = {
        "synthesis_has_doc1": "doc1.pdf" in synth,
        "synthesis_has_doc2": "doc2.pdf" in synth,
        "comparison_grouping": "doc1.pdf" in comp and "doc2.pdf" in comp,
        "extraction_mentions_assumptions": "assumptions" in extr.lower(),
    }

    path = os.path.join(OUTPUT_DIR, "e2_prompt_sanity.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== E2: Prompt Sanity Checks ===")
    print(results)
    return results


# Baseline prompt

def build_baseline_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(c["text"] for c in chunks)
    return (
        "You are an assistant answering questions using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


# E3 — Baseline vs Reasoning

def run_e3(client: LLMClient, reasoner: MultiDocReasoner):
    results = []

    print("\n=== E3: Baseline vs Reasoning ===")

    for qid, question, gold_type in EVAL_QUESTIONS:
        chunks = load_chunks_for_question(qid)

        # Baseline
        baseline_prompt = build_baseline_prompt(question, chunks)
        baseline_answer = client.generate(prompt=baseline_prompt)

        # Reasoning
        reasoning_prompt, qtype = reasoner.build_prompt(question, chunks)
        reasoning_answer = client.generate(prompt=reasoning_prompt)

        results.append({
            "qid": qid,
            "question": question,
            "gold_type": gold_type,
            "query_type_used": qtype,
            "baseline_prompt": baseline_prompt,
            "baseline_answer": baseline_answer,
            "reasoning_prompt": reasoning_prompt,
            "reasoning_answer": reasoning_answer,
        })

    out = os.path.join(OUTPUT_DIR, "e3_baseline_vs_reasoning.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {out}")
    return results


# E4 — Lost-in-the-middle ablation

def run_e4(client: LLMClient, reasoner: MultiDocReasoner):
    target_qids = ["q1", "q3", "q5"]
    results = []

    print("\n=== E4: Lost-in-the-middle Ablation ===")

    for qid, question, gold_type in EVAL_QUESTIONS:
        if qid not in target_qids:
            continue

        chunks = load_chunks_for_question(qid)

        # No mitigation
        p1, t1 = reasoner.build_prompt(question, chunks, apply_lost_in_middle=False)
        a1 = client.generate(prompt=p1)

        # With mitigation
        p2, t2 = reasoner.build_prompt(question, chunks, apply_lost_in_middle=True)
        a2 = client.generate(prompt=p2)

        results.append({
            "qid": qid,
            "question": question,
            "baseline_prompt": p1,
            "mitigated_prompt": p2,
            "answer_no_mitigation": a1,
            "answer_with_mitigation": a2,
        })

    out = os.path.join(OUTPUT_DIR, "e4_lost_in_middle.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {out}")
    return results


# Main

def main():
    reasoner = MultiDocReasoner()
    client = LLMClient(model_id="google/flan-t5-large")  # matches your repo exactly

    run_e1(reasoner)
    run_e2()
    run_e3(client, reasoner)
    run_e4(client, reasoner)


if __name__ == "__main__":
    main()
