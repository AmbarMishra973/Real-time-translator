"""
RAG Evaluation Engine.
Evaluates deterministic retrieval quality (Hit@K, Recall@K, MRR, No-Answer Accuracy, Latency)
and categorizes retrieval failure modes.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.rag_engine import rag_engine, RAGEngine


class RAGEvaluator:
    """Evaluates RAG retrieval against deterministic gold-standard test suites."""

    def __init__(self, engine: Optional[RAGEngine] = None):
        self.engine = engine or rag_engine

    def load_dataset(self, dataset_path: Optional[str] = None) -> List[Dict[str, Any]]:
        if dataset_path is None:
            dataset_path = os.path.join(
                os.path.dirname(__file__), "datasets", "rag_cases.json"
            )
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_relevant(self, chunk: Dict[str, Any], case: Dict[str, Any]) -> bool:
        """Determines if a retrieved chunk matches gold expectations."""
        expected_sources = [s.lower() for s in case.get("expected_sources", [])]
        expected_terms = [t.lower() for t in case.get("expected_terms", [])]

        chunk_source = (chunk.get("source_file") or "").lower()
        chunk_term = (chunk.get("term") or "").lower()

        source_match = (chunk_source in expected_sources) if expected_sources else False
        term_match = (chunk_term in expected_terms) if expected_terms else False

        # If both are specified, require term match and source match
        if expected_sources and expected_terms:
            return source_match and term_match
        if expected_sources:
            return source_match
        if expected_terms:
            return term_match
        return False

    def _classify_failure(self, case: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Categorizes why retrieval failed or underperformed."""
        if not case.get("answerable", True):
            # No-answer query that mistakenly retrieved something
            return "NO_RELEVANT_DOCUMENT"

        if not retrieved_chunks:
            # Nothing returned above threshold
            query_words = set(re.findall(r"\w+", case["query"].lower()))
            expected_words = set()
            for t in case.get("expected_terms", []):
                expected_words.update(re.findall(r"\w+", t.lower()))
            if not query_words.intersection(expected_words):
                return "PARAPHRASE"
            return "LOW_SIGNAL"

        # Chunks were returned, but wrong ones
        top_chunk = retrieved_chunks[0]
        category = case.get("category", "")
        if category == "paraphrase":
            return "PARAPHRASE"
        if category == "multilingual":
            return "TOKENIZATION"
        
        expected_terms = [t.lower() for t in case.get("expected_terms", [])]
        retrieved_terms = [c.get("term", "").lower() for c in retrieved_chunks]
        if any(exp in " ".join(retrieved_terms) for exp in expected_terms):
            return "CHUNKING"

        return "LEXICAL_MISMATCH"

    def evaluate(
        self,
        dataset: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        threshold: float = 0.30
    ) -> Dict[str, Any]:
        """
        Runs complete retrieval evaluation and latency benchmarking.
        """
        if dataset is None:
            dataset = self.load_dataset()

        total_cases = len(dataset)
        answerable_cases = [c for c in dataset if c.get("answerable", True)]
        no_answer_cases = [c for c in dataset if not c.get("answerable", True)]

        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        reciprocal_ranks = []
        recalls_at_5 = []
        correct_no_answers = 0
        failures = []
        latencies_ms = []

        for case in dataset:
            query = case["query"]
            domain = case.get("domain", "all")
            is_answerable = case.get("answerable", True)

            t0 = time.perf_counter()
            retrieval_res = self.engine.retrieve(query=query, domain=domain, top_k=top_k, threshold=threshold)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)

            chunks = retrieval_res.get("chunks", [])

            if not is_answerable:
                # Expect zero chunks retrieved above threshold
                if len(chunks) == 0:
                    correct_no_answers += 1
                else:
                    failures.append({
                        "id": case["id"],
                        "query": query,
                        "category": case.get("category", "no_answer"),
                        "expected_sources": [],
                        "expected_terms": [],
                        "retrieved_sources": retrieval_res.get("sources_used", []),
                        "retrieved_terms": [c.get("term") for c in chunks],
                        "top_score": chunks[0].get("similarity", 0.0) if chunks else 0.0,
                        "failure_reason": self._classify_failure(case, chunks)
                    })
                continue

            # Answerable cases evaluation
            ranks_of_relevant = []
            expected_sources = set(s.lower() for s in case.get("expected_sources", []))
            retrieved_sources_relevant = set()

            for rank, chunk in enumerate(chunks, start=1):
                if self._is_relevant(chunk, case):
                    ranks_of_relevant.append(rank)
                    retrieved_sources_relevant.add((chunk.get("source_file") or "").lower())

            # Hit@K calculations
            has_hit_1 = any(r <= 1 for r in ranks_of_relevant)
            has_hit_3 = any(r <= 3 for r in ranks_of_relevant)
            has_hit_5 = any(r <= 5 for r in ranks_of_relevant)

            if has_hit_1:
                hit_1 += 1
            if has_hit_3:
                hit_3 += 1
            if has_hit_5:
                hit_5 += 1

            # Reciprocal rank (for first hit)
            if ranks_of_relevant:
                reciprocal_ranks.append(1.0 / ranks_of_relevant[0])
            else:
                reciprocal_ranks.append(0.0)

            # Recall@5 for sources
            if expected_sources:
                recall = len(retrieved_sources_relevant.intersection(expected_sources)) / len(expected_sources)
                recalls_at_5.append(recall)

            if not has_hit_3:
                failures.append({
                    "id": case["id"],
                    "query": query,
                    "category": case.get("category", "general"),
                    "expected_sources": case.get("expected_sources", []),
                    "expected_terms": case.get("expected_terms", []),
                    "retrieved_sources": retrieval_res.get("sources_used", []),
                    "retrieved_terms": [c.get("term") for c in chunks],
                    "top_score": chunks[0].get("similarity", 0.0) if chunks else 0.0,
                    "failure_reason": self._classify_failure(case, chunks)
                })

        num_answerable = len(answerable_cases)
        num_no_answer = len(no_answer_cases)

        hit_1_rate = round(hit_1 / num_answerable, 4) if num_answerable else 0.0
        hit_3_rate = round(hit_3 / num_answerable, 4) if num_answerable else 0.0
        hit_5_rate = round(hit_5 / num_answerable, 4) if num_answerable else 0.0
        mrr = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0
        mean_recall = round(sum(recalls_at_5) / len(recalls_at_5), 4) if recalls_at_5 else 0.0
        no_answer_acc = round(correct_no_answers / num_no_answer, 4) if num_no_answer else 1.0

        sorted_latencies = sorted(latencies_ms)
        p50 = round(sorted_latencies[int(len(sorted_latencies) * 0.50)], 2) if sorted_latencies else 0.0
        p95 = round(sorted_latencies[int(len(sorted_latencies) * 0.95)], 2) if sorted_latencies else 0.0
        avg_latency = round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0

        failure_counts: Dict[str, int] = {}
        for f in failures:
            reason = f["failure_reason"]
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

        summary = {
            "total_cases": total_cases,
            "answerable_cases": num_answerable,
            "no_answer_cases": num_no_answer,
            "hit_at_1": hit_1_rate,
            "hit_at_1_count": f"{hit_1}/{num_answerable}",
            "hit_at_3": hit_3_rate,
            "hit_at_3_count": f"{hit_3}/{num_answerable}",
            "hit_at_5": hit_5_rate,
            "hit_at_5_count": f"{hit_5}/{num_answerable}",
            "mrr": mrr,
            "recall_at_5": mean_recall,
            "no_answer_accuracy": no_answer_acc,
            "no_answer_count": f"{correct_no_answers}/{num_no_answer}",
            "latency": {
                "avg_ms": avg_latency,
                "p50_ms": p50,
                "p95_ms": p95
            },
            "failure_summary": failure_counts,
            "failures": failures
        }
        return summary

    def save_report(self, summary: Dict[str, Any], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def run_cli_evaluation(save_path: Optional[str] = None):
    evaluator = RAGEvaluator()
    summary = evaluator.evaluate()

    print("=" * 60)
    print("           RAG RETRIEVAL EVALUATION REPORT            ")
    print("=" * 60)
    print(f"Total Test Cases:        {summary['total_cases']}")
    print(f"Answerable Cases:        {summary['answerable_cases']}")
    print(f"No-Answer Cases:         {summary['no_answer_cases']}")
    print("-" * 60)
    print(f"Hit@1:                   {summary['hit_at_1'] * 100:.1f}% ({summary['hit_at_1_count']})")
    print(f"Hit@3:                   {summary['hit_at_3'] * 100:.1f}% ({summary['hit_at_3_count']})")
    print(f"Hit@5:                   {summary['hit_at_5'] * 100:.1f}% ({summary['hit_at_5_count']})")
    print(f"MRR (Mean Recip. Rank):  {summary['mrr']}")
    print(f"Recall@5:                {summary['recall_at_5'] * 100:.1f}%")
    print(f"No-Answer Accuracy:      {summary['no_answer_accuracy'] * 100:.1f}% ({summary['no_answer_count']})")
    print("-" * 60)
    print("Latency Benchmark:")
    print(f"  Average Latency:       {summary['latency']['avg_ms']} ms")
    print(f"  p50 Latency:           {summary['latency']['p50_ms']} ms")
    print(f"  p95 Latency:           {summary['latency']['p95_ms']} ms")
    print("-" * 60)
    print("Observed Failures by Category:")
    for cat, cnt in summary["failure_summary"].items():
        print(f"  - {cat}: {cnt}")
    print("=" * 60)

    if save_path:
        evaluator.save_report(summary, save_path)
        print(f"Saved evaluation snapshot to: {save_path}")

    return summary


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "reports", "rag_baseline.json"
    )
    run_cli_evaluation(out_file)
