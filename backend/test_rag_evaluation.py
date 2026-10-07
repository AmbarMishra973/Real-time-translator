"""
Automated Test Suite for RAG Evaluation and Retrieval Quality.
Tests:
1. Deterministic evaluation dataset loading and schema validation.
2. Hit@1, Hit@3, Hit@5 computation and bounds.
3. MRR (Mean Reciprocal Rank) calculation.
4. No-answer query handling (accuracy on unsupported queries).
5. Substring collision resistance (ensuring 'api' does not trigger on 'capital', 'rag' not on 'storage').
6. Empty query handling.
7. Empty knowledge base handling.
8. Malformed evaluation records tolerance.
9. Domain filtering verification.
"""

import sys
import unittest
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.rag_engine import RAGEngine
from backend.evaluation.rag_eval import RAGEvaluator


class TestRAGEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RAGEngine()
        cls.evaluator = RAGEvaluator(engine=cls.engine)
        cls.dataset = cls.evaluator.load_dataset()

    def test_01_dataset_schema_and_size(self):
        """Verifies dataset has at least 25 cases and all required schema keys."""
        self.assertGreaterEqual(len(self.dataset), 25)
        for case in self.dataset:
            self.assertIn("id", case)
            self.assertIn("query", case)
            self.assertIn("category", case)
            self.assertIn("answerable", case)
            self.assertIsInstance(case.get("expected_sources", []), list)

    def test_02_hit_and_mrr_metrics(self):
        """Verifies Hit@K and MRR metrics execute cleanly and achieve high accuracy."""
        summary = self.evaluator.evaluate(self.dataset)
        self.assertGreaterEqual(summary["hit_at_1"], 0.90)
        self.assertEqual(summary["hit_at_3"], 1.0)
        self.assertEqual(summary["hit_at_5"], 1.0)
        self.assertGreaterEqual(summary["mrr"], 0.95)
        self.assertEqual(summary["recall_at_5"], 1.0)

    def test_03_no_answer_accuracy(self):
        """Verifies that unsupported queries do not produce false positive retrievals."""
        summary = self.evaluator.evaluate(self.dataset)
        self.assertEqual(summary["no_answer_accuracy"], 1.0)
        self.assertEqual(summary["no_answer_cases"], 5)

    def test_04_word_boundary_collision_protection(self):
        """Regression test ensuring short terms like 'API' and 'RAG' do not match inside other words."""
        # 'capital' must not match 'API'
        res_capital = self.engine.retrieve("What is the capital city of France?")
        terms_capital = [c["term"].lower() for c in res_capital.get("chunks", [])]
        self.assertNotIn("api", terms_capital)

        # 'storage' must not match 'RAG'
        res_storage = self.engine.retrieve("High speed temporary storage layer used to serve repeated queries faster.")
        top_term = res_storage["chunks"][0]["term"]
        self.assertEqual(top_term, "Caching")

    def test_05_empty_query(self):
        """Verifies empty or whitespace query returns empty retrieval result safely."""
        res_empty = self.engine.retrieve("")
        self.assertEqual(res_empty["chunks"], [])
        self.assertEqual(res_empty["total_matches"], 0)

        res_spaces = self.engine.retrieve("   ")
        self.assertEqual(res_spaces["chunks"], [])
        self.assertEqual(res_spaces["total_matches"], 0)

    def test_06_domain_filtering(self):
        """Verifies filtering by domain only returns chunks from that domain."""
        med_res = self.engine.retrieve("hypertension diagnosis", domain="medical")
        for chunk in med_res["chunks"]:
            self.assertEqual(chunk["domain"], "medical")

        # Querying medical term with business domain should return 0 results
        mismatch_res = self.engine.retrieve("hypertension blood pressure", domain="business")
        self.assertEqual(mismatch_res["chunks"], [])

    def test_07_empty_knowledge_base_engine(self):
        """Verifies engine handles an empty knowledge directory gracefully without crashing."""
        empty_engine = RAGEngine(knowledge_dir=str(REPO_ROOT / "non_existent_folder_xyz"))
        res = empty_engine.retrieve("test query")
        self.assertEqual(res["chunks"], [])
        self.assertEqual(res["total_matches"], 0)
        self.assertEqual(res["sources_used"], [])

    def test_08_malformed_case_tolerance(self):
        """Verifies evaluator tolerates missing optional fields or malformed records."""
        malformed = [
            {"id": "bad_1", "query": "Vector Database?", "answerable": True},
            {"id": "bad_2", "query": "Unknown query", "answerable": False}
        ]
        summary = self.evaluator.evaluate(malformed)
        self.assertEqual(summary["total_cases"], 2)


if __name__ == "__main__":
    unittest.main()
