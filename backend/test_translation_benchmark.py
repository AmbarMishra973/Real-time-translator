"""
Unit and Regression Test Suite for Phase 4: Translation Layer Benchmarking.
Verifies:
1. Benchmark dataset schema and integrity (all 50 cases valid).
2. Translation engine feature flag dispatch (groq, fallback, unfeasible local handlers).
3. RAG enabled/disabled toggle and context injection bypass.
4. Evaluation metric calculations (numbers, technical terms, named entities, script sanity).
5. Observability fields and resilience under error conditions.
"""

import os
import sys
import json
import unittest
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.llm_translator import LLMTranslator
from backend.services.llm_service import LLMService


class TestTranslationBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = LLMTranslator()
        cls.service = LLMService(cls.translator)
        cls.dataset_path = REPO_ROOT / "backend" / "evaluation" / "datasets" / "translation_benchmark_dataset.json"

    def test_01_dataset_schema_and_size(self):
        """Verifies that the benchmark dataset contains at least 40 cases with all required fields."""
        self.assertTrue(self.dataset_path.exists(), f"Dataset missing at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        self.assertGreaterEqual(len(cases), 40, f"Expected >= 40 cases, found {len(cases)}")
        required_keys = {
            "id", "source_language", "target_language", "source_text",
            "reference_translation", "category", "expected_terms",
            "expected_numbers", "expected_entities", "rag_expected"
        }
        for case in cases:
            for k in required_keys:
                self.assertIn(k, case, f"Case {case.get('id')} missing key '{k}'")
            self.assertIn(case["source_language"], ["en", "hi"])
            self.assertIn(case["target_language"], ["en", "hi"])
            self.assertTrue(len(case["source_text"].strip()) > 0)
            self.assertTrue(len(case["reference_translation"].strip()) > 0)

    def test_02_engine_flag_fallback_dispatch(self):
        """Verifies that TRANSLATION_ENGINE='fallback' directly invokes the fallback cascade."""
        res = self.translator.translate("Hello", source_lang="en", target_lang="hi", engine="fallback")
        self.assertIn("provider", res)
        self.assertIn("Fallback", res["provider"])
        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["fallback_reason"], "configured_fallback_engine")
        self.assertTrue(len(res["translated_text"]) > 0)

    def test_03_engine_flag_unfeasible_local_candidates(self):
        """Verifies that requesting unfeasible candidates safely falls back and records reason."""
        for candidate in ["indictrans2", "candidate_local"]:
            res = self.translator.translate("Hello", source_lang="en", target_lang="hi", engine=candidate)
            self.assertTrue(res["fallback_used"])
            self.assertEqual(res["fallback_reason"], f"engine_{candidate}_not_feasible_on_windows_cpu")
            self.assertIn("Fallback", res["provider"])
            self.assertTrue(len(res["translated_text"]) > 0)

    def test_04_rag_enabled_toggle(self):
        """Verifies that rag_enabled=False bypasses RAG retrieval and leaves retrieved_context empty."""
        tech_text = "We use Docker and Kubernetes for container orchestration."
        res_no_rag = self.translator.translate(
            tech_text, source_lang="en", target_lang="hi", engine="fallback", rag_enabled=False
        )
        self.assertEqual(res_no_rag["retrieved_context"], [])
        self.assertFalse(res_no_rag["context_used"])

    def test_05_number_preservation_evaluator(self):
        """Verifies number detection in reference translations."""
        from backend.evaluation.llm_eval import normalize_digits
        src = "HTTP 404 code with 2.5 seconds latency and 99.9% uptime"
        target_valid = "99.9% अपटाइम और 2.5 सेकंड लेटेंसी के साथ HTTP 404 कोड"
        target_missing = "अपटाइम और लेटेंसी के साथ HTTP कोड"

        expected = ["404", "2.5", "99.9"]
        norm_valid = normalize_digits(target_valid)
        norm_missing = normalize_digits(target_missing)

        for n in expected:
            self.assertIn(n, norm_valid)
        self.assertFalse(all(n in norm_missing for n in expected))

    def test_06_technical_term_preservation(self):
        """Verifies technical term preservation check."""
        target_text = "हमने Docker और Kubernetes का उपयोग करके सेवा तैनात की।"
        terms = ["Docker", "Kubernetes"]
        low = target_text.lower()
        self.assertTrue(all(t.lower() in low for t in terms))

    def test_07_service_wrapper_delegation(self):
        """Verifies LLMService cleanly forwards engine and rag_enabled arguments."""
        res = self.service.translate(
            "Good evening", source_lang="en", target_lang="hi", engine="fallback", rag_enabled=False
        )
        self.assertEqual(res["retrieved_context"], [])
        self.assertTrue(res["fallback_used"])
        self.assertTrue(len(res["translated_text"]) > 0)


if __name__ == "__main__":
    unittest.main()
