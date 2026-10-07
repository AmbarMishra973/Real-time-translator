"""
Automated Test Suite for LLM Translation Evaluation and Offline Quality Verification.
Tests:
1. Normal translation quality criteria (non-emptiness, script sanity).
2. Numerical preservation (ASCII and Devanagari numerals).
3. Named entities preservation.
4. Technical terminology preservation.
5. Dates and entity preservation.
6. Empty and whitespace-only input handling.
7. Very short token input handling.
8. Fallback cascade activation when Groq is unconfigured or returns error.
9. Observability metadata (fallback_used, context_used, provider).
10. Deterministic evaluation report generation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.llm_service import LLMService
from backend.llm_translator import LLMTranslator
from backend.evaluation.llm_eval import TranslationEvaluator, normalize_digits


class TestLLMEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = TranslationEvaluator()
        cls.dataset = cls.evaluator.load_dataset()

    def test_01_number_preservation_deterministic(self):
        """Verifies number preservation logic correctly detects present and missing numbers."""
        # Both Latin and Devanagari numerals must be recognized
        src = "The error code is 404 and count is 10."
        trans_latin = "त्रुटि कोड 404 है और गिनती 10 है।"
        trans_dev = "त्रुटि कोड ४०४ है और गिनती १० है।"
        trans_missing = "त्रुटि कोड है और गिनती है।"

        self.assertTrue(self.evaluator.check_number_preservation(src, trans_latin, ["404", "10"]))
        self.assertTrue(self.evaluator.check_number_preservation(src, trans_dev, ["404", "10"]))
        self.assertFalse(self.evaluator.check_number_preservation(src, trans_missing, ["404", "10"]))

    def test_02_technical_term_preservation(self):
        """Verifies technical term preservation with transliteration variants."""
        trans_with_terms = "हमने डॉकर और कुबेरनेट्स का उपयोग करके REST API तैनात किया।"
        terms = [["Docker", "डॉकर"], ["Kubernetes", "कुबेरनेट्स"], ["REST API"]]
        self.assertTrue(self.evaluator.check_term_preservation(trans_with_terms, terms))

        trans_without = "हमने कुछ कंटेनर तैनात किए।"
        self.assertFalse(self.evaluator.check_term_preservation(trans_without, terms))

    def test_03_named_entity_preservation(self):
        """Verifies named entities are retained."""
        trans_with_ent = "अमित शर्मा Microsoft में काम करते हैं।"
        entities = [["Amit", "अमित"], ["Sharma", "शर्मा"], ["Microsoft"]]
        self.assertTrue(self.evaluator.check_entity_preservation(trans_with_ent, entities))

    def test_04_script_sanity_checks(self):
        """Verifies target script validation for Devanagari and Latin."""
        self.assertTrue(self.evaluator.check_script_sanity("नमस्ते दुनिया", "devanagari"))
        self.assertFalse(self.evaluator.check_script_sanity("Hello world", "devanagari"))
        self.assertTrue(self.evaluator.check_script_sanity("Hello world", "latin"))
        self.assertFalse(self.evaluator.check_script_sanity("नमस्ते दुनिया", "latin"))

    def test_05_empty_and_whitespace_input(self):
        """Verifies translator returns empty results gracefully without crashing on empty inputs."""
        translator = LLMTranslator()
        res_empty = translator.translate("")
        self.assertEqual(res_empty["translated_text"], "")
        self.assertEqual(res_empty["retrieved_context"], [])
        self.assertEqual(res_empty["context_used"], False)

        res_spaces = translator.translate("    ")
        self.assertEqual(res_spaces["translated_text"], "")

    def test_06_very_short_and_single_word_input(self):
        """Verifies translation handles short single-word tokens cleanly."""
        translator = LLMTranslator()
        res = translator.translate("Hello", source_lang="en", target_lang="hi")
        self.assertTrue(len(res["translated_text"]) > 0)
        self.assertIn("provider", res)

    def test_07_fallback_activation_when_groq_unavailable(self):
        """Verifies fallback cascade activates and sets observability fields when Groq is absent."""
        translator = LLMTranslator()
        translator._groq_client = None  # Force absence of Groq client
        res = translator.translate("Good morning", source_lang="en", target_lang="hi")
        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["fallback_reason"], "groq_client_not_configured")
        self.assertEqual(res["provider"], "Local Multilingual Engine")
        self.assertTrue(len(res["translated_text"]) > 0)

    def test_08_provider_failure_resilience(self):
        """Verifies translator gracefully recovers via fallback if Groq raises an exception."""
        translator = LLMTranslator()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Simulated Groq 503 Outage")
        translator._groq_client = mock_client

        res = translator.translate("Thank you very much", source_lang="en", target_lang="hi")
        self.assertTrue(res["fallback_used"])
        self.assertIn("Simulated Groq 503 Outage", res["fallback_reason"])
        self.assertEqual(res["provider"], "Local Multilingual Engine")
        self.assertTrue(len(res["translated_text"]) > 0)

    def test_09_offline_deterministic_evaluation(self):
        """Verifies the evaluator evaluates an entire offline mock map without network calls."""
        mock_map = {
            case["id"]: ("त्रुटि कोड 404 और 2500 अनुरोध " + case["source_text"]) if case.get("target_script") == "devanagari" else ("English translated text " + case["source_text"])
            for case in self.dataset
        }
        summary = self.evaluator.evaluate(self.dataset, translation_override_map=mock_map)
        self.assertEqual(summary["total_cases"], len(self.dataset))
        self.assertIn("criteria_breakdown", summary)


if __name__ == "__main__":
    unittest.main()
