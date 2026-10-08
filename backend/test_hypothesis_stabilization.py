"""
Unit & Integration Tests for Phase 3: Hypothesis Stabilization / Local Agreement.

Tests:
  1. Case A: Monotonic growing confidence ("I want to" -> "I want to build" ...)
  2. Case B: Fluctuation & Revision resilience ("what is your name" -> "what is your main" -> "what is your name")
  3. Case C: Hindi Devanagari Unicode preservation
  4. Case D: Technical terms preservation ("vector database with RAG")
  5. Case E: Number & code preservation ("HTTP 404")
  6. Empty and whitespace hypothesis tolerance
  7. Repeated identical hypotheses (streak increments, promotion)
  8. Truncated / shorter hypotheses tolerance
  9. Session isolation across multiple sessions
  10. Reset and cancellation safety
  11. Final transcript reconciliation (final transcript remains 100% authoritative)
"""

import unittest
import sys
from pathlib import Path

# Ensure backend root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.hypothesis_service import (
    hypothesis_service,
    HypothesisService,
    HypothesisConfig,
    HypothesisSessionState,
    HypothesisResult
)
from backend.core.streaming_orchestrator import streaming_orchestrator, StreamingSession


class TestHypothesisStabilization(unittest.TestCase):
    def setUp(self):
        self.service = HypothesisService(HypothesisConfig(enabled=True, min_agreements=2, min_tokens=1))
        self.state = self.service.create_session_state("test_session_1")

    def test_01_case_a_growing_confidence(self):
        """Case A: Earlier stable text remains stable as hypotheses grow monotonically."""
        # Step 1
        res1 = self.service.process_hypothesis("I want to", self.state)
        self.assertEqual(res1.stable_text, "")
        self.assertEqual(res1.unstable_text, "I want to")

        # Step 2: "I want to" agrees consecutively -> promoted to stable prefix
        res2 = self.service.process_hypothesis("I want to build", self.state)
        self.assertEqual(res2.stable_text, "I want to")
        self.assertEqual(res2.unstable_text, "build")

        # Step 3: "build" agrees consecutively -> promoted to stable prefix
        res3 = self.service.process_hypothesis("I want to build a", self.state)
        self.assertEqual(res3.stable_text, "I want to build")
        self.assertEqual(res3.unstable_text, "a")

        # Step 4: "a" agrees consecutively -> promoted
        res4 = self.service.process_hypothesis("I want to build a backend", self.state)
        self.assertEqual(res4.stable_text, "I want to build a")
        self.assertEqual(res4.unstable_text, "backend")

        # Stable prefix must have grown monotonically without any revisions
        self.assertEqual(self.state.total_revisions, 0)
        self.assertEqual(self.state.total_token_churn, 0)

    def test_02_case_b_revision_scenario_no_false_lock(self):
        """Case B: Unstable fluctuated token ('main') must NOT be permanently committed."""
        # Step 1
        _ = self.service.process_hypothesis("what is your", self.state)
        # Step 2: "what is your" agreed -> stable
        res2 = self.service.process_hypothesis("what is your name", self.state)
        self.assertEqual(res2.stable_text, "what is your")
        self.assertEqual(res2.unstable_text, "name")

        # Step 3: Fluctuates to "main" (only once)
        res3 = self.service.process_hypothesis("what is your main", self.state)
        self.assertEqual(res3.stable_text, "what is your")
        self.assertEqual(res3.unstable_text, "main")
        self.assertTrue(res3.is_revised)
        self.assertEqual(res3.token_churn, 1)

        # Step 4: Returns to "name"
        res4 = self.service.process_hypothesis("what is your name", self.state)
        self.assertEqual(res4.stable_text, "what is your")
        self.assertEqual(res4.unstable_text, "name")
        self.assertTrue(res4.is_revised)

        # Step 5: "name" appears again -> 2nd agreement -> promoted to stable
        res5 = self.service.process_hypothesis("what is your name", self.state)
        self.assertEqual(res5.stable_text, "what is your name")
        self.assertEqual(res5.unstable_text, "")

        # "main" was never locked in stable prefix
        self.assertNotIn("main", self.state.stable_tokens)
        self.assertIn("name", self.state.stable_tokens)

    def test_03_case_c_hindi_devanagari_unicode(self):
        """Case C: Hindi Devanagari text is stabilized correctly without encoding corruption."""
        # Step 1
        _ = self.service.process_hypothesis("आपका नाम", self.state)
        # Step 2: "आपका नाम" agreed -> stable
        res2 = self.service.process_hypothesis("आपका नाम क्या है?", self.state)
        self.assertEqual(res2.stable_text, "आपका नाम")
        self.assertEqual(res2.unstable_text, "क्या है?")

        # Step 3: "क्या है" agreed
        res3 = self.service.process_hypothesis("आपका नाम क्या है?", self.state)
        self.assertEqual(res3.stable_text, "आपका नाम क्या है?")
        self.assertEqual(res3.unstable_text, "")

    def test_04_case_d_technical_terms(self):
        """Case D: Technical terms like 'vector database with RAG' stabilize cleanly."""
        _ = self.service.process_hypothesis("vector database", self.state)
        res2 = self.service.process_hypothesis("vector database with RAG", self.state)
        self.assertEqual(res2.stable_text, "vector database")
        self.assertEqual(res2.unstable_text, "with RAG")

        res3 = self.service.process_hypothesis("vector database with RAG", self.state)
        self.assertEqual(res3.stable_text, "vector database with RAG")
        self.assertEqual(res3.unstable_text, "")

    def test_05_case_e_numbers_and_codes(self):
        """Case E: Numbers like 'HTTP 404 error code' stabilize accurately."""
        _ = self.service.process_hypothesis("server returned HTTP 404", self.state)
        res2 = self.service.process_hypothesis("server returned HTTP 404 error code", self.state)
        self.assertEqual(res2.stable_text, "server returned HTTP 404")
        self.assertEqual(res2.unstable_text, "error code")

        res3 = self.service.process_hypothesis("server returned HTTP 404 error code", self.state)
        self.assertEqual(res3.stable_text, "server returned HTTP 404 error code")

    def test_06_empty_and_whitespace_hypothesis(self):
        """Empty or whitespace-only partial hypotheses are safely handled."""
        res1 = self.service.process_hypothesis("", self.state)
        self.assertEqual(res1.full_text, "")
        self.assertEqual(res1.stable_text, "")
        self.assertEqual(res1.unstable_text, "")

        res2 = self.service.process_hypothesis("   ", self.state)
        self.assertEqual(res2.full_text, "")

    def test_07_repeated_hypotheses_increment_streaks(self):
        """Identical hypotheses correctly satisfy agreement requirements."""
        text = "Kubernetes orchestration"
        res1 = self.service.process_hypothesis(text, self.state)
        self.assertEqual(res1.stable_text, "")
        self.assertEqual(res1.unstable_text, text)

        res2 = self.service.process_hypothesis(text, self.state)
        self.assertEqual(res2.stable_text, text)
        self.assertEqual(res2.unstable_text, "")
        self.assertEqual(res2.stable_ratio, 1.0)

    def test_08_shorter_hypothesis_handling(self):
        """When Whisper returns an unexpectedly shorter partial, stable prefix is preserved."""
        _ = self.service.process_hypothesis("Hello world today", self.state)
        _ = self.service.process_hypothesis("Hello world today", self.state)
        self.assertEqual(" ".join(self.state.stable_tokens), "Hello world today")

        # Suddenly Whisper returns just "Hello"
        res3 = self.service.process_hypothesis("Hello", self.state)
        # Committed stable prefix is NOT deleted
        self.assertEqual(res3.stable_text, "Hello world today")

    def test_09_session_isolation(self):
        """Two concurrent streaming sessions have completely isolated stabilization states."""
        session_a = streaming_orchestrator.create_session(session_id="session_A", hypothesis_enabled=True)
        session_b = streaming_orchestrator.create_session(session_id="session_B", hypothesis_enabled=True)

        # Process on A
        res_a1 = self.service.process_hypothesis("Session A unique token", session_a.hypothesis_state)
        res_a2 = self.service.process_hypothesis("Session A unique token", session_a.hypothesis_state)

        # Process on B
        res_b1 = self.service.process_hypothesis("Session B different token", session_b.hypothesis_state)

        self.assertEqual(res_a2.stable_text, "Session A unique token")
        self.assertEqual(res_b1.stable_text, "")
        self.assertEqual(res_b1.unstable_text, "Session B different token")
        self.assertNotIn("Session A", " ".join(session_b.hypothesis_state.stable_tokens))

    def test_10_reset_for_next_turn(self):
        """Reset clears all stabilization buffers, streaks, and token metrics for the next turn."""
        _ = self.service.process_hypothesis("First turn utterance", self.state)
        _ = self.service.process_hypothesis("First turn utterance", self.state)
        self.assertTrue(len(self.state.stable_tokens) > 0)

        self.state.reset_for_next_turn()
        self.assertEqual(len(self.state.stable_tokens), 0)
        self.assertEqual(len(self.state.unstable_tokens), 0)
        self.assertEqual(len(self.state.raw_hypotheses), 0)
        self.assertEqual(self.state.total_partials, 0)
        self.assertEqual(self.state.total_revisions, 0)

    def test_11_final_transcript_reconciliation(self):
        """Final transcript remains 100% authoritative; reconciliation measures distance."""
        _ = self.service.process_hypothesis("what is your main", self.state)
        _ = self.service.process_hypothesis("what is your name", self.state)

        reconciliation = self.service.reconcile_final("What is your name?", self.state)

        # Authoritative final transcript is exact
        self.assertEqual(reconciliation["authoritative_final"], "What is your name?")
        # Reconciliation stats are computed
        self.assertIn("reconciliation_token_distance", reconciliation)
        self.assertIn("reconciliation_wer", reconciliation)
        self.assertIn("total_revisions_incurred", reconciliation)
        self.assertEqual(reconciliation["total_partials_emitted"], 2)


if __name__ == "__main__":
    unittest.main()
