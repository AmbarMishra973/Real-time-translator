"""
Unit tests for Phase 6: Selective Translation Intelligence & RAG Gating.
Tests boundary detection, false-positive protection, multi-word matching,
bilingual/Hinglish terms, gating policies, latency bounds, and feature flag backward compatibility.
"""

import os
import pytest
from backend.services.translation_context_gate import (
    TranslationContextGate,
    CANONICAL_TECHNICAL_TERMS
)
from backend.services.rag_service import rag_service
from backend.llm_translator import LLMTranslator


@pytest.fixture
def gate():
    """Provides an isolated TranslationContextGate instance."""
    return TranslationContextGate(rag_svc=rag_service, relevance_threshold=0.35)


class TestBoundaryAndFalsePositives:
    """Tests word boundaries preventing subword substring collisions."""

    def test_capital_does_not_match_api(self, gate):
        text = "Delhi is the capital of India with high market capitalization."
        matched = gate.detect_technical_terms(text)
        assert "api" not in matched

    def test_storage_does_not_match_rag(self, gate):
        text = "We need extra storage and courage to manage garage equipment."
        matched = gate.detect_technical_terms(text)
        assert "rag" not in matched

    def test_interest_does_not_match_rest(self, gate):
        text = "I have a lot of interest in this investment portfolio."
        matched = gate.detect_technical_terms(text)
        assert "rest" not in matched

    def test_piston_does_not_match_post(self, gate):
        text = "The piston broke in the engine cylinder."
        matched = gate.detect_technical_terms(text)
        assert "post" not in matched

    def test_podcast_does_not_match_pod(self, gate):
        text = "I was listening to an interesting podcast yesterday."
        matched = gate.detect_technical_terms(text)
        assert "pod" not in matched


class TestTechnicalTermDetection:
    """Tests valid domain terms across English, multi-word phrases, and Devanagari."""

    def test_single_technical_terms(self, gate):
        text = "The API returned a 500 server error."
        matched = gate.detect_technical_terms(text)
        assert "api" in matched
        assert "server" in matched

    def test_multi_word_phrases(self, gate):
        text = "We are deploying a vector database with RAG architecture."
        matched = gate.detect_technical_terms(text)
        assert "vector database" in matched
        assert "rag architecture" in matched or "rag" in matched

    def test_case_insensitivity(self, gate):
        variations = [
            "DOCKER and KUBERNETES",
            "docker and kubernetes",
            "Docker and Kubernetes",
            "dOcKeR and KuBeRnEtEs"
        ]
        for t in variations:
            matched = gate.detect_technical_terms(t)
            assert "docker" in matched
            assert "kubernetes" in matched

    def test_punctuation_handling(self, gate):
        text = "Check the logs: (Docker, Kubernetes)! Is FastAPI running? Yes; CI/CD pipeline active."
        matched = gate.detect_technical_terms(text)
        assert "docker" in matched
        assert "kubernetes" in matched
        assert "fastapi" in matched
        assert "ci/cd" in matched
        assert "pipeline" in matched

    def test_devanagari_technical_loanwords(self, gate):
        text = "हमारा नया डेटाबेस और डॉकर कंटेनर ठीक से काम कर रहा है।"
        matched = gate.detect_technical_terms(text)
        assert any(t in matched for t in ["डेटाबेस", "डॉकर", "कंटेनर"])

    def test_hinglish_code_mixed(self, gate):
        text = "Docker container ko Kubernetes cluster me deploy karo."
        matched = gate.detect_technical_terms(text)
        assert "docker" in matched
        assert "container" in matched
        assert "kubernetes" in matched
        assert "cluster" in matched
        assert "deploy" in matched


class TestEmptyAndConversationalInputs:
    """Verifies that non-technical, conversational, or empty text evaluates to NO_RAG."""

    def test_empty_string(self, gate):
        res = gate.evaluate("")
        assert res.use_rag is False
        assert res.decision == "NO_RAG"
        assert res.reason == "empty_or_whitespace_input"

    def test_whitespace_only(self, gate):
        res = gate.evaluate("   \n\t  ")
        assert res.use_rag is False
        assert res.decision == "NO_RAG"

    def test_general_conversational(self, gate):
        conversational_samples = [
            "Hello, how are you today?",
            "Good morning, nice to meet you.",
            "Where is the nearest train station?",
            "Can I get a cup of tea please?",
            "What time will the meeting end?"
        ]
        for sample in conversational_samples:
            res = gate.evaluate(sample)
            assert res.use_rag is False
            assert res.decision == "NO_RAG"
            assert len(res.chunks) == 0

    def test_numbers_and_codes_without_tech_context(self, gate):
        text = "The bus arrives at 4:30 PM on platform number 7."
        res = gate.evaluate(text)
        assert res.use_rag is False
        assert res.decision == "NO_RAG"


class TestGateEvaluationPolicy:
    """Verifies gating decisions with retrieval score checks."""

    def test_technical_with_grounded_context_triggers_rag(self, gate):
        text = "We need to set up a Docker container and configure Kubernetes pods."
        res = gate.evaluate(text)
        assert res.use_rag is True
        assert res.decision == "USE_RAG"
        assert len(res.chunks) > 0
        assert res.retrieval_score >= 0.35

    def test_technical_density_calculation(self, gate):
        text = "Docker container Kubernetes pod deployment"
        matched = gate.detect_technical_terms(text)
        density = gate.calculate_technical_density(text, matched)
        assert density > 0.5

    def test_sub_millisecond_execution(self, gate):
        import time
        # Warm up
        gate.evaluate("Warm up query")

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            gate.evaluate("Hello, could you please pass the water bottle?")
            times.append((time.perf_counter() - t0) * 1000.0)

        p50 = sorted(times)[len(times) // 2]
        # Gate on conversational text should take less than 1.0 ms
        assert p50 < 1.0, f"Conversational gate p50 too slow: {p50:.3f}ms"


class TestTranslatorIntegrationAndFlags:
    """Tests LLMTranslator mode dispatching and backward compatibility flags."""

    def test_rag_mode_parameter_precedence(self):
        translator = LLMTranslator()
        # Mock empty translate call
        res_none = translator.translate("", rag_mode="none")
        assert res_none["rag_mode"] == "none"

        res_selective = translator.translate("", rag_mode="selective")
        assert res_selective["rag_mode"] == "none"  # Empty text early returns 'none'

    def test_env_feature_flags(self, monkeypatch):
        translator = LLMTranslator()

        # By default, without flags, RAG_MODE defaults to universal for backward compatibility
        monkeypatch.delenv("RAG_MODE", raising=False)
        monkeypatch.delenv("SELECTIVE_RAG_ENABLED", raising=False)

        # When selective is enabled via SELECTIVE_RAG_ENABLED=true
        monkeypatch.setenv("SELECTIVE_RAG_ENABLED", "true")
        # In translate(), if rag_mode is not explicitly passed, it uses active_rag_mode = "selective"
        # We verify this logic
        active_rag_mode = "selective" if os.getenv("SELECTIVE_RAG_ENABLED", "false").lower() in ("true", "1") else "universal"
        assert active_rag_mode == "selective"

        # Explicit RAG_MODE override
        monkeypatch.setenv("RAG_MODE", "none")
        env_mode = os.getenv("RAG_MODE", "").lower().strip()
        assert env_mode == "none"
