"""
Unit & Regression Test Suite for Phase 5 — TTS Benchmark & Service Architecture.
Validates dataset integrity, voice mapping, engine dispatch, empty input safety,
and metrics computation.
"""

import os
import json
import pytest
from fastapi import HTTPException

from backend.services.tts_service import (
    TTSService,
    VOICE_MAP,
    PIPER_VOICE_MODELS,
    pick_voice
)
from backend.benchmark_tts import (
    compute_wer,
    compute_cer,
    compute_levenshtein
)

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "evaluation",
    "datasets",
    "tts_benchmark_dataset.json"
)


class TestTTSBenchmarkSuite:
    """Test suite covering TTS dataset, service dispatch, and metrics."""

    def test_01_dataset_schema_and_size(self):
        """Verifies dataset exists, contains >=30 cases, and conforms to required schema."""
        assert os.path.exists(DATASET_PATH), f"Dataset file missing: {DATASET_PATH}"

        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        assert len(cases) >= 30, f"Expected at least 30 cases, got {len(cases)}"

        languages = set()
        categories = set()
        for c in cases:
            assert "id" in c and isinstance(c["id"], str)
            assert "language" in c and c["language"] in ("en", "hi", "hinglish")
            assert "text" in c and len(c["text"].strip()) > 0
            assert "category" in c and isinstance(c["category"], str)
            assert "expected_terms" in c and isinstance(c["expected_terms"], list)
            assert "expected_numbers" in c and isinstance(c["expected_numbers"], list)
            languages.add(c["language"])
            categories.add(c["category"])

        assert "en" in languages
        assert "hi" in languages
        assert "hinglish" in languages
        assert len([c for c in cases if c["language"] == "en"]) >= 10
        assert len([c for c in cases if c["language"] == "hi"]) >= 10
        assert len([c for c in cases if c["language"] == "hinglish"]) >= 5

    def test_02_voice_picker_and_mapping(self):
        """Verifies voice mapping selects expected neural voices for Edge-TTS."""
        assert pick_voice("en") == "en-US-JennyNeural"
        assert pick_voice("en-US") == "en-US-JennyNeural"
        assert pick_voice("hi") == "hi-IN-SwaraNeural"
        assert pick_voice("hi-IN") == "hi-IN-SwaraNeural"
        # Unknown language defaults to English Jenny
        assert pick_voice("xyz") == "en-US-JennyNeural"
        assert pick_voice(None) == "en-US-JennyNeural"

    def test_03_empty_and_whitespace_input(self):
        """Verifies empty or whitespace text raises 400 Bad Request."""
        import asyncio
        service = TTSService()
        with pytest.raises(HTTPException) as exc_info1:
            asyncio.run(service.synthesize(""))
        assert exc_info1.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info2:
            asyncio.run(service.synthesize("   \n\t  "))
        assert exc_info2.value.status_code == 400

    def test_04_engine_flag_default_edge(self):
        """Verifies default engine is 'edge' for production stability."""
        service = TTSService()
        assert service.get_active_engine() == "edge"

    def test_05_engine_flag_unfeasible_indic_tts(self):
        """Verifies Indic-TTS raises explicit 501 unfeasible error."""
        import asyncio
        service = TTSService(engine="indic_tts")
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(service.synthesize("Test input text"))
        assert exc_info.value.status_code == 501
        assert "unfeasible" in exc_info.value.detail.lower()

    def test_06_piper_model_configuration(self):
        """Verifies Piper model paths and config dictionary exist."""
        assert "en" in PIPER_VOICE_MODELS
        assert "hi" in PIPER_VOICE_MODELS
        assert "model_path" in PIPER_VOICE_MODELS["en"]
        assert "config_path" in PIPER_VOICE_MODELS["en"]

    def test_07_sapi5_voice_enumeration(self):
        """Verifies SAPI5 / pyttsx3 initialises and exposes Windows voice list."""
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        assert len(voices) > 0
        names = [v.name for v in voices]
        assert any("David" in n or "Zira" in n for n in names)

    def test_08_levenshtein_and_error_metrics(self):
        """Verifies auxiliary WER/CER/Levenshtein computation functions."""
        # Exact match
        assert compute_levenshtein("hello", "hello") == 0
        assert compute_wer("hello world", "hello world") == 0.0
        assert compute_cer("hello world", "hello world") == 0.0

        # Complete mismatch
        wer_mismatch = compute_wer("apple banana", "orange grape")
        assert wer_mismatch > 0.0

        # Empty handling
        assert compute_wer("", "") == 0.0
        assert compute_cer("", "") == 0.0
        assert compute_wer("test", "") == 1.0
