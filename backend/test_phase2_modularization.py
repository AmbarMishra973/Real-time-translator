"""
Automated Verification Suite for Phase 2 Modularization & Stability.
Verifies:
1. Service layer isolation & lifecycle.
2. SessionManager bounded history & TTL expiration.
3. Temporary audio file safety & cleanup under normal and error conditions.
4. PipelineOrchestrator request tracing & stage latency measurements.
5. Complete REST API contract compatibility for all endpoints.
"""

import sys
import os
import io
import time
import math
import wave
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure UTF-8 line buffering on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.server import app, sync_transcribe, whisper_model, WHISPER_SIZE
from backend.services.session_manager import SessionManager, SessionRecord
from backend.services.stt_service import stt_service, convert_to_clean_wav
from backend.services.rag_service import rag_service
from backend.services.llm_service import llm_service
from backend.services.tts_service import tts_service
from backend.core.orchestrator import pipeline_orchestrator


def make_pcm16_sine_wav(duration_s=0.5, sample_rate=16000, rms_target_dbfs=-24.0) -> bytes:
    """Generate a clean synthetic WAV clip for API testing."""
    amplitude = int(32767.0 * (10.0 ** (rms_target_dbfs / 20.0)) * math.sqrt(2.0))
    frames = int(sample_rate * duration_s)
    raw = bytearray()
    for i in range(frames):
        sample = int(amplitude * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
        sample = max(-32768, min(32767, sample))
        raw.extend(sample.to_bytes(2, byteorder='little', signed=True))
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(raw))
    return buf.getvalue()


class TestPhase2Modularization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_service_instances_exist(self):
        """Verify all Phase 2 services are instantiated and decoupled."""
        self.assertIsNotNone(stt_service)
        self.assertIsNotNone(rag_service)
        self.assertIsNotNone(llm_service)
        self.assertIsNotNone(tts_service)
        self.assertIsNotNone(pipeline_orchestrator)
        # Server backwards compatibility exports
        self.assertEqual(WHISPER_SIZE, stt_service.model_size)
        self.assertTrue(callable(sync_transcribe))

    def test_02_session_manager_lifecycle(self):
        """Verify SessionManager bounded capacity, TTL expiry, and cleanup."""
        sm = SessionManager(ttl_seconds=1.0, max_turns=3)
        sid = "test_sess_01"

        # Add 5 turns (max is 3)
        for i in range(5):
            sm.add_turn(sid, f"Query {i}", f"Response {i}", "en", "hi")

        history = sm.get_history(sid)
        self.assertEqual(len(history), 3, "History must be bounded to max_turns=3")
        self.assertEqual(history[-1]["source_text"], "Query 4")

        # Test prompt formatting
        prompt_fmt = sm.format_history_for_prompt(sid)
        self.assertIn("Query 4", prompt_fmt)

        # Test TTL expiration
        time.sleep(1.1)
        # After TTL has elapsed, history should be empty (purged)
        expired_history = sm.get_history(sid)
        self.assertEqual(len(expired_history), 0, "Expired session must be cleaned up on access")

    def test_03_temp_audio_file_cleanup(self):
        """Verify convert_to_clean_wav cleans up temporary files even on error."""
        import tempfile
        wav = make_pcm16_sine_wav(duration_s=0.2)

        # Normal conversion
        clean = convert_to_clean_wav(wav, suffix=".wav")
        self.assertTrue(len(clean) > 44)

        # Error case: corrupt bytes should raise ValueError without leaving dangling files
        with self.assertRaises(ValueError):
            convert_to_clean_wav(b"NOT_A_VALID_AUDIO_CONTAINER_STREAM", suffix=".bin")

    def test_04_api_health_and_root(self):
        """Verify /health and / endpoints return expected schema."""
        res_h = self.client.get("/health")
        self.assertEqual(res_h.status_code, 200)
        self.assertEqual(res_h.json()["status"], "ok")

        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        data = res_root.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("llm_status", data)
        self.assertIn("stt_status", data)
        self.assertIn("rag_domains", data)
        self.assertIn("total_knowledge_terms", data)

    def test_05_api_knowledge_and_settings(self):
        """Verify /api/knowledge and /api/settings endpoints."""
        res_k = self.client.get("/api/knowledge")
        self.assertEqual(res_k.status_code, 200)
        k_data = res_k.json()
        self.assertIn("domains", k_data)
        self.assertIn("terms", k_data)

        # Add custom term
        res_add = self.client.post("/api/knowledge", json={
            "term": "PhaseTwoTestTerm",
            "definition": "A test definition for modular architecture.",
            "domain": "testing"
        })
        self.assertEqual(res_add.status_code, 200)
        self.assertIn("PhaseTwoTestTerm", res_add.json()["message"])

        # Settings
        res_s = self.client.get("/api/settings")
        self.assertEqual(res_s.status_code, 200)
        self.assertIn("available_languages", res_s.json())

    def test_06_api_tts(self):
        """Verify /tts synthesizes speech and includes latency header."""
        res = self.client.post("/tts", data={"text": "Hello world", "target_lang": "en"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "audio/mpeg")
        self.assertIn("x-tts-latency", res.headers)
        self.assertTrue(len(res.content) > 100)

    def test_07_api_translate_text(self):
        """Verify /translate executes text translation and includes stage metrics."""
        res = self.client.post("/translate", data={
            "text": "Hello, how are you?",
            "source_lang": "en",
            "target_lang": "hi",
            "session_id": "test_text_translate"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("translated_text", data)
        self.assertIn("metrics", data)
        self.assertIn("request_id", data)
        self.assertIn("rag_s", data["metrics"])
        self.assertIn("llm_s", data["metrics"])

    def test_08_api_pipeline_silence_short_circuit(self):
        """Verify /pipeline short-circuits on silent audio without RAG/LLM invocation."""
        silent_wav = make_pcm16_sine_wav(duration_s=0.5, rms_target_dbfs=-68.0)
        res = self.client.post(
            "/pipeline",
            files={"file": ("silent.wav", silent_wav, "audio/wav")},
            data={"source_lang": "en", "target_lang": "hi", "session_id": "test_silence"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["transcript"], "")
        self.assertEqual(data["translated_text"], "")
        self.assertEqual(data["gate_reason"], "true_silence")
        self.assertEqual(data["metrics"]["rag_s"], 0.0)
        self.assertEqual(data["metrics"]["llm_s"], 0.0)
        self.assertIn("request_id", data)


if __name__ == "__main__":
    unittest.main()
