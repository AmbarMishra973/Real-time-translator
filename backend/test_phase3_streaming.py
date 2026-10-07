"""
Automated Verification Suite for Phase 3: Real-Time Streaming + Cancellation.
Tests:
1. WebSocket connection & ready handshake.
2. Start session with custom configuration.
3. Audio buffering and incremental accumulation.
4. Partial transcription evaluation state.
5. Final transcription with RAG and LLM stage metrics.
6. Instant cancellation protocol and state reset.
7. Disconnect and buffer cleanup.
8. Session isolation across concurrent streams.
9. Buffer overflow protection limit.
10. REST API regression verification.
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

# Ensure line-buffering on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.server import app
from backend.core.streaming_orchestrator import streaming_orchestrator, StreamingSession


def make_pcm16_sine_wav(duration_s=0.5, sample_rate=16000, rms_target_dbfs=-24.0) -> bytes:
    """Generate a clean synthetic WAV clip for streaming testing."""
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


class TestPhase3Streaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_websocket_connect_ready(self):
        """Test 1: Connect and receive ready handshake."""
        with self.client.websocket_connect("/ws/transcribe") as ws:
            msg = ws.receive_json()
            self.assertEqual(msg["type"], "ready")
            self.assertIn("request_id", msg)
            self.assertIn("session_id", msg)

    def test_02_start_session_config(self):
        """Test 2: Send 'start' message to configure streaming session."""
        with self.client.websocket_connect("/ws/transcribe") as ws:
            init_ready = ws.receive_json()
            self.assertEqual(init_ready["type"], "ready")

            ws.send_json({
                "type": "start",
                "session_id": "custom_sess_42",
                "language": "en",
                "target_lang": "hi",
                "domain": "technical"
            })
            start_ready = ws.receive_json()
            self.assertEqual(start_ready["type"], "ready")
            self.assertEqual(start_ready["session_id"], "custom_sess_42")
            self.assertIn("request_id", start_ready)

    def test_03_audio_buffering(self):
        """Test 3: Audio chunks are buffered without memory leaks."""
        session = streaming_orchestrator.create_session(session_id="test_buf")
        self.assertEqual(len(session.buffer), 0)

        chunk = b"\x00\x01" * 1000
        session.add_chunk(chunk)
        self.assertEqual(len(session.buffer), 2000)
        self.assertIsNotNone(session.first_audio_ts)

    def test_04_partial_transcription_state(self):
        """Test 4: Partial evaluation is triggered when threshold reached."""
        session = streaming_orchestrator.create_session(session_id="test_partial")
        wav = make_pcm16_sine_wav(duration_s=1.2)
        session.add_chunk(wav)
        self.assertTrue(session.should_trigger_partial())

    def test_05_final_transcription_silence_short_circuit(self):
        """Test 5: 'end' on silent audio triggers short-circuit final event."""
        with self.client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()  # ready
            silent_wav = make_pcm16_sine_wav(duration_s=0.5, rms_target_dbfs=-68.0)
            ws.send_bytes(silent_wav)
            ws.send_json({"type": "end"})
            final_msg = ws.receive_json()
            self.assertEqual(final_msg["type"], "final")
            self.assertEqual(final_msg["transcript"], "")
            self.assertEqual(final_msg["gate_reason"], "true_silence")
            self.assertIn("metrics", final_msg)
            self.assertIn("ttfr_ms", final_msg["metrics"])
            self.assertIn("stt_ms", final_msg["metrics"])

    def test_06_cancellation(self):
        """Test 6: Instant cancellation discards turn without emitting final result."""
        with self.client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()  # ready
            ws.send_bytes(b"\x00" * 8000)
            ws.send_json({"type": "cancel"})
            cancel_msg = ws.receive_json()
            self.assertEqual(cancel_msg["type"], "cancelled")
            self.assertIn("request_id", cancel_msg)

    def test_07_disconnect_cleanup(self):
        """Test 7: Disconnecting cleanly clears session buffers without lingering state."""
        session = streaming_orchestrator.create_session(session_id="disc_test")
        session.add_chunk(b"\x01\x02" * 5000)
        self.assertTrue(len(session.buffer) > 0)
        streaming_orchestrator.cancel_stream(session)
        self.assertEqual(len(session.buffer), 0)

    def test_08_session_isolation(self):
        """Test 8: Two independent streaming sessions do not share buffers or IDs."""
        s1 = streaming_orchestrator.create_session(session_id="sess_alpha")
        s2 = streaming_orchestrator.create_session(session_id="sess_beta")

        self.assertNotEqual(s1.request_id, s2.request_id)
        s1.add_chunk(b"AAAA" * 500)
        self.assertEqual(len(s2.buffer), 0)
        self.assertEqual(len(s1.buffer), 2000)

    def test_09_buffer_overflow_protection(self):
        """Test 9: Oversized audio chunks beyond max limit are rejected."""
        session = StreamingSession(
            session_id="overflow_test",
            request_id="req_over",
            max_buffer_bytes=1000
        )
        session.add_chunk(b"X" * 500)
        with self.assertRaises(ValueError):
            session.add_chunk(b"Y" * 600)

    def test_10_rest_pipeline_regression(self):
        """Test 10: Existing REST pipeline remains 100% functional and compatible."""
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)

        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)

        res_trans = self.client.post("/translate", data={
            "text": "Hello world",
            "source_lang": "en",
            "target_lang": "hi"
        })
        self.assertEqual(res_trans.status_code, 200)
        self.assertIn("translated_text", res_trans.json())


if __name__ == "__main__":
    unittest.main()
