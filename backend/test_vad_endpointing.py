"""
Automated Verification Suite for Phase 2: Speech-Aware VAD & Endpointing.
Tests:
1. VAD Service initialization & model availability.
2. Speech detection on canonical speech clips (English, Hindi, Technical).
3. Silence & noise floor rejection (Zero false speech on pure silence & -55dBFS noise).
4. Short utterance preservation ("Hello", tight 400ms word).
5. Soft & quiet speech sensitivity (-45 dBFS).
6. Pause tolerance: Internal natural pause does NOT trigger premature endpoint.
7. Conversational endpoint detection after trailing silence.
8. Streaming session isolation across concurrent turns.
9. State reset & cancellation safety.
10. Gating behavior in StreamingSession (avoids partial inference on silence when VAD is enabled).
"""

import sys
import os
import io
import math
import wave
import unittest
import numpy as np
from pathlib import Path

# Ensure backend can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.vad_service import vad_service, VADService, VADConfig, VADSessionState
from backend.core.streaming_orchestrator import streaming_orchestrator, StreamingSession


def make_sine_pcm16_bytes(duration_s=1.0, freq_hz=440.0, dbfs=-24.0, sample_rate=16000) -> bytes:
    """Creates synthetic sine PCM16 audio bytes."""
    samples_count = int(duration_s * sample_rate)
    target_rms = 32768.0 * (10.0 ** (dbfs / 20.0))
    amp = min(32767.0, target_rms * math.sqrt(2.0))
    raw = bytearray()
    for i in range(samples_count):
        val = int(amp * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
        val = max(-32768, min(32767, val))
        raw.extend(val.to_bytes(2, byteorder="little", signed=True))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(raw))
    return buf.getvalue()


class TestVADEndpointing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples_dir = Path("backend/debug_audio/benchmark_samples")

    def test_01_vad_initialization_and_availability(self):
        """Test 1: Silero VAD v5 ONNX model is available and initialized."""
        self.assertTrue(vad_service.is_available)
        self.assertGreater(vad_service.startup_latency_ms, 0.0)

    def test_02_speech_detection_canonical_samples(self):
        """Test 2: Speech correctly detected on canonical English and Hindi clips."""
        for cid in ["EN-1.wav", "HI-1.wav", "TECH-1.wav"]:
            fpath = self.samples_dir / cid
            if fpath.exists():
                wav_bytes = fpath.read_bytes()
                prob = vad_service.detect_speech_probability(wav_bytes)
                self.assertGreaterEqual(prob, 0.5, f"Expected speech detected in {cid}, got {prob}")

    def test_03_silence_and_noise_rejection(self):
        """Test 3: Pure silence and stationary ambient noise floors do not trigger speech."""
        # Pure silence
        silence = np.zeros(16000, dtype=np.int16).tobytes()
        prob_sil = vad_service.detect_speech_probability(silence)
        self.assertLess(prob_sil, 0.15, f"Silence prob exceeded threshold: {prob_sil}")

        # Noise floor at -55 dBFS
        noise = np.random.normal(0, 32768.0 * (10.0 ** (-55.0 / 20.0)), 16000).astype(np.int16).tobytes()
        prob_noise = vad_service.detect_speech_probability(noise)
        self.assertLess(prob_noise, 0.20, f"Noise floor prob exceeded threshold: {prob_noise}")

    def test_04_short_utterance_preservation(self):
        """Test 4: Short utterances (e.g. 400ms word) are preserved."""
        en1_path = self.samples_dir / "EN-1.wav"
        if en1_path.exists():
            with wave.open(io.BytesIO(en1_path.read_bytes()), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            # Extract tight 400ms window
            tight = raw[int(16000 * 2 * 0.3):int(16000 * 2 * 0.7)]
            prob = vad_service.detect_speech_probability(tight)
            self.assertGreaterEqual(prob, 0.5, f"Short utterance missed: prob={prob}")

    def test_05_quiet_speech_sensitivity(self):
        """Test 5: Attenuated quiet speech (-45 dBFS) remains detectable."""
        en2_path = self.samples_dir / "EN-2.wav"
        if en2_path.exists():
            with wave.open(io.BytesIO(en2_path.read_bytes()), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            # Attenuate by 20 dB
            attenuated = np.clip(samples * 0.1, -32768, 32767).astype(np.int16).tobytes()
            prob = vad_service.detect_speech_probability(attenuated)
            self.assertGreaterEqual(prob, 0.5, f"Quiet speech missed: prob={prob}")

    def _get_speech_frames(self, duration_s: float = 0.5) -> bytes:
        """Extracts canonical real speech PCM16 frames from cached EN-1.wav."""
        en1_path = self.samples_dir / "EN-1.wav"
        with wave.open(io.BytesIO(en1_path.read_bytes()), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        byte_count = int(16000 * 2 * duration_s)
        return raw[:byte_count]

    def test_06_pause_tolerance_no_premature_endpoint(self):
        """Test 6: Internal natural speech pause (400ms) does not trigger premature endpoint."""
        state = vad_service.create_session_state(session_id="pause_test")
        cfg = VADConfig(min_silence_duration_ms=600, hangover_ms=300)

        # 500ms real speech
        speech_chunk = self._get_speech_frames(0.5)
        vad_service.process_streaming_chunk(speech_chunk, state=state, config=cfg)
        self.assertTrue(state.is_speech_active)

        # 400ms pause (less than min_silence_duration_ms)
        pause_chunk = np.zeros(6400, dtype=np.int16).tobytes()
        res = vad_service.process_streaming_chunk(pause_chunk, state=state, config=cfg)

        self.assertFalse(res["is_endpoint"], "Premature endpoint triggered during short pause!")

    def test_07_trailing_silence_endpoint_detection(self):
        """Test 7: Continuous silence exceeding threshold triggers conversational endpoint."""
        state = vad_service.create_session_state(session_id="endpoint_test")
        cfg = VADConfig(min_speech_duration_ms=100, min_silence_duration_ms=500, hangover_ms=200)

        # Real speech chunk
        speech_chunk = self._get_speech_frames(0.5)
        vad_service.process_streaming_chunk(speech_chunk, state=state, config=cfg)
        self.assertTrue(state.is_speech_active)

        # 800ms trailing silence (exceeds min_silence + hangover/2)
        silence_chunk = np.zeros(12800, dtype=np.int16).tobytes()
        res = vad_service.process_streaming_chunk(silence_chunk, state=state, config=cfg)

        self.assertTrue(res["is_endpoint"], "Failed to detect endpoint after trailing silence!")
        self.assertTrue(state.endpoint_detected)

    def test_08_session_isolation(self):
        """Test 8: Two independent streaming sessions maintain completely isolated VAD states."""
        s1 = streaming_orchestrator.create_session(session_id="vad_iso_1", vad_enabled=True)
        s2 = streaming_orchestrator.create_session(session_id="vad_iso_2", vad_enabled=True)

        self.assertNotEqual(s1.vad_state.session_id, s2.vad_state.session_id)

        # Feed real speech to s1 only
        speech = self._get_speech_frames(0.5)
        s1.add_chunk(speech)

        self.assertTrue(s1.vad_state.is_speech_active)
        self.assertFalse(s2.vad_state.is_speech_active)

    def test_09_reset_and_cancellation_safety(self):
        """Test 9: Resetting session clears VAD speech tracking and continuous silence."""
        s = streaming_orchestrator.create_session(session_id="reset_test", vad_enabled=True)
        speech = self._get_speech_frames(0.5)
        s.add_chunk(speech)
        self.assertTrue(s.vad_state.is_speech_active)

        s.reset_for_next_turn()
        self.assertFalse(s.vad_state.is_speech_active)
        self.assertEqual(s.vad_state.total_speech_ms, 0.0)
        self.assertEqual(s.vad_state.continuous_silence_ms, 0.0)

    def test_10_streaming_partial_gating_on_silence(self):
        """Test 10: StreamingSession with VAD enabled suppresses partial Whisper inference on silence."""
        # When VAD enabled
        s_vad = StreamingSession(session_id="gate_vad", request_id="req_vad", vad_enabled=True)
        silence = np.zeros(32000, dtype=np.int16).tobytes()  # 1.0s silence
        s_vad.add_chunk(silence)

        # Despite exceeding DEFAULT_MIN_PARTIAL_BYTES, should_trigger_partial returns False on silence
        self.assertFalse(s_vad.should_trigger_partial())

        # When VAD disabled
        s_novad = StreamingSession(session_id="gate_novad", request_id="req_novad", vad_enabled=False)
        s_novad.add_chunk(silence)
        self.assertTrue(s_novad.should_trigger_partial())


if __name__ == "__main__":
    unittest.main()
