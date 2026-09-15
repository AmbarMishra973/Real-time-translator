"""Regression checks for audio validation before STT inference."""

import io
import math
import unittest
import wave

from backend.audio_diagnostics import inspect_pcm16_wav, upload_suffix, validate_normalized_audio


def pcm16_wav(sample_rate=16000, channels=1, duration_s=0.5, amplitude=1000):
    frames = int(sample_rate * duration_s)
    samples = bytearray()
    for frame in range(frames):
        sample = int(amplitude * math.sin(2 * math.pi * 220 * frame / sample_rate))
        for _ in range(channels):
            samples.extend(sample.to_bytes(2, byteorder="little", signed=True))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(samples))
    return output.getvalue()


class AudioDiagnosticsTests(unittest.TestCase):
    def test_reports_normalized_pcm_metadata(self):
        diagnostics = inspect_pcm16_wav(pcm16_wav())
        self.assertEqual(diagnostics["sample_rate_hz"], 16000)
        self.assertEqual(diagnostics["channels"], 1)
        self.assertEqual(diagnostics["bit_depth"], 16)
        self.assertFalse(diagnostics["is_silent"])
        validate_normalized_audio(diagnostics)

    def test_marks_silence_without_rejecting_a_valid_wav(self):
        diagnostics = inspect_pcm16_wav(pcm16_wav(amplitude=0))
        self.assertTrue(diagnostics["is_silent"])
        validate_normalized_audio(diagnostics)

    def test_rejects_non_normalized_audio(self):
        diagnostics = inspect_pcm16_wav(pcm16_wav(sample_rate=48000, channels=2))
        with self.assertRaises(ValueError):
            validate_normalized_audio(diagnostics)

    def test_preserves_safe_container_suffixes(self):
        self.assertEqual(upload_suffix("recording.webm", "audio/webm"), ".webm")
        self.assertEqual(upload_suffix(None, "audio/ogg; codecs=opus"), ".ogg")
        self.assertEqual(upload_suffix("untrusted.exe", None), ".bin")


if __name__ == "__main__":
    unittest.main()
