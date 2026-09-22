"""Small, dependency-free checks for decoded PCM WAV audio."""

from __future__ import annotations

import array
import math
import sys
import wave
from io import BytesIO
from pathlib import Path
from typing import Any


ALLOWED_AUDIO_SUFFIXES = {".webm", ".wav", ".ogg", ".mp3", ".m4a", ".mp4", ".aac", ".flac"}
CONTENT_TYPE_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
}


def upload_suffix(filename: str | None, content_type: str | None = None) -> str:
    """Return a safe suffix that helps FFmpeg identify an uploaded container."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_AUDIO_SUFFIXES:
        return suffix
    return CONTENT_TYPE_SUFFIXES.get((content_type or "").split(";", 1)[0].lower(), ".bin")


def inspect_pcm16_wav(wav_bytes: bytes) -> dict[str, Any]:
    """Return non-sensitive signal metadata for a 16-bit PCM WAV payload."""
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        raw_frames = wav_file.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit audio.")

    samples = array.array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()

    peak = max((abs(sample) for sample in samples), default=0)
    mean_square = sum(sample * sample for sample in samples) / len(samples) if samples else 0.0
    full_scale = 32768.0
    rms = math.sqrt(mean_square)
    rms_dbfs = 20 * math.log10(max(rms / full_scale, 1e-12))
    peak_dbfs = 20 * math.log10(max(peak / full_scale, 1e-12))

    return {
        "container": "wav",
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "bit_depth": sample_width * 8,
        "frames": frames,
        "duration_s": round(frames / sample_rate, 3) if sample_rate else 0.0,
        "rms_dbfs": round(rms_dbfs, 1),
        "peak_dbfs": round(peak_dbfs, 1),
        # This is diagnostic only. Quiet recordings must still reach STT and its non-VAD fallback.
        "is_silent": rms_dbfs < -65.0,
        "is_clipping": peak >= 32700,
    }


def validate_normalized_audio(diagnostics: dict[str, Any]) -> None:
    """Reject malformed conversion output, while treating silence as a normal STT outcome."""
    if diagnostics["sample_rate_hz"] != 16000 or diagnostics["channels"] != 1 or diagnostics["bit_depth"] != 16:
        raise ValueError("Audio conversion did not produce 16 kHz mono 16-bit PCM.")
    if diagnostics["duration_s"] < 0.08:
        raise ValueError("Recording is too short to transcribe reliably.")


def boost_quiet_pcm16_wav(
    wav_bytes: bytes,
    trigger_dbfs: float = -38.0,
    target_dbfs: float = -24.0,
    max_gain_db: float = 30.0,
) -> tuple[bytes, float]:
    """Disabled: returns raw audio untouched with 0.0 dB gain to prevent amplifying background noise."""
    return wav_bytes, 0.0

