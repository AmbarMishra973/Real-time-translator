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
    """
    Adaptive peak-safe gain boost for quiet PCM16 audio.
    - Preserves true silence (< -60.0 dBFS) to avoid amplifying the room noise floor.
    - Skips audio already at or above trigger_dbfs (default -38.0 dBFS).
    - Clamps applied gain so peak + gain never exceeds -1.0 dBFS (guaranteed 1 dB headroom).
    - Exact byte match returned if no gain is applied.
    """
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        raw_frames = wav_file.readframes(frames)

    if sample_width != 2 or frames == 0:
        return wav_bytes, 0.0

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

    # True silence: do not amplify room noise floor
    if rms_dbfs < -60.0:
        return wav_bytes, 0.0

    # Already sufficiently loud: return unchanged
    if rms_dbfs >= trigger_dbfs:
        return wav_bytes, 0.0

    wanted_gain_db = target_dbfs - rms_dbfs
    max_allowed_by_peak = -1.0 - peak_dbfs  # Ensure 1.0 dB headroom below 0 dBFS
    gain_db = min(wanted_gain_db, max_gain_db, max_allowed_by_peak)
    gain_db = max(gain_db, 0.0)

    if gain_db <= 0.01:
        return wav_bytes, 0.0

    factor = 10.0 ** (gain_db / 20.0)
    boosted = array.array("h")
    for s in samples:
        val = int(round(s * factor))
        if val > 32767:
            val = 32767
        elif val < -32768:
            val = -32768
        boosted.append(val)

    if sys.byteorder != "little":
        boosted.byteswap()

    out_buf = BytesIO()
    with wave.open(out_buf, "wb") as out_wav:
        out_wav.setnchannels(channels)
        out_wav.setsampwidth(sample_width)
        out_wav.setframerate(sample_rate)
        out_wav.writeframes(boosted.tobytes())

    return out_buf.getvalue(), round(gain_db, 1)


def pad_pcm16_wav(wav_bytes: bytes, pad_ms: int = 250) -> tuple[bytes, float]:
    """
    Prepend and append true digital silence (zero samples) to short clips (< 1.0s)
    to provide temporal context for Whisper phoneme recognition.
    """
    with wave.open(BytesIO(wav_bytes), "rb") as in_wav:
        channels = in_wav.getnchannels()
        sample_width = in_wav.getsampwidth()
        sample_rate = in_wav.getframerate()
        frames = in_wav.getnframes()
        raw_frames = in_wav.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit audio.")

    duration_s = frames / sample_rate if sample_rate else 0.0
    if duration_s >= 1.0:
        return wav_bytes, round(duration_s, 3)

    samples = array.array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()

    pad_samples_count = int(sample_rate * (pad_ms / 1000.0)) * channels
    pad_zeros = array.array("h", [0] * pad_samples_count)
    padded_samples = pad_zeros + samples + pad_zeros

    if sys.byteorder != "little":
        padded_samples.byteswap()

    out_buf = BytesIO()
    with wave.open(out_buf, "wb") as out_wav:
        out_wav.setnchannels(channels)
        out_wav.setsampwidth(sample_width)
        out_wav.setframerate(sample_rate)
        out_wav.writeframes(padded_samples.tobytes())

    total_frames = len(padded_samples) // channels
    new_duration_s = round(total_frames / sample_rate, 3) if sample_rate else 0.0
    return out_buf.getvalue(), new_duration_s


