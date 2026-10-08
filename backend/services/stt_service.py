"""
Speech-To-Text (STT) Service.
Encapsulates Faster-Whisper model lifecycle, DSP signal conditioning,
silence gating, script guidance prompts, and hallucination protection.
"""

import os
import io
import re
import sys
import uuid
import time
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Any, Dict

from fastapi import HTTPException
from faster_whisper import WhisperModel

from backend.core.logger import stt_log
from backend.services.llm_service import llm_service
from backend.audio_diagnostics import (
    boost_quiet_pcm16_wav,
    inspect_pcm16_wav,
    pad_pcm16_wav,
    upload_suffix,
    validate_normalized_audio
)

SUSPECTED_HALLUCINATIONS = {
    "thank you", "thank you very much", "thanks for watching", "you", "bye", "subscribe"
}


import struct


def convert_to_clean_wav_control(audio_bytes: bytes, suffix: str = ".bin") -> bytes:
    """
    Production Control Path: Converts incoming audio into 16kHz mono WAV PCM
    using temporary disk files.
    """
    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_f:
            in_f.write(audio_bytes)
            in_path = in_f.name

        out_path = in_path + ".wav"
        cmd = [
            "ffmpeg", "-y",
            "-err_detect", "ignore_err",
            "-i", in_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        ]
        if os.getenv("STT_NORMALIZE_AUDIO", "false").lower() == "true":
            cmd.extend(["-af", "dynaudnorm=p=0.9:s=5"])
        cmd.append(out_path)

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                converted = f.read()
            if len(converted) > 100:
                return converted

        # Fallback without extra flags
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            out_path
        ]
        res2 = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res2.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                return f.read()
    finally:
        for p in [in_path, out_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    raise ValueError("Audio conversion failed; verify that FFmpeg supports the uploaded audio format.")


def convert_to_clean_wav_in_memory(audio_bytes: bytes, suffix: str = ".bin") -> bytes:
    """
    Experimental In-Memory Path (Phase 1): Converts and resamples incoming audio into
    clean 16kHz mono WAV PCM without disk I/O, using FFmpeg stdin/stdout streaming pipes
    and in-memory RIFF/data chunk header patching.
    Fast-paths audio that is already valid 16kHz mono 16-bit PCM WAV.
    """
    if not audio_bytes or len(audio_bytes) < 44:
        raise ValueError("Audio data is empty or too short.")

    # 1. Fast-path: Check if already valid 16kHz mono 16-bit PCM WAV
    if os.getenv("STT_NORMALIZE_AUDIO", "false").lower() != "true":
        if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            try:
                import wave
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    if (
                        wf.getnchannels() == 1
                        and wf.getsampwidth() == 2
                        and wf.getframerate() == 16000
                        and wf.getcomptype() == "NONE"
                    ):
                        return audio_bytes
            except Exception:
                pass

    # 2. In-memory streaming via FFmpeg pipe:0 -> pipe:1
    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", "pipe:0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
    ]
    if os.getenv("STT_NORMALIZE_AUDIO", "false").lower() == "true":
        cmd.extend(["-af", "dynaudnorm=p=0.9:s=5"])
    cmd.extend(["-f", "wav", "pipe:1"])

    res = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0 and len(res.stdout) > 44:
        raw_out = bytearray(res.stdout)
        # Patch unseekable pipe streaming header (0xFFFFFFFF in RIFF size and data size)
        riff_len = len(raw_out) - 8
        raw_out[4:8] = struct.pack("<I", riff_len)
        data_idx = raw_out.find(b"data")
        if data_idx != -1:
            data_len = len(raw_out) - (data_idx + 8)
            raw_out[data_idx + 4 : data_idx + 8] = struct.pack("<I", data_len)
        if len(raw_out) > 100:
            return bytes(raw_out)

    # 3. Fallback without extra flags
    cmd_fallback = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-f", "wav", "pipe:1",
    ]
    res2 = subprocess.run(cmd_fallback, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res2.returncode == 0 and len(res2.stdout) > 44:
        raw_out = bytearray(res2.stdout)
        riff_len = len(raw_out) - 8
        raw_out[4:8] = struct.pack("<I", riff_len)
        data_idx = raw_out.find(b"data")
        if data_idx != -1:
            data_len = len(raw_out) - (data_idx + 8)
            raw_out[data_idx + 4 : data_idx + 8] = struct.pack("<I", data_len)
        return bytes(raw_out)

    raise ValueError("Audio conversion failed; verify that FFmpeg supports the uploaded audio format.")


def convert_to_clean_wav(audio_bytes: bytes, suffix: str = ".bin") -> bytes:
    """
    Converts and resamples incoming browser audio (WebM, OGG, MP4, WAV, etc.)
    into clean 16kHz mono WAV PCM.
    Dispatches to control (disk-based) or in_memory path based on AUDIO_PIPELINE_MODE.
    Default mode: 'control' (preserves production behavior).
    """
    mode = os.getenv("AUDIO_PIPELINE_MODE", "control").strip().lower()
    if mode == "in_memory":
        return convert_to_clean_wav_in_memory(audio_bytes, suffix)
    return convert_to_clean_wav_control(audio_bytes, suffix)


def parse_capture_metadata(raw_metadata: Optional[str]) -> dict:
    """Keep only non-identifying browser audio settings supplied with an upload."""
    if not raw_metadata:
        return {}
    import json
    try:
        metadata = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(metadata, dict):
        return {}
    allowed_keys = {
        "sampleRate", "channelCount", "sampleSize", "echoCancellation",
        "noiseSuppression", "autoGainControl", "mimeType"
    }
    return {
        key: metadata[key]
        for key in allowed_keys
        if key in metadata and isinstance(metadata[key], (str, int, float, bool, type(None)))
    }


class STTService:
    """Authoritative service managing local Faster-Whisper and cloud STT dispatch."""

    def __init__(self):
        self.model_size = os.getenv("WHISPER_SIZE", "base")
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.model: Optional[WhisperModel] = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"[*] Initializing Faster-Whisper ({self.model_size} on {self.device})...", flush=True)
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("[+] Whisper model loaded successfully.", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not load {self.model_size} model ({e}).", flush=True)
            if self.model_size != "base":
                try:
                    print("[*] Attempting cached base-model fallback...", flush=True)
                    self.model = WhisperModel("base", device="cpu", compute_type="int8")
                    self.model_size = "base"
                    print("[+] Base fallback model loaded successfully.", flush=True)
                except Exception as fallback_error:
                    print(f"[!] Local Whisper is unavailable: {fallback_error}", flush=True)

    def transcribe(
        self,
        audio_bytes: bytes,
        lang: Optional[str] = None,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        capture_metadata: Optional[dict] = None
    ) -> Tuple[str, Any, Dict[str, Any], str, str]:
        """
        Processes and transcribes audio with signal diagnostics, gain boost, silence gate,
        temporal padding, prompt guidance, and hallucination protection.

        Returns:
            (clean_text, info, post_diag, engine_used, model_used)
        """
        utterance_id = uuid.uuid4().hex[:12]
        stt_started_at = time.perf_counter()

        # 1. Convert and validate incoming audio into 16kHz mono WAV
        input_suffix = upload_suffix(filename, content_type)
        wav_bytes = convert_to_clean_wav(audio_bytes, input_suffix)
        pre_diag = inspect_pcm16_wav(wav_bytes)
        validate_normalized_audio(pre_diag)

        # 2. Adaptive Peak-Safe Gain Boost
        wav_bytes, gain_db_applied = boost_quiet_pcm16_wav(wav_bytes)
        post_diag = inspect_pcm16_wav(wav_bytes)
        post_diag["pre_rms_dbfs"] = pre_diag["rms_dbfs"]
        post_diag["pre_peak_dbfs"] = pre_diag["peak_dbfs"]
        post_diag["pre_duration_s"] = pre_diag["duration_s"]
        post_diag["gain_db_applied"] = gain_db_applied

        whisper_lang = None if (not lang or lang.lower() == 'auto') else lang.split('-')[0].lower()

        # 3. Pre-Flight Silence Gate (Frozen on post-gain, pre-padding signal levels)
        if post_diag["rms_dbfs"] < -55.0:
            if pre_diag["rms_dbfs"] < -60.0:
                gate_reason = "true_silence"
            elif pre_diag["peak_dbfs"] >= -3.0 or gain_db_applied < ((-24.0 - pre_diag["rms_dbfs"]) - 1.0):
                gate_reason = "headroom_limited"
            else:
                gate_reason = "true_silence"

            post_diag["is_silent_gate"] = True
            post_diag["gate_reason"] = gate_reason
            post_diag["gate_message"] = "No speech detected — please speak closer to the microphone."
            post_diag["suspected_hallucination"] = False
            post_diag["padded_duration_s"] = post_diag["duration_s"]

            stt_log(
                "silence_gate_triggered",
                utterance_id,
                gate_reason=gate_reason,
                pre_rms_dbfs=pre_diag["rms_dbfs"],
                post_rms_dbfs=post_diag["rms_dbfs"],
                gain_applied=gain_db_applied,
                duration_s=post_diag["duration_s"],
            )
            info = SimpleNamespace(language=whisper_lang or "en", language_probability=0.0, language_source="gate")
            return "", info, post_diag, "gate", "none"

        post_diag["is_silent_gate"] = False
        post_diag["gate_reason"] = None

        # 4. Temporal Padding: Short clips (< 1.0s) padded with 250ms silence for phonetic context
        if post_diag["duration_s"] < 1.0:
            wav_bytes, padded_duration = pad_pcm16_wav(wav_bytes, pad_ms=250)
            post_diag["padded_duration_s"] = padded_duration
        else:
            post_diag["padded_duration_s"] = post_diag["duration_s"]

        # Cache debug recordings locally if configured
        debug_dir = Path("backend/debug_audio")
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            (debug_dir / f"last_recording{input_suffix}").write_bytes(audio_bytes)
            (debug_dir / "last_recording.wav").write_bytes(wav_bytes)
        except Exception as e:
            print(f"[!] Warning: Could not write debug audio: {e}", flush=True)

        stt_log(
            "audio_validated",
            utterance_id,
            input_bytes=len(audio_bytes),
            input_container=input_suffix,
            input_content_type=content_type,
            capture_metadata=capture_metadata or {},
            normalized_bytes=len(wav_bytes),
            source_language=whisper_lang or "auto",
            **post_diag
        )

        # 5. Whisper Transcription (Local Faster-Whisper primary zero-cost engine; Groq optional)
        prefer_groq = os.getenv("STT_ENGINE", "local").lower() == "groq" and llm_service.has_groq_client
        if prefer_groq:
            engine_used = "groq"
            model_used = "whisper-large-v3-turbo"
            stt_log("transcription_started", utterance_id, engine=engine_used, model=model_used, source_language=whisper_lang or "auto")
            try:
                clean_text = llm_service.transcribe_with_groq(wav_bytes, whisper_lang)
                info = SimpleNamespace(language=whisper_lang or "en", language_probability=1.0, language_source="groq")
            except Exception as e:
                stt_log("groq_failed", utterance_id, error=str(e))
                if self.model is not None:
                    # Graceful fallback to local Faster-Whisper if cloud API fails
                    engine_used = "local"
                    model_used = self.model_size
                    initial_prompt = "यह हिंदी में बातचीत है।" if whisper_lang == "hi" else None
                    segments, info = self.model.transcribe(
                        io.BytesIO(wav_bytes),
                        language=whisper_lang,
                        beam_size=1,
                        temperature=0.0,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        initial_prompt=initial_prompt,
                    )
                    clean_text = ' '.join(seg.text for seg in segments).strip()
                else:
                    raise HTTPException(status_code=502, detail=f"Groq Whisper failed: {e}")
        else:
            if self.model is None:
                raise RuntimeError("No local Whisper model is available. Check WhisperModel installation/weights.")

            engine_used = "local"
            model_used = self.model_size
            stt_log("transcription_started", utterance_id, engine=engine_used, model=model_used, source_language=whisper_lang or "auto")

            # Guide Devanagari script tokenization for Hindi to prevent Urdu Perso-Arabic transcription
            initial_prompt = "यह हिंदी में बातचीत है।" if whisper_lang == "hi" else None

            segments, info = self.model.transcribe(
                io.BytesIO(wav_bytes),
                language=whisper_lang,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
            )
            clean_text = ' '.join(seg.text for seg in segments).strip()
            if info is None:
                info = SimpleNamespace(
                    language=whisper_lang,
                    language_probability=None,
                    language_source="selected" if whisper_lang else "unknown"
                )

        # 6. Normalized Post-Transcription Hallucination Guard
        clean_norm = re.sub(r'[^\w\s]', '', (clean_text or '').lower()).strip()
        is_suspected = False
        if (
            clean_norm in SUSPECTED_HALLUCINATIONS
            and post_diag.get("pre_duration_s", post_diag["duration_s"]) > 2.0
            and post_diag.get("pre_rms_dbfs", 0.0) < -38.0
        ):
            is_suspected = True
            stt_log(
                "suspected_hallucination",
                utterance_id,
                text=clean_text,
                normalized_text=clean_norm,
                pre_rms_dbfs=post_diag.get("pre_rms_dbfs"),
                duration_s=post_diag["duration_s"]
            )
            print(f"[STT] Suspected hallucination on quiet audio ('{clean_text}'), consider re-recording", flush=True)

        post_diag["suspected_hallucination"] = is_suspected

        detected_lang = getattr(info, "language", None) or (whisper_lang or "en")
        stt_log(
            "transcription_completed",
            utterance_id,
            engine=engine_used,
            model=model_used,
            detected_language=detected_lang,
            stt_latency_s=round(time.perf_counter() - stt_started_at, 3),
            text=clean_text,
            suspected_hallucination=is_suspected,
        )
        return clean_text, info, post_diag, engine_used, model_used

    def transcribe_partial(self, audio_bytes: bytes, lang: Optional[str] = None) -> str:
        """
        Lightweight incremental transcription for intermediate streaming partials.
        Converts available buffer window to WAV and performs fast single-beam decoding.
        Returns empty string if audio is too short (< 0.8s) or decoding fails.
        """
        if not audio_bytes or len(audio_bytes) < 4000:
            return ""

        if self.model is None:
            return ""

        try:
            wav_bytes = convert_to_clean_wav(audio_bytes)
        except Exception:
            return ""

        whisper_lang = None if (not lang or lang.lower() == 'auto') else lang.split('-')[0].lower()
        initial_prompt = "यह हिंदी में बातचीत है।" if whisper_lang == "hi" else None

        try:
            segments, _ = self.model.transcribe(
                io.BytesIO(wav_bytes),
                language=whisper_lang,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
            )
            text = ' '.join(seg.text for seg in segments).strip()
            return text
        except Exception:
            return ""


# Process-level singleton instance
stt_service = STTService()

