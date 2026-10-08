"""
Voice Activity Detection (VAD) & Endpointing Service (Phase 2).
Encapsulates Silero VAD (v5 ONNX) execution, configurable speech threshold gating,
short-utterance preservation, hangover timing, and conversational endpoint detection.
"""

import os
import io
import time
import wave
import struct
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from faster_whisper.vad import get_vad_model, SileroVADModel, VadOptions

DEFAULT_VAD_ENABLED = os.getenv("VAD_ENABLED", "false").lower() == "true"
DEFAULT_VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
DEFAULT_VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "100"))
DEFAULT_VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "600"))
DEFAULT_VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "150"))
DEFAULT_VAD_HANGOVER_MS = int(os.getenv("VAD_HANGOVER_MS", "300"))


@dataclass
class VADConfig:
    """Configurable settings for Silero VAD and conversational endpointing."""
    enabled: bool = DEFAULT_VAD_ENABLED
    threshold: float = DEFAULT_VAD_THRESHOLD
    min_speech_duration_ms: int = DEFAULT_VAD_MIN_SPEECH_MS
    min_silence_duration_ms: int = DEFAULT_VAD_MIN_SILENCE_MS
    speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS
    hangover_ms: int = DEFAULT_VAD_HANGOVER_MS
    sample_rate: int = 16000
    window_size_samples: int = 512  # 32 ms per window at 16kHz


@dataclass
class VADSessionState:
    """Session-isolated speech and endpoint tracking state."""
    session_id: str
    is_speech_active: bool = False
    speech_onset_ms: Optional[float] = None
    last_speech_ms: Optional[float] = None
    total_speech_ms: float = 0.0
    continuous_silence_ms: float = 0.0
    endpoint_detected: bool = False
    processed_ms: float = 0.0
    chunks_processed: int = 0
    speech_chunks_count: int = 0
    silence_chunks_count: int = 0

    def reset_for_next_turn(self) -> None:
        """Resets utterance-level state while keeping session identifier."""
        self.is_speech_active = False
        self.speech_onset_ms = None
        self.last_speech_ms = None
        self.total_speech_ms = 0.0
        self.continuous_silence_ms = 0.0
        self.endpoint_detected = False
        self.processed_ms = 0.0
        self.chunks_processed = 0
        self.speech_chunks_count = 0
        self.silence_chunks_count = 0


class VADService:
    """
    Authoritative service managing local Silero VAD v5 ONNX inference,
    speech probability scoring, and conversational endpoint detection.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._model: Optional[SileroVADModel] = None
        self.startup_latency_ms: float = 0.0
        self.first_inference_latency_ms: float = 0.0
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initializes the Silero VAD v5 ONNX model instance and benchmarks startup cost."""
        t0 = time.perf_counter()
        try:
            self._model = get_vad_model()
            self.startup_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            # Run one warm-up inference (16384 zero samples = 1.024s)
            t_warm0 = time.perf_counter()
            dummy = np.zeros(16384, dtype=np.float32)
            self._model(dummy.reshape(1, -1), num_samples=self.config.window_size_samples)
            self.first_inference_latency_ms = round((time.perf_counter() - t_warm0) * 1000.0, 2)
        except Exception as e:
            print(f"[!] Warning: Could not initialize Silero VAD model: {e}", flush=True)
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled and self.is_available

    def pcm_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """Converts raw 16-bit mono PCM bytes, WAV, or WebM audio to normalized float32 array in [-1.0, 1.0]."""
        if not pcm_bytes:
            return np.empty(0, dtype=np.float32)

        # Standard WAV container
        if pcm_bytes[:4] == b"RIFF" and pcm_bytes[8:12] == b"WAVE":
            try:
                with wave.open(io.BytesIO(pcm_bytes), "rb") as wf:
                    raw_frames = wf.readframes(wf.getnframes())
                return np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                pass

        # WebM / Matroska or other compressed audio containers
        if pcm_bytes[:4] == b"\x1a\x45\xdf\xa3" or pcm_bytes[:4] == b"OggS":
            try:
                from backend.services.stt_service import convert_to_clean_wav_in_memory
                clean_wav = convert_to_clean_wav_in_memory(pcm_bytes)
                with wave.open(io.BytesIO(clean_wav), "rb") as wf:
                    raw_frames = wf.readframes(wf.getnframes())
                return np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                pass

        # Raw PCM16: align to 2-byte boundary
        usable_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if usable_len == 0:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(pcm_bytes[:usable_len], dtype=np.int16).astype(np.float32) / 32768.0

    def get_speech_probabilities(self, audio: np.ndarray) -> np.ndarray:
        """
        Runs Silero VAD on float32 audio array and returns an array of speech
        probabilities across 32ms (512 sample) windows.
        """
        if self._model is None or len(audio) == 0:
            return np.empty(0, dtype=np.float32)

        window_size = self.config.window_size_samples
        pad_len = (window_size - (len(audio) % window_size)) % window_size
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))

        n_windows = len(audio) // window_size
        if n_windows == 0:
            return np.empty(0, dtype=np.float32)

        # Batch in smaller window chunks (max 16) to avoid ONNXRuntime Conv allocation spikes
        chunk_windows = 16
        probs_list = []
        for i in range(0, n_windows, chunk_windows):
            sub_audio = audio[i * window_size : min(len(audio), (i + chunk_windows) * window_size)]
            sub_probs = self._model(sub_audio.reshape(1, -1), num_samples=window_size).squeeze()
            if sub_probs.ndim == 0:
                probs_list.append(sub_probs.item())
            else:
                probs_list.extend(sub_probs.tolist())
        return np.array(probs_list, dtype=np.float32)

    def detect_speech_probability(self, audio_data: bytes | np.ndarray) -> float:
        """
        Calculates peak speech probability across all windows in the provided audio.
        Returns 0.0 if audio is empty or model unavailable.
        """
        if isinstance(audio_data, bytes):
            audio = self.pcm_to_float32(audio_data)
        else:
            audio = audio_data

        if len(audio) < self.config.window_size_samples:
            return 0.0

        probs = self.get_speech_probabilities(audio)
        if len(probs) == 0:
            return 0.0
        return float(probs.max())

    def has_speech(self, audio_data: bytes | np.ndarray, threshold: Optional[float] = None) -> bool:
        """Determines if any speech is present exceeding the specified or configured threshold."""
        th = threshold if threshold is not None else self.config.threshold
        return self.detect_speech_probability(audio_data) >= th

    def get_speech_timestamps(
        self,
        audio_data: bytes | np.ndarray,
        threshold: Optional[float] = None,
        min_speech_ms: Optional[int] = None,
        min_silence_ms: Optional[int] = None,
        speech_pad_ms: Optional[int] = None,
    ) -> List[Dict[str, int]]:
        """
        Identifies active speech intervals (start and end sample indices)
        using Silero VAD with hangover padding.
        """
        if isinstance(audio_data, bytes):
            audio = self.pcm_to_float32(audio_data)
        else:
            audio = audio_data

        if len(audio) < self.config.window_size_samples:
            return []

        th = threshold if threshold is not None else self.config.threshold
        min_sp = min_speech_ms if min_speech_ms is not None else self.config.min_speech_duration_ms
        min_sil = min_silence_ms if min_silence_ms is not None else self.config.min_silence_duration_ms
        pad_ms = speech_pad_ms if speech_pad_ms is not None else self.config.speech_pad_ms

        opts = VadOptions(
            threshold=th,
            min_speech_duration_ms=min_sp,
            min_silence_duration_ms=min_sil,
            speech_pad_ms=pad_ms,
        )

        from faster_whisper.vad import get_speech_timestamps as fv_timestamps
        return fv_timestamps(audio, vad_options=opts, sampling_rate=self.config.sample_rate)

    def create_session_state(self, session_id: str) -> VADSessionState:
        """Creates an isolated VAD session state for tracking an active streaming turn."""
        return VADSessionState(session_id=session_id)

    def process_streaming_chunk(
        self,
        chunk_data: bytes | np.ndarray,
        state: VADSessionState,
        config: Optional[VADConfig] = None
    ) -> Dict[str, Any]:
        """
        Incrementally processes an incoming streaming audio chunk against the session state.
        Updates speech onset, speech offset, continuous silence, and endpoint detection.

        Returns:
          {
            "is_speech": bool,
            "speech_prob": float,
            "is_endpoint": bool,
            "speech_active": bool,
            "continuous_silence_ms": float,
            "total_speech_ms": float,
            "processed_ms": float
          }
        """
        cfg = config or self.config

        if isinstance(chunk_data, bytes):
            audio = self.pcm_to_float32(chunk_data)
        else:
            audio = chunk_data

        chunk_duration_ms = (len(audio) / cfg.sample_rate) * 1000.0 if cfg.sample_rate else 0.0
        state.processed_ms += chunk_duration_ms
        state.chunks_processed += 1

        if len(audio) < cfg.window_size_samples or self._model is None:
            return {
                "is_speech": False,
                "speech_prob": 0.0,
                "is_endpoint": False,
                "speech_active": state.is_speech_active,
                "continuous_silence_ms": state.continuous_silence_ms,
                "total_speech_ms": state.total_speech_ms,
                "processed_ms": round(state.processed_ms, 1),
            }

        probs = self.get_speech_probabilities(audio)
        peak_prob = float(probs.max()) if len(probs) > 0 else 0.0
        mean_prob = float(probs.mean()) if len(probs) > 0 else 0.0

        is_chunk_speech = (peak_prob >= cfg.threshold)

        if is_chunk_speech:
            state.speech_chunks_count += 1
            state.continuous_silence_ms = 0.0
            state.last_speech_ms = state.processed_ms

            if not state.is_speech_active:
                state.is_speech_active = True
                state.speech_onset_ms = max(0.0, state.processed_ms - chunk_duration_ms)

            state.total_speech_ms += chunk_duration_ms
        else:
            state.silence_chunks_count += 1
            if state.is_speech_active:
                state.continuous_silence_ms += chunk_duration_ms

                # Check if continuous silence exceeds the endpoint threshold
                # (plus hangover window to preserve word endings)
                required_silence = cfg.min_silence_duration_ms + (cfg.hangover_ms // 2)
                if state.continuous_silence_ms >= required_silence:
                    # Verify total speech meets minimum threshold before triggering endpoint
                    if state.total_speech_ms >= cfg.min_speech_duration_ms:
                        state.endpoint_detected = True
                        state.is_speech_active = False

        return {
            "is_speech": is_chunk_speech,
            "speech_prob": round(peak_prob, 3),
            "mean_prob": round(mean_prob, 3),
            "is_endpoint": state.endpoint_detected,
            "speech_active": state.is_speech_active,
            "continuous_silence_ms": round(state.continuous_silence_ms, 1),
            "total_speech_ms": round(state.total_speech_ms, 1),
            "processed_ms": round(state.processed_ms, 1),
        }


# Global singleton instance
vad_service = VADService()
