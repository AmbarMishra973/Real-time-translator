"""
Whisper Call Profiler (Phase 8.1 & 8.2).
Provides structured, non-intrusive instrumentation for every Faster-Whisper invocation:
- Call sequence number and timestamp
- Invocation reason (partial, final, rest_pipeline, vad_gated, etc.)
- Audio duration and sample count supplied to Whisper
- Pure inference duration
- Cumulative audio processed vs actual speech duration (amplification factor)
- Partial vs final classification and hypothesis stabilization tracking
"""

import time
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class WhisperCallRecord:
    call_id: int
    session_id: str
    request_id: str
    timestamp: float
    reason: str                  # "partial_stream", "final_stream", "rest_pipeline", "benchmark", etc.
    call_type: str               # "partial" or "final"
    audio_duration_s: float
    sample_count: int
    inference_duration_s: float
    text: str
    accepted: bool
    hypothesis_applied: bool
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TurnSummary:
    session_id: str
    request_id: str
    call_count: int
    actual_speech_duration_s: float
    total_audio_processed_s: float
    processing_amplification: float
    total_inference_time_s: float
    rtf_cumulative: float
    calls: List[WhisperCallRecord] = field(default_factory=list)

class WhisperProfiler:
    """Singleton profiler tracking all Whisper invocations across sessions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._call_counter = 0
        self._history: List[WhisperCallRecord] = []
        self._active_sessions: Dict[str, List[WhisperCallRecord]] = {}

    def record_call(
        self,
        session_id: str,
        request_id: str,
        reason: str,
        call_type: str,
        audio_duration_s: float,
        sample_count: int,
        inference_duration_s: float,
        text: str,
        accepted: bool = True,
        hypothesis_applied: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> WhisperCallRecord:
        with self._lock:
            self._call_counter += 1
            record = WhisperCallRecord(
                call_id=self._call_counter,
                session_id=session_id or "default",
                request_id=request_id or "default",
                timestamp=time.time(),
                reason=reason,
                call_type=call_type,
                audio_duration_s=round(audio_duration_s, 3),
                sample_count=sample_count,
                inference_duration_s=round(inference_duration_s, 4),
                text=(text or "").strip(),
                accepted=accepted,
                hypothesis_applied=hypothesis_applied,
                extra=extra or {}
            )
            self._history.append(record)
            sid = record.session_id
            if sid not in self._active_sessions:
                self._active_sessions[sid] = []
            self._active_sessions[sid].append(record)
            return record

    def get_turn_summary(self, session_id: str, actual_duration_s: Optional[float] = None) -> TurnSummary:
        with self._lock:
            calls = list(self._active_sessions.get(session_id, []))
            if not calls:
                # Try finding by request_id
                calls = [c for c in self._history if c.request_id == session_id]

            call_count = len(calls)
            total_audio_processed_s = round(sum(c.audio_duration_s for c in calls), 3)
            total_inference_time_s = round(sum(c.inference_duration_s for c in calls), 4)

            # Determine actual duration: max single call audio duration or provided duration
            if actual_duration_s is not None and actual_duration_s > 0:
                speech_dur = actual_duration_s
            else:
                speech_dur = max([c.audio_duration_s for c in calls], default=0.0)

            amplification = round(total_audio_processed_s / max(speech_dur, 0.001), 2)
            rtf = round(total_inference_time_s / max(speech_dur, 0.001), 2)

            rid = calls[-1].request_id if calls else session_id
            return TurnSummary(
                session_id=session_id,
                request_id=rid,
                call_count=call_count,
                actual_speech_duration_s=round(speech_dur, 3),
                total_audio_processed_s=total_audio_processed_s,
                processing_amplification=amplification,
                total_inference_time_s=total_inference_time_s,
                rtf_cumulative=rtf,
                calls=calls
            )

    def get_latest_turn(self) -> Optional[TurnSummary]:
        with self._lock:
            if not self._history:
                return None
            latest_sid = self._history[-1].session_id
            return self.get_turn_summary(latest_sid)

    def reset_session(self, session_id: str):
        with self._lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]

    def reset_all(self):
        with self._lock:
            self._call_counter = 0
            self._history.clear()
            self._active_sessions.clear()

whisper_profiler = WhisperProfiler()
