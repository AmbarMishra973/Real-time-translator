"""
Real-Time Streaming Orchestrator (Phase 3).
Coordinates WebSocket audio streaming, incremental windowed STT buffering,
intermediate partials, final transcription, and instant cancellation.
"""

import os
import time
import uuid
import asyncio
from types import SimpleNamespace
from typing import Optional, Dict, Any, Tuple

from backend.core.logger import stream_log
from backend.services.stt_service import stt_service, STTService
from backend.services.rag_service import rag_service, RAGService
from backend.services.llm_service import llm_service, LLMService
from backend.services.session_manager import session_manager, SessionManager
from backend.services.vad_service import vad_service, VADService, VADSessionState
from backend.services.hypothesis_service import (
    hypothesis_service,
    HypothesisService,
    HypothesisSessionState,
    HypothesisResult
)

DEFAULT_MAX_BUFFER_BYTES = int(os.getenv("STREAM_MAX_BUFFER_BYTES", 5 * 1024 * 1024))  # 5 MB
DEFAULT_MIN_PARTIAL_BYTES = int(os.getenv("STREAM_MIN_PARTIAL_BYTES", 16000))
DEFAULT_PARTIAL_INTERVAL_BYTES = int(os.getenv("STREAM_PARTIAL_INTERVAL_BYTES", 24000))




class StreamingSession:
    """Encapsulates isolated state for an active streaming turn."""

    def __init__(
        self,
        session_id: str,
        request_id: str,
        language: str = "en",
        target_lang: str = "hi",
        domain: str = "all",
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        vad_enabled: Optional[bool] = None,
        hypothesis_enabled: Optional[bool] = None,
    ):
        self.session_id = session_id
        self.request_id = request_id
        self.language = language
        self.target_lang = target_lang
        self.domain = domain
        self.max_buffer_bytes = max_buffer_bytes

        # VAD & Endpointing configuration (defaults to VAD_ENABLED env var)
        self.vad_enabled = vad_enabled if vad_enabled is not None else (os.getenv("VAD_ENABLED", "false").lower() == "true")
        self.vad_state: VADSessionState = vad_service.create_session_state(session_id=self.session_id)

        # Hypothesis Stabilization configuration (defaults to HYPOTHESIS_STABILIZATION_ENABLED env var)
        self.hypothesis_enabled = (
            hypothesis_enabled if hypothesis_enabled is not None
            else (os.getenv("HYPOTHESIS_STABILIZATION_ENABLED", "true").strip().lower() in ("true", "1"))
        )
        self.hypothesis_state: HypothesisSessionState = hypothesis_service.create_session_state(session_id=self.session_id)
        self.last_stabilized_result: Optional[HypothesisResult] = None

        self.buffer = bytearray()
        self.first_audio_ts: Optional[float] = None
        self.first_result_ts: Optional[float] = None
        self.start_ts = time.perf_counter()

        self.is_cancelled = False
        self.is_evaluating_partial = False
        self.last_partial_text = ""
        self.last_partial_bytes_len = 0

    def add_chunk(self, chunk: bytes) -> bool:
        """
        Appends incoming audio chunk to buffer with protection against unbounded growth.
        Updates session VAD tracking when enabled.
        Returns True if chunk accepted, raises ValueError if buffer limit exceeded.
        """
        if self.is_cancelled:
            return False

        if len(self.buffer) + len(chunk) > self.max_buffer_bytes:
            raise ValueError(f"Stream buffer limit ({self.max_buffer_bytes} bytes) exceeded.")

        if self.first_audio_ts is None and len(chunk) > 0:
            self.first_audio_ts = time.perf_counter()

        self.buffer.extend(chunk)

        if self.vad_enabled and vad_service.is_available and len(chunk) > 0:
            vad_service.process_streaming_chunk(chunk, self.vad_state)

        return True

    def should_trigger_partial(self) -> bool:
        """
        Determines if enough incremental audio has arrived to evaluate a partial transcript.
        When VAD is enabled, avoids scheduling expensive Whisper inference on silence.
        """
        if self.is_cancelled or self.is_evaluating_partial:
            return False

        cur_len = len(self.buffer)
        if cur_len < DEFAULT_MIN_PARTIAL_BYTES:
            return False

        if (cur_len - self.last_partial_bytes_len) < DEFAULT_PARTIAL_INTERVAL_BYTES:
            return False

        # When VAD is active, skip partial inference if buffer contains no speech
        if self.vad_enabled and vad_service.is_available:
            if not self.vad_state.is_speech_active:
                snapshot = bytes(self.buffer)
                if not vad_service.has_speech(snapshot):
                    return False
                self.vad_state.is_speech_active = True

        # Atomically reserve in-flight execution to prevent concurrent duplicate partial dispatch
        self.is_evaluating_partial = True
        return True

    def reset_for_next_turn(self, new_request_id: Optional[str] = None) -> None:
        """Resets streaming buffers, VAD tracking, and hypothesis stabilization state while keeping session metadata intact."""
        self.request_id = new_request_id or uuid.uuid4().hex[:10]
        self.buffer.clear()
        self.first_audio_ts = None
        self.first_result_ts = None
        self.start_ts = time.perf_counter()
        self.is_cancelled = False
        self.is_evaluating_partial = False
        self.last_partial_text = ""
        self.last_partial_bytes_len = 0
        self.vad_state.reset_for_next_turn()
        self.hypothesis_state.reset_for_next_turn()
        self.last_stabilized_result = None


class StreamingOrchestrator:
    """
    Coordinates streaming audio lifecycle:
    Chunk Buffering ➔ Windowed Partial STT ➔ Final Pipeline (STT ➔ RAG ➔ LLM) ➔ Turn Persistence
    """

    def __init__(
        self,
        stt: STTService = stt_service,
        rag: RAGService = rag_service,
        llm: LLMService = llm_service,
        sessions: SessionManager = session_manager
    ):
        self.stt = stt
        self.rag = rag
        self.llm = llm
        self.sessions = sessions

    def create_session(
        self,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        language: str = "en",
        target_lang: str = "hi",
        domain: str = "all",
        vad_enabled: Optional[bool] = None,
        hypothesis_enabled: Optional[bool] = None,
    ) -> StreamingSession:
        sid = session_id or f"stream_{uuid.uuid4().hex[:8]}"
        rid = request_id or uuid.uuid4().hex[:10]
        session = StreamingSession(
            session_id=sid,
            request_id=rid,
            language=language,
            target_lang=target_lang,
            domain=domain,
            vad_enabled=vad_enabled,
            hypothesis_enabled=hypothesis_enabled,
        )
        stream_log("stream_started", rid, session_id=sid, language=language, target_lang=target_lang)
        return session

    async def evaluate_partial(self, session: StreamingSession) -> Optional[str]:
        """
        Executes non-blocking incremental transcription on currently accumulated audio.
        Applies hypothesis stabilization / local agreement when enabled.
        Returns text if new/updated intermediate tokens found, else None.
        """
        if session.is_cancelled or len(session.buffer) < DEFAULT_MIN_PARTIAL_BYTES:
            return None

        session.is_evaluating_partial = True
        snapshot = bytes(session.buffer)
        session.last_partial_bytes_len = len(snapshot)

        t_start = time.perf_counter()
        try:
            partial_text = await asyncio.to_thread(
                self.stt.transcribe_partial,
                audio_bytes=snapshot,
                lang=session.language,
                session_id=session.session_id,
                request_id=session.request_id,
                reason="partial_stream"
            )
        except Exception as e:
            partial_text = ""
        finally:
            session.is_evaluating_partial = False

        if session.is_cancelled:
            return None

        clean_text = (partial_text or "").strip()
        if clean_text and clean_text != session.last_partial_text:
            if session.first_result_ts is None:
                session.first_result_ts = time.perf_counter()

            session.last_partial_text = clean_text

            if session.hypothesis_enabled:
                stab_res = hypothesis_service.process_hypothesis(clean_text, session.hypothesis_state)
                session.last_stabilized_result = stab_res

            ttfr_ms = round((session.first_result_ts - (session.first_audio_ts or session.start_ts)) * 1000, 1)
            stream_log(
                "partial_transcription",
                session.request_id,
                partial=clean_text,
                stable_prefix=session.last_stabilized_result.stable_text if session.last_stabilized_result else "",
                ttfr_ms=ttfr_ms,
                latency_ms=round((time.perf_counter() - t_start) * 1000, 1)
            )
            return clean_text

        return None

    async def finalize_stream(self, session: StreamingSession) -> Dict[str, Any]:
        """
        Executes authoritative final transcription (with Phase 1 signal conditioning & gating)
        followed by RAG retrieval and LLM translation.
        """
        if session.is_cancelled:
            return {"type": "cancelled", "request_id": session.request_id}

        t_final_start = time.perf_counter()
        audio_snapshot = bytes(session.buffer)

        stream_log("final_transcription_started", session.request_id, buffer_bytes=len(audio_snapshot))

        # 1. Authoritative STT with Phase 8 Smart Finalization
        buffer_len = len(session.buffer)
        enable_smart_finalize = os.getenv("STREAM_SMART_FINALIZE", "true").lower() in ("true", "1")
        can_fast_path = (
            enable_smart_finalize
            and session.last_partial_bytes_len >= (buffer_len * 0.90)
            and bool(session.last_partial_text)
            and any(c.isalnum() for c in session.last_partial_text)
        )

        t_stt_start = time.perf_counter()
        if can_fast_path:
            transcript = session.last_partial_text
            engine_used = "local"
            model_used = self.stt.model_size
            info = SimpleNamespace(language=session.language, language_probability=1.0, language_source="streaming_partial")
            diagnostics = {"is_silent_gate": False, "gate_reason": None, "fast_path": True}
            t_stt_s = time.perf_counter() - t_stt_start
            stream_log(
                "final_transcription_fast_path",
                session.request_id,
                transcript=transcript,
                coverage_pct=round(session.last_partial_bytes_len / max(1, buffer_len) * 100, 1),
                stt_ms=round(t_stt_s * 1000, 1)
            )
        else:
            try:
                transcript, info, diagnostics, engine_used, model_used = await asyncio.to_thread(
                    self.stt.transcribe,
                    audio_bytes=audio_snapshot,
                    lang=session.language,
                    filename="stream_recording.wav",
                    content_type="audio/wav",
                    session_id=session.session_id,
                    request_id=session.request_id,
                    reason="final_stream"
                )
            except Exception as exc:
                stream_log("final_stt_error", session.request_id, error=str(exc))
                return {
                    "type": "error",
                    "request_id": session.request_id,
                    "message": f"Transcription error: {str(exc)}"
                }
            t_stt_s = time.perf_counter() - t_stt_start
        if session.first_result_ts is None and transcript:
            session.first_result_ts = time.perf_counter()

        first_ref = session.first_audio_ts or session.start_ts
        ttfr_ms = round(((session.first_result_ts or time.perf_counter()) - first_ref) * 1000, 1)

        # Silence Gate / Empty short-circuit: return without calling RAG/LLM
        if diagnostics.get("is_silent_gate") or not transcript or not any(c.isalnum() for c in transcript):
            gate_msg = diagnostics.get("gate_message") or "No speech detected in audio."
            stream_log("stream_short_circuited", session.request_id, reason=diagnostics.get("gate_reason") or "empty")
            return {
                "type": "final",
                "request_id": session.request_id,
                "session_id": session.session_id,
                "transcript": "",
                "translated_text": "",
                "translated": "",
                "retrieved_context": [],
                "sources_used": [],
                "provider": "none",
                "history": self.sessions.get_history(session.session_id),
                "metrics": {
                    "ttfr_ms": ttfr_ms,
                    "stt_ms": round(t_stt_s * 1000, 1),
                    "rag_ms": 0.0,
                    "llm_ms": 0.0,
                    "total_ms": round((time.perf_counter() - t_final_start) * 1000, 1),
                },
                "gate_reason": diagnostics.get("gate_reason"),
                "message": gate_msg,
                "suspected_hallucination": False,
                "ai_observability": {
                    "retrieval_attempted": False,
                    "retrieval_hit": False,
                    "retrieved_count": 0,
                    "top_retrieval_score": 0.0,
                    "context_used": False,
                    "llm_provider": "none",
                    "fallback_used": False,
                    "fallback_reason": None,
                    "source_language": session.language,
                    "target_language": session.target_lang
                }
            }

        # 2. RAG Retrieval
        t_rag_start = time.perf_counter()
        rag_res = self.rag.retrieve(query=transcript, domain=session.domain)
        t_rag_s = time.perf_counter() - t_rag_start
        rag_chunks = rag_res.get("chunks", []) if isinstance(rag_res, dict) else rag_res
        retrieval_hit = len(rag_chunks) > 0
        retrieved_count = len(rag_chunks)
        top_score = rag_chunks[0].get("similarity", 0.0) if retrieval_hit else 0.0

        # 3. LLM Translation
        t_llm_start = time.perf_counter()
        def run_translation():
            return self.llm.translate(
                text=transcript,
                source_lang=session.language,
                target_lang=session.target_lang,
                session_id=session.session_id,
                domain=session.domain
            )

        trans_result = await asyncio.to_thread(run_translation)
        t_llm_s = time.perf_counter() - t_llm_start
        total_s = time.perf_counter() - t_final_start

        fallback_used = trans_result.get("fallback_used", False)
        fallback_reason = trans_result.get("fallback_reason")
        context_used = trans_result.get("context_used", retrieval_hit)

        # Persist conversation turn
        self.sessions.add_turn(
            session_id=session.session_id,
            source_text=transcript,
            translated_text=trans_result.get("translated_text", ""),
            source_lang=session.language,
            target_lang=session.target_lang,
            metadata={
                "request_id": session.request_id,
                "mode": "streaming",
                "context_used": context_used,
                "fallback_used": fallback_used
            }
        )

        stream_log(
            "stream_completed",
            session.request_id,
            transcript=transcript,
            ttfr_ms=ttfr_ms,
            stt_ms=round(t_stt_s * 1000, 1),
            rag_ms=round(t_rag_s * 1000, 1),
            llm_ms=round(t_llm_s * 1000, 1),
            total_ms=round(total_s * 1000, 1),
            retrieval_hit=retrieval_hit,
            retrieved_count=retrieved_count,
            top_score=top_score,
            context_used=context_used,
            provider=trans_result.get("provider"),
            fallback_used=fallback_used
        )

        hypo_reconciliation = None
        if session.hypothesis_enabled:
            hypo_reconciliation = hypothesis_service.reconcile_final(transcript, session.hypothesis_state)

        return {
            "type": "final",
            "request_id": session.request_id,
            "session_id": session.session_id,
            "transcript": transcript,
            "translated_text": trans_result["translated_text"],
            "translated": trans_result["translated_text"],
            "retrieved_context": trans_result["retrieved_context"],
            "sources_used": trans_result.get("sources_used", []),
            "provider": trans_result["provider"],
            "history": trans_result["history"],
            "metrics": {
                "ttfr_ms": ttfr_ms,
                "stt_ms": round(t_stt_s * 1000, 1),
                "rag_ms": round(t_rag_s * 1000, 1),
                "llm_ms": round(t_llm_s * 1000, 1),
                "total_ms": round(total_s * 1000, 1),
            },
            "ai_observability": {
                "retrieval_attempted": True,
                "retrieval_hit": retrieval_hit,
                "retrieved_count": retrieved_count,
                "top_retrieval_score": top_score,
                "context_used": context_used,
                "llm_provider": trans_result.get("provider", "none"),
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "source_language": session.language,
                "target_language": session.target_lang
            },
            "stt_engine": engine_used,
            "stt_model": model_used,
            "gate_reason": None,
            "suspected_hallucination": diagnostics.get("suspected_hallucination", False),
            "hypothesis_reconciliation": hypo_reconciliation,
        }

    def cancel_stream(self, session: StreamingSession) -> Dict[str, Any]:
        """Instant cancellation of active turn and buffer release."""
        session.is_cancelled = True
        session.buffer.clear()
        stream_log("stream_cancelled", session.request_id, session_id=session.session_id)
        return {
            "type": "cancelled",
            "request_id": session.request_id,
            "session_id": session.session_id
        }


# Process-level singleton instance
streaming_orchestrator = StreamingOrchestrator()
