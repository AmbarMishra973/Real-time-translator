"""
Pipeline Orchestrator Module.
Coordinates STT, RAG, LLM, TTS, and Session Management with stage timing and request tracing.
"""

import os
import uuid
import time
import asyncio
from typing import Optional, Dict, Any

from fastapi import HTTPException

from backend.core.logger import pipeline_log
from backend.services.stt_service import stt_service, STTService
from backend.services.rag_service import rag_service, RAGService
from backend.services.llm_service import llm_service, LLMService
from backend.services.tts_service import tts_service, TTSService
from backend.services.session_manager import session_manager, SessionManager


class PipelineOrchestrator:
    """
    Coordinates multi-stage pipeline flow:
    Audio Input ➔ Whisper STT ➔ RAG Retrieval ➔ LLM Translation ➔ Session Persistence
    """

    def __init__(
        self,
        stt: STTService = stt_service,
        rag: RAGService = rag_service,
        llm: LLMService = llm_service,
        tts: TTSService = tts_service,
        sessions: SessionManager = session_manager
    ):
        self.stt = stt
        self.rag = rag
        self.llm = llm
        self.tts = tts
        self.sessions = sessions

    async def process_full_pipeline(
        self,
        audio_bytes: bytes,
        source_lang: str = "en",
        target_lang: str = "hi",
        session_id: str = "default",
        domain: str = "all",
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        capture_metadata: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end speech-to-speech translation pipeline with stage timing
        and short-circuit silence gating.
        """
        request_id = uuid.uuid4().hex[:10]
        t0 = time.perf_counter()

        pipeline_log(
            "pipeline_started",
            request_id,
            session_id=session_id,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            audio_bytes=len(audio_bytes)
        )

        # 1. STT Stage
        t_stt_start = time.perf_counter()
        try:
            transcript, info, diagnostics, engine_used, model_used = await asyncio.to_thread(
                self.stt.transcribe,
                audio_bytes=audio_bytes,
                lang=source_lang,
                filename=filename,
                content_type=content_type,
                capture_metadata=capture_metadata,
                session_id=session_id,
                request_id=request_id,
                reason="rest_pipeline"
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        t_stt = time.perf_counter() - t_stt_start
        pipeline_log("stt_completed", request_id, stt_ms=round(t_stt * 1000, 1), engine=engine_used, model=model_used)

        # Fast Short-Circuit on Silence Gate or Empty Spoken Content
        if diagnostics.get("is_silent_gate") or not transcript or not any(c.isalnum() for c in transcript):
            message = diagnostics.get("gate_message") or "No speech detected in audio."
            pipeline_log("pipeline_short_circuited", request_id, reason=diagnostics.get("gate_reason") or "empty_transcript")
            return {
                "request_id": request_id,
                "transcript": "",
                "translated_text": "",
                "translated": "",
                "retrieved_context": [],
                "sources_used": [],
                "history": self.sessions.get_history(session_id) or self.llm.conversation_manager.get_history(session_id),
                "metrics": {
                    "stt_s": round(t_stt, 2),
                    "rag_s": 0.0,
                    "llm_s": 0.0,
                    "total_s": round(t_stt, 2),
                    "stt_ms": round(t_stt * 1000, 1),
                    "rag_ms": 0.0,
                    "llm_ms": 0.0,
                    "total_ms": round(t_stt * 1000, 1)
                },
                "ai_observability": {
                    "retrieval_attempted": False,
                    "retrieval_hit": False,
                    "retrieved_count": 0,
                    "top_retrieval_score": 0.0,
                    "context_used": False,
                    "llm_provider": "none",
                    "fallback_used": False,
                    "fallback_reason": None,
                    "source_language": source_lang,
                    "target_language": target_lang
                },
                "message": message,
                "gate_reason": diagnostics.get("gate_reason"),
                "suspected_hallucination": False,
                "audio": diagnostics if os.getenv("STT_DEBUG", "false").lower() == "true" else None,
            }

        # 2. RAG Retrieval Stage
        t_rag_start = time.perf_counter()
        rag_res = self.rag.retrieve(query=transcript, domain=domain)
        t_rag = time.perf_counter() - t_rag_start
        rag_chunks = rag_res.get("chunks", []) if isinstance(rag_res, dict) else rag_res
        retrieval_hit = len(rag_chunks) > 0
        retrieved_count = len(rag_chunks)
        top_score = rag_chunks[0].get("similarity", 0.0) if retrieval_hit else 0.0

        pipeline_log(
            "rag_completed",
            request_id,
            rag_ms=round(t_rag * 1000, 1),
            retrieval_hit=retrieval_hit,
            retrieved_count=retrieved_count,
            top_score=top_score
        )

        # 3. LLM Translation Stage
        t_llm_start = time.perf_counter()
        def run_translation():
            return self.llm.translate(
                text=transcript,
                source_lang=source_lang,
                target_lang=target_lang,
                session_id=session_id,
                domain=domain
            )

        trans_result = await asyncio.to_thread(run_translation)
        t_llm = time.perf_counter() - t_llm_start
        t_total = time.perf_counter() - t0

        fallback_used = trans_result.get("fallback_used", False)
        fallback_reason = trans_result.get("fallback_reason")
        context_used = trans_result.get("context_used", retrieval_hit)

        # Persist to SessionManager
        self.sessions.add_turn(
            session_id=session_id,
            source_text=transcript,
            translated_text=trans_result.get("translated_text", ""),
            source_lang=source_lang,
            target_lang=target_lang,
            metadata={
                "request_id": request_id,
                "provider": trans_result.get("provider"),
                "context_used": context_used,
                "fallback_used": fallback_used
            }
        )

        pipeline_log(
            "pipeline_completed",
            request_id,
            total_ms=round(t_total * 1000, 1),
            llm_ms=round(t_llm * 1000, 1),
            provider=trans_result.get("provider"),
            retrieval_hit=retrieval_hit,
            retrieved_count=retrieved_count,
            top_score=top_score,
            context_used=context_used,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason
        )

        return {
            "request_id": request_id,
            "transcript": transcript,
            "detected_lang": getattr(info, "language", None) or source_lang,
            "language_source": getattr(info, "language_source", "detected"),
            "translated_text": trans_result["translated_text"],
            "translated": trans_result["translated_text"],
            "retrieved_context": trans_result["retrieved_context"],
            "sources_used": trans_result.get("sources_used", []),
            "provider": trans_result["provider"],
            "history": trans_result["history"],
            "metrics": {
                "stt_s": round(t_stt, 2),
                "rag_s": round(t_rag, 3),
                "llm_s": round(t_llm, 2),
                "total_s": round(t_total, 2),
                "stt_ms": round(t_stt * 1000, 1),
                "rag_ms": round(t_rag * 1000, 1),
                "llm_ms": round(t_llm * 1000, 1),
                "total_ms": round(t_total * 1000, 1)
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
                "source_language": source_lang,
                "target_language": target_lang
            },
            "stt_engine": engine_used,
            "stt_model": model_used,
            "suspected_hallucination": diagnostics.get("suspected_hallucination", False),
            "gate_reason": None,
            "audio": diagnostics if os.getenv("STT_DEBUG", "false").lower() == "true" else None,
        }

    async def process_text_translation(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "hi",
        session_id: str = "default",
        domain: str = "all"
    ) -> Dict[str, Any]:
        """
        Executes RAG retrieval + LLM translation for raw text requests.
        """
        request_id = uuid.uuid4().hex[:10]
        t0 = time.perf_counter()

        pipeline_log("text_translation_started", request_id, text_len=len(text), domain=domain)

        # 1. RAG Retrieval
        t_rag_start = time.perf_counter()
        rag_res = self.rag.retrieve(query=text, domain=domain)
        t_rag = time.perf_counter() - t_rag_start

        # 2. LLM Translation
        t_llm_start = time.perf_counter()
        def run_translation():
            return self.llm.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                session_id=session_id,
                domain=domain
            )

        result = await asyncio.to_thread(run_translation)
        t_llm = time.perf_counter() - t_llm_start
        t_total = time.perf_counter() - t0

        rag_chunks = rag_res.get("chunks", []) if isinstance(rag_res, dict) else rag_res
        retrieval_hit = len(rag_chunks) > 0
        retrieved_count = len(rag_chunks)
        top_score = rag_chunks[0].get("similarity", 0.0) if retrieval_hit else 0.0
        fallback_used = result.get("fallback_used", False)
        fallback_reason = result.get("fallback_reason")
        context_used = result.get("context_used", retrieval_hit)

        # Persist turn
        self.sessions.add_turn(
            session_id=session_id,
            source_text=text,
            translated_text=result.get("translated_text", ""),
            source_lang=source_lang,
            target_lang=target_lang,
            metadata={
                "request_id": request_id,
                "provider": result.get("provider"),
                "context_used": context_used,
                "fallback_used": fallback_used
            }
        )

        result["request_id"] = request_id
        result["translated"] = result["translated_text"]
        result["metrics"] = {
            "stt_s": 0.0,
            "rag_s": round(t_rag, 3),
            "llm_s": round(t_llm, 2),
            "total_s": round(t_total, 2),
            "stt_ms": 0.0,
            "rag_ms": round(t_rag * 1000, 1),
            "llm_ms": round(t_llm * 1000, 1),
            "total_ms": round(t_total * 1000, 1)
        }
        result["ai_observability"] = {
            "retrieval_attempted": True,
            "retrieval_hit": retrieval_hit,
            "retrieved_count": retrieved_count,
            "top_retrieval_score": top_score,
            "context_used": context_used,
            "llm_provider": result.get("provider", "none"),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "source_language": source_lang,
            "target_language": target_lang
        }
        pipeline_log(
            "text_translation_completed",
            request_id,
            total_ms=round(t_total * 1000, 1),
            retrieval_hit=retrieval_hit,
            retrieved_count=retrieved_count,
            top_score=top_score,
            context_used=context_used,
            fallback_used=fallback_used
        )
        return result


# Process-level orchestrator instance
pipeline_orchestrator = PipelineOrchestrator()
