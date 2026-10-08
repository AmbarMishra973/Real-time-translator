"""
HTTP Pipeline Routes.
Defines endpoints for STT, Translation, TTS, Pipeline orchestration, RAG management, and Session history.
"""

import io
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.stt_service import stt_service, parse_capture_metadata
from backend.services.tts_service import tts_service
from backend.services.rag_service import rag_service
from backend.services.llm_service import llm_service
from backend.services.session_manager import session_manager
from backend.core.orchestrator import pipeline_orchestrator

router = APIRouter()


class AddTermRequest(BaseModel):
    term: str
    definition: str
    domain: str = "custom"


class UpdateSettingsRequest(BaseModel):
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    capture_metadata: Optional[str] = Form(None),
):
    """
    Clean STT Diagnostic Endpoint:
    Receives audio, inspects, boosts, gates, and transcribes without LLM, RAG, or TTS.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    try:
        text, info, diagnostics, engine_used, model_used = await asyncio.to_thread(
            stt_service.transcribe,
            audio_bytes=audio_bytes,
            lang=lang,
            filename=file.filename,
            content_type=file.content_type,
            capture_metadata=parse_capture_metadata(capture_metadata)
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "mime_type": file.content_type,
        "bytes": len(audio_bytes),
        "duration_seconds": diagnostics.get("duration_s"),
        "sample_rate": diagnostics.get("sample_rate_hz"),
        "channels": diagnostics.get("channels"),
        "rms_dbfs": diagnostics.get("rms_dbfs"),
        "peak_dbfs": diagnostics.get("peak_dbfs"),
        "pre_rms_dbfs": diagnostics.get("pre_rms_dbfs"),
        "gain_db_applied": diagnostics.get("gain_db_applied"),
        "whisper_model": model_used,
        "transcript": text,
        "message": diagnostics.get("gate_message", ""),
        "gate_reason": diagnostics.get("gate_reason"),
        "suspected_hallucination": diagnostics.get("suspected_hallucination", False),
    }


@router.post("/translate")
async def translate_text(
    text: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("hi"),
    session_id: str = Form("default"),
    domain: str = Form("all"),
    capture_metadata: Optional[str] = Form(None),
):
    """
    RAG-Augmented LLM Translation for text inputs.
    """
    return await pipeline_orchestrator.process_text_translation(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        session_id=session_id,
        domain=domain
    )


@router.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: Optional[str] = Form(None),
    target_lang: Optional[str] = Form(None)
):
    """
    Synthesizes text to speech using Edge-TTS neural voice.
    """
    audio_bytes, latency_s = await tts_service.synthesize(
        text=text,
        voice=voice,
        target_lang=target_lang
    )
    headers = {
        "X-TTS-Latency": str(latency_s),
        "Access-Control-Expose-Headers": "X-TTS-Latency"
    }
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg", headers=headers)


@router.post("/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("hi"),
    session_id: str = Form("default"),
    domain: str = Form("all"),
    capture_metadata: Optional[str] = Form(None)
):
    """
    Full End-to-End Pipeline:
    Audio Input ➔ Whisper STT ➔ RAG Retrieval ➔ LLM Translation ➔ Session History
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    return await pipeline_orchestrator.process_full_pipeline(
        audio_bytes=audio_bytes,
        source_lang=source_lang,
        target_lang=target_lang,
        session_id=session_id,
        domain=domain,
        filename=file.filename,
        content_type=file.content_type,
        capture_metadata=parse_capture_metadata(capture_metadata)
    )


# === Knowledge Base Management Endpoints ===

@router.get("/api/knowledge")
def get_knowledge_base():
    """Returns all loaded knowledge items and domain categories."""
    return {
        "domains": rag_service.get_domains(),
        "total": len(rag_service.chunks),
        "terms": rag_service.get_all_terms()
    }


@router.post("/api/knowledge")
def add_custom_term(payload: AddTermRequest):
    """Dynamically adds a custom term to the vector knowledge base."""
    if not payload.term.strip() or not payload.definition.strip():
        raise HTTPException(status_code=400, detail="Term and definition cannot be empty.")

    chunk = rag_service.add_custom_term(
        term=payload.term,
        definition=payload.definition,
        domain=payload.domain
    )
    return {"message": f"Term '{chunk.term}' added and indexed successfully.", "term": chunk.to_dict()}


# === Settings & Configuration Endpoints ===

@router.get("/api/settings")
def get_settings():
    """Returns LLM status and available domains."""
    return {
        "llm_status": llm_service.get_status(),
        "domains": ["all"] + rag_service.get_domains(),
        "available_languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi"},
            {"code": "zh", "name": "Chinese"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ru", "name": "Russian"},
            {"code": "ar", "name": "Arabic"},
        ]
    }


@router.post("/api/settings")
def update_settings(payload: UpdateSettingsRequest):
    """Updates API keys dynamically in memory."""
    llm_service.update_keys(
        groq_key=payload.groq_api_key,
        openai_key=payload.openai_api_key
    )
    return {"message": "Settings updated.", "status": llm_service.get_status()}


# === Multi-Turn Session History Endpoints ===

@router.get("/api/history")
def get_history(session_id: str = "default"):
    """Fetches conversation history for a session."""
    history = session_manager.get_history(session_id) or llm_service.conversation_manager.get_history(session_id)
    return {"session_id": session_id, "history": history}


@router.delete("/api/history")
def clear_history(session_id: str = "default"):
    """Resets conversation history for a session."""
    session_manager.clear_history(session_id)
    llm_service.conversation_manager.clear_history(session_id)
    return {"message": f"History for session '{session_id}' cleared."}


# === Whisper Profiling Diagnostic Endpoint (Phase 8) ===

@router.get("/api/stt/profile")
def get_stt_profiler_summary(session_id: Optional[str] = None):
    """
    Returns structured Whisper call profiling metrics:
    - Number of Whisper invocations
    - Total audio duration processed vs speech duration (amplification)
    - Cumulative inference time and per-call breakdown
    """
    from backend.services.whisper_profiler import whisper_profiler
    if session_id:
        summary = whisper_profiler.get_turn_summary(session_id)
    else:
        summary = whisper_profiler.get_latest_turn()

    if not summary:
        return {"status": "no_data", "calls": [], "call_count": 0}
    return summary
