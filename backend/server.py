"""
FastAPI Modular Application Server for Real-Time AI Translator + RAG.
Coordinates application lifecycle, middleware, error handling, and route registration.
"""

import os
import io
import sys
import asyncio
from typing import Optional

# Ensure safe console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.pipeline_routes import router as pipeline_router
from backend.routes.websocket_routes import router as websocket_router
from backend.services.stt_service import stt_service, convert_to_clean_wav, parse_capture_metadata
from backend.services.rag_service import rag_service
from backend.services.llm_service import llm_service
from backend.services.tts_service import tts_service, pick_voice, VOICE_MAP
from backend.services.session_manager import session_manager
from backend.core.logger import stt_log, pipeline_log
from backend.core.orchestrator import pipeline_orchestrator

# Backward compatibility exports for existing test harnesses & investigations
whisper_model = stt_service.model
WHISPER_SIZE = stt_service.model_size
sync_transcribe = stt_service.transcribe

app = FastAPI(
    title="Real-Time AI Translator + RAG API",
    description="End-to-End Speech-to-Speech Translation powered by Whisper, RAG, LLM, and Edge-TTS.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*"
        }
    )


# === Root and Diagnostic Endpoints ===

@app.get("/health", status_code=200)
@app.head("/health", status_code=200)
def health_check():
    return {"status": "ok", "service": "Real-Time AI Translator + RAG"}


@app.get("/", status_code=200)
@app.head("/", status_code=200)
def root():
    return {
        "service": "Real-Time AI Translator + RAG",
        "version": "2.1.0",
        "status": "online",
        "llm_status": llm_service.get_status(),
        "stt_status": {
            "local_model_loaded": stt_service.model is not None,
            "requested_model": stt_service.model_size
        },
        "rag_domains": rag_service.get_domains(),
        "total_knowledge_terms": len(rag_service.chunks)
    }


# === Mount Application Routers ===
app.include_router(pipeline_router)
app.include_router(websocket_router)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
