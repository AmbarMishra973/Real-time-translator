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


# === Mount Pipeline Routes ===
app.include_router(pipeline_router)


# === Auxiliary Audio Format Helper for WebSockets ===

def pcm16_to_wav_bytes(pcm16_bytes: bytes, rate: int = 16000) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16_bytes)
    return buf.getvalue()


# === WebSocket Live Transcription (Deferred to Phase 3) ===

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    if os.getenv("ENABLE_EXPERIMENTAL_WEBSOCKET", "false").lower() != "true":
        await websocket.close(code=1008, reason="Experimental streaming STT is disabled; use /pipeline.")
        return
    if stt_service.model is None:
        await websocket.close(code=1011, reason="No local Whisper model is available.")
        return
    await websocket.accept()
    buffer = bytearray()
    print("\n[+] Microphone connected via WebSocket.", flush=True)

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)

            # Transcribe when ~1 sec of 16kHz 16-bit mono audio is accumulated
            if len(buffer) >= 32000:
                chunk = buffer[:32000]
                buffer = buffer[32000:]
                wav_data = pcm16_to_wav_bytes(chunk)

                def transcribe_ws_chunk(wav_bytes):
                    segments, info = stt_service.model.transcribe(
                        io.BytesIO(wav_bytes),
                        beam_size=1,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        language=None
                    )
                    return ' '.join([s.text for s in segments]).strip(), info

                text, info = await asyncio.to_thread(transcribe_ws_chunk, wav_data)

                if text:
                    print(f"-> Whisper heard: '{text}'", flush=True)
                    retrieved = rag_service.retrieve(text, top_k=2)
                    await websocket.send_json({
                        "text": text,
                        "detected_lang": info.language,
                        "confidence": round(info.language_probability, 2),
                        "retrieved_context": retrieved
                    })

    except WebSocketDisconnect:
        print("[-] WebSocket client disconnected.", flush=True)
    except Exception as e:
        print(f"[!] WebSocket Error: {str(e)}", flush=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
