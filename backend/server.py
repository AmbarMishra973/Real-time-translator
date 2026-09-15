"""
FastAPI Backend Server for Real-Time AI Translator + RAG.
Orchestrates:
  1. Audio Transcription (Faster-Whisper)
  2. Semantic Domain Knowledge Retrieval (RAG Engine)
  3. Context-Aware Translation with Conversation History (LLM Translator)
  4. Neural Speech Synthesis (Edge-TTS)
  5. WebSocket real-time audio stream handling
"""

import os
import io
import sys
import asyncio

# Ensure safe console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import edge_tts

from backend.rag_engine import rag_engine
from backend.llm_translator import llm_translator

app = FastAPI(
    title="Real-Time AI Translator + RAG API",
    description="End-to-End Speech-to-Speech Translation powered by Whisper, RAG, LLM, and Edge-TTS.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialize Whisper Model ===
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
print(f"[*] Initializing Faster-Whisper ({WHISPER_SIZE} on {WHISPER_DEVICE})...")
try:
    whisper_model = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE, compute_type="int8")
    print("[+] Whisper model loaded successfully.")
except Exception as e:
    print(f"[!] Warning: Could not load {WHISPER_SIZE} model ({e}). Attempting 'base' fallback...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# High-quality neural voices mapping
VOICE_MAP = {
    'en': 'en-US-JennyNeural',
    'hi': 'hi-IN-SwaraNeural',
    'zh': 'zh-CN-XiaoxiaoNeural',
    'es': 'es-ES-ElviraNeural',
    'fr': 'fr-FR-DeniseNeural',
    'de': 'de-DE-KatjaNeural',
    'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-SunHiNeural',
    'ru': 'ru-RU-SvetlanaNeural',
    'ar': 'ar-SA-ZariyahNeural',
}


def pick_voice(lang_code: str) -> str:
    base = (lang_code or 'en').split('-')[0].lower()
    return VOICE_MAP.get(base, 'en-US-JennyNeural')


def pcm16_to_wav_bytes(pcm16_bytes: bytes, rate: int = 16000) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16_bytes)
    return buf.getvalue()


def convert_to_clean_wav(audio_bytes: bytes) -> bytes:
    """
    Converts and resamples incoming browser audio (WebM, OGG, MP4, WAV, etc.) into clean 16kHz mono WAV PCM.
    Applies intelligent dynamic audio normalization (dynaudnorm) to boost soft voices WITHOUT clipping distortion.
    """
    import subprocess, tempfile, os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".input_audio") as in_f:
        in_f.write(audio_bytes)
        in_path = in_f.name

    out_path = in_path + ".wav"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-err_detect", "ignore_err",
            "-i", in_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            "-af", "dynaudnorm=p=0.9:s=5",
            out_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                converted = f.read()
            if len(converted) > 100:
                return converted
    except Exception as e:
        print(f"[!] Warning: Audio dynaudnorm conversion error: {e}, retrying without filter...")
        try:
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
        except Exception:
            pass
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

    return audio_bytes


WHISPER_HALLUCINATIONS = {
    "", "you", "you.", "you!", "thank you", "thank you.", "thank you for watching",
    "thank you for watching.", "thanks for watching", "thanks for watching!",
    "...", ".", "..", "....", ". . . .", "bye", "bye.", "amara.org", "subscribe",
    "subtitles by", "subtitles by the amara.org community", "please subscribe",
    "so", "so.", "yeah", "yeah."
}


def sync_transcribe(audio_bytes: bytes, lang: Optional[str] = None):
    # 1. Convert & resample to clean 16kHz mono WAV PCM with dynamic normalization
    wav_bytes = convert_to_clean_wav(audio_bytes)
    whisper_lang = None if (not lang or lang.lower() == 'auto') else lang.split('-')[0].lower()
    print(f"[AUDIO_RECEIVED] size: {len(audio_bytes)} bytes -> Clean WAV: {len(wav_bytes)} bytes")
    print(f"[TRANSCRIPTION_STARTED] forced_lang: {whisper_lang or 'auto-detect'}")

    # 2. Fast-Path: Groq LPU Cloud Whisper (whisper-large-v3-turbo, ~180-250ms latency, 99% accuracy)
    if llm_translator._groq_client:
        try:
            groq_text = llm_translator.transcribe_with_groq(wav_bytes, whisper_lang)
            if groq_text:
                clean_text = groq_text.strip()
                has_alnum = any(c.isalnum() for c in clean_text)
                stripped = clean_text.lower().strip(" .!?,;:-\"'\n\r\t")
                if has_alnum and stripped not in WHISPER_HALLUCINATIONS:
                    print(f"[TRANSCRIPTION_COMPLETED] engine: Groq LPU (whisper-large-v3-turbo) transcript: \"{clean_text}\"")
                    from collections import namedtuple
                    PseudoInfo = namedtuple("PseudoInfo", ["language", "language_probability"])
                    return clean_text, PseudoInfo(language=whisper_lang or "en", language_probability=0.99)
        except Exception as e:
            print(f"[!] Groq Whisper fast-path failed: {e}, falling back to local Whisper...")

    # 3. Local Faster-Whisper Fallback (beam_size=2 for accurate word boundaries)
    try:
        segments, info = whisper_model.transcribe(
            io.BytesIO(wav_bytes),
            language=whisper_lang,
            beam_size=2,
            best_of=2,
            temperature=0.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=450, speech_pad_ms=250, threshold=0.35)
        )
        text_parts = [seg.text for seg in segments]
        raw_text = ' '.join(text_parts).strip()
    except Exception as e:
        print(f"[TRANSCRIPTION_FAILED] VAD transcribe error: {e}, attempting non-VAD fallback...")
        raw_text = ""
        info = None

    # Fallback to non-VAD mode if VAD filter suppressed quiet/short speech
    if not raw_text:
        try:
            segments, info = whisper_model.transcribe(
                io.BytesIO(wav_bytes),
                language=whisper_lang,
                beam_size=2,
                best_of=2,
                temperature=0.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                vad_filter=False
            )
            text_parts = [seg.text for seg in segments]
            raw_text = ' '.join(text_parts).strip()
        except Exception as e:
            print(f"[TRANSCRIPTION_FAILED] Non-VAD fallback error: {e}")
            raw_text = ""
            info = None

    # 4. Filter silence hallucinations & punctuation-only artifacts
    has_alphanumeric = any(c.isalnum() for c in raw_text)
    stripped_word = raw_text.lower().strip(" .!?,;:-\"'\n\r\t")

    if not has_alphanumeric or stripped_word in WHISPER_HALLUCINATIONS:
        print(f"[-] Discarded silence / punctuation hallucination: '{raw_text}'")
        clean_text = ""
    else:
        clean_text = raw_text.strip()

    detected_lang = info.language if info else (whisper_lang or "en")
    prob = info.language_probability if info else 1.0
    print(f"[TRANSCRIPTION_COMPLETED] engine: Local Whisper transcript: \"{clean_text}\" (lang: {detected_lang}, confidence: {prob:.2f})")
    return clean_text, info


# === API Endpoints ===

@app.get("/health", status_code=200)
@app.head("/health", status_code=200)
def health_check():
    return {"status": "ok", "service": "Real-Time AI Translator + RAG"}


@app.get("/", status_code=200)
@app.head("/", status_code=200)
def root():
    return {
        "service": "Real-Time AI Translator + RAG",
        "version": "2.0.1",
        "status": "online",
        "llm_status": llm_translator.get_status(),
        "rag_domains": rag_engine.get_domains(),
        "total_knowledge_terms": len(rag_engine.chunks)
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str = Form("en")
):
    """Whisper Speech-to-Text transcription."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    text, info = await asyncio.to_thread(sync_transcribe, audio_bytes, lang)
    return {
        "text": text,
        "detected_lang": info.language,
        "confidence": round(info.language_probability, 3)
    }


@app.post("/translate")
async def translate_text(
    text: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("hi"),
    session_id: str = Form("default"),
    domain: str = Form("all")
):
    """
    RAG-Augmented LLM Translation.
    1. Retrieves relevant domain knowledge
    2. Incorporates recent conversation history
    3. Prompts LLM for context-aware, term-preserving translation
    """
    import time
    t0 = time.perf_counter()

    # 1. RAG Retrieval timing
    t_rag_start = time.perf_counter()
    rag_res = rag_engine.retrieve(query=text, domain=domain)
    t_rag = time.perf_counter() - t_rag_start

    # 2 & 3. LLM Translation
    t_llm_start = time.perf_counter()
    def run_translation():
        return llm_translator.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            session_id=session_id,
            domain=domain
        )

    result = await asyncio.to_thread(run_translation)
    t_llm = time.perf_counter() - t_llm_start
    t_total = time.perf_counter() - t0

    result["translated"] = result["translated_text"]
    result["metrics"] = {
        "stt_s": 0.0,
        "rag_s": round(t_rag, 3),
        "llm_s": round(t_llm, 2),
        "total_s": round(t_total, 2)
    }
    return result


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: Optional[str] = Form(None),
    target_lang: Optional[str] = Form(None)
):
    """Synthesizes text to speech using Edge-TTS neural voice."""
    import time
    t0 = time.perf_counter()
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    selected_voice = voice or (pick_voice(target_lang) if target_lang else 'hi-IN-SwaraNeural')
    communicate = edge_tts.Communicate(text, selected_voice)
    out = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            out.write(chunk["data"])

    tts_duration = round(time.perf_counter() - t0, 2)
    headers = {"X-TTS-Latency": str(tts_duration), "Access-Control-Expose-Headers": "X-TTS-Latency"}
    return StreamingResponse(io.BytesIO(out.getvalue()), media_type="audio/mpeg", headers=headers)


@app.post("/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("hi"),
    session_id: str = Form("default"),
    domain: str = Form("all")
):
    """
    Full End-to-End Pipeline with real latency benchmarks:
    Audio Input ➔ Whisper STT ➔ RAG Retrieval ➔ LLM Translation ➔ Session History
    """
    import time
    t0 = time.perf_counter()

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # 1. STT Timing
    t_stt_start = time.perf_counter()
    transcript, info = await asyncio.to_thread(sync_transcribe, audio_bytes, source_lang)
    t_stt = time.perf_counter() - t_stt_start

    if not transcript or not any(c.isalnum() for c in transcript):
        return {
            "transcript": "",
            "translated_text": "",
            "translated": "",
            "retrieved_context": [],
            "sources_used": [],
            "history": llm_translator.conversation_manager.get_history(session_id),
            "metrics": {
                "stt_s": round(t_stt, 2),
                "rag_s": 0.0,
                "llm_s": 0.0,
                "total_s": round(t_stt, 2)
            },
            "message": "No speech detected in audio."
        }

    # 2. RAG Timing
    t_rag_start = time.perf_counter()
    rag_res = rag_engine.retrieve(query=transcript, domain=domain)
    t_rag = time.perf_counter() - t_rag_start

    # 3. LLM Translation Timing
    t_llm_start = time.perf_counter()
    def run_translation():
        return llm_translator.translate(
            text=transcript,
            source_lang=source_lang,
            target_lang=target_lang,
            session_id=session_id,
            domain=domain
        )

    trans_result = await asyncio.to_thread(run_translation)
    t_llm = time.perf_counter() - t_llm_start
    t_total = time.perf_counter() - t0

    return {
        "transcript": transcript,
        "detected_lang": info.language,
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
            "total_s": round(t_total, 2)
        }
    }


# === Knowledge Base & RAG Management ===

@app.get("/api/knowledge")
def get_knowledge_base():
    """Returns all loaded knowledge items and domain categories."""
    return {
        "domains": rag_engine.get_domains(),
        "total": len(rag_engine.chunks),
        "terms": rag_engine.get_all_terms()
    }


class AddTermRequest(BaseModel):
    term: str
    definition: str
    domain: str = "custom"


@app.post("/api/knowledge")
def add_custom_term(payload: AddTermRequest):
    """Dynamically adds a custom term to the vector knowledge base."""
    if not payload.term.strip() or not payload.definition.strip():
        raise HTTPException(status_code=400, detail="Term and definition cannot be empty.")

    chunk = rag_engine.add_custom_term(
        term=payload.term,
        definition=payload.definition,
        domain=payload.domain
    )
    return {"message": f"Term '{chunk.term}' added and indexed successfully.", "term": chunk.to_dict()}


@app.get("/api/settings")
def get_settings():
    """Returns LLM status and available domains."""
    return {
        "llm_status": llm_translator.get_status(),
        "domains": ["all"] + rag_engine.get_domains(),
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


class UpdateSettingsRequest(BaseModel):
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


@app.post("/api/settings")
def update_settings(payload: UpdateSettingsRequest):
    """Updates API keys dynamically in memory."""
    llm_translator.update_keys(
        groq_key=payload.groq_api_key,
        openai_key=payload.openai_api_key
    )
    return {"message": "Settings updated.", "status": llm_translator.get_status()}


@app.get("/api/history")
def get_history(session_id: str = "default"):
    """Fetches conversation history for a session."""
    return {"session_id": session_id, "history": llm_translator.conversation_manager.get_history(session_id)}


@app.delete("/api/history")
def clear_history(session_id: str = "default"):
    """Resets conversation history for a session."""
    llm_translator.conversation_manager.clear_history(session_id)
    return {"message": f"History for session '{session_id}' cleared."}


# === WebSocket Live Transcription ===

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()
    print("\n[+] Microphone connected via WebSocket.")

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
                    segments, info = whisper_model.transcribe(
                        io.BytesIO(wav_bytes),
                        beam_size=5,
                        vad_filter=False,
                        language=None
                    )
                    return ' '.join([s.text for s in segments]).strip(), info

                text, info = await asyncio.to_thread(transcribe_ws_chunk, wav_data)

                if text:
                    print(f"-> Whisper heard: '{text}'")
                    # Also perform quick RAG retrieval on the live chunk
                    retrieved = rag_engine.retrieve(text, top_k=2)
                    await websocket.send_json({
                        "text": text,
                        "detected_lang": info.language,
                        "confidence": round(info.language_probability, 2),
                        "retrieved_context": retrieved
                    })

    except WebSocketDisconnect:
        print("[-] WebSocket client disconnected.")
    except Exception as e:
        print(f"[!] WebSocket Error: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass