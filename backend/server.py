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
import re
import asyncio
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

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
from backend.audio_diagnostics import boost_quiet_pcm16_wav, inspect_pcm16_wav, pad_pcm16_wav, upload_suffix, validate_normalized_audio

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


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Methods": "*"}
    )


# === Initialize Whisper Model ===
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
print(f"[*] Initializing Faster-Whisper ({WHISPER_SIZE} on {WHISPER_DEVICE})...")
whisper_model = None
try:
    whisper_model = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    print("[+] Whisper model loaded successfully.")
except Exception as e:
    print(f"[!] Warning: Could not load {WHISPER_SIZE} model ({e}).")
    if WHISPER_SIZE != "base":
        try:
            print("[*] Attempting cached base-model fallback...")
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("[+] Base fallback model loaded successfully.")
        except Exception as fallback_error:
            print(f"[!] Local Whisper is unavailable: {fallback_error}")

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


def stt_log(event: str, utterance_id: str, **fields) -> None:
    """Emit structured, non-sensitive STT diagnostics; transcript text requires STT_DEBUG=true."""
    payload = {"event": event, "utterance_id": utterance_id, **fields}
    print("[STT] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def parse_capture_metadata(raw_metadata: Optional[str]) -> dict:
    """Keep only non-identifying browser audio settings supplied with an upload."""
    if not raw_metadata:
        return {}
    try:
        metadata = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(metadata, dict):
        return {}
    allowed_keys = {"sampleRate", "channelCount", "sampleSize", "echoCancellation", "noiseSuppression", "autoGainControl", "mimeType"}
    return {key: metadata[key] for key in allowed_keys if key in metadata and isinstance(metadata[key], (str, int, float, bool, type(None)))}


def save_debug_audio(wav_bytes: bytes, utterance_id: str) -> None:
    """Persist decoded audio only after an explicit, local debug opt-in."""
    if os.getenv("STT_DEBUG_SAVE_AUDIO", "false").lower() != "true":
        return
    directory = os.getenv("STT_DEBUG_AUDIO_DIR", "").strip()
    if not directory:
        stt_log("debug_audio_skipped", utterance_id, reason="STT_DEBUG_AUDIO_DIR is not configured")
        return
    output_dir = Path(directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{utterance_id}.wav"
    output_path.write_bytes(wav_bytes)
    stt_log("debug_audio_saved", utterance_id, path=str(output_path))


def convert_to_clean_wav(audio_bytes: bytes, suffix: str = ".bin") -> bytes:
    """
    Converts and resamples incoming browser audio (WebM, OGG, MP4, WAV, etc.) into clean 16kHz mono WAV PCM.
    Normalization is opt-in because its benefit must be measured per microphone.
    """
    import subprocess, tempfile, os

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_f:
        in_f.write(audio_bytes)
        in_path = in_f.name

    out_path = in_path + ".wav"
    try:
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
    except Exception as e:
        print(f"[!] Warning: Audio conversion error: {e}")
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

    raise ValueError("Audio conversion failed; verify that FFmpeg supports the uploaded audio format.")


WHISPER_HALLUCINATIONS = {
    "", "you", "you.", "you!", "thank you", "thank you.", "thank you for watching",
    "thank you for watching.", "thanks for watching", "thanks for watching!",
    "...", ".", "..", "....", ". . . .", "bye", "bye.", "amara.org", "subscribe",
    "subtitles by", "subtitles by the amara.org community", "please subscribe",
    "so", "so.", "yeah", "yeah."
}


def sync_transcribe(audio_bytes: bytes, lang: Optional[str] = None, filename: Optional[str] = None, content_type: Optional[str] = None, capture_metadata: Optional[dict] = None):
    utterance_id = uuid.uuid4().hex[:12]
    stt_started_at = time.perf_counter()

    # 1. Convert and validate initial incoming audio container into 16kHz mono WAV
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

    # 3. Pre-Flight Silence Gate (FROZEN HERE on post-gain, pre-padding signal levels)
    # Gating uses pre-padding energy so zero-padding does not dilute RMS of short utterances
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
        print(
            f"[STT DIAGNOSTIC LOG (SILENCE GATE)]\n"
            f"  - recording MIME type:        {content_type}\n"
            f"  - Blob size:                  {len(audio_bytes)}\n"
            f"  - pre-gain RMS:               {pre_diag['rms_dbfs']} dBFS\n"
            f"  - pre-gain Peak:              {pre_diag['peak_dbfs']} dBFS\n"
            f"  - gain applied:               +{gain_db_applied} dB\n"
            f"  - post-gain RMS:              {post_diag['rms_dbfs']} dBFS\n"
            f"  - post-gain Peak:             {post_diag['peak_dbfs']} dBFS\n"
            f"  - gate reason:                {gate_reason}\n"
            f"  - action:                     Skipped Whisper call completely (gate fired)"
        )
        info = SimpleNamespace(language=whisper_lang or "en", language_probability=0.0, language_source="gate")
        return "", info, post_diag, "gate", "none"

    post_diag["is_silent_gate"] = False
    post_diag["gate_reason"] = None

    # 4. Temporal Padding: Pad short clips (< 1.0s) with 250ms digital silence for phoneme context
    if post_diag["duration_s"] < 1.0:
        wav_bytes, padded_duration = pad_pcm16_wav(wav_bytes, pad_ms=250)
        post_diag["padded_duration_s"] = padded_duration
    else:
        post_diag["padded_duration_s"] = post_diag["duration_s"]

    # Save audio files locally for external manual listening & diagnosis
    debug_dir = Path("backend/debug_audio")
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        (debug_dir / f"last_recording{input_suffix}").write_bytes(audio_bytes)
        (debug_dir / "last_recording.wav").write_bytes(wav_bytes)
    except Exception as e:
        print(f"[!] Warning: Could not write debug audio: {e}")

    stt_log("audio_validated", utterance_id, input_bytes=len(audio_bytes), input_container=input_suffix, input_content_type=content_type, capture_metadata=capture_metadata or {}, normalized_bytes=len(wav_bytes), source_language=whisper_lang or "auto", **post_diag)

    # 5. Whisper Transcription (Groq LPU primary, Local Faster-Whisper fallback)
    if llm_translator._groq_client:
        engine_used = "groq"
        model_used = "whisper-large-v3-turbo"
        stt_log("transcription_started", utterance_id, engine=engine_used, model=model_used, source_language=whisper_lang or "auto")
        try:
            clean_text = llm_translator.transcribe_with_groq(wav_bytes, whisper_lang)
            info = SimpleNamespace(language=whisper_lang or "en", language_probability=1.0, language_source="groq")
        except Exception as e:
            stt_log("groq_failed", utterance_id, error=str(e))
            raise HTTPException(status_code=502, detail=f"Groq Whisper failed: {e}")
    else:
        if whisper_model is None:
            raise RuntimeError("No STT engine is available. Configure Groq or install/cache the configured Faster-Whisper model.")

        engine_used = "local"
        model_used = WHISPER_SIZE
        stt_log("transcription_started", utterance_id, engine=engine_used, model=model_used, source_language=whisper_lang or "auto")

        segments, info = whisper_model.transcribe(
            io.BytesIO(wav_bytes),
            language=whisper_lang,
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
        )
        clean_text = ' '.join(seg.text for seg in segments).strip()
        if info is None:
            info = SimpleNamespace(language=whisper_lang, language_probability=None, language_source="selected" if whisper_lang else "unknown")

    # 6. Normalized Post-Transcription Hallucination Guard
    clean_norm = re.sub(r'[^\w\s]', '', (clean_text or '').lower()).strip()
    SUSPECTED_HALLUCINATIONS = {
        "thank you", "thank you very much", "thanks for watching", "you", "bye", "subscribe"
    }
    is_suspected = False
    if (
        clean_norm in SUSPECTED_HALLUCINATIONS
        and post_diag.get("pre_duration_s", post_diag["duration_s"]) > 2.0
        and post_diag.get("pre_rms_dbfs", 0.0) < -38.0
    ):
        is_suspected = True
        stt_log("suspected_hallucination", utterance_id, text=clean_text, normalized_text=clean_norm, pre_rms_dbfs=post_diag.get("pre_rms_dbfs"), duration_s=post_diag["duration_s"])
        print(f"[STT] Suspected hallucination on quiet audio ('{clean_text}'), consider re-recording", flush=True)

    post_diag["suspected_hallucination"] = is_suspected

    # 7. Complete Diagnostic Report
    print(
        f"[STT DIAGNOSTIC LOG]\n"
        f"  - recording MIME type:        {content_type}\n"
        f"  - Blob size:                  {len(audio_bytes)}\n"
        f"  - backend received byte size: {len(audio_bytes)}\n"
        f"  - pre-gain RMS:               {post_diag.get('pre_rms_dbfs')} dBFS\n"
        f"  - pre-gain Peak:              {post_diag.get('pre_peak_dbfs')} dBFS\n"
        f"  - gain applied:               +{post_diag.get('gain_db_applied')} dB\n"
        f"  - post-gain RMS:              {post_diag.get('rms_dbfs')} dBFS\n"
        f"  - post-gain Peak:             {post_diag.get('peak_dbfs')} dBFS\n"
        f"  - pre-pad duration:           {post_diag.get('pre_duration_s')}s\n"
        f"  - post-pad duration:          {post_diag.get('padded_duration_s')}s\n"
        f"  - sample rate:                {post_diag.get('sample_rate_hz')} Hz\n"
        f"  - channels:                   {post_diag.get('channels')}\n"
        f"  - Whisper model:              {model_used}\n"
        f"  - Whisper response:           \"{clean_text}\"\n"
        f"  - suspected hallucination:    {is_suspected}\n"
        f"  - final transcript:           \"{clean_text}\""
    )

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
        "stt_status": {"local_model_loaded": whisper_model is not None, "requested_model": WHISPER_SIZE},
        "rag_domains": rag_engine.get_domains(),
        "total_knowledge_terms": len(rag_engine.chunks)
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    capture_metadata: Optional[str] = Form(None),
):
    """
    Clean STT Diagnostic Endpoint:
    Receives audio, validates, transcribes without LLM, RAG, or TTS.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    try:
        text, info, diagnostics, engine_used, model_used = await asyncio.to_thread(
            sync_transcribe, audio_bytes, lang, file.filename, file.content_type, parse_capture_metadata(capture_metadata)
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


@app.post("/translate")
async def translate_text(
    text: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("hi"),
    session_id: str = Form("default"),
    domain: str = Form("all"),
    capture_metadata: Optional[str] = Form(None),
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
    domain: str = Form("all"),
    capture_metadata: Optional[str] = Form(None)
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
    try:
        transcript, info, diagnostics, engine_used, model_used = await asyncio.to_thread(
            sync_transcribe, audio_bytes, source_lang, file.filename, file.content_type, parse_capture_metadata(capture_metadata)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    t_stt = time.perf_counter() - t_stt_start

    # Fast Short-Circuit on Pre-Flight Silence Gate or Empty Spoken Content:
    # Immediately returns WITHOUT calling RAG retrieval or LLM translation!
    if diagnostics.get("is_silent_gate") or not transcript or not any(c.isalnum() for c in transcript):
        message = diagnostics.get("gate_message") or "No speech detected in audio."
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
            "message": message,
            "gate_reason": diagnostics.get("gate_reason"),
            "suspected_hallucination": False,
            "audio": diagnostics if os.getenv("STT_DEBUG", "false").lower() == "true" else None,
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
            "total_s": round(t_total, 2)
        },
        "stt_engine": engine_used,
        "stt_model": model_used,
        "suspected_hallucination": diagnostics.get("suspected_hallucination", False),
        "gate_reason": None,
        "audio": diagnostics if os.getenv("STT_DEBUG", "false").lower() == "true" else None,
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
    if os.getenv("ENABLE_EXPERIMENTAL_WEBSOCKET", "false").lower() != "true":
        await websocket.close(code=1008, reason="Experimental streaming STT is disabled; use /pipeline.")
        return
    if whisper_model is None:
        await websocket.close(code=1011, reason="No local Whisper model is available.")
        return
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
