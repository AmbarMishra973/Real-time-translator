from fastapi import FastAPI, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import io
import asyncio
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import edge_tts

app = FastAPI()

# Load model once at startup
model = WhisperModel("small", device="cpu", compute_type="int8")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile, lang: str = Form(...)):
    audio_bytes = await file.read()
    segments, info = model.transcribe(io.BytesIO(audio_bytes), language=lang, beam_size=5, vad_filter=False)
    text_parts = [seg.text for seg in segments]
    return {
        "text": ' '.join(text_parts).strip(),
        "detected_lang": info.language,
        "confidence": info.language_probability
    }


@app.post("/translate")
def translate(text: str = Form(...), target_lang: str = Form(...)):
    translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
    return {"translated": translated}


@app.post("/tts")
async def text_to_speech(text: str = Form(...), voice: str = Form(...)):
    communicate = edge_tts.Communicate(text, voice)
    out = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            out.write(chunk["data"])
    return StreamingResponse(io.BytesIO(out.getvalue()), media_type="audio/mpeg")


# === NEW: WebSocket Streaming Transcription ===
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    buffer = bytearray()

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer.extend(data)

            # If buffer is large enough (~1 sec audio), transcribe it
            if len(buffer) >= 32000:  # ≈ 1 sec @ 16-bit PCM 16kHz mono
                wav_data = pcm16_to_wav_bytes(buffer[:32000])
                buffer = buffer[32000:]  # Remove processed portion

                segments, info = model.transcribe(io.BytesIO(wav_data), beam_size=5, vad_filter=True, language="auto")
                text = ' '.join([s.text for s in segments]).strip()

                if text:
                    await websocket.send_json({
                        "text": text,
                        "detected_lang": info.language,
                        "confidence": round(info.language_probability, 2)
                    })

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    except Exception as e:
        await websocket.send_json({"error": str(e)})


# === Helper: Convert PCM16 to WAV (in-memory) ===
def pcm16_to_wav_bytes(pcm16_bytes: bytes, rate: int = 16000) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16_bytes)
    return buf.getvalue()  
add lang
