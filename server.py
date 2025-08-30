from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import io
import asyncio
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import edge_tts

app = FastAPI()

model = WhisperModel("small", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile, lang: str = Form(...)):
    audio_bytes = await file.read()
    segments, info = model.transcribe(io.BytesIO(audio_bytes), language=lang, beam_size=5, vad_filter=False)
    text_parts = [seg.text for seg in segments]
    return {"text": ' '.join(text_parts).strip(), "detected_lang": info.language, "confidence": info.language_probability}

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
