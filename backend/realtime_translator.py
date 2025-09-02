"""
Real-Time Two‑Way Speech Translator with confirmation before translation
"""

import argparse
import asyncio
import contextlib
import io
import time
from dataclasses import dataclass
import os
import tempfile
import platform

import numpy as np
import sounddevice as sd
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
import edge_tts
from pydub import AudioSegment

if platform.system() == "Windows":
    import winsound

# Map language codes to Edge TTS voices
VOICE_MAP = {
    'en': 'en-US-JennyNeural',
    'hi': 'hi-IN-SwaraNeural',
    'zh': 'zh-CN-XiaoxiaoNeural',
}


def norm_lang_for_voice(lang_code: str) -> str:
    if not lang_code:
        return 'en'
    return lang_code.split('-')[0].lower()


def pick_voice(lang_code: str) -> str:
    base = norm_lang_for_voice(lang_code)
    return VOICE_MAP.get(base, 'en-US-JennyNeural')


@dataclass
class AppArgs:
    lang_a: str
    lang_b: str
    device: str
    whisper_size: str
    rate: int
    min_utt_s: float


class SimpleRecorder:
    def __init__(self, rate=16000, duration_s=7):
        self.rate = rate
        self.duration_s = duration_s

    def record(self):
        print(f"Recording {self.duration_s}s audio...")
        audio = sd.rec(int(self.duration_s * self.rate), samplerate=self.rate, channels=1, dtype='float32')
        sd.wait()
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        pcm16 = (audio[:, 0] * 32767).astype(np.int16).tobytes()
        return pcm16

    @contextlib.contextmanager
    def pause_for_playback(self):
        yield


def pcm16_to_wav_bytes(pcm16: bytes, rate: int = 16000) -> bytes:
    """Convert raw PCM16 audio bytes to a WAV file in memory"""
    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


# Shared Whisper model instance
_model = None

def init_whisper(size: str, device: str) -> WhisperModel:
    global _model
    if _model is None:
        compute_type = 'float16' if device == 'cuda' else 'int8'
        _model = WhisperModel(model_size_or_path=size, device=device, compute_type=compute_type)
    return _model


def transcribe_whisper(model: WhisperModel, wav_bytes: bytes, forced_lang: str):
    with io.BytesIO(wav_bytes) as f:
        segments, info = model.transcribe(f, beam_size=5, vad_filter=False, language=forced_lang)
        text_parts = [seg.text for seg in segments]
    return ' '.join(text_parts).strip(), info.language, info.language_probability


def transcribe_realtime(wav_bytes: bytes):
    """
    Real-time WebSocket-compatible transcription using VAD
    """
    model = init_whisper("small", "cpu")  # You can change model size/device here
    with io.BytesIO(wav_bytes) as f:
        segments, info = model.transcribe(f, beam_size=5, vad_filter=True, language="auto")
        text_parts = [seg.text for seg in segments]
    return ' '.join(text_parts).strip(), info.language, info.language_probability


def translate_text(text: str, target_lang: str) -> str:
    return GoogleTranslator(source='auto', target=target_lang).translate(text)


async def tts_to_mp3_bytes(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    out = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            out.write(chunk["data"])
    return out.getvalue()


def play_mp3_bytes(mp3_bytes: bytes, recorder: SimpleRecorder):
    with recorder.pause_for_playback():
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav")
            tmp_path = tmp.name
        try:
            if platform.system() == "Windows":
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            else:
                from pydub.playback import play
                play(AudioSegment.from_wav(tmp_path))
        finally:
            os.remove(tmp_path)


def log(msg: str):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def run_app(args: AppArgs):
    log(f"Loading Whisper model: {args.whisper_size} on {args.device}…")
    model = init_whisper(args.whisper_size, args.device)

    recorder = SimpleRecorder(rate=args.rate, duration_s=args.min_utt_s)
    log("Listening… (Ctrl+C to stop)")

    try:
        while True:
            utter_pcm = recorder.record()
            wav_bytes = pcm16_to_wav_bytes(utter_pcm, args.rate)
            t0 = time.time()
            text, lang, p = transcribe_whisper(model, wav_bytes, args.lang_a)
            if not text:
                continue

            log(f"Heard [{lang or 'unknown'} {p:.2f}]: {text}")

            # Confirm with user before translation
            confirm = input(f"Did you say '{text}'? (y/n): ").strip().lower()
            if confirm != 'y':
                log("Re-recording...")
                continue

            target_lang = args.lang_b if lang.lower().startswith(args.lang_a.lower()) else args.lang_a
            try:
                translated = translate_text(text, target_lang)
            except Exception as e:
                log(f"Translate error: {e}")
                continue

            log(f"→ ({target_lang}) {translated}")

            voice = pick_voice(target_lang)
            try:
                mp3_bytes = asyncio.run(tts_to_mp3_bytes(translated, voice))
                play_mp3_bytes(mp3_bytes, recorder)
            except Exception as e:
                log(f"TTS/playback error: {e}")
                continue

            log(f"Latency: {(time.time() - t0) * 1000:.0f} ms\n")

    except KeyboardInterrupt:
        log("Stopping…")


def parse_args() -> AppArgs:
    print("Select source language (e.g., en for English, hi for Hindi, zh for Chinese):")
    lang_a = input("Source language: ").strip() or 'en'
    print("Select target language (e.g., en for English, hi for Hindi, zh for Chinese):")
    lang_b = input("Target language: ").strip() or 'hi'

    ap = argparse.ArgumentParser(description="Real-time two-way speech translator (VAD-free)")
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--whisper-size', default='small')
    ap.add_argument('--rate', type=int, default=16000)
    ap.add_argument('--min-utt-s', type=float, default=7.0, help='Recording chunk length in seconds')
    ns = ap.parse_args([])

    return AppArgs(
        lang_a=lang_a,
        lang_b=lang_b,
        device=ns.device,
        whisper_size=ns.whisper_size,
        rate=ns.rate,
        min_utt_s=ns.min_utt_s,
    )


if __name__ == '__main__':
    cfg = parse_args()
    print(f"\n=== Real-Time Two‑Way Speech Translator ===")
    print(f"A↔B languages: {cfg.lang_a} ↔ {cfg.lang_b}")
    print("Tip: wear headphones to avoid feedback. Speak a short sentence and pause.")
    print("------------------------------------------\n")
    run_app(cfg)
