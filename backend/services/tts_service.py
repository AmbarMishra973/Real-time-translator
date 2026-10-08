"""
Neural Text-To-Speech (TTS) Service with Engine Dispatch & Feature Flags.
Supports Edge-TTS (Production Control), Piper ONNX (Local Candidate),
and SAPI5 (Local Windows Fallback).
"""

import io
import os
import time
import wave
from typing import Optional, Tuple
from fastapi import HTTPException
import edge_tts

# Default TTS Engine Feature Flag (Production Default: edge)
TTS_ENGINE = os.environ.get("TTS_ENGINE", "edge").lower().strip()

# High-quality neural voices mapping (Edge-TTS Control)
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

# Piper Voice Model Configurations (Candidate A)
PIPER_VOICE_MODELS = {
    'en': {
        'model_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'piper', 'en_US-lessac-medium.onnx'),
        'config_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'piper', 'en_US-lessac-medium.onnx.json'),
    },
    'hi': {
        'model_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'piper', 'hi_IN-pratham-medium.onnx'),
        'config_path': os.path.join(os.path.dirname(__file__), '..', 'models', 'piper', 'hi_IN-pratham-medium.onnx.json'),
    }
}


class TTSService:
    """Encapsulates speech synthesis across Edge-TTS and candidate local engines."""

    def __init__(self, voice_map: Optional[dict] = None, engine: Optional[str] = None):
        self.voice_map = voice_map or VOICE_MAP
        self.engine = (engine or TTS_ENGINE).lower().strip()
        self._piper_cache = {}

    def get_active_engine(self) -> str:
        """Returns active engine name taking runtime env into account."""
        return os.environ.get("TTS_ENGINE", self.engine).lower().strip()

    def pick_voice(self, lang_code: Optional[str]) -> str:
        """Returns voice identifier for given language code under Edge-TTS."""
        base = (lang_code or 'en').split('-')[0].lower()
        return self.voice_map.get(base, 'en-US-JennyNeural')

    def _get_piper_voice(self, lang_code: str):
        """Loads and caches Piper ONNX voice model."""
        import piper
        base = (lang_code or 'en').split('-')[0].lower()
        if base not in PIPER_VOICE_MODELS:
            base = 'en'

        if base in self._piper_cache:
            return self._piper_cache[base]

        cfg = PIPER_VOICE_MODELS[base]
        if not os.path.exists(cfg['model_path']) or not os.path.exists(cfg['config_path']):
            raise FileNotFoundError(
                f"Piper model files missing for language '{base}': {cfg['model_path']}"
            )

        voice = piper.PiperVoice.load(cfg['model_path'], cfg['config_path'])
        self._piper_cache[base] = voice
        return voice

    async def _synthesize_edge(
        self,
        text: str,
        voice: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> Tuple[bytes, float]:
        """Synthesizes text using Edge-TTS neural voice (Control)."""
        t0 = time.perf_counter()
        selected_voice = voice or (self.pick_voice(target_lang) if target_lang else 'hi-IN-SwaraNeural')

        communicate = edge_tts.Communicate(text, selected_voice)
        out = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                out.write(chunk["data"])

        duration = round(time.perf_counter() - t0, 3)
        return out.getvalue(), duration

    def _synthesize_piper(
        self,
        text: str,
        target_lang: Optional[str] = None
    ) -> Tuple[bytes, float]:
        """Synthesizes text using Piper local ONNX model (Candidate A)."""
        t0 = time.perf_counter()
        lang = (target_lang or 'en').split('-')[0].lower()
        voice = self._get_piper_voice(lang)

        chunks = list(voice.synthesize(text))
        pcm_bytes = b''.join(c.audio_int16_bytes for c in chunks)

        # Wrap in valid PCM WAV container for browser compatibility
        out = io.BytesIO()
        with wave.open(out, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)
            wf.writeframes(pcm_bytes)

        duration = round(time.perf_counter() - t0, 3)
        return out.getvalue(), duration

    def _synthesize_sapi5(
        self,
        text: str
    ) -> Tuple[bytes, float]:
        """Synthesizes text using Windows SAPI5 COM dispatch (Candidate C)."""
        import tempfile
        t0 = time.perf_counter()
        tmp_path = tempfile.mktemp(suffix=".wav")

        try:
            import win32com.client
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(tmp_path, 3)  # 3 = SSFMCreateForWrite
            voice.AudioOutputStream = stream
            voice.Speak(text)
            stream.Close()
            del stream
            del voice

            if os.path.exists(tmp_path):
                with open(tmp_path, "rb") as f:
                    wav_bytes = f.read()
            else:
                wav_bytes = b""
        except Exception:
            wav_bytes = b""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        duration = round(time.perf_counter() - t0, 3)
        return wav_bytes, duration

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> Tuple[bytes, float]:
        """
        Synthesizes text to audio bytes using the active TTS engine.
        Returns: (audio_bytes, latency_seconds)
        """
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        active = self.get_active_engine()

        if active == "edge":
            return await self._synthesize_edge(text, voice=voice, target_lang=target_lang)
        elif active == "piper":
            return self._synthesize_piper(text, target_lang=target_lang)
        elif active in ("sapi5", "candidate_local"):
            return self._synthesize_sapi5(text)
        elif active == "indic_tts":
            raise HTTPException(
                status_code=501,
                detail="AI4Bharat Indic-TTS is unfeasible on Windows Python 3.13 without heavy PyTorch stack."
            )
        else:
            # Fallback to Edge-TTS
            return await self._synthesize_edge(text, voice=voice, target_lang=target_lang)


# Process-level singleton instance
tts_service = TTSService()


def pick_voice(lang_code: Optional[str]) -> str:
    """Convenience module function for neural voice selection."""
    return tts_service.pick_voice(lang_code)
