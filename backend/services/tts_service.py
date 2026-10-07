"""
Neural Text-To-Speech (TTS) Service using Edge-TTS.
Manages voice mappings, audio synthesis, and latency observation.
"""

import io
import time
from typing import Optional, Tuple
from fastapi import HTTPException
import edge_tts

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


class TTSService:
    """Encapsulates Edge-TTS neural speech synthesis and voice selection."""

    def __init__(self, voice_map: Optional[dict] = None):
        self.voice_map = voice_map or VOICE_MAP

    def pick_voice(self, lang_code: Optional[str]) -> str:
        base = (lang_code or 'en').split('-')[0].lower()
        return self.voice_map.get(base, 'en-US-JennyNeural')

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> Tuple[bytes, float]:
        """
        Synthesizes text to MP3 audio bytes using Edge-TTS neural voice.
        Returns: (audio_bytes, latency_seconds)
        """
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        t0 = time.perf_counter()
        selected_voice = voice or (self.pick_voice(target_lang) if target_lang else 'hi-IN-SwaraNeural')

        communicate = edge_tts.Communicate(text, selected_voice)
        out = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                out.write(chunk["data"])

        duration = round(time.perf_counter() - t0, 2)
        return out.getvalue(), duration


# Process-level singleton instance
tts_service = TTSService()


def pick_voice(lang_code: Optional[str]) -> str:
    """Convenience module function for neural voice selection."""
    return tts_service.pick_voice(lang_code)

