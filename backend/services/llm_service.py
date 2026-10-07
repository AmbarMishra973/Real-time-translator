"""
LLM Translation Service.
Encapsulates Groq/OpenAI client interactions, multi-tier fallback cascade, and prompt engineering.
"""

from typing import Dict, Any, Optional
from backend.llm_translator import llm_translator


class LLMService:
    """Wraps LLM-based translation and external API integration."""

    def __init__(self, translator=llm_translator):
        self._translator = translator

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "hi",
        session_id: str = "default",
        domain: str = "all"
    ) -> Dict[str, Any]:
        """Execute context-aware and domain-grounded translation."""
        return self._translator.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            session_id=session_id,
            domain=domain
        )

    def get_status(self) -> Dict[str, Any]:
        """Return provider connection and model status."""
        return self._translator.get_status()

    def update_keys(self, groq_key: Optional[str] = None, openai_key: Optional[str] = None) -> None:
        """Update API credentials dynamically in memory."""
        self._translator.update_keys(groq_key=groq_key, openai_key=openai_key)

    def transcribe_with_groq(self, wav_bytes: bytes, lang: Optional[str] = None) -> str:
        """Execute cloud Whisper transcription on Groq LPU if configured."""
        return self._translator.transcribe_with_groq(wav_bytes=wav_bytes, lang=lang)

    @property
    def has_groq_client(self) -> bool:
        """Check if Groq API client is authenticated and initialized."""
        return bool(self._translator._groq_client)

    @property
    def conversation_manager(self):
        """Access underlying conversation manager for backward compatibility."""
        return self._translator.conversation_manager


# Process-level singleton instance
llm_service = LLMService()
