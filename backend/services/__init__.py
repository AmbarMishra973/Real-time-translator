"""Modular Service Layer for Real-Time Translator."""

from backend.services.stt_service import stt_service, STTService, parse_capture_metadata, convert_to_clean_wav
from backend.services.rag_service import rag_service, RAGService
from backend.services.llm_service import llm_service, LLMService
from backend.services.tts_service import tts_service, TTSService, VOICE_MAP
from backend.services.session_manager import session_manager, SessionManager
from backend.services.vad_service import vad_service, VADService, VADConfig, VADSessionState
from backend.services.hypothesis_service import (
    hypothesis_service,
    HypothesisService,
    HypothesisConfig,
    HypothesisSessionState,
    HypothesisResult
)
from backend.services.translation_context_gate import (
    translation_context_gate,
    TranslationContextGate,
    GateDecision
)

__all__ = [
    "stt_service", "STTService", "parse_capture_metadata", "convert_to_clean_wav",
    "rag_service", "RAGService",
    "llm_service", "LLMService",
    "tts_service", "TTSService", "VOICE_MAP",
    "session_manager", "SessionManager",
    "vad_service", "VADService", "VADConfig", "VADSessionState",
    "hypothesis_service", "HypothesisService", "HypothesisConfig", "HypothesisSessionState", "HypothesisResult",
    "translation_context_gate", "TranslationContextGate", "GateDecision"
]

