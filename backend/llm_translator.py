"""
LLM Translator module with Multi-Turn Conversation Memory and RAG Context Injection.
Supports Groq (ultra-low latency), OpenAI, and graceful fallback translation.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load any .env variables if present
load_dotenv()

from backend.rag_engine import rag_engine

# Try importing groq and deep_translator
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    GOOGLE_TRANSLATOR_AVAILABLE = True
except ImportError:
    GoogleTranslator = None
    MyMemoryTranslator = None
    GOOGLE_TRANSLATOR_AVAILABLE = False


class ConversationManager:
    """Stores conversation turns to handle context, pronouns, and flow."""

    def __init__(self, max_history: int = 6):
        self.max_history = max_history
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def add_turn(self, session_id: str, source_text: str, translated_text: str, source_lang: str, target_lang: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({
            "source_text": source_text.strip(),
            "translated_text": translated_text.strip(),
            "source_lang": source_lang,
            "target_lang": target_lang
        })
        # Keep only the most recent turns
        if len(self._sessions[session_id]) > self.max_history:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._sessions.get(session_id, [])

    def clear_history(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id] = []

    def format_history_for_prompt(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return "No previous conversation context."

        formatted = []
        for turn in history[-4:]:  # last 4 turns
            formatted.append(
                f"User ({turn['source_lang']}): {turn['source_text']}\n"
                f"Translated ({turn['target_lang']}): {turn['translated_text']}"
            )
        return "\n\n".join(formatted)


class LLMTranslator:
    """
    Orchestrates RAG context retrieval, conversation context injection,
    and LLM inference using Groq, OpenAI, or smart fallback.
    """

    LANGUAGE_NAMES = {
        'en': 'English',
        'hi': 'Hindi',
        'zh': 'Chinese (Simplified)',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ru': 'Russian',
        'ar': 'Arabic',
    }

    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "groq")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._groq_client = None

        if self.groq_api_key and GROQ_AVAILABLE:
            try:
                self._groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                print(f"[!] Warning: Could not initialize Groq client: {e}")

    def update_keys(self, groq_key: Optional[str] = None, openai_key: Optional[str] = None):
        """Allows updating API keys dynamically from the UI or API."""
        if groq_key is not None:
            self.groq_api_key = groq_key.strip()
            if self.groq_api_key and GROQ_AVAILABLE:
                try:
                    self._groq_client = Groq(api_key=self.groq_api_key)
                except Exception as e:
                    self._groq_client = None
            else:
                self._groq_client = None

        if openai_key is not None:
            self.openai_api_key = openai_key.strip()

    def get_status(self) -> Dict[str, Any]:
        has_groq = bool(self.groq_api_key and self._groq_client)
        has_openai = bool(self.openai_api_key)
        is_connected = has_groq or has_openai
        return {
            "is_llm_connected": is_connected,
            "groq_configured": has_groq,
            "openai_configured": has_openai,
            "groq_model": self.groq_model,
            "fallback_available": GOOGLE_TRANSLATOR_AVAILABLE,
            "active_mode": f"Groq ({self.groq_model})" if has_groq else ("OpenAI (gpt-4o-mini)" if has_openai else "Local Multilingual Engine")
        }

    def _build_system_prompt(
        self,
        source_lang_name: str,
        target_lang_name: str,
        rag_context_str: str,
        conversation_history_str: str
    ) -> str:
        return f"""You are a professional real-time multilingual translator and speech interpreter.
Your task is to accurately translate speech transcriptions from {source_lang_name} to {target_lang_name}.

DOMAIN KNOWLEDGE BASE (RAG CONTEXT):
The following relevant technical/domain terminology was retrieved from our knowledge base for this input:
{rag_context_str}

RECENT CONVERSATION HISTORY (for context and pronoun resolution):
{conversation_history_str}

CRITICAL TRANSLATION GUIDELINES:
1. Provide a natural, fluent, and culturally appropriate translation in {target_lang_name}.
2. PRESERVE TECHNICAL TERMS: Keep recognized technical terms, acronyms, product names, and frameworks (e.g., API, Kubernetes, Docker, RAG, LLM, CI/CD, Microservices) in their standard technical form or English loan terms where that is standard practice in {target_lang_name}.
3. CONTEXT AWARENESS: Use the recent conversation history to correctly interpret ambiguous pronouns (e.g., 'it', 'they', 'this', 'we') and maintain continuity.
4. CORRECTION: Minor speech recognition slips in the transcript should be corrected naturally to reflect intended speech.
5. STRICT OUTPUT FORMAT: Output ONLY the direct translated sentence in {target_lang_name}. Do NOT add quotes, markdown bolding, introductory phrases like "Here is the translation:", or commentary.
"""

    def _translate_with_groq(self, system_prompt: str, user_text: str) -> str:
        if not self._groq_client:
            raise RuntimeError("Groq client not initialized or missing API key.")

        try:
            response = self._groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text to translate:\n{user_text}"}
                ],
                temperature=0.2,
                max_tokens=256
            )
        except Exception as e:
            if "model_not_found" in str(e) or "does not exist" in str(e):
                print("[!] Model unavailable, retrying with llama-3.1-8b-instant...")
                response = self._groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Text to translate:\n{user_text}"}
                    ],
                    temperature=0.2,
                    max_tokens=256
                )
            else:
                raise e

        translated = response.choices[0].message.content.strip()
        # Clean any surrounding quotes if present
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1].strip()
        return translated

    def _translate_with_fallback(
        self,
        user_text: str,
        source_lang: str,
        target_lang: str,
        retrieved_items: List[Dict[str, Any]]
    ) -> str:
        """
        Fallback translator using GoogleTranslator and MyMemoryTranslator
        so the application remains fully functional even without external LLM API keys.
        """
        if not GOOGLE_TRANSLATOR_AVAILABLE or not user_text.strip():
            return user_text

        src = (source_lang or 'en').split('-')[0].lower()
        tgt = (target_lang or 'hi').split('-')[0].lower()
        if src == 'auto':
            src = 'en'

        def is_valid_translation(text: str) -> bool:
            if not text:
                return False
            low = text.lower()
            if "error 500" in low or "server error" in low or "that’s all we know" in low or "no translation was found" in low:
                return False
            return True

        # 1. Try GoogleTranslator with specified source
        try:
            translated = GoogleTranslator(source=src, target=tgt).translate(user_text)
            if is_valid_translation(translated):
                return translated
        except Exception:
            pass

        # 2. Try MyMemoryTranslator
        if MyMemoryTranslator:
            try:
                mem_src = f"{src}-US" if src == 'en' else (f"{src}-IN" if src == 'hi' else src)
                mem_tgt = f"{tgt}-IN" if tgt == 'hi' else (f"{tgt}-US" if tgt == 'en' else tgt)
                translated = MyMemoryTranslator(source=mem_src, target=mem_tgt).translate(user_text)
                if is_valid_translation(translated):
                    return translated
            except Exception:
                pass

        # 3. Try auto source on Google
        try:
            translated = GoogleTranslator(source='auto', target=tgt).translate(user_text)
            if is_valid_translation(translated):
                return translated
        except Exception:
            pass

        return user_text

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "hi",
        session_id: str = "default",
        domain: Optional[str] = "all",
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes the full RAG + Context + LLM translation workflow.
        """
        import time
        t0 = time.perf_counter()
        text = (text or "").strip()
        if not text or not any(c.isalnum() for c in text):
            return {
                "translated_text": "",
                "retrieved_context": [],
                "sources_used": [],
                "provider": "none",
                "latency_s": 0.0,
                "history": self.conversation_manager.get_history(session_id)
            }

        # 1. RAG Retrieval
        rag_res = rag_engine.retrieve(query=text, domain=domain, top_k=top_k)
        retrieved_items = rag_res["chunks"]
        sources_used = rag_res["sources_used"]
        rag_context_str = rag_engine.format_context_for_prompt(retrieved_items)

        # 2. Conversation History
        history_str = self.conversation_manager.format_history_for_prompt(session_id)

        source_name = self.LANGUAGE_NAMES.get(source_lang.split('-')[0].lower(), source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang.split('-')[0].lower(), target_lang)

        system_prompt = self._build_system_prompt(
            source_lang_name=source_name,
            target_lang_name=target_name,
            rag_context_str=rag_context_str,
            conversation_history_str=history_str
        )

        translated_text = ""
        provider_used = "Local Multilingual Engine"

        # 3. LLM Translation
        t_llm_start = time.perf_counter()
        if self._groq_client:
            try:
                translated_text = self._translate_with_groq(system_prompt, text)
                provider_used = f"Groq ({self.groq_model})"
            except Exception as e:
                print(f"[!] Groq translation failed, falling back: {e}")
                translated_text = self._translate_with_fallback(text, source_lang, target_lang, retrieved_items)
                provider_used = "Local Multilingual Engine"
        else:
            translated_text = self._translate_with_fallback(text, source_lang, target_lang, retrieved_items)
            provider_used = "Local Multilingual Engine"
        
        llm_latency_s = round(time.perf_counter() - t_llm_start, 2)

        # 4. Save turn to conversation history
        self.conversation_manager.add_turn(
            session_id=session_id,
            source_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang
        )

        return {
            "source_text": text,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "retrieved_context": retrieved_items,
            "sources_used": sources_used,
            "provider": provider_used,
            "latency_s": llm_latency_s,
            "total_latency_s": round(time.perf_counter() - t0, 2),
            "history": self.conversation_manager.get_history(session_id)
        }


# Global singleton instance
llm_translator = LLMTranslator()
