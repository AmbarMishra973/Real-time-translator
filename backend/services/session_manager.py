"""
In-Memory Session Manager with Bounded History and Inactivity Lifecycle.
Ensures session memory cannot grow unboundedly across sustained operation.
"""

import time
import threading
from typing import Dict, List, Optional, Any


class SessionRecord:
    """Represents an active multi-turn conversation session."""

    def __init__(self, session_id: str, max_turns: int = 20):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.last_accessed_at = time.time()

    def touch(self) -> None:
        self.last_accessed_at = time.time()

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.last_accessed_at) > ttl_seconds

    def add_turn(
        self,
        source_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.touch()
        turn = {
            "source_text": source_text.strip(),
            "translated_text": translated_text.strip(),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_history(self) -> List[Dict[str, Any]]:
        self.touch()
        return list(self.turns)

    def clear(self) -> None:
        self.touch()
        self.turns.clear()


class SessionManager:
    """
    Thread-safe conversation manager with bounded history per session
    and lazy TTL-based expiration for inactive conversations.
    """

    def __init__(self, ttl_seconds: int = 1800, max_turns: int = 20):
        """
        Args:
            ttl_seconds: Inactivity window before a session is purged (default 30 mins).
            max_turns: Maximum number of conversation turns preserved per session (default 20).
        """
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def _cleanup_expired_locked(self) -> int:
        """Purge sessions that exceeded inactivity TTL. Must be called with self._lock held."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired(self.ttl_seconds)]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def get_or_create(self, session_id: str) -> SessionRecord:
        with self._lock:
            self._cleanup_expired_locked()
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionRecord(session_id, max_turns=self.max_turns)
            session = self._sessions[session_id]
            session.touch()
            return session

    def add_turn(
        self,
        session_id: str,
        source_text: str,
        translated_text: str,
        source_lang: str = "en",
        target_lang: str = "hi",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        session = self.get_or_create(session_id)
        with self._lock:
            session.add_turn(source_text, translated_text, source_lang, target_lang, metadata)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return session.get_history()

    def clear_history(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def active_session_count(self) -> int:
        with self._lock:
            self._cleanup_expired_locked()
            return len(self._sessions)

    def format_history_for_prompt(self, session_id: str, last_k: int = 4) -> str:
        history = self.get_history(session_id)
        if not history:
            return "No previous conversation context."

        formatted = []
        for turn in history[-last_k:]:
            formatted.append(
                f"User ({turn['source_lang']}): {turn['source_text']}\n"
                f"Translated ({turn['target_lang']}): {turn['translated_text']}"
            )
        return "\n\n".join(formatted)


# Default process-level session manager instance
session_manager = SessionManager()
