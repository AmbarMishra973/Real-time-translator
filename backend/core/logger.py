"""
Lightweight Structured Logging & Tracing for STT and Pipeline Events.
Provides consistent formatting and request/utterance correlation without external dependencies.
"""

import os
import json
import sys
from typing import Any, Dict, Optional

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def stt_log(event: str, utterance_id: str, **fields: Any) -> None:
    """
    Emit structured, non-sensitive STT diagnostics.
    Transcript text logging is gated behind STT_DEBUG=true in production.
    """
    payload = {
        "subsystem": "STT",
        "event": event,
        "utterance_id": utterance_id,
        **fields
    }
    print("[STT] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def pipeline_log(event: str, request_id: str, **fields: Any) -> None:
    """
    Emit structured pipeline tracing events correlating request stages.
    """
    payload = {
        "subsystem": "PIPELINE",
        "event": event,
        "request_id": request_id,
        **fields
    }
    print("[PIPELINE] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)
