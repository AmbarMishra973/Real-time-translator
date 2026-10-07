"""
WebSocket Routes for Real-Time Audio Streaming (Phase 3).
Provides persistent low-latency bi-directional streaming, incremental partials,
final RAG-augmented translation, and instant turn cancellation.
"""

import os
import json
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.logger import stream_log
from backend.services.stt_service import stt_service
from backend.core.streaming_orchestrator import streaming_orchestrator, StreamingSession

router = APIRouter()


def is_streaming_enabled() -> bool:
    """Check if WebSocket streaming feature is enabled (defaults to true for Phase 3)."""
    return (
        os.getenv("ENABLE_STREAMING", "true").lower() == "true"
        or os.getenv("ENABLE_EXPERIMENTAL_WEBSOCKET", "false").lower() == "true"
    )


@router.websocket("/ws/transcribe")
@router.websocket("/ws/stream")
async def websocket_streaming_endpoint(websocket: WebSocket):
    """
    Dedicated WebSocket endpoint for real-time chunked speech streaming:
    1. Connect ➔ 'ready' event with request_id & session_id
    2. Audio frames (bytes) ➔ Buffered & periodic windowed 'partial' events
    3. 'end' control ➔ Authoritative 'final' event with metrics (TTFR, STT, RAG, LLM)
    4. 'cancel' control ➔ Immediate abort and 'cancelled' event
    """
    if not is_streaming_enabled():
        await websocket.close(code=1008, reason="Streaming STT is disabled; use /pipeline REST endpoint.")
        return

    if stt_service.model is None:
        await websocket.close(code=1011, reason="No local Faster-Whisper model available on server.")
        return

    await websocket.accept()

    # Query params fallback for session & language initialization
    query_params = dict(websocket.query_params)
    initial_session_id = query_params.get("session_id")
    initial_lang = query_params.get("lang") or query_params.get("language") or "en"
    initial_target = query_params.get("target_lang") or "hi"
    initial_domain = query_params.get("domain") or "all"

    session: StreamingSession = streaming_orchestrator.create_session(
        session_id=initial_session_id,
        language=initial_lang,
        target_lang=initial_target,
        domain=initial_domain
    )

    stream_log("stream_connected", session.request_id, session_id=session.session_id)

    # Initial ready handshake
    await websocket.send_json({
        "type": "ready",
        "request_id": session.request_id,
        "session_id": session.session_id
    })

    background_tasks = set()

    try:
        while True:
            msg = await websocket.receive()

            if "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    await websocket.send_json({
                        "type": "error",
                        "request_id": session.request_id,
                        "message": "Malformed JSON control payload."
                    })
                    continue

                msg_type = payload.get("type", "").lower()

                if msg_type == "start":
                    # Reconfigure or initialize turn
                    session = streaming_orchestrator.create_session(
                        session_id=payload.get("session_id", session.session_id),
                        language=payload.get("language", session.language),
                        target_lang=payload.get("target_lang", session.target_lang),
                        domain=payload.get("domain", session.domain)
                    )
                    await websocket.send_json({
                        "type": "ready",
                        "request_id": session.request_id,
                        "session_id": session.session_id
                    })

                elif msg_type == "cancel":
                    # Cancel any active partial background tasks
                    for task in list(background_tasks):
                        task.cancel()
                    background_tasks.clear()

                    cancel_res = streaming_orchestrator.cancel_stream(session)
                    await websocket.send_json(cancel_res)
                    session.reset_for_next_turn()

                elif msg_type == "end":
                    # Finalize audio turn
                    for task in list(background_tasks):
                        task.cancel()
                    background_tasks.clear()

                    final_res = await streaming_orchestrator.finalize_stream(session)
                    await websocket.send_json(final_res)
                    # Prepare session for potential subsequent utterance
                    session.reset_for_next_turn()

                else:
                    await websocket.send_json({
                        "type": "error",
                        "request_id": session.request_id,
                        "message": f"Unrecognized control message type: '{msg_type}'"
                    })

            elif "bytes" in msg and msg["bytes"]:
                raw_chunk = msg["bytes"]
                try:
                    session.add_chunk(raw_chunk)
                except ValueError as ve:
                    # Buffer overflow protection
                    await websocket.send_json({
                        "type": "error",
                        "request_id": session.request_id,
                        "message": str(ve)
                    })
                    session.reset_for_next_turn()
                    continue

                if session.should_trigger_partial():
                    async def evaluate_and_emit(s: StreamingSession):
                        try:
                            partial = await streaming_orchestrator.evaluate_partial(s)
                            if partial and not s.is_cancelled:
                                await websocket.send_json({
                                    "type": "partial",
                                    "text": partial,
                                    "request_id": s.request_id
                                })
                        except Exception:
                            pass

                    t = asyncio.create_task(evaluate_and_emit(session))
                    background_tasks.add(t)
                    t.add_done_callback(background_tasks.discard)

    except (WebSocketDisconnect, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "disconnect" not in str(e).lower():
            stream_log("stream_error", session.request_id, error=str(e))
        else:
            stream_log("stream_disconnected", session.request_id, session_id=session.session_id)
    except Exception as exc:
        stream_log("stream_error", session.request_id, error=str(exc))
        try:
            await websocket.send_json({
                "type": "error",
                "request_id": session.request_id,
                "message": "Stream connection encountered an error."
            })
        except Exception:
            pass
    finally:
        for task in list(background_tasks):
            task.cancel()
        session.buffer.clear()
