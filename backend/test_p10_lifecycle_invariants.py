"""
P10 Lifecycle Invariants & Concurrency Regression Tests
=========================================================
Automated test suite verifying:
1. Single-Whisper-Inference Invariant: max(concurrent_whisper) == 1 per session.
   Partial and Final Whisper can NEVER execute simultaneously.
2. Short Utterance Lifecycle: Incomplete partial (<90%) drains cleanly, is discarded,
   and full-buffer authoritative Whisper executes with 0 concurrency.
3. 90% Reuse Invariant: Partials with >=90% coverage are reused safely with 0ms STT latency.
4. Stale Result / Cross-Turn Protection: Results from Turn 1 cannot contaminate Turn 2.
5. WebSocket End-to-End: Full streaming session over WebSocket terminates cleanly with max_concurrent == 1.
"""

import os
import sys
import io
import time
import wave
import asyncio
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.server import app
from backend.services.stt_service import stt_service
from backend.services.whisper_profiler import whisper_profiler
from backend.core.streaming_orchestrator import streaming_orchestrator, StreamingSession

SAMPLE_DIR = Path("backend/debug_audio/benchmark_samples_p8")
SHORT_EN = SAMPLE_DIR / "EN-S1.wav"   # 2.28s
SHORT_EN2 = SAMPLE_DIR / "EN-S2.wav"  # 2.11s
SHORT_HI = SAMPLE_DIR / "HI-1.wav"    # 3.00s
TECH_EN = SAMPLE_DIR / "TECH-1.wav"   # 4.63s


@pytest.fixture(autouse=True)
def reset_profiler():
    import gc
    gc.collect()
    whisper_profiler.reset_all()
    yield
    whisper_profiler.reset_all()
    gc.collect()


def test_single_whisper_invariant_under_concurrent_trigger():
    """
    PROVES INVARIANT: Even when finalize_stream is called while evaluate_partial
    is actively running on CPU, max_concurrent_whisper MUST NEVER exceed 1.
    """
    async def _run():
        wav_path = SHORT_EN if SHORT_EN.exists() else Path("backend/debug_audio/benchmark_samples/EN-1.wav")
        assert wav_path.exists(), f"Missing test audio: {wav_path}"
        audio_bytes = wav_path.read_bytes()

        sid = "test_invariant_session"
        session = streaming_orchestrator.create_session(
            session_id=sid,
            request_id="req_inv_1",
            language="en",
            hypothesis_enabled=False
        )
        # Add enough audio for partial
        session.add_chunk(audio_bytes[:48000])

        # Launch evaluate_partial as background task
        partial_task = asyncio.create_task(streaming_orchestrator.evaluate_partial(session))
        session.active_partial_task = partial_task

        # Small sleep to ensure partial enters inference lock
        await asyncio.sleep(0.01)

        # Immediately call finalize_stream while partial is running
        final_res = await streaming_orchestrator.finalize_stream(session)

        # Await partial task to complete cleanly
        await partial_task

        # Invariant assertion
        max_concurrent = whisper_profiler.get_max_concurrent(sid)
        assert max_concurrent == 1, (
            f"VIOLATION OF SINGLE-WHISPER INVARIANT: max_concurrent was {max_concurrent}, expected <= 1"
        )
        assert final_res["type"] == "final"
        assert len(final_res.get("transcript", "")) > 0

    asyncio.run(_run())


def test_short_utterance_no_truncation_no_concurrency():
    """
    CRITICAL P10 REGRESSION CASE: Short utterance (~2.2s).
    A partial representing ~1.0s must NOT be reused as final.
    Full buffer must be transcribed with max_concurrent == 1.
    """
    async def _run():
        wav_path = SHORT_EN2 if SHORT_EN2.exists() else SHORT_EN
        assert wav_path.exists()
        audio_bytes = wav_path.read_bytes()

        sid = "test_short_utterance_session"
        session = streaming_orchestrator.create_session(
            session_id=sid,
            request_id="req_short_1",
            language="en",
            hypothesis_enabled=False
        )

        # Feed 1.0s (32000 bytes) -> triggers partial
        session.add_chunk(audio_bytes[:44 + 32000])
        partial_text = await streaming_orchestrator.evaluate_partial(session)

        # Feed remaining audio (~1.1s)
        session.add_chunk(audio_bytes[44 + 32000:])

        # Finalize
        final_res = await streaming_orchestrator.finalize_stream(session)

        max_concurrent = whisper_profiler.get_max_concurrent(sid)
        assert max_concurrent <= 1, f"Concurrency violation: {max_concurrent}"

        transcript = final_res.get("transcript", "")
        assert len(transcript) > 0

        # Ensure no truncation: the partial was < 90% (1.0s of 2.1s = 47.6%)
        # So fast_path must NOT have been used
        assert final_res.get("metrics", {}).get("stt_ms", 0) > 100, (
            "Short utterance should execute authoritative full-buffer Whisper, not reuse incomplete partial."
        )

    asyncio.run(_run())


def test_smart_finalization_reused_when_coverage_ge_90():
    """
    Smart finalization must activate when completed partial covers >= 90%
    of the turn audio, achieving 0ms additional STT latency with max_concurrent == 1.
    """
    async def _run():
        wav_path = SHORT_EN if SHORT_EN.exists() else TECH_EN
        assert wav_path.exists()
        audio_bytes = wav_path.read_bytes()

        sid = "test_smart_final_session"
        session = streaming_orchestrator.create_session(
            session_id=sid,
            request_id="req_smart_1",
            language="en",
            hypothesis_enabled=False
        )

        # Feed 95% of audio and run partial
        cutoff = int(len(audio_bytes) * 0.95)
        session.add_chunk(audio_bytes[:cutoff])
        partial_text = await streaming_orchestrator.evaluate_partial(session)
        assert partial_text is not None and len(partial_text) > 0

        # Feed remaining 5% of audio
        session.add_chunk(audio_bytes[cutoff:])

        # Finalize
        final_res = await streaming_orchestrator.finalize_stream(session)

        max_concurrent = whisper_profiler.get_max_concurrent(sid)
        assert max_concurrent <= 1, f"Concurrency violation: {max_concurrent}"

        # Fast path verification
        assert final_res.get("metrics", {}).get("stt_ms", 999) < 50.0, (
            f"Expected fast path reuse (<50ms STT), got {final_res.get('metrics', {}).get('stt_ms')}ms"
        )
        assert final_res.get("transcript") == partial_text

    asyncio.run(_run())


def test_cross_turn_isolation_no_stale_contamination():
    """
    Guarantees that a completed partial from Turn 1 cannot be reused
    or contaminate Turn 2.
    """
    async def _run():
        wav_path = SHORT_EN if SHORT_EN.exists() else TECH_EN
        assert wav_path.exists()
        audio_bytes = wav_path.read_bytes()

        sid = "test_cross_turn_session"
        session = streaming_orchestrator.create_session(
            session_id=sid,
            request_id="turn_1",
            language="en"
        )

        # Turn 1: feed audio, evaluate partial, finalize
        session.add_chunk(audio_bytes)
        p1 = await streaming_orchestrator.evaluate_partial(session)
        r1 = await streaming_orchestrator.finalize_stream(session)
        assert r1["request_id"] == "turn_1"

        # Reset for Turn 2
        session.reset_for_next_turn(new_request_id="turn_2")
        assert session.request_id == "turn_2"
        assert session.is_finalizing is False
        assert session.last_partial_turn_id == ""
        assert session.last_partial_text == ""
        assert session.last_partial_bytes_len == 0

        # Turn 2: feed only 500ms of audio (silence / short burst)
        session.add_chunk(audio_bytes[:16000])
        r2 = await streaming_orchestrator.finalize_stream(session)
        assert r2["request_id"] == "turn_2"
        # Ensure Turn 1 transcript did not leak as fast-path in Turn 2
        if r2.get("transcript"):
            assert r2.get("metrics", {}).get("stt_ms", 0) > 50.0, "Turn 2 must not fast-path reuse Turn 1!"

    asyncio.run(_run())


def test_websocket_streaming_e2e_max_concurrent_invariant():
    """
    End-to-End WebSocket Test: Streams audio frames and sends 'end'.
    Asserts that max_concurrent_whisper == 1 over the full lifecycle.
    """
    wav_path = SHORT_EN if SHORT_EN.exists() else TECH_EN
    assert wav_path.exists()
    audio_bytes = wav_path.read_bytes()

    client = TestClient(app)
    sid = "test_ws_e2e_session"

    mock_trans = {
        "translated_text": "Mock translation",
        "source_lang": "en",
        "target_lang": "hi",
        "provider": "Mock",
        "fallback_used": False,
        "fallback_reason": None,
        "context_used": False,
        "retrieved_context": []
    }
    with patch.object(streaming_orchestrator.llm, "translate", return_value=mock_trans):
        with client.websocket_connect(f"/ws/transcribe?session_id={sid}&lang=en&target_lang=hi") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"

            # Stream chunks (250ms = 8000 bytes)
            CHUNK = 8000
            for offset in range(0, len(audio_bytes), CHUNK):
                chunk = audio_bytes[offset:offset + CHUNK]
                ws.send_bytes(chunk)
                time.sleep(0.01)

            # Send 'end' control message
            ws.send_json({"type": "end"})

            # Collect messages until 'final' or 'error'
            final_msg = None
            for _ in range(50):
                try:
                    msg = ws.receive_json()
                    if msg.get("type") in ("final", "error"):
                        final_msg = msg
                        break
                except Exception:
                    break

            assert final_msg is not None, "Did not receive final message from websocket endpoint"
            if final_msg.get("type") == "final":
                assert "transcript" in final_msg

    # Verify concurrency invariant on server
    max_concurrent = whisper_profiler.get_max_concurrent(sid)
    assert max_concurrent == 1, (
        f"E2E Concurrency violation: max_concurrent was {max_concurrent}, expected 1"
    )
