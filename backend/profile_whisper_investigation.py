"""
Phase 8 Profiler: Investigate STT Latency on 9-second Utterance.
Utterance: "Hello, what is your name? What are you doing? What is your job?"
Tests:
1. Generate audio clip via Edge-TTS and inspect duration and sample count.
2. Measure single-pass full transcribe() latency (as in REST /pipeline).
3. Measure streaming session with 250ms chunks (as in WebSocket /ws/stream).
"""

import os
import sys
import time
import io
import asyncio
import wave
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.stt_service import stt_service, convert_to_clean_wav
from backend.core.streaming_orchestrator import streaming_orchestrator
from backend.test_stt_benchmark import generate_speech_wav, inspect_pcm16_wav

async def main():
    text = "Hello, what is your name? What are you doing? What is your job?"
    print(f"[*] Generating audio for text: '{text}'")
    wav_bytes = await generate_speech_wav(text, "en-US-JennyNeural")
    diag = inspect_pcm16_wav(wav_bytes)
    duration_s = diag["duration_s"]
    print(f"[+] Audio generated: {len(wav_bytes)} bytes, duration: {duration_s:.2f}s, sample_rate: {diag['sample_rate_hz']}")

    # --- Test 1: Single-pass transcribe() (REST /pipeline path) ---
    print("\n" + "="*60)
    print("TEST 1: Direct Single-Pass STT (REST /pipeline path)")
    print("="*60)
    t0 = time.perf_counter()
    transcript, info, post_diag, engine, model = stt_service.transcribe(
        audio_bytes=wav_bytes,
        lang="en",
        filename="test.wav",
        content_type="audio/wav"
    )
    t_stt = time.perf_counter() - t0
    print(f"[Result] Transcript: '{transcript}'")
    print(f"[Result] Single-pass STT Latency: {t_stt:.3f}s")
    print(f"[Result] Real-Time Factor (RTF = STT / Audio): {t_stt / duration_s:.2f}x")

    # --- Test 2: Streaming Simulation (WebSocket /ws/stream path) ---
    print("\n" + "="*60)
    print("TEST 2: Streaming WebSocket Simulation (250ms chunks)")
    print("="*60)
    session = streaming_orchestrator.create_session(
        session_id="prof_stream_test",
        language="en",
        target_lang="hi",
        vad_enabled=False,
        hypothesis_enabled=True
    )

    chunk_size = 8000 # 250ms of 16kHz 16-bit mono PCM
    whisper_calls = []
    stream_t0 = time.perf_counter()

    # Feed chunk 1 with WAV header, subsequent chunks as 250ms PCM slices
    session.add_chunk(wav_bytes[:44 + chunk_size])

    chunk_idx = 1
    for offset in range(44 + chunk_size, len(wav_bytes), chunk_size):
        chunk_idx += 1
        raw_chunk = wav_bytes[offset : offset + chunk_size]
        session.add_chunk(raw_chunk)

        if session.should_trigger_partial():
            t_part_start = time.perf_counter()
            partial = await streaming_orchestrator.evaluate_partial(session)
            part_dur = time.perf_counter() - t_part_start
            buffer_dur_s = len(session.buffer) / 32000.0 # approx PCM duration
            whisper_calls.append({
                "type": "partial",
                "chunk_idx": chunk_idx,
                "audio_supplied_s": buffer_dur_s,
                "inference_s": part_dur,
                "text": partial
            })
            print(f"  [Partial #{len(whisper_calls)}] buffer={buffer_dur_s:.2f}s | time={part_dur:.3f}s | text='{partial}'")

    # Now finalize stream
    t_fin_start = time.perf_counter()
    fin_res = await streaming_orchestrator.finalize_stream(session)
    fin_dur = time.perf_counter() - t_fin_start
    stream_total_s = time.perf_counter() - stream_t0

    whisper_calls.append({
        "type": "final",
        "audio_supplied_s": len(session.buffer) / 32000.0,
        "inference_s": fin_res.get("metrics", {}).get("stt_ms", 0.0) / 1000.0,
        "text": fin_res.get("transcript")
    })

    print(f"\n[Final] Transcript: '{fin_res.get('transcript')}'")
    print(f"[Final] finalize_stream duration: {fin_dur:.3f}s")
    print(f"[Final] STT metric inside finalize: {fin_res.get('metrics', {}).get('stt_ms', 0.0)/1000.0:.3f}s")
    print(f"[Final] Total stream simulation wall time: {stream_total_s:.3f}s")

    # Summary of all calls
    total_audio_processed = sum(c["audio_supplied_s"] for c in whisper_calls)
    total_whisper_time = sum(c["inference_s"] for c in whisper_calls)
    print("\n" + "="*60)
    print("WHISPER INVOCATION BREAKDOWN")
    print("="*60)
    print(f"Total Whisper Invocations: {len(whisper_calls)}")
    for i, c in enumerate(whisper_calls, 1):
        print(f"  Call #{i:02d} [{c['type'].upper()}]: audio={c['audio_supplied_s']:.2f}s, inference={c['inference_s']:.3f}s, text='{c.get('text', '')}'")
    print(f"\nActual speech duration:           {duration_s:.2f}s")
    print(f"Total audio supplied to Whisper:  {total_audio_processed:.2f}s")
    print(f"Processing amplification:         {total_audio_processed / max(duration_s, 0.01):.2f}x")
    print(f"Total Whisper inference time:     {total_whisper_time:.3f}s")
    print(f"Cumulative time / speech duration:{total_whisper_time / max(duration_s, 0.01):.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
