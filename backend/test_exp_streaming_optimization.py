"""
Experiment 2 & 3 (EXP-P8-2 & EXP-P8-3): Streaming STT Optimization A/B Benchmark.
Compares:
Arm A: P7 Baseline (24KB partial interval, growing buffer 0..T, full re-decode on finalize)
Arm B: Optimized Cadence (40KB partial interval, strict non-overlapping mutex)
Arm C: Optimized Cadence + Sliding Tail Window (max 4.0s buffer for partials)
Arm D: Optimized Cadence + Smart Finalize (if last partial covers >=95% buffer, reuse stable text)

Evaluates on the 30 fixed benchmark recordings:
- Total Whisper calls per turn
- Total audio processed & amplification factor
- Mean first useful transcript latency
- Mean final transcription latency
- Average WER & exact matches
"""

import os
import sys
import json
import time
import io
import wave
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.stt_service import stt_service, inspect_pcm16_wav
from backend.services.whisper_profiler import whisper_profiler
from backend.core.streaming_orchestrator import streaming_orchestrator, StreamingSession
from backend.services.hypothesis_service import hypothesis_service
from backend.test_stt_benchmark import calculate_wer

DATASET_DIR = Path("backend/debug_audio/benchmark_samples_p8")
MANIFEST_FILE = DATASET_DIR / "dataset_manifest.json"

async def run_streaming_arm(manifest, arm_name: str, interval_bytes: int, max_partial_window_bytes: int, smart_finalize: bool):
    print("\n" + "="*70)
    print(f"RUNNING STREAMING ARM: {arm_name}")
    print(f"  interval_bytes={interval_bytes}, max_window={max_partial_window_bytes}, smart_finalize={smart_finalize}")
    print("="*70)

    whisper_profiler.reset_all()
    results = []

    for i, item in enumerate(manifest, 1):
        wav_path = DATASET_DIR / item["wav_file"]
        wav_bytes = wav_path.read_bytes()
        req_id = f"{arm_name}_{item['id']}"

        session = streaming_orchestrator.create_session(
            session_id=req_id,
            request_id=req_id,
            language=item["lang"],
            target_lang="hi",
            vad_enabled=False,
            hypothesis_enabled=True
        )

        header = wav_bytes[:44]
        chunk_size = 8000 # 250ms chunks

        first_useful_ts = None
        first_useful_text = ""
        last_partial_text = ""
        last_partial_bytes = 0
        is_busy = False

        stream_t0 = time.perf_counter()

        # Push header + first chunk
        session.add_chunk(wav_bytes[:44 + chunk_size])

        for offset in range(44 + chunk_size, len(wav_bytes), chunk_size):
            chunk = wav_bytes[offset : offset + chunk_size]
            session.add_chunk(chunk)

            cur_len = len(session.buffer)
            # Custom interval check for this arm
            if not is_busy and cur_len >= 16000 and (cur_len - last_partial_bytes) >= interval_bytes:
                is_busy = True
                last_partial_bytes = cur_len

                # Determine audio slice to decode
                if max_partial_window_bytes > 0 and cur_len > (44 + max_partial_window_bytes):
                    # Slice trailing window with valid WAV header
                    tail_pcm = session.buffer[-max_partial_window_bytes:]
                    tail_buf = io.BytesIO()
                    with wave.open(tail_buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(tail_pcm)
                    partial_snapshot = tail_buf.getvalue()
                else:
                    partial_snapshot = bytes(session.buffer)

                t_p_start = time.perf_counter()
                try:
                    partial = await asyncio.to_thread(
                        stt_service.transcribe_partial,
                        audio_bytes=partial_snapshot,
                        lang=session.language,
                        session_id=session.session_id,
                        request_id=session.request_id,
                        reason=f"partial_{arm_name}"
                    )
                except Exception:
                    partial = ""
                finally:
                    is_busy = False

                if partial:
                    clean_p = partial.strip()
                    if clean_p != last_partial_text:
                        last_partial_text = clean_p
                        if not first_useful_text:
                            first_useful_text = clean_p
                            first_useful_ts = time.perf_counter() - stream_t0
                        if session.hypothesis_enabled:
                            session.last_stabilized_result = hypothesis_service.process_hypothesis(clean_p, session.hypothesis_state)

        # Finalize turn
        t_fin_start = time.perf_counter()
        
        # Check Smart Finalize condition:
        # If last partial was evaluated on >= 90% of total audio and we have stabilized text
        buffer_len = len(session.buffer)
        can_fast_path = (
            smart_finalize
            and last_partial_bytes >= (buffer_len * 0.90)
            and bool(last_partial_text)
            and len(last_partial_text) > 3
        )

        if can_fast_path:
            # Fast-path finalization using stabilized hypothesis
            final_transcript = last_partial_text
            final_stt_s = time.perf_counter() - t_fin_start
            # Record fast-path call in profiler
            whisper_profiler.record_call(
                session_id=session.session_id,
                request_id=session.request_id,
                reason="smart_finalize_fast_path",
                call_type="final",
                audio_duration_s=0.0,
                sample_count=0,
                inference_duration_s=final_stt_s,
                text=final_transcript,
                accepted=True,
                hypothesis_applied=True,
                extra={"fast_path": True}
            )
        else:
            # Authoritative full pass
            audio_snapshot = bytes(session.buffer)
            transcript, _, _, _, _ = await asyncio.to_thread(
                stt_service.transcribe,
                audio_bytes=audio_snapshot,
                lang=session.language,
                filename="stream.wav",
                content_type="audio/wav",
                session_id=session.session_id,
                request_id=session.request_id,
                reason=f"final_{arm_name}"
            )
            final_transcript = transcript
            final_stt_s = time.perf_counter() - t_fin_start

        wer_val, _, _ = calculate_wer(item["reference_text"], final_transcript)
        exact = (wer_val == 0.0)

        turn_prof = whisper_profiler.get_turn_summary(req_id, actual_duration_s=item["duration_s"])

        results.append({
            "id": item["id"],
            "calls": turn_prof.call_count,
            "amplification": turn_prof.processing_amplification,
            "first_useful_s": round(first_useful_ts, 3) if first_useful_ts else None,
            "final_stt_s": round(final_stt_s, 3),
            "wer": wer_val,
            "exact": exact,
            "transcript": final_transcript
        })

    total_calls = sum(r["calls"] for r in results)
    avg_amp = sum(r["amplification"] for r in results) / len(results)
    avg_first_useful = sum(r["first_useful_s"] for r in results if r["first_useful_s"]) / max(1, sum(1 for r in results if r["first_useful_s"]))
    avg_final_stt = sum(r["final_stt_s"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)
    exact_count = sum(1 for r in results if r["exact"])

    print(f"\n[{arm_name} RESULTS SUMMARY]")
    print(f"  Total Whisper Calls:          {total_calls} (mean: {total_calls/len(results):.2f}/turn)")
    print(f"  Mean Audio Amplification:     {avg_amp:.2f}x")
    print(f"  Mean First Useful Transcript: {avg_first_useful:.3f}s")
    print(f"  Mean Final STT Latency:       {avg_final_stt:.3f}s")
    print(f"  Average WER:                  {avg_wer:.4f}")
    print(f"  Exact Matches:                {exact_count} / {len(results)} ({exact_count/len(results)*100:.1f}%)")

    return {
        "arm": arm_name,
        "total_calls": total_calls,
        "mean_calls": round(total_calls/len(results), 2),
        "mean_amp": round(avg_amp, 2),
        "mean_first_useful_s": round(avg_first_useful, 3),
        "mean_final_stt_s": round(avg_final_stt, 3),
        "average_wer": round(avg_wer, 4),
        "exact_matches": exact_count,
        "exact_pct": round(exact_count/len(results)*100, 1),
        "results": results
    }

async def main():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Arm A: P7 Baseline (24KB = 0.75s interval, growing window, full final pass)
    arm_a = await run_streaming_arm(manifest, "Arm_A_P7_Baseline", interval_bytes=24000, max_partial_window_bytes=0, smart_finalize=False)

    # Arm B: Optimized Cadence (40KB = 1.25s interval, growing window, full final pass)
    arm_b = await run_streaming_arm(manifest, "Arm_B_Opt_Cadence", interval_bytes=40000, max_partial_window_bytes=0, smart_finalize=False)

    # Arm C: Optimized Cadence + Smart Finalize
    arm_c = await run_streaming_arm(manifest, "Arm_C_Opt_Smart_Finalize", interval_bytes=40000, max_partial_window_bytes=0, smart_finalize=True)

    # Save comparative report
    out_file = Path("backend/benchmark_results/p8_streaming_ab_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"arms": [arm_a, arm_b, arm_c]}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved comparative streaming A/B report to {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
