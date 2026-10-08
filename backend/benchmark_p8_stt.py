"""
Phase 8 STT Benchmark & Profiler Harness (P8.5).
Evaluates the 30 fixed audio recordings under:
1. Single-pass REST STT
2. Streaming chunked STT with Local Agreement Hypothesis Stabilization

Calculates:
- WER, exact match, transcript accuracy
- First partial latency (TTFR / first useful transcript)
- Final transcript latency
- Whisper call count & invocation reasons
- Total audio processed & processing amplification factor
- Pure Whisper inference time vs wall time
"""

import os
import sys
import io
import json
import time
import math
import wave
try:
    import psutil
    def get_ram_mb():
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
except ImportError:
    def get_ram_mb():
        return 0.0

import asyncio
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.stt_service import stt_service, inspect_pcm16_wav
from backend.services.whisper_profiler import whisper_profiler
from backend.core.streaming_orchestrator import streaming_orchestrator, StreamingSession
from backend.test_stt_benchmark import calculate_wer

DATASET_DIR = Path("backend/debug_audio/benchmark_samples_p8")
MANIFEST_FILE = DATASET_DIR / "dataset_manifest.json"
RESULTS_DIR = Path("backend/benchmark_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

async def run_single_pass_benchmark(manifest: List[Dict[str, Any]], config_name: str) -> Dict[str, Any]:
    print(f"\n" + "="*70)
    print(f"RUNNING SINGLE-PASS BENCHMARK [{config_name}]")
    print(f"="*70)

    results = []
    whisper_profiler.reset_all()
    ram_start_mb = get_ram_mb()

    t_bench_start = time.perf_counter()

    for i, item in enumerate(manifest, 1):
        wav_path = DATASET_DIR / item["wav_file"]
        wav_bytes = wav_path.read_bytes()
        req_id = f"sp_{item['id']}"

        t0 = time.perf_counter()
        transcript, info, post_diag, engine_used, model_used = stt_service.transcribe(
            audio_bytes=wav_bytes,
            lang=item["lang"],
            filename=item["wav_file"],
            content_type="audio/wav",
            session_id=req_id,
            request_id=req_id,
            reason="single_pass_benchmark"
        )
        latency_s = time.perf_counter() - t0

        wer_val, edits, ref_len = calculate_wer(item["reference_text"], transcript)
        exact = (wer_val == 0.0)

        turn_prof = whisper_profiler.get_turn_summary(req_id, actual_duration_s=item["duration_s"])

        res_item = {
            "id": item["id"],
            "category": item["category"],
            "lang": item["lang"],
            "reference": item["reference_text"],
            "transcript": transcript,
            "duration_s": item["duration_s"],
            "latency_s": round(latency_s, 3),
            "wer": wer_val,
            "exact": exact,
            "whisper_calls": turn_prof.call_count,
            "audio_processed_s": turn_prof.total_audio_processed_s,
            "amplification": turn_prof.processing_amplification,
            "pure_inference_s": turn_prof.total_inference_time_s,
            "rtf": round(latency_s / max(item["duration_s"], 0.01), 3)
        }
        results.append(res_item)
        print(f"[{i:02d}/30] {item['id']} ({item['category']}): dur={item['duration_s']:.2f}s | lat={latency_s:.3f}s (RTF={res_item['rtf']}x) | WER={wer_val:.4f} | exact={exact}")
        print(f"       Ref: '{item['reference_text'][:60]}'")
        print(f"       Hyp: '{transcript[:60]}'")

    ram_end_mb = get_ram_mb()
    total_dur_s = sum(r["duration_s"] for r in results)
    total_lat_s = sum(r["latency_s"] for r in results)
    avg_wer = sum(r["wer"] for r in results) / len(results)
    exact_count = sum(1 for r in results if r["exact"])

    summary = {
        "benchmark": "P8_SinglePass",
        "config_name": config_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_clips": len(results),
        "total_audio_s": round(total_dur_s, 2),
        "total_latency_s": round(total_lat_s, 2),
        "mean_latency_s": round(total_lat_s / len(results), 3),
        "median_latency_s": round(sorted(r["latency_s"] for r in results)[len(results)//2], 3),
        "p95_latency_s": round(sorted(r["latency_s"] for r in results)[int(len(results)*0.95)], 3),
        "average_wer": round(avg_wer, 4),
        "exact_matches": exact_count,
        "exact_match_pct": round(exact_count / len(results) * 100, 2),
        "ram_start_mb": round(ram_start_mb, 2),
        "ram_end_mb": round(ram_end_mb, 2),
        "ram_delta_mb": round(ram_end_mb - ram_start_mb, 2),
        "results": results
    }
    return summary

async def run_streaming_simulation_benchmark(manifest: List[Dict[str, Any]], config_name: str) -> Dict[str, Any]:
    print(f"\n" + "="*70)
    print(f"RUNNING STREAMING CHUNKED BENCHMARK [{config_name}]")
    print(f"="*70)

    results = []
    whisper_profiler.reset_all()
    ram_start_mb = get_ram_mb()

    for i, item in enumerate(manifest, 1):
        wav_path = DATASET_DIR / item["wav_file"]
        wav_bytes = wav_path.read_bytes()
        req_id = f"st_{item['id']}"

        # Initialize session
        session = streaming_orchestrator.create_session(
            session_id=req_id,
            request_id=req_id,
            language=item["lang"],
            target_lang="hi",
            vad_enabled=False,
            hypothesis_enabled=True
        )

        header = wav_bytes[:44]
        pcm_data = wav_bytes[44:]
        chunk_size = 8000 # 250ms of 16kHz 16-bit mono PCM

        first_useful_ts = None
        first_useful_text = ""
        partial_count = 0
        stream_t0 = time.perf_counter()

        # Simulate progressive arrival of chunks starting with header
        # Chunk 1 includes header + 250ms of audio, subsequent chunks are 250ms audio slices
        session.add_chunk(wav_bytes[:44 + chunk_size])
        
        for offset in range(44 + chunk_size, len(wav_bytes), chunk_size):
            chunk = wav_bytes[offset : offset + chunk_size]
            session.add_chunk(chunk)

            if session.should_trigger_partial():
                partial = await streaming_orchestrator.evaluate_partial(session)
                if partial and not first_useful_text:
                    first_useful_text = partial
                    first_useful_ts = time.perf_counter() - stream_t0
                if partial:
                    partial_count += 1

        # Simulate user releasing mic / clicking stop
        # In real streaming, the complete buffer with header is finalized
        # Note: if session.buffer lacks WAV header, synthesize header for clean conversion
        if not session.buffer.startswith(b"RIFF"):
            full_wav_buf = io.BytesIO()
            with wave.open(full_wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(bytes(session.buffer))
            session.buffer = bytearray(full_wav_buf.getvalue())

        t_fin_start = time.perf_counter()
        final_res = await streaming_orchestrator.finalize_stream(session)
        final_transcription_s = time.perf_counter() - t_fin_start
        total_stream_wall_s = time.perf_counter() - stream_t0

        transcript = final_res.get("transcript", "")
        wer_val, edits, ref_len = calculate_wer(item["reference_text"], transcript)
        exact = (wer_val == 0.0)

        turn_prof = whisper_profiler.get_turn_summary(req_id, actual_duration_s=item["duration_s"])

        res_item = {
            "id": item["id"],
            "category": item["category"],
            "lang": item["lang"],
            "reference": item["reference_text"],
            "transcript": transcript,
            "duration_s": item["duration_s"],
            "first_useful_transcript_s": round(first_useful_ts, 3) if first_useful_ts else None,
            "first_useful_text": first_useful_text,
            "final_transcription_latency_s": round(final_transcription_s, 3),
            "total_stream_wall_s": round(total_stream_wall_s, 3),
            "wer": wer_val,
            "exact": exact,
            "whisper_calls": turn_prof.call_count,
            "audio_processed_s": turn_prof.total_audio_processed_s,
            "amplification": turn_prof.processing_amplification,
            "pure_inference_s": turn_prof.total_inference_time_s
        }
        results.append(res_item)
        print(f"[{i:02d}/30] {item['id']} ({item['category']}): dur={item['duration_s']:.2f}s | calls={turn_prof.call_count} (amp={turn_prof.processing_amplification}x) | first_useful={res_item['first_useful_transcript_s']}s | fin_lat={final_transcription_s:.3f}s | WER={wer_val:.4f}")

    ram_end_mb = get_ram_mb()
    total_calls = sum(r["whisper_calls"] for r in results)
    avg_amp = sum(r["amplification"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)
    exact_count = sum(1 for r in results if r["exact"])

    summary = {
        "benchmark": "P8_StreamingSimulation",
        "config_name": config_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_clips": len(results),
        "total_calls": total_calls,
        "mean_calls_per_turn": round(total_calls / len(results), 2),
        "mean_amplification": round(avg_amp, 2),
        "mean_first_useful_s": round(sum(r["first_useful_transcript_s"] for r in results if r["first_useful_transcript_s"]) / max(1, sum(1 for r in results if r["first_useful_transcript_s"])), 3),
        "mean_final_transcription_s": round(sum(r["final_transcription_latency_s"] for r in results) / len(results), 3),
        "average_wer": round(avg_wer, 4),
        "exact_matches": exact_count,
        "exact_match_pct": round(exact_count / len(results) * 100, 2),
        "ram_start_mb": round(ram_start_mb, 2),
        "ram_end_mb": round(ram_end_mb, 2),
        "ram_delta_mb": round(ram_end_mb - ram_start_mb, 2),
        "results": results
    }
    return summary

async def main():
    if not MANIFEST_FILE.exists():
        print(f"[!] Manifest not found at {MANIFEST_FILE}")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Run Single-Pass Benchmark (P7 Frozen Baseline)
    sp_summary = await run_single_pass_benchmark(manifest, config_name="P7_Frozen_SinglePass")
    out_sp = RESULTS_DIR / "p8_single_pass_baseline.json"
    with open(out_sp, "w", encoding="utf-8") as f:
        json.dump(sp_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved Single-Pass Baseline to {out_sp}")

    # 2. Run Streaming Benchmark (P7 Frozen Baseline)
    st_summary = await run_streaming_simulation_benchmark(manifest, config_name="P7_Frozen_Streaming")
    out_st = RESULTS_DIR / "p8_streaming_baseline.json"
    with open(out_st, "w", encoding="utf-8") as f:
        json.dump(st_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved Streaming Baseline to {out_st}")

if __name__ == "__main__":
    asyncio.run(main())
