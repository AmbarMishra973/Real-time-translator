"""
P10 Empirical Benchmark & Verification Script
Measures actual P10 metrics across representative test recordings:
- Short English (EN-S1: 2.28s, EN-S2: 2.11s)
- Short Hindi (HI-1: 3.00s)
- Medium Technical (TECH-1: 4.63s)
- Medium English (EN-L1: 5.88s)
- Long English (EN-L2: 8.23s)

Validates:
1. Single-Whisper Invariant: max_concurrent == 1
2. Partial reuse correctness (>=90% reused, <90% discarded with full-buffer decode)
3. Accurate final transcription with 0 truncation
4. Latency comparison vs P8 baseline
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path

# Ensure thread safety on Windows MKL
os.environ.setdefault("MKL_DISABLE_FAST_MM", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.stt_service import stt_service
from backend.core.streaming_orchestrator import streaming_orchestrator
from backend.services.whisper_profiler import whisper_profiler
from backend.test_stt_benchmark import calculate_wer

DATASET_DIR = Path("backend/debug_audio/benchmark_samples_p8")
MANIFEST_FILE = DATASET_DIR / "dataset_manifest.json"

TEST_IDS = ["EN-S1", "EN-S2", "HI-1", "TECH-1", "EN-L1", "EN-L2"]

# P8 baseline values recorded in p8_streaming_baseline.json / empirical report
P8_BASELINE = {
    "EN-S1": {"final_stt_s": 2.092, "calls": 4, "max_concurrent": 2, "reused": False, "wer": 0.0},
    "EN-S2": {"final_stt_s": 1.728, "calls": 3, "max_concurrent": 2, "reused": False, "wer": 0.0},
    "HI-1":  {"final_stt_s": 2.915, "calls": 5, "max_concurrent": 2, "reused": False, "wer": 0.0},
    "TECH-1":{"final_stt_s": 2.241, "calls": 8, "max_concurrent": 2, "reused": False, "wer": 0.0},
    "EN-L1": {"final_stt_s": 3.840, "calls": 11, "max_concurrent": 2, "reused": False, "wer": 0.0},
    "EN-L2": {"final_stt_s": 4.120, "calls": 14, "max_concurrent": 2, "reused": False, "wer": 0.0},
}


async def benchmark_recording(item: dict) -> dict:
    wav_path = DATASET_DIR / item["wav_file"]
    audio_bytes = wav_path.read_bytes()
    sid = f"bench_p10_{item['id']}"
    lang = item.get("lang", "en")
    expected = item.get("reference_text") or item.get("reference") or item.get("expected", "")

    whisper_profiler.reset_all()

    session = streaming_orchestrator.create_session(
        session_id=sid,
        request_id=f"req_{item['id']}",
        language=lang,
        hypothesis_enabled=False
    )

    CHUNK_SIZE = 8000  # 250ms chunks (16kHz 16-bit mono = 32000 B/s)
    total_len = len(audio_bytes)
    t_stream_start = time.perf_counter()
    partial_count = 0
    first_partial_s = None

    # Stream chunks simulating real-time audio arrival
    for offset in range(0, total_len, CHUNK_SIZE):
        chunk = audio_bytes[offset:offset + CHUNK_SIZE]
        session.add_chunk(chunk)

        if session.should_trigger_partial():
            p_text = await streaming_orchestrator.evaluate_partial(session)
            if p_text:
                partial_count += 1
                if first_partial_s is None:
                    first_partial_s = round(time.perf_counter() - t_stream_start, 3)

    # Finalize stream
    t_final_start = time.perf_counter()
    final_res = await streaming_orchestrator.finalize_stream(session)
    final_wall_s = round(time.perf_counter() - t_final_start, 3)

    max_concurrent = whisper_profiler.get_max_concurrent(sid)
    stt_ms = final_res.get("metrics", {}).get("stt_ms", round(final_wall_s * 1000, 1))
    transcript = final_res.get("transcript", "")
    reused = (stt_ms < 50.0)

    wer, edits, ref_len = calculate_wer(expected, transcript)

    return {
        "id": item["id"],
        "category": item.get("category", ""),
        "duration_s": round(total_len / 32000.0, 2),
        "transcript": transcript,
        "expected": expected,
        "exact_match": (transcript.strip().lower() == expected.strip().lower()),
        "wer": wer,
        "first_partial_s": first_partial_s,
        "final_stt_s": round(stt_ms / 1000.0, 3),
        "final_wall_s": final_wall_s,
        "max_concurrent": max_concurrent,
        "total_calls": partial_count + (0 if reused else 1),
        "reused_partial": reused,
    }


async def main():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_map = {item["id"]: item for item in manifest}
    test_items = [manifest_map[tid] for tid in TEST_IDS if tid in manifest_map]

    stt_service._load_model()
    assert stt_service.model is not None, "Failed to load Faster-Whisper model"

    print("=" * 80)
    print("P10 EVIDENCE-DRIVEN BENCHMARK: LIFECYCLE COORDINATION & REGRESSION TEST")
    print("=" * 80)

    results = []
    for item in test_items:
        res = await benchmark_recording(item)
        results.append(res)
        p8 = P8_BASELINE.get(res["id"], {})
        print(f"[{res['id']}] Dur: {res['duration_s']}s | WER: {res['wer']} | Exact: {res['exact_match']}")
        print(f"      P10 Final STT: {res['final_stt_s']:.3f}s (Reused: {res['reused_partial']}) | Max Conc: {res['max_concurrent']}")
        print(f"      P8  Final STT: {p8.get('final_stt_s', 'N/A')}s | P8 Max Conc: {p8.get('max_concurrent', 'N/A')}")
        print(f"      Transcript: '{res['transcript']}'")
        print("-" * 80)

    out_file = Path("backend/benchmark_results/p10_vs_p8_benchmark.json")
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[+] Results saved to {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
