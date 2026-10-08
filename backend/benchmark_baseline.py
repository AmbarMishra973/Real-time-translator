"""
Phase 0 Golden Baseline Benchmark
Freezes and measures the existing Real-Time AI Speech Translator pipeline
prior to any architectural or component modifications.

Non-destructive: Does NOT alter production code, routes, or configurations.
Measures:
  1. Streaming First Partial Latency (ms)
  2. Final STT Latency (ms)
  3. RAG Retrieval Latency (ms)
  4. Translation Latency (ms) [with provider and fallback tracking]
  5. TTS Synthesis Latency (ms)
  6. End-to-End Latency (ms)
  7. Cold-Start vs Warm-Run (N=5 repetitions)
  8. Host CPU, RAM, and Peak Process Memory
"""

import os
import sys
import io
import time
import json
import math
import wave
import array
import asyncio
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import edge_tts
from backend.services.stt_service import stt_service, convert_to_clean_wav
from backend.services.rag_service import rag_service
from backend.services.llm_service import llm_service
from backend.services.tts_service import tts_service
from backend.services.session_manager import session_manager
from backend.core.streaming_orchestrator import StreamingOrchestrator, StreamingSession


BENCHMARK_CASES = [
    {
        "id": "EN-1",
        "category": "English Conversational (Short)",
        "expected": "Hello",
        "lang": "en",
        "target_lang": "hi",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "EN-2",
        "category": "English Conversational",
        "expected": "What is your name?",
        "lang": "en",
        "target_lang": "hi",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "EN-3",
        "category": "English Conversational",
        "expected": "How are you doing today?",
        "lang": "en",
        "target_lang": "hi",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "HI-1",
        "category": "Hindi Conversational",
        "expected": "आपका नाम क्या है?",
        "lang": "hi",
        "target_lang": "en",
        "voice": "hi-IN-SwaraNeural"
    },
    {
        "id": "HI-2",
        "category": "Hindi Conversational",
        "expected": "आप कैसे हैं?",
        "lang": "hi",
        "target_lang": "en",
        "voice": "hi-IN-SwaraNeural"
    },
    {
        "id": "TECH-1",
        "category": "Technical English",
        "expected": "We need to implement a vector database with RAG.",
        "lang": "en",
        "target_lang": "hi",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "TECH-2",
        "category": "Technical English",
        "expected": "Kubernetes orchestration.",
        "lang": "en",
        "target_lang": "hi",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "NUM-1",
        "category": "Technical with Numbers",
        "expected": "The web server returned HTTP 404 error code.",
        "lang": "en",
        "target_lang": "hi",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "LONG-1",
        "category": "Longer Natural Utterance",
        "expected": "We deployed the backend service using Docker containers and Kubernetes.",
        "lang": "en",
        "target_lang": "hi",
        "voice": "en-US-JennyNeural"
    }
]


async def generate_and_cache_audio(item: Dict[str, Any], samples_dir: Path) -> bytes:
    """Loads existing cached benchmark WAV or generates using Edge-TTS if absent."""
    wav_path = samples_dir / f"{item['id']}.wav"
    if wav_path.exists():
        return wav_path.read_bytes()

    print(f"[*] Generating reference audio for {item['id']}: '{item['expected']}'...", flush=True)
    comm = edge_tts.Communicate(item["expected"], item["voice"])
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    raw_mp3 = buf.getvalue()

    # Convert to standard 16kHz mono PCM16 WAV via existing helper
    wav_bytes = convert_to_clean_wav(raw_mp3, suffix=".mp3")
    wav_path.write_bytes(wav_bytes)
    return wav_bytes


def compute_stats(values: List[float]) -> Dict[str, Any]:
    """Calculates n, mean, median/p50, p95, min, max, std from observations."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None, "std": None}

    clean_sorted = sorted(clean)
    mean_val = sum(clean_sorted) / n
    p50_val = clean_sorted[int(n * 0.50)]
    p95_idx = min(int(n * 0.95), n - 1)
    p95_val = clean_sorted[p95_idx]
    min_val = clean_sorted[0]
    max_val = clean_sorted[-1]
    variance = sum((x - mean_val) ** 2 for x in clean_sorted) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)

    return {
        "n": n,
        "mean": round(mean_val, 2),
        "p50": round(p50_val, 2),
        "p95": round(p95_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "std": round(std_val, 2)
    }


def get_system_environment() -> Dict[str, Any]:
    """Inspects host machine resources safely without adding new dependencies."""
    env = {
        "python_version": sys.version.split()[0],
        "os": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "total_ram_mb": None,
        "free_ram_mb": None,
        "peak_process_ram_mb": None
    }

    try:
        out = subprocess.check_output(
            ["powershell", "-Command", "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | ConvertTo-Json"],
            timeout=5
        ).decode()
        data = json.loads(out)
        env["total_ram_mb"] = round(data.get("TotalVisibleMemorySize", 0) / 1024, 1)
        env["free_ram_mb"] = round(data.get("FreePhysicalMemory", 0) / 1024, 1)
    except Exception:
        pass

    try:
        pid = os.getpid()
        out_proc = subprocess.check_output(
            ["powershell", "-Command", f"Get-Process -Id {pid} | Select-Object PeakWorkingSet64 | ConvertTo-Json"],
            timeout=5
        ).decode()
        data_proc = json.loads(out_proc)
        env["peak_process_ram_mb"] = round(data_proc.get("PeakWorkingSet64", 0) / (1024 * 1024), 2)
    except Exception:
        pass

    return env


async def run_streaming_simulation(
    orchestrator: StreamingOrchestrator,
    audio_bytes: bytes,
    lang: str,
    target_lang: str,
    chunk_size_ms: int = 250
) -> Dict[str, Any]:
    """
    Simulates real-world browser WebSocket chunked streaming (250ms chunks)
    against the existing StreamingOrchestrator without modifying it.
    Returns: { "first_partial_latency_ms", "first_partial_text", "chunks_sent" }
    """
    # 16kHz 16-bit mono PCM = 32,000 bytes per second -> 250ms = 8,000 bytes
    bytes_per_chunk = int(16000 * 2 * (chunk_size_ms / 1000.0))

    session = orchestrator.create_session(
        language=lang,
        target_lang=target_lang,
        domain="all"
    )

    first_partial_latency_ms = None
    first_partial_text = None
    chunks_sent = 0

    t_stream_start = time.perf_counter()

    for offset in range(0, len(audio_bytes), bytes_per_chunk):
        chunk = audio_bytes[offset:offset + bytes_per_chunk]
        chunks_sent += 1
        session.add_chunk(chunk)

        if session.should_trigger_partial():
            t_partial_eval_start = time.perf_counter()
            partial = await orchestrator.evaluate_partial(session)
            if partial and first_partial_latency_ms is None:
                first_partial_latency_ms = round((time.perf_counter() - t_stream_start) * 1000, 2)
                first_partial_text = partial

    return {
        "first_partial_latency_ms": first_partial_latency_ms,
        "first_partial_text": first_partial_text,
        "chunks_sent": chunks_sent
    }


async def execute_golden_baseline():
    print("=" * 70, flush=True)
    print("      PHASE 0: GOLDEN BASELINE BENCHMARK EXECUTION", flush=True)
    print("=" * 70, flush=True)

    samples_dir = Path("backend/debug_audio/benchmark_samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inspect Environment & Configuration
    sys_env = get_system_environment()
    config = {
        "stt_engine": os.getenv("STT_ENGINE", "local"),
        "whisper_size": getattr(stt_service, "model_size", "base"),
        "whisper_device": getattr(stt_service, "device", "cpu"),
        "whisper_compute_type": getattr(stt_service, "compute_type", "int8"),
        "default_llm_provider": os.getenv("DEFAULT_LLM_PROVIDER", "groq"),
        "groq_model": getattr(llm_service._translator, "groq_model", "llama-3.3-70b-versatile"),
        "has_groq_client": llm_service.has_groq_client,
        "tts_engine": "edge-tts",
        "tts_voices": {
            "en": "en-US-JennyNeural",
            "hi": "hi-IN-SwaraNeural"
        }
    }

    print(f"Environment: Python {sys_env['python_version']} | {sys_env['os']} | CPU: {sys_env['processor']}")
    print(f"System RAM: {sys_env['total_ram_mb']} MB (Free: {sys_env['free_ram_mb']} MB)")
    print(f"STT: Faster-Whisper ({config['whisper_size']}, {config['whisper_compute_type']} on {config['whisper_device']}) [LOCAL]")
    print(f"RAG: Scikit-learn TF-IDF ({len(rag_service.chunks)} chunks) [LOCAL]")
    print(f"LLM: {config['default_llm_provider']} (Groq configured: {config['has_groq_client']}) [NETWORK]")
    print(f"TTS: Edge-TTS Neural [NETWORK]\n", flush=True)

    # 2. Prepare Reference Audio Samples
    print("[*] Preparing and validating 9 reference audio clips...", flush=True)
    loaded_cases = []
    for case in BENCHMARK_CASES:
        audio_bytes = await generate_and_cache_audio(case, samples_dir)
        loaded_cases.append({**case, "audio_bytes": audio_bytes, "audio_len": len(audio_bytes)})
        print(f"  ✓ {case['id']} ({case['category']}): {len(audio_bytes)} bytes", flush=True)

    streaming_orch = StreamingOrchestrator(
        stt=stt_service,
        rag=rag_service,
        llm=llm_service,
        sessions=session_manager
    )

    # 3. Cold-Start Measurement (First Execution of Full Pipeline on EN-1)
    print("\n" + "-" * 70, flush=True)
    print("COLD-START MEASUREMENT (First invocation after initialization)", flush=True)
    print("-" * 70, flush=True)

    cold_case = loaded_cases[0]
    t_cold_start = time.perf_counter()

    t0 = time.perf_counter()
    cold_stream = await run_streaming_simulation(
        streaming_orch, cold_case["audio_bytes"], cold_case["lang"], cold_case["target_lang"]
    )
    t_cold_stream = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cold_stt_text, _, _, cold_stt_engine, _ = stt_service.transcribe(
        cold_case["audio_bytes"], lang=cold_case["lang"], filename="cold.wav"
    )
    t_cold_stt = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cold_rag = rag_service.retrieve(query=cold_stt_text, domain="all")
    t_cold_rag = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cold_trans = llm_service.translate(
        text=cold_stt_text,
        source_lang=cold_case["lang"],
        target_lang=cold_case["target_lang"],
        session_id="cold_sess"
    )
    t_cold_trans = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cold_audio, _ = await tts_service.synthesize(
        cold_trans.get("translated_text", "Hello"), target_lang=cold_case["target_lang"]
    )
    t_cold_tts = (time.perf_counter() - t0) * 1000
    t_cold_total = (time.perf_counter() - t_cold_start) * 1000

    cold_metrics = {
        "first_partial_latency_ms": cold_stream["first_partial_latency_ms"],
        "final_stt_ms": round(t_cold_stt, 2),
        "rag_ms": round(t_cold_rag, 2),
        "translation_ms": round(t_cold_trans, 2),
        "tts_ms": round(t_cold_tts, 2),
        "total_e2e_ms": round(t_cold_total, 2),
        "provider": cold_trans.get("provider", "unknown"),
        "fallback_used": cold_trans.get("fallback_used", False)
    }

    print(f"Cold First Partial:  {cold_metrics['first_partial_latency_ms']} ms")
    print(f"Cold Final STT:      {cold_metrics['final_stt_ms']} ms")
    print(f"Cold RAG:            {cold_metrics['rag_ms']} ms")
    print(f"Cold Translation:    {cold_metrics['translation_ms']} ms ({cold_metrics['provider']})")
    print(f"Cold TTS:            {cold_metrics['tts_ms']} ms")
    print(f"Cold End-to-End:     {cold_metrics['total_e2e_ms']} ms\n", flush=True)

    # 4. Warm-Run Measurements (N=5 Repetitions Across All 9 Test Cases)
    WARM_REPETITIONS = 5
    print("-" * 70, flush=True)
    print(f"WARM-RUN MEASUREMENTS (N={WARM_REPETITIONS} repetitions x {len(loaded_cases)} cases = {WARM_REPETITIONS * len(loaded_cases)} runs)", flush=True)
    print("-" * 70, flush=True)

    all_first_partial: List[float] = []
    all_final_stt: List[float] = []
    all_rag: List[float] = []
    all_translation: List[float] = []
    all_tts: List[float] = []
    all_e2e: List[float] = []

    case_results: List[Dict[str, Any]] = []

    for idx, case in enumerate(loaded_cases, 1):
        print(f"[{idx}/{len(loaded_cases)}] Benchmarking {case['id']} ('{case['expected']}') across {WARM_REPETITIONS} warm runs...", flush=True)
        c_fp: List[float] = []
        c_stt: List[float] = []
        c_rag: List[float] = []
        c_trans: List[float] = []
        c_tts: List[float] = []
        c_e2e: List[float] = []
        c_trans_provider = None
        c_fallback_used = False
        c_transcript = ""
        c_translated_text = ""

        for rep in range(WARM_REPETITIONS):
            t_run_start = time.perf_counter()

            # A. Streaming Partial Simulation
            stream_res = await run_streaming_simulation(
                streaming_orch, case["audio_bytes"], case["lang"], case["target_lang"]
            )
            fp_ms = stream_res["first_partial_latency_ms"]
            if fp_ms is not None:
                c_fp.append(fp_ms)
                all_first_partial.append(fp_ms)

            # B. Final STT
            t0 = time.perf_counter()
            transcript, _, _, _, _ = stt_service.transcribe(
                case["audio_bytes"], lang=case["lang"], filename=f"{case['id']}.wav"
            )
            stt_ms = (time.perf_counter() - t0) * 1000
            c_stt.append(stt_ms)
            all_final_stt.append(stt_ms)
            c_transcript = transcript

            # C. RAG Retrieval
            t0 = time.perf_counter()
            rag_service.retrieve(query=transcript, domain="all")
            rag_ms = (time.perf_counter() - t0) * 1000
            c_rag.append(rag_ms)
            all_rag.append(rag_ms)

            # D. Translation
            t0 = time.perf_counter()
            trans_res = llm_service.translate(
                text=transcript,
                source_lang=case["lang"],
                target_lang=case["target_lang"],
                session_id=f"bench_{case['id']}"
            )
            trans_ms = (time.perf_counter() - t0) * 1000
            c_trans.append(trans_ms)
            all_translation.append(trans_ms)
            c_trans_provider = trans_res.get("provider", "unknown")
            c_fallback_used = trans_res.get("fallback_used", False)
            c_translated_text = trans_res.get("translated_text", "")

            # E. TTS
            t0 = time.perf_counter()
            try:
                await tts_service.synthesize(text=c_translated_text or case["expected"], target_lang=case["target_lang"])
                tts_ms = (time.perf_counter() - t0) * 1000
            except Exception:
                tts_ms = 0.0
            c_tts.append(tts_ms)
            all_tts.append(tts_ms)

            # F. Total E2E
            total_e2e_ms = (time.perf_counter() - t_run_start) * 1000
            c_e2e.append(total_e2e_ms)
            all_e2e.append(total_e2e_ms)

        case_entry = {
            "id": case["id"],
            "category": case["category"],
            "expected": case["expected"],
            "transcript_produced": c_transcript,
            "translation_produced": c_translated_text,
            "provider": c_trans_provider,
            "fallback_used": c_fallback_used,
            "metrics": {
                "first_partial_latency_ms": compute_stats(c_fp),
                "final_stt_latency_ms": compute_stats(c_stt),
                "rag_latency_ms": compute_stats(c_rag),
                "translation_latency_ms": compute_stats(c_trans),
                "tts_latency_ms": compute_stats(c_tts),
                "e2e_latency_ms": compute_stats(c_e2e)
            }
        }
        case_results.append(case_entry)
        print(f"  → STT mean: {compute_stats(c_stt)['mean']} ms | Trans mean: {compute_stats(c_trans)['mean']} ms | E2E mean: {compute_stats(c_e2e)['mean']} ms", flush=True)

    # 5. Compile Statistical Summary
    summary = {
        "first_partial": compute_stats(all_first_partial),
        "final_stt": compute_stats(all_final_stt),
        "rag": compute_stats(all_rag),
        "translation": compute_stats(all_translation),
        "tts": compute_stats(all_tts),
        "e2e": compute_stats(all_e2e)
    }

    # Re-inspect process peak RAM post-benchmark
    final_env = get_system_environment()

    # 6. Save JSON Report
    results_dir = Path("backend/benchmark_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"baseline_{timestamp_str}.json"

    report_payload = {
        "benchmark": "phase_0_golden_baseline",
        "timestamp": datetime.now().isoformat(),
        "environment": final_env,
        "configuration": config,
        "cold_start": cold_metrics,
        "warm_aggregates": summary,
        "cases": case_results,
        "notes": [
            "Faster-Whisper inference executed locally on CPU via CTranslate2 INT8.",
            "RAG executed locally via Scikit-learn TF-IDF Cosine Similarity.",
            "First partial latency measured across simulated 250ms chunks fed into StreamingSession.",
            "Cases where audio length was under MIN_PARTIAL_BYTES yielded null first partials as designed.",
            "Translation executed via configured provider (Groq or local cascade fallback).",
            "TTS executed via Edge-TTS neural async synthesis stream."
        ]
    }

    report_file.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[+] Saved golden baseline report to: {report_file}", flush=True)

    # 7. Print Standardized Console Report
    print("\n" + "=" * 70, flush=True)
    print("                    GOLDEN BASELINE BENCHMARK REPORT", flush=True)
    print("=" * 70, flush=True)
    print(f"Python:       {final_env['python_version']} ({final_env['architecture']})")
    print(f"OS:           {final_env['os']}")
    print(f"CPU:          {final_env['processor']}")
    print(f"Total RAM:    {final_env['total_ram_mb']} MB (Free: {final_env['free_ram_mb']} MB)")
    print(f"Peak Proc RAM:{final_env['peak_process_ram_mb']} MB\n")
    print(f"STT Model:    Faster-Whisper ({config['whisper_size']}_{config['whisper_compute_type']} on {config['whisper_device']})")
    print(f"LLM:          {config['default_llm_provider']} ({config['groq_model']})")
    print(f"TTS:          {config['tts_engine']}")
    print("-" * 70, flush=True)
    print("STAGE LATENCY SUMMARY (WARM RUNS across all test cases):")
    print(f"{'Stage':<18} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 70, flush=True)
    for name, st in [
        ("First Partial", summary["first_partial"]),
        ("Final STT", summary["final_stt"]),
        ("RAG", summary["rag"]),
        ("Translation", summary["translation"]),
        ("TTS", summary["tts"]),
        ("End-to-End", summary["e2e"])
    ]:
        print(f"{name:<18} | {str(st['mean']):<10} | {str(st['p50']):<10} | {str(st['p95']):<10} | {str(st['min']):<10} | {str(st['max']):<10}")
    print("=" * 70, flush=True)

    return report_payload


if __name__ == "__main__":
    asyncio.run(execute_golden_baseline())
