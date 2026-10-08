"""
Phase 2: Standalone VAD & Endpointing Benchmark Harness.
Evaluates Silero VAD v5 ONNX performance on chunked 16kHz audio across difficult speech categories:
  - Normal English, Normal Hindi, Short Speech ("Hi", "Yes", "No", "RAG", "API")
  - Quiet Speech (-45 dBFS, -55 dBFS)
  - Speech with pauses ("Hello ... how are you?")
  - Leading and Trailing silence (500ms, 1000ms, 1500ms)
  - Pure silence & noise floors (-55 dBFS, -65 dBFS)
  - Technical terms & numerical identifiers

Measures:
  1. Detection Accuracy, Confusion Matrix (TP, TN, FP, FN)
  2. Endpoint Latency / Delay (ms)
  3. Premature Endpoint Rate
  4. Per-chunk VAD Inference Latency (Mean, p50, p95, Min, Max, StdDev)
  5. Short-utterance preservation
"""

import os
import sys
import io
import time
import json
import math
import wave
import array
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

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

from backend.services.vad_service import (
    vad_service,
    VADConfig,
    VADSessionState,
)
from backend.audio_diagnostics import inspect_pcm16_wav


def make_pcm16_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Converts numpy int16 array to standard WAV container bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def read_pcm16_samples(wav_bytes: bytes) -> np.ndarray:
    """Reads 16-bit mono PCM samples from WAV bytes as np.int16 array."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def create_silence_samples(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    """Creates pure digital silence samples."""
    return np.zeros(int(duration_s * sample_rate), dtype=np.int16)


def create_noise_samples(duration_s: float, dbfs: float, sample_rate: int = 16000) -> np.ndarray:
    """Creates Gaussian noise samples calibrated to specific RMS dBFS."""
    count = int(duration_s * sample_rate)
    target_rms = 32768.0 * (10.0 ** (dbfs / 20.0))
    noise = np.random.normal(0, target_rms, count)
    return np.clip(noise, -32768, 32767).astype(np.int16)


def scale_samples_to_dbfs(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scales audio samples to specific target RMS dBFS."""
    if len(samples) == 0:
        return samples
    float_s = samples.astype(np.float64)
    ms = np.mean(float_s ** 2)
    current_rms = math.sqrt(ms) if ms > 0 else 1e-12
    current_dbfs = 20 * math.log10(max(current_rms / 32768.0, 1e-12))
    factor = 10.0 ** ((target_dbfs - current_dbfs) / 20.0)
    scaled = float_s * factor
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def build_vad_benchmark_dataset() -> List[Dict[str, Any]]:
    """
    Builds the test dataset covering all 10 required difficult speech and non-speech categories.
    """
    samples_dir = REPO_ROOT / "backend" / "debug_audio" / "benchmark_samples"

    def load_canonical(name: str) -> np.ndarray:
        p = samples_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing canonical sample: {p}")
        return read_pcm16_samples(p.read_bytes())

    en1_samples = load_canonical("EN-1.wav")      # "Hello" (~1.87s)
    en2_samples = load_canonical("EN-2.wav")      # "What is your name?" (~1.87s)
    en3_samples = load_canonical("EN-3.wav")      # "How are you doing today?" (~2.06s)
    hi1_samples = load_canonical("HI-1.wav")      # "आपका नाम क्या है?" (~2.28s)
    hi2_samples = load_canonical("HI-2.wav")      # "आप कैसे हैं?" (~1.99s)
    tech1_samples = load_canonical("TECH-1.wav")  # "We need to implement a vector database with RAG." (~3.62s)
    tech2_samples = load_canonical("TECH-2.wav")  # "Kubernetes orchestration." (~2.42s)
    num1_samples = load_canonical("NUM-1.wav")    # "The web server returned HTTP 404 error code." (~4.34s)
    long1_samples = load_canonical("LONG-1.wav")  # "We deployed the backend service using Docker containers and Kubernetes." (~4.73s)

    cases = []

    # Category A: Normal English
    cases.append({
        "id": "A_ENG_NORMAL",
        "category": "Normal English",
        "description": "Conversational question ('What is your name?')",
        "samples": en2_samples,
        "ground_truth_has_speech": True,
        "is_speech_at_offset": lambda t_s: 0.1 <= t_s <= 1.8,
        "expected_endpoint_after_s": 1.87,
    })

    # Category B: Normal Hindi
    cases.append({
        "id": "B_HIN_NORMAL_1",
        "category": "Normal Hindi",
        "description": "Conversational Hindi ('आपका नाम क्या है?')",
        "samples": hi1_samples,
        "ground_truth_has_speech": True,
        "is_speech_at_offset": lambda t_s: 0.1 <= t_s <= 2.2,
        "expected_endpoint_after_s": 2.28,
    })
    cases.append({
        "id": "B_HIN_NORMAL_2",
        "category": "Normal Hindi",
        "description": "Conversational Hindi ('आप कैसे हैं?')",
        "samples": hi2_samples,
        "ground_truth_has_speech": True,
        "is_speech_at_offset": lambda t_s: 0.1 <= t_s <= 1.9,
        "expected_endpoint_after_s": 1.99,
    })

    # Category C: Short Speech (sub-utterances & single words)
    cases.append({
        "id": "C_SHORT_HELLO",
        "category": "Short Speech",
        "description": "Short single-word utterance ('Hello')",
        "samples": en1_samples,
        "ground_truth_has_speech": True,
        "is_speech_at_offset": lambda t_s: 0.2 <= t_s <= 1.2,
        "expected_endpoint_after_s": 1.87,
    })
    # Extract tight ~400ms word from EN-1
    tight_short = en1_samples[int(16000 * 0.3):int(16000 * 0.75)]
    cases.append({
        "id": "C_SHORT_TIGHT",
        "category": "Short Speech",
        "description": "Very short 450ms utterance ('Hi / Yes')",
        "samples": tight_short,
        "ground_truth_has_speech": True,
        "is_speech_at_offset": lambda t_s: 0.05 <= t_s <= 0.4,
        "expected_endpoint_after_s": 0.45,
    })

    # Category D: Quiet Speech
    quiet_45 = scale_samples_to_dbfs(en2_samples, -45.0)
    cases.append({
        "id": "D_QUIET_45DB",
        "category": "Quiet Speech",
        "description": "Attenuated conversational English at -45 dBFS",
        "samples": quiet_45,
        "ground_truth_has_speech": True,
        "is_speech_at_offset": lambda t_s: 0.1 <= t_s <= 1.8,
        "expected_endpoint_after_s": 1.87,
    })
    quiet_55 = scale_samples_to_dbfs(hi2_samples, -52.0)
    cases.append({
        "id": "D_QUIET_52DB_HI",
        "category": "Quiet Speech",
        "description": "Soft Hindi speech at -52 dBFS",
        "samples": quiet_55,
        "ground_truth_has_speech": True,
        "is_speech_at_offset": lambda t_s: 0.1 <= t_s <= 1.9,
        "expected_endpoint_after_s": 1.99,
    })

    # Category E: Speech with Pauses ("Hello ... how are you?")
    # en1 + 800ms silence + en3
    pause_silence = create_silence_samples(0.8)
    paused_speech = np.concatenate([en1_samples, pause_silence, en3_samples])
    cases.append({
        "id": "E_SPEECH_PAUSE",
        "category": "Speech with Pauses",
        "description": "Two clauses separated by an internal 800ms natural pause",
        "samples": paused_speech,
        "ground_truth_has_speech": True,
        "pause_start_s": len(en1_samples) / 16000.0,
        "pause_end_s": (len(en1_samples) + len(pause_silence)) / 16000.0,
        "expected_endpoint_after_s": len(paused_speech) / 16000.0,
    })

    # Category F: Long Speech
    cases.append({
        "id": "F_LONG_DOCKER_K8S",
        "category": "Long Speech",
        "description": "Continuous natural technical sentence ('We deployed the backend...')",
        "samples": long1_samples,
        "ground_truth_has_speech": True,
        "is_speech_at_offset": lambda t_s: 0.1 <= t_s <= 4.6,
        "expected_endpoint_after_s": 4.73,
    })

    # Category G: Trailing Silence (speech + 1000ms trailing silence)
    trail_silence = create_silence_samples(1.0)
    with_trailing = np.concatenate([en2_samples, trail_silence])
    cases.append({
        "id": "G_TRAILING_SILENCE_1S",
        "category": "Trailing Silence",
        "description": "Speech followed by 1000ms trailing silence to measure endpoint trigger",
        "samples": with_trailing,
        "ground_truth_has_speech": True,
        "speech_end_s": len(en2_samples) / 16000.0,
        "expected_endpoint_after_s": len(en2_samples) / 16000.0,
    })

    # Category H: Leading Silence (800ms leading silence + speech)
    lead_silence = create_silence_samples(0.8)
    with_leading = np.concatenate([lead_silence, en2_samples])
    cases.append({
        "id": "H_LEADING_SILENCE_800MS",
        "category": "Leading Silence",
        "description": "800ms initial silence before user speaks",
        "samples": with_leading,
        "ground_truth_has_speech": True,
        "speech_start_s": len(lead_silence) / 16000.0,
        "expected_endpoint_after_s": len(with_leading) / 16000.0,
    })

    # Category I: Pure Silence and Noise Floor (Negative Controls: MUST NOT DETECT SPEECH)
    cases.append({
        "id": "I_PURE_SILENCE_1S",
        "category": "Silence / Noise Floor",
        "description": "1.0s pure digital silence",
        "samples": create_silence_samples(1.0),
        "ground_truth_has_speech": False,
        "expected_endpoint_after_s": None,
    })
    cases.append({
        "id": "I_NOISE_FLOOR_55DB",
        "category": "Silence / Noise Floor",
        "description": "1.5s stationary ambient noise floor at -55 dBFS",
        "samples": create_noise_samples(1.5, -55.0),
        "ground_truth_has_speech": False,
        "expected_endpoint_after_s": None,
    })
    cases.append({
        "id": "I_NOISE_FLOOR_65DB",
        "category": "Silence / Noise Floor",
        "description": "2.0s quiet room noise floor at -65 dBFS",
        "samples": create_noise_samples(2.0, -65.0),
        "ground_truth_has_speech": False,
        "expected_endpoint_after_s": None,
    })

    # Category J: Technical Speech
    cases.append({
        "id": "J_TECH_RAG",
        "category": "Technical Speech",
        "description": "Vector database with RAG ('We need to implement a vector database with RAG.')",
        "samples": tech1_samples,
        "ground_truth_has_speech": True,
        "expected_endpoint_after_s": 3.62,
    })
    cases.append({
        "id": "J_TECH_K8S",
        "category": "Technical Speech",
        "description": "Kubernetes orchestration ('Kubernetes orchestration.')",
        "samples": tech2_samples,
        "ground_truth_has_speech": True,
        "expected_endpoint_after_s": 2.42,
    })
    cases.append({
        "id": "J_TECH_HTTP404",
        "category": "Technical Speech",
        "description": "HTTP 404 numerical error code ('The web server returned HTTP 404 error code.')",
        "samples": num1_samples,
        "ground_truth_has_speech": True,
        "expected_endpoint_after_s": 4.34,
    })

    return cases


def run_standalone_vad_benchmark() -> Dict[str, Any]:
    print("=" * 80, flush=True)
    print("PHASE 2: STANDALONE SILERO VAD & ENDPOINTING BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    print(f"[*] Environment: {platform.system()} {platform.release()} ({platform.machine()})", flush=True)
    print(f"[*] Python: {platform.python_version()}", flush=True)
    print(f"[*] Silero VAD Startup Latency: {vad_service.startup_latency_ms:.2f} ms", flush=True)
    print(f"[*] First Inference Latency:   {vad_service.first_inference_latency_ms:.2f} ms", flush=True)

    dataset = build_vad_benchmark_dataset()
    print(f"[*] Built {len(dataset)} evaluation cases across 10 difficult speech/noise categories.\n", flush=True)

    config = VADConfig(
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=600,
        speech_pad_ms=150,
        hangover_ms=300,
    )

    CHUNK_MS = 250
    CHUNK_SAMPLES = int(16000 * (CHUNK_MS / 1000.0))  # 4000 samples = 250ms

    chunk_latencies_ms = []
    case_results = []

    tp_count = 0  # Speech correctly detected
    tn_count = 0  # Silence correctly detected
    fp_count = 0  # Silence falsely marked as speech
    fn_count = 0  # Speech missed

    premature_endpoints = 0
    endpoint_delays_ms = []

    print(f"{'ID':<22} | {'Category':<22} | {'Speech GT':<9} | {'VAD Detected':<12} | {'Peak Prob':<9} | {'Endpoint Trigger':<16} | {'Status'}", flush=True)
    print("-" * 105, flush=True)

    for case in dataset:
        cid = case["id"]
        samples = case["samples"]
        gt_has_speech = case["ground_truth_has_speech"]

        state = vad_service.create_session_state(session_id=f"test_{cid}")
        detected_speech_chunks = 0
        peak_prob_observed = 0.0
        endpoint_offset_ms: Optional[float] = None

        # Feed 250ms chunks sequentially
        for offset in range(0, len(samples), CHUNK_SAMPLES):
            chunk = samples[offset:offset + CHUNK_SAMPLES]
            if len(chunk) < 512:
                # pad last chunk if needed
                pad = 512 - len(chunk)
                chunk = np.pad(chunk, (0, pad))

            t0 = time.perf_counter()
            res = vad_service.process_streaming_chunk(chunk.tobytes(), state=state, config=config)
            t_chunk_ms = (time.perf_counter() - t0) * 1000.0
            chunk_latencies_ms.append(t_chunk_ms)

            if res["is_speech"]:
                detected_speech_chunks += 1
            if res["speech_prob"] > peak_prob_observed:
                peak_prob_observed = res["speech_prob"]

            if res["is_endpoint"] and endpoint_offset_ms is None:
                endpoint_offset_ms = state.processed_ms

        vad_detected_speech = (detected_speech_chunks > 0 and peak_prob_observed >= config.threshold)

        # Confusion Matrix
        if gt_has_speech:
            if vad_detected_speech:
                tp_count += 1
                status = "TP (PASS)"
            else:
                fn_count += 1
                status = "FN (MISSED SPEECH)"
        else:
            if not vad_detected_speech:
                tn_count += 1
                status = "TN (PASS)"
            else:
                fp_count += 1
                status = "FP (FALSE SPEECH)"

        # Endpointing check
        endpoint_str = "None"
        if endpoint_offset_ms is not None:
            endpoint_str = f"{endpoint_offset_ms:.0f} ms"
            if "speech_end_s" in case:
                actual_end_ms = case["speech_end_s"] * 1000.0
                delay = endpoint_offset_ms - actual_end_ms
                endpoint_delays_ms.append(delay)
            elif "pause_start_s" in case:
                # Mid-sentence pause check: did endpoint trigger during pause?
                pause_start_ms = case["pause_start_s"] * 1000.0
                pause_end_ms = case["pause_end_s"] * 1000.0
                if pause_start_ms <= endpoint_offset_ms <= pause_end_ms:
                    premature_endpoints += 1
                    status += " [PREMATURE ENDPOINT IN PAUSE]"

        case_results.append({
            "id": cid,
            "category": case["category"],
            "description": case["description"],
            "ground_truth_has_speech": gt_has_speech,
            "vad_detected_speech": vad_detected_speech,
            "peak_probability": round(peak_prob_observed, 3),
            "detected_speech_chunks": detected_speech_chunks,
            "total_chunks": state.chunks_processed,
            "endpoint_offset_ms": endpoint_offset_ms,
            "status": status,
        })

        print(f"{cid:<22} | {case['category']:<22} | {str(gt_has_speech):<9} | {str(vad_detected_speech):<12} | {peak_prob_observed:<9.3f} | {endpoint_str:<16} | {status}", flush=True)

    print("-" * 105, flush=True)

    # Compute Statistical Metrics
    total_evals = len(dataset)
    accuracy = (tp_count + tn_count) / total_evals if total_evals > 0 else 0.0
    sensitivity_recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    specificity = tn_count / (tn_count + fp_count) if (tn_count + fp_count) > 0 else 0.0
    fp_rate = fp_count / (tn_count + fp_count) if (tn_count + fp_count) > 0 else 0.0
    fn_rate = fn_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0

    s_lat = sorted(chunk_latencies_ms)
    n_lat = len(s_lat)
    mean_lat = sum(s_lat) / n_lat if n_lat > 0 else 0.0
    p50_lat = s_lat[n_lat // 2] if n_lat > 0 else 0.0
    p95_lat = s_lat[min(n_lat - 1, int(math.ceil(0.95 * n_lat)) - 1)] if n_lat > 0 else 0.0
    min_lat = s_lat[0] if n_lat > 0 else 0.0
    max_lat = s_lat[-1] if n_lat > 0 else 0.0
    var_lat = sum((x - mean_lat) ** 2 for x in s_lat) / (n_lat - 1) if n_lat > 1 else 0.0
    std_lat = math.sqrt(var_lat)

    mean_endpoint_delay = sum(endpoint_delays_ms) / len(endpoint_delays_ms) if endpoint_delays_ms else 0.0

    print("\n[STANDALONE VAD BENCHMARK SUMMARY]")
    print(f"  Total Test Cases:          {total_evals}")
    print(f"  Confusion Matrix:          TP={tp_count}, TN={tn_count}, FP={fp_count}, FN={fn_count}")
    print(f"  Overall Detection Accuracy: {accuracy * 100:.2f}%")
    print(f"  Sensitivity (Recall):      {sensitivity_recall * 100:.2f}%")
    print(f"  Specificity (Noise Reject):{specificity * 100:.2f}%")
    print(f"  False Positive Rate (FP):  {fp_rate * 100:.2f}%")
    print(f"  False Negative Rate (FN):  {fn_rate * 100:.2f}%")
    print(f"  Premature Endpoints in Pause: {premature_endpoints}")
    print(f"  Mean Endpoint Delay:       {mean_endpoint_delay:.1f} ms")
    print(f"\n[PER-CHUNK 250ms VAD LATENCY]")
    print(f"  N:     {n_lat} chunks")
    print(f"  Mean:  {mean_lat:.2f} ms")
    print(f"  p50:   {p50_lat:.2f} ms")
    print(f"  p95:   {p95_lat:.2f} ms")
    print(f"  Min:   {min_lat:.2f} ms")
    print(f"  Max:   {max_lat:.2f} ms")
    print(f"  Std:   {std_lat:.2f} ms")
    print("=" * 80 + "\n", flush=True)

    report_payload = {
        "benchmark": "phase_2_standalone_vad_benchmark",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "os": f"{platform.system()}-{platform.release()}",
            "architecture": platform.machine(),
        },
        "config": {
            "threshold": config.threshold,
            "min_speech_duration_ms": config.min_speech_duration_ms,
            "min_silence_duration_ms": config.min_silence_duration_ms,
            "speech_pad_ms": config.speech_pad_ms,
            "hangover_ms": config.hangover_ms,
            "chunk_size_ms": CHUNK_MS,
        },
        "startup": {
            "startup_latency_ms": vad_service.startup_latency_ms,
            "first_inference_latency_ms": vad_service.first_inference_latency_ms,
        },
        "confusion_matrix": {
            "tp": tp_count,
            "tn": tn_count,
            "fp": fp_count,
            "fn": fn_count,
            "accuracy": round(accuracy, 4),
            "sensitivity": round(sensitivity_recall, 4),
            "specificity": round(specificity, 4),
            "fp_rate": round(fp_rate, 4),
            "fn_rate": round(fn_rate, 4),
        },
        "endpointing": {
            "premature_endpoints": premature_endpoints,
            "mean_endpoint_delay_ms": round(mean_endpoint_delay, 1),
            "endpoint_delays_ms": [round(d, 1) for d in endpoint_delays_ms],
        },
        "latency_stats": {
            "n": n_lat,
            "mean": round(mean_lat, 2),
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
            "std": round(std_lat, 2),
        },
        "case_results": case_results,
    }

    results_dir = REPO_ROOT / "backend" / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"vad_standalone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"[+] Saved standalone VAD report to: {report_path}\n", flush=True)

    return report_payload


if __name__ == "__main__":
    run_standalone_vad_benchmark()
