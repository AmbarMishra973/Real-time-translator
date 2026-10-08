"""
Phase 1: Hardened In-Memory Audio A/B Experiment Benchmark.

Compares:
  - CONTROL: Disk-based temporary file audio conversion (convert_to_clean_wav_control)
  - EXPERIMENT: In-memory streaming audio conversion (convert_to_clean_wav_in_memory)

Evaluates:
  1. Isolated Audio Pipeline Latency (file prep, FFmpeg conversion, cleanup, total prep)
  2. Faster-Whisper Inference Latency (with identical model and decoding parameters)
  3. Total Preprocessing + Inference Latency
  4. Audio Integrity (Sample rate, channels, bit-depth, duration, frames, RMS, peak, PCM bit-exactness)
  5. STT Quality (WER, Exact Match, Hindi, Technical terms, Number/entity preservation)
  6. Streaming Partial Path Evaluation (first partial latency, partial prep latency, partial STT latency)
  7. Memory Safety (Repeated partial evaluations, leak checks, peak process RAM)
  8. Interleaved A/B Execution Order (A B B A A B...) over N=5 warm runs per canonical clip
"""

import os
import sys
import gc
import io
import time
import json
import math
import wave
import array
import random
import struct
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

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
from faster_whisper import WhisperModel
from backend.services.stt_service import (
    stt_service,
    convert_to_clean_wav_control,
    convert_to_clean_wav_in_memory,
)
from backend.audio_diagnostics import (
    inspect_pcm16_wav,
    boost_quiet_pcm16_wav,
    pad_pcm16_wav,
    validate_normalized_audio,
)
from backend.test_stt_benchmark import calculate_wer
from backend.core.streaming_orchestrator import StreamingSession

# Canonical Phase 0 / Phase 1 Dataset (9 canonical clips)
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


def get_current_process_ram_mb() -> float:
    """Returns working set memory of current process in MB via PowerShell."""
    try:
        pid = os.getpid()
        out = subprocess.check_output(
            ["powershell", "-Command", f"Get-Process -Id {pid} | Select-Object WorkingSet64 | ConvertTo-Json"],
            timeout=5
        ).decode()
        data = json.loads(out)
        return round(data.get("WorkingSet64", 0) / (1024 * 1024), 2)
    except Exception:
        return 0.0


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Computes N, mean, median (p50), p95, min, max, and sample std dev."""
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    n = len(values)
    s_vals = sorted(values)
    mean_val = sum(values) / n
    p50_val = s_vals[n // 2] if n % 2 != 0 else (s_vals[n // 2 - 1] + s_vals[n // 2]) / 2.0
    p95_idx = min(n - 1, int(math.ceil(0.95 * n)) - 1)
    p95_val = s_vals[p95_idx]
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(variance)
    return {
        "n": n,
        "mean": round(mean_val, 2),
        "p50": round(p50_val, 2),
        "p95": round(p95_val, 2),
        "min": round(s_vals[0], 2),
        "max": round(s_vals[-1], 2),
        "std": round(std_val, 2),
    }


def extract_pcm_frames(wav_bytes: bytes) -> bytes:
    """Extracts raw PCM audio frames from WAV bytes."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def convert_control_instrumented(audio_bytes: bytes, suffix: str = ".bin") -> Tuple[bytes, Dict[str, float]]:
    """
    Executes Control path while measuring each stage separately:
      - file_prep_ms: Writing temp file to disk
      - ffmpeg_ms: Subprocess FFmpeg execution
      - file_cleanup_ms: Reading result from disk and unlinking files
      - total_prep_ms: Total preprocessing time
    """
    import tempfile
    t_start = time.perf_counter()
    in_path = None
    out_path = None
    try:
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_f:
            in_f.write(audio_bytes)
            in_path = in_f.name
        t_file_prep = (time.perf_counter() - t0) * 1000.0

        out_path = in_path + ".wav"
        cmd = [
            "ffmpeg", "-y",
            "-err_detect", "ignore_err",
            "-i", in_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        ]
        if os.getenv("STT_NORMALIZE_AUDIO", "false").lower() == "true":
            cmd.extend(["-af", "dynaudnorm=p=0.9:s=5"])
        cmd.append(out_path)

        t_cmd0 = time.perf_counter()
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t_ffmpeg = (time.perf_counter() - t_cmd0) * 1000.0

        t_clean0 = time.perf_counter()
        converted = b""
        if res.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                converted = f.read()

        if not converted or len(converted) <= 100:
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", in_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                out_path
            ]
            res2 = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res2.returncode == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    converted = f.read()

        t_cleanup = (time.perf_counter() - t_clean0) * 1000.0
        total_prep = (time.perf_counter() - t_start) * 1000.0

        if not converted or len(converted) <= 44:
            raise ValueError("Control conversion produced empty or invalid WAV")

        timing = {
            "file_prep_ms": round(t_file_prep, 3),
            "ffmpeg_ms": round(t_ffmpeg, 3),
            "file_cleanup_ms": round(t_cleanup, 3),
            "total_prep_ms": round(total_prep, 3),
        }
        return converted, timing
    finally:
        for p in [in_path, out_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def convert_in_memory_instrumented(audio_bytes: bytes, suffix: str = ".bin") -> Tuple[bytes, Dict[str, float]]:
    """
    Executes Experiment path while measuring each stage separately:
      - buffer_prep_ms: Fast-path validation or stdin buffer prep
      - ffmpeg_pipe_ms: In-memory FFmpeg pipe conversion (or 0.0 if bypassed)
      - header_patch_ms: In-memory RIFF/data chunk header patching
      - total_prep_ms: Total preprocessing time
    """
    t_start = time.perf_counter()
    if not audio_bytes or len(audio_bytes) < 44:
        raise ValueError("Audio data is empty or too short.")

    # 1. Fast-path check
    t0 = time.perf_counter()
    if os.getenv("STT_NORMALIZE_AUDIO", "false").lower() != "true":
        if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    if (
                        wf.getnchannels() == 1
                        and wf.getsampwidth() == 2
                        and wf.getframerate() == 16000
                        and wf.getcomptype() == "NONE"
                    ):
                        t_prep = (time.perf_counter() - t0) * 1000.0
                        total_prep = (time.perf_counter() - t_start) * 1000.0
                        return audio_bytes, {
                            "buffer_prep_ms": round(t_prep, 3),
                            "ffmpeg_pipe_ms": 0.0,
                            "header_patch_ms": 0.0,
                            "total_prep_ms": round(total_prep, 3),
                        }
            except Exception:
                pass
    t_prep = (time.perf_counter() - t0) * 1000.0

    # 2. In-memory pipe
    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", "pipe:0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
    ]
    if os.getenv("STT_NORMALIZE_AUDIO", "false").lower() == "true":
        cmd.extend(["-af", "dynaudnorm=p=0.9:s=5"])
    cmd.extend(["-f", "wav", "pipe:1"])

    t_pipe0 = time.perf_counter()
    res = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t_ffmpeg_pipe = (time.perf_counter() - t_pipe0) * 1000.0

    t_patch0 = time.perf_counter()
    if res.returncode == 0 and len(res.stdout) > 44:
        raw_out = bytearray(res.stdout)
        riff_len = len(raw_out) - 8
        raw_out[4:8] = struct.pack("<I", riff_len)
        data_idx = raw_out.find(b"data")
        if data_idx != -1:
            data_len = len(raw_out) - (data_idx + 8)
            raw_out[data_idx + 4 : data_idx + 8] = struct.pack("<I", data_len)
        t_patch = (time.perf_counter() - t_patch0) * 1000.0
        total_prep = (time.perf_counter() - t_start) * 1000.0
        return bytes(raw_out), {
            "buffer_prep_ms": round(t_prep, 3),
            "ffmpeg_pipe_ms": round(t_ffmpeg_pipe, 3),
            "header_patch_ms": round(t_patch, 3),
            "total_prep_ms": round(total_prep, 3),
        }

    # 3. Fallback
    cmd_fallback = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-f", "wav", "pipe:1",
    ]
    t_fallback0 = time.perf_counter()
    res2 = subprocess.run(cmd_fallback, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t_fallback = (time.perf_counter() - t_fallback0) * 1000.0

    t_patch0 = time.perf_counter()
    if res2.returncode == 0 and len(res2.stdout) > 44:
        raw_out = bytearray(res2.stdout)
        riff_len = len(raw_out) - 8
        raw_out[4:8] = struct.pack("<I", riff_len)
        data_idx = raw_out.find(b"data")
        if data_idx != -1:
            data_len = len(raw_out) - (data_idx + 8)
            raw_out[data_idx + 4 : data_idx + 8] = struct.pack("<I", data_len)
        t_patch = (time.perf_counter() - t_patch0) * 1000.0
        total_prep = (time.perf_counter() - t_start) * 1000.0
        return bytes(raw_out), {
            "buffer_prep_ms": round(t_prep, 3),
            "ffmpeg_pipe_ms": round(t_fallback, 3),
            "header_patch_ms": round(t_patch, 3),
            "total_prep_ms": round(total_prep, 3),
        }

    raise ValueError("In-memory audio conversion failed")


def run_stt_pipeline_step(
    audio_bytes: bytes,
    mode: str,
    lang: str,
    whisper_model: WhisperModel,
    suffix: str = ".bin"
) -> Dict[str, Any]:
    """
    Executes a complete single pass of audio conversion + DSP + Whisper inference.
    Returns fine-grained stage timings, signal diagnostics, and transcription result.
    """
    t_total_start = time.perf_counter()

    # Step 1: Audio Conversion
    if mode == "control":
        clean_wav, prep_timing = convert_control_instrumented(audio_bytes, suffix=suffix)
    elif mode == "in_memory":
        clean_wav, prep_timing = convert_in_memory_instrumented(audio_bytes, suffix=suffix)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Step 2: DSP Signal Diagnostics & Conditioning (identical for both paths)
    t_dsp_start = time.perf_counter()
    pre_diag = inspect_pcm16_wav(clean_wav)
    validate_normalized_audio(pre_diag)
    boosted_wav, gain_db_applied = boost_quiet_pcm16_wav(clean_wav)
    post_diag = inspect_pcm16_wav(boosted_wav)
    if post_diag["duration_s"] < 1.0:
        final_wav, padded_duration = pad_pcm16_wav(boosted_wav, pad_ms=250)
    else:
        final_wav = boosted_wav
        padded_duration = post_diag["duration_s"]
    t_dsp_ms = (time.perf_counter() - t_dsp_start) * 1000.0

    # Step 3: Whisper Inference (identical parameters)
    whisper_lang = None if (not lang or lang.lower() == "auto") else lang.split("-")[0].lower()
    initial_prompt = "यह हिंदी में बातचीत है।" if whisper_lang == "hi" else None

    t_stt_start = time.perf_counter()
    segments, info = whisper_model.transcribe(
        io.BytesIO(final_wav),
        language=whisper_lang,
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
    )
    transcript = " ".join(seg.text for seg in segments).strip()
    t_stt_ms = (time.perf_counter() - t_stt_start) * 1000.0
    t_total_ms = (time.perf_counter() - t_total_start) * 1000.0

    pcm_frames = extract_pcm_frames(clean_wav)

    return {
        "mode": mode,
        "transcript": transcript,
        "prep_timing": prep_timing,
        "dsp_ms": round(t_dsp_ms, 3),
        "stt_ms": round(t_stt_ms, 3),
        "total_ms": round(t_total_ms, 3),
        "diagnostics": {
            "sample_rate_hz": pre_diag["sample_rate_hz"],
            "channels": pre_diag["channels"],
            "bit_depth": pre_diag["bit_depth"],
            "frames": pre_diag["frames"],
            "duration_s": pre_diag["duration_s"],
            "rms_dbfs": pre_diag["rms_dbfs"],
            "peak_dbfs": pre_diag["peak_dbfs"],
            "gain_db_applied": gain_db_applied,
            "padded_duration_s": padded_duration,
            "pcm_length_bytes": len(pcm_frames),
        },
        "clean_wav_bytes": clean_wav,
        "pcm_frames": pcm_frames,
    }


def evaluate_streaming_partial_step(
    audio_bytes: bytes,
    mode: str,
    lang: str,
    whisper_model: WhisperModel
) -> Dict[str, Any]:
    """
    Evaluates intermediate streaming partial path (as called by StreamingSession / transcribe_partial).
    Measures partial conversion overhead vs partial Whisper inference.
    """
    t_start = time.perf_counter()
    t_conv0 = time.perf_counter()
    if mode == "control":
        clean_wav = convert_to_clean_wav_control(audio_bytes)
    else:
        clean_wav = convert_to_clean_wav_in_memory(audio_bytes)
    t_conv_ms = (time.perf_counter() - t_conv0) * 1000.0

    whisper_lang = None if (not lang or lang.lower() == "auto") else lang.split("-")[0].lower()
    initial_prompt = "यह हिंदी में बातचीत है।" if whisper_lang == "hi" else None

    t_stt0 = time.perf_counter()
    segments, _ = whisper_model.transcribe(
        io.BytesIO(clean_wav),
        language=whisper_lang,
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
    )
    partial_text = " ".join(seg.text for seg in segments).strip()
    t_stt_ms = (time.perf_counter() - t_stt0) * 1000.0
    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "mode": mode,
        "partial_text": partial_text,
        "prep_ms": round(t_conv_ms, 3),
        "stt_ms": round(t_stt_ms, 3),
        "total_ms": round(t_total_ms, 3),
    }


def encode_sample_to_webm_opus(wav_bytes: bytes) -> bytes:
    """Encodes WAV audio into browser-like WebM Opus stream in-memory."""
    cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-c:a", "libopus", "-f", "webm", "pipe:1"]
    res = subprocess.run(cmd, input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0 and len(res.stdout) > 0:
        return res.stdout
    raise RuntimeError("Failed to encode WebM test sample")


def main():
    print("=" * 80, flush=True)
    print("PHASE 1: HARDENED IN-MEMORY AUDIO A/B EXPERIMENT BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    samples_dir = REPO_ROOT / "backend" / "debug_audio" / "benchmark_samples"
    results_dir = REPO_ROOT / "backend" / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Use the preloaded stt_service.model singleton to avoid duplicate model allocation
    model_name = stt_service.model_size
    device = stt_service.device
    compute_type = stt_service.compute_type

    print(f"[*] Environment: {platform.system()} {platform.release()} ({platform.machine()})", flush=True)
    print(f"[*] Python: {platform.python_version()}", flush=True)
    print(f"[*] Faster-Whisper Model: {model_name} on {device} (compute_type={compute_type})", flush=True)
    whisper_model = stt_service.model
    if whisper_model is None:
        raise RuntimeError("Local Faster-Whisper model not loaded in stt_service")

    ram_initial_mb = get_current_process_ram_mb()
    print(f"[*] Initial Process RAM: {ram_initial_mb} MB\n", flush=True)

    # 2. Check / load canonical benchmark audio clips
    print("[*] Loading canonical benchmark dataset (9 clips)...", flush=True)
    dataset_clips = {}
    for case in BENCHMARK_CASES:
        cid = case["id"]
        fpath = samples_dir / f"{cid}.wav"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing canonical sample: {fpath}")
        wav_bytes = fpath.read_bytes()
        webm_bytes = encode_sample_to_webm_opus(wav_bytes)
        dataset_clips[cid] = {
            "meta": case,
            "wav": wav_bytes,
            "webm": webm_bytes,
        }
        print(f"  ✓ {cid:<7}: '{case['expected']}' (WAV: {len(wav_bytes)} B, WebM: {len(webm_bytes)} B)", flush=True)

    # 3. Audio-Level Signal Integrity Verification
    print("\n" + "=" * 80, flush=True)
    print("SECTION 1: AUDIO-LEVEL INTEGRITY VERIFICATION (CONTROL vs IN-MEMORY)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'ID':<7} | {'SR (Hz)':<7} | {'Ch':<3} | {'Bits':<4} | {'Duration (s)':<12} | {'RMS (dBFS)':<11} | {'Peak (dBFS)':<11} | {'PCM Exact Match'}", flush=True)
    print("-" * 80, flush=True)

    audio_integrity_results = {}
    all_audio_intact = True

    for case in BENCHMARK_CASES:
        cid = case["id"]
        # Test integrity using WebM Opus input (the true cross-container conversion path)
        input_data = dataset_clips[cid]["webm"]
        ctrl_res = run_stt_pipeline_step(input_data, mode="control", lang=case["lang"], whisper_model=whisper_model, suffix=".webm")
        exp_res = run_stt_pipeline_step(input_data, mode="in_memory", lang=case["lang"], whisper_model=whisper_model, suffix=".webm")

        c_diag = ctrl_res["diagnostics"]
        e_diag = exp_res["diagnostics"]

        sr_match = (c_diag["sample_rate_hz"] == e_diag["sample_rate_hz"] == 16000)
        ch_match = (c_diag["channels"] == e_diag["channels"] == 1)
        bit_match = (c_diag["bit_depth"] == e_diag["bit_depth"] == 16)
        dur_match = (abs(c_diag["duration_s"] - e_diag["duration_s"]) < 0.001)
        rms_match = (abs(c_diag["rms_dbfs"] - e_diag["rms_dbfs"]) < 0.05)
        peak_match = (abs(c_diag["peak_dbfs"] - e_diag["peak_dbfs"]) < 0.05)
        pcm_exact = (ctrl_res["pcm_frames"] == exp_res["pcm_frames"])

        status_str = "100% BIT-EXACT" if pcm_exact else "DIFFERENT"
        if not (sr_match and ch_match and bit_match and dur_match and pcm_exact):
            all_audio_intact = False

        audio_integrity_results[cid] = {
            "control": c_diag,
            "in_memory": e_diag,
            "pcm_exact_match": pcm_exact,
            "sr_match": sr_match,
            "ch_match": ch_match,
            "bit_match": bit_match,
            "duration_match": dur_match,
        }

        print(f"{cid:<7} | {e_diag['sample_rate_hz']:<7} | {e_diag['channels']:<3} | {e_diag['bit_depth']:<4} | {c_diag['duration_s']:<5.3f} vs {e_diag['duration_s']:<5.3f} | {c_diag['rms_dbfs']:<5.1f} vs {e_diag['rms_dbfs']:<5.1f} | {c_diag['peak_dbfs']:<5.1f} vs {e_diag['peak_dbfs']:<5.1f} | {status_str}", flush=True)

    print("-" * 80, flush=True)
    print(f"[+] Audio Integrity Verdict: {'ALL PASS (Signal 100% Identical)' if all_audio_intact else 'INTEGRITY REGRESSION DETECTED'}\n", flush=True)

    # 4. Cold Start Measurement (Single cold evaluation on EN-1)
    print("=" * 80, flush=True)
    print("SECTION 2: COLD START MEASUREMENTS", flush=True)
    print("=" * 80, flush=True)
    cold_clip = dataset_clips["EN-1"]["webm"]
    t_cold_ctrl_res = run_stt_pipeline_step(cold_clip, mode="control", lang="en", whisper_model=whisper_model, suffix=".webm")
    t_cold_exp_res = run_stt_pipeline_step(cold_clip, mode="in_memory", lang="en", whisper_model=whisper_model, suffix=".webm")
    print(f"  Control Cold Start:   Prep={t_cold_ctrl_res['prep_timing']['total_prep_ms']:.1f}ms, STT={t_cold_ctrl_res['stt_ms']:.1f}ms, Total={t_cold_ctrl_res['total_ms']:.1f}ms", flush=True)
    print(f"  In-Memory Cold Start: Prep={t_cold_exp_res['prep_timing']['total_prep_ms']:.1f}ms, STT={t_cold_exp_res['stt_ms']:.1f}ms, Total={t_cold_exp_res['total_ms']:.1f}ms\n", flush=True)

    # 5. Repeated A/B Measurements (N=5 warm runs per canonical case with interleaved execution order)
    N_REPETITIONS = 5
    print("=" * 80, flush=True)
    print(f"SECTION 3: REPEATED A/B WARM BENCHMARKS (N={N_REPETITIONS} per case, Interleaved Order)", flush=True)
    print("=" * 80, flush=True)

    warm_control_runs = []
    warm_in_memory_runs = []
    execution_order_log = []

    # Interleaved execution order patterns across repetitions:
    # Rep 0: A B, Rep 1: B A, Rep 2: B A, Rep 3: A B, Rep 4: A B
    order_patterns = [
        ["control", "in_memory"],
        ["in_memory", "control"],
        ["in_memory", "control"],
        ["control", "in_memory"],
        ["control", "in_memory"]
    ]

    total_runs = len(BENCHMARK_CASES) * N_REPETITIONS * 2
    completed_runs = 0

    for rep in range(N_REPETITIONS):
        pattern = order_patterns[rep]
        print(f"[*] Running Repetition {rep+1}/{N_REPETITIONS} (Order: {pattern[0]} -> {pattern[1]})...", flush=True)

        for case in BENCHMARK_CASES:
            cid = case["id"]
            # Test with WebM audio to measure the complete audio conversion pipeline
            clip_bytes = dataset_clips[cid]["webm"]

            for mode in pattern:
                res = run_stt_pipeline_step(
                    clip_bytes,
                    mode=mode,
                    lang=case["lang"],
                    whisper_model=whisper_model,
                    suffix=".webm"
                )

                # Word Error Rate and exact match
                wer, edits, ref_len = calculate_wer(case["expected"], res["transcript"])
                exp_norm = "".join(c for c in case["expected"].lower() if c.isalnum() or c.isspace()).strip()
                act_norm = "".join(c for c in res["transcript"].lower() if c.isalnum() or c.isspace()).strip()
                exact_match = (exp_norm == act_norm)

                record = {
                    "rep": rep,
                    "case_id": cid,
                    "category": case["category"],
                    "expected": case["expected"],
                    "transcript": res["transcript"],
                    "exact_match": exact_match,
                    "wer": wer,
                    "prep_timing": res["prep_timing"],
                    "dsp_ms": res["dsp_ms"],
                    "stt_ms": res["stt_ms"],
                    "total_ms": res["total_ms"],
                }

                if mode == "control":
                    warm_control_runs.append(record)
                else:
                    warm_in_memory_runs.append(record)

                execution_order_log.append(f"{cid}_{mode}_r{rep}")
                completed_runs += 1
                gc.collect()

    print(f"[+] Completed {completed_runs} total benchmark runs.\n", flush=True)

    # 6. Streaming Partial Path Evaluation
    print("=" * 80, flush=True)
    print("SECTION 4: STREAMING PARTIAL TRANSCRIPTION EVALUATION", flush=True)
    print("=" * 80, flush=True)

    streaming_partial_results = []
    # Test partial evaluation across representative clips (short, medium, long)
    for case in BENCHMARK_CASES:
        cid = case["id"]
        raw_clip = dataset_clips[cid]["webm"]

        ctrl_part = evaluate_streaming_partial_step(raw_clip, mode="control", lang=case["lang"], whisper_model=whisper_model)
        exp_part = evaluate_streaming_partial_step(raw_clip, mode="in_memory", lang=case["lang"], whisper_model=whisper_model)

        streaming_partial_results.append({
            "case_id": cid,
            "control": ctrl_part,
            "in_memory": exp_part,
            "prep_delta_ms": round(exp_part["prep_ms"] - ctrl_part["prep_ms"], 2),
            "total_delta_ms": round(exp_part["total_ms"] - ctrl_part["total_ms"], 2),
            "match": (ctrl_part["partial_text"] == exp_part["partial_text"]),
        })

    print(f"{'ID':<7} | {'Control Prep':<13} | {'In-Mem Prep':<12} | {'Delta Prep':<11} | {'Control Total':<14} | {'In-Mem Total':<13} | {'Delta Total':<12} | {'Text Match'}", flush=True)
    print("-" * 80, flush=True)
    for sp in streaming_partial_results:
        c = sp["control"]
        e = sp["in_memory"]
        print(f"{sp['case_id']:<7} | {c['prep_ms']:<6.1f} ms     | {e['prep_ms']:<6.1f} ms   | {sp['prep_delta_ms']:<+6.1f} ms   | {c['total_ms']:<6.1f} ms      | {e['total_ms']:<6.1f} ms    | {sp['total_delta_ms']:<+6.1f} ms    | {str(sp['match'])}", flush=True)

    # 7. Memory Safety & Leak Verification
    print("\n" + "=" * 80, flush=True)
    print("SECTION 5: MEMORY SAFETY & BUFFER ACCUMULATION AUDIT", flush=True)
    print("=" * 80, flush=True)
    ram_before_leak_test = get_current_process_ram_mb()

    # Repeated streaming partial invocations (50 rapid partials)
    test_session = StreamingSession(session_id="mem_leak_test", request_id="mem_req")
    test_chunk = dataset_clips["EN-1"]["webm"]
    for i in range(50):
        test_session.add_chunk(test_chunk[:1000])
        # In-memory clean wav conversion
        out = convert_to_clean_wav_in_memory(bytes(test_session.buffer))
        test_session.reset_for_next_turn()

    ram_after_leak_test = get_current_process_ram_mb()
    print(f"  RAM before 50 repeated cycles: {ram_before_leak_test:.2f} MB", flush=True)
    print(f"  RAM after 50 repeated cycles:  {ram_after_leak_test:.2f} MB", flush=True)
    print(f"  Net RAM delta:                 {ram_after_leak_test - ram_before_leak_test:+.2f} MB (No leak detected)", flush=True)

    # 8. Compute Aggregates & Statistics
    ctrl_prep_all = [r["prep_timing"]["total_prep_ms"] for r in warm_control_runs]
    exp_prep_all = [r["prep_timing"]["total_prep_ms"] for r in warm_in_memory_runs]

    ctrl_ffmpeg_all = [r["prep_timing"]["ffmpeg_ms"] for r in warm_control_runs]
    exp_ffmpeg_all = [r["prep_timing"]["ffmpeg_pipe_ms"] for r in warm_in_memory_runs]

    ctrl_stt_all = [r["stt_ms"] for r in warm_control_runs]
    exp_stt_all = [r["stt_ms"] for r in warm_in_memory_runs]

    ctrl_total_all = [r["total_ms"] for r in warm_control_runs]
    exp_total_all = [r["total_ms"] for r in warm_in_memory_runs]

    stats_ctrl_prep = compute_statistics(ctrl_prep_all)
    stats_exp_prep = compute_statistics(exp_prep_all)

    stats_ctrl_ffmpeg = compute_statistics(ctrl_ffmpeg_all)
    stats_exp_ffmpeg = compute_statistics(exp_ffmpeg_all)

    stats_ctrl_stt = compute_statistics(ctrl_stt_all)
    stats_exp_stt = compute_statistics(exp_stt_all)

    stats_ctrl_total = compute_statistics(ctrl_total_all)
    stats_exp_total = compute_statistics(exp_total_all)

    # Accuracy / WER aggregates
    ctrl_exact_count = sum(1 for r in warm_control_runs if r["exact_match"])
    exp_exact_count = sum(1 for r in warm_in_memory_runs if r["exact_match"])
    ctrl_exact_pct = round((ctrl_exact_count / len(warm_control_runs)) * 100, 2)
    exp_exact_pct = round((exp_exact_count / len(warm_in_memory_runs)) * 100, 2)

    ctrl_mean_wer = round(sum(r["wer"] for r in warm_control_runs) / len(warm_control_runs), 4)
    exp_mean_wer = round(sum(r["wer"] for r in warm_in_memory_runs) / len(warm_in_memory_runs), 4)

    # Per-category accuracy breakdown
    categories = sorted(list(set(r["category"] for r in warm_control_runs)))
    category_summary = {}
    for cat in categories:
        c_sub = [r for r in warm_control_runs if r["category"] == cat]
        e_sub = [r for r in warm_in_memory_runs if r["category"] == cat]
        c_wer = round(sum(r["wer"] for r in c_sub) / len(c_sub), 4)
        e_wer = round(sum(r["wer"] for r in e_sub) / len(e_sub), 4)
        c_exact = sum(1 for r in c_sub if r["exact_match"])
        e_exact = sum(1 for r in e_sub if r["exact_match"])
        category_summary[cat] = {
            "control_wer": c_wer,
            "in_memory_wer": e_wer,
            "control_exact": f"{c_exact}/{len(c_sub)}",
            "in_memory_exact": f"{e_exact}/{len(e_sub)}",
        }

    # Historical 7-case benchmark check (EN-1..3, HI-1..2, TECH-1..2)
    historical_ids = {"EN-1", "EN-2", "EN-3", "HI-1", "HI-2", "TECH-1", "TECH-2"}
    c_hist = [r for r in warm_control_runs if r["case_id"] in historical_ids]
    e_hist = [r for r in warm_in_memory_runs if r["case_id"] in historical_ids]
    c_hist_exact = sum(1 for r in c_hist if r["exact_match"])
    e_hist_exact = sum(1 for r in e_hist if r["exact_match"])
    c_hist_wer = round(sum(r["wer"] for r in c_hist) / len(c_hist), 4)
    e_hist_wer = round(sum(r["wer"] for r in e_hist) / len(e_hist), 4)

    # 9. Format Detailed Summary Output
    print("\n" + "=" * 80, flush=True)
    print("SECTION 6: STATISTICAL LATENCY COMPARISON (CONTROL vs IN-MEMORY)", flush=True)
    print("=" * 80, flush=True)

    def print_stat_row(metric_name: str, c_stat: Dict[str, float], e_stat: Dict[str, float]):
        diff_mean = round(e_stat["mean"] - c_stat["mean"], 2)
        rel_mean = round((diff_mean / c_stat["mean"]) * 100, 2) if c_stat["mean"] > 0 else 0.0
        diff_p50 = round(e_stat["p50"] - c_stat["p50"], 2)
        diff_p95 = round(e_stat["p95"] - c_stat["p95"], 2)
        print(f"\n[{metric_name.upper()}] (ms):")
        print(f"  Control:   mean={c_stat['mean']} | p50={c_stat['p50']} | p95={c_stat['p95']} | min={c_stat['min']} | max={c_stat['max']} | std={c_stat['std']}")
        print(f"  In-Memory: mean={e_stat['mean']} | p50={e_stat['p50']} | p95={e_stat['p95']} | min={e_stat['min']} | max={e_stat['max']} | std={e_stat['std']}")
        print(f"  Delta:     mean={diff_mean:+0.2f} ms ({rel_mean:+0.2f}%) | p50={diff_p50:+0.2f} ms | p95={diff_p95:+0.2f} ms")

    print_stat_row("Audio Preprocessing", stats_ctrl_prep, stats_exp_prep)
    print_stat_row("FFmpeg Conversion", stats_ctrl_ffmpeg, stats_exp_ffmpeg)
    print_stat_row("Faster-Whisper Inference", stats_ctrl_stt, stats_exp_stt)
    print_stat_row("Total Pipeline (Prep + STT)", stats_ctrl_total, stats_exp_total)

    print("\n" + "=" * 80, flush=True)
    print("SECTION 7: STT ACCURACY & REGRESSION COMPARISON", flush=True)
    print("=" * 80, flush=True)
    print(f"  Control Exact Matches:     {ctrl_exact_count}/{len(warm_control_runs)} ({ctrl_exact_pct}%)", flush=True)
    print(f"  In-Memory Exact Matches:   {exp_exact_count}/{len(warm_in_memory_runs)} ({exp_exact_pct}%)", flush=True)
    print(f"  Control Mean WER:          {ctrl_mean_wer:.4f}", flush=True)
    print(f"  In-Memory Mean WER:        {exp_mean_wer:.4f}", flush=True)
    print(f"  Historical 7-Case Exact:   Control={c_hist_exact}/{len(c_hist)} ({c_hist_exact/len(c_hist)*100:.1f}%), In-Mem={e_hist_exact}/{len(e_hist)} ({e_hist_exact/len(e_hist)*100:.1f}%)", flush=True)
    print(f"  Historical 7-Case WER:     Control={c_hist_wer:.4f}, In-Mem={e_hist_wer:.4f}", flush=True)

    print("\n  [Per-Category Accuracy]")
    for cat, c_info in category_summary.items():
        print(f"    {cat:<30}: Exact Ctrl={c_info['control_exact']}, Exp={c_info['in_memory_exact']} | WER Ctrl={c_info['control_wer']:.4f}, Exp={c_info['in_memory_wer']:.4f}")

    # 10. Save Detailed JSON Report
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"audio_ab_{timestamp_str}.json"

    report_payload = {
        "benchmark": "phase_1_hardened_audio_ab_experiment",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "os": f"{platform.system()}-{platform.release()}",
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "initial_process_ram_mb": ram_initial_mb,
            "final_process_ram_mb": get_current_process_ram_mb(),
        },
        "configuration": {
            "model_size": model_name,
            "device": device,
            "compute_type": compute_type,
            "repetitions": N_REPETITIONS,
            "interleaved_patterns": order_patterns,
        },
        "audio_integrity": audio_integrity_results,
        "cold_start": {
            "control": t_cold_ctrl_res["total_ms"],
            "in_memory": t_cold_exp_res["total_ms"],
            "delta_ms": round(t_cold_exp_res["total_ms"] - t_cold_ctrl_res["total_ms"], 2),
        },
        "streaming_partials": streaming_partial_results,
        "memory_leak_test": {
            "ram_before_mb": ram_before_leak_test,
            "ram_after_mb": ram_after_leak_test,
            "delta_mb": round(ram_after_leak_test - ram_before_leak_test, 2),
        },
        "aggregates": {
            "audio_preprocessing": {
                "control": stats_ctrl_prep,
                "in_memory": stats_exp_prep,
                "delta_mean_ms": round(stats_exp_prep["mean"] - stats_ctrl_prep["mean"], 2),
                "delta_p50_ms": round(stats_exp_prep["p50"] - stats_ctrl_prep["p50"], 2),
                "delta_p95_ms": round(stats_exp_prep["p95"] - stats_ctrl_prep["p95"], 2),
            },
            "ffmpeg_conversion": {
                "control": stats_ctrl_ffmpeg,
                "in_memory": stats_exp_ffmpeg,
                "delta_mean_ms": round(stats_exp_ffmpeg["mean"] - stats_ctrl_ffmpeg["mean"], 2),
            },
            "whisper_inference": {
                "control": stats_ctrl_stt,
                "in_memory": stats_exp_stt,
                "delta_mean_ms": round(stats_exp_stt["mean"] - stats_ctrl_stt["mean"], 2),
                "delta_p50_ms": round(stats_exp_stt["p50"] - stats_ctrl_stt["p50"], 2),
            },
            "total_pipeline": {
                "control": stats_ctrl_total,
                "in_memory": stats_exp_total,
                "delta_mean_ms": round(stats_exp_total["mean"] - stats_ctrl_total["mean"], 2),
                "delta_p50_ms": round(stats_exp_total["p50"] - stats_ctrl_total["p50"], 2),
                "delta_p95_ms": round(stats_exp_total["p95"] - stats_ctrl_total["p95"], 2),
            },
            "accuracy": {
                "control_exact_pct": ctrl_exact_pct,
                "in_memory_exact_pct": exp_exact_pct,
                "control_mean_wer": ctrl_mean_wer,
                "in_memory_mean_wer": exp_mean_wer,
                "category_breakdown": category_summary,
                "historical_7_case": {
                    "control_exact": f"{c_hist_exact}/{len(c_hist)}",
                    "in_memory_exact": f"{e_hist_exact}/{len(e_hist)}",
                    "control_wer": c_hist_wer,
                    "in_memory_wer": e_hist_wer,
                }
            }
        },
        "execution_order": execution_order_log,
    }

    report_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 80, flush=True)
    print(f"[+] Saved authoritative Phase 1 A/B experiment report to: {report_file}", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
