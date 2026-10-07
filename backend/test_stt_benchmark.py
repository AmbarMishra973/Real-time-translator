"""
Reproducible STT Benchmark & Regression Harness (Phase 1G & 1H)
Measures Word Error Rate (WER), exact match, latency, and signal levels across:
1. English Conversational ("Hello", "What is your name?", "How are you doing today?")
2. Hindi Conversational ("आपका नाम क्या है?", "आप कैसे हैं?")
3. Technical Domain ("We need to implement a vector database with RAG.", "Kubernetes orchestration.")
"""

import os
import sys
import io
import re
import time
import math
import wave
import array
import asyncio
from pathlib import Path

# Ensure UTF-8 line-buffering output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

import edge_tts
from faster_whisper import WhisperModel
from backend.audio_diagnostics import (
    inspect_pcm16_wav,
    boost_quiet_pcm16_wav,
    pad_pcm16_wav,
)

def calculate_wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """
    Standard Word Error Rate (WER) using Levenshtein distance on word tokens.
    Returns: (wer_ratio, edit_distance, reference_length)
    """
    # Clean punctuation and normalize whitespace
    ref_clean = re.sub(r'[^\w\s]', '', reference.lower()).strip()
    hyp_clean = re.sub(r'[^\w\s]', '', hypothesis.lower()).strip()

    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()

    if not ref_words:
        return (0.0 if not hyp_words else 1.0, len(hyp_words), 0)

    r_len, h_len = len(ref_words), len(hyp_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # substitution
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j] + 1       # deletion
                )

    edits = dp[r_len][h_len]
    wer = edits / float(r_len)
    return round(wer, 4), edits, r_len

async def generate_speech_wav(text: str, voice: str) -> bytes:
    """Generate audio for testing via Edge-TTS and convert to 16kHz mono WAV."""
    comm = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    raw_mp3 = buf.getvalue()

    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as in_f:
        in_f.write(raw_mp3)
        in_p = in_f.name
    out_p = in_p + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_p, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_p],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        with open(out_p, "rb") as f:
            wav_bytes = f.read()
    finally:
        for p in [in_p, out_p]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
    return wav_bytes

def scale_wav(wav_bytes: bytes, target_rms: float) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        ch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = array.array('h', frames)
    ms = sum(s*s for s in samples) / len(samples) if samples else 0
    cur_rms = 20 * math.log10(max(math.sqrt(ms) / 32768.0, 1e-12))
    factor = 10.0 ** ((target_rms - cur_rms) / 20.0)
    scaled = array.array('h', [max(-32768, min(32767, int(round(s * factor)))) for s in samples])
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(scaled.tobytes())
    return buf.getvalue()

# Benchmark Dataset as specified in Phase 1G
BENCHMARK_DATASET = [
    {
        "id": "EN-1",
        "category": "English Conversational",
        "expected": "Hello",
        "lang": "en",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "EN-2",
        "category": "English Conversational",
        "expected": "What is your name?",
        "lang": "en",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "EN-3",
        "category": "English Conversational",
        "expected": "How are you doing today?",
        "lang": "en",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "HI-1",
        "category": "Hindi Conversational",
        "expected": "आपका नाम क्या है?",
        "lang": "hi",
        "voice": "hi-IN-SwaraNeural"
    },
    {
        "id": "HI-2",
        "category": "Hindi Conversational",
        "expected": "आप कैसे हैं?",
        "lang": "hi",
        "voice": "hi-IN-SwaraNeural"
    },
    {
        "id": "TECH-1",
        "category": "Technical English",
        "expected": "We need to implement a vector database with RAG.",
        "lang": "en",
        "voice": "en-US-JennyNeural"
    },
    {
        "id": "TECH-2",
        "category": "Technical English",
        "expected": "Kubernetes orchestration.",
        "lang": "en",
        "voice": "en-US-JennyNeural"
    },
]

async def run_benchmark():
    print("=" * 75, flush=True)
    print("STT REGRESSION & BENCHMARK HARNESS (PHASE 1G & 1H)", flush=True)
    print("=" * 75, flush=True)

    # Cache pre-generated audio files in test_audio directory
    audio_dir = Path("backend/debug_audio/benchmark_samples")
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Preparing 7 reference audio clips in {audio_dir}...", flush=True)
    for item in BENCHMARK_DATASET:
        fpath = audio_dir / f"{item['id']}.wav"
        if not fpath.exists():
            wav = await generate_speech_wav(item["expected"], item["voice"])
            fpath.write_bytes(wav)
            print(f"  + Generated {item['id']}: '{item['expected']}'", flush=True)
        else:
            print(f"  ✓ Cached {item['id']}: '{item['expected']}'", flush=True)

    # Test Local Whisper model
    model_name = os.getenv("WHISPER_SIZE", "base")
    print(f"\n[*] Initializing Local Faster-Whisper (model='{model_name}', device='cpu', compute_type='int8')...", flush=True)
    t0 = time.perf_counter()
    whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"[+] Loaded Faster-Whisper in {time.perf_counter() - t0:.2f}s\n", flush=True)

    results = []

    print("-" * 75, flush=True)
    print(f"{'ID':<7} | {'Category':<22} | {'Exact':<5} | {'WER':<6} | {'Lat':<5} | {'RMS':<6} | {'Transcript'}", flush=True)
    print("-" * 75, flush=True)

    for item in BENCHMARK_DATASET:
        fpath = audio_dir / f"{item['id']}.wav"
        wav_bytes = fpath.read_bytes()
        diag = inspect_pcm16_wav(wav_bytes)

        # Apply standard DSP pipeline (adaptive gain + padding for sub-1s clips)
        boosted_wav, gain_db = boost_quiet_pcm16_wav(wav_bytes)
        post_diag = inspect_pcm16_wav(boosted_wav)
        if post_diag["duration_s"] < 1.0:
            final_wav, _ = pad_pcm16_wav(boosted_wav, 250)
        else:
            final_wav = boosted_wav

        # Transcribe with script guidance for Hindi to prevent Perso-Arabic transcription
        initial_prompt = "यह हिंदी में बातचीत है।" if item["lang"] == "hi" else None
        t_start = time.perf_counter()
        segments, info = whisper_model.transcribe(
            io.BytesIO(final_wav),
            language=item["lang"],
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt
        )
        actual = " ".join(s.text for s in segments).strip()
        latency = round(time.perf_counter() - t_start, 2)

        # Compare
        wer, edits, ref_words_len = calculate_wer(item["expected"], actual)
        
        # Exact match normalization
        exp_norm = re.sub(r'[^\w\s]', '', item["expected"].lower()).strip()
        act_norm = re.sub(r'[^\w\s]', '', actual.lower()).strip()
        exact_match = (exp_norm == act_norm)

        res = {
            "id": item["id"],
            "category": item["category"],
            "expected": item["expected"],
            "actual": actual,
            "exact_match": exact_match,
            "wer": wer,
            "latency": latency,
            "duration_s": diag["duration_s"],
            "rms_dbfs": diag["rms_dbfs"],
            "gain_applied": gain_db,
            "model_config": f"Faster-Whisper ({model_name}, cpu, beam=1, vad=False)"
        }
        results.append(res)

        print(f"{res['id']:<7} | {res['category']:<22} | {str(res['exact_match']):<5} | {res['wer']:<6.2f} | {res['latency']:<4.2f}s | {res['rms_dbfs']:<5.1f} | \"{res['actual']}\"", flush=True)

    print("-" * 75, flush=True)
    exact_count = sum(1 for r in results if r["exact_match"])
    mean_wer = sum(r["wer"] for r in results) / len(results)
    mean_lat = sum(r["latency"] for r in results) / len(results)
    print(f"\n[BENCHMARK SUMMARY]", flush=True)
    print(f"  Exact Matches:        {exact_count}/{len(results)} ({exact_count/len(results)*100:.1f}%)", flush=True)
    print(f"  Average WER:          {mean_wer:.4f}", flush=True)
    print(f"  Average Latency:      {mean_lat:.2f}s", flush=True)
    print(f"  Total Test Cases:     {len(results)}", flush=True)
    print("=" * 75, flush=True)

    return results

if __name__ == "__main__":
    asyncio.run(run_benchmark())
