"""
P9 EXPERIMENT 4 -- CPU Contention vs Whisper Output Quality
=============================================================
DIAGNOSTIC ONLY. Does NOT modify any production code.

Question: Does running Whisper under heavy CPU contention (another concurrent
          Whisper inference or CPU-saturating burn) degrade transcription *quality*
          (i.e. different text, missing words, hallucinations, corruption),
          or does it ONLY degrade transcription *latency*?

Methodology:
  Take test audio samples across English, Hindi, Hinglish, and Technical speech.
  Run 3 conditions:
    1. ISOLATED: Single Whisper invocation on quiet machine.
    2. WHISPER-CONTENTION: Whisper invocation running concurrently alongside
       another active Whisper inference thread (competing for same CPU cores/threads).
    3. CPU-BURN CONTENTION: Whisper invocation running while CPU is 100% saturated
       with synthetic multi-threaded math workers.

  Metrics:
    - Text exact match (True/False)
    - Normalized text match (case/punctuation-insensitive)
    - Character Edit Distance / Levenshtein
    - Latency (seconds) & Slowdown ratio

Run: python -m backend.exp4_contention_quality
Output: backend/exp_results/exp4_contention_quality.txt
"""

import io
import os
import sys
import time
import wave
import threading
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("WHISPER_SIZE", "base")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_COMPUTE_TYPE", "int8")
os.environ.setdefault("WHISPER_CPU_THREADS", "2")
os.environ.setdefault("STT_ENGINE", "local")
os.environ.setdefault("AUDIO_PIPELINE_MODE", "in_memory")

from backend.services.stt_service import stt_service

SEP = "=" * 80
OUT_DIR = Path("backend/exp_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "exp4_contention_quality.txt"

LINES = []

def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))

def get_dur(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dels = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]

def normalize_text(t):
    return " ".join("".join(c.lower() for c in t if c.isalnum() or c.isspace()).split())

# CPU burner worker
def cpu_burner(stop_evt):
    x = 1.0001
    while not stop_evt.is_set():
        x = (x * 1.00001) % 1000.0

def run_isolated(audio_bytes, lang, sample_id):
    t0 = time.perf_counter()
    txt, detected_lang, dur, meta, pcm = stt_service.transcribe(
        audio_bytes=audio_bytes,
        lang=lang,
        filename=f"{sample_id}.wav",
        content_type="audio/wav",
        session_id="exp4",
        request_id=f"iso_{sample_id}",
        reason="exp4_isolated"
    )
    wall = time.perf_counter() - t0
    return {"text": txt.strip(), "wall": wall, "lang": detected_lang}

def run_whisper_contention(audio_bytes, lang, sample_id, background_audio):
    stop_bg = threading.Event()
    bg_done = threading.Event()

    def bg_whisper_worker():
        while not stop_bg.is_set():
            try:
                stt_service.transcribe(
                    audio_bytes=background_audio,
                    lang="en",
                    filename="bg.wav",
                    content_type="audio/wav",
                    session_id="exp4_bg",
                    request_id="bg",
                    reason="exp4_bg_contention"
                )
            except Exception:
                pass
        bg_done.set()

    bg_thread = threading.Thread(target=bg_whisper_worker, daemon=True)
    bg_thread.start()
    time.sleep(0.1) # Ensure background Whisper is actively spinning

    t0 = time.perf_counter()
    try:
        txt, detected_lang, dur, meta, pcm = stt_service.transcribe(
            audio_bytes=audio_bytes,
            lang=lang,
            filename=f"{sample_id}.wav",
            content_type="audio/wav",
            session_id="exp4",
            request_id=f"wcont_{sample_id}",
            reason="exp4_whisper_contention"
        )
    finally:
        stop_bg.set()
        bg_thread.join(timeout=5.0)

    wall = time.perf_counter() - t0
    return {"text": txt.strip(), "wall": wall, "lang": detected_lang}

def run_cpu_burn_contention(audio_bytes, lang, sample_id):
    stop_evt = threading.Event()
    num_burners = max(2, os.cpu_count() or 4)
    threads = [threading.Thread(target=cpu_burner, args=(stop_evt,), daemon=True) for _ in range(num_burners)]
    for t in threads:
        t.start()
    time.sleep(0.1) # Let CPU saturate

    t0 = time.perf_counter()
    try:
        txt, detected_lang, dur, meta, pcm = stt_service.transcribe(
            audio_bytes=audio_bytes,
            lang=lang,
            filename=f"{sample_id}.wav",
            content_type="audio/wav",
            session_id="exp4",
            request_id=f"cpuburn_{sample_id}",
            reason="exp4_cpu_burn"
        )
    finally:
        stop_evt.set()
        for t in threads:
            t.join(timeout=1.0)

    wall = time.perf_counter() - t0
    return {"text": txt.strip(), "wall": wall, "lang": detected_lang}

def main():
    log(SEP)
    log("P9 EXPERIMENT 4 -- CPU Contention vs Whisper Transcription Quality")
    log(SEP)
    log(f"Model: {stt_service.model_size} | Device: {stt_service.device} | "
        f"Compute: {stt_service.compute_type} | Threads: {stt_service.cpu_threads} | System CPUs: {os.cpu_count()}")

    # Find test files
    samples = []
    p8_dir = Path("backend/debug_audio/benchmark_samples_p8")
    p7_dir = Path("backend/debug_audio/benchmark_samples")

    search_dirs = [p8_dir, p7_dir]
    for sdir in search_dirs:
        if sdir.exists():
            for f in sorted(sdir.glob("*.wav")):
                name = f.stem
                if any(x[0].stem == name for x in samples):
                    continue
                lang = "en"
                if name.startswith("HI"):
                    lang = "hi"
                elif name.startswith("HG"):
                    lang = "hi"
                samples.append((f, lang, name))

    log(f"Found {len(samples)} distinct audio benchmark files.")

    # Warmup
    log("\nWarming up Whisper...")
    if samples:
        stt_service.transcribe_partial(samples[0][0].read_bytes(), "en", "exp4", "warmup", "warmup")
    log("Warmup complete.\n")

    # Background audio for whisper contention
    bg_audio = samples[-1][0].read_bytes() if samples else None

    results = []

    for path, lang, name in samples[:8]: # Test top 8 diverse samples
        audio_b = path.read_bytes()
        dur = get_dur(audio_b)
        log(f"\n--- Testing Sample: {name} ({dur:.2f}s, lang={lang}) ---")

        # 1. Isolated
        iso = run_isolated(audio_b, lang, name)
        log(f"  [1] Isolated       : wall={iso['wall']:.3f}s -> \"{iso['text']}\"")

        # 2. Whisper Contention
        wcont = run_whisper_contention(audio_b, lang, name, bg_audio)
        log(f"  [2] Whisper-Contend: wall={wcont['wall']:.3f}s (x{wcont['wall']/max(iso['wall'],0.001):.2f}) -> \"{wcont['text']}\"")

        # 3. CPU Burn Contention
        cburn = run_cpu_burn_contention(audio_b, lang, name)
        log(f"  [3] CPU-Burn       : wall={cburn['wall']:.3f}s (x{cburn['wall']/max(iso['wall'],0.001):.2f}) -> \"{cburn['text']}\"")

        # Diff analysis
        norm_iso = normalize_text(iso['text'])
        norm_wcont = normalize_text(wcont['text'])
        norm_cburn = normalize_text(cburn['text'])

        exact_match_w = (iso['text'] == wcont['text'])
        exact_match_c = (iso['text'] == cburn['text'])
        norm_match_w = (norm_iso == norm_wcont)
        norm_match_c = (norm_iso == norm_cburn)

        edit_dist_w = levenshtein(norm_iso, norm_wcont)
        edit_dist_c = levenshtein(norm_iso, norm_cburn)

        results.append({
            "name": name,
            "dur": dur,
            "iso_wall": iso['wall'],
            "wcont_wall": wcont['wall'],
            "cburn_wall": cburn['wall'],
            "wcont_ratio": wcont['wall'] / max(iso['wall'], 0.001),
            "cburn_ratio": cburn['wall'] / max(iso['wall'], 0.001),
            "iso_text": iso['text'],
            "wcont_text": wcont['text'],
            "cburn_text": cburn['text'],
            "exact_match_w": exact_match_w,
            "exact_match_c": exact_match_c,
            "norm_match_w": norm_match_w,
            "norm_match_c": norm_match_c,
            "edit_dist_w": edit_dist_w,
            "edit_dist_c": edit_dist_c
        })

    log(f"\n{SEP}")
    log("EXPERIMENT 4 SUMMARY TABLE")
    log(SEP)
    log(f"{'Sample':<12} {'Dur':>5} {'Iso(s)':>7} {'WCont(s)':>9} {'Ratio_W':>8} {'CBurn(s)':>9} {'Ratio_C':>8} {'W_Match':>8} {'C_Match':>8}")
    log("-" * 80)
    for r in results:
        w_m = "IDENTICAL" if r['exact_match_w'] else f"DIFF(d={r['edit_dist_w']})"
        c_m = "IDENTICAL" if r['exact_match_c'] else f"DIFF(d={r['edit_dist_c']})"
        log(f"{r['name']:<12} {r['dur']:>4.1f}s {r['iso_wall']:>7.2f} {r['wcont_wall']:>9.2f} {r['wcont_ratio']:>7.2f}x {r['cburn_wall']:>9.2f} {r['cburn_ratio']:>7.2f}x {w_m:>8} {c_m:>8}")

    log("\nQUALITY VERDICT:")
    all_exact_w = all(r['exact_match_w'] for r in results)
    all_exact_c = all(r['exact_match_c'] for r in results)
    all_norm_w = all(r['norm_match_w'] for r in results)
    all_norm_c = all(r['norm_match_c'] for r in results)

    if all_exact_w and all_exact_c:
        log("  [CONFIRMED HYPOTHESIS REFUTED] CPU contention caused 0% quality difference (100% identical outputs).")
        log("  CPU contention only causes LATENCY slowdown, NOT quality/accuracy degradation.")
    elif all_norm_w and all_norm_c:
        log("  [MINOR FORMATTING ONLY] Normalized transcripts are 100% identical across all conditions.")
        log("  No semantic degradation observed under CPU contention.")
    else:
        log("  [QUALITY DEGRADATION DETECTED] Transcripts differed under contention:")
        for r in results:
            if not r['exact_match_w']:
                log(f"    - {r['name']} (WCont): '{r['iso_text']}' vs '{r['wcont_text']}'")
            if not r['exact_match_c']:
                log(f"    - {r['name']} (CBurn): '{r['iso_text']}' vs '{r['cburn_text']}'")

    OUT_FILE.write_text("\n".join(LINES), encoding="utf-8")
    log(f"\n[+] Results written to {OUT_FILE}")

if __name__ == "__main__":
    main()
