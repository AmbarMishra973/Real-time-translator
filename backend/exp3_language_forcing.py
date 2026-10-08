"""
P9 EXPERIMENT 3 -- Language Forcing vs Auto-Detection
======================================================
DIAGNOSTIC ONLY. Does NOT modify any production code.

Question: Does forcing language=en on Hindi/Hinglish audio cause transcript
          failure, and does language=None (auto-detect) fix it?

Conditions per file:
  A) language=en  (forced English -- simulates user leaving dropdown on English)
  B) language=hi  (forced Hindi   -- correct for Hindi files)
  C) language=None (auto-detect   -- what Whisper would choose)

Files tested:
  - HI-* samples  (Hindi speech)
  - HG-* samples  (Hinglish / code-switching)
  - EN-S* samples (English, as control -- all conditions should match)

Run:  python -m backend.exp3_language_forcing
Output: backend/exp_results/exp3_language_forcing.txt
"""

import io, os, sys, time, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("WHISPER_SIZE", "base")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_COMPUTE_TYPE", "int8")
os.environ.setdefault("WHISPER_CPU_THREADS", "2")
os.environ.setdefault("STT_ENGINE", "local")
os.environ.setdefault("AUDIO_PIPELINE_MODE", "in_memory")

from backend.services.stt_service import stt_service

SEP = "=" * 70
OUT_DIR = Path("backend/exp_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "exp3_language_forcing.txt"
LINES = []

def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))

def get_dur(b):
    with wave.open(io.BytesIO(b), "rb") as w:
        return w.getnframes() / w.getframerate()

def has_devanagari(text):
    return any('\u0900' <= c <= '\u097F' for c in text)

def has_latin(text):
    return any(c.isalpha() and c.isascii() for c in text)

def classify(text, expected_lang):
    """Classify whether transcript is plausibly in the expected script/language."""
    if not text or not text.strip():
        return "EMPTY"
    if expected_lang == "hi":
        if has_devanagari(text):
            return "CORRECT_DEVANAGARI"
        elif has_latin(text):
            return "WRONG_LATIN (transliterated or English words)"
        else:
            return "UNKNOWN_SCRIPT"
    elif expected_lang == "en":
        if has_latin(text) and not has_devanagari(text):
            return "CORRECT_LATIN"
        else:
            return "UNEXPECTED_SCRIPT"
    elif expected_lang == "hinglish":
        # Accept any mix
        if has_devanagari(text) or has_latin(text):
            return "CONTAINS_EXPECTED_CONTENT"
        return "EMPTY_OR_WRONG"
    return "UNCLASSIFIED"

TEST_FILES = [
    # (path, true_language_for_classification, display_label)
    ("backend/debug_audio/benchmark_samples_p8/HI-1.wav", "hi", "Hindi-1 (3.0s)"),
    ("backend/debug_audio/benchmark_samples_p8/HI-2.wav", "hi", "Hindi-2 (3.8s)"),
    ("backend/debug_audio/benchmark_samples_p8/HI-3.wav", "hi", "Hindi-3 (4.0s)"),
    ("backend/debug_audio/benchmark_samples_p8/HG-1.wav", "hinglish", "Hinglish-1 (2.7s)"),
    ("backend/debug_audio/benchmark_samples_p8/HG-2.wav", "hinglish", "Hinglish-2 (5.0s)"),
    ("backend/debug_audio/benchmark_samples_p8/EN-S1.wav", "en", "English-S1 (2.3s)"),
    ("backend/debug_audio/benchmark_samples_p8/EN-S2.wav", "en", "English-S2 (2.1s)"),
]

def run_one(audio_bytes, lang_param, label, true_lang):
    t0 = time.perf_counter()
    # lang_param=None means auto-detect (pass None to transcribe)
    txt, info, diag, engine, model = stt_service.transcribe(
        audio_bytes=audio_bytes,
        lang=lang_param or "auto",  # "auto" routes to whisper_lang=None
        filename="exp3.wav",
        content_type="audio/wav",
        session_id="exp3",
        request_id=f"exp3_{label}",
        reason="exp3"
    )
    wall = time.perf_counter() - t0
    detected = getattr(info, "language", "unknown") if info else "unknown"
    classification = classify(txt, true_lang)
    return {"txt": txt, "wall_s": wall, "detected_lang": detected, "classification": classification}

def run_file(path_str, true_lang, file_label):
    path = Path(path_str)
    if not path.exists():
        log(f"[!] Missing {path}")
        return None
    audio_bytes = path.read_bytes()
    dur = get_dur(audio_bytes)
    log(f"\n{SEP}")
    log(f"FILE: {file_label}  ({path.name}, true_lang={true_lang})")
    log(SEP)

    rA = run_one(audio_bytes, "en",   f"{true_lang}_en",   true_lang)
    rB = run_one(audio_bytes, "hi",   f"{true_lang}_hi",   true_lang)
    rC = run_one(audio_bytes, None,   f"{true_lang}_auto", true_lang)

    log(f"  [A] lang=en   ({rA['wall_s']:.3f}s, detected={rA['detected_lang']})")
    log(f"      transcript: '{rA['txt']}'")
    log(f"      classify:   {rA['classification']}")
    log(f"  [B] lang=hi   ({rB['wall_s']:.3f}s, detected={rB['detected_lang']})")
    log(f"      transcript: '{rB['txt']}'")
    log(f"      classify:   {rB['classification']}")
    log(f"  [C] lang=auto ({rC['wall_s']:.3f}s, detected={rC['detected_lang']})")
    log(f"      transcript: '{rC['txt']}'")
    log(f"      classify:   {rC['classification']}")

    # Key diagnostic
    if true_lang == "hi":
        if rA["classification"] != "CORRECT_DEVANAGARI" and rC["classification"] == "CORRECT_DEVANAGARI":
            log(f"  >> CONFIRMED: lang=en fails for Hindi, lang=auto produces Devanagari")
        elif rA["classification"] == "CORRECT_DEVANAGARI":
            log(f"  NOTE: lang=en still produced Devanagari (Whisper may override forced lang)")
        if rB["classification"] == "CORRECT_DEVANAGARI":
            log(f"  >> lang=hi gives correct Devanagari output")
    elif true_lang == "en":
        all_match = (rA["txt"].strip().lower() == rC["txt"].strip().lower())
        log(f"  Control check (en): A vs C match = {'YES' if all_match else 'NO'}")

    return {"label": file_label, "true_lang": true_lang, "dur": dur,
            "A": rA, "B": rB, "C": rC}

def main():
    log(SEP)
    log("P9 EXP 3 -- Language Forcing vs Auto-Detection")
    log("Does forcing lang=en on Hindi/Hinglish audio cause failure?")
    log(SEP)
    log(f"Model={stt_service.model_size} threads={stt_service.cpu_threads}")

    log("\n[*] Warmup...")
    w = Path("backend/debug_audio/benchmark_samples_p8/EN-S1.wav")
    if w.exists():
        stt_service.transcribe_partial(w.read_bytes(), "en", "wu", "wu", "wu")
    log("[+] Done\n")

    results = []
    for path, true_lang, label in TEST_FILES:
        r = run_file(path, true_lang, label)
        if r:
            results.append(r)

    log(f"\n{SEP}\nSUMMARY TABLE\n{SEP}")
    log(f"{'File':<22} {'TrueLang':>9} {'A(en)':>20} {'B(hi)':>20} {'C(auto)':>20}")
    for r in results:
        log(f"{r['label']:<22} {r['true_lang']:>9} "
            f"{r['A']['classification']:>20} {r['B']['classification']:>20} {r['C']['classification']:>20}")

    log("\nKEY QUESTIONS:")
    log("  1. For Hindi files: does lang=en produce garbage? (expected: YES)")
    log("  2. For Hindi files: does lang=auto produce Devanagari? (expected: YES)")
    log("  3. For English files: do all conditions produce the same output? (expected: YES)")

    OUT_FILE.write_text("\n".join(LINES), encoding="utf-8")
    log(f"\n[+] Results saved: {OUT_FILE}")

if __name__ == "__main__":
    main()
