"""
P9 EXPERIMENT 5 -- Partial Reusability vs Coverage Threshold
============================================================
DIAGNOSTIC ONLY. Does NOT modify any production code.

Question: At what coverage threshold (50%, 60%, 70%, 80%, 90%, 100%) is a partial
          transcript safe to reuse as the final transcript without missing words,
          truncation, or degrading WER?

Methodology:
  1. Run full 100% audio to establish the authoritative final transcript (reference).
  2. For coverage levels C in [50%, 60%, 70%, 80%, 90%, 100%]:
     - Slice the first C% of audio PCM.
     - Decode the slice using Whisper (same greedy config as streaming partial).
     - Compare partial against authoritative final:
       * exact_match (case/punctuation normalized)
       * WER (Word Error Rate)
       * is_truncated (partial has fewer words than authoritative)
       * missing_trailing_words (words from end of authoritative missing in partial)
       * complete_speech_ending (ends with exact same word as authoritative)
       * decode latency
  3. Evaluate reusability trade-off:
     - If threshold T is accepted: what is the risk of truncation vs latency savings?

Run: python -m backend.exp5_partial_reusability
Output: backend/exp_results/exp5_partial_reusability.txt
"""

import io
import os
import re
import sys
import time
import wave
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("WHISPER_SIZE", "base")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_COMPUTE_TYPE", "int8")
os.environ.setdefault("WHISPER_CPU_THREADS", "2")
os.environ.setdefault("STT_ENGINE", "local")
os.environ.setdefault("AUDIO_PIPELINE_MODE", "in_memory")

from backend.services.stt_service import stt_service
from backend.test_stt_benchmark import calculate_wer

SEP = "=" * 80
OUT_DIR = Path("backend/exp_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "exp5_partial_reusability.txt"

LINES = []

def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))

def normalize_text(text: str) -> str:
    cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
    return " ".join(cleaned.split())

def slice_wav(wav_bytes: bytes, pct: float) -> bytes:
    """Slice first pct% (0.0 to 1.0) of PCM frames from WAV and rewrite header."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw_pcm = wf.readframes(nframes)

    target_frames = max(1, int(nframes * pct))
    bytes_per_frame = nchannels * sampwidth
    target_bytes = target_frames * bytes_per_frame
    sliced_pcm = raw_pcm[:target_bytes]

    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as out_wf:
        out_wf.setnchannels(nchannels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)
        out_wf.writeframes(sliced_pcm)

    return out_io.getvalue()

def get_dur(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()

def main():
    log(SEP)
    log("P9 EXPERIMENT 5 -- Empirical Partial Reusability vs Coverage Threshold")
    log(SEP)
    log(f"Model: {stt_service.model_size} | Threads: {stt_service.cpu_threads} | Device: {stt_service.device}")

    # Dataset: Short, Medium, Long recordings covering English and Hindi
    p8_dir = Path("backend/debug_audio/benchmark_samples_p8")
    test_files = [
        # (filename, lang, category)
        ("EN-S1.wav", "en", "Short EN (2.3s)"),
        ("EN-S2.wav", "en", "Short EN (2.1s)"),
        ("HI-1.wav",  "hi", "Short HI (3.0s)"),
        ("TECH-1.wav","en", "Medium EN (4.6s)"),
        ("HG-2.wav",  "hi", "Medium HG (5.0s)"),
        ("EN-L1.wav", "en", "Medium EN (5.9s)"),
        ("EN-L2.wav", "en", "Long EN (8.2s)"),
        ("EN-L3.wav", "en", "Long EN (9.0s)"),
        ("EN-L5.wav", "en", "Long EN (8.7s)"),
    ]

    # Verify files exist
    valid_files = []
    for fname, lang, cat in test_files:
        p = p8_dir / fname
        if p.exists():
            valid_files.append((p, lang, cat))
        else:
            log(f"[!] Warning: missing {p}")

    if not valid_files:
        log("[!] No benchmark files found!")
        return

    # Warmup
    log("\nWarming up Whisper...")
    stt_service.transcribe_partial(valid_files[0][0].read_bytes(), "en", "exp5", "wu", "wu")
    log("Warmup complete.\n")

    COVERAGES = [50, 60, 70, 80, 90, 100]

    all_data = [] # per-sample detailed results
    threshold_stats = {c: {
        "exact_matches": 0,
        "total": 0,
        "wer_sum": 0.0,
        "truncated_count": 0,
        "missing_trailing_words_sum": 0,
        "complete_ending_count": 0,
        "latency_sum": 0.0,
    } for c in COVERAGES}

    for path, lang, cat in valid_files:
        full_wav = path.read_bytes()
        full_dur = get_dur(full_wav)
        log(f"\n{SEP}")
        log(f"FILE: {path.name} | Category: {cat} | Full Duration: {full_dur:.2f}s | Lang: {lang}")
        log(SEP)

        # 1. Authoritative full 100% transcript
        t0 = time.perf_counter()
        auth_txt, _, _, _, _ = stt_service.transcribe(
            audio_bytes=full_wav,
            lang=lang,
            filename=path.name,
            content_type="audio/wav",
            session_id="exp5",
            request_id="auth",
            reason="exp5_authoritative"
        )
        auth_wall = time.perf_counter() - t0
        auth_norm = normalize_text(auth_txt)
        auth_words = auth_norm.split()
        log(f"Authoritative [100%]: \"{auth_txt}\" (wall={auth_wall:.2f}s, words={len(auth_words)})")

        sample_coverages = {}

        for cov in COVERAGES:
            pct = cov / 100.0
            cov_wav = slice_wav(full_wav, pct)
            cov_dur = get_dur(cov_wav)

            t0 = time.perf_counter()
            cov_txt = stt_service.transcribe_partial(
                cov_wav,
                lang,
                session_id="exp5",
                request_id=f"cov_{cov}",
                reason=f"exp5_cov_{cov}"
            )
            cov_wall = time.perf_counter() - t0

            cov_norm = normalize_text(cov_txt)
            cov_words = cov_norm.split()

            # Exact match
            exact = (cov_norm == auth_norm)

            # WER
            wer, edit_d, _ = calculate_wer(auth_norm, cov_norm)

            # Truncation check
            is_truncated = len(cov_words) < len(auth_words)

            # Trailing words missing
            # Compare suffix of auth_words that is missing in cov_words
            missing_trailing = []
            if len(cov_words) < len(auth_words):
                # Check how many words from the end of auth_words are not in cov_words
                overlap = 0
                for k in range(1, len(cov_words) + 1):
                    if auth_words[-k:] == cov_words[-k:]:
                        overlap = k
                missing_trailing = auth_words[len(cov_words):] if overlap == 0 else auth_words[:-overlap]

            # Does partial end with the authoritative sentence ending?
            complete_ending = (len(cov_words) > 0 and len(auth_words) > 0 and cov_words[-1] == auth_words[-1])

            sample_coverages[cov] = {
                "dur": cov_dur,
                "txt": cov_txt,
                "wall": cov_wall,
                "exact": exact,
                "wer": wer,
                "is_truncated": is_truncated,
                "missing_trailing_count": len(missing_trailing),
                "missing_trailing_words": " ".join(missing_trailing[-3:]), # last up to 3 words
                "complete_ending": complete_ending
            }

            # Update aggregate stats
            st = threshold_stats[cov]
            st["total"] += 1
            if exact:
                st["exact_matches"] += 1
            st["wer_sum"] += wer
            if is_truncated:
                st["truncated_count"] += 1
            st["missing_trailing_words_sum"] += len(missing_trailing)
            if complete_ending:
                st["complete_ending_count"] += 1
            st["latency_sum"] += cov_wall

            tag = "MATCH" if exact else ("TRUNC" if is_truncated else "DIFF")
            log(f"  [{cov:>3}% coverage ({cov_dur:.2f}s)]: tag={tag:<5} wer={wer:.2f} "
                f"words={len(cov_words):>2}/{len(auth_words):<2} wall={cov_wall:.2f}s -> \"{cov_txt}\"")
            if is_truncated and missing_trailing:
                log(f"         missing end: ...{' '.join(missing_trailing)}")

        all_data.append({
            "name": path.name,
            "cat": cat,
            "dur": full_dur,
            "auth_txt": auth_txt,
            "auth_wall": auth_wall,
            "coverages": sample_coverages
        })

    # Summary table across coverage thresholds
    log(f"\n{SEP}")
    log("EXPERIMENT 5 SUMMARY TABLE: METRICS BY COVERAGE THRESHOLD")
    log(SEP)
    log(f"{'Coverage':>8} {'Exact%':>8} {'Avg_WER':>9} {'Trunc%':>8} {'EndingMatch%':>14} {'AvgMissingWords':>16} {'AvgPartialWall':>15}")
    log("-" * 80)
    for cov in COVERAGES:
        st = threshold_stats[cov]
        n = max(1, st["total"])
        exact_pct = (st["exact_matches"] / n) * 100.0
        avg_wer = st["wer_sum"] / n
        trunc_pct = (st["truncated_count"] / n) * 100.0
        ending_pct = (st["complete_ending_count"] / n) * 100.0
        avg_missing = st["missing_trailing_words_sum"] / n
        avg_wall = st["latency_sum"] / n
        log(f"{cov:>7}% {exact_pct:>7.1f}% {avg_wer:>8.2f} {trunc_pct:>7.1f}% {ending_pct:>13.1f}% {avg_missing:>15.2f} {avg_wall:>13.2f}s")

    log(f"\n{SEP}")
    log("BREAKDOWN BY AUDIO LENGTH CATEGORY")
    log(SEP)
    for category in ["Short EN (2.3s)", "Short EN (2.1s)", "Short HI (3.0s)", "Medium EN (4.6s)", "Medium HG (5.0s)", "Medium EN (5.9s)", "Long EN (8.2s)", "Long EN (9.0s)", "Long EN (8.7s)"]:
        sample = next((s for s in all_data if s["cat"] == category), None)
        if not sample:
            continue
        log(f"\nSample: {sample['name']} ({sample['cat']})")
        log(f"{'Coverage':>8} {'Match':>7} {'WER':>6} {'Trunc':>7} {'EndMatch':>9} {'Transcript':<50}")
        log("-" * 80)
        for cov in COVERAGES:
            c = sample["coverages"][cov]
            m = "YES" if c["exact"] else "NO"
            tr = "YES" if c["is_truncated"] else "NO"
            em = "YES" if c["complete_ending"] else "NO"
            log(f"{cov:>7}% {m:>7} {c['wer']:>6.2f} {tr:>7} {em:>9} \"{c['txt'][:48]}\"")

    log(f"\n{SEP}")
    log("REUSABILITY DECISION CURVE & TRADEOFF ANALYSIS")
    log(SEP)
    log("If you set smart-finalization threshold T:")
    for cov in COVERAGES[:-1]:
        st = threshold_stats[cov]
        n = max(1, st["total"])
        exact_pct = (st["exact_matches"] / n) * 100.0
        trunc_pct = (st["truncated_count"] / n) * 100.0
        avg_missing = st["missing_trailing_words_sum"] / n
        log(f"  Threshold >= {cov}%:")
        log(f"    - Safe exact reuse rate : {exact_pct:.1f}%")
        log(f"    - Risk of truncating end : {trunc_pct:.1f}% (average {avg_missing:.1f} trailing words lost)")
        if trunc_pct == 0.0:
            log(f"    -> [SAFE THRESHOLD CANDIDATE] Zero truncation observed across test corpus.")
        else:
            log(f"    -> [UNSAFE FOR BLIND REUSE] Will cause truncated user sentences {trunc_pct:.1f}% of the time.")

    OUT_FILE.write_text("\n".join(LINES), encoding="utf-8")
    log(f"\n[+] Results saved: {OUT_FILE}")

if __name__ == "__main__":
    main()
