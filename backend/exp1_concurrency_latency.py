"""
P9 EXPERIMENT 1 -- Concurrency Latency Proof
=============================================
DIAGNOSTIC ONLY. Does NOT modify any production code.

Hypothesis: P8 concurrent partial+final Whisper execution on the same CPU
            is the primary cause of the 15.67s wall-clock STT latency.

Three conditions on the SAME audio:
  A) Final-only  : single Whisper call  -> true single-pass baseline
  B) Sequential  : partial completes THEN final starts -> serialized baseline
  C) Concurrent  : both launched simultaneously  -> simulates actual P8 behavior

Run:  python -m backend.exp1_concurrency_latency
Output: backend/exp_results/exp1_concurrency_latency.txt
"""

import io, os, sys, time, wave, threading, concurrent.futures
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
OUT_FILE = OUT_DIR / "exp1_concurrency_latency.txt"

AUDIO_SHORT = Path("backend/debug_audio/benchmark_samples_p8/EN-S1.wav")   # 2.28s
AUDIO_MID   = Path("backend/debug_audio/benchmark_samples_p8/HI-1.wav")    # 3.00s
AUDIO_LONG  = Path("backend/debug_audio/benchmark_samples_p8/EN-L1.wav")   # 5.88s

LINES = []

def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))

def get_dur(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()

def cond_A(audio_bytes, lang, dur):
    log(f"\n[A] Final-only  (audio={dur:.2f}s lang={lang})")
    t0 = time.perf_counter()
    txt, _, _, _, _ = stt_service.transcribe(
        audio_bytes=audio_bytes, lang=lang,
        filename="exp1.wav", content_type="audio/wav",
        session_id="exp1", request_id="cA", reason="exp1_A")
    wall = time.perf_counter() - t0
    log(f"    wall={wall:.3f}s  rtf={wall/max(dur,0.001):.2f}x  transcript='{txt}'")
    return {"wall_s": wall, "transcript": txt}

def cond_B(audio_bytes, lang, dur, partial_bytes):
    log(f"\n[B] Sequential partial->final  (partial window={get_dur(partial_bytes):.2f}s)")
    t_seq = time.perf_counter()
    t0 = time.perf_counter()
    p_txt = stt_service.transcribe_partial(partial_bytes, lang, "exp1", "cB_p", "exp1_B_p")
    t_part = time.perf_counter() - t0
    t0 = time.perf_counter()
    f_txt, _, _, _, _ = stt_service.transcribe(
        audio_bytes=audio_bytes, lang=lang,
        filename="exp1.wav", content_type="audio/wav",
        session_id="exp1", request_id="cB_f", reason="exp1_B_f")
    t_fin = time.perf_counter() - t0
    total = time.perf_counter() - t_seq
    log(f"    partial={t_part:.3f}s -> '{p_txt}'")
    log(f"    final  ={t_fin:.3f}s -> '{f_txt}'")
    log(f"    total  ={total:.3f}s")
    return {"partial_s": t_part, "final_s": t_fin, "total_s": total,
            "partial_txt": p_txt, "final_txt": f_txt}

def cond_C(audio_bytes, lang, dur, partial_bytes):
    log(f"\n[C] Concurrent partial+final  (simulates P8 actual behavior)")
    p_res = {}
    f_res = {}
    def do_partial():
        t0 = time.perf_counter()
        txt = stt_service.transcribe_partial(partial_bytes, lang, "exp1", "cC_p", "exp1_C_p")
        p_res["wall_s"] = time.perf_counter() - t0
        p_res["txt"] = txt
    def do_final():
        t0 = time.perf_counter()
        txt, _, _, _, _ = stt_service.transcribe(
            audio_bytes=audio_bytes, lang=lang,
            filename="exp1.wav", content_type="audio/wav",
            session_id="exp1", request_id="cC_f", reason="exp1_C_f")
        f_res["wall_s"] = time.perf_counter() - t0
        f_res["txt"] = txt
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(do_partial), ex.submit(do_final)]
        concurrent.futures.wait(futures)
    total = time.perf_counter() - t0
    log(f"    partial={p_res.get('wall_s',0):.3f}s -> '{p_res.get('txt','')}'")
    log(f"    final  ={f_res.get('wall_s',0):.3f}s -> '{f_res.get('txt','')}'")
    log(f"    total  ={total:.3f}s")
    return {"partial_s": p_res.get("wall_s",0), "final_s": f_res.get("wall_s",0),
            "total_s": total, "partial_txt": p_res.get("txt",""), "final_txt": f_res.get("txt","")}

def run_set(path, lang, label):
    log(f"\n{SEP}\nSET: {label}  ({path.name})\n{SEP}")
    ab = path.read_bytes()
    dur = get_dur(ab)
    # Partial window: first ~1s of PCM = 32000 bytes + 44 header
    pwin = min(44 + 32000, len(ab) // 3)
    partial_ab = ab[:pwin]
    rA = cond_A(ab, lang, dur)
    rB = cond_B(ab, lang, dur, partial_ab)
    rC = cond_C(ab, lang, dur, partial_ab)
    log(f"\n--- Comparison ---")
    log(f"  A final-only   : {rA['wall_s']:.3f}s")
    log(f"  B seq p->f     : partial={rB['partial_s']:.3f}s + final={rB['final_s']:.3f}s = {rB['total_s']:.3f}s")
    log(f"  C concurrent   : final_wall={rC['final_s']:.3f}s, total_wall={rC['total_s']:.3f}s")
    slowdown = rC["final_s"] / max(rA["wall_s"], 0.001)
    log(f"  C.final / A.final slowdown = {slowdown:.2f}x")
    if slowdown > 1.3:
        log(f"  *** CONCURRENCY DEGRADES FINAL WHISPER SPEED by {slowdown:.2f}x ***")
    else:
        log(f"  NOTE: No significant slowdown (<1.3x) — check if local CPU has slack")
    # Quality check
    if rA["transcript"].strip().lower() != rC["final_txt"].strip().lower():
        log(f"  QUALITY DIFFERS: A='{rA['transcript']}' vs C='{rC['final_txt']}'")
    else:
        log(f"  QUALITY: transcripts match (concurrent did not change output)")
    return {"label": label, "dur": dur, "A": rA, "B": rB, "C": rC, "slowdown": slowdown}

def main():
    log(SEP)
    log("P9 EXP 1 -- Concurrency Latency Proof")
    log(SEP)
    log(f"Model={stt_service.model_size} device={stt_service.device} "
        f"compute={stt_service.compute_type} threads={stt_service.cpu_threads}")

    # Warmup
    log("\n[*] Warming up model...")
    if AUDIO_SHORT.exists():
        stt_service.transcribe_partial(AUDIO_SHORT.read_bytes(), "en", "wu", "wu", "wu")
    log("[+] Warmup done\n")

    results = []
    for path, lang, label in [
        (AUDIO_SHORT, "en", "Short English 2.28s"),
        (AUDIO_MID,   "hi", "Hindi 3.00s"),
        (AUDIO_LONG,  "en", "Long English 5.88s"),
    ]:
        if path.exists():
            results.append(run_set(path, lang, label))
        else:
            log(f"[!] Missing {path}")

    log(f"\n{SEP}\nFINAL TABLE\n{SEP}")
    log(f"{'Label':<25} {'Audio':>7} {'A_wall':>8} {'B_final':>8} {'C_final':>8} {'Slowdown':>10} {'QMatch':>7}")
    for r in results:
        a = r["A"]["wall_s"]
        bf = r["B"]["final_s"]
        cf = r["C"]["final_s"]
        sl = r["slowdown"]
        qm = "YES" if r["A"]["transcript"].strip().lower() == r["C"]["final_txt"].strip().lower() else "NO"
        log(f"{r['label']:<25} {r['dur']:>6.2f}s {a:>7.3f}s {bf:>7.3f}s {cf:>7.3f}s {sl:>9.2f}x {qm:>7}")

    log("\nINTERPRETATION:")
    log("  Slowdown >1.3x -> Concurrency is degrading final Whisper speed (confirms RC-1)")
    log("  QMatch=NO      -> Concurrency also changes transcript (proves quality degradation)")
    log("  QMatch=YES     -> Concurrency affects ONLY latency, not quality")

    OUT_FILE.write_text("\n".join(LINES), encoding="utf-8")
    log(f"\n[+] Results saved: {OUT_FILE}")

if __name__ == "__main__":
    main()
