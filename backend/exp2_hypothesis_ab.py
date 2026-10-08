"""
P9 EXPERIMENT 2 -- Hypothesis Stabilization A/B Test
=====================================================
DIAGNOSTIC ONLY. Does NOT modify any production code.

Question: Does enabling HYPOTHESIS_STABILIZATION (P8 default=true vs P7 default=false)
          affect final transcript quality or observed behavior?

Conditions:
  A) stabilization=OFF  (P7 default behavior)
  B) stabilization=ON   (P8 default behavior)

For each condition, simulate a streaming session with the full partial->final
pipeline using the existing orchestrator. Compare:
  - Final transcript
  - Partial transcripts seen during streaming
  - Whether stable prefix affected the final result
  - Latency difference

Run:  python -m backend.exp2_hypothesis_ab
Output: backend/exp_results/exp2_hypothesis_ab.txt
"""

import io, os, sys, asyncio, time, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("WHISPER_SIZE", "base")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_COMPUTE_TYPE", "int8")
os.environ.setdefault("WHISPER_CPU_THREADS", "2")
os.environ.setdefault("STT_ENGINE", "local")
os.environ.setdefault("AUDIO_PIPELINE_MODE", "in_memory")

from backend.services.stt_service import stt_service
from backend.core.streaming_orchestrator import streaming_orchestrator
from backend.services.hypothesis_service import hypothesis_service

SEP = "=" * 70
OUT_DIR = Path("backend/exp_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "exp2_hypothesis_ab.txt"
LINES = []

def log(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))

def get_dur(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()

# Audio files covering English and Hindi
TEST_FILES = [
    ("backend/debug_audio/benchmark_samples_p8/EN-S1.wav", "en",
     "What is the best way to learn machine learning?"),
    ("backend/debug_audio/benchmark_samples_p8/EN-S2.wav", "en", None),
    ("backend/debug_audio/benchmark_samples_p8/HI-1.wav", "hi", None),
    ("backend/debug_audio/benchmark_samples_p8/TECH-1.wav", "en",
     "We need to implement RAG architecture with a vector database."),
]

async def simulate_streaming(audio_bytes, lang, hypothesis_enabled, label):
    """
    Simulate the WebSocket streaming path:
    Feed audio in 250ms chunks, trigger partials, then finalize.
    Returns final transcript, list of partial transcripts, and wall-clock time.
    """
    session = streaming_orchestrator.create_session(
        session_id=f"exp2_{label}",
        language=lang,
        target_lang="hi",
        vad_enabled=False,
        hypothesis_enabled=hypothesis_enabled,
    )

    # Chunk size: 250ms at 16kHz 16-bit mono = 8000 bytes PCM
    # WAV is already decoded, so we split it directly
    CHUNK = 8000
    partials_seen = []
    t_stream_start = time.perf_counter()

    # Feed audio header + first chunk
    session.add_chunk(audio_bytes[:44 + CHUNK])

    for offset in range(44 + CHUNK, len(audio_bytes), CHUNK):
        chunk = audio_bytes[offset:offset + CHUNK]
        session.add_chunk(chunk)
        if session.should_trigger_partial():
            partial = await streaming_orchestrator.evaluate_partial(session)
            if partial:
                partials_seen.append(partial)

    # Finalize
    t_fin_start = time.perf_counter()
    result = await streaming_orchestrator.finalize_stream(session)
    t_fin = time.perf_counter() - t_fin_start
    total_wall = time.perf_counter() - t_stream_start

    return {
        "final_transcript": result.get("transcript", ""),
        "partials": partials_seen,
        "stt_ms": result.get("metrics", {}).get("stt_ms", 0),
        "fast_path": result.get("metrics", {}).get("stt_ms", 0) < 50,  # <50ms = fast path used
        "total_wall_s": total_wall,
        "fin_wall_s": t_fin,
    }

async def run_file(path_str, lang, reference, file_idx):
    path = Path(path_str)
    if not path.exists():
        log(f"[!] Missing: {path}")
        return
    audio_bytes = path.read_bytes()
    dur = get_dur(audio_bytes)
    log(f"\n{SEP}")
    log(f"FILE {file_idx}: {path.name}  (duration={dur:.2f}s, lang={lang})")
    if reference:
        log(f"  Reference: '{reference}'")
    log(SEP)

    # Condition A: stabilization OFF
    log("\n[A] hypothesis_enabled=False  (P7 default)")
    rA = await simulate_streaming(audio_bytes, lang, False, f"f{file_idx}_A")
    log(f"    Final:    '{rA['final_transcript']}'")
    log(f"    Partials: {rA['partials']}")
    log(f"    stt_ms:   {rA['stt_ms']:.1f}ms  (fast_path={rA['fast_path']})")
    log(f"    Wall:     {rA['total_wall_s']:.3f}s")

    # Condition B: stabilization ON
    log("\n[B] hypothesis_enabled=True   (P8 default)")
    rB = await simulate_streaming(audio_bytes, lang, True, f"f{file_idx}_B")
    log(f"    Final:    '{rB['final_transcript']}'")
    log(f"    Partials: {rB['partials']}")
    log(f"    stt_ms:   {rB['stt_ms']:.1f}ms  (fast_path={rB['fast_path']})")
    log(f"    Wall:     {rB['total_wall_s']:.3f}s")

    # Compare
    log(f"\n--- Comparison ---")
    finals_match = rA["final_transcript"].strip().lower() == rB["final_transcript"].strip().lower()
    log(f"  Final transcripts match: {'YES' if finals_match else 'NO -- hypothesis changes final output!'}")
    log(f"  Partials A: {len(rA['partials'])} events")
    log(f"  Partials B: {len(rB['partials'])} events")
    log(f"  Wall delta (B-A): {rB['total_wall_s'] - rA['total_wall_s']:+.3f}s")
    if not finals_match:
        log(f"  !!! CRITICAL: hypothesis_enabled changed the final transcript !!!")
        log(f"      A='{rA['final_transcript']}'")
        log(f"      B='{rB['final_transcript']}'")
    return {"path": path.name, "lang": lang, "dur": dur,
            "A": rA, "B": rB, "finals_match": finals_match}

async def main():
    log(SEP)
    log("P9 EXP 2 -- Hypothesis Stabilization A/B Test")
    log("Does enabling stabilization change the final transcript or latency?")
    log(SEP)
    log(f"Model={stt_service.model_size} threads={stt_service.cpu_threads}")

    # Warmup
    log("\n[*] Warmup...")
    warmup = Path("backend/debug_audio/benchmark_samples_p8/EN-S1.wav")
    if warmup.exists():
        stt_service.transcribe_partial(warmup.read_bytes(), "en", "wu", "wu", "wu")
    log("[+] Done\n")

    all_results = []
    for i, (path, lang, ref) in enumerate(TEST_FILES, 1):
        r = await run_file(path, lang, ref, i)
        if r:
            all_results.append(r)

    log(f"\n{SEP}\nSUMMARY\n{SEP}")
    log(f"{'File':<20} {'Lang':>5} {'Dur':>6} {'FinalMatch':>12} {'WallDelta':>11}")
    for r in all_results:
        delta = r["B"]["total_wall_s"] - r["A"]["total_wall_s"]
        log(f"{r['path']:<20} {r['lang']:>5} {r['dur']:>5.2f}s "
            f"{'YES' if r['finals_match'] else 'NO ':>12} {delta:>+10.3f}s")

    log("\nKEY FINDINGS:")
    changed = [r for r in all_results if not r["finals_match"]]
    if changed:
        log(f"  REGRESSION: hypothesis changed final transcript for {len(changed)} files")
        for r in changed:
            log(f"    {r['path']}: A='{r['A']['final_transcript']}' B='{r['B']['final_transcript']}'")
    else:
        log("  CONFIRMED: hypothesis stabilization does NOT change final transcripts")
        log("  (hypothesis only affects intermediate partial display, not final result)")

    OUT_FILE.write_text("\n".join(LINES), encoding="utf-8")
    log(f"\n[+] Results saved: {OUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
