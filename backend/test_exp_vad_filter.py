"""
Experiment 1 (EXP-P8-1): Faster-Whisper vad_filter=True vs vad_filter=False on 30 clips.
Evaluates single-pass STT accuracy (WER, exact match) and latency across all 30 clips.
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.test_stt_benchmark import calculate_wer
from faster_whisper import WhisperModel

DATASET_DIR = Path("backend/debug_audio/benchmark_samples_p8")
MANIFEST_FILE = DATASET_DIR / "dataset_manifest.json"

def run_vad_filter_ab():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)

    for vf_setting in [False, True]:
        print("\n" + "="*70)
        print(f"Testing Faster-Whisper vad_filter={vf_setting}")
        print("="*70)

        results = []
        t_start = time.perf_counter()

        for i, item in enumerate(manifest, 1):
            wav_bytes = (DATASET_DIR / item["wav_file"]).read_bytes()
            whisper_lang = None if item["lang"] == "auto" else item["lang"]
            prompt = "यह हिंदी में बातचीत है।" if whisper_lang == "hi" else None

            t0 = time.perf_counter()
            import io
            segments, info = model.transcribe(
                io.BytesIO(wav_bytes),
                language=whisper_lang,
                beam_size=1,
                temperature=0.0,
                vad_filter=vf_setting,
                condition_on_previous_text=False,
                initial_prompt=prompt,
            )
            text = ' '.join(s.text for s in segments).strip()
            lat = time.perf_counter() - t0

            wer_val, edits, ref_len = calculate_wer(item["reference_text"], text)
            exact = (wer_val == 0.0)
            results.append({
                "id": item["id"],
                "category": item["category"],
                "lat": lat,
                "wer": wer_val,
                "exact": exact,
                "text": text
            })

        total_lat = sum(r["lat"] for r in results)
        mean_lat = total_lat / len(results)
        avg_wer = sum(r["wer"] for r in results) / len(results)
        exacts = sum(1 for r in results if r["exact"])

        print(f"\n[vad_filter={vf_setting} Summary]")
        print(f"  Mean Latency: {mean_lat:.3f}s (Total: {total_lat:.2f}s)")
        print(f"  Average WER:  {avg_wer:.4f}")
        print(f"  Exact Matches:{exacts} / {len(results)} ({exacts/len(results)*100:.1f}%)")

if __name__ == "__main__":
    run_vad_filter_ab()
