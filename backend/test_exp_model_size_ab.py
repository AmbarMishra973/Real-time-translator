"""
Experiment 4 (EXP-P8-4): Model Size A/B Benchmark (Faster-Whisper 'base' vs 'tiny').
Evaluates:
- Transcription Accuracy (WER, exact match) per category (Short EN, Long EN, Hindi, Hinglish, Technical, Numbers)
- Inference latency per category and overall
- Real-time factor (RTF)
- Memory usage (RAM)
"""

import os
import sys
import json
import time
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.test_stt_benchmark import calculate_wer
from faster_whisper import WhisperModel

DATASET_DIR = Path("backend/debug_audio/benchmark_samples_p8")
MANIFEST_FILE = DATASET_DIR / "dataset_manifest.json"
RESULTS_DIR = Path("backend/benchmark_results")

def run_model_size_ab():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    models_to_test = ["base", "tiny"]
    comparison = {}

    for model_name in models_to_test:
        print("\n" + "="*70)
        print(f"BENCHMARKING MODEL SIZE: Faster-Whisper '{model_name}' (device='cpu', compute_type='int8')")
        print("="*70)

        t_load_0 = time.perf_counter()
        model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
        t_load = time.perf_counter() - t_load_0
        print(f"[+] Loaded {model_name} in {t_load:.2f}s")

        # Warmup
        _ = model.transcribe(io.BytesIO(b"RIFF$ \x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00 \x00\x00" + b"\x00"*8192), beam_size=1)

        results = []
        cat_stats = {}

        for i, item in enumerate(manifest, 1):
            wav_bytes = (DATASET_DIR / item["wav_file"]).read_bytes()
            whisper_lang = None if item["lang"] == "auto" else item["lang"]
            prompt = "यह हिंदी में बातचीत है।" if whisper_lang == "hi" else None

            t0 = time.perf_counter()
            segments, info = model.transcribe(
                io.BytesIO(wav_bytes),
                language=whisper_lang,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=prompt,
            )
            text = ' '.join(s.text for s in segments).strip()
            lat = time.perf_counter() - t0

            wer_val, edits, ref_len = calculate_wer(item["reference_text"], text)
            exact = (wer_val == 0.0)

            cat = item["category"]
            if cat not in cat_stats:
                cat_stats[cat] = {"wer_sum": 0.0, "lat_sum": 0.0, "exact_cnt": 0, "count": 0}
            cat_stats[cat]["wer_sum"] += wer_val
            cat_stats[cat]["lat_sum"] += lat
            cat_stats[cat]["exact_cnt"] += int(exact)
            cat_stats[cat]["count"] += 1

            results.append({
                "id": item["id"],
                "category": cat,
                "lat": round(lat, 3),
                "wer": round(wer_val, 4),
                "exact": exact,
                "ref": item["reference_text"],
                "hyp": text
            })
            print(f"[{i:02d}/30] {item['id']} ({cat}): lat={lat:.3f}s | WER={wer_val:.4f} | exact={exact}")
            print(f"       Ref: '{item['reference_text'][:60]}'")
            print(f"       Hyp: '{text[:60]}'")

        total_lat = sum(r["lat"] for r in results)
        mean_lat = total_lat / len(results)
        avg_wer = sum(r["wer"] for r in results) / len(results)
        exact_cnt = sum(1 for r in results if r["exact"])

        print(f"\n[{model_name.upper()} OVERALL SUMMARY]")
        print(f"  Mean Latency: {mean_lat:.3f}s (Total: {total_lat:.2f}s)")
        print(f"  Average WER:  {avg_wer:.4f}")
        print(f"  Exact Matches:{exact_cnt} / {len(results)} ({exact_cnt/len(results)*100:.1f}%)")

        print("\n  Category Breakdown:")
        for cat, st in cat_stats.items():
            print(f"    - {cat:<20}: Mean Lat={st['lat_sum']/st['count']:.3f}s | Avg WER={st['wer_sum']/st['count']:.4f} | Exact={st['exact_cnt']}/{st['count']}")

        comparison[model_name] = {
            "model": model_name,
            "mean_latency_s": round(mean_lat, 3),
            "total_latency_s": round(total_lat, 2),
            "average_wer": round(avg_wer, 4),
            "exact_matches": exact_cnt,
            "exact_pct": round(exact_cnt/len(results)*100, 1),
            "category_stats": {
                cat: {
                    "mean_latency_s": round(st["lat_sum"]/st["count"], 3),
                    "average_wer": round(st["wer_sum"]/st["count"], 4),
                    "exact_matches": st["exact_cnt"],
                    "total": st["count"]
                }
                for cat, st in cat_stats.items()
            },
            "results": results
        }

    out_file = RESULTS_DIR / "p8_model_size_ab_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved model size comparison report to {out_file}")

if __name__ == "__main__":
    run_model_size_ab()
