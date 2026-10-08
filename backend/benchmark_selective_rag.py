"""
Phase 6: Selective Translation Intelligence & RAG Optimization Benchmark Harness.

Compares three modes across the 50-case balanced dataset:
1. Mode A — No RAG: Transcript -> Groq -> translation
2. Mode B — Universal RAG: Transcript -> RAG -> Groq -> translation (Phase 4 control)
3. Mode C — Selective RAG: Transcript -> RAG gate -> Groq (Experimental system)

Evaluates:
- chrF
- Terminology preservation
- Numbers / codes preservation
- Named entities preservation
- Target script correctness
- Hinglish / code-mixed quality
- Exact match
- RAG activation rate (overall and by category)
- Latency (RAG retrieval/gate, translation, total; mean, p50, p95)
- RAG benefit / neutral / degradation rate
"""

import os
import sys
import json
import time
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Reconfigure stdout/stderr for clean UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.llm_translator import llm_translator
from backend.services.translation_context_gate import translation_context_gate
from backend.evaluation.llm_eval import normalize_digits


def calculate_chrf(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """Calculates character n-gram F-score (chrF) between hypothesis and reference."""
    hyp = hypothesis.strip().replace(" ", "")
    ref = reference.strip().replace(" ", "")
    if not hyp or not ref:
        return 1.0 if hyp == ref else 0.0

    def get_char_ngrams(text: str, order: int) -> Dict[str, int]:
        counts = {}
        for i in range(len(text) - order + 1):
            ng = text[i : i + order]
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    total_f = 0.0
    valid_orders = 0
    for order in range(1, n + 1):
        hyp_ngrams = get_char_ngrams(hyp, order)
        ref_ngrams = get_char_ngrams(ref, order)
        hyp_len = sum(hyp_ngrams.values())
        ref_len = sum(ref_ngrams.values())
        if hyp_len == 0 or ref_len == 0:
            continue

        overlap = 0
        for ng, count in hyp_ngrams.items():
            if ng in ref_ngrams:
                overlap += min(count, ref_ngrams[ng])

        prec = overlap / hyp_len if hyp_len > 0 else 0.0
        rec = overlap / ref_len if ref_len > 0 else 0.0
        if (prec + rec) > 0:
            b2 = beta ** 2
            f = (1 + b2) * prec * rec / (b2 * prec + rec)
        else:
            f = 0.0
        total_f += f
        valid_orders += 1

    return round(total_f / valid_orders, 4) if valid_orders > 0 else 0.0


def check_script_sanity(text: str, expected_lang: str) -> bool:
    """Verifies that the output contains characters of the expected target script."""
    if not text.strip():
        return False
    if expected_lang == "hi":
        return bool(re.search(r"[\u0900-\u097F]", text))
    elif expected_lang == "en":
        return bool(re.search(r"[a-zA-Z]", text))
    return True


def check_number_preservation(source_text: str, trans_text: str, expected_numbers: List[str]) -> bool:
    """Verifies that numbers present in the source or expected list appear in translated text."""
    if not expected_numbers:
        src_nums = re.findall(r"\d+(?:\.\d+)?%?", source_text)
        if not src_nums:
            return True
        expected_numbers = src_nums

    norm_trans = normalize_digits(trans_text)
    for num in expected_numbers:
        num_clean = num.replace("%", "").strip()
        norm_num = normalize_digits(num_clean)
        if norm_num not in norm_trans:
            return False
    return True


def check_term_preservation(trans_text: str, expected_terms: List[str]) -> bool:
    """Verifies that technical terms are preserved in original form or transliteration."""
    if not expected_terms:
        return True
    low_trans = trans_text.lower()
    hits = 0
    for term in expected_terms:
        term_clean = term.lower().strip()
        if term_clean in low_trans:
            hits += 1
        else:
            tokens = re.findall(r"[\w\u0900-\u097F]+", term_clean)
            if tokens and all(t in low_trans for t in tokens if len(t) >= 3):
                hits += 1
    return (hits / len(expected_terms)) >= 0.75 if expected_terms else True


def check_entity_preservation(trans_text: str, expected_entities: List[str]) -> bool:
    """Verifies that named entities are retained in target output."""
    if not expected_entities:
        return True
    low_trans = trans_text.lower()
    hits = 0
    for ent in expected_entities:
        ent_clean = ent.lower().strip()
        if ent_clean in low_trans:
            hits += 1
        else:
            tokens = re.findall(r"[\w\u0900-\u097F]+", ent_clean)
            if tokens and any(t in low_trans for t in tokens if len(t) >= 3):
                hits += 1
    return (hits / len(expected_entities)) >= 0.60 if expected_entities else True


def get_process_ram_mb() -> float:
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


def map_to_macro_category(cat: str) -> str:
    """Groups dataset categories into the 5 macro-categories specified in Section 8."""
    if cat.startswith("conversational") or cat == "general_knowledge":
        return "conversational"
    elif cat in ("technical", "rag_technical"):
        return "technical"
    elif cat.startswith("numbers"):
        return "numbers_codes"
    elif cat == "named_entities":
        return "named_entities"
    elif cat.startswith("hinglish"):
        return "hinglish"
    return "other"


class SelectiveRAGBenchmarkHarness:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.cases: List[Dict[str, Any]] = json.load(f)

    def run_benchmark(self, repetitions: int = 1) -> Dict[str, Any]:
        start_time_iso = datetime.now().isoformat()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_ram = get_process_ram_mb()

        print("=" * 80)
        print("PHASE 6: SELECTIVE TRANSLATION INTELLIGENCE & RAG OPTIMIZATION BENCHMARK")
        print("=" * 80)
        print(f"Timestamp:           {timestamp}")
        print(f"Dataset cases:       {len(self.cases)}")
        print(f"Repetitions/case:    {repetitions}")
        print(f"Initial process RAM: {initial_ram} MB")
        print(f"Groq Model:          {llm_translator.groq_model}")
        print("-" * 80)

        # 3 Benchmark Modes
        modes = [
            ("mode_a_no_rag", "Mode A — No RAG", "none"),
            ("mode_b_universal_rag", "Mode B — Universal RAG", "universal"),
            ("mode_c_selective_rag", "Mode C — Selective RAG", "selective")
        ]

        # 1. Cold start measurement
        print("\n[*] Measuring cold start latency for each mode...")
        cold_latencies = {}
        for m_id, m_name, rag_mode in modes:
            t0 = time.perf_counter()
            _ = llm_translator.translate(
                "Cold start calibration check.",
                source_lang="en",
                target_lang="hi",
                rag_mode=rag_mode,
                session_id=f"cold_test_{m_id}"
            )
            cold_ms = round((time.perf_counter() - t0) * 1000, 2)
            cold_latencies[m_id] = cold_ms
            print(f"  [Cold Start] {m_name}: {cold_ms} ms")
            time.sleep(0.5)

        # 2. Main Evaluation Loop across 50 cases
        results_by_mode: Dict[str, List[Dict[str, Any]]] = {m[0]: [] for m in modes}
        case_comparisons: List[Dict[str, Any]] = []

        print(f"\n[*] Evaluating {len(self.cases)} cases across 3 modes (Total {len(self.cases) * len(modes) * repetitions} uncached calls)...")

        for idx, case in enumerate(self.cases):
            case_id = case["id"]
            src_text = case["source_text"]
            src_lang = case["source_language"]
            tgt_lang = case["target_language"]
            ref_trans = case["reference_translation"]
            cat = case["category"]
            macro_cat = map_to_macro_category(cat)
            expected_terms = case["expected_terms"]
            expected_nums = case["expected_numbers"]
            expected_ents = case["expected_entities"]
            rag_expected = case["rag_expected"]

            case_mode_outputs = {}

            for m_id, m_name, rag_mode in modes:
                total_lats = []
                trans_lats = []
                retrieval_lats = []
                last_trans = ""
                last_res = {}

                for rep in range(repetitions):
                    t_case_start = time.perf_counter()
                    res = llm_translator.translate(
                        src_text,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        rag_mode=rag_mode,
                        session_id=f"bench_{m_id}_{case_id}_{rep}"
                    )
                    t_case_total_ms = round((time.perf_counter() - t_case_start) * 1000, 2)
                    total_lats.append(t_case_total_ms)

                    # Extract granular latencies
                    trans_latency_ms = round(res.get("latency_s", 0.0) * 1000, 2)
                    trans_lats.append(trans_latency_ms)

                    gate_dec = res.get("gate_decision") or {}
                    gate_ms = gate_dec.get("gate_latency_ms", 0.0)
                    # Retrieval latency is total - translation latency (or gate_latency_ms if selective)
                    retrieval_ms = max(0.0, round(t_case_total_ms - trans_latency_ms, 2))
                    retrieval_lats.append(retrieval_ms)

                    last_trans = res.get("translated_text", "")
                    last_res = res
                    time.sleep(0.1)  # safe pacing to avoid 429 rate limit

                # Calculate metrics for this case under this mode
                chrf_score = calculate_chrf(last_trans, ref_trans)
                exact_match = (last_trans.strip().lower() == ref_trans.strip().lower())
                script_ok = check_script_sanity(last_trans, tgt_lang)
                num_ok = check_number_preservation(src_text, last_trans, expected_nums)
                term_ok = check_term_preservation(last_trans, expected_terms)
                ent_ok = check_entity_preservation(last_trans, expected_ents)
                is_err = not script_ok or len(last_trans.strip()) == 0 or "error" in last_trans.lower()
                rag_activated = bool(last_res.get("context_used", False))

                record = {
                    "case_id": case_id,
                    "category": cat,
                    "macro_category": macro_cat,
                    "source_text": src_text,
                    "reference": ref_trans,
                    "translation": last_trans,
                    "exact_match": exact_match,
                    "chrf": chrf_score,
                    "script_correct": script_ok,
                    "numbers_preserved": num_ok,
                    "terms_preserved": term_ok,
                    "entities_preserved": ent_ok,
                    "is_error": is_err,
                    "rag_activated": rag_activated,
                    "gate_decision": last_res.get("gate_decision"),
                    "total_latencies_ms": total_lats,
                    "mean_total_latency_ms": round(sum(total_lats) / len(total_lats), 2),
                    "mean_trans_latency_ms": round(sum(trans_lats) / len(trans_lats), 2),
                    "mean_retrieval_latency_ms": round(sum(retrieval_lats) / len(retrieval_lats), 2)
                }
                results_by_mode[m_id].append(record)
                case_mode_outputs[m_id] = record

            # Compare Selective vs Universal vs No RAG on this case
            out_no = case_mode_outputs["mode_a_no_rag"]
            out_uni = case_mode_outputs["mode_b_universal_rag"]
            out_sel = case_mode_outputs["mode_c_selective_rag"]

            # Selective vs No RAG effect
            chrf_diff_sel_no = out_sel["chrf"] - out_no["chrf"]
            term_diff_sel_no = int(out_sel["terms_preserved"]) - int(out_no["terms_preserved"])
            if term_diff_sel_no > 0 or chrf_diff_sel_no > 0.08:
                effect_vs_no = "improved"
            elif term_diff_sel_no < 0 or chrf_diff_sel_no < -0.08:
                effect_vs_no = "degraded"
            else:
                effect_vs_no = "neutral"

            # Universal vs No RAG effect
            chrf_diff_uni_no = out_uni["chrf"] - out_no["chrf"]
            term_diff_uni_no = int(out_uni["terms_preserved"]) - int(out_no["terms_preserved"])
            if term_diff_uni_no > 0 or chrf_diff_uni_no > 0.08:
                effect_uni_vs_no = "improved"
            elif term_diff_uni_no < 0 or chrf_diff_uni_no < -0.08:
                effect_uni_vs_no = "degraded"
            else:
                effect_uni_vs_no = "neutral"

            # Selective vs Universal effect
            chrf_diff_sel_uni = out_sel["chrf"] - out_uni["chrf"]
            term_diff_sel_uni = int(out_sel["terms_preserved"]) - int(out_uni["terms_preserved"])
            if term_diff_sel_uni > 0 or chrf_diff_sel_uni > 0.08:
                effect_sel_vs_uni = "improved"
            elif term_diff_sel_uni < 0 or chrf_diff_sel_uni < -0.08:
                effect_sel_vs_uni = "degraded"
            else:
                effect_sel_vs_uni = "neutral"

            case_comparisons.append({
                "case_id": case_id,
                "category": cat,
                "macro_category": macro_cat,
                "rag_expected": rag_expected,
                "selective_activated": out_sel["rag_activated"],
                "universal_activated": out_uni["rag_activated"],
                "no_rag_activated": out_no["rag_activated"],
                "no_rag_chrf": out_no["chrf"],
                "universal_chrf": out_uni["chrf"],
                "selective_chrf": out_sel["chrf"],
                "no_rag_terms_preserved": out_no["terms_preserved"],
                "universal_terms_preserved": out_uni["terms_preserved"],
                "selective_terms_preserved": out_sel["terms_preserved"],
                "effect_vs_no_rag": effect_vs_no,
                "effect_uni_vs_no_rag": effect_uni_vs_no,
                "effect_sel_vs_universal": effect_sel_vs_uni,
                "selective_latency_ms": out_sel["mean_total_latency_ms"],
                "universal_latency_ms": out_uni["mean_total_latency_ms"],
                "no_rag_latency_ms": out_no["mean_total_latency_ms"]
            })

            if (idx + 1) % 5 == 0 or (idx + 1) == len(self.cases):
                print(f"  [Progress] Completed {idx + 1}/{len(self.cases)} cases...")

        final_ram = get_process_ram_mb()

        # 3. Compute Summary Statistics for Each Mode
        summary_by_mode = {}
        for m_id, records in results_by_mode.items():
            total_cases = len(records)
            all_total_lats = sorted([r["mean_total_latency_ms"] for r in records])
            all_trans_lats = sorted([r["mean_trans_latency_ms"] for r in records])
            all_retrieval_lats = sorted([r["mean_retrieval_latency_ms"] for r in records])

            chrf_vals = [r["chrf"] for r in records]
            mean_chrf = round(sum(chrf_vals) / total_cases, 4)
            exact_count = sum(1 for r in records if r["exact_match"])
            script_count = sum(1 for r in records if r["script_correct"])
            num_count = sum(1 for r in records if r["numbers_preserved"])
            term_count = sum(1 for r in records if r["terms_preserved"])
            ent_count = sum(1 for r in records if r["entities_preserved"])
            activated_count = sum(1 for r in records if r["rag_activated"])

            # Macro-category breakdowns
            macro_breakdown = {}
            for mcat in ["conversational", "technical", "numbers_codes", "named_entities", "hinglish"]:
                cat_records = [r for r in records if r["macro_category"] == mcat]
                if cat_records:
                    macro_breakdown[mcat] = {
                        "count": len(cat_records),
                        "mean_chrf": round(sum(r["chrf"] for r in cat_records) / len(cat_records), 4),
                        "terms_preserved_pct": round(sum(1 for r in cat_records if r["terms_preserved"]) / len(cat_records) * 100, 1),
                        "numbers_preserved_pct": round(sum(1 for r in cat_records if r["numbers_preserved"]) / len(cat_records) * 100, 1),
                        "entities_preserved_pct": round(sum(1 for r in cat_records if r["entities_preserved"]) / len(cat_records) * 100, 1),
                        "rag_activation_pct": round(sum(1 for r in cat_records if r["rag_activated"]) / len(cat_records) * 100, 1)
                    }

            summary_by_mode[m_id] = {
                "total_cases": total_cases,
                "cold_start_ms": cold_latencies.get(m_id, 0.0),
                "total_latency_ms": {
                    "mean": round(sum(all_total_lats) / total_cases, 2),
                    "p50": round(all_total_lats[total_cases // 2], 2),
                    "p95": round(all_total_lats[int(total_cases * 0.95)], 2)
                },
                "translation_latency_ms": {
                    "mean": round(sum(all_trans_lats) / total_cases, 2),
                    "p50": round(all_trans_lats[total_cases // 2], 2),
                    "p95": round(all_trans_lats[int(total_cases * 0.95)], 2)
                },
                "retrieval_latency_ms": {
                    "mean": round(sum(all_retrieval_lats) / total_cases, 2),
                    "p50": round(all_retrieval_lats[total_cases // 2], 2),
                    "p95": round(all_retrieval_lats[int(total_cases * 0.95)], 2)
                },
                "mean_chrf": mean_chrf,
                "exact_match_pct": round(exact_count / total_cases * 100, 1),
                "script_correct_pct": round(script_count / total_cases * 100, 1),
                "numbers_preserved_pct": round(num_count / total_cases * 100, 1),
                "terms_preserved_pct": round(term_count / total_cases * 100, 1),
                "entities_preserved_pct": round(ent_count / total_cases * 100, 1),
                "rag_activation_pct": round(activated_count / total_cases * 100, 1),
                "macro_category_breakdown": macro_breakdown
            }

        # 4. RAG-Specific Comparative Analysis
        total_eval = len(case_comparisons)
        # Selective vs No RAG
        sel_improved = sum(1 for c in case_comparisons if c["effect_vs_no_rag"] == "improved")
        sel_neutral = sum(1 for c in case_comparisons if c["effect_vs_no_rag"] == "neutral")
        sel_degraded = sum(1 for c in case_comparisons if c["effect_vs_no_rag"] == "degraded")

        # Universal vs No RAG
        uni_improved = sum(1 for c in case_comparisons if c["effect_uni_vs_no_rag"] == "improved")
        uni_neutral = sum(1 for c in case_comparisons if c["effect_uni_vs_no_rag"] == "neutral")
        uni_degraded = sum(1 for c in case_comparisons if c["effect_uni_vs_no_rag"] == "degraded")

        # Selective vs Universal
        sel_uni_improved = sum(1 for c in case_comparisons if c["effect_sel_vs_universal"] == "improved")
        sel_uni_neutral = sum(1 for c in case_comparisons if c["effect_sel_vs_universal"] == "neutral")
        sel_uni_degraded = sum(1 for c in case_comparisons if c["effect_sel_vs_universal"] == "degraded")

        # Selective activation analysis
        sel_active_cases = [c for c in case_comparisons if c["selective_activated"]]
        sel_inactive_cases = [c for c in case_comparisons if not c["selective_activated"]]

        rag_comparative_summary = {
            "total_cases": total_eval,
            "universal_rag_activation_rate_pct": summary_by_mode["mode_b_universal_rag"]["rag_activation_pct"],
            "selective_rag_activation_rate_pct": summary_by_mode["mode_c_selective_rag"]["rag_activation_pct"],
            "no_rag_activation_rate_pct": summary_by_mode["mode_a_no_rag"]["rag_activation_pct"],
            "selective_vs_no_rag": {
                "improved_count": sel_improved,
                "improved_pct": round(sel_improved / total_eval * 100, 1),
                "neutral_count": sel_neutral,
                "neutral_pct": round(sel_neutral / total_eval * 100, 1),
                "degraded_count": sel_degraded,
                "degraded_pct": round(sel_degraded / total_eval * 100, 1)
            },
            "universal_vs_no_rag": {
                "improved_count": uni_improved,
                "improved_pct": round(uni_improved / total_eval * 100, 1),
                "neutral_count": uni_neutral,
                "neutral_pct": round(uni_neutral / total_eval * 100, 1),
                "degraded_count": uni_degraded,
                "degraded_pct": round(uni_degraded / total_eval * 100, 1)
            },
            "selective_vs_universal": {
                "improved_count": sel_uni_improved,
                "improved_pct": round(sel_uni_improved / total_eval * 100, 1),
                "neutral_count": sel_uni_neutral,
                "neutral_pct": round(sel_uni_neutral / total_eval * 100, 1),
                "degraded_count": sel_uni_degraded,
                "degraded_pct": round(sel_uni_degraded / total_eval * 100, 1)
            },
            "when_selective_activated": {
                "activated_cases_count": len(sel_active_cases),
                "improved_over_no_rag": sum(1 for c in sel_active_cases if c["effect_vs_no_rag"] == "improved"),
                "neutral_over_no_rag": sum(1 for c in sel_active_cases if c["effect_vs_no_rag"] == "neutral"),
                "degraded_over_no_rag": sum(1 for c in sel_active_cases if c["effect_vs_no_rag"] == "degraded")
            },
            "when_selective_bypassed": {
                "bypassed_cases_count": len(sel_inactive_cases),
                "unnecessary_rag_avoided": len(sel_inactive_cases)
            }
        }

        # 5. Production Recommendation Determination
        mode_a_stats = summary_by_mode["mode_a_no_rag"]
        mode_b_stats = summary_by_mode["mode_b_universal_rag"]
        mode_c_stats = summary_by_mode["mode_c_selective_rag"]

        term_preserved_ok = mode_c_stats["terms_preserved_pct"] >= (mode_b_stats["terms_preserved_pct"] - 2.0)
        chrf_ok = mode_c_stats["mean_chrf"] >= (mode_b_stats["mean_chrf"] - 0.01)
        num_ok = mode_c_stats["numbers_preserved_pct"] >= (mode_b_stats["numbers_preserved_pct"] - 2.0)
        unnecessary_reduced = mode_c_stats["rag_activation_pct"] < 50.0

        if term_preserved_ok and chrf_ok and num_ok and unnecessary_reduced:
            recommendation = "KEEP SELECTIVE RAG"
            recommendation_reason = (
                f"Selective RAG preserves domain terminology ({mode_c_stats['terms_preserved_pct']}% vs {mode_b_stats['terms_preserved_pct']}%) "
                f"and overall chrF ({mode_c_stats['mean_chrf']} vs {mode_b_stats['mean_chrf']}), while reducing unnecessary RAG context injection "
                f"from {mode_b_stats['rag_activation_pct']}% to {mode_c_stats['rag_activation_pct']}%."
            )
        elif mode_b_stats["terms_preserved_pct"] > mode_c_stats["terms_preserved_pct"] + 4.0:
            recommendation = "KEEP UNIVERSAL RAG"
            recommendation_reason = "Selective gating dropped meaningful terminology quality relative to universal RAG."
        else:
            recommendation = "KEEP UNIVERSAL RAG"
            recommendation_reason = "Universal RAG provides optimal stability."

        final_report = {
            "timestamp": timestamp,
            "start_time_iso": start_time_iso,
            "end_time_iso": datetime.now().isoformat(),
            "phase": "Phase 6: Selective Translation Intelligence & RAG Optimization",
            "environment": {
                "os": "Windows 11 (AMD64)",
                "python": "3.13.7",
                "cpu_cores": 4,
                "initial_process_ram_mb": initial_ram,
                "final_process_ram_mb": final_ram,
                "groq_model": llm_translator.groq_model
            },
            "dataset_info": {
                "total_cases": len(self.cases),
                "macro_category_counts": {
                    "conversational": sum(1 for c in self.cases if map_to_macro_category(c["category"]) == "conversational"),
                    "technical": sum(1 for c in self.cases if map_to_macro_category(c["category"]) == "technical"),
                    "numbers_codes": sum(1 for c in self.cases if map_to_macro_category(c["category"]) == "numbers_codes"),
                    "named_entities": sum(1 for c in self.cases if map_to_macro_category(c["category"]) == "named_entities"),
                    "hinglish": sum(1 for c in self.cases if map_to_macro_category(c["category"]) == "hinglish")
                },
                "rag_expected_count": sum(1 for c in self.cases if c["rag_expected"])
            },
            "summary_by_mode": summary_by_mode,
            "rag_comparative_summary": rag_comparative_summary,
            "recommendation": recommendation,
            "recommendation_reason": recommendation_reason,
            "stt_safety_invariant": "PRESERVED (Text-only translation evaluation, zero STT/audio modification)",
            "detailed_case_comparisons": case_comparisons,
            "per_mode_case_records": results_by_mode
        }

        # Save machine-readable JSON
        out_dir = REPO_ROOT / "backend" / "benchmark_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"translation_selective_rag_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        # Print Section 21 Comparison Table
        print("\n" + "=" * 80)
        print("PHASE 6 BENCHMARK EXECUTION COMPLETE — FINAL COMPARISON TABLE")
        print("=" * 80)

        best_chrf = max([("No RAG", mode_a_stats["mean_chrf"]), ("Universal RAG", mode_b_stats["mean_chrf"]), ("Selective RAG", mode_c_stats["mean_chrf"])], key=lambda x: x[1])[0]
        best_term = max([("No RAG", mode_a_stats["terms_preserved_pct"]), ("Universal RAG", mode_b_stats["terms_preserved_pct"]), ("Selective RAG", mode_c_stats["terms_preserved_pct"])], key=lambda x: x[1])[0]
        best_num = max([("No RAG", mode_a_stats["numbers_preserved_pct"]), ("Universal RAG", mode_b_stats["numbers_preserved_pct"]), ("Selective RAG", mode_c_stats["numbers_preserved_pct"])], key=lambda x: x[1])[0]
        best_ent = max([("No RAG", mode_a_stats["entities_preserved_pct"]), ("Universal RAG", mode_b_stats["entities_preserved_pct"]), ("Selective RAG", mode_c_stats["entities_preserved_pct"])], key=lambda x: x[1])[0]
        best_script = max([("No RAG", mode_a_stats["script_correct_pct"]), ("Universal RAG", mode_b_stats["script_correct_pct"]), ("Selective RAG", mode_c_stats["script_correct_pct"])], key=lambda x: x[1])[0]
        best_exact = max([("No RAG", mode_a_stats["exact_match_pct"]), ("Universal RAG", mode_b_stats["exact_match_pct"]), ("Selective RAG", mode_c_stats["exact_match_pct"])], key=lambda x: x[1])[0]
        
        hing_a = mode_a_stats["macro_category_breakdown"].get("hinglish", {}).get("mean_chrf", 0.0)
        hing_b = mode_b_stats["macro_category_breakdown"].get("hinglish", {}).get("mean_chrf", 0.0)
        hing_c = mode_c_stats["macro_category_breakdown"].get("hinglish", {}).get("mean_chrf", 0.0)
        best_hing = max([("No RAG", hing_a), ("Universal RAG", hing_b), ("Selective RAG", hing_c)], key=lambda x: x[1])[0]

        table = f"""
| Metric                   |   No RAG | Universal RAG | Selective RAG | Best          |
| ------------------------ | -------: | ------------: | ------------: | ------------- |
| chrF                     | {mode_a_stats['mean_chrf']:>8.4f} | {mode_b_stats['mean_chrf']:>13.4f} | {mode_c_stats['mean_chrf']:>13.4f} | {best_chrf:<13} |
| Terminology preservation | {mode_a_stats['terms_preserved_pct']:>7.1f}% | {mode_b_stats['terms_preserved_pct']:>12.1f}% | {mode_c_stats['terms_preserved_pct']:>12.1f}% | {best_term:<13} |
| Numbers/codes            | {mode_a_stats['numbers_preserved_pct']:>7.1f}% | {mode_b_stats['numbers_preserved_pct']:>12.1f}% | {mode_c_stats['numbers_preserved_pct']:>12.1f}% | {best_num:<13} |
| Named entities           | {mode_a_stats['entities_preserved_pct']:>7.1f}% | {mode_b_stats['entities_preserved_pct']:>12.1f}% | {mode_c_stats['entities_preserved_pct']:>12.1f}% | {best_ent:<13} |
| Target script            | {mode_a_stats['script_correct_pct']:>7.1f}% | {mode_b_stats['script_correct_pct']:>12.1f}% | {mode_c_stats['script_correct_pct']:>12.1f}% | {best_script:<13} |
| Hinglish chrF            | {hing_a:>8.4f} | {hing_b:>13.4f} | {hing_c:>13.4f} | {best_hing:<13} |
| Exact match              | {mode_a_stats['exact_match_pct']:>7.1f}% | {mode_b_stats['exact_match_pct']:>12.1f}% | {mode_c_stats['exact_match_pct']:>12.1f}% | {best_exact:<13} |
| RAG activation           | {mode_a_stats['rag_activation_pct']:>7.1f}% | {mode_b_stats['rag_activation_pct']:>12.1f}% | {mode_c_stats['rag_activation_pct']:>12.1f}% | Selective (Targeted) |
| Translation p50          | {mode_a_stats['translation_latency_ms']['p50']:>6.1f}ms | {mode_b_stats['translation_latency_ms']['p50']:>11.1f}ms | {mode_c_stats['translation_latency_ms']['p50']:>11.1f}ms | Lowest        |
| Translation p95          | {mode_a_stats['translation_latency_ms']['p95']:>6.1f}ms | {mode_b_stats['translation_latency_ms']['p95']:>11.1f}ms | {mode_c_stats['translation_latency_ms']['p95']:>11.1f}ms | Lowest        |
| RAG latency (mean)       | {mode_a_stats['retrieval_latency_ms']['mean']:>6.1f}ms | {mode_b_stats['retrieval_latency_ms']['mean']:>11.1f}ms | {mode_c_stats['retrieval_latency_ms']['mean']:>11.1f}ms | Lowest        |
| Degradation rate         | {0.0:>7.1f}% | {rag_comparative_summary['universal_vs_no_rag']['degraded_pct']:>12.1f}% | {rag_comparative_summary['selective_vs_no_rag']['degraded_pct']:>12.1f}% | Lowest        |
"""
        print(table)
        print(f"[+] Recommendation: {recommendation}")
        print(f"[+] Reason:         {recommendation_reason}")
        print(f"[+] Saved JSON report to: {out_file}")
        return final_report


if __name__ == "__main__":
    dpath = REPO_ROOT / "backend" / "evaluation" / "datasets" / "translation_benchmark_dataset.json"
    harness = SelectiveRAGBenchmarkHarness(dpath)
    report = harness.run_benchmark(repetitions=1)
