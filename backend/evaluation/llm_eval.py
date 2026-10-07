"""
LLM and Translation Quality Evaluator.
Performs deterministic verification of translation outputs:
- Output presence & non-emptiness
- Numerical preservation (ASCII & Devanagari digits)
- Named entity preservation
- Technical terminology preservation
- Target language script sanity (Devanagari / Latin)
- Reference-based token overlap (clearly labeled, non-hallucinated metric)
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.llm_service import llm_service, LLMService


# Devanagari digits mapping
DEV_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

def normalize_digits(text: str) -> str:
    """Normalize both Devanagari and Latin digits to standard ASCII for robust comparison."""
    res = []
    for ch in text:
        res.append(DEV_DIGIT_MAP.get(ch, ch))
    return "".join(res)


class TranslationEvaluator:
    """Evaluates translation outputs across deterministic quality criteria."""

    def __init__(self, service: Optional[LLMService] = None):
        self.service = service or llm_service

    def load_dataset(self, dataset_path: Optional[str] = None) -> List[Dict[str, Any]]:
        if dataset_path is None:
            dataset_path = os.path.join(
                os.path.dirname(__file__), "datasets", "translation_cases.json"
            )
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_number_preservation(self, source_text: str, translated_text: str, expected_numbers: List[str]) -> bool:
        """Verifies all source numbers are preserved in target text."""
        if not expected_numbers:
            # Extract numbers from source if not explicitly listed
            src_nums = re.findall(r"\d+(?:[./]\d+)*", source_text)
            if not src_nums:
                return True
            expected_numbers = src_nums

        norm_trans = normalize_digits(translated_text)
        for num in expected_numbers:
            # Normalize candidate number
            norm_num = normalize_digits(num)
            if "/" in norm_num:
                # e.g. 130/80 can appear as "130/80" or "130 और 80" or "130 / 80"
                parts = norm_num.split("/")
                if not all(p in norm_trans for p in parts):
                    return False
            else:
                if norm_num not in norm_trans:
                    return False
        return True

    def check_term_preservation(self, translated_text: str, expected_terms: List[Any]) -> bool:
        """Verifies critical technical terms are preserved or appropriately transliterated."""
        if not expected_terms:
            return True
        low_trans = translated_text.lower()
        hits = 0
        for term_item in expected_terms:
            variants = term_item if isinstance(term_item, list) else [term_item]
            item_hit = False
            for var in variants:
                var_clean = str(var).lower().strip()
                if var_clean in low_trans:
                    item_hit = True
                    break
                tokens = re.findall(r"\w+", var_clean)
                if any(t in low_trans for t in tokens if len(t) >= 2):
                    item_hit = True
                    break
            if item_hit:
                hits += 1
        return (hits / len(expected_terms)) >= 0.70

    def check_entity_preservation(self, translated_text: str, expected_entities: List[Any]) -> bool:
        """Verifies named entities are present or transliterated."""
        if not expected_entities:
            return True
        low_trans = translated_text.lower()
        hits = 0
        for ent_item in expected_entities:
            variants = ent_item if isinstance(ent_item, list) else [ent_item]
            item_hit = False
            for var in variants:
                var_clean = str(var).lower().strip()
                if var_clean in low_trans:
                    item_hit = True
                    break
                tokens = var_clean.split()
                if any(t in low_trans for t in tokens if len(t) >= 3):
                    item_hit = True
                    break
            if item_hit:
                hits += 1
        return hits == len(expected_entities)

    def check_script_sanity(self, translated_text: str, target_script: str) -> bool:
        """Verifies that output contains the expected target script."""
        if not translated_text:
            return False
        if target_script == "devanagari":
            # Must contain Devanagari Unicode range \u0900-\u097F
            return bool(re.search(r"[\u0900-\u097F]", translated_text))
        elif target_script == "latin":
            # Must contain Latin characters [a-zA-Z]
            return bool(re.search(r"[a-zA-Z]", translated_text))
        return True

    def compute_token_overlap(self, text_a: str, text_b: str) -> float:
        """Lightweight token overlap metric (clearly designated as reference-based similarity)."""
        tokens_a = set(re.findall(r"\w+", text_a.lower()))
        tokens_b = set(re.findall(r"\w+", text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))

    def evaluate_case(self, case: Dict[str, Any], candidate_translation: str) -> Dict[str, Any]:
        """Evaluates a single translation output against quality checks."""
        source_text = case["source_text"]
        target_script = case.get("target_script", "devanagari")
        expected_numbers = case.get("expected_numbers", [])
        expected_terms = case.get("expected_technical_terms", [])
        expected_entities = case.get("expected_entities", [])

        is_non_empty = bool(candidate_translation and candidate_translation.strip())
        num_ok = self.check_number_preservation(source_text, candidate_translation, expected_numbers)
        terms_ok = self.check_term_preservation(candidate_translation, expected_terms)
        entities_ok = self.check_entity_preservation(candidate_translation, expected_entities)
        script_ok = self.check_script_sanity(candidate_translation, target_script)

        passed_all = is_non_empty and num_ok and terms_ok and entities_ok and script_ok

        return {
            "id": case["id"],
            "source_text": source_text,
            "translated_text": candidate_translation,
            "passed": passed_all,
            "checks": {
                "non_empty": is_non_empty,
                "numbers_preserved": num_ok,
                "terms_preserved": terms_ok,
                "entities_preserved": entities_ok,
                "script_sanity": script_ok
            }
        }

    def evaluate(
        self,
        dataset: Optional[List[Dict[str, Any]]] = None,
        translation_override_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Runs complete translation evaluation over dataset.
        Uses translation_override_map if provided (for offline deterministic testing).
        """
        if dataset is None:
            dataset = self.load_dataset()

        total = len(dataset)
        passed_count = 0
        non_empty_count = 0
        numbers_preserved_count = 0
        terms_preserved_count = 0
        entities_preserved_count = 0
        script_sanity_count = 0

        case_results = []
        latencies_ms = []

        for case in dataset:
            cid = case["id"]
            if translation_override_map and cid in translation_override_map:
                translated_text = translation_override_map[cid]
            else:
                t0 = time.perf_counter()
                try:
                    res = self.service.translate(
                        text=case["source_text"],
                        source_lang=case.get("source_lang", "en"),
                        target_lang=case.get("target_lang", "hi"),
                        session_id=f"eval_{cid}"
                    )
                    translated_text = res.get("translated_text", "")
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                except Exception as e:
                    translated_text = ""

            eval_res = self.evaluate_case(case, translated_text)
            checks = eval_res["checks"]

            if eval_res["passed"]:
                passed_count += 1
            if checks["non_empty"]:
                non_empty_count += 1
            if checks["numbers_preserved"]:
                numbers_preserved_count += 1
            if checks["terms_preserved"]:
                terms_preserved_count += 1
            if checks["entities_preserved"]:
                entities_preserved_count += 1
            if checks["script_sanity"]:
                script_sanity_count += 1

            case_results.append(eval_res)

        avg_lat = round(sum(latencies_ms) / len(latencies_ms), 1) if latencies_ms else 0.0

        summary = {
            "total_cases": total,
            "passed_cases": passed_count,
            "overall_pass_rate": round(passed_count / total, 4) if total else 0.0,
            "criteria_breakdown": {
                "non_empty": f"{non_empty_count}/{total} ({round(non_empty_count/total*100, 1)}%)",
                "numbers_preserved": f"{numbers_preserved_count}/{total} ({round(numbers_preserved_count/total*100, 1)}%)",
                "terms_preserved": f"{terms_preserved_count}/{total} ({round(terms_preserved_count/total*100, 1)}%)",
                "entities_preserved": f"{entities_preserved_count}/{total} ({round(entities_preserved_count/total*100, 1)}%)",
                "script_sanity": f"{script_sanity_count}/{total} ({round(script_sanity_count/total*100, 1)}%)",
            },
            "average_latency_ms": avg_lat,
            "case_results": case_results
        }
        return summary

    def save_report(self, summary: Dict[str, Any], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def run_cli_translation_eval(save_path: Optional[str] = None):
    evaluator = TranslationEvaluator()
    summary = evaluator.evaluate()

    print("=" * 60)
    print("        LLM / TRANSLATION QUALITY EVALUATION REPORT        ")
    print("=" * 60)
    print(f"Total Cases:             {summary['total_cases']}")
    print(f"Passed All Checks:       {summary['passed_cases']}/{summary['total_cases']} ({summary['overall_pass_rate'] * 100:.1f}%)")
    print("-" * 60)
    print("Deterministic Criteria Verification:")
    for crit, val in summary["criteria_breakdown"].items():
        print(f"  - {crit:20s}: {val}")
    print("-" * 60)
    if summary["average_latency_ms"] > 0:
        print(f"Average Latency:         {summary['average_latency_ms']} ms")
    print("=" * 60)

    if save_path:
        evaluator.save_report(summary, save_path)
        print(f"Saved translation evaluation snapshot to: {save_path}")

    return summary


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "reports", "translation_evaluation.json"
    )
    run_cli_translation_eval(out_file)
