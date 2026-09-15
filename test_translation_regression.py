"""
Automated Translation Correctness & Quality Regression Suite.
Tests critical linguistic requirements:
- Conversational greeting
- Proper nouns (IIIT Bhopal, Amazon)
- Numbers & Currencies (2500 rupees, port 8080)
- Questions as questions (NOT answering the question)
- Dates & Times (September 20 at 10 AM)
- Technical terminology
"""

import sys
import os
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from backend.llm_translator import llm_translator

REGRESSION_TEST_CASES = [
    {
        "id": "Simple Greeting",
        "input": "Hello, how are you?",
        "src": "en",
        "tgt": "hi",
        "must_contain_any": ["नमस्ते", "आप कैसे हैं", "क्या हाल", "नमस्कार"],
        "must_not_equal_input": True,
    },
    {
        "id": "Question (Must NOT Answer)",
        "input": "What is your name?",
        "src": "en",
        "tgt": "hi",
        "must_contain_any": ["आपका नाम क्या है", "तुम्हारा नाम क्या है", "आपका क्या नाम"],
        "must_not_equal_input": True,
    },
    {
        "id": "Numbers & Currency",
        "input": "The price is 2500 rupees.",
        "src": "en",
        "tgt": "hi",
        "must_contain_any": ["2500", "रुपये", "कीमत"],
        "must_not_equal_input": True,
    },
    {
        "id": "Proper Noun Preservation",
        "input": "I am studying at IIIT Bhopal.",
        "src": "en",
        "tgt": "hi",
        "must_contain_any": ["IIIT Bhopal", "आईआईआईटी भोपाल", "IIIT", "भोपाल"],
        "must_not_equal_input": True,
    },
    {
        "id": "Technical Terminology & Ports",
        "input": "The Spring Boot server is running on port 8080.",
        "src": "en",
        "tgt": "hi",
        "must_contain_any": ["Spring Boot", "8080", "पोर्ट", "port"],
        "must_not_equal_input": True,
    },
    {
        "id": "Mixed Dates, Times & Proper Nouns",
        "input": "Amazon interview is on September 20 at 10 AM.",
        "src": "en",
        "tgt": "hi",
        "must_contain_any": ["Amazon", "अमेज़न", "20", "10"],
        "must_not_equal_input": True,
    },
    {
        "id": "European Language (French)",
        "input": "Good morning everyone, have a nice day.",
        "src": "en",
        "tgt": "fr",
        "must_contain_any": ["bonjour", "bonne journée", "monde"],
        "must_not_equal_input": True,
    }
]

def run_regression():
    print("=" * 70)
    print("       TRANSLATION CORRECTNESS & STABILITY REGRESSION SUITE       ")
    print("=" * 70)

    passed = 0
    failed = 0

    for case in REGRESSION_TEST_CASES:
        cid = case["id"]
        inp = case["input"]
        src = case["src"]
        tgt = case["tgt"]

        t0 = time.perf_counter()
        res = llm_translator.translate(text=inp, source_lang=src, target_lang=tgt)
        lat = time.perf_counter() - t0
        output = res["translated_text"].strip()
        provider = res["provider"]

        # Check 1: Not identical to input
        not_echo = (output.lower() != inp.lower()) if case.get("must_not_equal_input") else True

        # Check 2: Contains expected key concepts
        contains_expected = any(kw.lower() in output.lower() for kw in case.get("must_contain_any", []))

        is_pass = not_echo and contains_expected and len(output) > 0

        print(f"\n[CASE: {cid}]")
        print(f"  Input ({src.upper()}):       \"{inp}\"")
        print(f"  Output ({tgt.upper()}):      \"{output}\"")
        print(f"  Engine:            {provider} (latency: {lat:.2f}s)")
        if is_pass:
            print(f"  Verdict:           [PASS] Correct & Grounded")
            passed += 1
        else:
            print(f"  Verdict:           [FAIL] Check translation quality or keywords")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(REGRESSION_TEST_CASES)} Passed | {failed} Failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_regression()
