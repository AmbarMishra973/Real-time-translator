"""
Phase 8.5: Controlled STT Benchmark Generator (30 Fixed Audio Clips).
Generates deterministic 16kHz mono WAV clips across 6 categories:
1. Short English (5)
2. Long English (5)
3. Hindi (5)
4. Hinglish (5)
5. Technical/Software (5)
6. Numbers/Entities (5)
"""

import os
import sys
import io
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.test_stt_benchmark import generate_speech_wav, inspect_pcm16_wav

OUTPUT_DIR = Path("backend/debug_audio/benchmark_samples_p8")
METADATA_FILE = OUTPUT_DIR / "dataset_manifest.json"

DATASET_SPECS = [
    # Category 1: Short English
    {"id": "EN-S1", "category": "Short English", "text": "Hello, how are you?", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "EN-S2", "category": "Short English", "text": "What time is the meeting?", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "EN-S3", "category": "Short English", "text": "Can you hear me clearly?", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "EN-S4", "category": "Short English", "text": "Thank you for your help.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "EN-S5", "category": "Short English", "text": "Please send me the report.", "lang": "en", "voice": "en-US-JennyNeural"},

    # Category 2: Long English
    {"id": "EN-L1", "category": "Long English", "text": "Hello, what is your name? What are you doing? What is your job?", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "EN-L2", "category": "Long English", "text": "Good morning everyone, today we are going to discuss the architecture of our real time translation system and how to optimize inference latency.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "EN-L3", "category": "Long English", "text": "The speech recognition pipeline processes audio in small windows and produces stabilized transcripts for downstream translation and speech synthesis.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "EN-L4", "category": "Long English", "text": "If you have any questions during the presentation, please feel free to interrupt me or write them down in the chat window.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "EN-L5", "category": "Long English", "text": "We observed that running deep learning models on limited CPU resources requires careful memory management and thread tuning to avoid bottlenecks.", "lang": "en", "voice": "en-US-JennyNeural"},

    # Category 3: Hindi
    {"id": "HI-1", "category": "Hindi", "text": "नमस्ते, आप कैसे हैं?", "lang": "hi", "voice": "hi-IN-SwaraNeural"},
    {"id": "HI-2", "category": "Hindi", "text": "आपका नाम क्या है और आप कहाँ रहते हैं?", "lang": "hi", "voice": "hi-IN-SwaraNeural"},
    {"id": "HI-3", "category": "Hindi", "text": "मुझे आज का मौसम बहुत अच्छा लग रहा है।", "lang": "hi", "voice": "hi-IN-SwaraNeural"},
    {"id": "HI-4", "category": "Hindi", "text": "कृपया मुझे स्टेशन का रास्ता बता दीजिए।", "lang": "hi", "voice": "hi-IN-SwaraNeural"},
    {"id": "HI-5", "category": "Hindi", "text": "क्या हम कल सुबह दस बजे मिल सकते हैं?", "lang": "hi", "voice": "hi-IN-SwaraNeural"},

    # Category 4: Hinglish
    {"id": "HG-1", "category": "Hinglish", "text": "Aaj meeting ka time kya hai?", "lang": "hi", "voice": "hi-IN-SwaraNeural"},
    {"id": "HG-2", "category": "Hinglish", "text": "Mera laptop restart ho gaya hai aur code save nahi hua.", "lang": "hi", "voice": "hi-IN-SwaraNeural"},
    {"id": "HG-3", "category": "Hinglish", "text": "Server down ho gaya hai, please deployment check karo.", "lang": "hi", "voice": "hi-IN-MadhurNeural"},
    {"id": "HG-4", "category": "Hinglish", "text": "Mujhe project documentation submit karni hai.", "lang": "hi", "voice": "hi-IN-SwaraNeural"},
    {"id": "HG-5", "category": "Hinglish", "text": "Latency bahut zyada hai, isko optimize karna padega.", "lang": "hi", "voice": "hi-IN-MadhurNeural"},

    # Category 5: Technical / Software
    {"id": "TECH-1", "category": "Technical", "text": "We need to deploy our containerized microservices to Kubernetes.", "lang": "en", "voice": "en-US-GuyNeural"},
    {"id": "TECH-2", "category": "Technical", "text": "The vector database stores dense embeddings for retrieval augmented generation.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "TECH-3", "category": "Technical", "text": "Check the API gateway logs for rate limit errors and high response times.", "lang": "en", "voice": "en-US-GuyNeural"},
    {"id": "TECH-4", "category": "Technical", "text": "We configure Faster-Whisper with INT8 quantization on CPU.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "TECH-5", "category": "Technical", "text": "Continuous integration pipeline runs unit tests before pushing to production.", "lang": "en", "voice": "en-US-GuyNeural"},

    # Category 6: Numbers / Entities
    {"id": "NUM-1", "category": "Numbers & Entities", "text": "The server IP address is 192.168.1.105 on port 8000.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "NUM-2", "category": "Numbers & Entities", "text": "Flight AI 302 departs at 5:45 PM from terminal 3.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "NUM-3", "category": "Numbers & Entities", "text": "Your verification code is 8 4 9 2 0 1.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "NUM-4", "category": "Numbers & Entities", "text": "The invoice total is $1,250.75 due on November 15th.", "lang": "en", "voice": "en-US-JennyNeural"},
    {"id": "NUM-5", "category": "Numbers & Entities", "text": "Doctor Sharma will be available on room 402 between 2 and 4 PM.", "lang": "en", "voice": "en-US-JennyNeural"},
]

async def generate_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    print(f"[*] Generating {len(DATASET_SPECS)} fixed benchmark audio clips...")

    for i, spec in enumerate(DATASET_SPECS, 1):
        wav_path = OUTPUT_DIR / f"{spec['id']}.wav"
        if not wav_path.exists():
            print(f"  [{i:02d}/{len(DATASET_SPECS)}] Generating {spec['id']} ({spec['category']})...")
            wav_bytes = await generate_speech_wav(spec["text"], spec["voice"])
            wav_path.write_bytes(wav_bytes)
        else:
            wav_bytes = wav_path.read_bytes()

        diag = inspect_pcm16_wav(wav_bytes)
        manifest.append({
            "id": spec["id"],
            "category": spec["category"],
            "reference_text": spec["text"],
            "lang": spec["lang"],
            "voice": spec["voice"],
            "wav_file": str(wav_path.name),
            "duration_s": diag["duration_s"],
            "sample_rate_hz": diag["sample_rate_hz"],
            "rms_dbfs": diag["rms_dbfs"],
            "peak_dbfs": diag["peak_dbfs"],
            "bytes": len(wav_bytes)
        })

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[+] Dataset generated successfully: {len(manifest)} clips in {OUTPUT_DIR}")
    total_dur = sum(m["duration_s"] for m in manifest)
    print(f"[+] Total audio duration: {total_dur:.2f} seconds ({total_dur/60:.2f} minutes)")

if __name__ == "__main__":
    asyncio.run(generate_dataset())
