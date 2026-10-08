"""
Generate a realistic 9-second utterance with conversational pauses:
'Hello, what is your name? What are you doing? What is your job?'
And profile:
1. Audio parameters, duration, RMS
2. CPU thread benchmarks (1, 2, 4 threads)
3. vad_filter parameter inside Faster-Whisper (True vs False)
4. Streaming WebSocket simulation with real WebM/WAV chunks
"""

import os
import sys
import time
import io
import asyncio
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.stt_service import stt_service, inspect_pcm16_wav
from backend.test_stt_benchmark import generate_speech_wav
from faster_whisper import WhisperModel

async def create_9s_paused_audio():
    # Generate segments
    w1 = await generate_speech_wav("Hello.", "en-US-JennyNeural")
    w2 = await generate_speech_wav("What is your name?", "en-US-JennyNeural")
    w3 = await generate_speech_wav("What are you doing?", "en-US-JennyNeural")
    w4 = await generate_speech_wav("What is your job?", "en-US-JennyNeural")

    # Read PCM samples from each (skipping 44-byte header)
    pcm1 = w1[44:]
    pcm2 = w2[44:]
    pcm3 = w3[44:]
    pcm4 = w4[44:]

    # 1 second of silence at 16kHz 16-bit mono = 32000 bytes
    silence_800ms = b'\x00' * int(16000 * 2 * 0.8)
    silence_1000ms = b'\x00' * int(16000 * 2 * 1.0)

    combined_pcm = (
        pcm1 + silence_800ms +
        pcm2 + silence_1000ms +
        pcm3 + silence_800ms +
        pcm4 + silence_800ms
    )

    # Write WAV
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(combined_pcm)
    
    wav_bytes = buf.getvalue()
    return wav_bytes

async def main():
    wav_bytes = await create_9s_paused_audio()
    diag = inspect_pcm16_wav(wav_bytes)
    print(f"[+] Created paused audio: duration={diag['duration_s']:.2f}s, bytes={len(wav_bytes)}")

    # 1. Profile Faster-Whisper with different cpu_threads
    print("\n" + "="*70)
    print("PROFILING 1: CPU Threads Comparison (1 vs 2 vs 4 vs 6)")
    print("="*70)
    for threads in [1, 2, 4]:
        model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=threads)
        # Warmup
        _ = model.transcribe(io.BytesIO(wav_bytes[:32000]), beam_size=1)
        
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            io.BytesIO(wav_bytes),
            language="en",
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False
        )
        text = ' '.join(s.text for s in segments).strip()
        dur = time.perf_counter() - t0
        print(f"Threads={threads}: time={dur:.3f}s | RTF={dur/diag['duration_s']:.2f}x | text='{text}'")

    # 2. Profile vad_filter=True vs vad_filter=False
    print("\n" + "="*70)
    print("PROFILING 2: vad_filter inside Faster-Whisper (False vs True)")
    print("="*70)
    for vf in [False, True]:
        model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            io.BytesIO(wav_bytes),
            language="en",
            beam_size=1,
            temperature=0.0,
            vad_filter=vf,
            condition_on_previous_text=False
        )
        text = ' '.join(s.text for s in segments).strip()
        dur = time.perf_counter() - t0
        print(f"vad_filter={vf}: time={dur:.3f}s | RTF={dur/diag['duration_s']:.2f}x | text='{text}'")

    # 3. Save this 9s WAV file as a benchmark sample
    out_path = Path("backend/debug_audio/benchmark_samples/EN-9S-PAUSED.wav")
    out_path.write_bytes(wav_bytes)
    print(f"\n[+] Saved test clip to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
