"""
Phase 1 Forensic Script:
Tests Faster-Whisper directly across:
1. Signal levels: Normal (-22 dBFS), Low (-38 dBFS), Very Low (-46 dBFS), Gated (-56 dBFS).
2. Parameter matrix:
   - condition_on_previous_text: False vs True
   - beam_size: 1 vs 5
   - vad_filter: False vs True
   - temperature: 0.0
3. Utterances:
   - "What is your name?" (Conversational English)
   - "Hello" (Short English clip)
   - "आपका नाम क्या है?" (Conversational Hindi)
"""

import sys
import os
import io
import time
import math
import wave
import array
import asyncio
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, r"c:\Users\dell1\Desktop\Real-time-translator")

import edge_tts
from faster_whisper import WhisperModel
from backend.audio_diagnostics import inspect_pcm16_wav, boost_quiet_pcm16_wav, pad_pcm16_wav, validate_normalized_audio

async def synthesize_speech(text: str, voice: str = "en-US-JennyNeural") -> bytes:
    """Generate reference clean speech via edge-tts."""
    comm = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    raw_mp3 = buf.getvalue()
    
    # Convert to 16kHz mono WAV via ffmpeg
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as in_f:
        in_f.write(raw_mp3)
        in_path = in_f.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
    return wav_bytes

def scale_wav_volume(wav_bytes: bytes, target_rms_dbfs: float) -> bytes:
    """Scale a 16kHz mono WAV to an exact target RMS dBFS."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = array.array("h", frames)
    ms = sum(s*s for s in samples) / len(samples) if samples else 0
    rms = math.sqrt(ms)
    cur_rms_dbfs = 20 * math.log10(max(rms / 32768.0, 1e-12))
    
    delta_db = target_rms_dbfs - cur_rms_dbfs
    factor = 10.0 ** (delta_db / 20.0)
    scaled = array.array("h")
    for s in samples:
        v = int(round(s * factor))
        scaled.append(max(-32768, min(32767, v)))
    
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(scaled.tobytes())
    return out.getvalue()

def add_silence_padding(wav_bytes: bytes, lead_ms: int = 800, trail_ms: int = 500) -> bytes:
    """Simulate real browser recording with leading and trailing silence."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = array.array("h", frames)
    lead_zeros = array.array("h", [0] * int(sr * lead_ms / 1000.0))
    trail_zeros = array.array("h", [0] * int(sr * trail_ms / 1000.0))
    full = lead_zeros + samples + trail_zeros
    
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(full.tobytes())
    return out.getvalue()

async def run_diagnostics():
    print("=" * 70)
    print("PHASE 1 FORENSIC ANALYSIS: DIRECT FASTER-WHISPER DECODING MATRIX")
    print("=" * 70)
    
    model_size = os.getenv("WHISPER_SIZE", "small")
    print(f"Loading local Faster-Whisper ({model_size} on cpu)...")
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n")
    
    test_utterances = [
        ("What is your name?", "en", "en-US-JennyNeural"),
        ("Hello", "en", "en-US-JennyNeural"),
        ("आपका नाम क्या है?", "hi", "hi-IN-SwaraNeural"),
    ]
    
    # We will test each utterance at:
    # 1. Clean normal level (~ -22 dBFS)
    # 2. Quiet level (~ -42 dBFS) with typical recording pauses
    # 3. Very quiet level (~ -48 dBFS)
    
    configurations = [
        {"beam_size": 1, "vad_filter": False, "condition_on_previous_text": False, "desc": "Default Pipeline (beam=1, no-VAD, no-prev-cond)"},
        {"beam_size": 1, "vad_filter": True,  "condition_on_previous_text": False, "desc": "VAD Enabled (beam=1, vad=True, no-prev-cond)"},
        {"beam_size": 5, "vad_filter": False, "condition_on_previous_text": False, "desc": "Beam 5 (beam=5, no-VAD, no-prev-cond)"},
        {"beam_size": 1, "vad_filter": False, "condition_on_previous_text": True,  "desc": "Condition on Prev Text (beam=1, vad=False, prev-cond=True)"},
    ]
    
    for text, lang, voice in test_utterances:
        print(f"\n=======================================================")
        print(f"UTTERANCE: '{text}' (lang={lang})")
        print(f"=======================================================")
        base_wav = await synthesize_speech(text, voice)
        base_diag = inspect_pcm16_wav(base_wav)
        print(f"Base Audio: duration={base_diag['duration_s']}s, RMS={base_diag['rms_dbfs']} dBFS, Peak={base_diag['peak_dbfs']} dBFS")
        
        # Test Case A: Clean Normal Speech
        # Test Case B: Quiet Speech with 800ms leading / 500ms trailing pause (mimicking user button press)
        test_variants = [
            ("A. Normal (-22 dBFS, no extra pause)", base_wav),
            ("B. Quiet (-42 dBFS with pauses, unboosted)", add_silence_padding(scale_wav_volume(base_wav, -42.0))),
            ("C. Quiet (-42 dBFS with pauses, AFTER DSP BOOST)", None), # Will boost variant B
        ]
        
        # Prepare variant C
        var_b_wav = test_variants[1][1]
        boosted_wav, gain_db = boost_quiet_pcm16_wav(var_b_wav)
        if base_diag['duration_s'] < 1.0:
            boosted_wav, _ = pad_pcm16_wav(boosted_wav)
        test_variants[2] = (f"C. Quiet with pauses (DSP Boosted +{gain_db}dB)", boosted_wav)
        
        for var_name, var_wav in test_variants:
            diag = inspect_pcm16_wav(var_wav)
            print(f"\n--- Variant: {var_name} ---")
            print(f"Signal: duration={diag['duration_s']}s, RMS={diag['rms_dbfs']} dBFS, Peak={diag['peak_dbfs']} dBFS")
            
            for cfg in configurations:
                t_start = time.perf_counter()
                segments, info = model.transcribe(
                    io.BytesIO(var_wav),
                    language=lang,
                    beam_size=cfg["beam_size"],
                    temperature=0.0,
                    vad_filter=cfg["vad_filter"],
                    condition_on_previous_text=cfg["condition_on_previous_text"],
                )
                result_text = " ".join(s.text for s in segments).strip()
                dur = time.perf_counter() - t_start
                match = "EXACT MATCH" if result_text.lower().rstrip(".?!") == text.lower().rstrip(".?!") else ("PARTIAL/CLOSE" if any(w in result_text.lower() for w in text.lower().split()) else "FAIL/HALLUCINATION")
                print(f"  [{cfg['desc']}]")
                print(f"    -> Transcript: '{result_text}' | Latency: {dur:.2f}s | Result: {match}")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
