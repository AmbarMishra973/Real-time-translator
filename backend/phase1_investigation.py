"""
Phase 1 Forensic Investigation Script
Isolates:
1. Raw audio vs converted WAV vs DSP boosted vs silence gate.
2. Direct Faster-Whisper decoding matrix (beam_size, vad_filter, condition_on_previous_text).
3. Exact response of sync_transcribe from backend.server.
"""

import sys
import os
import io
import time
import math
import wave
import array

# Ensure line-buffering on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, r"c:\Users\dell1\Desktop\Real-time-translator")

from faster_whisper import WhisperModel
from backend.audio_diagnostics import (
    inspect_pcm16_wav,
    boost_quiet_pcm16_wav,
    pad_pcm16_wav,
    validate_normalized_audio,
)
from backend.server import sync_transcribe, whisper_model, WHISPER_SIZE

def scale_wav(wav_bytes: bytes, target_rms: float) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        ch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = array.array('h', frames)
    ms = sum(s*s for s in samples) / len(samples) if samples else 0
    cur_rms = 20 * math.log10(max(math.sqrt(ms) / 32768.0, 1e-12))
    factor = 10.0 ** ((target_rms - cur_rms) / 20.0)
    scaled = array.array('h', [max(-32768, min(32767, int(round(s * factor)))) for s in samples])
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(scaled.tobytes())
    return buf.getvalue()

def truncate_wav(wav_bytes: bytes, duration_s: float) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        ch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        n = int(sr * duration_s)
        frames = wf.readframes(n)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(frames)
    return buf.getvalue()

def main():
    print("=" * 65, flush=True)
    print("PHASE 1 FORENSIC STT AUDIT & DIRECT FASTER-WHISPER DECODING", flush=True)
    print("=" * 65, flush=True)

    base_wav_path = r"backend\debug_audio\last_recording.wav"
    if not os.path.exists(base_wav_path):
        print(f"Error: {base_wav_path} does not exist!", flush=True)
        return

    with open(base_wav_path, "rb") as f:
        clean_wav = f.read()

    diag = inspect_pcm16_wav(clean_wav)
    print(f"\n[Baseline Audio Artifact]", flush=True)
    print(f"  File: {base_wav_path}", flush=True)
    print(f"  Duration: {diag['duration_s']}s, Rate: {diag['sample_rate_hz']}Hz, Channels: {diag['channels']}", flush=True)
    print(f"  RMS: {diag['rms_dbfs']} dBFS, Peak: {diag['peak_dbfs']} dBFS", flush=True)

    # Prepare Test Variants
    variants = {}
    
    # Variant 1: Baseline clean audio (-22.8 dBFS)
    variants["1_clean_normal"] = {
        "desc": "Clean speech (-22.8 dBFS, 1.87s)",
        "wav": clean_wav
    }
    
    # Variant 2: Quiet speech unboosted (-42.0 dBFS)
    variants["2_quiet_unboosted"] = {
        "desc": "Quiet speech (-42.0 dBFS unboosted, 1.87s)",
        "wav": scale_wav(clean_wav, -42.0)
    }
    
    # Variant 3: Quiet speech WITH DSP Boost (-42 dBFS boosted)
    boosted_wav, gain_applied = boost_quiet_pcm16_wav(variants["2_quiet_unboosted"]["wav"])
    variants["3_quiet_dsp_boosted"] = {
        "desc": f"Quiet speech DSP boosted (+{gain_applied}dB applied)",
        "wav": boosted_wav
    }

    # Variant 4: Very quiet speech (-52.0 dBFS unboosted)
    variants["4_very_quiet_unboosted"] = {
        "desc": "Very quiet speech (-52.0 dBFS unboosted)",
        "wav": scale_wav(clean_wav, -52.0)
    }
    
    # Variant 5: Sub-second short clip (0.6s) unpadded
    short_unpadded = truncate_wav(clean_wav, 0.6)
    variants["5_short_unpadded"] = {
        "desc": "Short clip (0.6s, unpadded)",
        "wav": short_unpadded
    }
    
    # Variant 6: Sub-second short clip (0.6s) padded with 250ms silence
    padded_wav, pad_dur = pad_pcm16_wav(short_unpadded, 250)
    variants["6_short_padded"] = {
        "desc": f"Short clip padded (0.6s -> {pad_dur}s with 250ms silence)",
        "wav": padded_wav
    }

    # Variant 7: Pure room silence (-65 dBFS)
    variants["7_silence"] = {
        "desc": "Near true silence (-65.0 dBFS)",
        "wav": scale_wav(clean_wav, -65.0)
    }

    # Initialize model
    print(f"\n[STT Model] WhisperModel({WHISPER_SIZE}, device='cpu', compute_type='int8')", flush=True)

    # Matrix of decoding configurations to test on each variant
    matrix = [
        {"beam": 1, "vad": False, "prev_cond": False, "name": "beam=1, vad=False, prev_cond=False"},
        {"beam": 1, "vad": True,  "prev_cond": False, "name": "beam=1, vad=True (Silero), prev_cond=False"},
        {"beam": 5, "vad": False, "prev_cond": False, "name": "beam=5, vad=False, prev_cond=False"},
        {"beam": 1, "vad": False, "prev_cond": True,  "name": "beam=1, vad=False, prev_cond=True (DEFAULT)"},
    ]

    print("\n" + "=" * 65, flush=True)
    print("PART 1: DIRECT FASTER-WHISPER DECODING MATRIX", flush=True)
    print("=" * 65, flush=True)

    for v_key, v_info in variants.items():
        v_diag = inspect_pcm16_wav(v_info["wav"])
        print(f"\n>>> VARIANT: {v_info['desc']}", flush=True)
        print(f"    Signal: Dur={v_diag['duration_s']}s, RMS={v_diag['rms_dbfs']} dBFS, Peak={v_diag['peak_dbfs']} dBFS", flush=True)

        for cfg in matrix:
            t0 = time.perf_counter()
            try:
                segments, info = whisper_model.transcribe(
                    io.BytesIO(v_info["wav"]),
                    language="en",
                    beam_size=cfg["beam"],
                    temperature=0.0,
                    vad_filter=cfg["vad"],
                    condition_on_previous_text=cfg["prev_cond"],
                )
                txt = " ".join(s.text for s in segments).strip()
                elapsed = time.perf_counter() - t0
                print(f"    [{cfg['name']}] -> '{txt}' (lat: {elapsed:.2f}s)", flush=True)
            except Exception as e:
                print(f"    [{cfg['name']}] -> ERROR: {e}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("PART 2: FULL SERVER sync_transcribe PIPELINE ROUTING TEST", flush=True)
    print("=" * 65, flush=True)

    for v_key, v_info in variants.items():
        print(f"\n>>> Testing server.py sync_transcribe on: {v_info['desc']}", flush=True)
        t0 = time.perf_counter()
        try:
            transcript, info, out_diag, engine_used, model_used = sync_transcribe(
                v_info["wav"],
                lang="en",
                filename="test.wav",
                content_type="audio/wav"
            )
            elapsed = time.perf_counter() - t0
            print(f"    Engine: {engine_used} ({model_used}) | Latency: {elapsed:.2f}s", flush=True)
            print(f"    Transcript: '{transcript}'", flush=True)
            print(f"    Gate Fired: {out_diag.get('is_silent_gate')} (reason: {out_diag.get('gate_reason')})", flush=True)
            print(f"    Hallucination Flag: {out_diag.get('suspected_hallucination')}", flush=True)
        except Exception as e:
            print(f"    Server error: {e}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("PHASE 1 FORENSIC AUDIT COMPLETE", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
