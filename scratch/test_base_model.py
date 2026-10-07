import time, io, os
from faster_whisper import WhisperModel

wav_bytes = open(r"backend\debug_audio\last_recording.wav", "rb").read()

print("--- Testing Faster-Whisper 'base' model on CPU ---")
t0 = time.perf_counter()
m_base = WhisperModel("base", device="cpu", compute_type="int8")
t_load = time.perf_counter() - t0
print(f"Base model loaded in {t_load:.2f}s")

t0 = time.perf_counter()
segs, info = m_base.transcribe(io.BytesIO(wav_bytes), language="en", beam_size=1, temperature=0.0, vad_filter=False)
txt = " ".join(s.text for s in segs).strip()
t_trans = time.perf_counter() - t0
print(f"Base model transcript: '{txt}' | Latency: {t_trans:.2f}s (Lang: {info.language}, Prob: {info.language_probability})")
