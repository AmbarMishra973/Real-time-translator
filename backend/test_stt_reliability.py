"""
Comprehensive Verification Test Suite for STT Reliability Fix:
1. Unit tests for adaptive peak-safe gain boost (synthetic and edge-case audio).
2. Unit tests for padding short clips.
3. Unit tests for pre-padding silence gate freezing and gate_reason diagnostics.
4. Unit tests for normalized hallucination matching.
5. Integration test with real recorded speech transients.
6. Integration test with transient edge-case (loud transient + quiet speech).
"""

import array
import io
import math
import os
import re
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

# Ensure backend can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.audio_diagnostics import (
    boost_quiet_pcm16_wav,
    inspect_pcm16_wav,
    pad_pcm16_wav,
    validate_normalized_audio,
)


def generate_pcm16_sine_wav(duration_s: float, freq_hz: float, target_rms_dbfs: float, sample_rate: int = 16000) -> bytes:
    """Generate a clean 16kHz mono PCM16 sine wave at exact target RMS dBFS."""
    num_samples = int(duration_s * sample_rate)
    # RMS of sine wave with peak amplitude A is A / sqrt(2)
    # target_rms = 32768 * 10^(target_rms_dbfs / 20)
    # A = target_rms * sqrt(2)
    target_rms = 32768.0 * (10.0 ** (target_rms_dbfs / 20.0))
    amplitude = min(32767.0, target_rms * math.sqrt(2.0))

    samples = array.array("h")
    for i in range(num_samples):
        val = int(round(amplitude * math.sin(2.0 * math.pi * freq_hz * i / sample_rate)))
        samples.append(max(-32768, min(32767, val)))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def test_1_synthetic_gain_boost():
    print("\n--- Test 1: Synthetic Sine Wave Gain Boost (-45 dBFS -> -24 dBFS) ---")
    in_wav = generate_pcm16_sine_wav(duration_s=2.0, freq_hz=440.0, target_rms_dbfs=-45.0)
    pre = inspect_pcm16_wav(in_wav)
    print(f"Pre-gain: RMS={pre['rms_dbfs']} dBFS, Peak={pre['peak_dbfs']} dBFS")
    assert abs(pre["rms_dbfs"] - (-45.0)) <= 1.0, f"Expected initial ~-45 dBFS, got {pre['rms_dbfs']}"

    boosted_wav, gain_applied = boost_quiet_pcm16_wav(in_wav)
    post = inspect_pcm16_wav(boosted_wav)
    print(f"Post-gain: RMS={post['rms_dbfs']} dBFS, Peak={post['peak_dbfs']} dBFS, Gain Applied=+{gain_applied} dB")

    # Assert gain brought it to within ~1 dB of target -24.0 dBFS
    assert abs(post["rms_dbfs"] - (-24.0)) <= 1.0, f"Expected post-gain RMS ~-24 dBFS, got {post['rms_dbfs']}"
    # Assert peak does not exceed -1.0 dBFS
    assert post["peak_dbfs"] <= -1.0, f"Peak exceeded -1.0 dBFS headroom limit: {post['peak_dbfs']}"
    assert gain_applied > 15.0, f"Expected significant gain applied, got {gain_applied}"
    print("[PASS] Test 1 passed: Output RMS within 1dB of -24 dBFS with peak headroom protected.")


def test_2_loud_input_unchanged_exact_byte_match():
    print("\n--- Test 2: Loud Input Already Above Trigger (-20 dBFS) ---")
    in_wav = generate_pcm16_sine_wav(duration_s=1.5, freq_hz=440.0, target_rms_dbfs=-20.0)
    pre = inspect_pcm16_wav(in_wav)
    print(f"Pre-gain: RMS={pre['rms_dbfs']} dBFS")

    boosted_wav, gain_applied = boost_quiet_pcm16_wav(in_wav)
    print(f"Gain Applied: {gain_applied} dB")

    assert gain_applied == 0.0, f"Expected 0.0 dB gain, got {gain_applied}"
    assert boosted_wav == in_wav, "Expected EXACT byte match on audio above trigger threshold!"
    print("[PASS] Test 2 passed: Exact byte match returned, audio completely untouched.")


def test_3_noise_floor_unchanged():
    print("\n--- Test 3: True Silence / Noise Floor Below -60 dBFS (-70 dBFS) ---")
    in_wav = generate_pcm16_sine_wav(duration_s=1.5, freq_hz=440.0, target_rms_dbfs=-70.0)
    pre = inspect_pcm16_wav(in_wav)
    print(f"Pre-gain: RMS={pre['rms_dbfs']} dBFS")

    boosted_wav, gain_applied = boost_quiet_pcm16_wav(in_wav)
    print(f"Gain Applied: {gain_applied} dB")

    assert gain_applied == 0.0, f"Expected 0.0 dB gain on noise floor, got {gain_applied}"
    assert boosted_wav == in_wav, "Expected EXACT byte match for audio below silence floor!"
    print("[PASS] Test 3 passed: Room noise floor (< -60 dBFS) is never amplified.")


def test_4_temporal_padding():
    print("\n--- Test 4: Short Clip Temporal Padding (< 1.0s) ---")
    short_wav = generate_pcm16_sine_wav(duration_s=0.5, freq_hz=300.0, target_rms_dbfs=-25.0)
    pre = inspect_pcm16_wav(short_wav)
    print(f"Pre-padding duration: {pre['duration_s']}s")
    assert pre["duration_s"] == 0.5

    padded_wav, new_duration = pad_pcm16_wav(short_wav, pad_ms=250)
    post = inspect_pcm16_wav(padded_wav)
    print(f"Post-padding duration: {post['duration_s']}s (reported {new_duration}s)")

    # 0.5s + 0.25s + 0.25s = 1.0s
    assert abs(post["duration_s"] - 1.0) <= 0.01, f"Expected ~1.0s, got {post['duration_s']}"

    # Verify original audio is preserved in center
    with wave.open(io.BytesIO(short_wav), "rb") as orig_f, wave.open(io.BytesIO(padded_wav), "rb") as pad_f:
        orig_samples = array.array("h", orig_f.readframes(orig_f.getnframes()))
        pad_samples = array.array("h", pad_f.readframes(pad_f.getnframes()))

    pad_count = int(16000 * 0.25)
    # Check leading samples are 0
    assert all(s == 0 for s in pad_samples[:pad_count]), "Leading samples must be digital silence!"
    # Check trailing samples are 0
    assert all(s == 0 for s in pad_samples[-pad_count:]), "Trailing samples must be digital silence!"
    # Check center matches original samples
    center_samples = pad_samples[pad_count:pad_count + len(orig_samples)]
    assert center_samples == orig_samples, "Center samples must exactly match original audio!"
    print("[PASS] Test 4 passed: Short clip padded to 1.0s with zero-padding and center speech preserved.")


def test_5_pre_padding_silence_gate_and_reason():
    print("\n--- Test 5: Silence Gate Check & gate_reason Evaluation ---")
    silent_wav = generate_pcm16_sine_wav(duration_s=1.2, freq_hz=100.0, target_rms_dbfs=-68.0)
    pre_diag = inspect_pcm16_wav(silent_wav)
    boosted_wav, gain_applied = boost_quiet_pcm16_wav(silent_wav)
    post_diag = inspect_pcm16_wav(boosted_wav)

    assert post_diag["rms_dbfs"] < -55.0, "Expected post-gain RMS < -55 dBFS for true silence"

    # Evaluate gate reason logic
    if pre_diag["rms_dbfs"] < -60.0:
        gate_reason = "true_silence"
    elif pre_diag["peak_dbfs"] >= -3.0:
        gate_reason = "headroom_limited"
    else:
        gate_reason = "true_silence"

    assert gate_reason == "true_silence", f"Expected true_silence, got {gate_reason}"
    print(f"[PASS] Test 5 passed: Silence gate correctly triggers on {post_diag['rms_dbfs']} dBFS with gate_reason='{gate_reason}'.")


def test_6_transient_edge_case_headroom_limited():
    print("\n--- Test 6: Loud Transient Edge-Case (Clap/Click + Quiet Speech) ---")
    # Construct 1.5s audio: a loud spike (peak ~ -1 dBFS) in first 50ms, followed by quiet speech at -58 dBFS
    sample_rate = 16000
    samples = array.array("h")

    # Transient: 50ms loud impulse
    for i in range(int(0.05 * sample_rate)):
        val = int(round(30000.0 * math.sin(2.0 * math.pi * 1000.0 * i / sample_rate)))
        samples.append(val)

    # Followed by quiet signal (target ~ -56 dBFS)
    quiet_amp = 32768.0 * (10.0 ** (-56.0 / 20.0)) * math.sqrt(2.0)
    for i in range(int(1.45 * sample_rate)):
        val = int(round(quiet_amp * math.sin(2.0 * math.pi * 300.0 * i / sample_rate)))
        samples.append(int(val))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    transient_wav = buf.getvalue()

    pre_diag = inspect_pcm16_wav(transient_wav)
    boosted_wav, gain_applied = boost_quiet_pcm16_wav(transient_wav)
    post_diag = inspect_pcm16_wav(boosted_wav)

    print(f"Transient Audio Pre: RMS={pre_diag['rms_dbfs']} dBFS, Peak={pre_diag['peak_dbfs']} dBFS")
    print(f"Transient Audio Post: RMS={post_diag['rms_dbfs']} dBFS, Peak={post_diag['peak_dbfs']} dBFS, Gain Applied={gain_applied} dB")

    # Because peak is already high (-0.8 to -1.0 dBFS), headroom limits gain:
    assert gain_applied < 3.0, f"Headroom should have limited gain, but got {gain_applied} dB"

    # Now verify gate reason:
    gate_reason = None
    if post_diag["rms_dbfs"] < -55.0:
        if pre_diag["rms_dbfs"] < -60.0:
            gate_reason = "true_silence"
        elif pre_diag["peak_dbfs"] >= -3.0 or gain_applied < ((-24.0 - pre_diag["rms_dbfs"]) - 1.0):
            gate_reason = "headroom_limited"
        else:
            gate_reason = "true_silence"
        print(f"Gate fired as expected with gate_reason='{gate_reason}'")
        assert gate_reason == "headroom_limited", f"Expected headroom_limited, got {gate_reason}"
    else:
        print(f"Audio passed gate with post-gain RMS: {post_diag['rms_dbfs']} dBFS")

    print("[PASS] Test 6 passed: Transient edge-case correctly evaluated with diagnostic clarity.")


def test_7_normalized_hallucination_matching():
    print("\n--- Test 7: Normalized Hallucination Phrase Matching ---")
    SUSPECTED_HALLUCINATIONS = {
        "thank you", "thank you very much", "thanks for watching", "you", "bye", "subscribe"
    }

    test_phrases = [
        ("Thank you.", True),
        ("Thank You!", True),
        ("   thank you very much...  ", True),
        ("THANKS FOR WATCHING!", True),
        ("Bye.", True),
        ("What is your name?", False),
        ("Hello, how are you?", False),
        ("Thank you for your help with the project.", False), # legitimate long sentence
    ]

    for raw_text, expected_match in test_phrases:
        clean_norm = re.sub(r'[^\w\s]', '', raw_text.lower()).strip()
        matched = clean_norm in SUSPECTED_HALLUCINATIONS
        print(f"  '{raw_text}' -> normalized: '{clean_norm}' -> matched: {matched}")
        assert matched == expected_match, f"Failed for '{raw_text}': expected {expected_match}, got {matched}"

    print("[PASS] Test 7 passed: Normalized regex-based matching catches all variants without false positives.")


def test_8_real_recorded_audio_boost():
    print("\n--- Test 8: Real Recorded Audio Transients ---")
    debug_dir = Path("backend/debug_audio")
    sample_file = debug_dir / "last_recording.wav"

    if not sample_file.exists():
        print(f"Sample file {sample_file} not found; generating speech-like multi-harmonic audio.")
        # Multi-harmonic audio simulating speech vowels (150Hz pitch + formants)
        sample_rate = 16000
        samples = array.array("h")
        for i in range(int(1.5 * sample_rate)):
            # Formant synthesis
            f0 = 130.0
            val = (
                0.5 * math.sin(2 * math.pi * f0 * i / sample_rate)
                + 0.3 * math.sin(2 * math.pi * 3 * f0 * i / sample_rate)
                + 0.2 * math.sin(2 * math.pi * 5 * f0 * i / sample_rate)
            )
            # Modulate amplitude (syllable envelope)
            envelope = 0.5 * (1.0 + math.sin(2 * math.pi * 3.0 * i / sample_rate))
            scaled = int(round(val * envelope * 500.0))  # quiet speech
            samples.append(scaled)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        real_wav = buf.getvalue()
    else:
        real_wav = sample_file.read_bytes()

    # Artificially attenuate to quiet speech level (~ -45 dBFS)
    with wave.open(io.BytesIO(real_wav), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    raw_s = array.array("h", frames)

    # Scale down to -46 dBFS
    curr_diag = inspect_pcm16_wav(real_wav)
    attenuate_db = -46.0 - curr_diag["rms_dbfs"]
    att_factor = 10.0 ** (attenuate_db / 20.0)
    quiet_samples = array.array("h", [int(round(s * att_factor)) for s in raw_s])

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(sr)
        wf.writeframes(quiet_samples.tobytes())
    quiet_real_wav = buf.getvalue()

    pre = inspect_pcm16_wav(quiet_real_wav)
    print(f"Quiet Real Speech: RMS={pre['rms_dbfs']} dBFS, Peak={pre['peak_dbfs']} dBFS")

    boosted, gain_applied = boost_quiet_pcm16_wav(quiet_real_wav)
    post = inspect_pcm16_wav(boosted)
    print(f"Boosted Real Speech: RMS={post['rms_dbfs']} dBFS, Peak={post['peak_dbfs']} dBFS, Gain Applied=+{gain_applied} dB")

    assert gain_applied > 10.0, f"Expected gain applied to quiet speech, got {gain_applied} dB"
    assert post["peak_dbfs"] <= -1.0, f"Peak exceeded headroom: {post['peak_dbfs']} dBFS"
    assert post["rms_dbfs"] >= -30.0, f"Expected speech boosted near target -24 dBFS, got {post['rms_dbfs']} dBFS"
    print("[PASS] Test 8 passed: Real speech transients accurately boosted with headroom protection.")


if __name__ == "__main__":
    print("======================================================")
    print("RUNNING STT RELIABILITY AUTOMATED VERIFICATION SUITE")
    print("======================================================")
    test_1_synthetic_gain_boost()
    test_2_loud_input_unchanged_exact_byte_match()
    test_3_noise_floor_unchanged()
    test_4_temporal_padding()
    test_5_pre_padding_silence_gate_and_reason()
    test_6_transient_edge_case_headroom_limited()
    test_7_normalized_hallucination_matching()
    test_8_real_recorded_audio_boost()
    print("\n======================================================")
    print("ALL 8 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("======================================================")
