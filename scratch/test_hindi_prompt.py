import io, sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
from faster_whisper import WhisperModel

m_small = WhisperModel('small', device='cpu', compute_type='int8')
wav_hi1 = open(r'backend\debug_audio\benchmark_samples\HI-1.wav', 'rb').read()
wav_hi2 = open(r'backend\debug_audio\benchmark_samples\HI-2.wav', 'rb').read()

print('--- Test small model WITH script-guiding initial_prompt ---')
segs1, _ = m_small.transcribe(io.BytesIO(wav_hi1), language='hi', beam_size=1, temperature=0.0, initial_prompt='यह हिंदी में है।')
res1 = ' '.join(s.text for s in segs1).strip()
print('HI-1 small with prompt:', res1)

segs2, _ = m_small.transcribe(io.BytesIO(wav_hi2), language='hi', beam_size=1, temperature=0.0, initial_prompt='यह हिंदी में है।')
res2 = ' '.join(s.text for s in segs2).strip()
print('HI-2 small with prompt:', res2)
