import React, { useState, useRef } from 'react';

function App() {
  const [recording, setRecording] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('hi');
  const [status, setStatus] = useState('');
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Start recording
  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Your browser does not support audio recording.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = event => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = handleRecordingStop;

      mediaRecorderRef.current.start();
      setRecording(true);
      setStatus('Recording...');
    } catch (err) {
      alert('Could not start recording: ' + err.message);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      setStatus('Processing audio...');
    }
  };

  // Handle audio after recording stopped
  const handleRecordingStop = async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
    // Send to backend
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');
      formData.append('lang', sourceLang);

      setStatus('Sending audio for transcription...');
      const res = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Transcription failed: ${res.statusText}`);
      }

      const data = await res.json();
      setTranscribedText(data.text || '');
      setStatus(`Detected language: ${data.detected_lang} (confidence: ${data.confidence.toFixed(2)})`);
    } catch (err) {
      setStatus('Error: ' + err.message);
    }
  };

  // Handle translation
  const translateText = async () => {
    if (!transcribedText.trim()) {
      alert('Please provide some text to translate.');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('text', transcribedText);
      formData.append('target_lang', targetLang);

      setStatus('Translating...');
      const res = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Translation failed: ${res.statusText}`);
      }

      const data = await res.json();
      setTranslatedText(data.translated);
      setStatus('Translation complete.');
    } catch (err) {
      setStatus('Error: ' + err.message);
    }
  };

  // Play TTS audio
  const playTTS = async () => {
    if (!translatedText.trim()) {
      alert('No translated text to play.');
      return;
    }

    try {
      // Choose voice based on targetLang (simple map)
      const voiceMap = {
        en: 'en-US-JennyNeural',
        hi: 'hi-IN-SwaraNeural',
        zh: 'zh-CN-XiaoxiaoNeural',
      };
      const voice = voiceMap[targetLang] || 'en-US-JennyNeural';

      const formData = new FormData();
      formData.append('text', translatedText);
      formData.append('voice', voice);

      setStatus('Generating speech...');
      const res = await fetch('http://localhost:8000/tts', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`TTS failed: ${res.statusText}`);
      }

      const audioBlob = await res.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
      setStatus('Playing audio...');
    } catch (err) {
      setStatus('Error: ' + err.message);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: 'auto', padding: 20, fontFamily: 'Arial, sans-serif' }}>
      <h1>Real-time Speech Translator</h1>

      <div style={{ marginBottom: 10 }}>
        <label>
          Source Language:{' '}
          <input
            value={sourceLang}
            onChange={e => setSourceLang(e.target.value)}
            placeholder="e.g., en"
            style={{ width: 50 }}
          />
        </label>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>
          Target Language:{' '}
          <input
            value={targetLang}
            onChange={e => setTargetLang(e.target.value)}
            placeholder="e.g., hi"
            style={{ width: 50 }}
          />
        </label>
      </div>

      <div style={{ marginBottom: 10 }}>
        {!recording ? (
          <button onClick={startRecording}>Start Recording</button>
        ) : (
          <button onClick={stopRecording}>Stop Recording</button>
        )}
      </div>

      <div style={{ marginBottom: 10 }}>
        <strong>Status:</strong> {status}
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Transcribed Text:</label>
        <textarea
          rows={4}
          cols={50}
          value={transcribedText}
          onChange={e => setTranscribedText(e.target.value)}
          placeholder="Transcription will appear here"
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: 10 }}>
        <button onClick={translateText} disabled={!transcribedText.trim()}>
          Translate
        </button>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Translated Text:</label>
        <textarea
          rows={4}
          cols={50}
          value={translatedText}
          readOnly
          placeholder="Translated text will appear here"
          style={{ width: '100%', backgroundColor: '#f0f0f0' }}
        />
      </div>

      <div>
        <button onClick={playTTS} disabled={!translatedText.trim()}>
          Play Translated Audio
        </button>
      </div>
    </div>
  );
}

export default App;
