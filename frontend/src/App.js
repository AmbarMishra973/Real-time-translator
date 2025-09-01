import React, { useState, useRef } from 'react';
import Waveform from './Waveform';

function App() {
  const [recording, setRecording] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('hi');
  const [status, setStatus] = useState('');
  const [audioBlob, setAudioBlob] = useState(null);
  const [playing, setPlaying] = useState(false);

  const socketRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const languageOptions = [
    { label: 'English', value: 'en' },
    { label: 'Hindi', value: 'hi' },
    { label: 'Chinese', value: 'zh' },
  ];

  // Start recording with streaming via WebSocket
  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Your browser does not support audio recording.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // Use MediaRecorder for capturing audio blob for waveform
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = e => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
      };

      mediaRecorderRef.current.start();

      // Setup WebSocket streaming as before
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      socketRef.current = new WebSocket('ws://localhost:8000/ws/transcribe');

      socketRef.current.onopen = () => {
        setStatus('Streaming to server...');
        setRecording(true);
      };

      socketRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.text) {
          setTranscribedText(prev => prev ? prev + ' ' + data.text : data.text);
          setStatus(`Detected language: ${data.detected_lang} (confidence: ${data.confidence})`);
        }
      };

      socketRef.current.onerror = (err) => {
        setStatus('WebSocket error: ' + err.message);
      };

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const pcmData = convertFloat32ToInt16(inputData);
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
          socketRef.current.send(pcmData);
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      socketRef.current.audioContext = audioContext;
      socketRef.current.processor = processor;
    } catch (err) {
      alert('Error starting recording: ' + err.message);
    }
  };

  const stopRecording = () => {
    if (socketRef.current) {
      socketRef.current.close();
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
    }

    if (socketRef.current?.processor) {
      socketRef.current.processor.disconnect();
    }

    if (socketRef.current?.audioContext) {
      socketRef.current.audioContext.close();
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    setRecording(false);
    setStatus('Stopped streaming.');
  };

  const convertFloat32ToInt16 = (buffer) => {
    let l = buffer.length;
    const buf = new Int16Array(l);
    for (let i = 0; i < l; i++) {
      buf[i] = Math.min(1, buffer[i]) * 0x7FFF;
    }
    return new Int16Array(buf).buffer;
  };

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

      const data = await res.json();
      setTranslatedText(data.translated);
      setStatus('Translation complete.');
    } catch (err) {
      setStatus('Error: ' + err.message);
    }
  };

  const playTTS = async () => {
    if (!translatedText.trim()) {
      alert('No translated text to play.');
      return;
    }

    const voiceMap = {
      en: 'en-US-JennyNeural',
      hi: 'hi-IN-SwaraNeural',
      zh: 'zh-CN-XiaoxiaoNeural',
    };
    const voice = voiceMap[targetLang] || 'en-US-JennyNeural';

    try {
      const formData = new FormData();
      formData.append('text', translatedText);
      formData.append('voice', voice);

      setStatus('Generating speech...');
      const res = await fetch('http://localhost:8000/tts', {
        method: 'POST',
        body: formData,
      });

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
      <h1>Real-Time Speech Translator</h1>

      <div style={{ marginBottom: 10 }}>
        <label>
          Source Language:{' '}
          <select value={sourceLang} onChange={e => setSourceLang(e.target.value)}>
            {languageOptions.map(lang => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>
          Target Language:{' '}
          <select value={targetLang} onChange={e => setTargetLang(e.target.value)}>
            {languageOptions.map(lang => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ marginBottom: 10 }}>
        {!recording ? (
          <button onClick={startRecording}>Start Real-Time Recording</button>
        ) : (
          <button onClick={stopRecording}>Stop Recording</button>
        )}
      </div>

      <div style={{ marginBottom: 10 }}>
        <strong>Status:</strong> {status}
      </div>

      {/* Waveform visualization */}
      <div style={{ marginBottom: 20 }}>
        <Waveform audioBlob={audioBlob} playing={playing} onFinish={() => setPlaying(false)} />
        <button
          onClick={() => setPlaying(p => !p)}
          disabled={!audioBlob}
          style={{ marginTop: 10 }}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label>Transcribed Text:</label>
        <textarea
          rows={4}
          cols={50}
          value={transcribedText}
          onChange={e => setTranscribedText(e.target.value)}
          placeholder="Real-time transcription appears here"
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
          placeholder="Translation will appear here"
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
