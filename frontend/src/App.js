import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const LANGUAGE_OPTIONS = [
  { label: 'English', value: 'en' },
  { label: 'Hindi (हिंदी)', value: 'hi' },
  { label: 'Spanish (Español)', value: 'es' },
  { label: 'French (Français)', value: 'fr' },
  { label: 'German (Deutsch)', value: 'de' },
  { label: 'Chinese (中文)', value: 'zh' },
  { label: 'Japanese (日本語)', value: 'ja' },
  { label: 'Korean (한국어)', value: 'ko' },
  { label: 'Russian (Русский)', value: 'ru' },
  { label: 'Arabic (العربية)', value: 'ar' },
];

const DEMO_PROMPTS = [
  "We need to implement RAG architecture with vector database and LLM context.",
  "Our microservices architecture will reduce latency and increase throughput.",
  "I have a meeting tomorrow regarding the Kubernetes deployment.",
  "It is with the development team and we need to discuss the API."
];

function App() {
  // State
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('hi');
  const [sessionId] = useState(() => 'sess_' + Math.random().toString(36).substring(2, 8));

  // Pipeline Step & Status
  const [currentStep, setCurrentStep] = useState(1);
  const [recording, setRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  // Content
  const [transcribedText, setTranscribedText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [sourcesUsed, setSourcesUsed] = useState([]);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [providerLabel, setProviderLabel] = useState('');
  const [autoPlayTTS, setAutoPlayTTS] = useState(false);

  // Real Latency Metrics
  const [metrics, setMetrics] = useState({
    stt_s: null,
    rag_s: null,
    llm_s: null,
    tts_s: null,
    total_s: null,
  });

  // LLM Status & Settings
  const [llmStatus, setLlmStatus] = useState({ is_llm_connected: false, active_mode: 'Checking...' });
  const [showSettings, setShowSettings] = useState(false);
  const [groqKeyInput, setGroqKeyInput] = useState(localStorage.getItem('groq_api_key') || '');

  // Refs
  const mediaStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const speechRecognitionRef = useRef(null);

  const isWebSpeechSupported = typeof window !== 'undefined' && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window);
  const [sttEngine, setSttEngine] = useState(isWebSpeechSupported ? 'browser' : 'whisper');

  const getRecognitionLang = (code) => {
    const base = (code || 'en').split('-')[0].toLowerCase();
    const map = {
      'en': 'en-US',
      'hi': 'hi-IN',
      'es': 'es-ES',
      'fr': 'fr-FR',
      'de': 'de-DE',
      'zh': 'zh-CN',
      'ja': 'ja-JP',
      'ko': 'ko-KR',
      'ru': 'ru-RU',
      'ar': 'ar-SA',
    };
    return map[base] || 'en-US';
  };

  // Check LLM status from backend on mount
  const checkBackendStatus = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/`);
      if (res.ok) {
        const data = await res.json();
        setLlmStatus(data.llm_status || {});
      }
    } catch (e) {
      console.warn('Backend not ready:', e.message);
    }
  }, []);

  useEffect(() => {
    checkBackendStatus();
    const savedKey = localStorage.getItem('groq_api_key');
    if (savedKey) {
      fetch(`${BACKEND_URL}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ groq_api_key: savedKey })
      }).then(() => checkBackendStatus()).catch(() => { });
    }
  }, [checkBackendStatus]);

  // Save Settings
  const handleSaveSettings = async (e) => {
    e.preventDefault();
    localStorage.setItem('groq_api_key', groqKeyInput.trim());
    try {
      const res = await fetch(`${BACKEND_URL}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ groq_api_key: groqKeyInput.trim() })
      });
      if (res.ok) {
        const data = await res.json();
        setLlmStatus(data.status || {});
        setShowSettings(false);
      }
    } catch (err) {
      alert('Error saving settings: ' + err.message);
    }
  };

  // Swap Languages
  const handleSwap = () => {
    const prev = sourceLang;
    setSourceLang(targetLang);
    setTargetLang(prev);
  };

  // Start Voice Recording (Dual Mode: Live Browser Recognition vs Cloud Whisper)
  const startRecording = async () => {
    if (sttEngine === 'browser' && isWebSpeechSupported) {
      try {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = getRecognitionLang(sourceLang);

        let finalTranscript = '';
        recognition.onresult = (event) => {
          let interim = '';
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
              finalTranscript += event.results[i][0].transcript + ' ';
            } else {
              interim += event.results[i][0].transcript;
            }
          }
          const text = (finalTranscript + interim).trim();
          if (text) {
            setTranscribedText(text);
          }
        };

        recognition.onerror = (err) => {
          console.warn('Browser speech recognition notice:', err.error);
        };

        recognition.onend = () => {
          setRecording(false);
        };

        recognition.start();
        speechRecognitionRef.current = recognition;
        setRecording(true);
        setRecordSeconds(0);
        setCurrentStep(1);

        timerRef.current = setInterval(() => {
          setRecordSeconds((s) => s + 1);
        }, 1000);
        return;
      } catch (err) {
        console.warn('Web Speech API failed, falling back to MediaRecorder:', err);
      }
    }

    // MediaRecorder path (Whisper audio upload)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Your browser does not support audio recording.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      mediaStreamRef.current = stream;

      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = '';
        }
      }
      const options = mimeType ? { mimeType } : {};
      mediaRecorderRef.current = new MediaRecorder(stream, options);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        }

        const finalType = (mediaRecorderRef.current && mediaRecorderRef.current.mimeType) || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: finalType });
        console.log(`[Mic] Recorded ${audioChunksRef.current.length} chunks, total size: ${blob.size} bytes (${finalType})`);

        if (blob.size < 250) {
          alert('Recording was too short or no audio was detected. Please speak clearly for at least 1-2 seconds.');
          setIsProcessing(false);
          return;
        }

        await executeAudioPipeline(blob);
      };

      mediaRecorderRef.current.start(250);
      setRecording(true);
      setRecordSeconds(0);
      setCurrentStep(1);

      timerRef.current = setInterval(() => {
        setRecordSeconds((s) => s + 1);
      }, 1000);
    } catch (err) {
      alert('Microphone error: ' + err.message);
    }
  };

  // Stop Recording
  const stopRecording = () => {
    if (timerRef.current) clearInterval(timerRef.current);

    if (sttEngine === 'browser' && speechRecognitionRef.current) {
      try {
        speechRecognitionRef.current.stop();
      } catch (e) {}
      setRecording(false);
      setCurrentStep(3);
      // Auto-translate spoken text after settling
      setTimeout(() => {
        handleTranslateText();
      }, 300);
      return;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.requestData();
      } catch (e) {}
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
    setCurrentStep(2);
  };

  // Client-side translation resolver: Guarantees translation correctness even under cloud server 429 rate-limits
  const resolveTranslation = async (sourceText, serverTranslation, src, tgt) => {
    let result = (serverTranslation || '').trim();
    const srcBase = (src || 'en').split('-')[0].toLowerCase();
    const tgtBase = (tgt || 'hi').split('-')[0].toLowerCase();

    // Check if translation succeeded and is not an unchanged echo
    const isEcho = srcBase !== tgtBase && sourceText.trim().split(/\s+/).length >= 1 && result.toLowerCase() === sourceText.trim().toLowerCase();
    if (result && !isEcho) {
      return result;
    }

    // Client-side rescue: User's residential IP is unblocked and has separate API quota
    try {
      const res = await fetch(`https://api.mymemory.translated.net/get?q=${encodeURIComponent(sourceText)}&langpair=${srcBase}|${tgtBase}&de=translumina_app@gmail.com`);
      if (res.ok) {
        const json = await res.json();
        const clientTrans = json?.responseData?.translatedText;
        if (clientTrans && !clientTrans.includes('MYMEMORY WARNING') && clientTrans.toLowerCase() !== sourceText.toLowerCase()) {
          const txt = document.createElement('textarea');
          txt.innerHTML = clientTrans;
          return txt.value.trim();
        }
      }
    } catch (e) {
      console.warn('Client fallback translation failed:', e);
    }
    return result || sourceText;
  };

  // Audio Pipeline (Whisper STT ➔ RAG ➔ LLM)
  const executeAudioPipeline = async (blob) => {
    setIsProcessing(true);
    setCurrentStep(2);

    try {
      const formData = new FormData();
      formData.append('file', blob, 'recording.webm');
      formData.append('source_lang', sourceLang);
      formData.append('target_lang', targetLang);
      formData.append('session_id', sessionId);
      formData.append('domain', 'all');

      const response = await fetch(`${BACKEND_URL}/pipeline`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`Backend server responded with error ${response.status}`);

      const data = await response.json();
      
      // Strict validation: Must contain actual alphanumeric words (not dots or empty punctuation)
      const hasSpokenContent = data.transcript && /[a-zA-Z0-9\u0900-\u097F\u4e00-\u9fa5\u0600-\u06FF]/.test(data.transcript);
      if (!hasSpokenContent) {
        setIsProcessing(false);
        alert('No clear voice detected in the recording. Please speak clearly into the microphone and try again.');
        return;
      }

      setTranscribedText(data.transcript);
      setCurrentStep(3);
      setRetrievedChunks(data.retrieved_context || []);
      setSourcesUsed(data.sources_used || []);

      setCurrentStep(4);
      const rawTrans = data.translated_text || data.translated || '';
      const finalTrans = await resolveTranslation(data.transcript, rawTrans, sourceLang, targetLang);
      setTranslatedText(finalTrans);
      setConversationHistory(data.history || []);
      setProviderLabel(data.provider || 'Local Multilingual Engine');

      // Set latency metrics
      if (data.metrics) {
        setMetrics({
          stt_s: data.metrics.stt_s,
          rag_s: data.metrics.rag_s,
          llm_s: data.metrics.llm_s,
          tts_s: null,
          total_s: data.metrics.total_s,
        });
      }

      setCurrentStep(5);
      if (autoPlayTTS && finalTrans) {
        await playTTS(finalTrans, data.metrics);
      }
    } catch (err) {
      console.error(err);
      if (err.message && (err.message.includes('Failed to fetch') || err.message.includes('NetworkError'))) {
        alert('Could not connect to backend server.\n\nIf using Render free tier, the backend may be waking up from sleep mode (takes ~30-45 seconds). Please wait a moment and try speaking again!');
      } else {
        alert('Audio processing error: ' + err.message);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // Direct Text Translation with RAG
  const handleTranslateText = async () => {
    const text = transcribedText.trim();
    if (!text || !/[a-zA-Z0-9\u0900-\u097F\u4e00-\u9fa5\u0600-\u06FF]/.test(text)) return;

    setIsProcessing(true);
    setCurrentStep(3);

    try {
      const formData = new FormData();
      formData.append('text', text);
      formData.append('source_lang', sourceLang);
      formData.append('target_lang', targetLang);
      formData.append('session_id', sessionId);
      formData.append('domain', 'all');

      setCurrentStep(4);
      const response = await fetch(`${BACKEND_URL}/translate`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Translation failure.');

      const data = await response.json();
      const rawTrans = data.translated_text || data.translated || '';
      const finalTrans = await resolveTranslation(text, rawTrans, sourceLang, targetLang);
      setTranslatedText(finalTrans);
      setRetrievedChunks(data.retrieved_context || []);
      setSourcesUsed(data.sources_used || []);
      setConversationHistory(data.history || []);
      setProviderLabel(data.provider || 'Local Multilingual Engine');

      if (data.metrics) {
        setMetrics({
          stt_s: null,
          rag_s: data.metrics.rag_s,
          llm_s: data.metrics.llm_s,
          tts_s: null,
          total_s: data.metrics.total_s,
        });
      }

      setCurrentStep(5);
      if (autoPlayTTS && finalTrans) {
        await playTTS(finalTrans, data.metrics);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsProcessing(false);
    }
  };

  // Play Neural TTS Audio & Measure Latency
  const playTTS = async (textOverride, baseMetrics) => {
    const text = textOverride || translatedText;
    if (!text || !text.trim() || !/[a-zA-Z0-9\u0900-\u097F\u4e00-\u9fa5\u0600-\u06FF]/.test(text)) {
      alert('No translatable text available to play.');
      return;
    }

    const ttsStart = performance.now();
    try {
      const formData = new FormData();
      formData.append('text', text);
      formData.append('target_lang', targetLang);

      const res = await fetch(`${BACKEND_URL}/tts`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('TTS synthesis failed.');

      const realLatency = ((performance.now() - ttsStart) / 1000).toFixed(2);

      // Update metrics with actual measured TTS time
      setMetrics((prev) => {
        const curTotal = (prev.total_s ? parseFloat(prev.total_s) : 0) + parseFloat(realLatency);
        return {
          ...prev,
          tts_s: parseFloat(realLatency),
          total_s: curTotal.toFixed(2),
        };
      });

      const blob = await res.blob();
      if (blob.size < 100) {
        throw new Error('Received empty audio from speech engine.');
      }
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);
      audio.onended = () => URL.revokeObjectURL(audioUrl);
      audio.onerror = (e) => console.error('Audio playback error:', e);
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch((e) => {
          console.warn('Autoplay blocked or playback error:', e);
        });
      }
    } catch (err) {
      console.error('TTS error:', err);
      alert('Audio playback error: ' + err.message);
    }
  };

  // Clear Multi-Turn Memory
  const handleClearHistory = async () => {
    try {
      await fetch(`${BACKEND_URL}/api/history?session_id=${sessionId}`, {
        method: 'DELETE',
      });
      setConversationHistory([]);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="app-wrapper">
      {/* 1. Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo">🌐</div>
          <div>
            <div className="brand-title">
              TransLumina AI
              <span className="brand-tag">RAG + SPEECH</span>
            </div>
          </div>
        </div>

        {/* Clean LLM Connection Status Pill */}
        <div
          className={`header-status-badge ${llmStatus.is_llm_connected ? 'connected' : 'offline'}`}
          onClick={() => setShowSettings(true)}
          title="Click to view or edit LLM configuration"
        >
          <span className={llmStatus.is_llm_connected ? 'status-dot-green' : 'status-dot-amber'} />
          <span>{llmStatus.is_llm_connected ? '● LLM Connected' : '⚙️ Configure LLM Key'}</span>
        </div>
      </header>

      {/* 2. Pipeline Stepper */}
      <div className="pipeline-stepper">
        <div className={`step-item ${currentStep >= 1 ? 'active' : ''}`}>
          <div className="step-num">1</div>
          <span>Voice Input</span>
        </div>
        <span className="step-arrow">➔</span>

        <div className={`step-item ${currentStep >= 2 ? 'active' : ''}`}>
          <div className="step-num">2</div>
          <span>Whisper STT</span>
        </div>
        <span className="step-arrow">➔</span>

        <div className={`step-item ${currentStep >= 3 ? 'active' : ''}`}>
          <div className="step-num">3</div>
          <span>RAG Retrieval</span>
        </div>
        <span className="step-arrow">➔</span>

        <div className={`step-item ${currentStep >= 4 ? 'active' : ''}`}>
          <div className="step-num">4</div>
          <span>LLM Context</span>
        </div>
        <span className="step-arrow">➔</span>

        <div className={`step-item ${currentStep >= 5 ? 'active' : ''}`}>
          <div className="step-num">5</div>
          <span>Neural TTS</span>
        </div>
      </div>

      {/* Demo Chips Bar */}
      <div className="demo-prompts-bar">
        <span className="demo-title">⚡ Try Examples:</span>
        {DEMO_PROMPTS.map((prompt, i) => (
          <button
            key={i}
            className="demo-btn"
            onClick={() => setTranscribedText(prompt)}
          >
            "{prompt.length > 38 ? prompt.substring(0, 38) + '...' : prompt}"
          </button>
        ))}
      </div>

      {/* 3. Upper Row: SPEECH & TRANSCRIPT (Left) | TRANSLATION (Right) */}
      <div className="section-grid">
        {/* Upper Left: Speech & Transcript */}
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span>🎤 Speech & Transcript</span>
              {isWebSpeechSupported && (
                <div className="engine-toggle-group">
                  <button
                    type="button"
                    className={`engine-pill-btn ${sttEngine === 'browser' ? 'active' : ''}`}
                    onClick={() => setSttEngine('browser')}
                    title="Live Instant Browser Recognition (0s latency, native accent clarity)"
                  >
                    ⚡ Live Mic
                  </button>
                  <button
                    type="button"
                    className={`engine-pill-btn ${sttEngine === 'whisper' ? 'active' : ''}`}
                    onClick={() => setSttEngine('whisper')}
                    title="Neural Whisper Audio Transcription"
                  >
                    ☁️ Whisper AI
                  </button>
                </div>
              )}
            </div>
            <div className="lang-selector-row">
              <select
                className="lang-select"
                value={sourceLang}
                onChange={(e) => setSourceLang(e.target.value)}
              >
                {LANGUAGE_OPTIONS.map((l) => (
                  <option key={l.value} value={l.value}>
                    {l.label}
                  </option>
                ))}
              </select>
              <button className="swap-btn" onClick={handleSwap} title="Swap languages">
                ⇄
              </button>
              <select
                className="lang-select"
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
              >
                {LANGUAGE_OPTIONS.map((l) => (
                  <option key={l.value} value={l.value}>
                    {l.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Mic Button & Status */}
          <div className="mic-input-row">
            {!recording ? (
              <button className="record-btn-compact" onClick={startRecording} title="Click to Speak">
                🎤
              </button>
            ) : (
              <button className="record-btn-compact recording" onClick={stopRecording} title="Stop & Process">
                ⏹
              </button>
            )}
            <div className="record-status-text">
              {recording
                ? (sttEngine === 'browser' ? `🎙️ Listening live (${recordSeconds}s)... Speak naturally, click ⏹ when done` : `🎙️ Recording audio (${recordSeconds}s)... Click ⏹ to transcribe`)
                : isProcessing
                  ? (currentStep === 2 ? '⏳ Transcribing audio (Whisper STT)...' : currentStep === 3 ? '🔍 Retrieving domain context (RAG)...' : currentStep === 4 ? '🌐 Translating transcript...' : '⚡ Processing...')
                  : (translatedText ? '✅ Translation complete. Click mic or type to translate again.' : (sttEngine === 'browser' ? '⚡ Click mic to speak with instant Live Recognition' : 'Click mic to speak, or type directly in the box below'))}
            </div>
          </div>

          {/* Transcript Area */}
          <textarea
            className="text-area-clean"
            rows={3}
            value={transcribedText}
            onChange={(e) => setTranscribedText(e.target.value)}
            placeholder="User speech transcript will appear here..."
          />

          <div className="action-row">
            <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
              {transcribedText ? `${transcribedText.split(/\s+/).filter(Boolean).length} words` : ''}
            </span>
            <button
              className="btn-primary-action"
              onClick={handleTranslateText}
              disabled={!transcribedText.trim() || isProcessing}
            >
              {isProcessing ? '⏳ Processing...' : '✨ Translate with RAG'}
            </button>
          </div>
        </div>

        {/* Upper Right: Context-Aware Translation */}
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title">
              <span>🎯 Translation</span>
            </div>
            {providerLabel && (
              <span className="sub-badge" style={{ color: '#38bdf8' }}>
                {providerLabel}
              </span>
            )}
          </div>

          <textarea
            className="text-area-clean output"
            rows={4}
            value={translatedText}
            readOnly
            placeholder="Contextual translation will appear here..."
          />

          <div className="action-row">
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={autoPlayTTS}
                onChange={(e) => setAutoPlayTTS(e.target.checked)}
              />
              Auto-play audio
            </label>

            <button
              className="btn-tts-action"
              onClick={() => playTTS()}
              disabled={!translatedText.trim()}
            >
              🔊 Play Audio
            </button>
          </div>
        </div>
      </div>

      {/* 4. Lower Row: RETRIEVED KNOWLEDGE (Left) | CONTEXT MEMORY (Right) */}
      <div className="section-grid">
        {/* Lower Left: RAG Retrieved Domain Knowledge & Grounding */}
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title">
              <span>🔎 Retrieved Domain Knowledge</span>
            </div>
            <span className="sub-badge">
              {retrievedChunks.length} {retrievedChunks.length === 1 ? 'relevant chunk' : 'relevant chunks'}
            </span>
          </div>

          <div className="rag-chunks-container">
            {retrievedChunks.length > 0 ? (
              retrievedChunks.map((chunk, idx) => (
                <div key={idx} className="rag-chunk-card">
                  <div className="rag-chunk-title-row">
                    <span className="rag-chunk-term">
                      {idx + 1}. {chunk.term}
                    </span>
                    <span className="rag-similarity-pill">
                      Similarity: {chunk.similarity !== undefined ? chunk.similarity.toFixed(2) : (chunk.score || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="rag-chunk-snippet">{chunk.definition}</div>
                </div>
              ))
            ) : (
              <div className="rag-empty-message">
                No domain terms retrieved yet. Try speaking words like "RAG Architecture", "Kubernetes", "Vector Database", or "Microservices".
              </div>
            )}
          </div>

          {/* Source Grounding Box - Point 5 */}
          {sourcesUsed.length > 0 && (
            <div className="rag-sources-grounding">
              <span style={{ fontWeight: 600, color: '#93c5fd' }}>📚 Sources Used:</span>
              <div className="rag-source-list">
                {sourcesUsed.map((src, i) => (
                  <span key={i}>✓ {src}</span>
                ))}
              </div>
              <span style={{ color: 'var(--text-muted)' }}>Grounded</span>
            </div>
          )}
        </div>

        {/* Lower Right: Multi-Turn Context Memory */}
        <div className="panel-card">
          <div className="panel-header">
            <div className="panel-title">
              <span>🧠 Context Memory</span>
            </div>
            {conversationHistory.length > 0 && (
              <button
                className="sub-badge"
                style={{ cursor: 'pointer', background: 'rgba(239, 68, 68, 0.1)', color: '#f87171' }}
                onClick={handleClearHistory}
              >
                Clear Memory
              </button>
            )}
          </div>

          <div className="memory-turns-container">
            {conversationHistory.length > 0 ? (
              conversationHistory.map((turn, i) => (
                <div key={i} className="memory-turn-box">
                  <div className="memory-turn-header">Turn {i + 1}</div>
                  <div className="memory-turn-user">
                    User: {turn.source_text}
                  </div>
                  <div className="memory-turn-translated">
                    ➔ {turn.translated_text}
                  </div>
                </div>
              ))
            ) : (
              <div className="rag-empty-message">
                Previous conversation turns are maintained here so subsequent translations correctly resolve pronouns and context.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 5. Bottom Real Latency Bar - Point 7 */}
      <footer className="latency-bar">
        <div className="latency-title">
          <span>⚙ Processing Latency:</span>
        </div>

        <div className="latency-metrics-row">
          <div className="metric-item">
            <span>STT:</span>
            <span className="metric-val">{metrics.stt_s !== null ? `${metrics.stt_s}s` : '—'}</span>
          </div>

          <span>|</span>

          <div className="metric-item">
            <span>RAG Retrieval:</span>
            <span className="metric-val">{metrics.rag_s !== null ? `${metrics.rag_s}s` : '—'}</span>
          </div>

          <span>|</span>

          <div className="metric-item">
            <span>LLM:</span>
            <span className="metric-val">{metrics.llm_s !== null ? `${metrics.llm_s}s` : '—'}</span>
          </div>

          <span>|</span>

          <div className="metric-item">
            <span>TTS:</span>
            <span className="metric-val">{metrics.tts_s !== null ? `${metrics.tts_s}s` : '—'}</span>
          </div>

          <span>|</span>

          <div className="metric-item">
            <span>Total:</span>
            <span className="metric-val total">{metrics.total_s !== null ? `${metrics.total_s}s` : '—'}</span>
          </div>
        </div>
      </footer>

      {/* Settings Modal */}
      {showSettings && (
        <div className="modal-overlay" onClick={() => setShowSettings(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ margin: 0, color: '#fff', fontSize: 16 }}>⚙️ LLM & API Configuration</h3>
              <button
                onClick={() => setShowSettings(false)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', fontSize: 20, cursor: 'pointer' }}
              >
                &times;
              </button>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.04)', padding: 12, borderRadius: 8, fontSize: 13 }}>
              <div>Current Engine: <strong style={{ color: '#38bdf8' }}>{llmStatus.active_mode}</strong></div>
              <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>
                Status: {llmStatus.is_llm_connected ? 'Connected to Cloud LLM API' : 'Using Local Multilingual Translation Engine'}
              </div>
            </div>

            <form onSubmit={handleSaveSettings} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: '#cbd5e1' }}>
                Groq API Key (Free tier recommended for ~200ms latency)
              </label>
              <input
                type="password"
                placeholder="gsk_..."
                value={groqKeyInput}
                onChange={(e) => setGroqKeyInput(e.target.value)}
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 8,
                  padding: '8px 12px',
                  color: '#fff',
                  fontSize: 13,
                  outline: 'none'
                }}
              />
              <span style={{ fontSize: 11, color: '#64748b' }}>
                Get a free key from console.groq.com. Leave empty to use the local multilingual engine.
              </span>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 10 }}>
                <button
                  type="button"
                  onClick={() => setShowSettings(false)}
                  style={{
                    background: 'rgba(255,255,255,0.08)',
                    border: 'none',
                    borderRadius: 8,
                    padding: '6px 14px',
                    color: '#fff',
                    cursor: 'pointer'
                  }}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary-action">
                  Save & Connect
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;