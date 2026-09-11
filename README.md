# 🌐 Real-Time AI Translator + RAG

An end-to-end Speech-to-Speech translation system combining **Faster-Whisper STT**, a **Semantic RAG Vector Retrieval Engine**, a **Context-Aware LLM Translator**, and **Microsoft Edge-TTS**.

```
                         USER
                          │
                          ▼
                    🎤 Microphone
                          │
                          ▼
                    ┌──────────┐
                    │ Whisper  │
                    │   STT    │
                    └────┬─────┘
                         │
                         ▼
                     Transcript
                         │
                         ▼
                 ┌───────────────┐
                 │   Embedding   │
                 │   & Vector    │
                 │    Search     │
                 └───────┬───────┘
                         │
                         ▼
                  Relevant Context (RAG)
                         │
                         ▼
                ┌─────────────────┐
                │      LLM        │
                │  (Groq / Llama) │
                │  + Conversation │
                │     Context     │
                └────────┬────────┘
                         │
                         ▼
                  Translated Text
                         │
                         ▼
                     Edge-TTS
                         │
                         ▼
                    🔊 Speaker
```

---

## ✨ Features

- **🎙️ Real-Time Speech-to-Text (Whisper)**: Accurate transcription using `faster-whisper` running locally with VAD filtering.
- **🔍 Domain Knowledge Retrieval (RAG)**: Built-in glossaries (`technical_terms.txt`, `business_terms.txt`, `medical_terms.txt`) indexed into an in-memory vector database with TF-IDF/n-gram cosine similarity. Also supports on-the-fly custom term addition and dynamic vector re-indexing.
- **🧠 Context-Aware LLM Translation**: Uses Groq (`llama-3.3-70b-versatile` or `llama-3.1-8b-instant`) for ultra-low latency contextual translations, preserving technical terminology and loan words.
- **💬 Multi-Turn Conversation Memory**: Tracks previous conversation turns per session to resolve ambiguous pronouns (*"it"*, *"they"*, *"we"*) and maintain natural continuity.
- **🔊 Neural Speech Synthesis (Edge-TTS)**: Produces natural multilingual speech output in Hindi (`hi-IN-SwaraNeural`), English, Spanish, French, German, Chinese, etc.
- **⚡ Zero-Crash Fallback**: Automatically falls back to a term-preserving translator if no API key is provided, ensuring out-of-the-box functionality.
- **💎 Modern Web Interface**: Glassmorphism dark mode UI featuring live pipeline status, audio visualizer waveform, real-time RAG context inspector cards, conversation memory timeline, and knowledge base manager.
- **💻 Standalone CLI**: Interactive terminal translator (`python -m backend.realtime_translator`) for quick console tests.

---

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

*(Optional) Configure API Keys for Groq / OpenAI:*
Copy `.env.example` to `.env` and add your Groq key (get a free key at [console.groq.com](https://console.groq.com)):
```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Start the FastAPI Server:
```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend Setup

```bash
cd frontend
npm install
npm start
```
The web app opens automatically at [http://localhost:3000](http://localhost:3000).

### 3. Standalone CLI Translator

You can also run the full pipeline entirely in your terminal:
```bash
python -m backend.realtime_translator
```

---

## 🧪 Running Automated Tests

A complete verification test suite is included:
```bash
python test_system.py
```
This verifies:
1. Knowledge base parsing and vector indexing
2. RAG semantic retrieval accuracy
3. LLM translation and multi-turn context retention
4. Edge-TTS neural audio synthesis
5. FastAPI REST API endpoints (`/`, `/translate`, `/tts`, `/api/knowledge`)

---

## 📂 Project Structure

```
Real-time-translator/
├── backend/
│   ├── knowledge/               # Domain Knowledge Base Documents
│   │   ├── technical_terms.txt  # Kubernetes, RAG, Docker, API, etc.
│   │   ├── business_terms.txt   # ROI, Sprint, KPI, Stakeholders, etc.
│   │   └── medical_terms.txt    # Hypertension, Triage, Prognosis, etc.
│   ├── rag_engine.py            # Vector store, chunking & semantic retrieval
│   ├── llm_translator.py        # Groq/OpenAI client, prompt engine & history
│   ├── server.py                # FastAPI REST API & WebSocket server
│   ├── realtime_translator.py   # Standalone CLI speech translator
│   ├── requirements.txt         # Python dependencies
│   └── .env.example             # Environment variables template
├── frontend/
│   ├── src/
│   │   ├── App.js               # Glassmorphism UI & audio pipeline
│   │   ├── App.css              # Custom styling, dark mode & animations
│   │   ├── Waveform.js          # Audio waveform visualizer (WaveSurfer)
│   │   └── index.css            # Base design system & typography
│   └── package.json
├── test_system.py               # Automated verification suite
└── README.md
```
