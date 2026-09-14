# Real-Time AI Speech Translator with RAG

An end-to-end Speech-to-Speech translation pipeline combining **Faster-Whisper Speech-to-Text**, a **Vector RAG Retrieval Engine**, a **Context-Aware LLM Translator** (Groq LLaMA 3.3 70B / Local Multilingual Engine), and **Microsoft Edge-TTS**.

```
                   [ User Speech Input ]
                             │
                             ▼
                   🎤 Microphone / Audio
                             │
                             ▼
                    ┌─────────────────┐
                    │ Faster-Whisper  │ (Speech-to-Text)
                    └────────┬────────┘
                             │
                        Transcript
                             │
                             ▼
                    ┌─────────────────┐
                    │   RAG Engine    │ (Vector Retrieval & Grounding)
                    └────────┬────────┘
                             │
                     Domain Context
                             │
                             ▼
                    ┌─────────────────┐
                    │ LLM Translator  │ (Groq LLaMA 3.3 70B / Local Engine)
                    └────────┬────────┘ + Multi-Turn History
                             │
                      Translated Text
                             │
                             ▼
                    ┌─────────────────┐
                    │    Edge-TTS     │ (Neural Speech Synthesis)
                    └────────┬────────┘
                             │
                             ▼
                   🔊 Translated Audio Output
```

---

## Technical Architecture & Core System Design

### 1. Speech-to-Text (STT) Layer
- Powered by `faster-whisper` (`small`/`base` model using `int8` quantization on CPU).
- Automatic WebM/Opus to 16kHz mono WAV conversion with audio gain normalization (`-af "volume=1.8"`) to boost soft microphone recordings.
- Dual-pass Silero VAD filtering and hallucination suppression to filter out silent artifacts.

### 2. Retrieval-Augmented Generation (RAG) Engine
- Domain-specific glossaries (`technical_terms.txt`, `business_terms.txt`, `medical_terms.txt`) indexed into an in-memory vector store.
- Cosine similarity retrieval over n-gram TF-IDF embeddings to extract technical terminology and grounding context.
- Prevents spurious matches on casual speech while preserving domain jargon (e.g., *Kubernetes*, *RAG Architecture*, *Vector Database*).

### 3. Context-Aware LLM Translation
- Primary engine: **Groq Cloud API** (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) for ultra-low latency translations (~200ms).
- Zero-crash fallback: Local multilingual engine operating out-of-the-box without requiring API keys.
- Multi-turn conversation manager: Retains session state to resolve ambiguous pronouns (*it*, *they*, *this*) across dialogue turns.

### 4. Neural Speech Synthesis (TTS) Layer
- Synthesizes translated text into high-quality neural voice streams using `edge_tts`.
- Language-to-Voice mapping (`hi-IN-SwaraNeural` for Hindi, `en-US-JennyNeural` for English, `es-ES-ElviraNeural` for Spanish, etc.).

### 5. Web Interface & Benchmarking
- Built with React 19 and custom Vanilla CSS.
- Real-time pipeline step indicator and end-to-end execution latency benchmarking bar (`STT`, `RAG`, `LLM`, `TTS`, `Total`).

---

## Local Setup & Execution

### Prerequisites
- Python 3.10+ installed
- Node.js 18+ installed
- FFmpeg installed and available on system PATH

### 1. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt
```

*(Optional) Configure Groq API Key for Cloud LLM:*
Copy `.env.example` to `.env` and set your key from [console.groq.com](https://console.groq.com):
```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Start the FastAPI server:
```bash
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
```
Interactive API Documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Frontend Setup

In a new terminal window:

```bash
cd frontend

# Install Node dependencies
npm install

# Start React development server
npm start
```
The application will open automatically at [http://localhost:3000](http://localhost:3000).

---

## Running Automated Verification Suite

Run the end-to-end automated test suite to verify RAG retrieval, translation accuracy, session history retention, and audio synthesis:

```bash
python test_system.py
```

---

## Project Directory Structure

```
Real-time-translator/
├── backend/
│   ├── knowledge/               # Domain Knowledge Base Documents
│   │   ├── technical_terms.txt  # RAG, Kubernetes, Vector DB, API, etc.
│   │   ├── business_terms.txt   # ROI, KPI, Stakeholders, Sprint, etc.
│   │   └── medical_terms.txt    # Triage, Prognosis, Hypertension, etc.
│   ├── rag_engine.py            # Vector store, chunking & similarity retrieval
│   ├── llm_translator.py        # Groq/OpenAI client, prompt engine & history manager
│   ├── server.py                # FastAPI REST API & WebSocket server
│   ├── realtime_translator.py   # CLI speech translator
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # Production container configuration
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main application & audio pipeline
│   │   ├── App.css              # Glassmorphism dark mode UI styling
│   │   └── index.css            # Base design system & tokens
│   └── package.json
├── test_system.py               # System verification test suite
└── README.md
```
