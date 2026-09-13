# LatientDesk.ai 🦷🎙️

An autonomous AI receptionist for dental practices capable of real-time voice conversations, appointment booking, and insurance benefit verification.

## Features

- **Practice Management System (PMS) Integration**: SQLite-backed calendar grid with providers, patient profiles, and multi-slot procedure bookings (e.g. 90-minute root canals consuming consecutive 30-minute grid blocks).
- **Insurance Verification (Mock 270/271)**: Validates patient eligibility, copays, deductibles, and procedure waiting periods before quoting prices.
- **Tool-Calling Agent**: Powered by Groq-hosted LLMs with functions for slot checking, insurance verification, and appointment booking.
- **Streaming Voice Pipeline**: Built with [Pipecat](https://github.com/pipecat-ai/pipecat), integrating Silero VAD, Groq Whisper STT, Groq LLM, and local Piper TTS.
- **Evaluation Harness**: Automated benchmark scoring booking accuracy, verify-before-quote enforcement, benefit quote accuracy, and turn latency.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │     Caller (Voice/CLI)  │
                    └───────────┬─────────────┘
                                │
                 [voice_agent.py] (Pipecat Pipeline)
                 - Silero VAD (voice detection)
                 - Groq Whisper STT
                 - Local Piper TTS
                                │
                    ┌───────────▼─────────────┐
                    │        agent.py         │
                    │       (Groq LLM)        │
                    └───────┬───────────┬─────┘
           Tools            │           │
      ┌─────────────────────┘           └───────────────────────┐
      ▼                                                         ▼
[pms.py]                                              [insurance.py]
- SQLite (demo.db)                                    - Mock X12 270/271 Eligibility
- Providers & Appointments                            - Deductibles & Coinsurance
- Multi-block slot booking                            - Waiting period evaluation
```

---

## Quick Start

### 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-voice.txt
```

### 2. Configuration

Copy the example environment file and add your Groq API key:

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 3. Initialize the Practice Database

Seed the practice database with providers and open calendar slots:

```bash
python pms.py
```

### 4. Run the Receptionist

**Interactive Streamlit Web Dashboard (Voice & Text GUI):**
```bash
streamlit run app.py
```

**Interactive Text CLI:**
```bash
python agent.py
```

**Real-Time Streaming Voice Receptionist:**
```bash
python voice_agent.py
```

**Run Evaluation Harness:**
```bash
# Offline mock benchmark
python eval/evaluate.py --scenarios eval/scenarios.txt --offline

# Live LLM evaluation
python eval/evaluate.py --scenarios eval/scenarios.txt
```
