"""LatientDesk.ai — Streamlit Interactive Web Application.

Provides a full GUI for:
1. Live Receptionist Voice & Text Chat with visible tool-call tracing.
2. In-browser audio input (microphone) -> Groq Whisper STT.
3. Spoken audio responses powered by local Piper TTS (Amy voice).
4. Practice Management System (PMS) live schedule and booked appointments.
5. Insurance verification playground (mock 270/271).
"""

import hashlib
import io
import json
import os
import re
import sqlite3
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from piper.voice import PiperVoice

import agent
import insurance
import pms

# Page configuration
st.set_page_config(
    page_title="LatientDesk.ai — Dental AI Receptionist",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .voice-badge {
        background-color: #E0E7FF;
        color: #3730A3;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_db():
    return pms.connect(pms.DB_PATH)


# -----------------------------------------------------------------------------
# Speech Services (STT & TTS)
# -----------------------------------------------------------------------------

@st.cache_resource
def get_piper_voice():
    """Load local Piper TTS voice model once into memory."""
    model_path = Path(__file__).resolve().parent / "voices" / "en_US-amy-medium.onnx"
    config_path = Path(__file__).resolve().parent / "voices" / "en_US-amy-medium.onnx.json"
    if model_path.exists() and config_path.exists():
        return PiperVoice.load(str(model_path), config_path=str(config_path))
    return None


def clean_text_for_speech(text: str) -> str:
    """Sanitize text for natural speech synthesis.

    Strips markdown formatting (asterisks, bullet points, hashes, backticks)
    and converts raw ISO dates/datetimes into natural spoken phrases.
    """
    # Remove markdown bold/italics asterisks
    text = re.sub(r"\*+", "", text)
    # Remove markdown headers and list bullets
    text = re.sub(r"^[#\-*]\s+", "", text, flags=re.MULTILINE)
    # Remove backticks and symbols
    text = text.replace("`", "").replace("~", "")

    # Convert ISO datetimes like '2026-09-14 09:00:00' to 'September 14 at 9:00 AM'
    def _replace_dt(m):
        raw = m.group(0)
        try:
            dt = datetime.fromisoformat(raw)
            return dt.strftime("%B %d at %I:%M %p").replace(" 0", " ")
        except Exception:
            return raw

    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\b", _replace_dt, text)

    # Convert ISO dates like '2026-09-14' to 'September 14'
    def _replace_date(m):
        raw = m.group(0)
        try:
            d = datetime.strptime(raw, "%Y-%m-%d")
            return d.strftime("%B %d").replace(" 0", " ")
        except Exception:
            return raw

    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", _replace_date, text)
    return text.strip()


def synthesize_speech(text: str) -> bytes | None:
    """Synthesize text to spoken 16-bit PCM WAV using Piper TTS."""
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        return None
    try:
        voice = get_piper_voice()
        if not voice:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            for chunk in voice.synthesize(clean_text):
                audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        st.warning(f"Voice synthesis note: {exc}")
        return None


def transcribe_audio_groq(audio_bytes: bytes, api_key: str) -> str:
    """Transcribe browser microphone audio with Groq Whisper."""
    if not audio_bytes:
        return ""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("caller_speech.wav", audio_bytes, "audio/wav")},
            data={"model": "whisper-large-v3-turbo"},
            timeout=30.0,
        )
        if r.status_code == 200:
            return r.json().get("text", "").strip()
        else:
            st.error(f"Whisper transcription failed ({r.status_code}): {r.text[:200]}")
            return ""
    except Exception as exc:
        st.error(f"Speech transcription error: {exc}")
        return ""


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dental-braces.png", width=64)
    st.title("LatientDesk.ai")
    st.caption("Autonomous Dental AI Receptionist")

    st.markdown("---")
    st.subheader("🎙️ Voice & Audio Settings")

    enable_voice_out = st.toggle(
        "🔊 Voice Responses (TTS)",
        value=True,
        help="Use Piper TTS (Amy voice) to read receptionist replies aloud.",
    )

    model_choice = st.selectbox(
        "LLM Model",
        options=["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
        index=0,
        help="Groq high-speed tool-calling models.",
    )

    has_key = bool(os.environ.get("GROQ_API_KEY"))
    if has_key:
        st.success("API Key Active (Groq)", icon="✅")
    else:
        st.error("GROQ_API_KEY not found in .env", icon="⚠️")

    st.markdown("---")
    st.subheader("💡 Quick Test Scenarios")
    st.caption("Click to inject test prompts:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Maya Okafor\n(100% Free)"):
            st.session_state.quick_prompt = "Hi, I'm Maya Okafor, born 1988-03-14. I have ABC Dental. I'd like to book a cleaning. What will it cost?"
    with col2:
        if st.button("Liam Barnes\n($50 Ded, 80%)"):
            st.session_state.quick_prompt = "Hi, I'm Liam Barnes, born 1992-07-02 with Delta Prime insurance. How much will a filling cost me?"

    col3, col4 = st.columns(2)
    with col3:
        if st.button("Sofia Reyes\n(Wait Period)"):
            st.session_state.quick_prompt = "Hello, my name is Sofia Reyes, born 1995-11-27, with Guardian Shield. I need a filling done."
    with col4:
        if st.button("No Insurance\n(Self-Pay)"):
            st.session_state.quick_prompt = "Hi, I don't have dental insurance. How much is a checkup and when is Dr. Chen free?"

    st.markdown("---")
    if st.button("🔄 Reset Dental Calendar (PMS)", help="Reseed ~200 fresh calendar slots"):
        conn = get_db()
        n = pms.seed(conn)
        conn.close()
        st.toast(f"Successfully reseeded {n} open slots!", icon="📅")
        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.last_processed_audio = ""
        st.rerun()

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_pms, tab_insurance = st.tabs(
    ["🎙️ Live Receptionist (Voice & Text)", "📅 PMS Calendar & Appointments", "🛡️ Insurance 270/271 Inspector"]
)

# -----------------------------------------------------------------------------
# Tab 1: Live Voice & Text Receptionist
# -----------------------------------------------------------------------------
with tab_chat:
    st.markdown('<div class="main-title">Dental Front Desk Receptionist</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="voice-badge">🎙️ Voice Mode Active</span> '
        '<span style="color:#6B7280; font-size: 0.95rem;">Speak with your microphone or type below. LatientDesk.ai verifies insurance, checks availability, and books appointments.</span>',
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []  # Raw LLM message history

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello! Welcome to our dental clinic. How can I help you today?",
                "tools": [],
                "audio": None,
            }
        ]

    if "last_processed_audio" not in st.session_state:
        st.session_state.last_processed_audio = ""

    # Audio input widget for speaking directly into the browser
    st.markdown("#### 🎙️ Voice Input")
    recorded_audio = st.audio_input("Click the microphone to speak to the receptionist:")

    user_input = ""
    # Process audio if new recording detected
    if recorded_audio is not None:
        audio_bytes = recorded_audio.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_processed_audio:
            st.session_state.last_processed_audio = audio_hash
            with st.spinner("Transcribing your voice with Groq Whisper..."):
                key = os.environ.get("GROQ_API_KEY", "")
                transcribed = transcribe_audio_groq(audio_bytes, key)
                if transcribed:
                    user_input = transcribed

    # Quick scenario button override
    if "quick_prompt" in st.session_state and st.session_state.quick_prompt:
        user_input = st.session_state.quick_prompt
        del st.session_state.quick_prompt

    # Text input box
    typed_input = st.chat_input("Or type your message here...")
    if typed_input:
        user_input = typed_input

    # Render previous conversation history
    st.markdown("---")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🦷"):
                st.write(msg["content"])
                if msg.get("audio"):
                    st.audio(msg["audio"], format="audio/wav")

    # Process new user input (from voice or text)
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🦷"):
            with st.spinner("Receptionist is checking calendar & insurance..."):
                conn = get_db()
                try:
                    reply, st.session_state.messages = agent.agent_reply(
                        user_input,
                        st.session_state.messages,
                        conn,
                        model=model_choice,
                    )
                finally:
                    conn.close()

                st.write(reply)

                # Generate speech response if enabled
                reply_audio = None
                if enable_voice_out:
                    with st.spinner("Generating voice response (Piper TTS)..."):
                        reply_audio = synthesize_speech(reply)
                        if reply_audio:
                            st.audio(reply_audio, format="audio/wav", autoplay=True)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply, "audio": reply_audio}
                )

# -----------------------------------------------------------------------------
# Tab 2: PMS Live Schedule & Database
# -----------------------------------------------------------------------------
with tab_pms:
    st.markdown("### 📅 Practice Management System (PMS)")
    st.caption("Live view of appointments and calendar slots stored in SQLite (`demo.db`).")

    conn = get_db()
    try:
        total_booked = conn.execute("SELECT COUNT(*) FROM appointments WHERE status = 'booked'").fetchone()[0]
        total_open = conn.execute("SELECT COUNT(*) FROM appointments WHERE status = 'open'").fetchone()[0]
        total_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Booked Appointments", total_booked)
        m2.metric("Available Open Slots", total_open)
        m3.metric("Registered Patients", total_patients)

        st.markdown("---")
        st.subheader("📋 Booked Appointments")
        booked_df = pd.read_sql_query(
            """
            SELECT 
                a.appointment_id AS ID,
                p.full_name AS Patient,
                p.phone_number AS Phone,
                pr.name AS Provider,
                pr.specialty AS Specialty,
                a.appt_type AS Procedure,
                a.start_time AS 'Start Time',
                a.end_time AS 'End Time'
            FROM appointments a
            JOIN patients p ON p.patient_id = a.patient_id
            JOIN providers pr ON pr.provider_id = a.provider_id
            WHERE a.status = 'booked'
            ORDER BY a.start_time DESC
            """,
            conn,
        )
        if not booked_df.empty:
            st.dataframe(booked_df, use_container_width=True, hide_index=True)
        else:
            st.info("No booked appointments yet. Ask the receptionist to book one!")

        st.markdown("---")
        st.subheader("🗓️ Next Available Open Slots")
        provider_filter = st.radio(
            "Filter by Provider:",
            options=["All Providers", "Dr. Alice Chen (General Dentistry)", "Dr. Marcus Webb (Orthodontics)"],
            horizontal=True,
        )

        query = """
            SELECT 
                a.slot_id AS 'Slot ID',
                pr.name AS Provider,
                a.start_time AS 'Start Time',
                a.end_time AS 'End Time'
            FROM appointments a
            JOIN providers pr ON pr.provider_id = a.provider_id
            WHERE a.status = 'open'
        """
        if "Alice Chen" in provider_filter:
            query += " AND a.provider_id = 1"
        elif "Marcus Webb" in provider_filter:
            query += " AND a.provider_id = 2"
        query += " ORDER BY a.start_time LIMIT 25"

        open_df = pd.read_sql_query(query, conn)
        st.dataframe(open_df, use_container_width=True, hide_index=True)

    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Tab 3: Insurance 270/271 Test Bench
# -----------------------------------------------------------------------------
with tab_insurance:
    st.markdown("### 🛡️ Insurance Eligibility & Benefits Playground")
    st.caption("Simulate an ANSI X12 270 Inquiry and view the parsed 271 EDI response.")

    with st.form("insurance_form"):
        col_name, col_dob, col_payer = st.columns(3)
        with col_name:
            test_name = st.text_input("Patient Name", value="Maya Okafor")
        with col_dob:
            test_dob = st.text_input("Date of Birth (YYYY-MM-DD)", value="1988-03-14")
        with col_payer:
            test_payer = st.selectbox("Insurance Payer", options=insurance.list_payers())

        submit_ins = st.form_submit_button("🔍 Verify Coverage")

    if submit_ins:
        try:
            res = insurance.verify_insurance(test_name, test_dob, test_payer)
            plan = res.get("eligibility", {}).get("plan", {})

            col_status, col_ded, col_cov = st.columns(3)
            status = res.get("eligibility", {}).get("status", "unknown").upper()
            col_status.metric("Plan Status", status)
            col_ded.metric("Deductible", f"${plan.get('deductible', 0):.2f}")
            col_cov.metric("Coinsurance / Coverage", f"{plan.get('coverage_percent', 0)}%")

            active_waits = res.get("response", {}).get("active_waiting_periods", [])
            if active_waits:
                st.warning(f"⚠️ Active Waiting Periods: {json.dumps(active_waits, indent=2)}")
            else:
                st.success("✅ No Active Waiting Periods. Benefits immediately effective.")

            with st.expander("📄 Full Parsed 271 JSON Payload", expanded=True):
                st.json(res)

        except ValueError as exc:
            st.error(f"Eligibility Verification Failed: {exc}")
