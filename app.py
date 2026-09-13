"""LatientDesk.ai — Streamlit Interactive Web Application.

Provides a full GUI for:
1. Live Receptionist Chat with visible tool-call tracing.
2. Practice Management System (PMS) live schedule and booked appointments.
3. Insurance verification playground (mock 270/271).
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

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
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_db():
    return pms.connect(pms.DB_PATH)


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dental-braces.png", width=64)
    st.title("LatientDesk.ai")
    st.caption("Autonomous Dental AI Receptionist")

    st.markdown("---")
    st.subheader("⚙️ Settings")

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
        st.rerun()

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_pms, tab_insurance = st.tabs(
    ["💬 AI Receptionist Chat", "📅 PMS Calendar & Appointments", "🛡️ Insurance 270/271 Inspector"]
)

# -----------------------------------------------------------------------------
# Tab 1: Live Chat Interface
# -----------------------------------------------------------------------------
with tab_chat:
    st.markdown('<div class="main-title">Dental Front Desk Receptionist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Chat with LatientDesk.ai in real time. The receptionist checks calendar availability, verifies insurance, and books appointments.</div>',
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []  # Internal raw LLM history

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! Welcome to our dental clinic. How can I assist you today?"}
        ]

    # Render previous chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🦷"):
                st.write(msg["content"])
                if msg.get("tools"):
                    with st.expander("🔧 Tools Executed", expanded=False):
                        for t in msg["tools"]:
                            st.markdown(f"**Tool:** `{t['name']}`")
                            st.json(t["result"])

    # Handle quick scenario or chat input
    user_input = st.chat_input("Say something to the receptionist...")
    if "quick_prompt" in st.session_state and st.session_state.quick_prompt:
        user_input = st.session_state.quick_prompt
        del st.session_state.quick_prompt

    if user_input:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        # Generate agent reply
        with st.chat_message("assistant", avatar="🦷"):
            with st.spinner("Receptionist is thinking and checking systems..."):
                conn = get_db()
                tools_used = []

                # Intercept tool executions to display in UI
                original_exec = agent.execute_tool

                def _intercept_tool(name, args, db_conn):
                    res = original_exec(name, args, db_conn)
                    try:
                        parsed = json.loads(res)
                    except Exception:
                        parsed = res
                    tools_used.append({"name": name, "args": args, "result": parsed})
                    return res

                agent.execute_tool = _intercept_tool
                try:
                    reply, st.session_state.messages = agent.agent_reply(
                        user_input,
                        st.session_state.messages,
                        conn,
                        model=model_choice,
                    )
                finally:
                    agent.execute_tool = original_exec
                    conn.close()

                st.write(reply)
                if tools_used:
                    with st.expander("🔧 Tools Executed", expanded=True):
                        for t in tools_used:
                            st.markdown(f"**Tool:** `{t['name']}`")
                            st.json(t["result"])

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply, "tools": tools_used}
                )

# -----------------------------------------------------------------------------
# Tab 2: PMS Live Schedule & Database
# -----------------------------------------------------------------------------
with tab_pms:
    st.markdown("### 📅 Practice Management System (PMS)")
    st.caption("Live view of appointments and calendar slots stored in SQLite (`demo.db`).")

    conn = get_db()
    try:
        # Metrics
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
