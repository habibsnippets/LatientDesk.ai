"""LatientDesk.ai receptionist agent.

A tool-calling voice-agent skeleton for a dental front desk.

Flow: user text -> LLM -> if tool call: execute locally, feed result back ->
repeat -> final spoken response.

LLM: Groq-hosted Llama 3.1 via its OpenAI-compatible chat-completions API.
Default model is the free "llama-3.1-8b-instant"; switch with --model (e.g.
"llama-3.3-70b-versatile"). Auth reads GROQ_API_KEY from the environment.
No SDK: plain `requests`.

With no API key the CLI falls back to an offline scripted mode so the whole
loop (tools, state, conversation shape) can still be exercised end to end.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

import requests

import insurance
import pms

# Auto-load .env if present
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ---------------------------------------------------------------------------
# Groq client (OpenAI-compatible REST, no SDK)
# ---------------------------------------------------------------------------

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
MODELS_WITH_TOOLS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
}


class GroqError(RuntimeError):
    """Raised for any Groq API failure (auth, rate limit, bad request...)."""


def groq_chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """One chat-completion round-trip. Returns the full message dict."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise GroqError("GROQ_API_KEY is not set in the environment")

    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools

    resp = requests.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if resp.status_code == 401:
        raise GroqError("Groq rejected the API key (401). Check GROQ_API_KEY.")
    if resp.status_code == 404 and "model" in resp.text.lower():
        raise GroqError(
            f"Groq does not know model {model!r} (404). "
            f"Known tool-calling models: {sorted(MODELS_WITH_TOOLS)}"
        )
    if resp.status_code == 429:
        raise GroqError("Groq rate limit hit (429); slow down or retry shortly.")
    if resp.status_code != 200:
        raise GroqError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    if data.get("choices"):
        return data["choices"][0]["message"]
    raise GroqError(f"Unexpected Groq response shape: {str(data)[:300]}")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the front desk receptionist at a dental practice. \
Greet the caller, find out what they need, collect details like name, DOB, \
phone, check their insurance using the verify_insurance tool before \
confirming the price and then book the patient using the book_appointment \
tool. Speak in short precise and clear sentences."""


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI/Groq function-calling JSON Schema)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_open_slots",
            "description": (
                "Look up open appointment slots for a provider. Use this when "
                "the caller wants an appointment. Omit date_range to search "
                "all upcoming days; appointment_type must exactly match one "
                "of the practice's appointment type names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_id": {
                        "type": "integer",
                        "enum": [1, 2],
                        "description": "1 = Dr. Alice Chen (General Dentistry), 2 = Dr. Marcus Webb (Orthodontics)",
                    },
                    "date_range": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Optional inclusive [start, end] dates in YYYY-MM-DD",
                    },
                    "appointment_type": {
                        "type": "string",
                        "enum": [
                            "30 min checkup",
                            "45 min cleaning",
                            "60 min filling",
                            "60 min extraction",
                            "90 min root canal",
                        ],
                        "description": "Exact appointment type name",
                    },
                },
                "required": ["provider_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_appointment",
            "description": (
                "Book an open slot for a patient. slot_id must come from a "
                "previous get_open_slots result in this conversation. Do not "
                "call this before insurance has been verified and the caller "
                "has agreed to the price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {
                        "type": "string",
                        "description": "Slot identifier from get_open_slots, e.g. '1-20260914-0900'",
                    },
                    "patient_name": {"type": "string", "description": "Patient full name"},
                    "dob": {"type": "string", "description": "Patient date of birth, YYYY-MM-DD"},
                    "phone_number": {"type": "string", "description": "Patient contact phone"},
                    "appointment_type": {
                        "type": "string",
                        "enum": [
                            "30 min checkup",
                            "45 min cleaning",
                            "60 min filling",
                            "60 min extraction",
                            "90 min root canal",
                        ],
                        "description": "Exact appointment type name",
                    },
                },
                "required": ["slot_id", "patient_name", "dob", "phone_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_insurance",
            "description": (
                "Check a patient's dental insurance eligibility and benefits "
                "before quoting a price. payer must be one of the practice's "
                "accepted payers: ABC Dental, Delta Prime, Guardian Shield."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Patient full name as registered with the payer"},
                    "dob": {"type": "string", "description": "Patient date of birth, YYYY-MM-DD"},
                    "payer": {
                        "type": "string",
                        "enum": ["ABC Dental", "Delta Prime", "Guardian Shield"],
                        "description": "Insurance payer name",
                    },
                    "service_date": {
                        "type": "string",
                        "description": "Optional planned appointment date YYYY-MM-DD for waiting-period checks",
                    },
                },
                "required": ["name", "dob", "payer"],
            },
        },
    },
]

# Practice price list used to answer "what will it cost" after verification.
PRICE_LIST: dict[str, str] = {
    "30 min checkup": "$95",
    "45 min cleaning": "$140",
    "60 min filling": "$240",
    "60 min extraction": "$300",
    "90 min root canal": "$650",
}


def _summarize_eligibility(result: dict[str, Any]) -> str:
    """Compact, model-friendly summary of a verify_insurance result."""
    plan = result.get("eligibility", {}).get("plan", {})
    waits = result.get("response", {}).get("active_waiting_periods") or []
    lines = [
        f"eligible={result.get('eligibility', {}).get('status')}",
        f"deductible=${plan.get('deductible', 0):.2f}",
        f"coverage_percent={plan.get('coverage_percent', 0)}",
    ]
    if waits:
        lines.append("active_waiting_periods=" + json.dumps(waits))
    else:
        lines.append("active_waiting_periods=none")
    return "; ".join(lines)


def execute_tool(name: str, arguments: dict[str, Any], conn: sqlite3.Connection) -> str:
    """Dispatch a tool call against the local fake PMS / mock insurance."""
    try:
        if name == "get_open_slots":
            slots = pms.get_open_slots(
                int(arguments["provider_id"]),
                date_range=tuple(arguments["date_range"]) if arguments.get("date_range") else None,
                appointment_type=arguments.get("appointment_type"),
            )
            if not slots:
                return "[] (no open slots match; offer the caller other days or the other provider)"
            # Cap the payload; the model only needs a handful of options.
            return json.dumps(
                {
                    "slots": slots[:8],
                    "total_open": len(slots),
                    "note": "showing first 8; ask for a narrower date_range for more",
                }
            )

        if name == "create_appointment":
            booked = pms.create_appointment(
                slot_id=arguments["slot_id"],
                patient_name=arguments["patient_name"],
                dob=arguments["dob"],
                phone_number=arguments["phone_number"],
                appointment_type=arguments.get("appointment_type"),
            )
            return json.dumps({"booked": booked})

        if name == "verify_insurance":
            result = insurance.verify_insurance(
                name=arguments["name"],
                dob=arguments["dob"],
                payer=arguments["payer"],
                service_date=arguments.get("service_date"),
            )
            return json.dumps({"summary": _summarize_eligibility(result)})

        return f"ERROR unknown tool {name!r}"
    except KeyError as exc:
        return f"ERROR missing required argument {exc}"
    except (ValueError, TypeError) as exc:
        return f"ERROR {exc}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 6  # safety cap: user text -> (tool -> result ->) final reply


def agent_reply(
    user_text: str,
    messages: list[dict[str, Any]],
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the loop until the model produces a final spoken response.

    messages is mutated in place (system prompt + history + this turn's
    tool traffic). Returns (final_text, messages).
    """
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_text})

    for _ in range(max_rounds):
        msg = groq_chat(model, messages, tools=TOOLS, api_key=api_key)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            content = (msg.get("content") or "").strip()
            messages.append({"role": "assistant", "content": content})
            return content, messages

        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = execute_tool(name, arguments, conn)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": result,
                }
            )

    # Safety valve: never leave the caller hanging.
    fallback = "I'm sorry, I couldn't complete that. Let me transfer you to a human colleague."
    messages.append({"role": "assistant", "content": fallback})
    return fallback, messages


# ---------------------------------------------------------------------------
# Offline fallback (no API key): scripted plumbing demo
# ---------------------------------------------------------------------------

def _offline_receptionist(conn: sqlite3.Connection) -> None:
    """Scripted walk-through proving the tool wiring works without a key."""
    print("[offline mode] No GROQ_API_KEY found — running a scripted demo of "
          "the tool wiring (no LLM). Get a free key at https://console.groq.com")

    name = "Maya Okafor"
    dob = "1988-03-14"
    print(f'caller: "Hi, I need a cleaning."')

    try:
        checks = insurance.verify_insurance(name, dob, "ABC Dental")
        print(f"[tool] verify_insurance -> {_summarize_eligibility(checks)}")
    except ValueError as exc:
        print(f"[tool] verify_insurance -> skipped ({exc})")
        print("[note] Fill the fixture slots in insurance.py to enable insurance checks.")

    slots = pms.get_open_slots(1, appointment_type="45 min cleaning")
    if not slots:
        print("[tool] get_open_slots -> no slots; demo ends")
        return
    first = slots[0]
    print(f'[tool] get_open_slots -> {len(slots)} open; first: {first["slot_id"]} at {first["start_time"]}')

    print(f'agent: "We have {first["start_time"][:10]} at {first["start_time"][11:16]}. '
          f'Your cleaning is fully covered. Shall I book it?"')
    print('caller: "Yes please."')

    booked = pms.create_appointment(
        first["slot_id"], name, dob, "5035550142", "45 min cleaning"
    )
    print(f'[tool] create_appointment -> booked appointment_id={booked["appointment_id"]} '
          f'for {booked["start_time"]}')
    print(f'agent: "You\'re booked for {booked["start_time"]}. See you then!"')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LatientDesk.ai dental receptionist agent")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model id (default {DEFAULT_MODEL})")
    parser.add_argument("--db", default=str(pms.DB_PATH), help="SQLite database path")
    parser.add_argument("--reseed", action="store_true", help="Reseed the demo database before starting")
    parser.add_argument("--max-rounds", type=int, default=MAX_TOOL_ROUNDS, help="Max tool rounds per turn")
    args = parser.parse_args()

    conn = pms.connect(args.db)
    if args.reseed or not Path(args.db).exists():
        n = pms.seed(conn)
        print(f"[db] seeded {n} open slots at {args.db}")

    if not os.environ.get("GROQ_API_KEY"):
        _offline_receptionist(conn)
        conn.close()
        return

    print(f"[agent] {args.model} ready. Ctrl-C to exit.\n")
    messages: list[dict[str, Any]] = []
    try:
        while True:
            try:
                user_text = input("caller> ").strip()
            except EOFError:
                break
            if not user_text:
                continue
            if user_text.lower() in {"quit", "exit", "bye"} and len(messages) <= 1:
                break
            reply, messages = agent_reply(
                user_text, messages, conn, model=args.model, max_rounds=args.max_rounds
            )
            print(f"agent> {reply}\n")
    except KeyboardInterrupt:
        print("\n[agent] goodbye")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
