"""LatientDesk.ai evaluation harness — runs scripted caller dialogues through the
receptionist agent in text mode and scores the full pipeline.

Metrics
  1. booking accuracy                - create_appointment succeeded
  2. verification-before-quote       - verify_insurance ran (successfully)
                                       before any price was cited in agent text
  3. end-to-end turn latency         - p50/p95 seconds from end of user speech
                                       (text submitted) to start of agent
                                       speech (reply produced). In text mode
                                       each turn is one LLM round-trip, so this
                                       isolates Groq network+generation time;
                                       the same probes wrap the voice pipeline
                                       later without changing the metric.

Usage
  .venv/bin/python eval/evaluate.py --scenarios eval/scenarios.txt          # live (needs GROQ_API_KEY)
  .venv/bin/python eval/evaluate.py --scenarios eval/selftest.txt --offline # mock LLM, no key
  .venv/bin/python eval/evaluate.py --model llama-3.3-70b-versatile

agent.py is never modified: probes and audit hooks wrap core.groq_chat and
core.execute_tool at runtime, and each dialogue runs against a freshly seeded
in-memory database.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as core  # noqa: E402
import insurance  # noqa: E402
import pms  # noqa: E402

PRICE_RE = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)", re.IGNORECASE)

DASH = "-" * 78


# ---------------------------------------------------------------------------
# Scenario parsing
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    label: str
    utterances: list[str]


def parse_scenarios(path: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    current: Scenario | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current = Scenario(label=line[3:].strip(), utterances=[])
            scenarios.append(current)
        elif not line or line.startswith("#"):
            continue
        elif current is not None:
            current.utterances.append(line)
    return scenarios


# ---------------------------------------------------------------------------
# Instrumentation: audit trail + latency probes (no agent.py edits)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    utterance: str
    reply: str = ""
    llm_latency: float | None = None


@dataclass
class DialogueAudit:
    scenario: str = ""
    turns: list[Turn] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    quoted_prices: list[tuple[int, float]] = field(default_factory=list)

    def first_quote_turn(self) -> int | None:
        return self.quoted_prices[0][0] if self.quoted_prices else None

    def first_verify_tool_turn(self) -> int | None:
        for ev in self.tool_events:
            if ev["tool"] == "verify_insurance" and not ev["error"]:
                return ev["turn"]
        return None

    def verified_patient(self) -> str | None:
        for ev in self.tool_events:
            if ev["tool"] == "verify_insurance" and not ev["error"]:
                return ev.get("patient")
        return None

    def waiting_period_surfaced(self) -> bool:
        for ev in self.tool_events:
            if ev["tool"] == "verify_insurance" and not ev["error"]:
                if "active_waiting_periods=[" in ev.get("result", "") and "none" not in ev.get("result", ""):
                    return True
        return False

    def booked_ok(self) -> bool:
        return any(
            ev["tool"] == "create_appointment" and not ev["error"]
            for ev in self.tool_events
        )


class Instrumented:
    """Patches core.groq_chat (latency/quotes) and core.execute_tool (audit)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.audit = DialogueAudit()
        self._turn_idx = -1
        self._last_verified: tuple[str, str] | None = None  # (name, dob)
        self._orig_chat = core.groq_chat
        self._orig_exec = core.execute_tool
        self._patches: list = []
        self._last_summary = ""

    # -- core.execute_tool wrapper ------------------------------------------
    def _exec(self, name: str, arguments: dict[str, Any], conn: sqlite3.Connection) -> str:
        result = self._orig_exec(name, arguments, self.conn)
        event = {
            "turn": self._turn_idx,
            "tool": name,
            "error": result.startswith("ERROR"),
            "result": result,
        }
        if name == "verify_insurance":
            # Record the identity on attempt (error or not) so the mock can
            # continue its scripted flow; scoring only counts successful ones.
            self._last_verified = (arguments.get("name", ""), arguments.get("dob", ""))
            if not event["error"]:
                event["patient"] = arguments.get("name", "")
        self.audit.tool_events.append(event)
        return result

    # -- shared per-turn recording -------------------------------------------
    def _record(self, content: str, latency: float) -> None:
        if 0 <= self._turn_idx < len(self.audit.turns):
            turn = self.audit.turns[self._turn_idx]
            turn.reply = content
            turn.llm_latency = latency
            for m in PRICE_RE.finditer(content):
                self.audit.quoted_prices.append((self._turn_idx, float(m.group(1))))

    # -- live LLM wrapper -----------------------------------------------------
    def _chat(self, model, messages, tools=None, api_key=None, timeout=60.0):
        t0 = time.perf_counter()
        msg = self._orig_chat(model, messages, tools=tools, api_key=api_key, timeout=timeout)
        content = (msg.get("content") or "").strip()
        self._record(content, time.perf_counter() - t0)
        return msg

    # -- offline mock LLM ------------------------------------------------------
    def _mock_chat(self, model, messages, tools=None, api_key=None, timeout=60.0):
        """Scripted stand-in for the LLM so the harness runs with no API key.

        Mirrors the receptionist flow: verify insurance for a known fixture
        patient, offer a slot, book on confirmation. Not a quality evaluation
        of the LLM — it exercises the harness, scoring, and latency probes.
        """
        t0 = time.perf_counter()
        time.sleep(0.01)  # simulate network+generation so percentiles are non-trivial

        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = (user_msgs[-1].get("content") or "").lower() if user_msgs else ""
        called = {ev["tool"] for ev in self.audit.tool_events}

        known = {
            "maya okafor": ("Maya Okafor", "1988-03-14", "ABC Dental"),
            "liam barnes": ("Liam Barnes", "1992-07-02", "Delta Prime"),
            "sofia reyes": ("Sofia Reyes", "1995-11-27", "Guardian Shield"),
        }
        content = ""
        summary = self._last_summary
        if "verify_insurance" not in called:
            for key, (name, dob, payer) in known.items():
                if key in last_user:
                    result = core.execute_tool(
                        "verify_insurance",
                        {"name": name, "dob": dob, "payer": payer},
                        self.conn,
                    )
                    summary = result[8:] if result.startswith("ERROR ") else json.loads(result).get("summary", "")
                    self._last_summary = summary
                    content = f"Thanks, {name}. I checked your benefits: {summary}."
                    break
            else:
                content = "Could I get your full name and date of birth please?"
        elif "yes" in last_user or "book" in last_user:
            if "create_appointment" not in called and self._last_verified:
                name, dob = self._last_verified
                slots = json.loads(
                    core.execute_tool(
                        "get_open_slots",
                        {"provider_id": 1, "appointment_type": "30 min checkup"},
                        self.conn,
                    )
                )
                if slots.get("slots"):
                    s = slots["slots"][0]
                    booked = core.execute_tool(
                        "create_appointment",
                        {
                            "slot_id": s["slot_id"],
                            "patient_name": name,
                            "dob": dob,
                            "phone_number": "5035550142",
                            "appointment_type": "30 min checkup",
                        },
                        self.conn,
                    )
                    if not booked.startswith("ERROR"):
                        content = f"Booked! Your benefits: {summary}"
                    else:
                        content = f"Sorry, booking failed: {booked[6:]}"
                else:
                    content = "I couldn't find any open slots."
            else:
                content = "You're all set. Anything else?"
        else:
            content = "What day works for you? We have openings this week and next."

        self._record(content, time.perf_counter() - t0)
        return {"content": content}

    # -- context manager --------------------------------------------------------
    def __enter__(self) -> "Instrumented":
        self._patches = [
            patch.object(core, "execute_tool", self._exec),
            patch.object(core, "groq_chat", self._chat),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc) -> None:
        for p in self._patches:
            p.stop()


# ---------------------------------------------------------------------------
# Fixture expectations for scoring quoted benefits
# ---------------------------------------------------------------------------

EXPECTED_PLAN = {
    "Maya Okafor": {"deductible": 0, "coverage_percent": 100, "waits": False},
    "Liam Barnes": {"deductible": 50, "coverage_percent": 80, "waits": False},
    "Sofia Reyes": {"deductible": 0, "coverage_percent": 100, "waits": True},
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    label: str
    booked: bool
    verified_before_quote: bool
    quoted_benefit_matches_fixture: bool
    p50: float
    p95: float
    notes: list[str]


def score_dialogue(audit: DialogueAudit, expected_book: bool = True) -> Score:
    notes: list[str] = []

    # 1. booking accuracy
    booked = audit.booked_ok()
    if expected_book and not booked:
        notes.append("expected booking but create_appointment never succeeded")
    if not expected_book and booked:
        notes.append("booking happened although scenario expected none")

    # 2. verification before quote
    q_turn = audit.first_quote_turn()
    v_turn = audit.first_verify_tool_turn()
    verified_before_quote = v_turn is not None and q_turn is not None and v_turn <= q_turn
    if q_turn is None:
        notes.append("no price quoted in dialogue")
    elif v_turn is None:
        notes.append("price quoted but verify_insurance never ran")

    # 3. quoted benefit matches fixture
    matches = False
    patient = audit.verified_patient()
    if patient and q_turn is not None:
        plan = EXPECTED_PLAN.get(patient)
        quotes = [p for (_, p) in audit.quoted_prices]
        all_text = " ".join(t.reply.lower() for t in audit.turns)
        if plan is None:
            notes.append(f"no fixture expectation for patient {patient!r}")
        elif plan["waits"] and audit.waiting_period_surfaced():
            # Sofia: correct if the waiting period was surfaced and either the
            # coverage numbers are right or the wait was called out.
            matches = (
                "wait" in all_text
                and any(p == plan["deductible"] for p in quotes)
            ) or "wait" in all_text
        elif plan["deductible"] == 0 and plan["coverage_percent"] == 100:
            matches = any(p == 0 for p in quotes) or "fully covered" in all_text
        else:
            matches = (
                any(p == plan["deductible"] for p in quotes)
                or str(plan["coverage_percent"]) in all_text
            )
    elif q_turn is None and v_turn is None:
        matches = True  # nothing quoted, nothing verified: vacuous
    else:
        notes.append("quote/verify present but patient attribution failed")

    lat = [t.llm_latency for t in audit.turns if t.llm_latency is not None]
    return Score(
        label=audit.scenario,
        booked=booked,
        verified_before_quote=verified_before_quote,
        quoted_benefit_matches_fixture=matches,
        p50=percentile(lat, 0.50) if lat else float("nan"),
        p95=percentile(lat, 0.95) if lat else float("nan"),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# The three documented template fixtures from insurance.py. Injected into the
# eval process ONLY in --offline mode when the user hasn't hand-written real
# ones yet, so the harness selftest exercises the full happy path.
TEMPLATE_FIXTURES = {
    "fully_covered": {
        "payer": {"id": "PAYER-ABC", "name": "ABC Dental"},
        "subscriber": {"name": "Maya Okafor", "member_id": "ABC-100234", "dob": "1988-03-14"},
        "eligibility": {
            "status": "active",
            "plan": {"deductible": 0, "coverage_percent": 100, "waiting_periods": []},
        },
    },
    "deductible_80": {
        "payer": {"id": "PAYER-DP", "name": "Delta Prime"},
        "subscriber": {"name": "Liam Barnes", "member_id": "DP-884213", "dob": "1992-07-02"},
        "eligibility": {
            "status": "active",
            "plan": {"deductible": 50, "coverage_percent": 80, "waiting_periods": []},
        },
    },
    "filling_wait": {
        "payer": {"id": "PAYER-GS", "name": "Guardian Shield"},
        "subscriber": {"name": "Sofia Reyes", "member_id": "GS-550091", "dob": "1995-11-27"},
        "eligibility": {
            "status": "active",
            "plan": {
                "deductible": 0,
                "coverage_percent": 100,
                "waiting_periods": [
                    {"service_type": "filling", "code": "203", "months": 6, "start_date": "2027-01-01"}
                ],
            },
        },
    },
}


def _ensure_offline_fixtures() -> None:
    """Offline-only: use template fixtures if the real ones aren't written yet."""
    import insurance as ins

    if not ins.FIXTURES and not (ins._fixtures_dir().exists() and any(ins._fixtures_dir().glob("*.json"))):
        ins.FIXTURES.update(TEMPLATE_FIXTURES)


def run_dialogue(
    scenario: Scenario,
    conn: sqlite3.Connection,
    offline: bool,
    model: str,
) -> tuple[DialogueAudit, Score]:
    inst = Instrumented(conn)
    inst.audit.scenario = scenario.label
    # Convention: a label containing "(no book" marks scenarios where booking
    # should NOT happen (e.g. caller just asks a question and hangs up).
    expected_book = "(no book" not in scenario.label.lower()

    with inst:
        messages: list[dict[str, Any]] = []
        for i, utterance in enumerate(scenario.utterances):
            inst._turn_idx = i
            inst.audit.turns.append(Turn(utterance=utterance))
            if offline:
                inst._mock_chat(model, [{"role": "user", "content": utterance}])
            else:
                core.agent_reply(utterance, messages, conn, model=model)

        score = score_dialogue(inst.audit, expected_book=expected_book)
    return inst.audit, score


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> None:
    parser = argparse.ArgumentParser(description="LatientDesk.ai evaluation harness")
    parser.add_argument("--scenarios", default="eval/scenarios.txt")
    parser.add_argument("--model", default=core.DEFAULT_MODEL)
    parser.add_argument("--offline", action="store_true", help="Mock LLM; no API key needed")
    parser.add_argument("--json", action="store_true", help="Also emit machine-readable JSON")
    args = parser.parse_args()

    scenarios = parse_scenarios(Path(args.scenarios))
    non_empty = [s for s in scenarios if s.utterances]
    print(f"{len(scenarios)} scenarios parsed, {len(non_empty)} with content")

    if not non_empty:
        print("No scenarios with utterances found. Fill the scenario file first.")
        return

    if args.offline:
        _ensure_offline_fixtures()
    elif not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Use --offline for the mock-LLM mode, or export the key.")
        return

    rows: list[Score] = []
    for s in non_empty:
        conn = pms.connect(":memory:")
        pms.seed(conn)
        _, score = run_dialogue(s, conn, args.offline, args.model)
        rows.append(score)
        conn.close()

    print(DASH)
    print(
        f"{'scenario':<28} {'booked':>6} {'verif<quote':>11} {'benefit ok':>10} "
        f"{'p50(s)':>7} {'p95(s)':>7}"
    )
    print(DASH)
    for r in rows:
        print(
            f"{r.label[:27]:<28} {str(r.booked):>6} {str(r.verified_before_quote):>11} "
            f"{str(r.quoted_benefit_matches_fixture):>10} {r.p50:>7.2f} {r.p95:>7.2f}"
        )
        for n in r.notes:
            print(f"{'':<28} note: {n}")
    print(DASH)

    p50s = [r.p50 for r in rows if r.p50 == r.p50]
    p95s = [r.p95 for r in rows if r.p95 == r.p95]
    overall_p50 = statistics.median(p50s) if p50s else float("nan")
    overall_p95 = max(p95s) if p95s else float("nan")
    print(
        f"summary: booked {sum(r.booked for r in rows)}/{len(rows)} | "
        f"verified-before-quote {sum(r.verified_before_quote for r in rows)}/{len(rows)} | "
        f"benefit-match {sum(r.quoted_benefit_matches_fixture for r in rows)}/{len(rows)} | "
        f"turn latency p50={overall_p50:.2f}s p95={overall_p95:.2f}s"
    )

    if args.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2, default=str))


if __name__ == "__main__":
    main()
