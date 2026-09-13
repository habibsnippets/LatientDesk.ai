"""Fake PMS for LatientDesk.ai.

A tiny stand-in for a real practice-management system (Open Dental style):
SQLite with three tables — patients, providers, appointments.

Appointment *types* are a fixed catalog of five (durations drive how many
30-minute grid slots a booking consumes). "Slots" are bookable openings on a
provider's calendar over the next two weeks. Booking a slot creates/reuses a
patient row and marks the slot (and any extra grid slots its duration needs)
as taken.

Usage:
    python pms.py        # reseeds demo.db, then runs a smoke-test demo
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).with_name("demo.db")

# Fixed catalog of appointment types: name -> duration in minutes.
APPOINTMENT_TYPES: dict[str, int] = {
    "30 min checkup": 30,
    "45 min cleaning": 45,
    "60 min filling": 60,
    "60 min extraction": 60,
    "90 min root canal": 90,
}

PROVIDERS = [
    {"provider_id": 1, "name": "Dr. Alice Chen", "specialty": "General Dentistry"},
    {"provider_id": 2, "name": "Dr. Marcus Webb", "specialty": "Orthodontics"},
]

# Office hours per weekday (0=Monday ... 6=Sunday), 24h clock.
OFFICE_HOURS = {
    0: (9, 17),  # Mon 09:00-17:00
    1: (9, 17),
    2: (9, 17),
    3: (9, 17),
    4: (9, 15),  # Fri short day
    5: None,     # Sat closed
    6: None,     # Sun closed
}

SLOT_MINUTES = 30   # slots are offered on a 30-minute grid
HORIZON_DAYS = 14   # ~two weeks of slots
TARGET_SLOTS = 200  # total open slots to seed across all providers

_rng_state = {"n": 987654321}


def _rand() -> float:
    """Deterministic LCG random in [0,1) so seeds are reproducible."""
    _rng_state["n"] = (1103515245 * _rng_state["n"] + 12345) % (2**31)
    return _rng_state["n"] / (2**31)


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit; explicit BEGIN below
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS providers (
            provider_id INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            specialty   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS patients (
            patient_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name     TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,           -- ISO date
            phone_number  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_id        TEXT UNIQUE NOT NULL,     -- e.g. '1-20260914-0900'
            provider_id    INTEGER NOT NULL REFERENCES providers(provider_id),
            patient_id     INTEGER REFERENCES patients(patient_id),
            appt_type      TEXT,                     -- NULL while slot is open
            start_time     TEXT NOT NULL,            -- ISO datetime
            end_time       TEXT NOT NULL,            -- ISO datetime
            status         TEXT NOT NULL DEFAULT 'open'  -- 'open' | 'booked'
        );

        CREATE INDEX IF NOT EXISTS idx_appts_start ON appointments(start_time);
        CREATE INDEX IF NOT EXISTS idx_appts_status ON appointments(status);
        """
    )


def _slot_id(provider_id: int, start: datetime) -> str:
    return f"{provider_id}-{start.strftime('%Y%m%d-%H%M')}"


def seed(conn: sqlite3.Connection, target_slots: int = TARGET_SLOTS) -> int:
    """Reset the database and seed providers + open slots for the next two weeks.

    Returns the number of slots created.
    """
    _rng_state["n"] = 987654321  # reset so seeds are reproducible
    conn.executescript(
        """
        DROP TABLE IF EXISTS appointments;
        DROP TABLE IF EXISTS patients;
        DROP TABLE IF EXISTS providers;
        """
    )
    init_schema(conn)

    conn.executemany(
        "INSERT INTO providers (provider_id, name, specialty) VALUES (:provider_id, :name, :specialty)",
        PROVIDERS,
    )

    # Slots start tomorrow morning so "today" is never half-booked.
    first_day = (datetime.now() + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    slot_rows: list[tuple[str, int, str, str]] = []  # (slot_id, provider_id, start, end)
    day_offset = 0
    while len(slot_rows) < target_slots and day_offset <= HORIZON_DAYS + 7:
        day = first_day + timedelta(days=day_offset)
        hours = OFFICE_HOURS.get(day.weekday())
        if hours is not None:
            open_h, close_h = hours
            for provider in PROVIDERS:
                t = day.replace(hour=open_h)
                while t < day.replace(hour=close_h) and len(slot_rows) < target_slots:
                    # Skip ~12% of grid positions to simulate buffers/lunch.
                    if _rand() < 0.12:
                        t += timedelta(minutes=SLOT_MINUTES)
                        continue
                    end = t + timedelta(minutes=SLOT_MINUTES)
                    slot_rows.append((_slot_id(provider["provider_id"], t), provider["provider_id"], t.isoformat(sep=" "), end.isoformat(sep=" ")))
                    t += timedelta(minutes=SLOT_MINUTES)
        day_offset += 1

    conn.executemany(
        """
        INSERT INTO appointments (slot_id, provider_id, patient_id, appt_type, start_time, end_time, status)
        VALUES (?, ?, NULL, NULL, ?, ?, 'open')
        """,
        slot_rows,
    )
    return len(slot_rows)


# ------------------------------------------------------------------ public API

def get_open_slots(
    provider_id: int,
    date_range: tuple[str, str] | None = None,
    appointment_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return open slots for a provider, soonest first.

    provider_id: 1 or 2.
    date_range: optional ("YYYY-MM-DD", "YYYY-MM-DD"), inclusive; None = all seeded slots.
    appointment_type: optional name from APPOINTMENT_TYPES. When given, only
        returns slots where enough *consecutive* open grid slots exist to fit
        the type's duration (e.g. "60 min filling" needs 2 consecutive blocks).
        Each returned dict then includes "duration_minutes" and "blocks_needed".

    Raises ValueError for an unknown provider_id or appointment_type.
    """
    if provider_id not in {p["provider_id"] for p in PROVIDERS}:
        raise ValueError(f"Unknown provider_id {provider_id!r}; valid: {[p['provider_id'] for p in PROVIDERS]}")
    if appointment_type is not None and appointment_type not in APPOINTMENT_TYPES:
        raise ValueError(
            f"Unknown appointment_type {appointment_type!r}; valid: {sorted(APPOINTMENT_TYPES)}"
        )

    conn = connect()
    try:
        params: list[Any] = [provider_id]
        where = "a.provider_id = ? AND a.status = 'open'"
        if date_range:
            start_d, end_d = date_range
            try:
                date.fromisoformat(start_d)
                date.fromisoformat(end_d)
            except ValueError as exc:
                raise ValueError("date_range must be ('YYYY-MM-DD', 'YYYY-MM-DD')") from exc
            where += " AND date(start_time) BETWEEN date(?) AND date(?)"
            params += [start_d, end_d]

        rows = conn.execute(
            f"""
            SELECT a.slot_id, a.provider_id, p.name AS provider_name,
                   a.start_time, a.end_time
            FROM appointments a
            JOIN providers p ON p.provider_id = a.provider_id
            WHERE {where}
            ORDER BY a.start_time
            """,
            params,
        ).fetchall()
        slots = [dict(r) for r in rows]

        if appointment_type is None:
            return slots

        duration = APPOINTMENT_TYPES[appointment_type]
        blocks = math.ceil(duration / SLOT_MINUTES)
        open_starts = {datetime.fromisoformat(s["start_time"]) for s in slots}
        result: list[dict[str, Any]] = []
        for s in slots:
            start = datetime.fromisoformat(s["start_time"])
            chain = [start + timedelta(minutes=SLOT_MINUTES * k) for k in range(blocks)]
            if all(t in open_starts for t in chain[1:]):
                enriched = dict(s)
                enriched["end_time"] = (start + timedelta(minutes=duration)).isoformat(sep=" ")
                enriched["duration_minutes"] = duration
                enriched["blocks_needed"] = blocks
                result.append(enriched)
        return result
    finally:
        conn.close()


def create_appointment(
    slot_id: str,
    patient_name: str,
    dob: str,
    phone_number: str,
    appointment_type: str | None = None,
) -> dict[str, Any]:
    """Book an open slot for a patient.

    slot_id: from get_open_slots().
    patient_name: full name, non-empty.
    dob: date of birth, "YYYY-MM-DD", must be in the past.
    phone_number: at least 7 digits (formatting is normalized to digits).
    appointment_type: optional name from APPOINTMENT_TYPES. If its duration
        exceeds one 30-minute grid slot, the following consecutive slots are
        booked too (a 45-min booking conservatively occupies 2 grid slots).

    Returns the booked appointment dict. Raises ValueError for invalid input
    or if the slot is not open (double-booking is rejected).
    """
    patient_name = (patient_name or "").strip()
    if not patient_name:
        raise ValueError("patient_name is required")

    try:
        dob_date = date.fromisoformat(dob)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid dob {dob!r}; expected 'YYYY-MM-DD'") from exc
    if dob_date >= date.today():
        raise ValueError("dob must be in the past")

    digits = "".join(c for c in (phone_number or "") if c.isdigit())
    if len(digits) < 7:
        raise ValueError(f"Invalid phone_number {phone_number!r}; need at least 7 digits")
    phone_number = digits

    if appointment_type is not None and appointment_type not in APPOINTMENT_TYPES:
        raise ValueError(
            f"Unknown appointment_type {appointment_type!r}; valid: {sorted(APPOINTMENT_TYPES)}"
        )

    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        slot = conn.execute(
            "SELECT appointment_id, provider_id, start_time, end_time, status FROM appointments WHERE slot_id = ?",
            (slot_id,),
        ).fetchone()
        if slot is None:
            raise ValueError(f"Unknown slot_id {slot_id!r}")
        if slot["status"] != "open":
            raise ValueError(f"Slot {slot_id!r} is not open (status={slot['status']!r})")

        start = datetime.fromisoformat(slot["start_time"])
        duration = APPOINTMENT_TYPES.get(appointment_type, SLOT_MINUTES) if appointment_type else SLOT_MINUTES
        blocks = math.ceil(duration / SLOT_MINUTES)

        # Find-or-create the patient (reuse record when name+dob match).
        patient = conn.execute(
            "SELECT patient_id, phone_number FROM patients WHERE lower(full_name) = lower(?) AND date_of_birth = ?",
            (patient_name, dob_date.isoformat()),
        ).fetchone()
        if patient:
            patient_id = patient["patient_id"]
            if patient["phone_number"] != phone_number:
                conn.execute(
                    "UPDATE patients SET phone_number = ? WHERE patient_id = ?",
                    (phone_number, patient_id),
                )
        else:
            cur = conn.execute(
                "INSERT INTO patients (full_name, date_of_birth, phone_number) VALUES (?, ?, ?)",
                (patient_name, dob_date.isoformat(), phone_number),
            )
            patient_id = cur.lastrowid

        # Book the anchor slot.
        end = start + timedelta(minutes=duration)
        conn.execute(
            """
            UPDATE appointments
            SET status = 'booked', patient_id = ?, appt_type = ?, end_time = ?
            WHERE appointment_id = ?
            """,
            (patient_id, appointment_type or "Unspecified", end.isoformat(sep=" "), slot["appointment_id"]),
        )

        # A multi-block booking also consumes the following grid slots.
        grid_ends = start + timedelta(minutes=SLOT_MINUTES * blocks)
        t = start + timedelta(minutes=SLOT_MINUTES)
        while t < grid_ends:
            nxt = conn.execute(
                """
                SELECT appointment_id, status FROM appointments
                WHERE provider_id = ? AND start_time = ?
                """,
                (slot["provider_id"], t.isoformat(sep=" ")),
            ).fetchone()
            if nxt is None:
                raise ValueError(
                    f"Cannot fit {duration} minutes: missing consecutive slot at {t.isoformat(sep=' ')}"
                )
            if nxt["status"] != "open":
                raise ValueError(
                    f"Cannot fit {duration} minutes: slot at {t.isoformat(sep=' ')} is already taken"
                )
            conn.execute(
                "UPDATE appointments SET status = 'booked', patient_id = ?, appt_type = ? WHERE appointment_id = ?",
                (patient_id, appointment_type or "Unspecified", nxt["appointment_id"]),
            )
            t += timedelta(minutes=SLOT_MINUTES)

        appointment = dict(
            conn.execute(
                "SELECT appointment_id, slot_id, provider_id, patient_id, appt_type, start_time, end_time, status FROM appointments WHERE appointment_id = ?",
                (slot["appointment_id"],),
            ).fetchone()
        )
        conn.execute("COMMIT")
        return appointment
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# ----------------------------------------------------------------------- CLI

def _demo() -> None:
    conn = connect()
    n = seed(conn)
    conn.close()
    print(f"Seeded {n} open slots at {DB_PATH}")

    week_start = (date.today() + timedelta(days=1)).isoformat()
    week_end = (date.today() + timedelta(days=7)).isoformat()
    slots = get_open_slots(1, (week_start, week_end), "60 min filling")
    print(f"Dr. Chen, {week_start}..{week_end}, 60 min filling: {len(slots)} matching starts")
    if slots:
        print("  first:", slots[0])

    first = get_open_slots(1)[0]
    booked = create_appointment(
        first["slot_id"],
        "Jamie Rivera",
        "1991-04-12",
        "(503) 555-0142",
        "45 min cleaning",
    )
    print("Booked:", booked)

    try:
        create_appointment(first["slot_id"], "Someone Else", "1980-01-01", "5551234567")
    except ValueError as e:
        print("Double-book correctly rejected:", e)


if __name__ == "__main__":
    _demo()
