"""Insurance verification for LatientDesk.ai (mock layer).

In production, a 270/271 exchange (X12 005010X279) goes out to the payer and the
returned 271 EDI is parsed into the JSON shape below. In this mock, no EDI is
generated or parsed: `verify_insurance(name, dob, payer)` returns one of the
hand-written fixture responses (or raises for unknown payers). The JSON
abstraction is the same one a production system uses internally *after* it has
parsed the 271, so downstream code written against it survives the swap to a
real clearinghouse.

Overview of the public X12 example set (005010X279), as published at
https://x12.org/examples/005010x279 — the canonical demo trading partners:

  Payer (Information Source):   ABC Company, ID 842610001
  Provider (Info Receiver):     Bone and Joint Clinic, Service Provider Number 2000035
                                Individual physician: Marcus Jones, SPN 0202034
  Subscriber (the patient):     Robert B. Smith, Member ID 11122333301,
                                DOB 1943-05-19, Group/Policy 599119
  Dependent (the patient):      Mary Smith, DOB 1978-10-14, relationship: Child

The official example files are distributed as a Windows self-extractor:
https://x12.org/sites/default/files/examples_downloads/X279-Examples_0.exe
(a copy is kept at tests/x12_examples/X279-Examples.exe). It must be opened on
a Windows machine or via a compatible extractor; do not execute unknown
binaries on a dev box. The example transactions in the archive cover:
  Example 01 — Subscriber Who is Also the Patient (270 request and 271 response)
  Example 02 — Patient Who is the Dependent of a Subscriber (270 and 271)

The JSON fixtures below deliberately mirror the fields those examples exercise
(subscriber vs. dependent hierarchy, member IDs, EB benefit segments, DTP date
qualifiers, MSG notes) in a flattened, query-friendly form.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

TESTS_DIR = Path(__file__).resolve().parent / "tests"
X279_ARCHIVE = TESTS_DIR / "x12_examples" / "X279-Examples.exe"

# ---------------------------------------------------------------------------
# Payer registry (fictional)
# ---------------------------------------------------------------------------

PAYER_DB: dict[str, dict[str, str]] = {
    "ABC Dental": {
        "payer_id": "PAYER-ABC",
        "eligibility_payer_id": "842610001",  # mirrors the x12.org example payer
    },
    "Delta Prime": {
        "payer_id": "PAYER-DP",
        "eligibility_payer_id": "842610002",
    },
    "Guardian Shield": {
        "payer_id": "PAYER-GS",
        "eligibility_payer_id": "842610003",
    },
}


def list_payers() -> list[str]:
    """Names of payers known to the mock eligibility service."""
    return sorted(PAYER_DB)


# ---------------------------------------------------------------------------
# Hand-written 271 fixtures live here.
#
# == 271 FIXTURES (hand-written by the user — do not generate programmatically) ==
# == Format contract: top-level dict matching `validate_fixture` in this file. ==
# ---------------------------------------------------------------------------
#
# == 271 FIXTURE SLOT 1 — "fully covered patient" ==
# Key: "fully_covered". Expected scenario: active plan, $0 deductible, 100%
# coverage, no waiting periods. Matched by verify_insurance("Maya Okafor", ...).
# Template (replace the placeholder below with your handwritten dict):
#
# {
#     "payer": { "id": "PAYER-ABC", "name": "ABC Dental" },
#     "subscriber": {
#         "name": "Maya Okafor",
#         "member_id": "ABC-100234",
#         "dob": "1988-03-14"
#     },
#     "eligibility": {
#         "status": "active",
#         "plan": {
#             "deductible": 0,
#             "coverage_percent": 100,
#             "waiting_periods": []
#         }
#     },
#     "verification_id": "demo-fully-covered",
#     "verified_at": null
# }
#
# == Placeholder — put the handwritten fixture here ==
#
# == 271 FIXTURE SLOT 2 — "$50 deductible / 80% coverage" ==
# Key: "deductible_80". Expected scenario: deductible 50.00, 80% coinsurance.
# Matched by verify_insurance("Liam Barnes", ...).
# Template:
#
# {
#     "payer": { "id": "PAYER-DP", "name": "Delta Prime" },
#     "subscriber": {
#         "name": "Liam Barnes",
#         "member_id": "DP-884213",
#         "dob": "1992-07-02"
#     },
#     "eligibility": {
#         "status": "active",
#         "plan": {
#             "deductible": 50,
#             "coverage_percent": 80,
#             "waiting_periods": []
#         }
#     },
#     "verification_id": "demo-deductible-80",
#     "verified_at": null
# }
#
# == Placeholder — put the handwritten fixture here ==
#
# == 271 FIXTURE SLOT 3 — "six-month waiting period on fillings" ==
# Key: "filling_wait". Expected scenario: 6-month waiting period on service
# type "filling" (X12 Service Type Code 203) not yet satisfied; everything
# else covered. Matched by verify_insurance("Sofia Reyes", ...).
# Template:
#
# {
#     "payer": { "id": "PAYER-GS", "name": "Guardian Shield" },
#     "subscriber": {
#         "name": "Sofia Reyes",
#         "member_id": "GS-550091",
#         "dob": "1995-11-27"
#     },
#     "eligibility": {
#         "status": "active",
#         "plan": {
#             "deductible": 0,
#             "coverage_percent": 100,
#             "waiting_periods": [
#                 { "service_type": "filling", "code": "203", "months": 6,
#                   "start_date": "2027-01-01" }
#             ]
#         }
#     },
#     "verification_id": "demo-filling-wait",
#     "verified_at": null
# }
#
# ---------------------------------------------------------------------------

FIXTURES: dict[str, dict[str, Any]] = {
    "fully_covered": {
        "payer": {"id": "PAYER-ABC", "name": "ABC Dental"},
        "subscriber": {
            "name": "Maya Okafor",
            "member_id": "ABC-100234",
            "dob": "1988-03-14",
        },
        "eligibility": {
            "status": "active",
            "plan": {
                "deductible": 0,
                "coverage_percent": 100,
                "waiting_periods": [],
            },
        },
        "verification_id": "demo-fully-covered",
        "verified_at": None,
    },
    "deductible_80": {
        "payer": {"id": "PAYER-DP", "name": "Delta Prime"},
        "subscriber": {
            "name": "Liam Barnes",
            "member_id": "DP-884213",
            "dob": "1992-07-02",
        },
        "eligibility": {
            "status": "active",
            "plan": {
                "deductible": 50,
                "coverage_percent": 80,
                "waiting_periods": [],
                "notes": ["$50 deductible applies to basic and major services"],
            },
        },
        "verification_id": "demo-deductible-80",
        "verified_at": None,
    },
    "filling_wait": {
        "payer": {"id": "PAYER-GS", "name": "Guardian Shield"},
        "subscriber": {
            "name": "Sofia Reyes",
            "member_id": "GS-550091",
            "dob": "1995-11-27",
        },
        "eligibility": {
            "status": "active",
            "plan": {
                "deductible": 0,
                "coverage_percent": 100,
                "waiting_periods": [
                    {
                        "service_type": "filling",
                        "code": "203",
                        "months": 6,
                        "start_date": "2027-01-01",
                    }
                ],
            },
        },
        "verification_id": "demo-filling-wait",
        "verified_at": None,
    },
}


def _fixtures_dir() -> Path:
    return TESTS_DIR / "fixtures" / "insurance" / "handwritten"


def _load_fixtures() -> dict[str, dict[str, Any]]:
    """Merge hand-written fixtures with any staged JSON files in tests/fixtures.

    Staged files live in tests/fixtures/insurance/handwritten/<key>.json.
    Hand-written in-code fixtures take precedence.
    """
    merged: dict[str, dict[str, Any]] = {}

    dir_ = _fixtures_dir()
    if dir_.is_dir():
        for path in sorted(dir_.glob("*.json")):
            try:
                merged[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Fixture {path.name} is not valid JSON: {exc}") from exc

    merged.update({k: v for k, v in FIXTURES.items() if v})
    return merged


# ---------------------------------------------------------------------------
# Shape validation — mirrors what production applies right after parsing a
# raw 271, so malformed fixtures fail loudly instead of leaking downstream.
# ---------------------------------------------------------------------------

_REQUIRED_SUBSCRIBER_FIELDS = {"name", "member_id", "dob"}


def validate_fixture(fx: dict[str, Any]) -> None:
    """Raise ValueError if `fx` doesn't match the internal 271 JSON contract."""
    if not isinstance(fx, dict):
        raise ValueError("fixture must be a dict")

    payer = fx.get("payer") or {}
    if not isinstance(payer, dict) or not payer.get("id") or not payer.get("name"):
        raise ValueError("payer.id and payer.name are required")

    subscriber = fx.get("subscriber") or {}
    missing = _REQUIRED_SUBSCRIBER_FIELDS - set(subscriber)
    if missing:
        raise ValueError(f"subscriber is missing required fields: {sorted(missing)}")
    if not isinstance(subscriber.get("name"), str) or not subscriber["name"].strip():
        raise ValueError("subscriber.name must be a non-empty string")

    dob = subscriber.get("dob")
    if not isinstance(dob, str):
        raise ValueError("subscriber.dob must be a 'YYYY-MM-DD' string")
    try:
        date.fromisoformat(dob)
    except ValueError as exc:
        raise ValueError(f"subscriber.dob {dob!r} is not a valid date") from exc

    eligibility = fx.get("eligibility") or {}
    if not isinstance(eligibility, dict):
        raise ValueError("eligibility must be a dict")
    if eligibility.get("status") not in {"active", "inactive", "pending"}:
        raise ValueError("eligibility.status must be one of: active, inactive, pending")

    plan = eligibility.get("plan") or {}
    if not isinstance(plan, dict):
        raise ValueError("eligibility.plan must be a dict")

    deductible = plan.get("deductible")
    if not isinstance(deductible, (int, float)) or deductible < 0:
        raise ValueError("plan.deductible must be a number >= 0")

    coverage = plan.get("coverage_percent")
    if not isinstance(coverage, (int, float)) or not 0 <= coverage <= 100:
        raise ValueError("plan.coverage_percent must be a number between 0 and 100")

    waits = plan.get("waiting_periods", [])
    if not isinstance(waits, list):
        raise ValueError("plan.waiting_periods must be a list")
    for wp in waits:
        if not isinstance(wp, dict):
            raise ValueError("each waiting_period must be a dict")
        if "service_type" not in wp:
            raise ValueError("waiting_periods[].service_type is required")
        start = wp.get("start_date")
        if start is not None:
            try:
                date.fromisoformat(start)
            except ValueError as exc:
                raise ValueError(
                    f"waiting_periods[].start_date {start!r} is not a valid date"
                ) from exc


# ---------------------------------------------------------------------------
# verify_insurance
# ---------------------------------------------------------------------------

def verify_insurance(
    name: str,
    dob: str,
    payer: str,
    service_date: str | None = None,
) -> dict[str, Any]:
    """Return the internal JSON view of a 271 eligibility response.

    Args:
        name: patient full name as registered with the payer.
        dob: patient date of birth, "YYYY-MM-DD".
        payer: one of `list_payers()` (e.g. "ABC Dental").
        service_date: optional "YYYY-MM-DD" used to evaluate waiting periods;
            defaults to today.

    Returns: the fixture dict augmented with `verification_id`,
    `verified_at`, and a `response` object echoing the lookup identity.

    Raises:
        ValueError: unknown payer, malformed DOB, patient not found at that
            payer, or (for fixtures loaded from disk) an invalid fixture.
    """
    if payer not in PAYER_DB:
        raise ValueError(f"Unknown payer {payer!r}; valid payers: {list_payers()}")

    try:
        dob_date = date.fromisoformat(dob)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid dob {dob!r}; expected 'YYYY-MM-DD'") from exc

    fixtures = _load_fixtures()

    # Match is case-insensitive on the subscriber's last name token; the mock
    # assumes first names are unique per payer in the fixture set.
    wanted_last = name.strip().split()[-1].lower() if name.strip() else ""
    for key, fx in fixtures.items():
        sub = fx.get("subscriber", {})
        fixture_last = str(sub.get("name", "")).strip().split()[-1].lower()
        fixture_dob = sub.get("dob", "")
        if fixture_last == wanted_last and fixture_dob == dob_date.isoformat():
            validate_fixture(fx)

            # Waiting-period evaluation against the service date.
            service_day = (
                date.fromisoformat(service_date)
                if service_date
                else date.today()
            )
            active_waits: list[dict[str, Any]] = []
            for wp in (fx.get("eligibility", {}).get("plan", {}).get("waiting_periods") or []):
                start = wp.get("start_date")
                if start and date.fromisoformat(start) > service_day:
                    active_waits.append(wp)

            response = dict(fx)
            response["response"] = {
                "service_date": service_day.isoformat(),
                "active_waiting_periods": active_waits,
            }
            return response

    raise ValueError(f"No fixture for patient {name!r} (dob {dob}) at payer {payer!r}")


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _demo() -> None:
    print("Known payers:", list_payers())
    try:
        result = verify_insurance("Maya Okafor", "1988-03-14", "ABC Dental")
        print(json.dumps(result, indent=2))
    except ValueError as exc:
        print("No fixtures populated yet (expected until hand-written):", exc)


if __name__ == "__main__":
    _demo()
