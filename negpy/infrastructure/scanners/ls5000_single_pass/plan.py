"""Load the frozen LS-5000 RGBI4x command plan with integrity checks."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any


CANONICAL_PLAN_FILENAME = "replay-first-rgbi4-plan.jsonl"
CANONICAL_PLAN_SHA256 = "f44223965db6e3a3ece7806c76cd8653ef4f2695c0841353b7107240e3861d16"
CANONICAL_PLAN_LINE_COUNT = 607
CANONICAL_FINE_READ_CDB = "280000000001032c0080"
CANONICAL_FINE_READ_BYTES = 207_872
CANONICAL_FINE_READ_COUNT = 2_980


class CanonicalPlanError(ValueError):
    """The bundled wire plan is missing, corrupted, or structurally wrong."""


def canonical_plan_bytes() -> bytes:
    """Read the plan from package resources and verify its exact provenance hash."""

    asset = resources.files(__package__).joinpath("data", CANONICAL_PLAN_FILENAME)
    payload = asset.read_bytes()
    verify_canonical_plan(payload)
    return payload


def verify_canonical_plan(payload: bytes) -> list[dict[str, Any]]:
    """Fail closed unless *payload* is the exact proven plan and expected schema."""

    payload = bytes(payload)
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != CANONICAL_PLAN_SHA256:
        raise CanonicalPlanError(f"canonical plan sha256 mismatch: expected {CANONICAL_PLAN_SHA256}, got {actual_hash}")

    lines = payload.splitlines()
    if len(lines) != CANONICAL_PLAN_LINE_COUNT:
        raise CanonicalPlanError(f"canonical plan has {len(lines)} entries; expected {CANONICAL_PLAN_LINE_COUNT}")
    try:
        entries = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalPlanError("canonical plan is not valid UTF-8 JSONL") from exc
    if not all(isinstance(entry, dict) for entry in entries):
        raise CanonicalPlanError("every canonical plan entry must be a JSON object")
    if [entry.get("seq") for entry in entries] != list(range(1, CANONICAL_PLAN_LINE_COUNT + 1)):
        raise CanonicalPlanError("canonical plan sequence numbers are not contiguous 1..607")

    inquiry = entries[0]
    if (
        inquiry.get("name") != "INQUIRY"
        or inquiry.get("cdb") != "120000002480"
        or inquiry.get("expected_data_in") != "068002021f0000004e696b6f6e2020204c532d35303030204544202020202020312e3033"
    ):
        raise CanonicalPlanError("canonical plan scanner identity preamble is wrong")

    fine = entries[-1]
    required = {
        "name": "READ",
        "role": "fine-rgbi4-template",
        "cdb": CANONICAL_FINE_READ_CDB,
        "request_len": CANONICAL_FINE_READ_BYTES,
        "repeat": CANONICAL_FINE_READ_COUNT,
        "capture": True,
    }
    if any(fine.get(key) != value for key, value in required.items()):
        raise CanonicalPlanError("canonical plan fine-read template is wrong")
    return entries


def load_canonical_plan() -> list[dict[str, Any]]:
    """Return a fresh mutable plan after exact hash and schema validation."""

    return verify_canonical_plan(canonical_plan_bytes())
