"""Load the pcap-derived Nikon RGBI4x later-frame continuation recipe.

The first-frame replay plan owns device discovery, reservation, roll indexing,
and the shared frame table.  This smaller recipe records only what Nikon Scan
repeated for each later selected frame while the same reservation remained
active.  The live worker compiles these validated semantics against its pinned
command templates; this module itself deliberately exposes no USB execution
path.

The compact semantic-step codes in the asset are: ``R`` ready, ``A``
autofocus, ``X`` execute, ``F`` focus readback, ``Q`` vendor 0x93 read,
``S``/``G`` set/get window, ``N`` scan, ``V`` vendor 0x87 sense read, and
``M`` meter-data read.  ``$Y``, ``$FOCUS``, and ``$EXPOSURE`` mark the fields
that legitimately vary by selected frame.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, cast


CANONICAL_CONTINUATION_PLAN_FILENAME = "replay-next-rgbi4-plan.json"
CANONICAL_CONTINUATION_PLAN_SHA256 = (
    "f0153cfc644d1b9949f3f9a8045676b34398b95135f1e963485aa5da5ea739c2"
)
CANONICAL_CONTINUATION_SEMANTIC_SHA256 = (
    "16be52b4b5fbf377f698d2fdcbf1b261e5ec3c1d7681ce4e09d4c36f1336ca35"
)


class CanonicalContinuationPlanError(ValueError):
    """The bundled later-frame recipe failed provenance or schema checks."""


@dataclass(frozen=True)
class ContinuationTraceBlock:
    """Immutable control-plane evidence extracted from one Nikon frame."""

    command_range: tuple[int, int]
    raw_command_count: int
    ready_groups: tuple[tuple[tuple[str, int], ...], ...]
    autofocus_y: int
    focus_readback: int
    window_y: int
    semantic_sha256: str

    @property
    def ready_poll_count(self) -> int:
        """Number of raw TEST UNIT READY commands in this trace block."""

        return sum(
            count
            for group in self.ready_groups
            for _sense, count in group
        )

    @property
    def semantic_step_count(self) -> int:
        """Command count after each variable-length ready poll becomes one step."""

        return self.raw_command_count - self.ready_poll_count + len(
            self.ready_groups
        )

    @property
    def autofocus_window_delta(self) -> int:
        """Nikon's autofocus point relative to the selected frame origin."""

        return self.autofocus_y - self.window_y


def _ready_groups(value: object) -> tuple[tuple[tuple[str, int], ...], ...]:
    if not isinstance(value, list) or not value:
        raise CanonicalContinuationPlanError(
            "continuation evidence must contain ready-poll groups"
        )
    parsed: list[tuple[tuple[str, int], ...]] = []
    for raw_group in value:
        if not isinstance(raw_group, list) or not raw_group:
            raise CanonicalContinuationPlanError(
                "continuation evidence contains an empty ready-poll group"
            )
        group: list[tuple[str, int]] = []
        for raw_run in raw_group:
            if (
                not isinstance(raw_run, list)
                or len(raw_run) != 2
                or not isinstance(raw_run[0], str)
                or raw_run[0] not in ("020401", "000000")
                or isinstance(raw_run[1], bool)
                or not isinstance(raw_run[1], int)
                or raw_run[1] < 1
            ):
                raise CanonicalContinuationPlanError(
                    "continuation evidence has an invalid ready-poll run"
                )
            sense = raw_run[0]
            count = cast(int, raw_run[1])
            group.append((sense, count))
        if group[-1][0] != "000000":
            raise CanonicalContinuationPlanError(
                "a continuation ready-poll group does not end ready"
            )
        parsed.append(tuple(group))
    return tuple(parsed)


def _required_integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalContinuationPlanError(
            f"continuation evidence {key} must be an integer"
        )
    return value


def derive_equivalent_continuation_blocks(
    plan: dict[str, Any],
) -> tuple[ContinuationTraceBlock, ...]:
    """Validate and reduce both Nikon later-frame blocks to one semantics.

    Busy-poll counts, frame origin, focus result, exposure, and meter pixels
    are observations, not control-flow steps.  The pinned semantic hash was
    calculated from the remaining ordered command geometry and sense chains.
    """

    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary")
    evidence = plan.get("trace_equivalence")
    if not isinstance(evidence, dict):
        raise CanonicalContinuationPlanError(
            "canonical continuation plan has no trace-equivalence evidence"
        )
    normalization = evidence.get("normalization")
    expected_normalization = {
        "dynamic_fields": [
            "autofocus_y",
            "focus_readback",
            "window_y",
            "exposure",
            "meter_pixels",
        ],
        "ready": (
            "collapse each consecutive TEST_UNIT_READY group after terminal "
            "000000"
        ),
    }
    if normalization != expected_normalization:
        raise CanonicalContinuationPlanError(
            "continuation evidence has the wrong normalization contract"
        )
    declared_steps = evidence.get("semantic_step_count")
    declared_digest = evidence.get("semantic_sha256")
    if declared_steps != 89:
        raise CanonicalContinuationPlanError(
            "continuation evidence does not reduce to 89 semantic steps"
        )
    if declared_digest != CANONICAL_CONTINUATION_SEMANTIC_SHA256:
        raise CanonicalContinuationPlanError(
            "continuation evidence has the wrong semantic digest"
        )
    semantic_steps = evidence.get("semantic_steps")
    if not isinstance(semantic_steps, list) or len(semantic_steps) != declared_steps:
        raise CanonicalContinuationPlanError(
            "continuation evidence has the wrong semantic step list"
        )
    encoded_steps = json.dumps(
        semantic_steps,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(encoded_steps).hexdigest() != declared_digest:
        raise CanonicalContinuationPlanError(
            "continuation semantic steps do not match their declared digest"
        )
    raw_blocks = evidence.get("blocks")
    if not isinstance(raw_blocks, list) or len(raw_blocks) != 2:
        raise CanonicalContinuationPlanError(
            "continuation evidence must contain exactly two trace blocks"
        )

    blocks: list[ContinuationTraceBlock] = []
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            raise CanonicalContinuationPlanError(
                "continuation trace block must be a JSON object"
            )
        command_range = raw.get("command_range")
        if (
            not isinstance(command_range, list)
            or len(command_range) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in command_range
            )
        ):
            raise CanonicalContinuationPlanError(
                "continuation trace block has an invalid command range"
            )
        block = ContinuationTraceBlock(
            command_range=(command_range[0], command_range[1]),
            raw_command_count=_required_integer(raw, "raw_command_count"),
            ready_groups=_ready_groups(raw.get("ready_groups")),
            autofocus_y=_required_integer(raw, "autofocus_y"),
            focus_readback=_required_integer(raw, "focus_readback"),
            window_y=_required_integer(raw, "window_y"),
            semantic_sha256=str(raw.get("semantic_sha256", "")),
        )
        start, stop = block.command_range
        if block.raw_command_count != stop - start + 1:
            raise CanonicalContinuationPlanError(
                "continuation trace command count disagrees with its range"
            )
        if block.semantic_step_count != declared_steps:
            raise CanonicalContinuationPlanError(
                "continuation trace does not reduce to the declared step count"
            )
        if block.autofocus_window_delta != 2979:
            raise CanonicalContinuationPlanError(
                "continuation autofocus is not frame origin plus 2979 rows"
            )
        if block.semantic_sha256 != declared_digest:
            raise CanonicalContinuationPlanError(
                "continuation trace does not match the declared semantics"
            )
        blocks.append(block)

    source = plan.get("source")
    source_ranges = source.get("command_blocks") if isinstance(source, dict) else None
    if source_ranges != [list(block.command_range) for block in blocks]:
        raise CanonicalContinuationPlanError(
            "continuation source ranges and equivalence evidence disagree"
        )
    if blocks[0].ready_groups == blocks[1].ready_groups:
        raise CanonicalContinuationPlanError(
            "continuation evidence does not preserve observed poll variation"
        )
    return tuple(blocks)


def canonical_continuation_plan_bytes() -> bytes:
    """Return the exact package asset after hash and schema verification."""

    asset = resources.files(__package__).joinpath(
        "data", CANONICAL_CONTINUATION_PLAN_FILENAME
    )
    payload = asset.read_bytes()
    verify_canonical_continuation_plan(payload)
    return payload


def verify_canonical_continuation_plan(payload: bytes) -> dict[str, Any]:
    """Fail closed unless *payload* is the pinned two-block Nikon recipe."""

    payload = bytes(payload)
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != CANONICAL_CONTINUATION_PLAN_SHA256:
        raise CanonicalContinuationPlanError(
            "canonical continuation plan sha256 mismatch: "
            f"expected {CANONICAL_CONTINUATION_PLAN_SHA256}, got {actual_hash}"
        )
    try:
        plan = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalContinuationPlanError(
            "canonical continuation plan is not valid UTF-8 JSON"
        ) from error
    if not isinstance(plan, dict):
        raise CanonicalContinuationPlanError(
            "canonical continuation plan must be a JSON object"
        )
    expected_source = {
        "capture": "oracle-ice-on-1.pcapng",
        "capture_sha256": (
            "5e3a890fa7c61f4f6c9cf597f55c0e7ea71628818d449aa9c53760fb017d07be"
        ),
        "command_blocks": [[3581, 3719], [6700, 6849]],
        "usb_bus": 1,
        "usb_device": 2,
    }
    if plan.get("schema_version") != 1 or plan.get("source") != expected_source:
        raise CanonicalContinuationPlanError(
            "canonical continuation plan has the wrong source provenance"
        )
    contract = plan.get("session_contract")
    if not isinstance(contract, dict):
        raise CanonicalContinuationPlanError(
            "canonical continuation plan has no session contract"
        )
    if contract.get("requires_existing_reservation") is not True:
        raise CanonicalContinuationPlanError(
            "later-frame recipe must require an existing reservation"
        )
    if contract.get("requires_existing_frame_table") is not True:
        raise CanonicalContinuationPlanError(
            "later-frame recipe must require an existing frame table"
        )
    expected_forbidden = [
        "RESERVE_UNIT",
        "SEND:sub_008f",
        "RELEASE_UNIT",
        "EJECT",
    ]
    if contract.get("forbidden_commands") != expected_forbidden:
        raise CanonicalContinuationPlanError(
            "later-frame recipe has the wrong forbidden-command contract"
        )
    if plan.get("autofocus") != {
        "x": 1973,
        "y": "resolved_native_origin+2979",
    }:
        raise CanonicalContinuationPlanError(
            "canonical continuation autofocus geometry is wrong"
        )
    metering = plan.get("metering")
    if metering != {"colors": [9, 1, 2, 3], "passes": 3, "resolution": 285}:
        raise CanonicalContinuationPlanError(
            "canonical continuation metering recipe is wrong"
        )
    fine = plan.get("fine")
    required_fine = {
        "colors": [9, 1, 2, 3],
        "read_cdb": "280000000001032c0080",
        "read_count": 2980,
        "resolution": 4000,
        "samples_per_scan": 4,
    }
    if fine != required_fine:
        raise CanonicalContinuationPlanError(
            "canonical continuation fine-scan recipe is wrong"
        )
    derive_equivalent_continuation_blocks(plan)
    return plan


def load_canonical_continuation_plan() -> dict[str, Any]:
    """Return a fresh recipe after exact provenance and schema validation."""

    return verify_canonical_continuation_plan(canonical_continuation_plan_bytes())


__all__ = [
    "CANONICAL_CONTINUATION_PLAN_FILENAME",
    "CANONICAL_CONTINUATION_PLAN_SHA256",
    "CANONICAL_CONTINUATION_SEMANTIC_SHA256",
    "CanonicalContinuationPlanError",
    "ContinuationTraceBlock",
    "canonical_continuation_plan_bytes",
    "derive_equivalent_continuation_blocks",
    "load_canonical_continuation_plan",
    "verify_canonical_continuation_plan",
]
