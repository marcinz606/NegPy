"""Offline, fail-closed audit of completed LS-5000 acceptance outputs.

The live acceptance runner deliberately performs only a cheap publication
check while the scanner is reserved.  This module composes the stronger
loaders used by the production write path after the scanner has been closed:
Digital ICE replay, retained native-builder re-derivation, portable CML4
re-evaluation, Hybrid disclosure binding, and exact TIFF/ICC verification.

Nothing in this module opens a scanner or imports a hardware backend.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, NoReturn, Protocol, Sequence, cast

import numpy as np

from negpy.infrastructure.roll.repair import RepairAcquisition, RepairMode, RepairResult
from negpy.services.repair.hybrid_runtime_manifest import (
    HybridRuntimeManifestError,
    load_default_hybrid_runtime_manifest,
)
from negpy.services.roll import exact_color, nikon_icc
from negpy.services.roll import service as roll_service
from negpy.services.roll.portable_builder import PortableStage1Builder
from negpy.services.roll.portable_cms import PortableCMSOnEvaluator
from negpy.services.roll.service import RollFrameOutput


class _FcntlAPI(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, descriptor: int, operation: int) -> None: ...


try:
    _fcntl_module = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - this audit targets the macOS workflow
    _fcntl: _FcntlAPI | None = None
else:
    _fcntl = cast(_FcntlAPI, _fcntl_module)


SCHEMA = "negpy.ls5000-deep-acceptance.v1"
SLOTS = (1, 2, 3, 4, 5, 6)
# Historical fixture/export compatibility only.  The production audit derives
# the actual approved slots from the sealed frame evidence.
APPROVED_SLOTS = frozenset((1, 6))
_OUTPUT_FIELDS = (
    "rgb_path",
    "ir_path",
    "repaired_rgb_path",
    "repaired_ir_path",
    "positive_path",
    "receipt_path",
    "synthesis_mask_path",
    "native_synthesis_mask_path",
    "hybrid_receipt_path",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024


class _HybridRuntime(Protocol):
    core_source_manifest_sha256: str
    hybrid_source_manifest_sha256: str
    iopaint_source_manifest_sha256: str
    model_weights_sha256: str
    inpaint_device: str
    inpaint_threads: int
    inpaint_seed: int

    def validate_files(self) -> None: ...


class DeepAcceptanceError(ValueError):
    """A completed output failed an offline acceptance invariant."""


@dataclass(frozen=True)
class _FrameAudit:
    summary: dict[str, Any]
    referenced_files: frozenset[Path]
    receipt_path: Path
    receipt: dict[str, Any]
    ownership: object
    builder_receipt: exact_color.NativeValidatedBuilderReceipt
    output_artifacts: dict[str, str]


def _fail(label: str, message: str) -> NoReturn:
    raise DeepAcceptanceError(f"{label}: {message}")


def _public_output_slot(
    output: object,
    *,
    expected_slot: int | None,
) -> tuple[RollFrameOutput, int]:
    if type(output) is not RollFrameOutput:
        raise DeepAcceptanceError("output must be one RollFrameOutput")
    slot = output.slot if expected_slot is None else expected_slot
    if type(slot) is not int:
        raise DeepAcceptanceError("expected slot must be an integer")
    return output, slot


def _validated_runtime(runtime: _HybridRuntime | None) -> _HybridRuntime:
    try:
        active = runtime if runtime is not None else load_default_hybrid_runtime_manifest()
    except (HybridRuntimeManifestError, OSError, TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"pinned Hybrid runtime manifest is invalid: {error}") from error
    if active is None:
        raise DeepAcceptanceError("the pinned Hybrid runtime is not installed")
    try:
        active.validate_files()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"pinned Hybrid runtime is invalid: {error}") from error
    return active


def _color_dependencies(
    builder: PortableStage1Builder | None,
    evaluator: PortableCMSOnEvaluator | None,
) -> tuple[PortableStage1Builder, PortableCMSOnEvaluator]:
    try:
        active_builder = builder if builder is not None else PortableStage1Builder()
        active_evaluator = evaluator if evaluator is not None else PortableCMSOnEvaluator()
    except (exact_color.ExactColorUnavailable, OSError, TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"portable exact-color dependencies are unavailable: {error}") from error
    return active_builder, active_evaluator


def _canonical_json(
    value: object,
    *,
    newline: bool = False,
    ensure_ascii: bool = False,
) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_dict(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(label, "must be an object")
    return cast(dict[str, Any], value)


def _require_str(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(label, "must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(label, "must be one lowercase SHA-256 digest")
    return value


def _output_root(path: str | os.PathLike[str]) -> Path:
    try:
        candidate = Path(path)
        linked = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"output directory is unavailable: {error}") from error
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
        raise DeepAcceptanceError("output directory must be a real directory")
    return resolved


def _regular_file(
    path_value: str | os.PathLike[str],
    *,
    root: Path,
    label: str,
) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        _fail(label, "path must be absolute")
    lexical = Path(os.path.abspath(os.fspath(path)))
    if not lexical.is_relative_to(root):
        _fail(label, "escaped the output directory")
    try:
        current = root
        for component in lexical.relative_to(root).parts[:-1]:
            current /= component
            parent_metadata = current.lstat()
            if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
                _fail(label, "has a symlink or non-directory parent")
        linked = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise DeepAcceptanceError(f"{label}: unavailable: {error}") from error
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
        _fail(label, "must be a regular non-symlink file")
    if resolved != lexical:
        _fail(label, "resolved through a symbolic link")
    return resolved


def _stable_bytes(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        return roll_service._stable_regular_bytes(  # noqa: SLF001 - production integrity boundary
            str(path),
            maximum_bytes=maximum_bytes,
            label=label,
        )
    except (OSError, ValueError) as error:
        raise DeepAcceptanceError(f"{label}: {error}") from error


def _load_frame_receipt(
    receipt_path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    payload = _stable_bytes(
        receipt_path,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        label=label,
    )
    try:
        document = roll_service._strict_json_loads(payload)  # noqa: SLF001
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise DeepAcceptanceError(f"{label}: invalid JSON: {error}") from error
    document = _require_dict(document, label=label)
    try:
        canonical = json.dumps(
            document,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise DeepAcceptanceError(f"{label}: cannot canonicalize JSON: {error}") from error
    if payload != canonical:
        _fail(label, "is not in the production canonical frame-receipt encoding")
    return document, payload


def _nested_receipt_value(
    document: object,
    *keys: str,
    label: str,
) -> object:
    current = document
    for key in keys:
        current = _require_dict(current, label=label).get(key)
    return current


def _receipt_artifact_path(
    document: object,
    *keys: str,
    root: Path,
    label: str,
) -> Path:
    value = _nested_receipt_value(document, *keys, label=label)
    if type(value) is not str or not Path(value).is_absolute():
        _fail(label, "must be an absolute artifact path")
    return _regular_file(value, root=root, label=label)


def _absolute_receipt_paths(value: object, *, key: str | None = None) -> set[str]:
    found: set[str] = set()
    if type(value) is dict:
        for child_key, child in cast(dict[str, object], value).items():
            found.update(_absolute_receipt_paths(child, key=child_key))
    elif type(value) is list:
        for child in value:
            found.update(_absolute_receipt_paths(child, key=key))
    elif (
        type(value) is str
        and key != "relative_path"
        and (key == "path" or (key is not None and key.endswith("_path")))
        and Path(value).is_absolute()
    ):
        found.add(value)
    return found


def _collect_completed_frame_files_locked(
    output: RollFrameOutput,
    *,
    root: Path,
    expected_slot: int,
) -> tuple[dict[str, Any], bytes, frozenset[Path]]:
    """Resolve one exact completed receipt while its frame lock is held."""

    label = f"slot {expected_slot}"
    if type(output) is not RollFrameOutput or output.slot != expected_slot:
        _fail(label, "RollFrameOutput identity is wrong")
    receipt_path = _regular_file(
        output.receipt_path,
        root=root,
        label=f"{label} frame receipt",
    )
    receipt, receipt_bytes = _load_frame_receipt(
        receipt_path,
        label=f"{label} frame receipt",
    )
    if (
        receipt.get("version") != 1
        or receipt.get("slot") != expected_slot
        or receipt.get("dpi") != 4_000
        or receipt.get("depth") != 16
        or type(receipt.get("device_id")) is not str
        or not receipt.get("device_id")
        or receipt.get("device_model") != "Nikon LS-5000 ED 1.03"
    ):
        _fail(label, "frame or LS-5000 capture identity is wrong")
    outputs = _require_dict(receipt.get("outputs"), label=f"{label} outputs")
    unrepaired = _require_dict(outputs.get("unrepaired"), label=f"{label} unrepaired tier")
    repaired = _require_dict(outputs.get("repaired"), label=f"{label} repaired tier")
    positive = _require_dict(outputs.get("positive"), label=f"{label} positive tier")
    dice = _require_dict(
        outputs.get("repair_acquisition_evidence"),
        label=f"{label} DICE evidence",
    )
    native = _require_dict(
        outputs.get("native_color_evidence"),
        label=f"{label} native color evidence",
    )
    if any(row.get("written") is not True for row in (unrepaired, repaired, positive)):
        _fail(label, "all three output tiers must be complete")
    if (
        repaired.get("mode_requested") != "hybrid"
        or repaired.get("mode_resolved") != "hybrid"
        or repaired.get("degraded") is not False
        or positive.get("color_mode") != "nikon-exact"
        or positive.get("exact_nikon_color") is not True
        or positive.get("native_per_acquisition_builder") is not True
        or positive.get("builder_validated") is not True
        or positive.get("cms_verified") is not True
        or dice.get("retained") is not True
        or dice.get("replayable") is not True
        or native.get("retained") is not True
        or native.get("native_per_acquisition_builder") is not True
    ):
        _fail(label, "completed Hybrid/Nikon-exact evidence flags changed")
    artifacts = _require_dict(receipt.get("artifacts"), label=f"{label} artifacts")
    if set(artifacts) != {"rgb", "ir"}:
        _fail(label, "frame artifact inventory is not exactly RGB and IR")

    required: list[Path] = [
        _receipt_artifact_path(
            outputs,
            "unrepaired",
            "rgb_path",
            root=root,
            label=f"{label} RGB",
        ),
        _receipt_artifact_path(
            outputs,
            "unrepaired",
            "ir_path",
            root=root,
            label=f"{label} IR",
        ),
        _receipt_artifact_path(
            outputs,
            "repaired",
            "rgb_path",
            root=root,
            label=f"{label} repaired RGB",
        ),
        _receipt_artifact_path(
            outputs,
            "repaired",
            "ir_path",
            root=root,
            label=f"{label} repaired IR",
        ),
        _receipt_artifact_path(
            outputs,
            "positive",
            "rgb_path",
            root=root,
            label=f"{label} positive",
        ),
        _receipt_artifact_path(
            outputs,
            "repair_acquisition_evidence",
            "binding",
            "path",
            root=root,
            label=f"{label} DICE binding",
        ),
    ]
    for key in ("prepass_rgbi", "ir_validity"):
        required.append(
            _receipt_artifact_path(
                outputs,
                "repair_acquisition_evidence",
                "artifacts",
                key,
                "path",
                root=root,
                label=f"{label} DICE {key}",
            )
        )
    for key in ("storage_rgb_tiff", "storage_ir_tiff"):
        required.append(
            _receipt_artifact_path(
                outputs,
                "repair_acquisition_evidence",
                "sources",
                key,
                "path",
                root=root,
                label=f"{label} DICE {key}",
            )
        )
    for branch, name in (
        (("applied_final", "storage"), "storage mask"),
        (("applied_final", "native"), "native mask"),
        (("routed_raw", "native"), "routed native mask"),
    ):
        required.append(
            _receipt_artifact_path(
                outputs,
                "repaired",
                "disclosure_mask",
                *branch,
                "path",
                root=root,
                label=f"{label} {name}",
            )
        )
    for key, name in (
        ("hybrid_receipt", "Hybrid receipt"),
        ("hybrid_evidence_binding", "Hybrid binding"),
    ):
        required.append(
            _receipt_artifact_path(
                outputs,
                "repaired",
                key,
                "path",
                root=root,
                label=f"{label} {name}",
            )
        )

    retained = _require_dict(
        native.get("retained_builder_evidence"),
        label=f"{label} retained native builder",
    )
    if positive.get("retained_builder_evidence") != retained:
        _fail(label, "positive and native tiers reference different builder evidence")
    for key in (
        "builder_receipt",
        "analyzer_rgb",
        "evidence_receipt",
        "frame_ownership_receipt",
        "density_evidence_receipt",
    ):
        required.append(
            _receipt_artifact_path(
                retained,
                key,
                "path",
                root=root,
                label=f"{label} native {key}",
            )
        )
    luts = retained.get("pre_f_luts")
    if type(luts) is not list or len(luts) != 3:
        _fail(label, "retained native builder must contain exactly three LUTs")
    for index, row in enumerate(luts):
        required.append(
            _receipt_artifact_path(
                row,
                "path",
                root=root,
                label=f"{label} native LUT {index}",
            )
        )

    required_set = set(required)
    if len(required_set) != 21:
        _fail(label, "receipt-bound artifact paths alias or are incomplete")
    recorded_absolute = _absolute_receipt_paths(outputs)
    if recorded_absolute != {str(path) for path in required_set}:
        missing = sorted(str(path) for path in required_set - {Path(item) for item in recorded_absolute})
        extra = sorted(recorded_absolute - {str(path) for path in required_set})
        raise DeepAcceptanceError(f"{label}: receipt path inventory changed (missing={missing}, extra={extra})")

    expected_output_paths = {
        "rgb_path": required[0],
        "ir_path": required[1],
        "repaired_rgb_path": required[2],
        "repaired_ir_path": required[3],
        "positive_path": required[4],
        "receipt_path": receipt_path,
        "synthesis_mask_path": required[10],
        "native_synthesis_mask_path": required[11],
        "hybrid_receipt_path": required[13],
    }
    if set(expected_output_paths) != set(_OUTPUT_FIELDS):
        raise AssertionError("internal RollFrameOutput field inventory changed")
    for field, path in expected_output_paths.items():
        if getattr(output, field) != str(path):
            _fail(label, f"RollFrameOutput {field} differs from its receipt")

    receipt_lock = _regular_file(
        _receipt_lock_path(receipt_path),
        root=root,
        label=f"{label} frame receipt lock",
    )
    return receipt, receipt_bytes, frozenset((*required_set, receipt_path, receipt_lock))


def collect_completed_frame_files(
    output: RollFrameOutput,
    *,
    output_dir: str | os.PathLike[str],
    expected_slot: int | None = None,
) -> list[str]:
    """Cheaply collect one completed frame's exact owned-file inventory.

    This hardware-inert checkpoint validates only the bounded frame receipt,
    path bindings, regular files, and persistent frame lock. It deliberately
    does not decode TIFF/PNG/NPY data, hash model or raster bytes, load the
    Hybrid runtime, or recompute color.
    """

    checked_output, slot = _public_output_slot(
        output,
        expected_slot=expected_slot,
    )
    root = _output_root(output_dir)
    with _hold_frame_locks((checked_output,), root=root):
        _, _, referenced = _collect_completed_frame_files_locked(
            checked_output,
            root=root,
            expected_slot=slot,
        )
        return sorted(str(path) for path in referenced)


def _artifact_row(
    row_value: object,
    *,
    root: Path,
    label: str,
    expected_path: Path | None = None,
    maximum_bytes: int = _MAX_EVIDENCE_BYTES,
    digest_key: str = "sha256",
) -> tuple[Path, bytes]:
    row = _require_dict(row_value, label=label)
    path_value = row.get("path")
    if type(path_value) is not str:
        _fail(label, "has no path")
    path = _regular_file(path_value, root=root, label=label)
    if expected_path is not None and path != expected_path:
        _fail(label, "path disagrees with the published RollFrameOutput")
    payload = _stable_bytes(path, maximum_bytes=maximum_bytes, label=label)
    if row.get("bytes") != len(payload):
        _fail(label, "byte count changed")
    digest = _require_sha256(row.get(digest_key), label=f"{label} SHA-256")
    if not hmac.compare_digest(digest, _sha256(payload)):
        _fail(label, "SHA-256 changed")
    return path, payload


def _published_path(
    output: RollFrameOutput,
    field: str,
    recorded: object,
    *,
    root: Path,
    label: str,
) -> Path:
    value = getattr(output, field)
    if type(value) is not str or type(recorded) is not str or value != recorded:
        _fail(label, "receipt and RollFrameOutput paths disagree")
    return _regular_file(value, root=root, label=label)


def _array_artifact_matches(row_value: object, array: np.ndarray, *, label: str) -> None:
    row = _require_dict(row_value, label=label)
    contiguous = np.ascontiguousarray(array)
    expected = {
        "byte_length": contiguous.nbytes,
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
        "shape": list(contiguous.shape),
    }
    if row != expected:
        _fail(label, "does not bind the decoded array")


def _decode_tiff(path: Path, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    try:
        return roll_service._stable_tiff_array(  # noqa: SLF001 - production integrity boundary
            str(path),
            expected_shape=shape,
            label=label,
        )
    except (OSError, ValueError) as error:
        raise DeepAcceptanceError(f"{label}: {error}") from error


def _bound_archive_path(
    row_value: object,
    *,
    binding_path: Path,
    root: Path,
    label: str,
) -> Path:
    row = _require_dict(row_value, label=label)
    relative = row.get("relative_path")
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        _fail(label, "has no safe relative path")
    return _regular_file(
        binding_path.parent / relative,
        root=root,
        label=label,
    )


def _validate_dice_archive_paths(
    dice: dict[str, Any],
    *,
    binding_path: Path,
    binding_bytes: bytes,
    root: Path,
    rgb_path: Path,
    ir_path: Path,
    referenced: set[Path],
    label: str,
) -> None:
    """Bind sidecar paths to the exact relative members consumed by replay."""

    try:
        binding = roll_service._strict_json_loads(binding_bytes)  # noqa: SLF001
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise DeepAcceptanceError(f"{label}: DICE binding JSON is invalid: {error}") from error
    binding = _require_dict(binding, label=f"{label} DICE binding")
    if binding_path.name != "acquisition-binding.json" or binding_path.parent.parent != root / ".negpy-dice-acquisition":
        _fail(label, "DICE evidence escaped its production archive directory")
    bound_acquisition = _require_dict(binding.get("acquisition"), label=f"{label} bound DICE acquisition")
    if (
        dice.get("acquisition_id") != bound_acquisition.get("acquisition_id")
        or dice.get("replay") != binding.get("replay")
        or dice.get("schema") != binding.get("schema")
    ):
        _fail(label, "DICE sidecar identity or replay contract changed")
    outer_artifacts = _require_dict(dice.get("artifacts"), label=f"{label} DICE artifacts")
    bound_artifacts = _require_dict(binding.get("artifacts"), label=f"{label} bound DICE artifacts")
    outer_sources = _require_dict(dice.get("sources"), label=f"{label} DICE sources")
    bound_sources = _require_dict(binding.get("sources"), label=f"{label} bound DICE sources")
    if set(outer_artifacts) != {"prepass_rgbi", "ir_validity"} or set(bound_artifacts) != {"prepass_rgbi", "ir_validity"}:
        _fail(label, "DICE artifact inventory changed")
    if set(outer_sources) != {"storage_rgb_tiff", "storage_ir_tiff"} or set(bound_sources) != {"storage_rgb_tiff", "storage_ir_tiff"}:
        _fail(label, "DICE source inventory changed")

    expected_artifact_names = {
        "prepass_rgbi": "prepass.rgbi16.npy",
        "ir_validity": "ir-validity.npy",
    }
    for key, expected_name in expected_artifact_names.items():
        outer = _require_dict(outer_artifacts.get(key), label=f"{label} DICE {key}")
        bound = _require_dict(bound_artifacts.get(key), label=f"{label} bound DICE {key}")
        if {name: value for name, value in outer.items() if name != "path"} != bound:
            _fail(label, f"DICE {key} sidecar differs from the replay binding")
        outer_path, _ = _artifact_row(
            outer,
            root=root,
            label=f"{label} DICE {key}",
            digest_key="file_sha256",
        )
        bound_path = _bound_archive_path(
            bound,
            binding_path=binding_path,
            root=root,
            label=f"{label} bound DICE {key}",
        )
        if outer_path != bound_path:
            _fail(label, f"DICE {key} path differs from the replay binding")
        if bound_path.parent != binding_path.parent or bound_path.name != expected_name:
            _fail(label, f"DICE {key} escaped its production archive directory")
        referenced.add(bound_path)

    for key, expected_path in (
        ("storage_rgb_tiff", rgb_path),
        ("storage_ir_tiff", ir_path),
    ):
        outer = _require_dict(outer_sources.get(key), label=f"{label} DICE {key}")
        bound = _require_dict(bound_sources.get(key), label=f"{label} bound DICE {key}")
        if {name: value for name, value in outer.items() if name != "path"} != bound:
            _fail(label, f"DICE {key} sidecar differs from the replay binding")
        recorded = outer.get("path")
        if type(recorded) is not str:
            _fail(label, f"DICE {key} has no path")
        outer_path = _regular_file(
            recorded,
            root=root,
            label=f"{label} DICE {key}",
        )
        bound_path = _bound_archive_path(
            bound,
            binding_path=binding_path,
            root=root,
            label=f"{label} bound DICE {key}",
        )
        if outer_path != expected_path or bound_path != expected_path:
            _fail(label, f"DICE {key} does not bind the published Tier-1 TIFF")


def _stored_receipt_hash(document: object) -> str:
    return _sha256(_canonical_json(document))


def _receipt_lock_path(receipt_path: Path) -> Path:
    """Return the persistent transaction-lock path for one frame receipt."""

    lock_name = hashlib.sha256(str(receipt_path).encode("utf-8")).hexdigest()
    return receipt_path.parent / ".negpy-locks" / f"{lock_name}.lock"


@contextmanager
def _hold_frame_locks(
    outputs: Sequence[RollFrameOutput],
    *,
    root: Path,
) -> Iterator[tuple[Path, ...]]:
    """Hold every production frame lock for the entire audit snapshot."""

    if _fcntl is None:
        raise DeepAcceptanceError("safe frame locking is unavailable")
    lock_paths: set[Path] = set()
    for index, output in enumerate(outputs, start=1):
        if type(output) is not RollFrameOutput or type(output.receipt_path) is not str:
            _fail(f"frame {index}", "RollFrameOutput identity is wrong")
        receipt_path = _regular_file(
            output.receipt_path,
            root=root,
            label=f"frame {index} receipt",
        )
        lock_paths.add(
            _regular_file(
                _receipt_lock_path(receipt_path),
                root=root,
                label=f"frame {index} receipt lock",
            )
        )

    held: list[tuple[Path, int, tuple[int, int]]] = []
    try:
        for path in sorted(lock_paths):
            before = path.lstat()
            flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                identity = (opened.st_dev, opened.st_ino)
                if not stat.S_ISREG(opened.st_mode) or identity != (before.st_dev, before.st_ino):
                    raise DeepAcceptanceError(f"frame lock changed while opening: {path}")
                try:
                    _fcntl.flock(descriptor, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise DeepAcceptanceError(f"frame output is busy in another process: {path}") from error
                visible = path.lstat()
                if identity != (visible.st_dev, visible.st_ino):
                    raise DeepAcceptanceError(f"frame lock changed while acquiring ownership: {path}")
            except BaseException:
                os.close(descriptor)
                raise
            held.append((path, descriptor, identity))
        yield tuple(path for path, _, _ in held)
        for path, _, identity in held:
            visible = path.lstat()
            if not stat.S_ISREG(visible.st_mode) or identity != (visible.st_dev, visible.st_ino):
                raise DeepAcceptanceError(f"frame lock changed during deep acceptance: {path}")
    except OSError as error:
        raise DeepAcceptanceError(f"cannot hold frame output locks: {error}") from error
    finally:
        for _, descriptor, _ in reversed(held):
            try:
                _fcntl.flock(descriptor, _fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _require_receipt_copy(
    document: object,
    digest: object,
    expected_document: dict[str, Any],
    expected_digest: str,
    *,
    label: str,
) -> None:
    if document != expected_document:
        _fail(label, "payload differs from a fresh production evaluation")
    if digest != expected_digest or _stored_receipt_hash(document) != expected_digest:
        _fail(label, "SHA-256 differs from a fresh production evaluation")


def _runtime_receipt_binding(
    receipt: dict[str, Any],
    *,
    acquisition: RepairAcquisition,
    repaired: dict[str, Any],
    runtime: _HybridRuntime,
    label: str,
) -> None:
    inputs = _require_dict(receipt.get("inputs"), label=f"{label} inputs")
    main = _require_dict(inputs.get("main"), label=f"{label} main input")
    prepass = _require_dict(inputs.get("prepass"), label=f"{label} prepass input")
    geometry = _require_dict(inputs.get("geometry"), label=f"{label} geometry")
    provenance = _require_dict(inputs.get("provenance"), label=f"{label} provenance")
    main_rgbi = np.asarray(getattr(acquisition, "main_rgbi"))
    prepass_rgbi = np.asarray(getattr(acquisition, "prepass_rgbi"))
    if main != {
        "canonical_encoding": "uint16_little_endian_c_order",
        "raw_sha256": getattr(acquisition, "main_rgbi_sha256"),
        "shape": list(main_rgbi.shape),
    }:
        _fail(label, "main input does not bind the retained acquisition")
    if prepass != {
        "canonical_encoding": "uint16_little_endian_c_order",
        "raw_sha256": getattr(acquisition, "prepass_rgbi_sha256"),
        "shape": list(prepass_rgbi.shape),
    }:
        _fail(label, "prepass input does not bind the retained acquisition")
    if geometry.get("mask_shape") != list(main_rgbi.shape[:2]) or geometry.get("output_shape") != [*main_rgbi.shape[:2], 3]:
        _fail(label, "geometry does not bind the retained acquisition")
    assertion = {
        "assertions": {
            "focus_exposure_locked": True,
            "same_frame_id": getattr(acquisition, "acquisition_id"),
        },
        "inputs": {
            "main": {"raw_sha256": getattr(acquisition, "main_rgbi_sha256")},
            "prepass": {"raw_sha256": getattr(acquisition, "prepass_rgbi_sha256")},
        },
        "provenance_class": "caller_asserted_bare_npy",
        "schema": "negpy.fauxce-hybrid-acquisition-assertion-v1",
    }
    if provenance.get("basis") != "caller_asserted" or provenance.get("source_manifest_sha256") != _sha256(_canonical_json(assertion)):
        _fail(label, "acquisition assertion binding changed")

    core = _require_dict(receipt.get("core"), label=f"{label} core")
    backend = _require_dict(core.get("backend"), label=f"{label} backend")
    generation = _require_dict(receipt.get("generation"), label=f"{label} generation")
    if core.get("source_manifest_sha256") != getattr(runtime, "core_source_manifest_sha256") or generation.get(
        "hybrid_source_manifest_sha256"
    ) != getattr(runtime, "hybrid_source_manifest_sha256"):
        _fail(label, "source manifests differ from the pinned runtime")
    if core.get("version") != repaired.get("engine_version"):
        _fail(label, "Hybrid core version differs from the repaired receipt")
    if (
        backend.get("requested") != repaired.get("backend_requested")
        or backend.get("used") != repaired.get("backend_used")
        or backend.get("reason") != repaired.get("backend_selection_reason")
    ):
        _fail(label, "backend disclosure differs from the repaired receipt")

    inpainting = _require_dict(receipt.get("inpainting"), label=f"{label} inpainting")
    if type(inpainting.get("invoked")) is not bool:
        _fail(label, "inpainting invocation disclosure is malformed")
    if inpainting["invoked"]:
        model = _require_dict(inpainting.get("model"), label=f"{label} model")
        tool = _require_dict(inpainting.get("tool"), label=f"{label} tool")
        run = _require_dict(inpainting.get("runtime"), label=f"{label} runtime")
        if (
            model.get("weights_sha256") != getattr(runtime, "model_weights_sha256")
            or tool.get("iopaint_source_manifest_sha256") != getattr(runtime, "iopaint_source_manifest_sha256")
            or run.get("device") != getattr(runtime, "inpaint_device")
            or run.get("threads") != getattr(runtime, "inpaint_threads")
            or run.get("seed") != getattr(runtime, "inpaint_seed")
        ):
            _fail(label, "inpainting runtime differs from the pinned runtime")


def _hybrid_result(
    repaired: dict[str, Any],
    *,
    acquisition: RepairAcquisition,
    repaired_rgb: np.ndarray,
    output: RollFrameOutput,
    root: Path,
    runtime: _HybridRuntime,
    referenced: set[Path],
    label: str,
) -> RepairResult:
    disclosure = _require_dict(repaired.get("disclosure_mask"), label=f"{label} disclosure mask")
    applied = _require_dict(disclosure.get("applied_final"), label=f"{label} applied mask")
    routed = _require_dict(disclosure.get("routed_raw"), label=f"{label} routed mask")
    storage_row = _require_dict(applied.get("storage"), label=f"{label} storage mask")
    native_row = _require_dict(applied.get("native"), label=f"{label} native mask")
    routed_row = _require_dict(routed.get("native"), label=f"{label} routed native mask")
    hybrid_row = _require_dict(repaired.get("hybrid_receipt"), label=f"{label} Hybrid receipt")
    binding_row = _require_dict(
        repaired.get("hybrid_evidence_binding"),
        label=f"{label} Hybrid binding",
    )

    storage_path = _published_path(
        output,
        "synthesis_mask_path",
        storage_row.get("path"),
        root=root,
        label=f"{label} storage mask",
    )
    native_path = _published_path(
        output,
        "native_synthesis_mask_path",
        native_row.get("path"),
        root=root,
        label=f"{label} native mask",
    )
    receipt_path = _published_path(
        output,
        "hybrid_receipt_path",
        hybrid_row.get("path"),
        root=root,
        label=f"{label} Hybrid receipt",
    )
    storage_path, storage_png = _artifact_row(
        storage_row,
        root=root,
        label=f"{label} storage mask",
        expected_path=storage_path,
    )
    native_path, native_png = _artifact_row(
        native_row,
        root=root,
        label=f"{label} native mask",
        expected_path=native_path,
    )
    routed_path, routed_png = _artifact_row(
        routed_row,
        root=root,
        label=f"{label} routed native mask",
    )
    receipt_path, receipt_bytes = _artifact_row(
        hybrid_row,
        root=root,
        label=f"{label} Hybrid receipt",
        expected_path=receipt_path,
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    binding_path, binding_bytes = _artifact_row(
        binding_row,
        root=root,
        label=f"{label} Hybrid binding",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    evidence_directory = receipt_path.parent
    hybrid_digest = _require_sha256(hybrid_row.get("sha256"), label=f"{label} Hybrid receipt SHA-256")
    expected_hidden_paths = {
        receipt_path: "hybrid-receipt.json",
        native_path: "synth-mask-applied-scanner-native.png",
        routed_path: "synth-mask-routed-scanner-native.png",
        binding_path: "negpy-binding.json",
    }
    if evidence_directory.name != hybrid_digest or any(
        path.parent != evidence_directory or path.name != expected_name for path, expected_name in expected_hidden_paths.items()
    ):
        _fail(label, "Hybrid evidence escaped its content-addressed directory")
    referenced.update((storage_path, native_path, routed_path, receipt_path, binding_path))

    try:
        receipt_document = roll_service._strict_json_loads(  # noqa: SLF001
            receipt_bytes
        )
        binding_document = roll_service._strict_json_loads(  # noqa: SLF001
            binding_bytes
        )
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise DeepAcceptanceError(f"{label}: retained Hybrid JSON is invalid: {error}") from error
    receipt_document = _require_dict(receipt_document, label=f"{label} Hybrid receipt")
    binding_document = _require_dict(binding_document, label=f"{label} Hybrid binding")
    if _canonical_json(receipt_document, newline=True, ensure_ascii=True) != receipt_bytes:
        _fail(label, "Hybrid receipt is not canonical")
    if _canonical_json(binding_document) != binding_bytes:
        _fail(label, "Hybrid retained binding is not canonical")

    engine = _require_str(repaired.get("engine"), label=f"{label} repair engine")
    engine_version = _require_str(repaired.get("engine_version"), label=f"{label} repair engine version")
    reason = _require_str(repaired.get("reason"), label=f"{label} repair reason")
    try:
        result = RepairResult(
            rgb=repaired_rgb,
            engine=engine,
            engine_version=engine_version,
            mode_requested=RepairMode(repaired.get("mode_requested")),
            mode_resolved=RepairMode(repaired.get("mode_resolved")),
            reason=reason,
            acquisition_id=repaired.get("acquisition_id"),
            slot=repaired.get("slot"),
            reservation_id=repaired.get("reservation_id"),
            evidence_sha256=repaired.get("evidence_sha256"),
            backend_requested=repaired.get("backend_requested"),
            backend_used=repaired.get("backend_used"),
            backend_selection_reason=repaired.get("backend_selection_reason"),
            native_output_rgb_sha256=repaired.get("native_output_rgb_sha256"),
            storage_output_rgb_sha256=repaired.get("storage_output_rgb_sha256"),
            native_synthesis_mask_png=native_png,
            native_synthesis_mask_sha256=native_row.get("sha256"),
            native_synthesis_mask_shape=tuple(native_row.get("shape", ())),
            routed_native_synthesis_mask_png=routed_png,
            routed_native_synthesis_mask_sha256=routed_row.get("sha256"),
            routed_native_synthesis_mask_shape=tuple(routed_row.get("shape", ())),
            storage_synthesis_mask_png=storage_png,
            storage_synthesis_mask_sha256=storage_row.get("sha256"),
            storage_synthesis_mask_shape=tuple(storage_row.get("shape", ())),
            synthesis_mask_transform=applied.get("transform"),
            synthesis_fraction=applied.get("fraction"),
            routing_counts=routed.get("routing_counts"),
            hybrid_receipt=receipt_bytes,
            hybrid_receipt_sha256=hybrid_row.get("sha256"),
            hybrid_provenance_class=hybrid_row.get("provenance_class"),
            hybrid_receipt_output_rgb_sha256=hybrid_row.get("verified_output_rgb_sha256"),
        )
    except (TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"{label}: Hybrid result receipt is malformed: {error}") from error
    try:
        roll_service._validate_repair_result_binding(  # noqa: SLF001
            acquisition,
            result,
            requested_mode=RepairMode.HYBRID,
        )
    except (TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"{label}: Hybrid result binding failed: {error}") from error
    if result.engine != "digital-fauxice" or result.degraded:
        _fail(label, "repair was not non-degraded digital-fauxice Hybrid")
    try:
        applied_mask = roll_service._decode_binary_mask(  # noqa: SLF001
            native_png,
            expected_shape=acquisition.main_rgbi.shape[:2],
            label=f"{label} final scanner-native disclosure mask",
        )
    except (OSError, TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"{label}: final disclosure mask could not be decoded: {error}") from error
    applied_pixel_count = int(np.count_nonzero(applied_mask))
    del applied_mask
    if applied.get("pixel_count") != applied_pixel_count:
        _fail(label, "outer synthesis pixel count differs from the decoded mask")

    expected_binding = {
        "acquisition": {
            "acquisition_id": getattr(acquisition, "acquisition_id"),
            "capture_attempt_id": getattr(acquisition, "capture_attempt_id"),
            "evidence_sha256": getattr(acquisition, "evidence_sha256"),
            "ir_validity_sha256": getattr(acquisition, "ir_validity_sha256"),
            "main_rgbi_sha256": getattr(acquisition, "main_rgbi_sha256"),
            "prepass_rgbi_sha256": getattr(acquisition, "prepass_rgbi_sha256"),
            "reservation_id": getattr(acquisition, "reservation_id"),
            "slot": getattr(acquisition, "slot"),
            "storage_transform": getattr(acquisition, "storage_transform"),
        },
        "hybrid_receipt_sha256": result.hybrid_receipt_sha256,
        "hybrid_receipt_output_rgb_sha256": result.hybrid_receipt_output_rgb_sha256,
        "native_output_rgb_sha256": result.native_output_rgb_sha256,
        "native_synthesis_mask_sha256": result.native_synthesis_mask_sha256,
        "routed_native_synthesis_mask_sha256": result.routed_native_synthesis_mask_sha256,
        "provenance_class": result.hybrid_provenance_class,
        "schema": "negpy.dice-hybrid-retained-evidence-v2",
        "storage_output_rgb_sha256": result.storage_output_rgb_sha256,
        "storage_synthesis_mask_sha256": result.storage_synthesis_mask_sha256,
        "synthesis_mask_transform": result.synthesis_mask_transform,
    }
    if binding_document != expected_binding:
        _fail(label, "Hybrid retained binding changed")
    _runtime_receipt_binding(
        receipt_document,
        acquisition=acquisition,
        repaired=repaired,
        runtime=runtime,
        label=label,
    )
    return result


def _retained_native_paths(
    retained_value: object,
    *,
    root: Path,
    label: str,
    referenced: set[Path],
) -> Path:
    retained = _require_dict(retained_value, label=label)
    if retained.get("scope") != exact_color.NATIVE_BUILDER_SCOPE or retained.get("native_per_acquisition_builder") is not True:
        _fail(label, "is not native per-acquisition builder evidence")
    receipt_path, _ = _artifact_row(retained.get("builder_receipt"), root=root, label=f"{label} receipt")
    if receipt_path.name != "native-builder-receipt.json":
        _fail(label, "native builder receipt filename changed")
    if receipt_path.parent.parent != root / ".negpy-native-builder":
        _fail(label, "native builder evidence escaped its production directory")
    referenced.add(receipt_path)
    directory = receipt_path.parent
    expected_rows = {
        "analyzer_rgb": directory / "analyzer-rgb-u16le.bin",
        "evidence_receipt": directory / "native-builder-evidence.json",
        "frame_ownership_receipt": (directory / "nikon-density-frame-ownership.json"),
        "density_evidence_receipt": directory / "nikon-density-evidence.json",
    }
    for key, expected_path in expected_rows.items():
        path, _ = _artifact_row(
            retained.get(key),
            root=root,
            label=f"{label} {key}",
            expected_path=expected_path,
        )
        referenced.add(path)
    luts = retained.get("pre_f_luts")
    if type(luts) is not list or len(luts) != 3:
        _fail(label, "must retain exactly three pre-F LUTs")
    for index, (channel, row) in enumerate(zip(("r", "g", "b"), luts, strict=True)):
        row_document = _require_dict(row, label=f"{label} pre-F LUT {index}")
        if row_document.get("channel") != channel:
            _fail(label, "pre-F LUT channel order changed")
        path, _ = _artifact_row(
            row_document,
            root=root,
            label=f"{label} pre-F LUT {index}",
            expected_path=directory / f"builder-preF-{channel}.bin",
        )
        referenced.add(path)
    return receipt_path


def _validate_frame(
    output: RollFrameOutput,
    *,
    root: Path,
    expected_slot: int,
    builder: PortableStage1Builder,
    evaluator: PortableCMSOnEvaluator,
    runtime: _HybridRuntime,
) -> _FrameAudit:
    label = f"slot {expected_slot}"
    receipt, receipt_bytes, completed_files = _collect_completed_frame_files_locked(
        output,
        root=root,
        expected_slot=expected_slot,
    )
    receipt_path = _regular_file(output.receipt_path, root=root, label=f"{label} frame receipt")
    receipt_lock_path = _regular_file(
        _receipt_lock_path(receipt_path),
        root=root,
        label=f"{label} frame receipt lock",
    )
    smear = receipt.get("transport_smear")
    if not isinstance(smear, dict):
        # Compatibility with an early compact receipt spelling.
        if receipt.get("transport_smear_verdict") != "clean":
            _fail(label, "transport-smear verdict is not clean")
    elif smear.get("verdict") != "clean":
        _fail(label, "transport-smear verdict is not clean")

    outputs = _require_dict(receipt.get("outputs"), label=f"{label} outputs")
    unrepaired = _require_dict(outputs.get("unrepaired"), label=f"{label} unrepaired tier")
    repaired = _require_dict(outputs.get("repaired"), label=f"{label} repaired tier")
    positive = _require_dict(outputs.get("positive"), label=f"{label} positive tier")
    if any(entry.get("written") is not True for entry in (unrepaired, repaired, positive)):
        _fail(label, "all three tiers were not written")
    if repaired.get("mode_requested") != "hybrid" or repaired.get("mode_resolved") != "hybrid" or repaired.get("degraded") is not False:
        _fail(label, "repair was not non-degraded Hybrid")
    if (
        positive.get("color_mode") != "nikon-exact"
        or positive.get("exact_nikon_color") is not True
        or positive.get("inversion_path") != "native-per-acquisition-builder-and-verified-portable-cms"
        or positive.get("native_per_acquisition_builder") is not True
        or positive.get("native_builder_scope") != exact_color.NATIVE_BUILDER_SCOPE
        or positive.get("builder_validated") is not True
        or positive.get("cms_verified") is not True
    ):
        _fail(label, "positive is not validated native Nikon exact color")

    rgb_path = _published_path(
        output,
        "rgb_path",
        unrepaired.get("rgb_path"),
        root=root,
        label=f"{label} RGB",
    )
    ir_path = _published_path(
        output,
        "ir_path",
        unrepaired.get("ir_path"),
        root=root,
        label=f"{label} IR",
    )
    repaired_rgb_path = _published_path(
        output,
        "repaired_rgb_path",
        repaired.get("rgb_path"),
        root=root,
        label=f"{label} repaired RGB",
    )
    repaired_ir_path = _published_path(
        output,
        "repaired_ir_path",
        repaired.get("ir_path"),
        root=root,
        label=f"{label} repaired IR",
    )
    positive_path = _published_path(
        output,
        "positive_path",
        positive.get("rgb_path"),
        root=root,
        label=f"{label} positive",
    )
    referenced = {
        receipt_path,
        receipt_lock_path,
        rgb_path,
        ir_path,
        repaired_rgb_path,
        repaired_ir_path,
        positive_path,
    }

    dice = _require_dict(
        outputs.get("repair_acquisition_evidence"),
        label=f"{label} DICE acquisition evidence",
    )
    if dice.get("retained") is not True or dice.get("replayable") is not True or dice.get("schema") != "negpy.dice-acquisition-replay-v1":
        _fail(label, "DICE acquisition evidence is not retained and replayable")
    binding_path, binding_bytes = _artifact_row(
        dice.get("binding"),
        root=root,
        label=f"{label} DICE acquisition binding",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    referenced.add(binding_path)
    _validate_dice_archive_paths(
        dice,
        binding_path=binding_path,
        binding_bytes=binding_bytes,
        root=root,
        rgb_path=rgb_path,
        ir_path=ir_path,
        referenced=referenced,
        label=label,
    )
    try:
        acquisition = roll_service.load_repair_acquisition_evidence(binding_path)
    except (OSError, TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"{label}: DICE replay failed: {error}") from error
    expected_rgb_name = f"acceptance_slot{expected_slot:02d}.tif"
    expected_dice_token = _sha256(f"{acquisition.acquisition_id}\0acceptance_slot{expected_slot:02d}".encode("utf-8"))
    if rgb_path.name != expected_rgb_name or binding_path.parent.name != expected_dice_token:
        _fail(label, "DICE evidence directory is not content-addressed")
    acquisition_entry = _require_dict(repaired.get("acquisition"), label=f"{label} repaired acquisition")
    expected_acquisition = {
        "acquisition_id": acquisition.acquisition_id,
        "slot": acquisition.slot,
        "reservation_id": acquisition.reservation_id,
        "capture_attempt_id": acquisition.capture_attempt_id,
        "evidence_sha256": acquisition.evidence_sha256,
        "storage_transform": acquisition.storage_transform,
        "main_rgbi_sha256": acquisition.main_rgbi_sha256,
        "prepass_rgbi_sha256": acquisition.prepass_rgbi_sha256,
        "ir_validity_sha256": acquisition.ir_validity_sha256,
    }
    if acquisition_entry != expected_acquisition:
        _fail(label, "repaired acquisition differs from retained DICE evidence")

    storage_shape = (
        acquisition.main_rgbi.shape[1],
        acquisition.main_rgbi.shape[0],
    )
    rgb = _decode_tiff(rgb_path, (*storage_shape, 3), label=f"{label} RGB")
    infrared = _decode_tiff(ir_path, storage_shape, label=f"{label} IR")
    expected_storage = np.rot90(acquisition.main_rgbi, k=1, axes=(0, 1))
    if not np.array_equal(rgb, expected_storage[..., :3]) or not np.array_equal(infrared, expected_storage[..., 3]):
        _fail(label, "Tier-1 TIFFs differ from the replayed DICE acquisition")
    artifacts = _require_dict(receipt.get("artifacts"), label=f"{label} artifacts")
    if set(artifacts) != {"rgb", "ir"}:
        _fail(label, "frame artifact inventory is not exactly RGB and IR")
    _array_artifact_matches(artifacts.get("rgb"), rgb, label=f"{label} RGB artifact")
    _array_artifact_matches(artifacts.get("ir"), infrared, label=f"{label} IR artifact")
    del rgb, expected_storage

    repaired_rgb = _decode_tiff(repaired_rgb_path, (*storage_shape, 3), label=f"{label} repaired RGB")
    repaired_ir = _decode_tiff(repaired_ir_path, storage_shape, label=f"{label} repaired IR")
    if not np.array_equal(repaired_ir, infrared):
        _fail(label, "repaired IR is not the unchanged Tier-1 IR")
    del repaired_ir, infrared

    repair_result = _hybrid_result(
        repaired,
        acquisition=acquisition,
        repaired_rgb=repaired_rgb,
        output=output,
        root=root,
        runtime=runtime,
        referenced=referenced,
        label=label,
    )
    if (
        positive.get("repair_engine") != repair_result.engine
        or positive.get("repair_engine_version") != repair_result.engine_version
        or positive.get("repair_mode") != str(repair_result.mode_resolved)
    ):
        _fail(label, "positive repair provenance differs from the repaired tier")
    acquisition_id = acquisition.acquisition_id
    acquisition_reservation_id = acquisition.reservation_id
    acquisition_capture_attempt_id = acquisition.capture_attempt_id
    repair_storage_output_rgb_sha256 = repair_result.storage_output_rgb_sha256
    hybrid_receipt_sha256 = repair_result.hybrid_receipt_sha256
    del repair_result, acquisition

    native_evidence = _require_dict(
        outputs.get("native_color_evidence"),
        label=f"{label} native color evidence",
    )
    if (
        native_evidence.get("retained") is not True
        or native_evidence.get("native_per_acquisition_builder") is not True
        or native_evidence.get("scope") != exact_color.NATIVE_BUILDER_SCOPE
    ):
        _fail(label, "native color evidence was not retained")
    retained = native_evidence.get("retained_builder_evidence")
    receipt_file = _retained_native_paths(
        retained,
        root=root,
        label=f"{label} native builder evidence",
        referenced=referenced,
    )
    if positive.get("retained_builder_evidence") != retained:
        _fail(label, "positive and native evidence reference different builders")
    try:
        builder_receipt = exact_color.load_native_builder_receipt(receipt_file)
    except exact_color.ExactColorUnavailable as error:
        raise DeepAcceptanceError(f"{label}: native builder reload failed: {error}") from error
    if (
        builder_receipt.slot != expected_slot
        or builder_receipt.reservation_id != acquisition_reservation_id
        or builder_receipt.capture_attempt_id != acquisition_capture_attempt_id
    ):
        _fail(label, "native builder belongs to another acquisition")
    expected_builder_document = exact_color.builder_receipt_payload(builder_receipt)
    if (
        native_evidence.get("builder_receipt") != expected_builder_document
        or native_evidence.get("builder_receipt_sha256") != builder_receipt.sha256
    ):
        _fail(label, "native color evidence does not bind the reloaded builder")

    try:
        color = exact_color.evaluate_exact_color(
            repaired_rgb,
            builder_receipt=builder_receipt,
            builder=builder,
            evaluator=evaluator,
        )
    except exact_color.ExactColorUnavailable as error:
        raise DeepAcceptanceError(f"{label}: exact color replay failed: {error}") from error
    application = color.builder_application_receipt
    if application is None:
        _fail(label, "exact color replay omitted its builder application receipt")
    _require_receipt_copy(
        positive.get("builder_receipt"),
        positive.get("builder_receipt_sha256"),
        exact_color.receipt_payload(color.builder_receipt),
        color.builder_receipt.sha256,
        label=f"{label} builder receipt",
    )
    _require_receipt_copy(
        positive.get("builder_application_receipt"),
        positive.get("builder_application_receipt_sha256"),
        exact_color.receipt_payload(application),
        application.sha256,
        label=f"{label} builder application receipt",
    )
    _require_receipt_copy(
        positive.get("cms_receipt"),
        positive.get("cms_receipt_sha256"),
        exact_color.receipt_payload(color.cms_receipt),
        color.cms_receipt.sha256,
        label=f"{label} CMS receipt",
    )
    if (
        positive.get("repaired_input_rgb_sha256") != color.source_rgb_sha256
        or positive.get("input_rgb_sha256") != color.input_rgb_sha256
        or positive.get("stage1_input_rgb_sha256") != color.input_rgb_sha256
        or positive.get("output_rgb_sha256") != color.output_rgb_sha256
        or repaired.get("storage_output_rgb_sha256") != repair_storage_output_rgb_sha256
    ):
        _fail(label, "color or repair content hashes changed")
    try:
        profile = nikon_icc.nikon_adobe_rgb_profile()
    except nikon_icc.NikonICCProfileError as error:
        raise DeepAcceptanceError(f"{label}: pinned Nikon ICC profile is unavailable: {error}") from error
    try:
        tiff_binding = roll_service._verify_exact_positive_tiff(  # noqa: SLF001
            str(positive_path),
            expected_rgb=color.rgb,
            expected_icc=profile,
        )
    except exact_color.ExactColorIntegrityError as error:
        raise DeepAcceptanceError(f"{label}: exact positive TIFF failed: {error}") from error
    if tiff_binding != positive.get("tiff_artifact"):
        _fail(label, "positive TIFF binding differs from its sidecar")
    if positive.get("icc_profile") != nikon_icc.profile_receipt_binding():
        _fail(label, "positive ICC receipt differs from the pinned Nikon profile")

    try:
        from coolscanpy.protocol.ls5000_single_pass.density import (
            NikonDensityFrameOwnershipReceipt,
        )

        ownership_document = roll_service._strict_json_loads(  # noqa: SLF001
            builder_receipt.frame_ownership_receipt
        )
        ownership = NikonDensityFrameOwnershipReceipt.from_dict(ownership_document)
    except (ImportError, TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"{label}: density ownership is invalid: {error}") from error
    outer_ownership = receipt.get("nikon_density_ownership")
    compact_ownership = {
        "reservation_id": ownership.reservation_id,
        "batch_session_id": ownership.batch_session_id,
        "preview_sha256": ownership.preview_sha256,
        "preview_identity_sha256": ownership.preview_identity_sha256,
        "transport_table_sha256": ownership.transport_table_sha256,
        "reviewed_fingerprint_sha256": ownership.reviewed_fingerprint_sha256,
        "fresh_fingerprint_sha256": ownership.fresh_fingerprint_sha256,
        "frame_capture_attempt_id": ownership.frame_capture_attempt_id,
        "frame_index": ownership.frame_index,
        "frame_total": ownership.frame_total,
        "selected_slots": list(ownership.selected_slots),
        "selected_slot": ownership.selected_slot,
    }
    if outer_ownership not in (compact_ownership, ownership.to_dict()):
        _fail(label, "public frame receipt and retained density ownership disagree")
    if (
        ownership.selected_slot != expected_slot
        or ownership.reservation_id != acquisition_reservation_id
        or ownership.frame_capture_attempt_id != acquisition_capture_attempt_id
        or receipt.get("reviewed_fingerprint_sha256") != ownership.reviewed_fingerprint_sha256
        or receipt.get("fresh_fingerprint_sha256") != ownership.fresh_fingerprint_sha256
    ):
        _fail(label, "frame, acquisition, and density ownership disagree")
    if (
        builder_receipt.preview_sha256 != ownership.preview_sha256
        or builder_receipt.preview_identity_sha256 != ownership.preview_identity_sha256
        or builder_receipt.batch_session_id != ownership.batch_session_id
    ):
        _fail(label, "native builder and density ownership disagree")

    if referenced != set(completed_files):
        missing = sorted(str(path) for path in set(completed_files) - referenced)
        extra = sorted(str(path) for path in referenced - set(completed_files))
        raise DeepAcceptanceError(f"{label}: deep artifact inventory changed (missing={missing}, extra={extra})")
    output_artifacts: dict[str, str] = {}
    for field in _OUTPUT_FIELDS:
        path = getattr(output, field)
        if type(path) is not str:
            _fail(label, "RollFrameOutput omitted a completed artifact")
        output_artifacts[field] = path

    return _FrameAudit(
        summary={
            "slot": expected_slot,
            "frame_receipt": {
                "path": str(receipt_path),
                "bytes": len(receipt_bytes),
                "sha256": _sha256(receipt_bytes),
            },
            "acquisition_id": acquisition_id,
            "reservation_id": acquisition_reservation_id,
            "capture_attempt_id": acquisition_capture_attempt_id,
            "transport_identity_sha256": ownership.transport_identity_sha256,
            "preview_identity_sha256": ownership.preview_identity_sha256,
            "builder_receipt_sha256": builder_receipt.sha256,
            "cms_receipt_sha256": color.cms_receipt.sha256,
            "positive_file_sha256": tiff_binding["file_sha256"],
            "hybrid_receipt_sha256": hybrid_receipt_sha256,
            "referenced_file_count": len(referenced),
            "referenced_files": sorted(str(path) for path in referenced),
        },
        referenced_files=frozenset(referenced),
        receipt_path=receipt_path,
        receipt=receipt,
        ownership=ownership,
        builder_receipt=builder_receipt,
        output_artifacts=output_artifacts,
    )


def validate_completed_frame(
    output: RollFrameOutput,
    *,
    output_dir: str | os.PathLike[str],
    expected_slot: int | None = None,
    builder: PortableStage1Builder | None = None,
    evaluator: PortableCMSOnEvaluator | None = None,
    hybrid_runtime: _HybridRuntime | None = None,
) -> dict[str, Any]:
    """Deeply validate one already-published frame without touching hardware."""

    checked_output, slot = _public_output_slot(
        output,
        expected_slot=expected_slot,
    )
    root = _output_root(output_dir)
    runtime = _validated_runtime(hybrid_runtime)
    active_builder, active_evaluator = _color_dependencies(builder, evaluator)
    with _hold_frame_locks((checked_output,), root=root):
        audit = _validate_frame(
            checked_output,
            root=root,
            expected_slot=slot,
            builder=active_builder,
            evaluator=active_evaluator,
            runtime=runtime,
        )
        return {
            "schema": SCHEMA,
            "status": "passed",
            "scope": "frame",
            "output_dir": str(root),
            **audit.summary,
        }


def _validate_manual_approval(
    frame: _FrameAudit,
    *,
    expected: bool | None,
) -> dict[str, Any] | None:
    slot = frame.summary["slot"]
    approval_value = frame.receipt.get("manual_approval")
    if expected is not None and (approval_value is not None) is not expected:
        _fail(f"slot {slot}", "manual approval presence is wrong")
    if approval_value is None:
        return None
    approval = _require_dict(approval_value, label=f"slot {slot} manual approval")
    if set(approval) != {
        "reviewed_fingerprint_sha256",
        "slot",
        "spacing_offset",
        "thumbnail_sha256",
        "reviewed_lookup_row",
        "reviewed_native_origin",
        "review_reasons",
    }:
        _fail(f"slot {slot}", "manual approval fields changed")
    review_reasons = approval.get("review_reasons")
    if type(review_reasons) is not list or not review_reasons or any(type(reason) is not str or not reason for reason in review_reasons):
        _fail(f"slot {slot}", "manual approval reasons are malformed")
    try:
        from coolscanpy.types import ApprovalReceipt

        parsed = ApprovalReceipt(
            reviewed_fingerprint_sha256=approval["reviewed_fingerprint_sha256"],
            slot=approval["slot"],
            spacing_offset=approval["spacing_offset"],
            thumbnail_sha256=approval["thumbnail_sha256"],
            reviewed_lookup_row=approval["reviewed_lookup_row"],
            reviewed_native_origin=approval["reviewed_native_origin"],
            review_reasons=tuple(review_reasons),
        )
    except (ImportError, KeyError, TypeError, ValueError) as error:
        raise DeepAcceptanceError(f"slot {slot}: manual approval is invalid: {error}") from error
    ownership = frame.ownership
    if (
        parsed.slot != slot
        or parsed.spacing_offset != frame.receipt.get("spacing_offset")
        or parsed.reviewed_fingerprint_sha256 != getattr(ownership, "reviewed_fingerprint_sha256")
    ):
        _fail(f"slot {slot}", "manual approval does not bind the reviewed frame")
    return {
        "binding_sha256": parsed.binding_sha256,
        "boundary_offset_rows": parsed.spacing_offset,
        "review_reasons": list(parsed.review_reasons),
        "reviewed_fingerprint_sha256": parsed.reviewed_fingerprint_sha256,
        "reviewed_lookup_row": parsed.reviewed_lookup_row,
        "reviewed_native_origin": parsed.reviewed_native_origin,
        "schema_version": 1,
        "slot": parsed.slot,
        "thumbnail_sha256": parsed.thumbnail_sha256,
    }


def _inventory(
    root: Path,
    frames: Sequence[_FrameAudit],
    *,
    allowed_output_lock_name: str | None,
) -> dict[str, Any]:
    expected_files = set().union(*(frame.referenced_files for frame in frames))
    if allowed_output_lock_name is not None:
        if (
            type(allowed_output_lock_name) is not str
            or not allowed_output_lock_name
            or Path(allowed_output_lock_name).name != allowed_output_lock_name
            or allowed_output_lock_name in {".", ".."}
        ):
            raise DeepAcceptanceError("allowed output lock name must be one safe basename")
        expected_files.add(root / allowed_output_lock_name)

    found_files: set[Path] = set()
    found_directories = {root}
    try:
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            parent = Path(directory)
            for name in names:
                child = parent / name
                metadata = child.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    _fail("inventory", f"directory entry is unsafe: {child}")
                found_directories.add(child.resolve(strict=True))
            for name in files:
                child = parent / name
                metadata = child.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    _fail("inventory", f"file entry is unsafe: {child}")
                found_files.add(child.resolve(strict=True))
    except OSError as error:
        raise DeepAcceptanceError(f"inventory walk failed: {error}") from error

    try:
        expected_files = {path.resolve(strict=True) for path in expected_files}
    except OSError as error:
        raise DeepAcceptanceError(f"inventory expected file is unavailable: {error}") from error
    expected_directories = {root}
    for path in expected_files:
        parent = path.parent
        while parent != root:
            expected_directories.add(parent)
            parent = parent.parent
        expected_directories.add(root)
    if found_files != expected_files:
        missing = sorted(str(path.relative_to(root)) for path in expected_files - found_files)
        extra = sorted(str(path.relative_to(root)) for path in found_files - expected_files)
        raise DeepAcceptanceError(f"inventory file set changed (missing={missing}, extra={extra})")
    if found_directories != expected_directories:
        missing = sorted(str(path.relative_to(root)) for path in expected_directories - found_directories)
        extra = sorted(str(path.relative_to(root)) for path in found_directories - expected_directories)
        raise DeepAcceptanceError(f"inventory directory set changed (missing={missing}, extra={extra})")
    allowed_lock_path = (root / allowed_output_lock_name).resolve(strict=True) if allowed_output_lock_name is not None else None
    visible = [
        path for path in found_files if path != allowed_lock_path and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
    if len(visible) != 42:
        raise DeepAcceptanceError(f"visible inventory contains {len(visible)} files; expected 42")
    return {
        "regular_file_count": len(found_files),
        "directory_count": len(found_directories),
        "visible_file_count": len(visible),
        "allowed_output_lock_path": (str(allowed_lock_path) if allowed_lock_path is not None else None),
        "exact": True,
    }


def _run_receipt_binding(
    run_receipt_path: str | os.PathLike[str] | None,
    *,
    root: Path,
    frames: Sequence[_FrameAudit],
    approvals: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    if run_receipt_path is None:
        return None
    path = Path(run_receipt_path).absolute()
    if path.is_relative_to(root):
        raise DeepAcceptanceError("run receipt must be outside the output directory")
    payload = _stable_bytes(
        path,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        label="live run receipt",
    )
    try:
        document = roll_service._strict_json_loads(payload)  # noqa: SLF001
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise DeepAcceptanceError(f"live run receipt is invalid JSON: {error}") from error
    document = _require_dict(document, label="live run receipt")
    if _canonical_json(document, newline=True, ensure_ascii=True) != payload:
        raise DeepAcceptanceError("live run receipt is not canonical JSON")
    rows = document.get("frames")
    settings = document.get("settings")
    close = document.get("close")
    output_lease = document.get("output_lease")
    operation_state = document.get("operation_state")
    deep_result = document.get("deep_acceptance")
    expected_device_id = frames[0].receipt.get("device_id") if frames else None
    approved_slots = sorted(approvals)
    if (
        document.get("schema") != "negpy.ls5000-live-acceptance.v2"
        or document.get("status") != "succeeded"
        or document.get("phase") != "succeeded"
        or document.get("device_id") != expected_device_id
        or document.get("output_dir") != str(root)
        or document.get("slots") != list(SLOTS)
        or document.get("approved_slots") != approved_slots
        or settings
        != {
            "write_unrepaired": True,
            "write_repaired": True,
            "write_positive": True,
            "repair_mode": "hybrid",
            "positive_mode": "nikon-exact",
            "filename_pattern": 'acceptance_slot{{ "%02d" % seq }}',
        }
        or close
        != {
            "iterator": {"attempted": True, "succeeded": True},
            "roll": {"attempted": True, "succeeded": True},
        }
        or type(output_lease) is not dict
        or output_lease.get("acquired") is not True
        or output_lease.get("release_attempted") is not True
        or output_lease.get("released") is not True
        or type(operation_state) is not dict
        or operation_state.get("batch_exhausted") is not True
        or operation_state.get("verified_slots") != list(SLOTS)
        or type(deep_result) is not dict
        or deep_result.get("status") != "passed"
        or deep_result.get("slots") != list(SLOTS)
        or document.get("retry_count") != 0
        or document.get("eject_requested") is not False
        or type(rows) is not list
        or len(rows) != 6
    ):
        raise DeepAcceptanceError("live run receipt did not record one successful six-frame run")
    for frame, row_value in zip(frames, rows, strict=True):
        row = _require_dict(row_value, label="live run frame")
        artifacts = _require_dict(row.get("artifacts"), label="live run frame artifacts")
        if (
            row.get("slot") != frame.summary["slot"]
            or row.get("expected_slot") != frame.summary["slot"]
            or row.get("frame_receipt_sha256") != frame.summary["frame_receipt"]["sha256"]
            or artifacts != frame.output_artifacts
        ):
            _fail("live run receipt", "frame sidecar binding changed")

    reviewed_row = _require_dict(
        document.get("reviewed_approval"),
        label="live run reviewed approval",
    )
    if set(reviewed_row) != {
        "path",
        "expected_sha256",
        "verified_sha256",
        "bytes",
        "reviewed_fingerprint_sha256",
        "contact_sheet",
    }:
        _fail("live run receipt", "reviewed-approval binding fields changed")
    reviewed_path_value = reviewed_row.get("path")
    if type(reviewed_path_value) is not str or not Path(reviewed_path_value).is_absolute():
        _fail("live run receipt", "reviewed-approval path is not absolute")
    reviewed_path = Path(reviewed_path_value).absolute()
    if reviewed_path.is_relative_to(root):
        _fail("live run receipt", "reviewed approval must be outside output")
    reviewed_bytes = _stable_bytes(
        reviewed_path,
        maximum_bytes=64 * 1024,
        label="reviewed approval",
    )
    reviewed_digest = _sha256(reviewed_bytes)
    if (
        reviewed_row.get("bytes") != len(reviewed_bytes)
        or reviewed_row.get("expected_sha256") != reviewed_digest
        or reviewed_row.get("verified_sha256") != reviewed_digest
    ):
        _fail("live run receipt", "reviewed-approval content binding changed")
    try:
        reviewed_document = roll_service._strict_json_loads(  # noqa: SLF001
            reviewed_bytes
        )
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise DeepAcceptanceError(f"live run reviewed approval is invalid JSON: {error}") from error
    reviewed_document = _require_dict(
        reviewed_document,
        label="live run reviewed approval",
    )
    if _canonical_json(reviewed_document, newline=True, ensure_ascii=True) != reviewed_bytes:
        _fail("live run receipt", "reviewed approval is not canonical JSON")
    expected_approval_rows = [approvals[slot] for slot in approved_slots]
    reviewed_contact = _require_dict(
        reviewed_document.get("contact_sheet"),
        label="reviewed contact sheet",
    )
    if (
        reviewed_document.get("schema") != "negpy.ls5000-reviewed-approval.v1"
        or reviewed_document.get("approvals") != expected_approval_rows
        or reviewed_document.get("reviewed_fingerprint_sha256") != getattr(frames[0].ownership, "reviewed_fingerprint_sha256")
        or reviewed_row.get("reviewed_fingerprint_sha256") != reviewed_document.get("reviewed_fingerprint_sha256")
        or reviewed_row.get("contact_sheet")
        != {
            "path": reviewed_contact.get("path"),
            "sha256": reviewed_contact.get("sha256"),
        }
    ):
        _fail("live run receipt", "reviewed approvals differ from frame receipts")
    return {"path": str(path), "bytes": len(payload), "sha256": _sha256(payload)}


def _validate_six_frame_batch_locked(
    outputs: Sequence[RollFrameOutput],
    *,
    output_dir: str | os.PathLike[str],
    run_receipt_path: str | os.PathLike[str] | None = None,
    allowed_output_lock_name: str | None = None,
    builder: PortableStage1Builder | None = None,
    evaluator: PortableCMSOnEvaluator | None = None,
    hybrid_runtime: _HybridRuntime | None = None,
) -> dict[str, Any]:
    """Deeply validate one exact six-frame batch and its recursive inventory."""

    if isinstance(outputs, (str, bytes, bytearray)):
        raise DeepAcceptanceError("batch must contain exactly six RollFrameOutput values")
    try:
        frame_outputs = tuple(outputs)
    except TypeError as error:
        raise DeepAcceptanceError("batch must contain exactly six RollFrameOutput values") from error
    if len(frame_outputs) != 6 or any(type(output) is not RollFrameOutput for output in frame_outputs):
        raise DeepAcceptanceError("batch must contain exactly six RollFrameOutput values")
    if tuple(output.slot for output in frame_outputs) != SLOTS:
        raise DeepAcceptanceError("batch slots must be exactly 1,2,3,4,5,6 in order")
    root = _output_root(output_dir)
    runtime = _validated_runtime(hybrid_runtime)
    active_builder, active_evaluator = _color_dependencies(builder, evaluator)
    frames = tuple(
        _validate_frame(
            output,
            root=root,
            expected_slot=slot,
            builder=active_builder,
            evaluator=active_evaluator,
            runtime=runtime,
        )
        for slot, output in zip(SLOTS, frame_outputs, strict=True)
    )

    common = None
    density_receipt_sha256 = None
    device_identity = None
    approval_rows: dict[int, dict[str, Any]] = {}
    for index, frame in enumerate(frames, start=1):
        ownership = frame.ownership
        if (
            getattr(ownership, "frame_index") != index
            or getattr(ownership, "frame_total") != 6
            or tuple(getattr(ownership, "selected_slots")) != SLOTS
            or getattr(ownership, "selected_slot") != index
        ):
            _fail(f"slot {index}", "batch coordinates are wrong")
        approval_row = _validate_manual_approval(frame, expected=None)
        if approval_row is not None:
            approval_rows[index] = approval_row
        candidate = (
            getattr(ownership, "reservation_id"),
            getattr(ownership, "batch_session_id"),
            getattr(ownership, "preview_sha256"),
            getattr(ownership, "preview_identity_sha256"),
            getattr(ownership, "transport_table_sha256"),
            getattr(ownership, "transport_identity_sha256"),
            getattr(ownership, "reviewed_fingerprint_sha256"),
            getattr(ownership, "fresh_fingerprint_sha256"),
        )
        if common is None:
            common = candidate
        elif candidate != common:
            raise DeepAcceptanceError("batch frames disagree on reservation, preview, transport, or fingerprint identity")
        current_density = frame.builder_receipt.density_evidence_receipt_sha256
        if density_receipt_sha256 is None:
            density_receipt_sha256 = current_density
        elif current_density != density_receipt_sha256:
            raise DeepAcceptanceError("batch frames disagree on Nikon density evidence")
        current_device = (
            frame.receipt.get("device_id"),
            frame.receipt.get("device_model"),
        )
        if device_identity is None:
            device_identity = current_device
        elif current_device != device_identity:
            raise DeepAcceptanceError("batch frames disagree on scanner identity")

    inventory = _inventory(
        root,
        frames,
        allowed_output_lock_name=allowed_output_lock_name,
    )
    run_receipt = _run_receipt_binding(
        run_receipt_path,
        root=root,
        frames=frames,
        approvals=approval_rows,
    )
    assert common is not None
    approved_slots = sorted(approval_rows)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed",
        "scope": "six-frame-batch",
        "output_dir": str(root),
        "slots": list(SLOTS),
        "approved_slots": approved_slots,
        "reservation_id": common[0],
        "preview_sha256": common[2],
        "preview_identity_sha256": common[3],
        "transport_identity_sha256": common[5],
        "reviewed_fingerprint_sha256": common[6],
        "fresh_fingerprint_sha256": common[7],
        "density_evidence_receipt_sha256": density_receipt_sha256,
        "frames": [frame.summary for frame in frames],
        "referenced_files": sorted(str(path) for path in set().union(*(frame.referenced_files for frame in frames))),
        "inventory": inventory,
    }
    if run_receipt is not None:
        result["run_receipt"] = run_receipt
    assert device_identity is not None
    result.update(
        device_id=device_identity[0],
        device_model=device_identity[1],
        manual_approval_bindings=[
            {
                "slot": slot,
                "binding_sha256": approval_rows[slot]["binding_sha256"],
            }
            for slot in approved_slots
        ],
    )
    return result


def validate_six_frame_batch(
    outputs: Sequence[RollFrameOutput],
    *,
    output_dir: str | os.PathLike[str],
    run_receipt_path: str | os.PathLike[str] | None = None,
    allowed_output_lock_name: str | None = None,
    builder: PortableStage1Builder | None = None,
    evaluator: PortableCMSOnEvaluator | None = None,
    hybrid_runtime: _HybridRuntime | None = None,
) -> dict[str, Any]:
    """Deeply validate one locked, exact six-frame batch and its inventory."""

    if isinstance(outputs, (str, bytes, bytearray)):
        raise DeepAcceptanceError("batch must contain exactly six RollFrameOutput values")
    try:
        frame_outputs = tuple(outputs)
    except TypeError as error:
        raise DeepAcceptanceError("batch must contain exactly six RollFrameOutput values") from error
    if len(frame_outputs) != 6 or any(type(output) is not RollFrameOutput for output in frame_outputs):
        raise DeepAcceptanceError("batch must contain exactly six RollFrameOutput values")
    if tuple(output.slot for output in frame_outputs) != SLOTS:
        raise DeepAcceptanceError("batch slots must be exactly 1,2,3,4,5,6 in order")
    root = _output_root(output_dir)
    with _hold_frame_locks(frame_outputs, root=root):
        return _validate_six_frame_batch_locked(
            frame_outputs,
            output_dir=root,
            run_receipt_path=run_receipt_path,
            allowed_output_lock_name=allowed_output_lock_name,
            builder=builder,
            evaluator=evaluator,
            hybrid_runtime=hybrid_runtime,
        )


__all__ = [
    "DeepAcceptanceError",
    "SCHEMA",
    "collect_completed_frame_files",
    "validate_completed_frame",
    "validate_six_frame_batch",
]
