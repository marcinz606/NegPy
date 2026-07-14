"""Artifact-first finalization for completed LS-5000 single-pass captures.

The USB worker owns acquisition.  This module starts only after that process
has released the scanner and left a journal plus packed stream in a unique
attempt directory.  It deliberately has no USB or SANE imports.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, cast

import numpy as np
import tifffile

from negpy.infrastructure.scanners.ls5000_single_pass.packed import (
    EXPECTED_RECORDS,
    FULL_RECORD_BYTES,
    HEIGHT,
    WIDTH,
    decode_full_records,
)
from negpy.infrastructure.scanners.ls5000_single_pass.continuation_plan import (
    CANONICAL_CONTINUATION_PLAN_SHA256,
)
from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    MAX_FRAME_START_DELTA_ROWS,
    MAX_FRAME_START_MEDIAN_DELTA_ROWS,
    MAX_NATIVE_ORIGIN_DELTA,
    MAX_NATIVE_ORIGIN_MEDIAN_DELTA,
    MAX_PREVIEW_HEIGHT_DELTA_ROWS,
    MAX_SELECTED_VISUAL_HAMMING,
    MAX_VISUAL_MEDIAN_HAMMING,
    MAX_VISUAL_P90_HAMMING,
    MIN_DISCRIMINATIVE_FRAME_COUNT,
    MIN_VISUAL_LOG_SPAN,
    ManualFrameApproval,
)
from negpy.infrastructure.scanners.result import ScanResult, SplitIrAlignment
from negpy.infrastructure.tiff_contract import has_linear_scanner_rgb_marker
from negpy.services.scanning.quality import (
    FocusDetailTelemetry,
    ScanClippingTelemetry,
    StoppedTransportSmearAssessment,
    assess_stopped_transport_smear,
    measure_focus_detail,
    measure_scan_clipping,
)
from negpy.services.scanning.writer import write_full_negative_tiff


WORKFLOW_KIND = "negpy.ls5000-single-pass-finalization"
WORKFLOW_VERSION = 2
EXPECTED_COLOR_IDS = frozenset({1, 2, 3, 9})
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SinglePassWorkflowError(RuntimeError):
    """A capture cannot safely become a committed NegPy artifact set."""


class SinglePassIntegrityError(SinglePassWorkflowError):
    """Durable evidence disagrees with live files or the proven wire layout."""


@dataclass(frozen=True)
class PackedCaptureContract:
    """Geometry of one packed stream, configurable only for fast unit tests."""

    records: int = EXPECTED_RECORDS
    record_bytes: int = FULL_RECORD_BYTES
    width: int = WIDTH
    height: int = HEIGHT
    dpi: int = 4000

    def __post_init__(self) -> None:
        for name in ("records", "record_bytes", "width", "height", "dpi"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.records != (self.height + 1) // 2:
            raise ValueError("packed record count must provide exactly two row slots per record")

    @property
    def stream_bytes(self) -> int:
        return self.records * self.record_bytes


@dataclass(frozen=True)
class SinglePassSession:
    """Durable roll session metadata; an exposure count is advisory only."""

    root: Path
    session_id: str
    nominal_exposure_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if not _SAFE_ID.fullmatch(self.session_id):
            raise ValueError("session_id must be a filesystem-safe identifier")
        count = self.nominal_exposure_count
        if count is not None and (type(count) is not int or not 1 <= count <= 40):
            raise ValueError("nominal exposure count must be an integer in 1..40")


class _AttemptPathsLike(Protocol):
    @property
    def directory(self) -> Path: ...

    @property
    def output(self) -> Path: ...

    @property
    def journal(self) -> Path: ...


class _AttemptRequestLike(Protocol):
    @property
    def selected_slot(self) -> int | None: ...

    @property
    def boundary_offset_rows(self) -> int: ...

    @property
    def mode(self) -> object: ...


class CaptureAttemptResultLike(Protocol):
    """Structural seam for ``capture_process.CaptureAttemptResult``."""

    @property
    def paths(self) -> _AttemptPathsLike: ...

    @property
    def request(self) -> _AttemptRequestLike: ...

    @property
    def returncode(self) -> int: ...


@dataclass(frozen=True)
class SinglePassAttempt:
    """Paths and explicit scanner slot belonging to one worker attempt."""

    session: SinglePassSession
    attempt_id: str
    directory: Path
    stream_path: Path
    journal_path: Path
    selected_slot: int
    worker_returncode: int
    boundary_offset_rows: int = 0
    batch_session_id: str | None = None
    batch_frame_index: int | None = None
    batch_frame_total: int | None = None
    batch_selected_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.attempt_id):
            raise ValueError("attempt_id must be a filesystem-safe identifier")
        directory = Path(self.directory).expanduser().resolve()
        stream_path = Path(self.stream_path).expanduser().resolve()
        journal_path = Path(self.journal_path).expanduser().resolve()
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "stream_path", stream_path)
        object.__setattr__(self, "journal_path", journal_path)
        if not _is_within(directory, self.session.root):
            raise ValueError("attempt directory must be inside its session root")
        if stream_path.parent != directory or journal_path.parent != directory:
            raise ValueError("worker stream and journal must be direct children of the attempt directory")
        if type(self.selected_slot) is not int or not 1 <= self.selected_slot <= 40:
            raise ValueError("selected scanner slot must be an integer in 1..40")
        if type(self.worker_returncode) is not int:
            raise TypeError("worker return code must be an integer")
        minimum_offset = 0 if self.selected_slot == 1 else -144
        if (
            type(self.boundary_offset_rows) is not int
            or not minimum_offset <= self.boundary_offset_rows <= 144
        ):
            raise ValueError(
                f"slot {self.selected_slot} boundary offset must be an integer "
                f"in {minimum_offset}..144"
            )
        object.__setattr__(self, "batch_selected_slots", tuple(self.batch_selected_slots))
        batch_values = (
            self.batch_session_id,
            self.batch_frame_index,
            self.batch_frame_total,
        )
        if all(value is None for value in batch_values) and not self.batch_selected_slots:
            return
        if any(value is None for value in batch_values):
            raise ValueError("batch attempt identity must be complete or absent")
        if not isinstance(self.batch_session_id, str) or not _SAFE_ID.fullmatch(
            self.batch_session_id
        ):
            raise ValueError("batch_session_id must be filesystem-safe")
        if (
            type(self.batch_frame_index) is not int
            or type(self.batch_frame_total) is not int
            or not 1 <= self.batch_frame_index <= self.batch_frame_total
        ):
            raise ValueError("batch frame index must be inside the batch total")
        if (
            len(self.batch_selected_slots) != self.batch_frame_total
            or tuple(sorted(set(self.batch_selected_slots)))
            != self.batch_selected_slots
            or any(type(slot) is not int or not 1 <= slot <= 40 for slot in self.batch_selected_slots)
        ):
            raise ValueError("batch selected slots must be ordered, unique, and complete")
        if self.batch_selected_slots[self.batch_frame_index - 1] != self.selected_slot:
            raise ValueError("batch frame order does not identify this selected slot")

    @classmethod
    def from_capture_result(
        cls,
        *,
        session: SinglePassSession,
        result: CaptureAttemptResultLike,
    ) -> SinglePassAttempt:
        """Adapt the process boundary without importing its concrete classes."""

        slot = result.request.selected_slot
        mode = getattr(result.request.mode, "value", result.request.mode)
        if mode != "full" or type(slot) is not int:
            raise ValueError("only a full capture with an explicit selected slot can be finalized")
        directory = Path(result.paths.directory)
        return cls(
            session=session,
            attempt_id=directory.name,
            directory=directory,
            stream_path=Path(result.paths.output),
            journal_path=Path(result.paths.journal),
            selected_slot=slot,
            worker_returncode=result.returncode,
            boundary_offset_rows=result.request.boundary_offset_rows,
            batch_session_id=getattr(result, "batch_session_id", None),
            batch_frame_index=getattr(result, "batch_frame_index", None),
            batch_frame_total=getattr(result, "batch_frame_total", None),
            batch_selected_slots=getattr(result, "batch_selected_slots", ()),
        )


@dataclass(frozen=True)
class SinglePassFinalizationResult:
    manifest_path: Path
    output_paths: dict[str, Path]
    manifest: dict[str, object]
    resumed: bool
    scratch_deleted: bool


Decoder = Callable[[Path], tuple[np.ndarray, Mapping[str, object]]]
Writer = Callable[..., dict[str, dict[str, object]]]
StableHasher = Callable[[Path], tuple[str, int]]
ManifestWriter = Callable[[Path, bytes], None]
StreamDeleter = Callable[[Path], None]
SmearAssessor = Callable[..., StoppedTransportSmearAssessment]
ClippingMeasurer = Callable[[np.ndarray], ScanClippingTelemetry]
FocusMeasurer = Callable[[np.ndarray], FocusDetailTelemetry]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _stable_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _hash_file_stable(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            while block := handle.read(chunk_bytes):
                digest.update(block)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise SinglePassIntegrityError(f"cannot hash {path}: {error}") from error
    if _stable_identity(before) != _stable_identity(after):
        raise SinglePassIntegrityError(f"{path} changed while it was being hashed")
    return digest.hexdigest(), before.st_size


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise SinglePassIntegrityError(f"JSON contains duplicate key {key!r}")
        parsed[key] = value
    return parsed


def _load_json_stable(path: Path) -> tuple[dict[str, Any], str, int]:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise SinglePassIntegrityError(f"cannot read {path}: {error}") from error
    if _stable_identity(before) != _stable_identity(after) or len(payload) != before.st_size:
        raise SinglePassIntegrityError(f"{path} changed while it was being read")
    try:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SinglePassIntegrityError(f"{path} is not valid JSON: {error}") from error
    if type(parsed) is not dict:
        raise SinglePassIntegrityError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], parsed), hashlib.sha256(payload).hexdigest(), len(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SinglePassIntegrityError(f"refusing to overwrite {path}") from error
        linked = True
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if not linked:
            temporary.unlink(missing_ok=True)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SinglePassIntegrityError(f"manifest evidence is not JSON serializable: {error}") from error


def _manifest_binding(payload: Mapping[str, object]) -> str:
    semantic = {key: value for key, value in payload.items() if key != "binding_sha256"}
    try:
        canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SinglePassIntegrityError(f"manifest evidence is not bindable JSON: {error}") from error
    return hashlib.sha256(canonical).hexdigest()


def _require_object(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if type(value) is not dict:
        raise SinglePassIntegrityError(f"journal {key} must be an object")
    return cast(dict[str, Any], value)


def _require_exact(journal: Mapping[str, Any], key: str, expected: object) -> None:
    actual = journal.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise SinglePassIntegrityError(f"journal {key} is {actual!r}, expected {expected!r}")


def _finite_number(value: object, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _validate_quality_control_evidence(raw: object) -> None:
    if type(raw) is not dict:
        raise SinglePassIntegrityError("quality control evidence is missing")
    quality = cast(dict[str, Any], raw)
    clipping = quality.get("capture_clipping")
    if type(clipping) is not dict:
        raise SinglePassIntegrityError("capture clipping evidence is missing")
    clipping = cast(dict[str, Any], clipping)
    fractions = clipping.get("fractions")
    if (
        set(clipping)
        != {"clip_level", "fractions", "warning", "warning_fraction"}
        or type(fractions) is not list
        or len(fractions) != 3
        or not all(
            _finite_number(value) and float(value) <= 1.0 for value in fractions
        )
        or not _finite_number(clipping.get("clip_level"))
        or not 0.0 < float(clipping["clip_level"]) <= 1.0
        or not _finite_number(clipping.get("warning_fraction"))
        or float(clipping["warning_fraction"]) > 1.0
        or type(clipping.get("warning")) is not bool
        or clipping["warning"]
        is not (max(float(value) for value in fractions) > clipping["warning_fraction"])
    ):
        raise SinglePassIntegrityError("capture clipping evidence is malformed")

    focus = quality.get("focus_detail")
    if type(focus) is not dict:
        raise SinglePassIntegrityError("focus detail evidence is missing")
    focus = cast(dict[str, Any], focus)
    verdict = focus.get("verdict")
    score = focus.get("score")
    if (
        set(focus) != {"method", "score", "texture_span", "verdict"}
        or focus.get("method") != "normalized-gradient-v1"
        or verdict not in ("measured", "indeterminate")
        or not _finite_number(focus.get("texture_span"))
        or (verdict == "measured" and not _finite_number(score))
        or (verdict == "indeterminate" and score is not None)
    ):
        raise SinglePassIntegrityError("focus detail evidence is malformed")


def _canonical_output_paths(attempt: SinglePassAttempt) -> tuple[Path, dict[str, Path]]:
    root = attempt.directory / "final"
    rgb = root / f"frame{attempt.selected_slot:03d}.tif"
    base = rgb.with_suffix("")
    return root / "manifest.json", {
        "rgb": rgb,
        "ir": Path(f"{base}_IR.tif"),
        "ir_valid_mask": Path(f"{base}_IR_VALID.tif"),
    }


def _attempt_identity(attempt: SinglePassAttempt) -> dict[str, object]:
    identity: dict[str, object] = {
        "session_id": attempt.session.session_id,
        "attempt_id": attempt.attempt_id,
        "selected_slot": attempt.selected_slot,
        "boundary_offset_rows": attempt.boundary_offset_rows,
    }
    if attempt.batch_session_id is not None:
        identity["batch_session"] = {
            "session_id": attempt.batch_session_id,
            "frame_index": attempt.batch_frame_index,
            "frame_total": attempt.batch_frame_total,
            "selected_slots": list(attempt.batch_selected_slots),
        }
    return identity


class LS5000SinglePassWorkflow:
    """Validate, decode, commit, verify, and optionally clean one attempt."""

    def __init__(
        self,
        *,
        contract: PackedCaptureContract | None = None,
        decoder: Decoder | None = None,
        writer: Writer = write_full_negative_tiff,
        stable_hasher: StableHasher = _hash_file_stable,
        manifest_writer: ManifestWriter = _write_exclusive_fsynced,
        stream_deleter: StreamDeleter = Path.unlink,
        smear_assessor: SmearAssessor = assess_stopped_transport_smear,
        clipping_measurer: ClippingMeasurer = measure_scan_clipping,
        focus_measurer: FocusMeasurer = measure_focus_detail,
    ) -> None:
        self._contract = PackedCaptureContract() if contract is None else contract
        self._decoder = decoder if decoder is not None else self._decode_default
        self._writer = writer
        self._hash = stable_hasher
        self._write_manifest = manifest_writer
        self._delete_stream = stream_deleter
        self._assess_smear = smear_assessor
        self._measure_clipping = clipping_measurer
        self._measure_focus = focus_measurer

    def _decode_default(self, path: Path) -> tuple[np.ndarray, Mapping[str, object]]:
        return decode_full_records(path, width=self._contract.width, height=self._contract.height)

    def finalize_attempt(
        self,
        attempt: SinglePassAttempt,
        *,
        delete_scratch: bool = True,
    ) -> SinglePassFinalizationResult:
        """Finalize one completed full capture; repeat calls safely resume."""

        if not isinstance(attempt, SinglePassAttempt):
            raise TypeError("attempt must be a SinglePassAttempt")
        if type(delete_scratch) is not bool:
            raise TypeError("delete_scratch must be bool")
        manifest_path, output_paths = _canonical_output_paths(attempt)
        if manifest_path.is_file():
            return self._resume_committed(attempt, manifest_path, output_paths)
        if manifest_path.exists() or manifest_path.is_symlink():
            raise SinglePassIntegrityError(f"manifest path is not a regular file: {manifest_path}")
        if attempt.worker_returncode != 0:
            raise SinglePassIntegrityError(f"worker return code {attempt.worker_returncode} is not complete")

        journal, journal_sha, journal_bytes = _load_json_stable(attempt.journal_path)
        stream_sha, stream_bytes = self._hash(attempt.stream_path)
        frame_evidence, exposure_evidence, scanner_identity = self._validate_completed_capture(
            attempt,
            journal,
            stream_sha=stream_sha,
            stream_bytes=stream_bytes,
        )

        try:
            decoded, raw_layout = self._decoder(attempt.stream_path)
        except Exception as error:
            if isinstance(error, SinglePassWorkflowError):
                raise
            raise SinglePassIntegrityError(f"packed-stream decode refused: {error}") from error
        rgbi = np.asarray(decoded)
        expected_shape = (self._contract.height, self._contract.width, 4)
        if rgbi.shape != expected_shape or rgbi.dtype != np.uint16:
            raise SinglePassIntegrityError(f"decoder returned {rgbi.shape} {rgbi.dtype}, expected {expected_shape} uint16")
        layout = dict(raw_layout)
        if layout.get("padding_validated_records") != self._contract.records:
            raise SinglePassIntegrityError("decoder did not validate padding in every full record")
        if layout.get("rgb_samples_decoded") != 4 or layout.get("ir_planes_transferred") != 1:
            raise SinglePassIntegrityError("decoder did not prove the RGB4x plus IR single-pass layout")

        try:
            smear_assessment = self._assess_smear(
                rgbi[..., :3],
                dpi=self._contract.dpi,
            )
        except Exception as error:
            raise SinglePassIntegrityError(
                f"stopped-transport smear QC could not assess decoded RGB: {error}"
            ) from error
        if not isinstance(smear_assessment, StoppedTransportSmearAssessment):
            raise SinglePassIntegrityError(
                "stopped-transport smear QC returned invalid evidence"
            )
        if smear_assessment.verdict != "clean":
            raise SinglePassIntegrityError(
                "stopped-transport smear QC refused decoded RGB: "
                f"{smear_assessment.verdict}: {smear_assessment.reason}"
            )

        final_stream_sha, final_stream_bytes = self._hash(attempt.stream_path)
        if final_stream_bytes != stream_bytes or not hmac.compare_digest(final_stream_sha, stream_sha):
            raise SinglePassIntegrityError("capture stream changed while it was being decoded")
        _, final_journal_sha, final_journal_bytes = _load_json_stable(attempt.journal_path)
        if final_journal_bytes != journal_bytes or not hmac.compare_digest(final_journal_sha, journal_sha):
            raise SinglePassIntegrityError("capture journal changed while the stream was being decoded")

        upright = np.ascontiguousarray(np.rot90(rgbi, k=1, axes=(0, 1)))
        rgb = upright[..., :3]
        ir = upright[..., 3]
        valid = np.ones(ir.shape, dtype=np.bool_)
        try:
            clipping = self._measure_clipping(rgb)
            focus_detail = self._measure_focus(rgb)
        except Exception as error:
            raise SinglePassIntegrityError(
                f"capture quality telemetry could not assess decoded RGB: {error}"
            ) from error
        if not isinstance(clipping, ScanClippingTelemetry):
            raise SinglePassIntegrityError(
                "capture clipping telemetry returned invalid evidence"
            )
        if not isinstance(focus_detail, FocusDetailTelemetry):
            raise SinglePassIntegrityError(
                "focus-detail telemetry returned invalid evidence"
            )
        clipping_evidence = asdict(clipping)
        # JSON has no tuple type. Normalize before binding the manifest so the
        # in-memory evidence and its durable round-trip remain byte-for-byte
        # comparable.
        clipping_evidence["fractions"] = list(clipping.fractions)
        result = ScanResult(
            rgb=rgb,
            ir=ir,
            dpi=self._contract.dpi,
            device_model=scanner_identity,
            ir_alignment=SplitIrAlignment(mode="identity", dx_px=0.0, dy_px=0.0),
            ir_valid_mask=valid,
        )

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw_output_evidence = self._writer(result, ir_valid_mask=valid, path=output_paths["rgb"])
        except Exception as error:
            raise SinglePassWorkflowError(f"transactional TIFF writer failed: {error}") from error
        output_evidence = self._verify_outputs(raw_output_evidence, output_paths)

        cleanup_checkpoint: dict[str, object] | None = None
        cleanup_checkpoint_evidence: dict[str, object] | None = None
        if delete_scratch:
            cleanup_checkpoint = {
                "version": 1,
                "kind": "negpy.ls5000-scratch-deletion-checkpoint",
                "source_path": attempt.stream_path.name,
                "source_sha256": stream_sha,
                "source_bytes": stream_bytes,
                "policy": "delete only after completion manifest and output hashes reverify",
            }
            checkpoint_path = manifest_path.parent / "scratch-deletion.json"
            checkpoint_payload = _json_bytes(cleanup_checkpoint)
            if checkpoint_path.exists():
                existing, _, _ = _load_json_stable(checkpoint_path)
                if existing != cleanup_checkpoint:
                    raise SinglePassIntegrityError("existing scratch-deletion checkpoint conflicts with this capture")
            else:
                _write_exclusive_fsynced(checkpoint_path, checkpoint_payload)
            checkpoint_sha, checkpoint_bytes = self._hash(checkpoint_path)
            cleanup_checkpoint_evidence = {
                "path": checkpoint_path.name,
                "sha256": checkpoint_sha,
                "bytes": checkpoint_bytes,
            }

        roll_metadata = self._roll_metadata(attempt, journal)
        batch_session = self._batch_session_evidence(attempt)
        capture_evidence: dict[str, object] = {
            "status": "frame-complete" if batch_session is not None else "complete",
            "mode": "full",
            "scanner_identity": scanner_identity,
            "fine_reads": self._contract.records,
            "fine_bytes": self._contract.stream_bytes,
            "unit_released": batch_session is None,
            "plan_sha256": journal["plan_sha256"],
            "capture_engine_sha256": journal["capture_engine_sha256"],
        }
        if batch_session is not None:
            capture_evidence.update(
                session_reservation_retained=True,
                batch_session=batch_session,
                continuation_plan_sha256=journal[
                    "continuation_plan_sha256"
                ],
            )
        manifest: dict[str, object] = {
            "version": WORKFLOW_VERSION,
            "kind": WORKFLOW_KIND,
            "identity": _attempt_identity(attempt),
            "sources": {
                "packed_stream": {
                    "path": attempt.stream_path.name,
                    "sha256": stream_sha,
                    "bytes": stream_bytes,
                },
                "capture_journal": {
                    "path": attempt.journal_path.name,
                    "sha256": journal_sha,
                    "bytes": journal_bytes,
                },
            },
            "capture": capture_evidence,
            "roll_metadata": roll_metadata,
            "frame_evidence": frame_evidence,
            "exposure_evidence": exposure_evidence,
            "record_geometry": {
                "records": self._contract.records,
                "record_bytes": self._contract.record_bytes,
                "stream_bytes": self._contract.stream_bytes,
                "width": self._contract.width,
                "height": self._contract.height,
                "channels": ["R", "G", "B", "IR"],
            },
            "decode_layout": deepcopy(layout),
            "quality_control": {
                "stopped_transport_smear": {
                    "coordinate_space": "scanner-native RGB before storage rotation",
                    "required_verdict": "clean",
                    "assessment": asdict(smear_assessment),
                },
                "capture_clipping": clipping_evidence,
                "focus_detail": asdict(focus_detail),
            },
            "alignment": {
                "mode": "identity",
                "dx_px": 0.0,
                "dy_px": 0.0,
                "basis": "RGB and IR decoded from the same packed scanner traversal",
            },
            "orientation": {
                "storage": "DIY-upright",
                "source_storage": "scanner-native portrait",
                "source_to_storage": "numpy rot90(k=1, axes=(0,1))",
                "downstream_rotation_degrees": 0,
            },
            "ir_valid_mask": {
                "policy": "all true: same-pass planes share complete geometry",
                "valid_pixels": int(valid.size),
                "total_pixels": int(valid.size),
            },
            "outputs": output_evidence,
            "scratch_cleanup": {
                "requested": delete_scratch,
                "checkpoint": cleanup_checkpoint_evidence,
                "source_sha256": stream_sha,
                "source_bytes": stream_bytes,
            },
        }
        _validate_quality_control_evidence(manifest["quality_control"])
        manifest["binding_sha256"] = _manifest_binding(manifest)
        manifest_payload = _json_bytes(manifest)
        self._write_manifest(manifest_path, manifest_payload)

        committed, _, _ = _load_json_stable(manifest_path)
        if committed != manifest:
            raise SinglePassIntegrityError("committed completion manifest differs from the manifest that was verified")
        self._verify_outputs(committed.get("outputs"), output_paths)
        if cleanup_checkpoint_evidence is not None:
            self._verify_cleanup_checkpoint(manifest_path.parent, cleanup_checkpoint_evidence, stream_sha, stream_bytes)

        scratch_deleted = False
        if delete_scratch:
            last_sha, last_bytes = self._hash(attempt.stream_path)
            if last_bytes != stream_bytes or not hmac.compare_digest(last_sha, stream_sha):
                raise SinglePassIntegrityError("capture stream changed before scratch cleanup")
            # Deliberately last: every operation which can invalidate the
            # archive has completed and been re-read before scratch is removed.
            self._delete_stream(attempt.stream_path)
            scratch_deleted = True

        return SinglePassFinalizationResult(
            manifest_path=manifest_path,
            output_paths=output_paths,
            manifest=manifest,
            resumed=False,
            scratch_deleted=scratch_deleted,
        )

    @staticmethod
    def _batch_session_evidence(
        attempt: SinglePassAttempt,
    ) -> dict[str, object] | None:
        if attempt.batch_session_id is None:
            return None
        return {
            "session_id": attempt.batch_session_id,
            "frame_index": attempt.batch_frame_index,
            "frame_total": attempt.batch_frame_total,
            "selected_slots": list(attempt.batch_selected_slots),
        }

    def _validate_completed_capture(
        self,
        attempt: SinglePassAttempt,
        journal: dict[str, Any],
        *,
        stream_sha: str,
        stream_bytes: int,
    ) -> tuple[dict[str, object], dict[str, object], str]:
        contract = self._contract
        for key, expected in (
            ("capture_mode", "full"),
            ("expected_reads", contract.records),
            ("completed_reads", contract.records),
            ("expected_bytes", contract.stream_bytes),
            ("completed_bytes", contract.stream_bytes),
            ("disk_bytes", contract.stream_bytes),
            ("preview_geometry_validated_before_reads", True),
        ):
            _require_exact(journal, key, expected)
        batch_session = self._batch_session_evidence(attempt)
        if batch_session is None:
            _require_exact(journal, "status", "complete")
            _require_exact(journal, "unit_released", True)
            if journal.get("session_reservation_retained") not in (None, False):
                raise SinglePassIntegrityError(
                    "single-frame capture cannot retain a batch reservation"
                )
        else:
            _require_exact(journal, "status", "frame-complete")
            _require_exact(journal, "frame_complete", True)
            _require_exact(journal, "unit_released", False)
            _require_exact(journal, "session_reservation_retained", True)
            _require_exact(journal, "batch_session", batch_session)
            _require_exact(
                journal,
                "continuation_plan_sha256",
                CANONICAL_CONTINUATION_PLAN_SHA256,
            )
        if journal.get("recovery_required") not in (None, "none"):
            raise SinglePassIntegrityError("completed worker journal requests scanner recovery")
        if stream_bytes != contract.stream_bytes:
            raise SinglePassIntegrityError(f"stream has {stream_bytes} bytes, expected {contract.stream_bytes}")
        if journal.get("output") != str(attempt.stream_path):
            raise SinglePassIntegrityError("journal output path does not identify this attempt stream")
        expected_stream_sha = journal.get("output_sha256")
        if not _is_sha256(expected_stream_sha) or not hmac.compare_digest(cast(str, expected_stream_sha), stream_sha):
            raise SinglePassIntegrityError("stream SHA-256 does not match the completed worker journal")
        for key in ("plan_sha256", "capture_engine_sha256"):
            if not _is_sha256(journal.get(key)):
                raise SinglePassIntegrityError(f"journal {key} is missing or malformed")
        identity = journal.get("scanner_identity")
        if type(identity) is not str or not identity.startswith("Nikon LS-5000 ED"):
            raise SinglePassIntegrityError("journal scanner identity is not a Nikon LS-5000 ED")
        _require_exact(journal, "requested_frame", attempt.selected_slot)
        _require_exact(
            journal,
            "requested_boundary_offset_rows",
            attempt.boundary_offset_rows,
        )
        _require_exact(
            journal,
            "applied_boundary_offset_rows",
            attempt.boundary_offset_rows,
        )

        selection = _require_object(journal, "live_frame_selection")
        _require_exact(selection, "frame", attempt.selected_slot)
        selected = _require_object(selection, "selected")
        _require_exact(selected, "frame", attempt.selected_slot)
        native_origin = selected.get("native_origin")
        if type(native_origin) is not int or native_origin < 0:
            raise SinglePassIntegrityError("selected scanner slot has no valid native origin")
        _require_exact(journal, "resolved_lookup_row", selected.get("lookup_row"))
        _require_exact(journal, "resolved_native_origin", native_origin)
        for key in ("preview_sha256", "table_sha256"):
            if not _is_sha256(selection.get(key)):
                raise SinglePassIntegrityError(f"live frame selection {key} is missing or malformed")
        roll_identity, manual_review_approval = self._validated_roll_evidence(
            attempt,
            journal,
            selection,
            selected,
            required=batch_session is not None,
        )

        preflight = self._validated_windows(
            journal.get("fine_set_windows_preflight"),
            native_origin=native_origin,
            require_interleave=False,
            label="fine SET_WINDOW preflight",
        )
        fine = self._validated_windows(
            journal.get("fine_windows"),
            native_origin=native_origin,
            require_interleave=True,
            label="fine GET_WINDOW echo",
        )
        for color_id in EXPECTED_COLOR_IDS:
            expected = {key: value for key, value in fine[color_id].items() if key != "interleave"}
            if preflight[color_id] != expected:
                raise SinglePassIntegrityError(f"fine window {color_id} changed between SET_WINDOW and GET_WINDOW")

        controller = _require_object(journal, "meter_controller_final_result")
        if controller.get("accepted") is not True:
            raise SinglePassIntegrityError("meter controller did not accept the final exposure contract")
        if journal.get("meter_evidence_persisted_before_fine_arm") is not True:
            raise SinglePassIntegrityError("meter evidence was not persisted before the fine scan")
        final_exposures = _require_object(journal, "meter_final_exposures")
        wire_exposures = _require_object(final_exposures, "wire_colors_raw_10ns")
        observed = {str(color_id): fine[color_id]["exposure_raw_10ns"] for color_id in EXPECTED_COLOR_IDS}
        if wire_exposures != observed:
            raise SinglePassIntegrityError("fine exposure echo does not match the accepted meter contract")
        controller_exposures = _require_object(final_exposures, "controller_channels_raw_10ns")
        expected_controller = {
            "R": observed["1"],
            "G": observed["2"],
            "B": observed["3"],
            "IR": observed["9"],
        }
        if controller_exposures != expected_controller or controller.get("final_exposures_raw_10ns") != expected_controller:
            raise SinglePassIntegrityError("meter controller exposure echo is inconsistent")

        frame_evidence: dict[str, object] = {
            "requested_slot": attempt.selected_slot,
            "boundary_offset_rows": attempt.boundary_offset_rows,
            "selected": deepcopy(selected),
            "preview_sha256": selection["preview_sha256"],
            "transport_table_sha256": selection["table_sha256"],
            "detection": deepcopy(selection.get("detection")),
            "transport_mapping": deepcopy(selection.get("transport_mapping")),
        }
        if roll_identity is not None:
            frame_evidence["roll_identity"] = roll_identity
            frame_evidence["manual_review_approval"] = manual_review_approval
        exposure_evidence: dict[str, object] = {
            "controller_final": deepcopy(controller),
            "accepted_contract": deepcopy(final_exposures),
            "fine_set_windows_preflight": [deepcopy(preflight[color]) for color in sorted(preflight)],
            "fine_get_windows": [deepcopy(fine[color]) for color in sorted(fine)],
        }
        return frame_evidence, exposure_evidence, identity

    @staticmethod
    def _validated_roll_evidence(
        attempt: SinglePassAttempt,
        journal: Mapping[str, Any],
        selection: Mapping[str, Any],
        selected: Mapping[str, Any],
        *,
        required: bool,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        raw_identity = selection.get("roll_identity")
        if type(raw_identity) is not dict:
            if required:
                raise SinglePassIntegrityError("batch frame has no roll identity evidence")
            return None, None
        roll_identity = cast(dict[str, Any], raw_identity)
        expected_identity_keys = {
            "comparison",
            "fresh_fingerprint_sha256",
            "reviewed_fingerprint_sha256",
            "selected_slot_comparison",
        }
        if set(roll_identity) != expected_identity_keys:
            raise SinglePassIntegrityError("roll identity evidence keys changed")
        if all(roll_identity.get(key) is None for key in expected_identity_keys):
            if required:
                raise SinglePassIntegrityError("batch frame has no roll identity evidence")
            return None, None

        reviewed_sha = roll_identity.get("reviewed_fingerprint_sha256")
        fresh_sha = roll_identity.get("fresh_fingerprint_sha256")
        if not _is_sha256(reviewed_sha):
            raise SinglePassIntegrityError("reviewed roll fingerprint identity is malformed")
        if not _is_sha256(fresh_sha):
            raise SinglePassIntegrityError("fresh roll fingerprint identity is malformed")
        if journal.get("reviewed_roll_fingerprint_sha256") != reviewed_sha:
            raise SinglePassIntegrityError(
                "reviewed roll fingerprint identity changed between journal fields"
            )

        comparison = roll_identity.get("comparison")
        comparison_keys = {
            "compared_frames",
            "discriminative_frames",
            "frame_start_max_delta_rows",
            "frame_start_median_delta_rows",
            "matches",
            "minimum_discriminative_frames",
            "minimum_visual_log_span",
            "native_origin_max_delta",
            "native_origin_median_delta",
            "preview_height_delta_rows",
            "reason",
            "visual_median_hamming",
            "visual_p90_hamming",
        }
        if type(comparison) is not dict or set(comparison) != comparison_keys:
            raise SinglePassIntegrityError("roll fingerprint comparison is malformed")
        comparison = cast(dict[str, Any], comparison)
        compared = comparison.get("compared_frames")
        discriminative = comparison.get("discriminative_frames")
        minimum_discriminative = comparison.get("minimum_discriminative_frames")
        integer_metrics = (
            comparison.get("preview_height_delta_rows"),
            comparison.get("visual_p90_hamming"),
            comparison.get("frame_start_max_delta_rows"),
            comparison.get("native_origin_max_delta"),
        )
        float_metrics = (
            comparison.get("visual_median_hamming"),
            comparison.get("frame_start_median_delta_rows"),
            comparison.get("native_origin_median_delta"),
            comparison.get("minimum_visual_log_span"),
        )
        if (
            comparison.get("matches") is not True
            or comparison.get("reason") != "matched"
            or type(compared) is not int
            or compared < 1
            or type(discriminative) is not int
            or type(minimum_discriminative) is not int
            or not 1 <= compared <= 40
            or minimum_discriminative
            != min(MIN_DISCRIMINATIVE_FRAME_COUNT, compared)
            or not 1 <= minimum_discriminative <= discriminative <= compared
            or any(type(value) is not int or value < 0 for value in integer_metrics)
            or any(not _finite_number(value) for value in float_metrics)
            or comparison["preview_height_delta_rows"]
            > MAX_PREVIEW_HEIGHT_DELTA_ROWS
            or comparison["visual_median_hamming"]
            > MAX_VISUAL_MEDIAN_HAMMING
            or comparison["visual_p90_hamming"] > MAX_VISUAL_P90_HAMMING
            or comparison["frame_start_median_delta_rows"]
            > MAX_FRAME_START_MEDIAN_DELTA_ROWS
            or comparison["frame_start_max_delta_rows"]
            > MAX_FRAME_START_DELTA_ROWS
            or comparison["native_origin_median_delta"]
            > MAX_NATIVE_ORIGIN_MEDIAN_DELTA
            or comparison["native_origin_max_delta"] > MAX_NATIVE_ORIGIN_DELTA
            or comparison["minimum_visual_log_span"] != MIN_VISUAL_LOG_SPAN
        ):
            raise SinglePassIntegrityError(
                "roll fingerprint comparison did not prove the reviewed roll"
            )

        selected_comparison = roll_identity.get("selected_slot_comparison")
        selected_keys = {
            "fresh_visual_log_span",
            "matches",
            "maximum_visual_hamming",
            "minimum_visual_log_span",
            "reason",
            "reviewed_visual_log_span",
            "slot",
            "visual_hamming",
        }
        if (
            type(selected_comparison) is not dict
            or set(selected_comparison) != selected_keys
        ):
            raise SinglePassIntegrityError(
                "selected-slot roll fingerprint comparison is malformed"
            )
        selected_comparison = cast(dict[str, Any], selected_comparison)
        visual_hamming = selected_comparison.get("visual_hamming")
        maximum_hamming = selected_comparison.get("maximum_visual_hamming")
        minimum_span = selected_comparison.get("minimum_visual_log_span")
        reviewed_span = selected_comparison.get("reviewed_visual_log_span")
        fresh_span = selected_comparison.get("fresh_visual_log_span")
        if (
            selected_comparison.get("matches") is not True
            or selected_comparison.get("reason") != "matched"
            or selected_comparison.get("slot") != attempt.selected_slot
            or type(visual_hamming) is not int
            or type(maximum_hamming) is not int
            or maximum_hamming != MAX_SELECTED_VISUAL_HAMMING
            or not 0 <= visual_hamming <= maximum_hamming
            or not _finite_number(minimum_span)
            or minimum_span != MIN_VISUAL_LOG_SPAN
            or not _finite_number(reviewed_span)
            or not _finite_number(fresh_span)
            or float(reviewed_span) < float(minimum_span)
            or float(fresh_span) < float(minimum_span)
        ):
            raise SinglePassIntegrityError(
                "selected-slot roll fingerprint did not prove the reviewed frame"
            )

        automatic = selected.get("automatic")
        manual_review = selected.get("manual_review")
        if type(automatic) is not bool or type(manual_review) is not bool:
            raise SinglePassIntegrityError(
                "selected frame manual-review state is malformed"
            )
        raw_approval = journal.get("manual_review_approval")
        approval_payload: dict[str, object] | None = None
        if not automatic or manual_review:
            try:
                approval = ManualFrameApproval.from_payload(raw_approval)
            except (TypeError, ValueError) as error:
                raise SinglePassIntegrityError(
                    f"manual review approval is missing or malformed: {error}"
                ) from error
            if (
                approval.reviewed_fingerprint_sha256 != reviewed_sha
                or approval.slot != attempt.selected_slot
                or approval.boundary_offset_rows != attempt.boundary_offset_rows
            ):
                raise SinglePassIntegrityError(
                    "manual review approval does not bind this frame selection"
                )
            approval_payload = approval.to_payload()
        elif raw_approval is not None:
            raise SinglePassIntegrityError(
                "automatic frame unexpectedly carries manual review approval"
            )
        return deepcopy(roll_identity), approval_payload

    def _validated_windows(
        self,
        raw_windows: object,
        *,
        native_origin: int,
        require_interleave: bool,
        label: str,
    ) -> dict[int, dict[str, object]]:
        if type(raw_windows) is not list or len(raw_windows) != 4:
            raise SinglePassIntegrityError(f"journal must contain exactly four {label} windows")
        normalized: dict[int, dict[str, object]] = {}
        fields = ("color_id", "exposure_raw_10ns", "origin", "resolution", "samples", "size")
        for index, raw_window in enumerate(raw_windows):
            if type(raw_window) is not dict:
                raise SinglePassIntegrityError(f"{label} window {index} must be an object")
            window = cast(dict[str, Any], raw_window)
            color_id = window.get("color_id")
            exposure = window.get("exposure_raw_10ns")
            if type(color_id) is not int or color_id in normalized:
                raise SinglePassIntegrityError(f"{label} window {index} has an invalid or duplicate color ID")
            if type(exposure) is not int or exposure <= 0:
                raise SinglePassIntegrityError(f"{label} window {index} has an invalid exposure")
            if window.get("origin") != [0, native_origin]:
                raise SinglePassIntegrityError(f"{label} window {index} does not use the selected slot origin")
            if window.get("resolution") != [self._contract.dpi, self._contract.dpi]:
                raise SinglePassIntegrityError(f"{label} window {index} has an unexpected resolution")
            if window.get("size") != [self._contract.width, self._contract.height] or window.get("samples") != 4:
                raise SinglePassIntegrityError(f"{label} window {index} is not the proven RGBI4x geometry")
            normalized_window = {field: deepcopy(window[field]) for field in fields}
            if require_interleave:
                if window.get("interleave") != 64:
                    raise SinglePassIntegrityError(f"{label} window {index} has an unexpected interleave")
                normalized_window["interleave"] = 64
            normalized[color_id] = normalized_window
        if set(normalized) != EXPECTED_COLOR_IDS:
            raise SinglePassIntegrityError(f"{label} color IDs are not R, G, B, and IR")
        return normalized

    def _roll_metadata(self, attempt: SinglePassAttempt, journal: Mapping[str, Any]) -> dict[str, object]:
        selection = cast(Mapping[str, Any], journal["live_frame_selection"])
        observed = selection.get("frame_count")
        observed_count = observed if type(observed) is int and observed > 0 else None
        nominal = attempt.session.nominal_exposure_count
        warnings: list[str] = []
        if nominal is not None and observed_count is not None and nominal != observed_count:
            warnings.append(f"nominal exposure count {nominal} differs from preview candidate count {observed_count}")
        if nominal is not None and attempt.selected_slot > nominal:
            warnings.append(f"selected scanner slot {attempt.selected_slot} is beyond nominal exposure count {nominal}")
        return {
            "nominal_exposure_count": nominal,
            "observed_candidate_count": observed_count,
            "count_is_advisory": True,
            "selection_model": "explicit scanner-addressable slot in 1..40",
            "warnings": warnings,
        }

    def _verify_outputs(
        self,
        raw_evidence: object,
        paths: Mapping[str, Path],
    ) -> dict[str, dict[str, object]]:
        if type(raw_evidence) is not dict or set(raw_evidence) != set(paths):
            raise SinglePassIntegrityError("TIFF writer returned incomplete artifact evidence")
        evidence_map = cast(dict[str, object], raw_evidence)
        expected_shapes = {
            "rgb": (self._contract.width, self._contract.height, 3),
            "ir": (self._contract.width, self._contract.height),
            "ir_valid_mask": (self._contract.width, self._contract.height),
        }
        expected_dtypes = {"rgb": np.dtype("uint16"), "ir": np.dtype("uint16"), "ir_valid_mask": np.dtype("uint8")}
        verified: dict[str, dict[str, object]] = {}
        for role, path in paths.items():
            item = cast(dict[str, object], evidence_map.get(role))
            if type(item) is not dict:
                raise SinglePassIntegrityError(f"TIFF writer returned invalid {role} evidence")
            expected_sha = item.get("sha256")
            if not _is_sha256(expected_sha):
                raise SinglePassIntegrityError(f"TIFF writer returned an invalid {role} SHA-256")
            actual_sha, actual_bytes = self._hash(path)
            if not hmac.compare_digest(cast(str, expected_sha), actual_sha):
                raise SinglePassIntegrityError(f"committed {role} hash disagrees with writer evidence")
            if item.get("bytes") != actual_bytes or item.get("path") != path.name:
                raise SinglePassIntegrityError(f"committed {role} size or basename disagrees with writer evidence")
            try:
                with tifffile.TiffFile(path) as tiff:
                    series = tiff.series[0]
                    if tuple(series.shape) != expected_shapes[role] or np.dtype(series.dtype) != expected_dtypes[role]:
                        raise SinglePassIntegrityError(f"committed {role} TIFF has unexpected shape or dtype")
                    page = cast(Any, tiff.pages[0])
                    if role == "rgb" and not has_linear_scanner_rgb_marker(cast(Mapping[Any, Any], page.tags)):
                        raise SinglePassIntegrityError("committed RGB TIFF is missing the scanner-linear source contract")
            except tifffile.TiffFileError as error:
                raise SinglePassIntegrityError(f"committed {role} is not a readable TIFF: {error}") from error
            verified[role] = deepcopy(item)
        return verified

    def _verify_cleanup_checkpoint(
        self,
        root: Path,
        raw_evidence: object,
        stream_sha: str,
        stream_bytes: int,
    ) -> None:
        if type(raw_evidence) is not dict:
            raise SinglePassIntegrityError("scratch cleanup checkpoint evidence is missing")
        evidence = cast(dict[str, object], raw_evidence)
        name = evidence.get("path")
        if type(name) is not str or Path(name).name != name:
            raise SinglePassIntegrityError("scratch cleanup checkpoint path is invalid")
        path = root / name
        actual_sha, actual_bytes = self._hash(path)
        if evidence.get("sha256") != actual_sha or evidence.get("bytes") != actual_bytes:
            raise SinglePassIntegrityError("scratch cleanup checkpoint hash or size changed")
        checkpoint, _, _ = _load_json_stable(path)
        if checkpoint.get("source_sha256") != stream_sha or checkpoint.get("source_bytes") != stream_bytes:
            raise SinglePassIntegrityError("scratch cleanup checkpoint identifies a different stream")

    def _resume_committed(
        self,
        attempt: SinglePassAttempt,
        manifest_path: Path,
        output_paths: dict[str, Path],
    ) -> SinglePassFinalizationResult:
        manifest, _, _ = _load_json_stable(manifest_path)
        binding = manifest.get("binding_sha256")
        if not _is_sha256(binding) or not hmac.compare_digest(cast(str, binding), _manifest_binding(manifest)):
            raise SinglePassIntegrityError("existing completion manifest binding no longer verifies")
        identity = manifest.get("identity")
        expected_identity = _attempt_identity(attempt)
        if manifest.get("kind") != WORKFLOW_KIND or manifest.get("version") != WORKFLOW_VERSION or identity != expected_identity:
            raise SinglePassIntegrityError("existing completion manifest does not identify this attempt")
        quality_control = manifest.get("quality_control")
        _validate_quality_control_evidence(quality_control)
        smear_control = (
            quality_control.get("stopped_transport_smear")
            if type(quality_control) is dict
            else None
        )
        smear_evidence = (
            smear_control.get("assessment")
            if type(smear_control) is dict
            else None
        )
        if (
            type(smear_control) is not dict
            or smear_control.get("coordinate_space")
            != "scanner-native RGB before storage rotation"
            or smear_control.get("required_verdict") != "clean"
            or type(smear_evidence) is not dict
            or smear_evidence.get("verdict") != "clean"
        ):
            raise SinglePassIntegrityError(
                "existing completion manifest has no verified stopped-transport smear QC"
            )
        self._verify_outputs(manifest.get("outputs"), output_paths)

        sources = manifest.get("sources")
        if type(sources) is not dict:
            raise SinglePassIntegrityError("existing completion manifest has no source evidence")
        journal_evidence = cast(dict[str, object], sources.get("capture_journal"))
        if type(journal_evidence) is not dict:
            raise SinglePassIntegrityError("existing completion manifest has no journal evidence")
        journal_sha, journal_bytes = _load_json_stable(attempt.journal_path)[1:]
        if journal_evidence.get("sha256") != journal_sha or journal_evidence.get("bytes") != journal_bytes:
            raise SinglePassIntegrityError("capture journal changed after completion")
        stream_evidence = cast(dict[str, object], sources.get("packed_stream"))
        if (
            type(stream_evidence) is not dict
            or not _is_sha256(stream_evidence.get("sha256"))
            or type(stream_evidence.get("bytes")) is not int
        ):
            raise SinglePassIntegrityError("existing completion manifest has invalid stream evidence")
        stream_sha = cast(str, stream_evidence["sha256"])
        stream_bytes = cast(int, stream_evidence["bytes"])

        cleanup = manifest.get("scratch_cleanup")
        if type(cleanup) is not dict or type(cleanup.get("requested")) is not bool:
            raise SinglePassIntegrityError("existing completion manifest has invalid scratch policy")
        requested = cast(bool, cleanup["requested"])
        if requested:
            self._verify_cleanup_checkpoint(manifest_path.parent, cleanup.get("checkpoint"), stream_sha, stream_bytes)

        scratch_exists = attempt.stream_path.is_file()
        if scratch_exists:
            actual_sha, actual_bytes = self._hash(attempt.stream_path)
            if actual_sha != stream_sha or actual_bytes != stream_bytes:
                raise SinglePassIntegrityError("retained capture stream differs from completion evidence")
            if requested:
                self._delete_stream(attempt.stream_path)
                scratch_exists = False
        elif attempt.stream_path.exists() or attempt.stream_path.is_symlink():
            raise SinglePassIntegrityError("capture stream path is not a regular file")
        elif not requested:
            raise SinglePassIntegrityError("completion manifest required retained scratch, but the stream is missing")

        return SinglePassFinalizationResult(
            manifest_path=manifest_path,
            output_paths=output_paths,
            manifest=cast(dict[str, object], manifest),
            resumed=True,
            scratch_deleted=not scratch_exists,
        )


__all__ = [
    "CaptureAttemptResultLike",
    "LS5000SinglePassWorkflow",
    "PackedCaptureContract",
    "SinglePassAttempt",
    "SinglePassFinalizationResult",
    "SinglePassIntegrityError",
    "SinglePassSession",
    "SinglePassWorkflowError",
]
