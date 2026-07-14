"""Offline LS-5000 whole-roll preview and selection state.

The scanner worker owns USB.  This module starts only after a preview attempt
has completed and released the device.  It revalidates the persisted journal,
whole-roll index, and same-traversal transport table before exposing slots to
the application.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import numpy as np

from negpy.infrastructure.scanners.ls5000_single_pass.capture_process import (
    AttemptPaths,
    CaptureAttemptResult,
    CaptureMode,
    CaptureOutcome,
    CaptureRequest,
    ManualFrameApproval,
    ReviewedRollFingerprint,
    build_reviewed_roll_fingerprint,
)
from negpy.infrastructure.scanners.ls5000_single_pass.plan import (
    CANONICAL_PLAN_SHA256,
)
from negpy.infrastructure.scanners.ls5000_single_pass.roll_index import (
    INDEX_BLOCK_BYTES,
    INDEX_ROW_WORDS,
    IndexGeometry,
    NativeFrameOrigin,
    RollDetection,
    TransportMapping,
    TransportRecord,
    decode_full_index_bytes,
    derive_transport_mapping,
    detect_roll_frames,
    parse_live_transport_records_bytes,
    transport_native_origin,
    validate_live_0x8e_bytes,
)
from negpy.services.scanning.roll_preview_controls import (
    ScanMaterial,
    validate_boundary_offset,
)


LS5000_NATIVE_RESOLUTION = 4_000
LS5000_FINE_NATIVE_HEIGHT = 5_959
SA30_ADAPTER_FRAME_CAPACITY = 40
SESSION_VERSION = 1
_PREVIEW_SLOT_SEMANTICS = "scanner-addressable preview slots; not an exposure count"


def _immutable_array(value: np.ndarray) -> np.ndarray:
    """Copy an array onto an immutable bytes buffer."""

    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


class RollSessionError(RuntimeError):
    """A roll session could not be created from the supplied preview."""


class RollSessionIntegrityError(RollSessionError):
    """Persisted preview evidence is missing, changed, or inconsistent."""


class CaptureRoute(StrEnum):
    """Physical acquisition implementation required by a scan recipe."""

    SINGLE_PASS_RGBI4 = "single-pass-rgbi4"
    SANE_RGB4 = "sane-rgb4"


@dataclass(frozen=True)
class CaptureRecipe:
    """Material-specific full-quality capture contract."""

    dpi: int
    bit_depth: int
    rgb_samples: int
    capture_route: CaptureRoute
    capture_ir: bool
    repair_with_ir_after_import: bool


def recipe_for_material(material: ScanMaterial) -> CaptureRecipe:
    """Return the explicit acquisition route for one supported material."""

    if not isinstance(material, ScanMaterial):
        raise TypeError("material must be a ScanMaterial")
    if material is ScanMaterial.COLOR_NEGATIVE:
        return CaptureRecipe(
            dpi=4_000,
            bit_depth=16,
            rgb_samples=4,
            capture_route=CaptureRoute.SINGLE_PASS_RGBI4,
            capture_ir=True,
            repair_with_ir_after_import=True,
        )
    return CaptureRecipe(
        dpi=4_000,
        bit_depth=16,
        rgb_samples=4,
        capture_route=CaptureRoute.SANE_RGB4,
        capture_ir=False,
        repair_with_ir_after_import=False,
    )


@dataclass(frozen=True)
class ArtifactIdentity:
    """One validated, immutable reference to a persisted preview artifact."""

    path: Path
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class ValidatedRollPreview:
    """Decoded whole-roll raster bound to its exact transport-table bytes."""

    preview_artifact: ArtifactIdentity
    table_artifact: ArtifactIdentity
    journal_artifact: ArtifactIdentity
    geometry: IndexGeometry
    usable_rows: int
    rgb: np.ndarray = field(repr=False, compare=False)
    transport_records: tuple[TransportRecord, ...] = field(repr=False)
    decode_report: Mapping[str, object] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.rgb.dtype != np.uint16 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("validated roll preview must be an HxWx3 uint16 raster")
        object.__setattr__(self, "rgb", _immutable_array(self.rgb))


@dataclass(frozen=True)
class RollPreviewSlot:
    """One fixed-order scanner slot and its advisory preview metadata."""

    slot_id: int
    start_boundary_row: int
    end_boundary_row: int
    base_origin: NativeFrameOrigin
    thumbnail: np.ndarray = field(repr=False, compare=False)
    warnings: tuple[str, ...] = ()
    manual_review: bool = False
    boundary_offset_rows: int = 0

    def __post_init__(self) -> None:
        if type(self.slot_id) is not int or self.slot_id < 1:
            raise ValueError("slot_id must be a positive integer")
        if not 0 <= self.start_boundary_row < self.end_boundary_row:
            raise ValueError("slot boundary rows must form a nonempty interval")
        if self.base_origin.frame != self.slot_id:
            raise ValueError("slot transport origin belongs to another slot")
        validate_boundary_offset(self.slot_id, self.boundary_offset_rows)
        warnings = tuple(dict.fromkeys(self.warnings))
        if any(type(item) is not str or not item for item in warnings):
            raise ValueError("slot warnings must be nonempty strings")
        object.__setattr__(self, "warnings", warnings)
        if self.thumbnail.dtype != np.uint16 or self.thumbnail.ndim != 3 or self.thumbnail.shape[2] != 3:
            raise ValueError("slot thumbnail must be an HxWx3 uint16 array")
        object.__setattr__(self, "thumbnail", _immutable_array(self.thumbnail))

    @property
    def boundary_rows(self) -> tuple[int, int]:
        return self.start_boundary_row, self.end_boundary_row


@dataclass(frozen=True)
class RollPreviewSession:
    """Immutable application state for one validated whole-roll traversal."""

    preview: ValidatedRollPreview
    geometry: IndexGeometry
    detection: RollDetection
    mapping: TransportMapping
    slots: tuple[RollPreviewSlot, ...]
    material: ScanMaterial
    recipe: CaptureRecipe
    selected_slots: tuple[int, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.geometry != self.preview.geometry:
            raise ValueError("session geometry differs from its preview")
        if self.recipe != recipe_for_material(self.material):
            raise ValueError("session recipe does not match its film material")
        expected_ids = tuple(range(1, len(self.slots) + 1))
        if tuple(item.slot_id for item in self.slots) != expected_ids:
            raise ValueError("roll slots must be fixed, contiguous, and 1-based")
        selected = _validate_selected_slots(self.selected_slots, len(self.slots))
        object.__setattr__(self, "selected_slots", selected)
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(self.warnings)))

    def resolve_origin(
        self,
        slot_id: int,
        boundary_offset_rows: int = 0,
    ) -> NativeFrameOrigin:
        """Resolve an offset to the exact record from this preview traversal."""

        slot = self._slot(slot_id)
        validate_boundary_offset(slot_id, boundary_offset_rows)
        if boundary_offset_rows == 0:
            return slot.base_origin
        resolved_row = slot.base_origin.lookup_row + boundary_offset_rows
        records = self.preview.transport_records
        if not 0 <= resolved_row < len(records):
            raise RollSessionError(f"slot {slot_id} boundary offset resolves outside the saved 0x8e table")
        record = records[resolved_row]
        if record.row != resolved_row or transport_native_origin(record.code, record.selector) != record.native_origin:
            raise RollSessionIntegrityError(f"slot {slot_id} resolved transport record has an invalid identity")
        return replace(
            slot.base_origin,
            lookup_row=record.row,
            code=record.code,
            selector=record.selector,
            native_origin=record.native_origin,
            method=f"{slot.base_origin.method}+operator-boundary-offset",
            automatic=False,
        )

    def reviewed_fingerprint(self) -> ReviewedRollFingerprint:
        """Bind the exact reviewed artifacts to a reread-tolerant roll identity."""

        return build_reviewed_roll_fingerprint(
            self.preview.rgb,
            frame_intervals=tuple(slot.boundary_rows for slot in self.slots),
            frame_native_origins=tuple(
                slot.base_origin.native_origin for slot in self.slots
            ),
            source_preview_sha256=self.preview.preview_artifact.sha256,
            source_table_sha256=self.preview.table_artifact.sha256,
        )

    def approve_manual_origin(
        self,
        slot_id: int,
        boundary_offset_rows: int = 0,
    ) -> ManualFrameApproval:
        """Create an immutable receipt for one visually reviewed manual origin."""

        slot = self._slot(slot_id)
        if slot.base_origin.automatic or not slot.base_origin.manual_review:
            raise ValueError(f"slot {slot_id} does not require manual review")
        thumbnail = reload_thumbnail(
            self.preview,
            slot,
            boundary_offset_rows,
        )
        origin = self.resolve_origin(slot_id, boundary_offset_rows)
        thumbnail_digest = hashlib.sha256()
        thumbnail_digest.update(str(thumbnail.shape).encode("ascii"))
        thumbnail_digest.update(thumbnail.dtype.str.encode("ascii"))
        thumbnail_digest.update(np.ascontiguousarray(thumbnail).tobytes())
        reasons = tuple(
            dict.fromkeys((*slot.warnings, *slot.base_origin.review_reasons))
        )
        if not reasons:
            reasons = ("transport-origin-manual-review",)
        return ManualFrameApproval(
            reviewed_fingerprint_sha256=self.reviewed_fingerprint().binding_sha256,
            slot=slot_id,
            boundary_offset_rows=boundary_offset_rows,
            thumbnail_sha256=thumbnail_digest.hexdigest(),
            reviewed_lookup_row=origin.lookup_row,
            reviewed_native_origin=origin.native_origin,
            review_reasons=reasons,
        )

    def validate_manual_approval(
        self,
        approval: ManualFrameApproval,
        *,
        slot_id: int,
        boundary_offset_rows: int,
    ) -> bool:
        """Return whether an approval exactly matches this reviewed thumbnail."""

        if not isinstance(approval, ManualFrameApproval):
            return False
        return approval == self.approve_manual_origin(
            slot_id,
            boundary_offset_rows,
        )

    def with_material(self, material: ScanMaterial) -> RollPreviewSession:
        """Return this preview with the material's explicit capture recipe."""

        return replace(
            self,
            material=material,
            recipe=recipe_for_material(material),
        )

    def with_selected_slots(
        self,
        selected_slots: Iterable[int],
    ) -> RollPreviewSession:
        """Return a session with an explicit ordered subset of physical slots."""

        selected = _validate_selected_slots(selected_slots, len(self.slots))
        return replace(self, selected_slots=selected)

    def with_boundary_offset(
        self,
        slot_id: int,
        boundary_offset_rows: int,
    ) -> RollPreviewSession:
        """Re-crop one saved thumbnail and persist its exact operator offset."""

        slot = self._slot(slot_id)
        thumbnail = reload_thumbnail(
            self.preview,
            slot,
            boundary_offset_rows,
        )
        updated = replace(
            slot,
            thumbnail=thumbnail,
            boundary_offset_rows=boundary_offset_rows,
        )
        slots = list(self.slots)
        slots[slot_id - 1] = updated
        return replace(self, slots=tuple(slots))

    def to_json(self) -> str:
        """Serialize operator state plus content identities, never pixel arrays."""

        payload = {
            "version": SESSION_VERSION,
            "journal": {
                "path": str(self.preview.journal_artifact.path),
                "sha256": self.preview.journal_artifact.sha256,
            },
            "preview_sha256": self.preview.preview_artifact.sha256,
            "table_sha256": self.preview.table_artifact.sha256,
            "slot_count": len(self.slots),
            "expected_frame_count": self.detection.expected_frame_count,
            "material": self.material.value,
            "selected_slots": list(self.selected_slots),
            "boundary_offsets": [item.boundary_offset_rows for item in self.slots],
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> RollPreviewSession:
        """Restore state only after revalidating and re-decoding source bytes."""

        session = _restore_roll_preview_session(payload)
        if not isinstance(session, cls):
            raise AssertionError("roll-session restore returned the wrong type")
        return session

    def _slot(self, slot_id: int) -> RollPreviewSlot:
        if type(slot_id) is not int or not 1 <= slot_id <= len(self.slots):
            raise ValueError(f"unknown roll slot: {slot_id!r}")
        return self.slots[slot_id - 1]


def _validate_selected_slots(values: Iterable[int], slot_count: int) -> tuple[int, ...]:
    selected = tuple(values)
    if any(type(item) is not int or not 1 <= item <= slot_count for item in selected):
        raise ValueError(f"selected slots must be integers in 1..{slot_count}")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("selected slots must be unique and increasing")
    return selected


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_file_stable(path: Path) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RollSessionIntegrityError(f"artifact is not a regular file: {path}")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise RollSessionIntegrityError(f"could not read artifact {path}: {error}") from error

    def identity(item: Any) -> tuple[int, int, int, int]:
        return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns

    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise RollSessionIntegrityError(f"artifact changed while it was read: {path}")
    return payload


def _load_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = _read_file_stable(path)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RollSessionIntegrityError(f"invalid JSON artifact {path}: {error}") from error
    if type(value) is not dict:
        raise RollSessionIntegrityError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value), payload


def _artifact_path(root: Path, value: object, label: str) -> Path:
    if type(value) is not str:
        raise RollSessionIntegrityError(f"{label} artifact path is missing")
    raw = Path(value)
    if not raw.is_absolute():
        raise RollSessionIntegrityError(f"{label} artifact path is not absolute")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise RollSessionIntegrityError(f"{label} artifact path is unavailable: {error}") from error
    if resolved != raw or not resolved.is_relative_to(root):
        raise RollSessionIntegrityError(f"{label} artifact path escapes or aliases the attempt directory")
    return resolved


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if type(value) is not dict:
        raise RollSessionIntegrityError(f"preview journal {key} is not an object")
    return cast(dict[str, Any], value)


def _require_exact(parent: Mapping[str, Any], key: str, expected: object) -> None:
    if parent.get(key) != expected:
        raise RollSessionIntegrityError(f"preview journal {key}={parent.get(key)!r}, expected {expected!r}")


def _validate_preview_result(
    attempt: CaptureAttemptResult,
) -> tuple[
    dict[str, Any],
    ArtifactIdentity,
    bytes,
    ArtifactIdentity,
    bytes,
    ArtifactIdentity,
    int,
]:
    if not isinstance(attempt, CaptureAttemptResult):
        raise TypeError("attempt must be a CaptureAttemptResult")
    if (
        attempt.outcome is not CaptureOutcome.COMPLETE
        or attempt.request.mode is not CaptureMode.PREVIEW
        or attempt.request.selected_slot is not None
        or attempt.request.boundary_offset_rows != 0
        or attempt.returncode != 0
        or attempt.journal is None
        or attempt.journal_error is not None
    ):
        raise RollSessionIntegrityError("roll session requires one COMPLETE preview-only capture attempt")
    try:
        root = attempt.paths.directory.resolve(strict=True)
    except OSError as error:
        raise RollSessionIntegrityError(f"preview attempt directory is unavailable: {error}") from error
    raw_journal_path = attempt.paths.journal
    if not raw_journal_path.is_absolute() or raw_journal_path.is_symlink():
        raise RollSessionIntegrityError("preview journal path is an alias")
    journal_path = _artifact_path(root, str(raw_journal_path), "journal")
    journal, journal_bytes = _load_json_object(journal_path)
    if journal != attempt.journal:
        raise RollSessionIntegrityError("persisted preview journal differs from the validated attempt result")
    for key, expected in (
        ("status", "complete"),
        ("capture_mode", "preview-only"),
        ("requested_frame", None),
        ("requested_boundary_offset_rows", 0),
        ("expected_frame_count", None),
        ("expected_reads", 0),
        ("completed_reads", 0),
        ("expected_bytes", 0),
        ("completed_bytes", 0),
        ("disk_bytes", 0),
        ("unit_released", True),
        ("plan_sha256", CANONICAL_PLAN_SHA256),
        ("preview_geometry_validated_before_reads", True),
    ):
        _require_exact(journal, key, expected)
    if journal.get("scanner_identity") != "Nikon LS-5000 ED 1.03":
        raise RollSessionIntegrityError("preview journal is not from the proven LS-5000 firmware")
    if not _is_sha256(journal.get("capture_engine_sha256")):
        raise RollSessionIntegrityError("preview journal capture-engine hash is missing")

    output_path = _artifact_path(root, journal.get("output"), "output")
    if output_path != attempt.paths.output.resolve():
        raise RollSessionIntegrityError("preview journal output path differs from the attempt")
    output = _read_file_stable(output_path)
    if output or journal.get("output_sha256") != hashlib.sha256(output).hexdigest():
        raise RollSessionIntegrityError("preview-only output file is not the recorded empty artifact")

    artifacts = _require_mapping(journal, "live_index_artifacts")
    if set(artifacts) != {"mapping", "preview", "table"}:
        raise RollSessionIntegrityError("preview journal has an unexpected live-index artifact set")
    preview_path = _artifact_path(root, artifacts.get("preview"), "preview")
    table_path = _artifact_path(root, artifacts.get("table"), "table")
    mapping_path = _artifact_path(root, artifacts.get("mapping"), "mapping")
    preview = _read_file_stable(preview_path)
    table = _read_file_stable(table_path)
    receipt, _mapping_bytes = _load_json_object(mapping_path)
    evidence = _require_mapping(journal, "live_index_evidence")
    journal_receipt = _require_mapping(journal, "preview_only_receipt")
    if receipt != journal_receipt:
        raise RollSessionIntegrityError("persisted preview receipt differs from the journal")
    preview_sha256 = hashlib.sha256(preview).hexdigest()
    table_sha256 = hashlib.sha256(table).hexdigest()
    expected_receipt = {
        "status": "preview-only-complete",
        "slot_capacity_hint": journal_receipt.get("slot_capacity_hint"),
        "slot_capacity_semantics": _PREVIEW_SLOT_SEMANTICS,
        "preview_bytes": len(preview),
        "preview_sha256": preview_sha256,
        "table_bytes": len(table),
        "table_sha256": table_sha256,
        "frame_detection": "deferred-offline",
    }
    if receipt != expected_receipt:
        raise RollSessionIntegrityError("preview receipt does not bind the saved artifacts")
    if evidence != {
        "status": "persisted-before-frame-detection",
        "preview_bytes": len(preview),
        "preview_sha256": preview_sha256,
        "table_bytes": len(table),
        "table_sha256": table_sha256,
    }:
        raise RollSessionIntegrityError("preview journal evidence does not bind the saved artifacts")
    slot_capacity = journal_receipt.get("slot_capacity_hint")
    startup = _require_mapping(journal, "live_startup_0x8f")
    if (
        type(slot_capacity) is not int
        or slot_capacity != SA30_ADAPTER_FRAME_CAPACITY
        or startup.get("count") != SA30_ADAPTER_FRAME_CAPACITY
    ):
        raise RollSessionIntegrityError(
            "preview does not contain matching live proof of an "
            "SA-30-compatible 40-slot adapter"
        )

    return (
        journal,
        ArtifactIdentity(preview_path, len(preview), preview_sha256),
        preview,
        ArtifactIdentity(table_path, len(table), table_sha256),
        table,
        ArtifactIdentity(journal_path, len(journal_bytes), hashlib.sha256(journal_bytes).hexdigest()),
        slot_capacity,
    )


def _derive_geometry(journal: Mapping[str, Any], preview_bytes: int) -> IndexGeometry:
    windows = journal.get("preview_windows")
    if type(windows) is not list or len(windows) != 3:
        raise RollSessionIntegrityError("preview journal must contain exactly three RGB windows")
    normalized: list[dict[str, Any]] = []
    for value in windows:
        if type(value) is not dict:
            raise RollSessionIntegrityError("preview window is not an object")
        window = cast(dict[str, Any], value)
        if set(window) != {"color_id", "resolution", "origin", "size", "bit_depth"}:
            raise RollSessionIntegrityError("preview window has an unexpected schema")
        normalized.append(window)
    if [item["color_id"] for item in normalized] != [1, 2, 3]:
        raise RollSessionIntegrityError("preview windows are not ordered RGB")
    first = normalized[0]
    for item in normalized[1:]:
        if {key: item[key] for key in item if key != "color_id"} != {key: first[key] for key in first if key != "color_id"}:
            raise RollSessionIntegrityError("preview RGB windows have inconsistent geometry")
    if first["resolution"] != [97, 97] or first["origin"] != [0, 0] or first["size"] != [3_946, 250_278] or first["bit_depth"] != 16:
        raise RollSessionIntegrityError("preview window is not the proven LS-5000 roll index")
    requested_resolution = 97
    pitch = LS5000_NATIVE_RESOLUTION // requested_resolution
    row_bytes = INDEX_ROW_WORDS * 2
    if preview_bytes % INDEX_BLOCK_BYTES or preview_bytes % row_bytes:
        raise RollSessionIntegrityError("preview byte length is not a complete block allocation")
    height = preview_bytes // row_bytes
    native_width, native_height = cast(list[int], first["size"])
    geometry = IndexGeometry(
        requested_resolution=requested_resolution,
        native_resolution=LS5000_NATIVE_RESOLUTION,
        pitch=pitch,
        native_width=native_width,
        native_height=native_height,
        width=native_width // pitch,
        height=height,
        block_bytes=INDEX_BLOCK_BYTES,
        expected_stream_bytes=preview_bytes,
    )
    if geometry.width != 96 or geometry.height != geometry.native_height // geometry.pitch or geometry.height % 2:
        raise RollSessionIntegrityError("preview geometry does not match the LS-5000 RGB96 allocation")
    return geometry


def _thumbnail(rgb: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, min(start, len(rgb)))
    end = max(start, min(end, len(rgb)))
    if end <= start:
        raise RollSessionError("preview slot crop is empty")
    return _immutable_array(np.asarray(rgb[start:end, :, :], dtype=np.uint16))


def reload_thumbnail(
    preview: ValidatedRollPreview,
    slot: RollPreviewSlot,
    boundary_offset_rows: int,
) -> np.ndarray:
    """Re-crop one slot from the saved whole-roll index at an exact 0x8e row.

    This does not shift an already-rendered thumbnail.  The selected offset is
    first resolved through the raw table captured during the same traversal;
    the original decoded RGB96 preview is then cropped again at that row delta.
    """

    if not isinstance(preview, ValidatedRollPreview):
        raise TypeError("preview must be a ValidatedRollPreview")
    if not isinstance(slot, RollPreviewSlot):
        raise TypeError("slot must be a RollPreviewSlot")
    validate_boundary_offset(slot.slot_id, boundary_offset_rows)
    resolved_row = slot.base_origin.lookup_row + boundary_offset_rows
    records = preview.transport_records
    if not 0 <= resolved_row < len(records):
        raise RollSessionError(f"slot {slot.slot_id} boundary offset resolves outside the saved 0x8e table")
    record = records[resolved_row]
    if record.row != resolved_row or transport_native_origin(record.code, record.selector) != record.native_origin:
        raise RollSessionIntegrityError(f"slot {slot.slot_id} resolved transport record has an invalid identity")
    row_delta = record.row - slot.base_origin.lookup_row
    start = slot.start_boundary_row + row_delta
    end = slot.end_boundary_row + row_delta
    if start < 0 or end > len(preview.rgb):
        raise RollSessionError(f"slot {slot.slot_id} boundary offset lies outside the saved preview")
    return _thumbnail(preview.rgb, start, end)


def _slot_warnings(
    slot_id: int,
    interval: Any,
    origin: NativeFrameOrigin,
    detection: RollDetection,
) -> tuple[str, ...]:
    warnings = [*interval.review_reasons, *origin.review_reasons]
    if detection.content_end_candidates:
        first_end = min(detection.content_end_candidates)
        last_end = max(detection.content_end_candidates)
        if len(detection.content_end_candidates) > 1 and first_end <= slot_id <= last_end:
            warnings.append("ambiguous-content-tail-boundary")
        if slot_id > last_end:
            warnings.append("beyond-advisory-content-end")
    return tuple(dict.fromkeys(warnings))


def build_roll_preview_session(
    attempt: CaptureAttemptResult,
    *,
    material: ScanMaterial = ScanMaterial.COLOR_NEGATIVE,
    selected_slots: Iterable[int] = (),
    expected_frame_count: int | None = None,
) -> RollPreviewSession:
    """Validate and decode a completed worker preview into fixed-order slots."""

    if not isinstance(material, ScanMaterial):
        raise TypeError("material must be a ScanMaterial")
    (
        journal,
        preview_artifact,
        preview_bytes,
        table_artifact,
        table_bytes,
        journal_artifact,
        capacity,
    ) = _validate_preview_result(attempt)
    geometry = _derive_geometry(journal, len(preview_bytes))
    validated_table, usable_rows = validate_live_0x8e_bytes(
        table_bytes,
        geometry.height,
    )
    rgb, known, decode_report = decode_full_index_bytes(
        preview_bytes,
        geometry,
        usable_rows=usable_rows,
    )
    detection = detect_roll_frames(
        rgb,
        known,
        nominal_frame_rows=LS5000_FINE_NATIVE_HEIGHT // geometry.pitch,
        expected_frame_count=expected_frame_count,
    )
    if detection.alignment_confidence == "low":
        raise RollSessionError("roll preview physical alignment confidence is low")
    records = parse_live_transport_records_bytes(
        validated_table,
        maximum_rows=geometry.height,
    )
    mapping = derive_transport_mapping(
        detection.boundaries,
        len(detection.intervals),
        records,
    )
    slot_count = min(capacity, len(detection.intervals), len(mapping.origins))
    if slot_count < 1:
        raise RollSessionError("roll preview produced no scanner-addressable slots")
    preview = ValidatedRollPreview(
        preview_artifact=preview_artifact,
        table_artifact=table_artifact,
        journal_artifact=journal_artifact,
        geometry=geometry,
        usable_rows=usable_rows,
        rgb=rgb,
        transport_records=records,
        decode_report=decode_report,
    )
    slots = []
    for interval, origin in zip(
        detection.intervals[:slot_count],
        mapping.origins[:slot_count],
    ):
        warnings = _slot_warnings(interval.frame, interval, origin, detection)
        slots.append(
            RollPreviewSlot(
                slot_id=interval.frame,
                start_boundary_row=interval.start_row,
                end_boundary_row=interval.end_row,
                base_origin=origin,
                thumbnail=_thumbnail(rgb, interval.start_row, interval.end_row),
                warnings=warnings,
                manual_review=bool(interval.manual_review or origin.manual_review or warnings),
            )
        )
    selected = _validate_selected_slots(selected_slots, len(slots))
    return RollPreviewSession(
        preview=preview,
        geometry=geometry,
        detection=detection,
        mapping=mapping,
        slots=tuple(slots),
        material=material,
        recipe=recipe_for_material(material),
        selected_slots=selected,
        warnings=tuple(detection.warnings),
    )


def _restore_roll_preview_session(payload: str) -> RollPreviewSession:
    if type(payload) is not str:
        raise TypeError("roll session JSON must be a string")
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise RollSessionIntegrityError(f"invalid roll session JSON: {error}") from error
    if type(value) is not dict:
        raise RollSessionIntegrityError("roll session JSON must be an object")
    state = cast(dict[str, Any], value)
    expected_keys = {
        "version",
        "journal",
        "preview_sha256",
        "table_sha256",
        "slot_count",
        "expected_frame_count",
        "material",
        "selected_slots",
        "boundary_offsets",
    }
    if set(state) != expected_keys or state.get("version") != SESSION_VERSION:
        raise RollSessionIntegrityError("roll session JSON has an unsupported schema")
    journal_identity = state.get("journal")
    if type(journal_identity) is not dict or set(journal_identity) != {"path", "sha256"}:
        raise RollSessionIntegrityError("roll session journal identity is malformed")
    identity = cast(dict[str, Any], journal_identity)
    if type(identity.get("path")) is not str or not _is_sha256(identity.get("sha256")):
        raise RollSessionIntegrityError("roll session journal identity is malformed")
    journal_path = Path(identity["path"])
    if not journal_path.is_absolute():
        raise RollSessionIntegrityError("roll session journal path is not absolute")
    try:
        resolved_journal = journal_path.resolve(strict=True)
    except OSError as error:
        raise RollSessionIntegrityError(f"roll session journal is unavailable: {error}") from error
    if resolved_journal != journal_path:
        raise RollSessionIntegrityError("roll session journal path aliases another path")
    journal, journal_bytes = _load_json_object(journal_path)
    if hashlib.sha256(journal_bytes).hexdigest() != identity["sha256"]:
        raise RollSessionIntegrityError("roll session journal changed since it was saved")
    root = journal_path.parent
    output_value = journal.get("output")
    if type(output_value) is not str:
        raise RollSessionIntegrityError("roll session journal output path is missing")
    output_path = Path(output_value)
    paths = AttemptPaths(
        directory=root,
        output=output_path,
        journal=journal_path,
        plan=root / "replay-first-rgbi4-plan.jsonl",
        manifest=root / "replay-first-rgbi4-manifest.json",
        stdout=root / "stdout.txt",
        stderr=root / "stderr.txt",
    )
    attempt = CaptureAttemptResult(
        outcome=CaptureOutcome.COMPLETE,
        request=CaptureRequest(mode=CaptureMode.PREVIEW),
        paths=paths,
        argv=(),
        returncode=0,
        stdout="",
        stderr="",
        journal=journal,
    )
    material_value = state.get("material")
    try:
        material = ScanMaterial(material_value)
    except (TypeError, ValueError) as error:
        raise RollSessionIntegrityError("roll session material is unsupported") from error
    selected_value = state.get("selected_slots")
    offsets_value = state.get("boundary_offsets")
    if type(selected_value) is not list or type(offsets_value) is not list:
        raise RollSessionIntegrityError("roll session selections or offsets are malformed")
    expected_frame_count = state.get("expected_frame_count")
    try:
        session = build_roll_preview_session(
            attempt,
            material=material,
            selected_slots=cast(list[int], selected_value),
            expected_frame_count=cast(int | None, expected_frame_count),
        )
    except (TypeError, ValueError) as error:
        raise RollSessionIntegrityError(f"roll session state is invalid: {error}") from error
    if (
        state.get("slot_count") != len(session.slots)
        or state.get("preview_sha256") != session.preview.preview_artifact.sha256
        or state.get("table_sha256") != session.preview.table_artifact.sha256
        or len(offsets_value) != len(session.slots)
    ):
        raise RollSessionIntegrityError("roll session source identities or slot geometry changed")
    for slot_id, offset in enumerate(offsets_value, start=1):
        if type(offset) is not int:
            raise RollSessionIntegrityError("roll session boundary offset is not an integer")
        if offset:
            try:
                session = session.with_boundary_offset(slot_id, offset)
            except (TypeError, ValueError) as error:
                raise RollSessionIntegrityError(f"roll session boundary offset is invalid: {error}") from error
    return session


__all__ = [
    "ArtifactIdentity",
    "CaptureRecipe",
    "CaptureRoute",
    "RollPreviewSession",
    "RollPreviewSlot",
    "RollSessionError",
    "RollSessionIntegrityError",
    "ValidatedRollPreview",
    "build_roll_preview_session",
    "reload_thumbnail",
    "recipe_for_material",
]
