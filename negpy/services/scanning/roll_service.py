"""Two-pass roll scanning with explicit human approval before fine scans.

The workflow deliberately treats a stop request as a *between-transfer* stop.
The event supplied by the UI is never passed into an in-flight scanner read;
this matters for film scanners that can be left in an unknown transport state
when a process interrupts a frame transfer.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from hashlib import sha256
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import ceil, isfinite
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence, cast
from uuid import uuid4

import numpy as np
import tifffile

from negpy.infrastructure.scanners.params import RegisteredScanGeometry, ScanParams
from negpy.infrastructure.scanners.result import SplitIrAlignment
from negpy.services.scanning.service import ScannerService
from negpy.services.scanning.quality import (
    assess_stopped_transport_smear,
    inspect_tiff_payload,
    measure_focus_detail,
    measure_scan_clipping,
    split_alignment_metrics_confident,
)
from negpy.services.scanning.writer import write_tiff_16bit
from negpy.services.scanning.provenance import PlanIdentity, canonical_semantic_sha256

if TYPE_CHECKING:
    from negpy.services.scanning.full_negative_workflow import FullNegativeWorkflowResult


PLAN_FILENAME = "roll-scan.json"
PLAN_VERSION = 3
FINE_QC_VERSION = 1


class RollStage(StrEnum):
    WIDE_PREVIEW = "wide_preview"
    ALIGNED_PREVIEW = "aligned_preview"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    FINE_SCANNING = "fine_scanning"
    COMPLETE = "complete"
    STOPPED = "stopped"


@dataclass(frozen=True)
class AlignmentVerification:
    """Measured leading margin in an aligned preview."""

    leading_margin_rows: int | None
    target_margin_rows: int
    tolerance_rows: int
    confidence: str

    @property
    def passed(self) -> bool:
        return (
            self.leading_margin_rows is not None
            and self.confidence == "high"
            and abs(self.leading_margin_rows - self.target_margin_rows) <= self.tolerance_rows
        )


@dataclass(frozen=True)
class FrameRegistration:
    """Measured bounds and their coupled scanner geometry for one exposure."""

    frame: int
    target_start_row: int
    usable_tail_row: float
    confidence: str
    geometry: RegisteredScanGeometry


@dataclass(frozen=True)
class PreviewScanRecipe:
    """All caller-controlled preview settings that affect safe resume."""

    dpi: int
    depth: int
    capture_ir: bool
    autofocus: bool
    samples_per_scan: int
    auto_exposure: bool

    def __post_init__(self) -> None:
        _validate_recipe_values(self)

    @classmethod
    def from_params(cls, params: ScanParams) -> PreviewScanRecipe:
        return cls(
            dpi=params.dpi,
            depth=params.depth,
            capture_ir=params.capture_ir,
            autofocus=params.autofocus,
            samples_per_scan=params.samples_per_scan,
            auto_exposure=params.auto_exposure,
        )


@dataclass(frozen=True)
class FineScanRecipe:
    """Quality-affecting fine-scan settings persisted for safe resume."""

    dpi: int
    depth: int
    capture_ir: bool
    autofocus: bool
    samples_per_scan: int
    auto_exposure: bool

    def __post_init__(self) -> None:
        _validate_recipe_values(self)

    @classmethod
    def from_params(cls, params: ScanParams) -> FineScanRecipe:
        return cls(
            dpi=params.dpi,
            depth=params.depth,
            capture_ir=params.capture_ir,
            autofocus=params.autofocus,
            samples_per_scan=params.samples_per_scan,
            auto_exposure=params.auto_exposure,
        )


def _validate_recipe_values(recipe: PreviewScanRecipe | FineScanRecipe) -> None:
    if type(recipe.dpi) is not int or recipe.dpi < 1:
        raise ValueError("scan recipe dpi must be a positive integer")
    if type(recipe.depth) is not int or recipe.depth not in (8, 16):
        raise ValueError("scan recipe depth must be 8 or 16")
    if type(recipe.samples_per_scan) is not int or recipe.samples_per_scan < 1:
        raise ValueError("scan recipe samples_per_scan must be a positive integer")
    for field in ("capture_ir", "autofocus", "auto_exposure"):
        if type(getattr(recipe, field)) is not bool:
            raise ValueError(f"scan recipe {field} must be a boolean")


@dataclass(frozen=True)
class RollFrameRecord:
    frame: int
    wide_path: str | None = None
    wide_rgb_shape: tuple[int, ...] | None = None
    wide_dtype: str | None = None
    wide_sha256: str | None = None
    wide_dpi: int | None = None
    registration: FrameRegistration | None = None
    aligned_path: str | None = None
    aligned_rgb_shape: tuple[int, ...] | None = None
    aligned_dtype: str | None = None
    aligned_sha256: str | None = None
    aligned_dpi: int | None = None
    verification: AlignmentVerification | None = None
    fine_path: str | None = None
    fine_rgb_shape: tuple[int, ...] | None = None
    fine_ir_shape: tuple[int, ...] | None = None
    fine_dtype: str | None = None
    fine_ir_dtype: str | None = None
    fine_dpi: int | None = None
    fine_recipe: FineScanRecipe | None = None


@dataclass(frozen=True)
class CalibrationPreviewRecord:
    """Exact decoded preview pixels used only as roll-wide calibration context."""

    frame: int
    sha256: str
    shape: tuple[int, ...]
    dtype: str
    dpi: int


@dataclass(frozen=True)
class RollScanPlan:
    identity: PlanIdentity
    device_id: str
    stage: RollStage
    approved: bool
    preview_dpi: int
    preview_depth: int
    preview_recipe: PreviewScanRecipe
    registration_signature: Mapping[str, object]
    visual_override_frames: tuple[int, ...]
    calibration_context: tuple[CalibrationPreviewRecord, ...]
    frames: tuple[RollFrameRecord, ...]
    version: int = PLAN_VERSION

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        return payload


class RollRegistration(Protocol):
    """Array-level registration policy, kept independent of scanner I/O."""

    @property
    def preview_dpi(self) -> int: ...

    @property
    def minimum_preview_count(self) -> int: ...

    @property
    def registration_signature(self) -> Mapping[str, object]: ...

    def calibrate(self, previews: Mapping[int, np.ndarray]) -> Mapping[int, FrameRegistration]: ...

    def verify(
        self,
        frame: int,
        preview: np.ndarray,
        registration: FrameRegistration,
    ) -> AlignmentVerification: ...


ProgressCallback = Callable[[RollStage, int, float], None]


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Durably replace a plan without exposing a partially-written JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r}")


def _exact_keys(value: object, keys: set[str], field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    payload = cast(dict[str, object], value)
    missing = keys - payload.keys()
    extra = payload.keys() - keys
    if missing or extra:
        raise ValueError(f"{field} keys differ (missing={sorted(missing)}, extra={sorted(extra)})")
    return payload


def _strict_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _strict_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _strict_string(value: object, field: str, *, nonempty: bool = False) -> str:
    if type(value) is not str or (nonempty and not value):
        raise ValueError(f"{field} must be {'a nonempty' if nonempty else 'a'} string")
    return value


def _finite_number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite number")
    number = cast(int | float, value)
    if not isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return float(number)


def _optional_path(value: object, field: str) -> str | None:
    if value is None:
        return None
    path = _strict_string(value, field, nonempty=True)
    candidate = Path(path)
    windows_candidate = PureWindowsPath(path)
    if candidate.is_absolute() or windows_candidate.is_absolute() or ".." in candidate.parts or ".." in windows_candidate.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path


def _optional_shape(value: object, field: str, *, channels: int | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if type(value) is not list:
        raise ValueError(f"{field} must be an array")
    expected_dimensions = 2 if channels is None else 3
    if len(value) != expected_dimensions:
        raise ValueError(f"{field} must have {expected_dimensions} dimensions")
    shape = tuple(_strict_int(item, f"{field}[{index}]", minimum=1) for index, item in enumerate(value))
    if channels is not None and shape[-1] != channels:
        raise ValueError(f"{field} must have {channels} channels")
    return shape


def _optional_dtype(value: object, field: str) -> str | None:
    if value is None:
        return None
    dtype = _strict_string(value, field, nonempty=True)
    if dtype not in {"uint8", "uint16"}:
        raise ValueError(f"{field} is not a supported scanner dtype")
    return dtype


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    digest = _strict_string(value, field, nonempty=True)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(
        json.dumps(
            {"shape": list(contiguous.shape), "dtype": contiguous.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


_RECIPE_KEYS = {
    "dpi",
    "depth",
    "capture_ir",
    "autofocus",
    "samples_per_scan",
    "auto_exposure",
}


def _parse_recipe(value: object, field: str, recipe_type):
    payload = _exact_keys(value, _RECIPE_KEYS, field)
    depth = _strict_int(payload["depth"], f"{field}.depth")
    if depth not in (8, 16):
        raise ValueError(f"{field}.depth must be 8 or 16")
    return recipe_type(
        dpi=_strict_int(payload["dpi"], f"{field}.dpi", minimum=1),
        depth=depth,
        capture_ir=_strict_bool(payload["capture_ir"], f"{field}.capture_ir"),
        autofocus=_strict_bool(payload["autofocus"], f"{field}.autofocus"),
        samples_per_scan=_strict_int(payload["samples_per_scan"], f"{field}.samples_per_scan", minimum=1),
        auto_exposure=_strict_bool(payload["auto_exposure"], f"{field}.auto_exposure"),
    )


def _optional_recipe(value: object, field: str, recipe_type):
    return None if value is None else _parse_recipe(value, field, recipe_type)


def _json_safe_tree(value: object, field: str) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return value
    if type(value) is list:
        return [_json_safe_tree(item, f"{field}[]") for item in value]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError(f"{field} keys must be strings")
        return {key: _json_safe_tree(item, f"{field}.{key}") for key, item in value.items()}
    raise ValueError(f"{field} contains unsupported JSON data")


def _coherent_artifact(path: str | None, shape: tuple[int, ...] | None, dtype: str | None, field: str) -> None:
    if (path is None) != (shape is None) or (path is None) != (dtype is None):
        raise ValueError(f"{field} path, shape, and dtype must be recorded together")


def _read_plan(path: Path) -> RollScanPlan:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not read roll plan {path}: {error}") from error
    if type(payload) is not dict or type(payload.get("version")) is not int or payload.get("version") != PLAN_VERSION:
        raise ValueError(f"unsupported roll plan version in {path}")
    try:
        root = _exact_keys(
            payload,
            {
                "version",
                "identity",
                "device_id",
                "stage",
                "approved",
                "preview_dpi",
                "preview_depth",
                "preview_recipe",
                "registration_signature",
                "visual_override_frames",
                "calibration_context",
                "frames",
            },
            "plan",
        )
        preview_recipe = _parse_recipe(root["preview_recipe"], "preview_recipe", PreviewScanRecipe)
        preview_dpi = _strict_int(root["preview_dpi"], "preview_dpi", minimum=1)
        preview_depth = _strict_int(root["preview_depth"], "preview_depth")
        if preview_dpi != preview_recipe.dpi or preview_depth != preview_recipe.depth:
            raise ValueError("preview resolution/depth disagrees with preview_recipe")
        signature = _json_safe_tree(root["registration_signature"], "registration_signature")
        if type(signature) is not dict or not signature:
            raise ValueError("registration_signature must be a nonempty object")
        registration_signature = cast(dict[str, object], signature)
        raw_frames = root["frames"]
        if type(raw_frames) is not list or not raw_frames:
            raise ValueError("frames must be a nonempty array")

        raw_context = root["calibration_context"]
        if type(raw_context) is not list:
            raise ValueError("calibration_context must be an array")
        calibration_context: list[CalibrationPreviewRecord] = []
        for index, raw_item in enumerate(raw_context):
            item = _exact_keys(
                raw_item,
                {"frame", "sha256", "shape", "dtype", "dpi"},
                f"calibration_context[{index}]",
            )
            context_dtype = _strict_string(item["dtype"], f"calibration_context[{index}].dtype", nonempty=True)
            if context_dtype != "uint16":
                raise ValueError(f"calibration_context[{index}].dtype must be uint16")
            context_dpi = _strict_int(item["dpi"], f"calibration_context[{index}].dpi", minimum=1)
            if context_dpi != preview_dpi:
                raise ValueError(f"calibration_context[{index}].dpi disagrees with preview recipe")
            context_sha256 = _optional_sha256(item["sha256"], f"calibration_context[{index}].sha256")
            context_shape = _optional_shape(
                item["shape"],
                f"calibration_context[{index}].shape",
                channels=3,
            )
            if context_sha256 is None or context_shape is None:
                raise ValueError(f"calibration_context[{index}] hash/shape evidence is required")
            calibration_context.append(
                CalibrationPreviewRecord(
                    frame=_strict_int(item["frame"], f"calibration_context[{index}].frame", minimum=1),
                    sha256=context_sha256,
                    shape=context_shape,
                    dtype=context_dtype,
                    dpi=context_dpi,
                )
            )
        context_frames = tuple(item.frame for item in calibration_context)
        if tuple(sorted(set(context_frames))) != context_frames:
            raise ValueError("calibration_context frames must be unique and increasing")

        records: list[RollFrameRecord] = []
        record_keys = {
            "frame",
            "wide_path",
            "wide_rgb_shape",
            "wide_dtype",
            "wide_sha256",
            "wide_dpi",
            "registration",
            "aligned_path",
            "aligned_rgb_shape",
            "aligned_dtype",
            "aligned_sha256",
            "aligned_dpi",
            "verification",
            "fine_path",
            "fine_rgb_shape",
            "fine_ir_shape",
            "fine_dtype",
            "fine_ir_dtype",
            "fine_dpi",
            "fine_recipe",
        }
        for index, raw_item in enumerate(raw_frames):
            item = _exact_keys(raw_item, record_keys, f"frames[{index}]")
            frame = _strict_int(item["frame"], f"frames[{index}].frame", minimum=1)
            registration_data = item["registration"]
            if registration_data is None:
                registration = None
            else:
                registration_payload = _exact_keys(
                    registration_data,
                    {"frame", "target_start_row", "usable_tail_row", "confidence", "geometry"},
                    f"frames[{index}].registration",
                )
                registration_frame = _strict_int(
                    registration_payload["frame"],
                    f"frames[{index}].registration.frame",
                    minimum=1,
                )
                target_start_row = _strict_int(
                    registration_payload["target_start_row"],
                    f"frames[{index}].registration.target_start_row",
                    minimum=0,
                )
                usable_tail_row = _finite_number(
                    registration_payload["usable_tail_row"],
                    f"frames[{index}].registration.usable_tail_row",
                )
                confidence = _strict_string(
                    registration_payload["confidence"],
                    f"frames[{index}].registration.confidence",
                    nonempty=True,
                )
                if confidence not in {"high", "medium"}:
                    raise ValueError(f"frames[{index}].registration.confidence is invalid")
                geometry_payload = _exact_keys(
                    registration_payload["geometry"],
                    {"frame", "subframe_mm", "br_y_device_px"},
                    f"frames[{index}].registration.geometry",
                )
                geometry_frame = _strict_int(
                    geometry_payload["frame"],
                    f"frames[{index}].registration.geometry.frame",
                    minimum=1,
                )
                if registration_frame != frame or geometry_frame not in {frame, frame - 1}:
                    raise ValueError(f"frames[{index}] registration/geometry frame mismatch")
                if usable_tail_row <= target_start_row:
                    raise ValueError(f"frames[{index}] usable tail must follow the target start")
                registration = FrameRegistration(
                    frame=registration_frame,
                    target_start_row=target_start_row,
                    usable_tail_row=usable_tail_row,
                    confidence=confidence,
                    geometry=RegisteredScanGeometry(
                        frame=geometry_frame,
                        subframe_mm=_finite_number(
                            geometry_payload["subframe_mm"],
                            f"frames[{index}].registration.geometry.subframe_mm",
                        ),
                        br_y_device_px=_strict_int(
                            geometry_payload["br_y_device_px"],
                            f"frames[{index}].registration.geometry.br_y_device_px",
                            minimum=0,
                        ),
                    ),
                )

            verification_data = item["verification"]
            if verification_data is None:
                verification = None
            else:
                verification_payload = _exact_keys(
                    verification_data,
                    {"leading_margin_rows", "target_margin_rows", "tolerance_rows", "confidence"},
                    f"frames[{index}].verification",
                )
                leading_value = verification_payload["leading_margin_rows"]
                leading_margin_rows = (
                    None
                    if leading_value is None
                    else _strict_int(leading_value, f"frames[{index}].verification.leading_margin_rows", minimum=0)
                )
                verification_confidence = _strict_string(
                    verification_payload["confidence"],
                    f"frames[{index}].verification.confidence",
                    nonempty=True,
                )
                if verification_confidence not in {"high", "medium", "unresolved"}:
                    raise ValueError(f"frames[{index}].verification.confidence is invalid")
                verification = AlignmentVerification(
                    leading_margin_rows=leading_margin_rows,
                    target_margin_rows=_strict_int(
                        verification_payload["target_margin_rows"],
                        f"frames[{index}].verification.target_margin_rows",
                        minimum=0,
                    ),
                    tolerance_rows=_strict_int(
                        verification_payload["tolerance_rows"],
                        f"frames[{index}].verification.tolerance_rows",
                        minimum=0,
                    ),
                    confidence=verification_confidence,
                )

            wide_path = _optional_path(item["wide_path"], f"frames[{index}].wide_path")
            wide_rgb_shape = _optional_shape(item["wide_rgb_shape"], f"frames[{index}].wide_rgb_shape", channels=3)
            wide_dtype = _optional_dtype(item["wide_dtype"], f"frames[{index}].wide_dtype")
            wide_sha256 = _optional_sha256(item["wide_sha256"], f"frames[{index}].wide_sha256")
            wide_dpi = None if item["wide_dpi"] is None else _strict_int(item["wide_dpi"], f"frames[{index}].wide_dpi", minimum=1)
            aligned_path = _optional_path(item["aligned_path"], f"frames[{index}].aligned_path")
            aligned_rgb_shape = _optional_shape(item["aligned_rgb_shape"], f"frames[{index}].aligned_rgb_shape", channels=3)
            aligned_dtype = _optional_dtype(item["aligned_dtype"], f"frames[{index}].aligned_dtype")
            aligned_sha256 = _optional_sha256(item["aligned_sha256"], f"frames[{index}].aligned_sha256")
            aligned_dpi = (
                None if item["aligned_dpi"] is None else _strict_int(item["aligned_dpi"], f"frames[{index}].aligned_dpi", minimum=1)
            )
            fine_path = _optional_path(item["fine_path"], f"frames[{index}].fine_path")
            fine_rgb_shape = _optional_shape(item["fine_rgb_shape"], f"frames[{index}].fine_rgb_shape", channels=3)
            fine_ir_shape = _optional_shape(item["fine_ir_shape"], f"frames[{index}].fine_ir_shape", channels=None)
            fine_dtype = _optional_dtype(item["fine_dtype"], f"frames[{index}].fine_dtype")
            fine_ir_dtype = _optional_dtype(item["fine_ir_dtype"], f"frames[{index}].fine_ir_dtype")
            fine_dpi = None if item["fine_dpi"] is None else _strict_int(item["fine_dpi"], f"frames[{index}].fine_dpi", minimum=1)
            fine_recipe = _optional_recipe(item["fine_recipe"], f"frames[{index}].fine_recipe", FineScanRecipe)

            _coherent_artifact(wide_path, wide_rgb_shape, wide_dtype, f"frames[{index}].wide")
            _coherent_artifact(aligned_path, aligned_rgb_shape, aligned_dtype, f"frames[{index}].aligned")
            if (wide_path is None) != (wide_sha256 is None) or (wide_path is None) != (wide_dpi is None):
                raise ValueError(f"frames[{index}] wide hash/DPI evidence is incomplete")
            if (aligned_path is None) != (aligned_sha256 is None) or (aligned_path is None) != (aligned_dpi is None):
                raise ValueError(f"frames[{index}] aligned hash/DPI evidence is incomplete")
            if wide_dpi is not None and wide_dpi != preview_dpi:
                raise ValueError(f"frames[{index}] wide DPI disagrees with the preview recipe")
            if aligned_dpi is not None and aligned_dpi != preview_dpi:
                raise ValueError(f"frames[{index}] aligned DPI disagrees with the preview recipe")
            _coherent_artifact(fine_path, fine_rgb_shape, fine_dtype, f"frames[{index}].fine")
            if registration is not None and wide_path is None:
                raise ValueError(f"frames[{index}] registration requires a wide preview")
            if aligned_path is not None and (registration is None or verification is None):
                raise ValueError(f"frames[{index}] aligned preview requires registration and verification")
            if fine_path is None:
                if any(value is not None for value in (fine_ir_shape, fine_ir_dtype, fine_dpi, fine_recipe)):
                    raise ValueError(f"frames[{index}] has partial fine-scan metadata")
            else:
                if registration is None or fine_recipe is None or fine_dpi != fine_recipe.dpi:
                    raise ValueError(f"frames[{index}] fine scan recipe/registration is incomplete")
                if fine_recipe.capture_ir != (fine_ir_shape is not None and fine_ir_dtype is not None):
                    raise ValueError(f"frames[{index}] IR metadata disagrees with the fine recipe")

            records.append(
                RollFrameRecord(
                    frame=frame,
                    wide_path=wide_path,
                    wide_rgb_shape=wide_rgb_shape,
                    wide_dtype=wide_dtype,
                    wide_sha256=wide_sha256,
                    wide_dpi=wide_dpi,
                    registration=registration,
                    aligned_path=aligned_path,
                    aligned_rgb_shape=aligned_rgb_shape,
                    aligned_dtype=aligned_dtype,
                    aligned_sha256=aligned_sha256,
                    aligned_dpi=aligned_dpi,
                    verification=verification,
                    fine_path=fine_path,
                    fine_rgb_shape=fine_rgb_shape,
                    fine_ir_shape=fine_ir_shape,
                    fine_dtype=fine_dtype,
                    fine_ir_dtype=fine_ir_dtype,
                    fine_dpi=fine_dpi,
                    fine_recipe=fine_recipe,
                )
            )

        frame_numbers = tuple(record.frame for record in records)
        if tuple(sorted(set(frame_numbers))) != frame_numbers:
            raise ValueError("plan frames must be unique and increasing")
        raw_overrides = root["visual_override_frames"]
        if type(raw_overrides) is not list:
            raise ValueError("visual_override_frames must be an array")
        visual_override_frames = tuple(
            _strict_int(value, f"visual_override_frames[{index}]", minimum=1) for index, value in enumerate(raw_overrides)
        )
        if tuple(sorted(set(visual_override_frames))) != visual_override_frames:
            raise ValueError("visual_override_frames must be unique and increasing")
        if not set(visual_override_frames).issubset(frame_numbers):
            raise ValueError("visual_override_frames contains a frame absent from the plan")
        approved = _strict_bool(root["approved"], "approved")
        stage = RollStage(_strict_string(root["stage"], "stage", nonempty=True))
        approved_stages = {RollStage.APPROVED, RollStage.FINE_SCANNING, RollStage.COMPLETE}
        if approved != (stage in approved_stages):
            raise ValueError("approved flag disagrees with plan stage")
        if not approved and visual_override_frames:
            raise ValueError("an unapproved plan cannot retain visual overrides")
        if approved:
            unverified = {record.frame for record in records if record.verification is None or not record.verification.passed}
            if not unverified.issubset(visual_override_frames):
                raise ValueError("approved plan has unverified frames without visual overrides")
        if stage in {RollStage.ALIGNED_PREVIEW, RollStage.REVIEW_REQUIRED, *approved_stages} and any(
            record.wide_path is None or record.registration is None for record in records
        ):
            raise ValueError(f"stage {stage.value} requires every wide preview and registration")
        if stage in {RollStage.REVIEW_REQUIRED, *approved_stages} and any(
            record.aligned_path is None or record.verification is None for record in records
        ):
            raise ValueError(f"stage {stage.value} requires every aligned preview and verification")
        if stage is RollStage.COMPLETE:
            if any(record.fine_path is None for record in records):
                raise ValueError("complete plan is missing a fine scan")
            complete_recipes = {record.fine_recipe for record in records}
            if len(complete_recipes) != 1 or None in complete_recipes:
                raise ValueError("complete plan contains mixed or missing fine recipes")

        return RollScanPlan(
            version=PLAN_VERSION,
            identity=PlanIdentity.from_dict(root["identity"]),
            device_id=_strict_string(root["device_id"], "device_id", nonempty=True),
            stage=stage,
            approved=approved,
            preview_dpi=preview_dpi,
            preview_depth=preview_depth,
            preview_recipe=preview_recipe,
            registration_signature=registration_signature,
            visual_override_frames=visual_override_frames,
            calibration_context=tuple(calibration_context),
            frames=tuple(records),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed roll plan {path}: {error}") from error


def _tiff_payload_matches(
    path: Path,
    *,
    shape: tuple[int, ...] | None,
    dtype: str | None,
    sha256_digest: str | None = None,
    dpi: int | None = None,
) -> bool:
    """Check saved geometry and that every TIFF strip lies inside the file."""

    if shape is None or dtype is None or not path.is_file():
        return False
    try:
        if sha256_digest is not None or dpi is not None:
            if sha256_digest is None or dpi is None:
                return False
            inspection = inspect_tiff_payload(path)
            expected_resolution = (dpi, 1)
            if not (
                inspection.sha256 == sha256_digest
                and inspection.x_resolution == expected_resolution
                and inspection.y_resolution == expected_resolution
                and inspection.resolution_unit == "INCH"
            ):
                return False
        file_size = path.stat().st_size
        with tifffile.TiffFile(path) as document:
            if len(document.pages) != 1:
                return False
            page = document.pages[0]
            if tuple(int(value) for value in page.shape) != shape:
                return False
            if np.dtype(page.dtype).name != dtype:
                return False
            if not page.dataoffsets or len(page.dataoffsets) != len(page.databytecounts):
                return False
            return all(
                int(offset) >= 0 and int(count) > 0 and int(offset) + int(count) <= file_size
                for offset, count in zip(page.dataoffsets, page.databytecounts, strict=True)
            )
    except (OSError, ValueError, tifffile.TiffFileError):
        return False


def _tiff_signature(path: Path) -> tuple[tuple[int, ...], str]:
    """Return the shape and dtype actually committed to a TIFF."""

    with tifffile.TiffFile(path) as document:
        if len(document.pages) != 1:
            raise ValueError(f"expected one TIFF page in {path}")
        page = document.pages[0]
        return (
            tuple(int(value) for value in page.shape),
            np.dtype(page.dtype).name,
        )


def _preview_signature(path: Path, *, dpi: int) -> tuple[tuple[int, ...], str, str, int]:
    shape, dtype = _tiff_signature(path)
    inspection = inspect_tiff_payload(path)
    expected_resolution = (dpi, 1)
    if (
        inspection.x_resolution != expected_resolution
        or inspection.y_resolution != expected_resolution
        or inspection.resolution_unit != "INCH"
    ):
        raise ValueError(f"preview TIFF {path} does not record {dpi} DPI")
    return shape, dtype, inspection.sha256, dpi


def _fine_record_valid(
    root: Path,
    record: RollFrameRecord,
    recipe: FineScanRecipe | None,
    device_id: str,
) -> bool:
    if recipe is None or record.fine_recipe != recipe or record.fine_path is None or record.fine_dpi != recipe.dpi:
        return False
    output = root / record.fine_path
    if not _tiff_payload_matches(
        output,
        shape=record.fine_rgb_shape,
        dtype=record.fine_dtype,
    ):
        return False
    if recipe.capture_ir:
        ir_sidecar = output.with_name(f"{output.stem}_IR.tif")
        if not _tiff_payload_matches(
            ir_sidecar,
            shape=record.fine_ir_shape,
            dtype=record.fine_ir_dtype,
        ):
            return False
    return _fine_qc_valid(output, record, recipe, device_id)


def _complete_plan_valid(root: Path, plan: RollScanPlan) -> bool:
    recipes = {record.fine_recipe for record in plan.frames}
    if len(recipes) != 1:
        return False
    recipe = next(iter(recipes))
    if recipe is None:
        return False
    return all(_fine_record_valid(root, record, recipe, plan.device_id) for record in plan.frames)


def _fine_artifact_name(
    *,
    frame: int,
    recipe: FineScanRecipe,
    geometry: RegisteredScanGeometry,
) -> str:
    """Key a fine artifact by every setting that can change its meaning."""

    payload = json.dumps(
        {
            "schema": 1,
            "recipe": asdict(recipe),
            "geometry": asdict(geometry),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"frame{frame:03d}-{sha256(payload).hexdigest()[:16]}.tif"


def _fine_qc_path(output: Path) -> Path:
    return Path(f"{output}.qc.json")


def _read_fine_qc(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return payload if type(payload) is dict else None


def _finite_json_number(value: object) -> bool:
    return type(value) in (int, float) and isfinite(value)


def _fine_qc_telemetry_valid(
    payload: Mapping[str, object],
    *,
    allow_indeterminate: bool,
    expected_dpi: int,
    image_height: int,
) -> bool:
    smear = payload.get("stopped_transport_smear")
    if type(smear) is not dict or set(smear) != {
        "verdict",
        "start_row",
        "suffix_rows",
        "minimum_matches",
        "tail_median_rms",
        "tail_min_corr",
        "pre_tail_median_rms",
        "texture_span",
        "reason",
    }:
        return False
    verdict = smear.get("verdict")
    if verdict != "clean" and not (verdict == "indeterminate" and allow_indeterminate):
        return False
    suffix_rows = smear.get("suffix_rows")
    minimum_matches = smear.get("minimum_matches")
    if (
        type(suffix_rows) is not int
        or suffix_rows < 0
        or suffix_rows >= image_height
        or type(minimum_matches) is not int
        or minimum_matches != ceil(expected_dpi * 0.016)
    ):
        return False
    if type(smear.get("reason")) is not str or not smear.get("reason").strip():
        return False
    start_row = smear.get("start_row")
    if start_row is not None and (type(start_row) is not int or start_row < 0):
        return False
    for field in ("tail_median_rms", "pre_tail_median_rms", "texture_span"):
        value = smear.get(field)
        if value is not None and (not _finite_json_number(value) or value < 0):
            return False
    tail_min_corr = smear.get("tail_min_corr")
    if tail_min_corr is not None and (not _finite_json_number(tail_min_corr) or not -1.0 <= tail_min_corr <= 1.0):
        return False
    tail_fields = ("tail_median_rms", "tail_min_corr", "pre_tail_median_rms")
    if verdict == "clean":
        if start_row is not None or suffix_rows != 0 or any(smear.get(field) is not None for field in tail_fields):
            return False
        if smear.get("texture_span") is None:
            return False
    elif suffix_rows == 0:
        if start_row is not None or any(smear.get(field) is not None for field in tail_fields):
            return False
    elif (
        start_row != image_height - suffix_rows
        or smear.get("tail_median_rms") is None
        or tail_min_corr is None
        or smear.get("texture_span") is None
    ):
        return False

    clipping = payload.get("clipping")
    if type(clipping) is not dict or set(clipping) != {
        "fractions",
        "clip_level",
        "warning_fraction",
        "warning",
    }:
        return False
    fractions = clipping.get("fractions")
    if (
        type(fractions) is not list
        or len(fractions) != 3
        or not all(_finite_json_number(value) and 0.0 <= value <= 1.0 for value in fractions)
    ):
        return False
    clip_level = clipping.get("clip_level")
    warning_fraction = clipping.get("warning_fraction")
    if (
        not _finite_json_number(clip_level)
        or not 0.0 < clip_level <= 1.0
        or not _finite_json_number(warning_fraction)
        or not 0.0 <= warning_fraction <= 1.0
    ):
        return False
    warning = clipping.get("warning")
    if type(warning) is not bool or warning is not (max(fractions) > warning_fraction):
        return False

    focus = payload.get("focus_detail")
    if type(focus) is not dict or set(focus) != {"method", "verdict", "score", "texture_span"}:
        return False
    if focus.get("method") != "normalized-gradient-v1":
        return False
    focus_verdict = focus.get("verdict")
    if focus_verdict not in ("measured", "indeterminate"):
        return False
    score = focus.get("score")
    if score is not None and (not _finite_json_number(score) or score < 0):
        return False
    focus_texture_span = focus.get("texture_span")
    if not _finite_json_number(focus_texture_span) or focus_texture_span < 0:
        return False
    if focus_verdict == "measured":
        return score is not None and focus_texture_span > 0
    return score is None


def _fine_qc_valid(
    output: Path,
    record: RollFrameRecord,
    recipe: FineScanRecipe,
    device_id: str,
) -> bool:
    payload = _read_fine_qc(_fine_qc_path(output))
    expected_keys = {
        "version",
        "accepted",
        "frame",
        "device_id",
        "fine_recipe",
        "registered_geometry",
        "artifacts",
        "split_alignment",
        "stopped_transport_smear",
        "clipping",
        "focus_detail",
        "human_overrides",
        "device_health",
    }
    if (
        payload is None
        or set(payload) != expected_keys
        or type(payload.get("version")) is not int
        or payload.get("version") != FINE_QC_VERSION
        or payload.get("accepted") is not True
    ):
        return False
    if record.registration is None:
        return False
    if (
        payload.get("frame") != record.frame
        or payload.get("device_id") != device_id
        or payload.get("fine_recipe") != asdict(recipe)
        or payload.get("registered_geometry") != asdict(record.registration.geometry)
    ):
        return False
    artifacts = payload.get("artifacts")
    if type(artifacts) is not dict:
        return False
    rgb = artifacts.get("rgb")
    if type(rgb) is not dict:
        return False
    try:
        root = output.parents[1]
        live_rgb = _fine_artifact_evidence(output, root=root)
        _require_fine_artifact(
            live_rgb,
            frame=record.frame,
            expected_shape=cast(tuple[int, ...], record.fine_rgb_shape),
            expected_dpi=recipe.dpi,
            label="RGB",
        )
        if rgb != live_rgb:
            return False
        if recipe.capture_ir:
            ir = artifacts.get("ir")
            if type(ir) is not dict:
                return False
            ir_output = output.with_name(f"{output.stem}_IR.tif")
            live_ir = _fine_artifact_evidence(ir_output, root=root)
            _require_fine_artifact(
                live_ir,
                frame=record.frame,
                expected_shape=cast(tuple[int, ...], record.fine_ir_shape),
                expected_dpi=recipe.dpi,
                label="IR",
            )
            if ir != live_ir:
                return False
            rgb_shape = cast(list[int], live_rgb["shape"])
            ir_shape = cast(list[int], live_ir["shape"])
            if rgb_shape[:2] != ir_shape:
                return False
        elif artifacts.get("ir") is not None:
            return False

        overrides = payload.get("human_overrides")
        if type(overrides) is not dict or set(overrides) != {"allow_indeterminate_stopped_transport_smear"}:
            return False
        allow_indeterminate = overrides.get("allow_indeterminate_stopped_transport_smear")
        if type(allow_indeterminate) is not bool:
            return False
        if not _fine_qc_telemetry_valid(
            payload,
            allow_indeterminate=allow_indeterminate,
            expected_dpi=recipe.dpi,
            image_height=int(cast(list[int], live_rgb["shape"])[0]),
        ):
            return False

        alignment_payload = payload.get("split_alignment")
        alignment = None
        if alignment_payload is not None:
            required_alignment_fields = {
                "mode",
                "dx_px",
                "dy_px",
                "phase_responses",
                "channel_spread_px",
                "ecc_coefficient",
            }
            optional_alignment_fields = {
                "tile_support_counts",
                "tile_shift_spread_px",
                "estimator_version",
                "multiscale_max_dimensions",
                "multiscale_channel_shifts_px",
                "multiscale_responses",
                "multiscale_tile_support_counts",
                "multiscale_tile_shift_spreads_px",
                "multiscale_global_alias_shifts_px",
            }
            if type(alignment_payload) is not dict:
                return False
            alignment_payload = cast(dict[str, object], alignment_payload)
            if not required_alignment_fields.issubset(alignment_payload) or not set(alignment_payload).issubset(
                required_alignment_fields | optional_alignment_fields
            ):
                return False
            phase_responses = alignment_payload.get("phase_responses")
            if type(phase_responses) is not list:
                return False
            phase_responses = cast(list[float], phase_responses)
            alignment = SplitIrAlignment(
                mode=cast(str, alignment_payload.get("mode")),
                dx_px=float(cast(float, alignment_payload.get("dx_px"))),
                dy_px=float(cast(float, alignment_payload.get("dy_px"))),
                phase_responses=tuple(float(value) for value in phase_responses),
                channel_spread_px=(
                    None
                    if alignment_payload.get("channel_spread_px") is None
                    else float(cast(float, alignment_payload.get("channel_spread_px")))
                ),
                ecc_coefficient=(
                    None
                    if alignment_payload.get("ecc_coefficient") is None
                    else float(cast(float, alignment_payload.get("ecc_coefficient")))
                ),
                tile_support_counts=tuple(int(value) for value in cast(list[int], alignment_payload.get("tile_support_counts", ()))),
                tile_shift_spread_px=(
                    None
                    if alignment_payload.get("tile_shift_spread_px") is None
                    else float(cast(float, alignment_payload.get("tile_shift_spread_px")))
                ),
                estimator_version=(
                    None
                    if alignment_payload.get("estimator_version") is None
                    else int(cast(int, alignment_payload.get("estimator_version")))
                ),
                multiscale_max_dimensions=tuple(
                    int(value) for value in cast(list[int], alignment_payload.get("multiscale_max_dimensions", ()))
                ),
                multiscale_channel_shifts_px=tuple(
                    tuple((float(pair[0]), float(pair[1])) for pair in scale)
                    for scale in cast(
                        list[list[list[float]]],
                        alignment_payload.get("multiscale_channel_shifts_px", ()),
                    )
                ),
                multiscale_responses=tuple(
                    tuple(float(value) for value in scale)
                    for scale in cast(
                        list[list[float]],
                        alignment_payload.get("multiscale_responses", ()),
                    )
                ),
                multiscale_tile_support_counts=tuple(
                    tuple(int(value) for value in scale)
                    for scale in cast(
                        list[list[int]],
                        alignment_payload.get("multiscale_tile_support_counts", ()),
                    )
                ),
                multiscale_tile_shift_spreads_px=tuple(
                    tuple(float(value) for value in scale)
                    for scale in cast(
                        list[list[float]],
                        alignment_payload.get("multiscale_tile_shift_spreads_px", ()),
                    )
                ),
                multiscale_global_alias_shifts_px=tuple(
                    tuple((float(pair[0]), float(pair[1])) for pair in scale)
                    for scale in cast(
                        list[list[list[float]]],
                        alignment_payload.get("multiscale_global_alias_shifts_px", ()),
                    )
                ),
            )
        split_capture = recipe.capture_ir and recipe.samples_per_scan > 1 and "coolscan" in device_id.lower()
        if split_capture and not split_alignment_metrics_confident(
            alignment,
            image_width=int(cast(list[int], live_rgb["shape"])[1]),
        ):
            return False

        health = payload.get("device_health")
        if (
            type(health) is not dict
            or set(health) != {"fresh_probe", "id", "vendor", "model"}
            or health.get("fresh_probe") is not True
            or health.get("id") != device_id
            or type(health.get("vendor")) is not str
            or not health.get("vendor")
            or type(health.get("model")) is not str
            or not health.get("model")
        ):
            return False
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError, tifffile.TiffFileError):
        return False
    return True


def _resolution_matches_dpi(resolution: tuple[int, int] | None, expected_dpi: int) -> bool:
    if resolution is None or expected_dpi <= 0:
        return False
    numerator, denominator = resolution
    return numerator > 0 and denominator > 0 and numerator == expected_dpi * denominator


def _fine_artifact_evidence(path: Path, *, root: Path) -> dict[str, object]:
    inspection = inspect_tiff_payload(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": inspection.sha256,
        "bytes": inspection.byte_length,
        "shape": list(inspection.shape),
        "dtype": inspection.dtype,
        "page_count": inspection.page_count,
        "payload_within_file": inspection.payload_within_file,
        "x_resolution": list(inspection.x_resolution) if inspection.x_resolution is not None else None,
        "y_resolution": list(inspection.y_resolution) if inspection.y_resolution is not None else None,
        "resolution_unit": inspection.resolution_unit,
    }


def _require_fine_artifact(
    evidence: Mapping[str, object],
    *,
    frame: int,
    expected_shape: tuple[int, ...],
    expected_dpi: int,
    label: str,
) -> None:
    x_resolution = evidence["x_resolution"]
    y_resolution = evidence["y_resolution"]
    x_pair = tuple(x_resolution) if isinstance(x_resolution, list) else None
    y_pair = tuple(y_resolution) if isinstance(y_resolution, list) else None
    valid = (
        evidence["shape"] == list(expected_shape)
        and evidence["dtype"] == "uint16"
        and evidence["page_count"] == 1
        and evidence["payload_within_file"] is True
        and evidence["resolution_unit"] == "INCH"
        and _resolution_matches_dpi(cast(tuple[int, int] | None, x_pair), expected_dpi)
        and _resolution_matches_dpi(cast(tuple[int, int] | None, y_pair), expected_dpi)
    )
    if not valid:
        raise RuntimeError(f"frame {frame}: fine {label} artifact failed committed TIFF metadata QC")


def _aligned_record_valid(root: Path, record: RollFrameRecord) -> bool:
    if record.aligned_path is None or record.verification is None:
        return False
    path = root / record.aligned_path
    if not _tiff_payload_matches(
        path,
        shape=record.aligned_rgb_shape,
        dtype=record.aligned_dtype,
        sha256_digest=record.aligned_sha256,
        dpi=record.aligned_dpi,
    ):
        return False
    try:
        preview = tifffile.imread(path)
    except (OSError, ValueError, tifffile.TiffFileError):
        return False
    return tuple(int(value) for value in preview.shape) == record.aligned_rgb_shape and np.dtype(preview.dtype).name == record.aligned_dtype


def approved_plan_binding_sha256(plan: RollScanPlan) -> str:
    """Bind an approval to its exact roll, context pixels, previews and geometry."""

    if not plan.approved:
        raise ValueError("cannot bind an unapproved roll plan")
    frames = []
    for record in plan.frames:
        if record.registration is None or record.verification is None:
            raise ValueError(f"frame {record.frame} is missing approval evidence")
        frames.append(
            {
                "frame": record.frame,
                "wide": {
                    "sha256": record.wide_sha256,
                    "shape": list(record.wide_rgb_shape or ()),
                    "dtype": record.wide_dtype,
                    "dpi": record.wide_dpi,
                },
                "aligned": {
                    "sha256": record.aligned_sha256,
                    "shape": list(record.aligned_rgb_shape or ()),
                    "dtype": record.aligned_dtype,
                    "dpi": record.aligned_dpi,
                },
                "registration": asdict(record.registration),
                "verification": asdict(record.verification),
            }
        )
    return canonical_semantic_sha256(
        {
            "schema": 1,
            "identity": plan.identity.to_dict(),
            "device_id": plan.device_id,
            "preview_recipe": asdict(plan.preview_recipe),
            "registration_signature": dict(plan.registration_signature),
            "visual_override_frames": list(plan.visual_override_frames),
            "calibration_context": [asdict(item) for item in plan.calibration_context],
            "frames": frames,
        }
    )


class RollScanService:
    """Orchestrate wide previews, registered previews, review, and fine scans."""

    def __init__(
        self,
        *,
        scanner: ScannerService,
        registration: RollRegistration,
    ) -> None:
        self._scanner = scanner
        self._registration = registration

    @staticmethod
    def _validate_frames(frames: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(frames)
        if not normalized:
            raise ValueError("at least one exposure frame is required")
        if any(type(frame) is not int or frame < 1 for frame in normalized):
            raise ValueError("exposure frames must be positive integers")
        if tuple(sorted(set(normalized))) != normalized:
            raise ValueError("exposure frames must be unique and increasing")
        return normalized

    @staticmethod
    def _progress(
        callback: ProgressCallback | None,
        stage: RollStage,
        frame: int,
    ) -> Callable[[float], None]:
        if callback is None:
            return lambda _value: None
        return lambda value: callback(stage, frame, value)

    def _current_registration_signature(self) -> dict[str, object]:
        signature = _json_safe_tree(
            dict(self._registration.registration_signature),
            "registration_signature",
        )
        if type(signature) is not dict or not signature:
            raise ValueError("registration policy must provide a nonempty JSON signature")
        return cast(dict[str, object], signature)

    @staticmethod
    def _write_preview(result, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_tiff_16bit(result, str(path))
        return str(path.relative_to(path.parents[1]))

    def _write_fine_qc(
        self,
        *,
        root: Path,
        output: Path,
        frame: int,
        device_id: str,
        recipe: FineScanRecipe,
        geometry: RegisteredScanGeometry,
        result,
        allow_indeterminate_smear: bool,
    ) -> None:
        rgb_evidence = _fine_artifact_evidence(output, root=root)
        _require_fine_artifact(
            rgb_evidence,
            frame=frame,
            expected_shape=tuple(int(value) for value in result.rgb.shape),
            expected_dpi=recipe.dpi,
            label="RGB",
        )

        ir_evidence = None
        if recipe.capture_ir:
            if result.ir is None:
                raise RuntimeError(f"frame {frame}: fine QC requires the requested IR artifact")
            ir_output = output.with_name(f"{output.stem}_IR.tif")
            ir_evidence = _fine_artifact_evidence(ir_output, root=root)
            _require_fine_artifact(
                ir_evidence,
                frame=frame,
                expected_shape=tuple(int(value) for value in result.ir.shape),
                expected_dpi=recipe.dpi,
                label="IR",
            )
            if tuple(result.rgb.shape[:2]) != tuple(result.ir.shape):
                raise RuntimeError(f"frame {frame}: committed RGB and IR shapes do not match")

        split_capture = (
            recipe.capture_ir
            and recipe.samples_per_scan > 1
            and ("coolscan" in device_id.lower() or "coolscan" in result.device_model.lower())
        )
        if split_capture and not split_alignment_metrics_confident(
            result.ir_alignment,
            image_width=int(result.rgb.shape[1]),
        ):
            raise RuntimeError(f"frame {frame}: Coolscan split-capture alignment failed QC")

        committed_rgb = tifffile.imread(output)
        smear = assess_stopped_transport_smear(committed_rgb, dpi=recipe.dpi)
        if smear.verdict == "smear":
            raise RuntimeError(f"frame {frame}: fine QC detected stopped-transport smear")
        if smear.verdict == "indeterminate" and not allow_indeterminate_smear:
            raise RuntimeError(
                f"frame {frame}: stopped-transport smear QC is {smear.verdict!r}; "
                "an indeterminate result requires an explicit human override"
            )
        clipping = measure_scan_clipping(committed_rgb)
        focus_detail = measure_focus_detail(committed_rgb)

        try:
            device = self._scanner.probe_device(device_id)
        except Exception as exc:
            raise RuntimeError(f"frame {frame}: fresh scanner health probe failed: {exc}") from exc
        if device.id != device_id:
            raise RuntimeError(f"frame {frame}: fresh scanner probe returned {device.id!r}, expected {device_id!r}")

        sidecar = {
            "version": FINE_QC_VERSION,
            "accepted": True,
            "frame": frame,
            "device_id": device_id,
            "fine_recipe": asdict(recipe),
            "registered_geometry": asdict(geometry),
            "artifacts": {
                "rgb": rgb_evidence,
                "ir": ir_evidence,
            },
            "split_alignment": asdict(result.ir_alignment) if result.ir_alignment is not None else None,
            "stopped_transport_smear": asdict(smear),
            "clipping": asdict(clipping),
            "focus_detail": asdict(focus_detail),
            "human_overrides": {
                "allow_indeterminate_stopped_transport_smear": allow_indeterminate_smear,
            },
            "device_health": {
                "fresh_probe": True,
                "id": device.id,
                "vendor": device.vendor,
                "model": device.model,
            },
        }
        _atomic_write_json(_fine_qc_path(output), sidecar)

    def prepare_roll(
        self,
        *,
        device_id: str,
        frames: Sequence[int],
        output_dir: str | Path,
        preview_params: ScanParams,
        stop: threading.Event,
        progress: ProgressCallback | None = None,
        calibration_previews: Mapping[int, np.ndarray] | None = None,
        identity: PlanIdentity | None = None,
    ) -> RollScanPlan:
        """Acquire wide and registered previews, then stop for visual review.

        ``frames`` is an explicit exposure list.  Scanner adapter capacity is
        only a transport limit and must never be treated as the exposure count.
        ``calibration_previews`` supplies committed wide previews from earlier
        transport-safe chunks of the same roll.  They improve the shared film-
        base model but are never rescanned or added to this chunk's plan.
        """

        selected = self._validate_frames(frames)
        context = dict(calibration_previews or {})
        if any(type(frame) is not int or frame < 1 for frame in context):
            raise ValueError("calibration preview frames must be positive integers")
        overlap = sorted(set(context) & set(selected))
        if overlap:
            raise ValueError("calibration context overlaps selected frames: " + ", ".join(str(frame) for frame in overlap))
        for frame, image in context.items():
            preview = np.asarray(image)
            if preview.ndim != 3 or preview.shape[2] < 3:
                raise ValueError(f"calibration preview {frame} must contain RGB channels")
            if preview.dtype != np.uint16:
                raise ValueError(f"calibration preview {frame} must be a committed uint16 TIFF payload")
        context_records = tuple(
            CalibrationPreviewRecord(
                frame=frame,
                sha256=_array_content_sha256(np.asarray(context[frame])),
                shape=tuple(int(value) for value in np.asarray(context[frame]).shape),
                dtype=np.dtype(np.asarray(context[frame]).dtype).name,
                dpi=preview_params.dpi,
            )
            for frame in sorted(context)
        )
        minimum_preview_count = self._registration.minimum_preview_count
        if type(minimum_preview_count) is not int or minimum_preview_count < 1:
            raise ValueError("registration minimum_preview_count must be a positive integer")
        calibration_count = len(selected) + len(context)
        if calibration_count < minimum_preview_count:
            raise ValueError(f"registration requires at least {minimum_preview_count} preview frames; got {calibration_count}")
        if preview_params.area is not None:
            raise ValueError("roll registration requires area=None")
        if preview_params.dpi != self._registration.preview_dpi:
            raise ValueError(f"preview DPI {preview_params.dpi} does not match registration DPI {self._registration.preview_dpi}")
        preview_recipe = PreviewScanRecipe.from_params(preview_params)
        registration_signature = self._current_registration_signature()
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / PLAN_FILENAME
        transfer_cancel = threading.Event()

        if plan_path.exists():
            plan = _read_plan(plan_path)
            if identity is not None and plan.identity != identity:
                raise ValueError("roll identity differs from the saved roll plan")
            if plan.calibration_context != context_records:
                raise ValueError("calibration context differs from the saved roll plan")
            if plan.device_id != device_id:
                raise ValueError(f"roll plan belongs to {plan.device_id!r}, not {device_id!r}")
            if tuple(record.frame for record in plan.frames) != selected:
                raise ValueError("requested exposure list differs from the saved roll plan")
            if plan.preview_recipe != preview_recipe:
                raise ValueError("preview recipe differs from the saved roll plan")
            if plan.registration_signature != registration_signature:
                raise ValueError("registration policy differs from the saved roll plan")
            if plan.stage in (RollStage.APPROVED, RollStage.COMPLETE):
                if all(_aligned_record_valid(root, record) for record in plan.frames):
                    if plan.stage is RollStage.COMPLETE and not _complete_plan_valid(root, plan):
                        plan = replace(plan, stage=RollStage.APPROVED)
                        _atomic_write_json(plan_path, plan.to_dict())
                    return plan
                plan = replace(
                    plan,
                    stage=RollStage.ALIGNED_PREVIEW,
                    approved=False,
                    visual_override_frames=(),
                )
                _atomic_write_json(plan_path, plan.to_dict())
        else:
            if identity is None:
                identity = PlanIdentity(str(uuid4()), str(uuid4()), str(uuid4()))
            plan = RollScanPlan(
                identity=identity,
                device_id=device_id,
                stage=RollStage.WIDE_PREVIEW,
                approved=False,
                preview_dpi=preview_params.dpi,
                preview_depth=preview_params.depth,
                preview_recipe=preview_recipe,
                registration_signature=registration_signature,
                visual_override_frames=(),
                calibration_context=context_records,
                frames=tuple(RollFrameRecord(frame=frame) for frame in selected),
            )
            _atomic_write_json(plan_path, plan.to_dict())

        wide_arrays: dict[int, np.ndarray] = {}
        records = list(plan.frames)
        for index, frame in enumerate(selected):
            record = records[index]
            existing_wide = root / record.wide_path if record.wide_path else None
            if existing_wide is not None and _tiff_payload_matches(
                existing_wide,
                shape=record.wide_rgb_shape,
                dtype=record.wide_dtype,
                sha256_digest=record.wide_sha256,
                dpi=record.wide_dpi,
            ):
                wide_arrays[frame] = tifffile.imread(existing_wide)
                continue
            if stop.is_set():
                return replace(plan, stage=RollStage.WIDE_PREVIEW)
            params = replace(
                preview_params,
                frame=frame,
                registered_geometry=None,
            )
            result = self._scanner.run_scan(
                device_id,
                params,
                self._progress(progress, RollStage.WIDE_PREVIEW, frame),
                transfer_cancel,
            )
            wide_path = self._write_preview(result, root / "wide" / f"frame{frame:03d}.tif")
            wide_rgb_shape, wide_dtype, wide_sha256, wide_dpi = _preview_signature(
                root / wide_path,
                dpi=preview_params.dpi,
            )
            # Registration always consumes the committed TIFF representation.
            # The writer promotes 8-bit scans to uint16, so using result.rgb
            # here would mix uint8 and uint16 scales after a partial resume.
            wide_arrays[frame] = tifffile.imread(root / wide_path)
            records[index] = replace(
                record,
                wide_path=wide_path,
                wide_rgb_shape=wide_rgb_shape,
                wide_dtype=wide_dtype,
                wide_sha256=wide_sha256,
                wide_dpi=wide_dpi,
            )
            plan = replace(
                plan,
                stage=RollStage.WIDE_PREVIEW,
                frames=tuple(records),
            )
            _atomic_write_json(plan_path, plan.to_dict())
            if stop.is_set():
                return plan

        calibration_arrays = {frame: np.asarray(context[frame]) for frame in sorted(context)}
        calibration_arrays.update(wide_arrays)
        registrations = dict(self._registration.calibrate(calibration_arrays))
        if set(registrations) != set(calibration_arrays):
            raise RuntimeError("registration did not return exactly the calibration frames")

        for index, frame in enumerate(selected):
            registration = registrations[frame]
            if registration.frame != frame:
                raise RuntimeError(f"registration frame mismatch: expected {frame}, got {registration.frame}")
            if registration.geometry.frame not in (None, frame, frame - 1):
                raise RuntimeError(f"geometry frame mismatch: expected {frame}, got {registration.geometry.frame}")
            geometry = replace(registration.geometry, frame=frame) if registration.geometry.frame is None else registration.geometry
            registration = replace(registration, geometry=geometry)
            record = records[index]
            if record.registration != registration:
                record = replace(
                    record,
                    registration=registration,
                    aligned_path=None,
                    aligned_rgb_shape=None,
                    aligned_dtype=None,
                    aligned_sha256=None,
                    aligned_dpi=None,
                    verification=None,
                    fine_path=None,
                    fine_rgb_shape=None,
                    fine_ir_shape=None,
                    fine_dtype=None,
                    fine_ir_dtype=None,
                    fine_dpi=None,
                    fine_recipe=None,
                )
            records[index] = record

        plan = replace(
            plan,
            stage=RollStage.ALIGNED_PREVIEW,
            frames=tuple(records),
        )
        _atomic_write_json(plan_path, plan.to_dict())

        for index, frame in enumerate(selected):
            record = records[index]
            registration = record.registration
            if registration is None:
                raise RuntimeError(f"frame {frame} has no registration")
            existing_aligned = root / record.aligned_path if record.aligned_path else None
            if existing_aligned is not None and _aligned_record_valid(root, record):
                continue
            if any(
                value is not None
                for value in (
                    record.aligned_path,
                    record.aligned_rgb_shape,
                    record.aligned_dtype,
                    record.aligned_sha256,
                    record.aligned_dpi,
                    record.verification,
                )
            ):
                record = replace(
                    record,
                    aligned_path=None,
                    aligned_rgb_shape=None,
                    aligned_dtype=None,
                    aligned_sha256=None,
                    aligned_dpi=None,
                    verification=None,
                )
                records[index] = record
                plan = replace(
                    plan,
                    stage=RollStage.ALIGNED_PREVIEW,
                    frames=tuple(records),
                )
                _atomic_write_json(plan_path, plan.to_dict())
            if stop.is_set():
                return plan
            params = replace(
                preview_params,
                frame=registration.geometry.frame or frame,
                registered_geometry=registration.geometry,
            )
            result = self._scanner.run_scan(
                device_id,
                params,
                self._progress(progress, RollStage.ALIGNED_PREVIEW, frame),
                transfer_cancel,
            )
            aligned_path = self._write_preview(result, root / "aligned" / f"frame{frame:03d}.tif")
            aligned_rgb_shape, aligned_dtype, aligned_sha256, aligned_dpi = _preview_signature(
                root / aligned_path,
                dpi=preview_params.dpi,
            )
            aligned_preview = tifffile.imread(root / aligned_path)
            verification = self._registration.verify(frame, aligned_preview, registration)
            records[index] = replace(
                record,
                aligned_path=aligned_path,
                aligned_rgb_shape=aligned_rgb_shape,
                aligned_dtype=aligned_dtype,
                aligned_sha256=aligned_sha256,
                aligned_dpi=aligned_dpi,
                verification=verification,
            )
            plan = replace(plan, frames=tuple(records))
            _atomic_write_json(plan_path, plan.to_dict())
            if stop.is_set():
                return plan

        plan = replace(
            plan,
            stage=RollStage.REVIEW_REQUIRED,
            approved=False,
            visual_override_frames=(),
            frames=tuple(records),
        )
        _atomic_write_json(plan_path, plan.to_dict())
        return plan

    def approve_roll(
        self,
        plan_path: str | Path,
        *,
        allow_unverified_frames: Sequence[int] = (),
    ) -> RollScanPlan:
        """Persist review approval, with frame-specific visual overrides.

        An unresolved dark/cut frame is never approved implicitly.  Its frame
        number must be named after a person has inspected the aligned preview.
        """

        path = Path(plan_path).expanduser().resolve()
        plan = _read_plan(path)
        if plan.registration_signature != self._current_registration_signature():
            raise ValueError("registration policy differs from the saved roll plan")
        if plan.stage is not RollStage.REVIEW_REQUIRED:
            raise RuntimeError(f"roll cannot be approved from stage {plan.stage.value!r}")
        invalid_previews = [record.frame for record in plan.frames if not _aligned_record_valid(path.parent, record)]
        if invalid_previews:
            joined = ", ".join(str(frame) for frame in invalid_previews)
            raise RuntimeError(f"roll cannot be approved: aligned preview is missing or invalid for frame(s) {joined}")
        raw_overrides = tuple(allow_unverified_frames)
        if any(type(frame) is not int or frame < 1 for frame in raw_overrides):
            raise ValueError("visual approval overrides must be positive integers")
        overrides = set(raw_overrides)
        available = {record.frame for record in plan.frames}
        unknown = sorted(overrides - available)
        if unknown:
            raise ValueError("visual approval overrides are absent from the plan: " + ", ".join(str(frame) for frame in unknown))
        failed = [
            record.frame
            for record in plan.frames
            if (record.frame not in overrides and (record.verification is None or not record.verification.passed))
        ]
        if failed:
            joined = ", ".join(str(frame) for frame in failed)
            raise RuntimeError(f"roll cannot be approved: alignment verification failed for {joined}")
        approved = replace(
            plan,
            stage=RollStage.APPROVED,
            approved=True,
            visual_override_frames=tuple(sorted(overrides)),
        )
        _atomic_write_json(path, approved.to_dict())
        return approved

    def scan_fine(
        self,
        *,
        plan_path: str | Path,
        fine_params: ScanParams,
        stop: threading.Event,
        frames: Sequence[int] | None = None,
        progress: ProgressCallback | None = None,
        allow_indeterminate_smear_frames: Sequence[int] = (),
    ) -> RollScanPlan:
        """Scan approved frames at fine resolution using saved geometry."""

        path = Path(plan_path).expanduser().resolve()
        plan = _read_plan(path)
        if plan.registration_signature != self._current_registration_signature():
            raise ValueError("registration policy differs from the saved roll plan")
        if not plan.approved or plan.stage not in (
            RollStage.APPROVED,
            RollStage.COMPLETE,
        ):
            raise RuntimeError("fine scan refused: approve the aligned previews first")
        invalid_previews = [record.frame for record in plan.frames if not _aligned_record_valid(path.parent, record)]
        if invalid_previews:
            joined = ", ".join(str(frame) for frame in invalid_previews)
            raise RuntimeError(
                f"fine scan refused: aligned preview is missing or invalid for frame(s) {joined}; run preview preparation and review again"
            )
        if fine_params.area is not None:
            raise ValueError("registered fine scans require area=None")
        recipe = FineScanRecipe.from_params(fine_params)

        available = tuple(record.frame for record in plan.frames)
        selected = available if frames is None else self._validate_frames(frames)
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError("fine-scan frames are absent from the approved plan: " + ", ".join(str(frame) for frame in unknown))
        override_frames = self._validate_frames(allow_indeterminate_smear_frames) if allow_indeterminate_smear_frames else ()
        unknown_overrides = sorted(set(override_frames) - set(selected))
        if unknown_overrides:
            raise ValueError(
                "indeterminate-smear override frames are absent from this fine-scan selection: "
                + ", ".join(str(frame) for frame in unknown_overrides)
            )

        root = path.parent
        transfer_cancel = threading.Event()
        records = list(plan.frames)
        record_indexes = {record.frame: index for index, record in enumerate(records)}
        requires_transfer = any(not _fine_record_valid(root, records[record_indexes[frame]], recipe, plan.device_id) for frame in selected)
        if requires_transfer and plan.stage is RollStage.COMPLETE:
            # Persist a schema-valid, resumable state before the first changed
            # artifact. Per-frame checkpoints may temporarily contain mixed
            # recipes while a whole-roll quality upgrade is in progress.
            plan = replace(plan, stage=RollStage.APPROVED)
            _atomic_write_json(path, plan.to_dict())
        for frame in selected:
            if stop.is_set():
                break
            index = record_indexes[frame]
            record = records[index]
            if _fine_record_valid(root, record, recipe, plan.device_id):
                continue
            registration = record.registration
            if registration is None:
                raise RuntimeError(f"approved roll plan has no registration for frame {frame}")
            params = replace(
                fine_params,
                frame=registration.geometry.frame or frame,
                registered_geometry=registration.geometry,
            )
            result = self._scanner.run_scan(
                plan.device_id,
                params,
                self._progress(progress, RollStage.FINE_SCANNING, frame),
                transfer_cancel,
            )
            if result.dpi != recipe.dpi:
                raise RuntimeError(f"frame {frame}: scanner returned {result.dpi} dpi for a {recipe.dpi} dpi request")
            if (result.ir is not None) != recipe.capture_ir:
                expected = "with" if recipe.capture_ir else "without"
                raise RuntimeError(f"frame {frame}: scanner result was not returned {expected} the requested IR channel")
            fine_filename = _fine_artifact_name(
                frame=frame,
                recipe=recipe,
                geometry=registration.geometry,
            )
            fine_path = self._write_preview(result, root / "fine" / fine_filename)
            fine_output = root / fine_path
            _fine_qc_path(fine_output).unlink(missing_ok=True)
            try:
                fine_rgb_shape, fine_dtype = _tiff_signature(fine_output)
                if result.ir is None:
                    fine_ir_shape = None
                    fine_ir_dtype = None
                else:
                    fine_ir_shape, fine_ir_dtype = _tiff_signature(fine_output.with_name(f"{fine_output.stem}_IR.tif"))
            except (OSError, ValueError, tifffile.TiffFileError) as exc:
                raise RuntimeError(f"frame {frame}: fine RGB artifact failed committed TIFF metadata QC") from exc
            self._write_fine_qc(
                root=root,
                output=fine_output,
                frame=frame,
                device_id=plan.device_id,
                recipe=recipe,
                geometry=registration.geometry,
                result=result,
                allow_indeterminate_smear=frame in override_frames,
            )
            records[index] = replace(
                record,
                fine_path=fine_path,
                fine_rgb_shape=fine_rgb_shape,
                fine_ir_shape=fine_ir_shape,
                fine_dtype=fine_dtype,
                fine_ir_dtype=fine_ir_dtype,
                fine_dpi=result.dpi,
                fine_recipe=recipe,
            )
            plan = replace(plan, frames=tuple(records))
            _atomic_write_json(path, plan.to_dict())

        stage = (
            RollStage.COMPLETE
            if all(_fine_record_valid(root, record, recipe, plan.device_id) for record in records)
            else RollStage.APPROVED
        )
        plan = replace(plan, stage=stage, frames=tuple(records))
        _atomic_write_json(path, plan.to_dict())
        return plan

    def scan_full_negative(
        self,
        *,
        plan_path: str | Path,
        output_dir: str | Path,
        identity: PlanIdentity,
        fine_params: ScanParams,
        stop: threading.Event,
        frames: Sequence[int] | None = None,
        progress: ProgressCallback | None = None,
        pitch_rows: int = 5959,
    ) -> Mapping[int, FullNegativeWorkflowResult]:
        """Capture approved frames as full-field RGB4x + IR source pairs."""

        from negpy.services.scanning.full_negative_workflow import FullNegativeWorkflow

        path = Path(plan_path).expanduser().resolve()
        plan = _read_plan(path)
        if plan.registration_signature != self._current_registration_signature():
            raise ValueError("registration policy differs from the saved roll plan")
        if not plan.approved or plan.stage not in (RollStage.APPROVED, RollStage.COMPLETE):
            raise RuntimeError("full-negative scan refused: approve the aligned previews first")
        if identity != plan.identity:
            raise RuntimeError("full-negative identity does not match the exact approved roll plan")
        if not plan.device_id.startswith("coolscan3:"):
            raise RuntimeError("full-negative capture currently requires a direct coolscan3 device")
        invalid_previews = [record.frame for record in plan.frames if not _aligned_record_valid(path.parent, record)]
        if invalid_previews:
            joined = ", ".join(str(frame) for frame in invalid_previews)
            raise RuntimeError(f"full-negative scan refused: aligned preview is missing or invalid for frame(s) {joined}")
        if (
            fine_params.dpi != 4000
            or fine_params.depth != 16
            or not fine_params.capture_ir
            or fine_params.samples_per_scan != 4
            or not fine_params.autofocus
            or not fine_params.auto_exposure
            or fine_params.area is not None
        ):
            raise ValueError("full-negative scan requires 4000 dpi, 16-bit, RGB4x+IR, autofocus, auto-exposure, and area=None")

        available = tuple(record.frame for record in plan.frames)
        selected = available if frames is None else self._validate_frames(frames)
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError("full-negative frames are absent from the approved plan: " + ", ".join(str(frame) for frame in unknown))
        records = {record.frame: record for record in plan.frames}
        approval_binding = approved_plan_binding_sha256(plan)
        output_root = Path(output_dir).expanduser().resolve()
        checkpoint_path = output_root / "roll-full-negative.json"
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"could not read full-negative roll checkpoint: {exc}") from exc
            if (
                type(checkpoint) is not dict
                or checkpoint.get("version") != 1
                or checkpoint.get("identity") != identity.to_dict()
                or checkpoint.get("approved_plan_binding_sha256") != approval_binding
                or checkpoint.get("approved_frames") != list(available)
                or checkpoint.get("requested_frames") != list(selected)
            ):
                raise RuntimeError("full-negative roll checkpoint belongs to a different approved roll plan")
        workflow = FullNegativeWorkflow(
            scanner=self._scanner,
            output_dir=output_dir,
            identity=identity,
            approved_plan_binding_sha256=approval_binding,
            pitch_rows=pitch_rows,
        )
        completed: dict[int, FullNegativeWorkflowResult] = {}

        def write_checkpoint() -> None:
            manifest_hashes = {
                str(frame): sha256(result.manifest_path.read_bytes()).hexdigest() for frame, result in sorted(completed.items())
            }
            completed_frames = sorted(completed)
            _atomic_write_json(
                checkpoint_path,
                {
                    "version": 1,
                    "identity": identity.to_dict(),
                    "approved_plan_binding_sha256": approval_binding,
                    "approved_frames": list(available),
                    "requested_frames": list(selected),
                    "completed_frames": completed_frames,
                    "complete": completed_frames == list(selected),
                    "manifest_sha256": manifest_hashes,
                },
            )

        write_checkpoint()
        for frame in selected:
            if stop.is_set():
                break
            registration = records[frame].registration
            if registration is None:
                raise RuntimeError(f"approved roll plan has no registration for frame {frame}")
            completed[frame] = workflow.capture_frame(
                device_id=plan.device_id,
                registration=registration,
                recipe=fine_params,
                stop=stop,
                progress=self._progress(progress, RollStage.FINE_SCANNING, frame),
            )
            write_checkpoint()
        if tuple(completed) != selected:
            missing = sorted(set(selected) - set(completed))
            raise RuntimeError(
                "full-negative roll is incomplete; live QC-valid manifests are missing for frame(s) "
                + ", ".join(str(frame) for frame in missing)
            )
        return completed

    def eject(self, device_id: str) -> bool:
        """Trigger a capability-gated film eject; False when unsupported.

        Thin passthrough to the wrapped ScannerService so callers that only
        hold a RollScanService (e.g. the roll-scan CLI) do not need direct
        access to the private scanner boundary.
        """
        return self._scanner.eject(device_id)
