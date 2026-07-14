"""Strict, content-bound provenance records for full-negative acquisition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256 as _sha256
from math import isfinite
from pathlib import Path, PureWindowsPath
from typing import Mapping, Protocol, cast
from uuid import UUID


def _exact_dict(value: object, keys: set[str], field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    payload = cast(dict[str, object], value)
    if set(payload) != keys:
        missing = sorted(keys - set(payload))
        extra = sorted(set(payload) - keys)
        raise ValueError(f"{field} keys differ (missing={missing}, extra={extra})")
    return payload


def _canonical_uuid(value: object, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise ValueError(f"{field} must be a canonical UUID string")
    return value


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    return value


def _strict_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _strict_string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _positive_float(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be finite and positive")
    number = cast(int | float, value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return float(number)


def _nonnegative_float(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be finite and non-negative")
    number = cast(int | float, value)
    if not isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return float(number)


def _sha256_hex(value: object, field: str) -> str:
    digest = _strict_string(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return digest


def _relative_path(value: object, field: str) -> str:
    path = _strict_string(value, field)
    native = Path(path)
    windows = PureWindowsPath(path)
    if native.is_absolute() or windows.is_absolute() or ".." in native.parts or ".." in windows.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path


def _shape(value: object, field: str) -> tuple[int, ...]:
    if type(value) not in (list, tuple) or not value:
        raise ValueError(f"{field} must be a nonempty array")
    items = cast(list[object] | tuple[object, ...], value)
    return tuple(_strict_int(item, f"{field}[{index}]", minimum=1) for index, item in enumerate(items))


def _resolution(value: object, field: str) -> tuple[int, int]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{field} must be a numerator/denominator pair")
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) != 2:
        raise ValueError(f"{field} must be a numerator/denominator pair")
    return (
        _strict_int(items[0], f"{field}[0]", minimum=1),
        _strict_int(items[1], f"{field}[1]", minimum=1),
    )


class _SemanticRecord(Protocol):
    def semantic_dict(self) -> Mapping[str, object]: ...


def canonical_semantic_sha256(value: _SemanticRecord | Mapping[str, object]) -> str:
    """Hash canonical semantic JSON, deliberately excluding storage location."""

    if isinstance(value, Mapping):
        semantic = value
    else:
        semantic = value.semantic_dict()
    try:
        encoded = json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"semantic value is not strict JSON: {exc}") from exc
    return _sha256(encoded).hexdigest()


def _json_safe_tree(value: object, field: str) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return value
    if type(value) in (list, tuple):
        items = cast(list[object] | tuple[object, ...], value)
        return [_json_safe_tree(item, f"{field}[]") for item in items]
    if type(value) is dict:
        payload = cast(dict[object, object], value)
        if not all(type(key) is str for key in payload):
            raise ValueError(f"{field} keys must be strings")
        return {cast(str, key): _json_safe_tree(item, f"{field}.{key}") for key, item in payload.items()}
    raise ValueError(f"{field} contains unsupported JSON data")


def _json_object(value: object, field: str, *, nonempty: bool = True) -> dict[str, object]:
    safe = _json_safe_tree(value, field)
    if type(safe) is not dict or (nonempty and not safe):
        adjective = "a nonempty" if nonempty else "an"
        raise ValueError(f"{field} must be {adjective} object")
    return cast(dict[str, object], safe)


@dataclass(frozen=True)
class PlanIdentity:
    """Stable roll, physical insertion, and plan-session identities."""

    roll_id: str
    insertion_id: str
    session_id: str

    def __post_init__(self) -> None:
        _canonical_uuid(self.roll_id, "roll_id")
        _canonical_uuid(self.insertion_id, "insertion_id")
        _canonical_uuid(self.session_id, "session_id")
        if len({self.roll_id, self.insertion_id, self.session_id}) != 3:
            raise ValueError("roll_id, insertion_id, and session_id must be distinct")

    def to_dict(self) -> dict[str, object]:
        return {
            "roll_id": self.roll_id,
            "insertion_id": self.insertion_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> PlanIdentity:
        payload = _exact_dict(value, {"roll_id", "insertion_id", "session_id"}, "identity")
        return cls(
            roll_id=_canonical_uuid(payload["roll_id"], "identity.roll_id"),
            insertion_id=_canonical_uuid(payload["insertion_id"], "identity.insertion_id"),
            session_id=_canonical_uuid(payload["session_id"], "identity.session_id"),
        )


@dataclass(frozen=True)
class ArtifactEvidence:
    """Verified TIFF evidence; ``path`` is locator data, not identity."""

    path: str
    sha256: str
    byte_length: int
    shape: tuple[int, ...]
    dtype: str
    page_count: int
    payload_within_file: bool
    x_resolution: tuple[int, int]
    y_resolution: tuple[int, int]
    resolution_unit: str

    def __post_init__(self) -> None:
        _relative_path(self.path, "path")
        _sha256_hex(self.sha256, "sha256")
        _strict_int(self.byte_length, "byte_length", minimum=1)
        _shape(self.shape, "shape")
        _strict_string(self.dtype, "dtype")
        _strict_int(self.page_count, "page_count", minimum=1)
        _strict_bool(self.payload_within_file, "payload_within_file")
        _resolution(self.x_resolution, "x_resolution")
        _resolution(self.y_resolution, "y_resolution")
        if self.resolution_unit != "INCH":
            raise ValueError("resolution_unit must be 'INCH'")

    def semantic_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "page_count": self.page_count,
            "payload_within_file": self.payload_within_file,
            "x_resolution": list(self.x_resolution),
            "y_resolution": list(self.y_resolution),
            "resolution_unit": self.resolution_unit,
        }

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, **self.semantic_dict()}

    @classmethod
    def from_dict(cls, value: object) -> ArtifactEvidence:
        keys = {
            "path",
            "sha256",
            "byte_length",
            "shape",
            "dtype",
            "page_count",
            "payload_within_file",
            "x_resolution",
            "y_resolution",
            "resolution_unit",
        }
        payload = _exact_dict(value, keys, "artifact")
        return cls(
            path=_relative_path(payload["path"], "artifact.path"),
            sha256=_sha256_hex(payload["sha256"], "artifact.sha256"),
            byte_length=_strict_int(payload["byte_length"], "artifact.byte_length", minimum=1),
            shape=_shape(payload["shape"], "artifact.shape"),
            dtype=_strict_string(payload["dtype"], "artifact.dtype"),
            page_count=_strict_int(payload["page_count"], "artifact.page_count", minimum=1),
            payload_within_file=_strict_bool(payload["payload_within_file"], "artifact.payload_within_file"),
            x_resolution=_resolution(payload["x_resolution"], "artifact.x_resolution"),
            y_resolution=_resolution(payload["y_resolution"], "artifact.y_resolution"),
            resolution_unit=_strict_string(payload["resolution_unit"], "artifact.resolution_unit"),
        )


@dataclass(frozen=True)
class CaptureStateEvidence:
    """Focus/exposure state measured once or replayed from a bound capture."""

    mode: str
    replayed_from_capture_id: str | None
    focus_position: int
    exposure_multiplier: float
    red_exposure_us: float
    green_exposure_us: float
    blue_exposure_us: float

    def __post_init__(self) -> None:
        if self.mode not in {"measured", "replayed"}:
            raise ValueError("capture state mode must be 'measured' or 'replayed'")
        if self.mode == "measured" and self.replayed_from_capture_id is not None:
            raise ValueError("measured capture state cannot name a replay source")
        if self.mode == "replayed":
            _sha256_hex(self.replayed_from_capture_id, "replayed_from_capture_id")
        _strict_int(self.focus_position, "focus_position", minimum=0)
        for field in (
            "exposure_multiplier",
            "red_exposure_us",
            "green_exposure_us",
            "blue_exposure_us",
        ):
            _positive_float(getattr(self, field), field)

    def values_dict(self) -> dict[str, object]:
        return {
            "focus_position": self.focus_position,
            "exposure_multiplier": self.exposure_multiplier,
            "red_exposure_us": self.red_exposure_us,
            "green_exposure_us": self.green_exposure_us,
            "blue_exposure_us": self.blue_exposure_us,
        }

    def semantic_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "replayed_from_capture_id": self.replayed_from_capture_id,
            **self.values_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return self.semantic_dict()

    @classmethod
    def from_dict(cls, value: object) -> CaptureStateEvidence:
        keys = {
            "mode",
            "replayed_from_capture_id",
            "focus_position",
            "exposure_multiplier",
            "red_exposure_us",
            "green_exposure_us",
            "blue_exposure_us",
        }
        payload = _exact_dict(value, keys, "capture_state")
        replayed_from = payload["replayed_from_capture_id"]
        if replayed_from is not None:
            replayed_from = _sha256_hex(replayed_from, "capture_state.replayed_from_capture_id")
        return cls(
            mode=_strict_string(payload["mode"], "capture_state.mode"),
            replayed_from_capture_id=replayed_from,
            focus_position=_strict_int(payload["focus_position"], "capture_state.focus_position", minimum=0),
            exposure_multiplier=_positive_float(payload["exposure_multiplier"], "capture_state.exposure_multiplier"),
            red_exposure_us=_positive_float(payload["red_exposure_us"], "capture_state.red_exposure_us"),
            green_exposure_us=_positive_float(payload["green_exposure_us"], "capture_state.green_exposure_us"),
            blue_exposure_us=_positive_float(payload["blue_exposure_us"], "capture_state.blue_exposure_us"),
        )


_SOURCE_ARTIFACT_ROLES = {
    "rgb4x",
    "rgb1x_proxy",
    "ir1x",
    "aligned_ir",
    "ir_valid_mask",
}


def _capture_geometry(value: object, field: str, *, physical_frame: int) -> dict[str, object]:
    payload = _exact_dict(
        value,
        {"physical_frame", "subframe_mm", "br_y_device_px"},
        field,
    )
    geometry_frame = _strict_int(payload["physical_frame"], f"{field}.physical_frame", minimum=1)
    if geometry_frame != physical_frame:
        raise ValueError(f"{field}.physical_frame disagrees with source physical_frame")
    return {
        "physical_frame": geometry_frame,
        "subframe_mm": _nonnegative_float(payload["subframe_mm"], f"{field}.subframe_mm"),
        "br_y_device_px": _strict_int(payload["br_y_device_px"], f"{field}.br_y_device_px", minimum=0),
    }


@dataclass(frozen=True)
class SourceCaptureRecord:
    """One immutable physical feeder-window capture and all of its evidence."""

    identity: PlanIdentity
    physical_frame: int
    geometry: Mapping[str, object]
    recipe: Mapping[str, object]
    capture_state: CaptureStateEvidence
    passes: tuple[Mapping[str, object], ...]
    artifacts: Mapping[str, ArtifactEvidence]
    split_alignment: Mapping[str, object]
    binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PlanIdentity):
            raise ValueError("source identity must be a PlanIdentity")
        _strict_int(self.physical_frame, "physical_frame", minimum=1)
        _capture_geometry(self.geometry, "geometry", physical_frame=self.physical_frame)
        _json_object(self.recipe, "recipe")
        if type(self.passes) is not tuple or not self.passes:
            raise ValueError("passes must be a nonempty tuple")
        for index, item in enumerate(self.passes):
            _json_object(item, f"passes[{index}]")
        if not isinstance(self.capture_state, CaptureStateEvidence):
            raise ValueError("capture_state must be CaptureStateEvidence")
        if type(self.artifacts) is not dict or set(self.artifacts) != _SOURCE_ARTIFACT_ROLES:
            raise ValueError(f"artifacts must contain exactly {sorted(_SOURCE_ARTIFACT_ROLES)}")
        if not all(isinstance(item, ArtifactEvidence) for item in self.artifacts.values()):
            raise ValueError("every source artifact must be ArtifactEvidence")
        _json_object(self.split_alignment, "split_alignment")
        _sha256_hex(self.binding_sha256, "binding_sha256")
        if self.binding_sha256 != canonical_semantic_sha256(self):
            raise ValueError("source capture binding does not match its semantic content")

    @property
    def capture_id(self) -> str:
        return self.binding_sha256

    def semantic_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "physical_frame": self.physical_frame,
            "geometry": _json_safe_tree(self.geometry, "geometry"),
            "recipe": _json_safe_tree(self.recipe, "recipe"),
            "capture_state": self.capture_state.semantic_dict(),
            "passes": _json_safe_tree(self.passes, "passes"),
            "artifacts": {role: self.artifacts[role].semantic_dict() for role in sorted(_SOURCE_ARTIFACT_ROLES)},
            "split_alignment": _json_safe_tree(self.split_alignment, "split_alignment"),
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.semantic_dict()
        payload["artifacts"] = {role: self.artifacts[role].to_dict() for role in sorted(_SOURCE_ARTIFACT_ROLES)}
        payload["binding_sha256"] = self.binding_sha256
        return payload

    @classmethod
    def create(
        cls,
        *,
        identity: PlanIdentity,
        physical_frame: int,
        geometry: Mapping[str, object],
        recipe: Mapping[str, object],
        capture_state: CaptureStateEvidence,
        passes: tuple[Mapping[str, object], ...],
        artifacts: Mapping[str, ArtifactEvidence],
        split_alignment: Mapping[str, object],
    ) -> SourceCaptureRecord:
        normalized_geometry = _capture_geometry(
            geometry,
            "geometry",
            physical_frame=_strict_int(physical_frame, "physical_frame", minimum=1),
        )
        normalized_recipe = _json_object(recipe, "recipe")
        normalized_passes = tuple(_json_object(item, f"passes[{index}]") for index, item in enumerate(passes))
        normalized_artifacts = dict(artifacts)
        normalized_alignment = _json_object(split_alignment, "split_alignment")
        semantic = {
            "identity": identity.to_dict(),
            "physical_frame": physical_frame,
            "geometry": normalized_geometry,
            "recipe": normalized_recipe,
            "capture_state": capture_state.semantic_dict(),
            "passes": list(normalized_passes),
            "artifacts": {role: normalized_artifacts[role].semantic_dict() for role in sorted(_SOURCE_ARTIFACT_ROLES)},
            "split_alignment": normalized_alignment,
        }
        return cls(
            identity=identity,
            physical_frame=physical_frame,
            geometry=normalized_geometry,
            recipe=normalized_recipe,
            capture_state=capture_state,
            passes=normalized_passes,
            artifacts=normalized_artifacts,
            split_alignment=normalized_alignment,
            binding_sha256=canonical_semantic_sha256(semantic),
        )

    @classmethod
    def from_dict(cls, value: object) -> SourceCaptureRecord:
        keys = {
            "identity",
            "physical_frame",
            "geometry",
            "recipe",
            "capture_state",
            "passes",
            "artifacts",
            "split_alignment",
            "binding_sha256",
        }
        payload = _exact_dict(value, keys, "source_capture")
        physical_frame = _strict_int(payload["physical_frame"], "source_capture.physical_frame", minimum=1)
        raw_passes = payload["passes"]
        if type(raw_passes) is not list or not raw_passes:
            raise ValueError("source_capture.passes must be a nonempty array")
        raw_artifacts = _exact_dict(
            payload["artifacts"],
            _SOURCE_ARTIFACT_ROLES,
            "source_capture.artifacts",
        )
        return cls(
            identity=PlanIdentity.from_dict(payload["identity"]),
            physical_frame=physical_frame,
            geometry=_capture_geometry(
                payload["geometry"],
                "source_capture.geometry",
                physical_frame=physical_frame,
            ),
            recipe=_json_object(payload["recipe"], "source_capture.recipe"),
            capture_state=CaptureStateEvidence.from_dict(payload["capture_state"]),
            passes=tuple(_json_object(item, f"source_capture.passes[{index}]") for index, item in enumerate(raw_passes)),
            artifacts={role: ArtifactEvidence.from_dict(raw_artifacts[role]) for role in sorted(_SOURCE_ARTIFACT_ROLES)},
            split_alignment=_json_object(payload["split_alignment"], "source_capture.split_alignment"),
            binding_sha256=_sha256_hex(payload["binding_sha256"], "source_capture.binding_sha256"),
        )


@dataclass(frozen=True)
class SourcePairRecord:
    """Ordered adjacent physical captures bound to one logical exposure."""

    identity: PlanIdentity
    logical_frame: int
    first_capture_id: str
    second_capture_id: str
    first_physical_frame: int
    second_physical_frame: int
    binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PlanIdentity):
            raise ValueError("source pair identity must be a PlanIdentity")
        _strict_int(self.logical_frame, "logical_frame", minimum=1)
        _sha256_hex(self.first_capture_id, "first_capture_id")
        _sha256_hex(self.second_capture_id, "second_capture_id")
        if self.first_capture_id == self.second_capture_id:
            raise ValueError("source pair capture IDs must be distinct")
        _strict_int(self.first_physical_frame, "first_physical_frame", minimum=1)
        _strict_int(self.second_physical_frame, "second_physical_frame", minimum=1)
        if self.second_physical_frame != self.first_physical_frame + 1:
            raise ValueError("source pair physical frames must be consecutive and ordered")
        _sha256_hex(self.binding_sha256, "binding_sha256")
        if self.binding_sha256 != canonical_semantic_sha256(self):
            raise ValueError("source pair binding does not match its semantic content")

    @property
    def pair_id(self) -> str:
        return self.binding_sha256

    def semantic_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "logical_frame": self.logical_frame,
            "first_capture_id": self.first_capture_id,
            "second_capture_id": self.second_capture_id,
            "first_physical_frame": self.first_physical_frame,
            "second_physical_frame": self.second_physical_frame,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_dict(), "binding_sha256": self.binding_sha256}

    def validate_sources(
        self,
        first: SourceCaptureRecord,
        second: SourceCaptureRecord,
    ) -> None:
        if first.identity != self.identity or second.identity != self.identity:
            raise ValueError("source pair captures must share the pair identity")
        if first.capture_id != self.first_capture_id or second.capture_id != self.second_capture_id:
            raise ValueError("source pair capture binding does not resolve to its ordered sources")
        if first.physical_frame != self.first_physical_frame or second.physical_frame != self.second_physical_frame:
            raise ValueError("source pair physical frames disagree with their captures")
        if first.recipe != second.recipe:
            raise ValueError("source pair captures must use the same recipe")
        if first.capture_state.mode != "measured":
            raise ValueError("the first source capture must contain measured state")
        if second.capture_state.mode != "replayed":
            raise ValueError("the second source capture must replay measured state")
        if second.capture_state.replayed_from_capture_id != first.capture_id:
            raise ValueError("the second source capture must replay the first capture")
        if second.capture_state.values_dict() != first.capture_state.values_dict():
            raise ValueError("source pair focus/exposure state differs")

    @classmethod
    def create(
        cls,
        *,
        logical_frame: int,
        first: SourceCaptureRecord,
        second: SourceCaptureRecord,
    ) -> SourcePairRecord:
        if not isinstance(first, SourceCaptureRecord) or not isinstance(second, SourceCaptureRecord):
            raise ValueError("source pair requires two SourceCaptureRecord values")
        semantic = {
            "identity": first.identity.to_dict(),
            "logical_frame": _strict_int(logical_frame, "logical_frame", minimum=1),
            "first_capture_id": first.capture_id,
            "second_capture_id": second.capture_id,
            "first_physical_frame": first.physical_frame,
            "second_physical_frame": second.physical_frame,
        }
        record = cls(
            identity=first.identity,
            logical_frame=logical_frame,
            first_capture_id=first.capture_id,
            second_capture_id=second.capture_id,
            first_physical_frame=first.physical_frame,
            second_physical_frame=second.physical_frame,
            binding_sha256=canonical_semantic_sha256(semantic),
        )
        record.validate_sources(first, second)
        return record

    @classmethod
    def from_dict(cls, value: object) -> SourcePairRecord:
        keys = {
            "identity",
            "logical_frame",
            "first_capture_id",
            "second_capture_id",
            "first_physical_frame",
            "second_physical_frame",
            "binding_sha256",
        }
        payload = _exact_dict(value, keys, "source_pair")
        return cls(
            identity=PlanIdentity.from_dict(payload["identity"]),
            logical_frame=_strict_int(payload["logical_frame"], "source_pair.logical_frame", minimum=1),
            first_capture_id=_sha256_hex(payload["first_capture_id"], "source_pair.first_capture_id"),
            second_capture_id=_sha256_hex(payload["second_capture_id"], "source_pair.second_capture_id"),
            first_physical_frame=_strict_int(
                payload["first_physical_frame"],
                "source_pair.first_physical_frame",
                minimum=1,
            ),
            second_physical_frame=_strict_int(
                payload["second_physical_frame"],
                "source_pair.second_physical_frame",
                minimum=1,
            ),
            binding_sha256=_sha256_hex(payload["binding_sha256"], "source_pair.binding_sha256"),
        )


_RECONSTRUCTION_ARTIFACT_ROLES = {"rgb", "ir", "ir_valid_mask"}


@dataclass(frozen=True)
class ReconstructionBindingRecord:
    """Derived full-negative raster bound to sources, crop policy, and outputs."""

    identity: PlanIdentity
    logical_frame: int
    source_pair_binding_sha256: str
    registration_binding_sha256: str
    algorithm_signature: Mapping[str, object]
    pitch_rows: int
    crop_start_row: int
    artifacts: Mapping[str, ArtifactEvidence]
    binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PlanIdentity):
            raise ValueError("reconstruction identity must be a PlanIdentity")
        _strict_int(self.logical_frame, "logical_frame", minimum=1)
        _sha256_hex(self.source_pair_binding_sha256, "source_pair_binding_sha256")
        _sha256_hex(self.registration_binding_sha256, "registration_binding_sha256")
        _json_object(self.algorithm_signature, "algorithm_signature")
        _strict_int(self.pitch_rows, "pitch_rows", minimum=1)
        _strict_int(self.crop_start_row, "crop_start_row", minimum=0)
        if self.crop_start_row >= self.pitch_rows:
            raise ValueError("crop_start_row must be smaller than pitch_rows")
        if type(self.artifacts) is not dict or set(self.artifacts) != _RECONSTRUCTION_ARTIFACT_ROLES:
            raise ValueError(f"reconstruction artifacts must contain exactly {sorted(_RECONSTRUCTION_ARTIFACT_ROLES)}")
        if not all(isinstance(item, ArtifactEvidence) for item in self.artifacts.values()):
            raise ValueError("every reconstruction artifact must be ArtifactEvidence")
        _sha256_hex(self.binding_sha256, "binding_sha256")
        if self.binding_sha256 != canonical_semantic_sha256(self):
            raise ValueError("reconstruction binding does not match its semantic content")

    @property
    def reconstruction_id(self) -> str:
        return self.binding_sha256

    def semantic_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "logical_frame": self.logical_frame,
            "source_pair_binding_sha256": self.source_pair_binding_sha256,
            "registration_binding_sha256": self.registration_binding_sha256,
            "algorithm_signature": _json_safe_tree(self.algorithm_signature, "algorithm_signature"),
            "pitch_rows": self.pitch_rows,
            "crop_start_row": self.crop_start_row,
            "artifacts": {role: self.artifacts[role].semantic_dict() for role in sorted(_RECONSTRUCTION_ARTIFACT_ROLES)},
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.semantic_dict()
        payload["artifacts"] = {role: self.artifacts[role].to_dict() for role in sorted(_RECONSTRUCTION_ARTIFACT_ROLES)}
        payload["binding_sha256"] = self.binding_sha256
        return payload

    @classmethod
    def create(
        cls,
        *,
        identity: PlanIdentity,
        logical_frame: int,
        source_pair: SourcePairRecord,
        registration_binding_sha256: str,
        algorithm_signature: Mapping[str, object],
        pitch_rows: int,
        crop_start_row: int,
        artifacts: Mapping[str, ArtifactEvidence],
    ) -> ReconstructionBindingRecord:
        if source_pair.identity != identity or source_pair.logical_frame != logical_frame:
            raise ValueError("source pair identity/logical frame disagrees with reconstruction")
        registration_binding = _sha256_hex(registration_binding_sha256, "registration_binding_sha256")
        algorithm = _json_object(algorithm_signature, "algorithm_signature")
        normalized_artifacts = dict(artifacts)
        semantic = {
            "identity": identity.to_dict(),
            "logical_frame": _strict_int(logical_frame, "logical_frame", minimum=1),
            "source_pair_binding_sha256": source_pair.binding_sha256,
            "registration_binding_sha256": registration_binding,
            "algorithm_signature": algorithm,
            "pitch_rows": _strict_int(pitch_rows, "pitch_rows", minimum=1),
            "crop_start_row": _strict_int(crop_start_row, "crop_start_row", minimum=0),
            "artifacts": {role: normalized_artifacts[role].semantic_dict() for role in sorted(_RECONSTRUCTION_ARTIFACT_ROLES)},
        }
        return cls(
            identity=identity,
            logical_frame=logical_frame,
            source_pair_binding_sha256=source_pair.binding_sha256,
            registration_binding_sha256=registration_binding,
            algorithm_signature=algorithm,
            pitch_rows=pitch_rows,
            crop_start_row=crop_start_row,
            artifacts=normalized_artifacts,
            binding_sha256=canonical_semantic_sha256(semantic),
        )

    @classmethod
    def from_dict(cls, value: object) -> ReconstructionBindingRecord:
        keys = {
            "identity",
            "logical_frame",
            "source_pair_binding_sha256",
            "registration_binding_sha256",
            "algorithm_signature",
            "pitch_rows",
            "crop_start_row",
            "artifacts",
            "binding_sha256",
        }
        payload = _exact_dict(value, keys, "reconstruction")
        raw_artifacts = _exact_dict(
            payload["artifacts"],
            _RECONSTRUCTION_ARTIFACT_ROLES,
            "reconstruction.artifacts",
        )
        return cls(
            identity=PlanIdentity.from_dict(payload["identity"]),
            logical_frame=_strict_int(payload["logical_frame"], "reconstruction.logical_frame", minimum=1),
            source_pair_binding_sha256=_sha256_hex(
                payload["source_pair_binding_sha256"],
                "reconstruction.source_pair_binding_sha256",
            ),
            registration_binding_sha256=_sha256_hex(
                payload["registration_binding_sha256"],
                "reconstruction.registration_binding_sha256",
            ),
            algorithm_signature=_json_object(payload["algorithm_signature"], "reconstruction.algorithm_signature"),
            pitch_rows=_strict_int(payload["pitch_rows"], "reconstruction.pitch_rows", minimum=1),
            crop_start_row=_strict_int(payload["crop_start_row"], "reconstruction.crop_start_row", minimum=0),
            artifacts={role: ArtifactEvidence.from_dict(raw_artifacts[role]) for role in sorted(_RECONSTRUCTION_ARTIFACT_ROLES)},
            binding_sha256=_sha256_hex(payload["binding_sha256"], "reconstruction.binding_sha256"),
        )


def _json_object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r}")


@dataclass(frozen=True)
class FullNegativeProvenanceManifest:
    """Manifest-last graph for one fully reconstructed logical negative."""

    identity: PlanIdentity
    captures: tuple[SourceCaptureRecord, SourceCaptureRecord]
    source_pair: SourcePairRecord
    reconstruction: ReconstructionBindingRecord
    binding_sha256: str
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported full-negative provenance manifest version")
        if not isinstance(self.identity, PlanIdentity):
            raise ValueError("manifest identity must be a PlanIdentity")
        if type(self.captures) is not tuple or len(self.captures) != 2:
            raise ValueError("manifest must contain exactly two source captures")
        first, second = self.captures
        if not isinstance(first, SourceCaptureRecord) or not isinstance(second, SourceCaptureRecord):
            raise ValueError("manifest captures must be SourceCaptureRecord values")
        if self.source_pair.identity != self.identity:
            raise ValueError("source pair identity disagrees with manifest")
        self.source_pair.validate_sources(first, second)
        if self.reconstruction.identity != self.identity:
            raise ValueError("reconstruction identity disagrees with manifest")
        if self.reconstruction.logical_frame != self.source_pair.logical_frame:
            raise ValueError("reconstruction logical frame disagrees with source pair")
        if self.reconstruction.source_pair_binding_sha256 != self.source_pair.binding_sha256:
            raise ValueError("reconstruction does not bind the manifest source pair")
        _sha256_hex(self.binding_sha256, "binding_sha256")
        if self.binding_sha256 != canonical_semantic_sha256(self):
            raise ValueError("manifest binding does not match its semantic graph")

    def semantic_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "identity": self.identity.to_dict(),
            "capture_ids": [capture.capture_id for capture in self.captures],
            "source_pair_binding_sha256": self.source_pair.binding_sha256,
            "reconstruction_binding_sha256": self.reconstruction.binding_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "identity": self.identity.to_dict(),
            "captures": [capture.to_dict() for capture in self.captures],
            "source_pair": self.source_pair.to_dict(),
            "reconstruction": self.reconstruction.to_dict(),
            "binding_sha256": self.binding_sha256,
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def create(
        cls,
        *,
        identity: PlanIdentity,
        captures: tuple[SourceCaptureRecord, SourceCaptureRecord],
        source_pair: SourcePairRecord,
        reconstruction: ReconstructionBindingRecord,
    ) -> FullNegativeProvenanceManifest:
        semantic = {
            "version": 1,
            "identity": identity.to_dict(),
            "capture_ids": [capture.capture_id for capture in captures],
            "source_pair_binding_sha256": source_pair.binding_sha256,
            "reconstruction_binding_sha256": reconstruction.binding_sha256,
        }
        return cls(
            identity=identity,
            captures=captures,
            source_pair=source_pair,
            reconstruction=reconstruction,
            binding_sha256=canonical_semantic_sha256(semantic),
        )

    @classmethod
    def from_dict(cls, value: object) -> FullNegativeProvenanceManifest:
        keys = {
            "version",
            "identity",
            "captures",
            "source_pair",
            "reconstruction",
            "binding_sha256",
        }
        payload = _exact_dict(value, keys, "manifest")
        version = _strict_int(payload["version"], "manifest.version", minimum=1)
        raw_captures = payload["captures"]
        if type(raw_captures) is not list or len(raw_captures) != 2:
            raise ValueError("manifest.captures must contain exactly two records")
        captures = (
            SourceCaptureRecord.from_dict(raw_captures[0]),
            SourceCaptureRecord.from_dict(raw_captures[1]),
        )
        return cls(
            version=version,
            identity=PlanIdentity.from_dict(payload["identity"]),
            captures=captures,
            source_pair=SourcePairRecord.from_dict(payload["source_pair"]),
            reconstruction=ReconstructionBindingRecord.from_dict(payload["reconstruction"]),
            binding_sha256=_sha256_hex(payload["binding_sha256"], "manifest.binding_sha256"),
        )

    @classmethod
    def from_json(cls, value: str) -> FullNegativeProvenanceManifest:
        if type(value) is not str:
            raise ValueError("manifest JSON must be a string")
        try:
            payload = json.loads(
                value,
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid full-negative provenance JSON: {exc}") from exc
        return cls.from_dict(payload)


__all__ = [
    "ArtifactEvidence",
    "CaptureStateEvidence",
    "FullNegativeProvenanceManifest",
    "PlanIdentity",
    "ReconstructionBindingRecord",
    "SourceCaptureRecord",
    "SourcePairRecord",
    "canonical_semantic_sha256",
]
