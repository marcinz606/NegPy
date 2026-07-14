"""Manifest-last full-negative acquisition and persistence workflow."""

from __future__ import annotations

import os
import tempfile
import threading
import fcntl
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, cast

import numpy as np
import tifffile

from negpy.infrastructure.scanners.params import ScanParams, ScannerCaptureState
from negpy.infrastructure.scanners.result import ScanResult, SplitIrAlignment, SplitSourceCapture
from negpy.services.scanning.full_negative_capture import (
    FullNegativeCaptureResult,
    FullNegativeCapturer,
    FullNegativeScanner,
)
from negpy.services.scanning.provenance import (
    ArtifactEvidence,
    CaptureStateEvidence,
    FullNegativeProvenanceManifest,
    PlanIdentity,
    ReconstructionBindingRecord,
    SourceCaptureRecord,
    SourcePairRecord,
    canonical_semantic_sha256,
)
from negpy.services.scanning.roll_service import FrameRegistration
from negpy.services.scanning.quality import inspect_tiff_payload
from negpy.services.scanning.writer import (
    write_full_negative_tiff,
    write_split_source_bundle,
)


class FullNegativeWorkflowScanner(FullNegativeScanner, Protocol):
    def probe_device(self, device_id: str): ...


@dataclass(frozen=True)
class FullNegativeWorkflowResult:
    manifest_path: Path
    provenance: FullNegativeProvenanceManifest
    capture: FullNegativeCaptureResult | None


def _artifact(evidence: Mapping[str, object], *, prefix: str) -> ArtifactEvidence:
    x_resolution = cast(list[int], evidence["x_resolution"])
    y_resolution = cast(list[int], evidence["y_resolution"])
    return ArtifactEvidence(
        path=f"{prefix}/{cast(str, evidence['path'])}",
        sha256=cast(str, evidence["sha256"]),
        byte_length=cast(int, evidence["bytes"]),
        shape=tuple(cast(list[int], evidence["shape"])),
        dtype=cast(str, evidence["dtype"]),
        page_count=cast(int, evidence["page_count"]),
        payload_within_file=cast(bool, evidence["payload_within_file"]),
        x_resolution=(x_resolution[0], x_resolution[1]),
        y_resolution=(y_resolution[0], y_resolution[1]),
        resolution_unit=cast(str, evidence["resolution_unit"]),
    )


def _capture_state(
    source: ScanResult,
    *,
    mode: str,
    replayed_from_capture_id: str | None,
) -> CaptureStateEvidence:
    state = source.capture_state
    if state is None:
        raise RuntimeError("validated full-negative source lost its focus/exposure state")
    return CaptureStateEvidence(
        mode=mode,
        replayed_from_capture_id=replayed_from_capture_id,
        focus_position=state.focus_position,
        exposure_multiplier=state.exposure_multiplier,
        red_exposure_us=state.red_exposure_us,
        green_exposure_us=state.green_exposure_us,
        blue_exposure_us=state.blue_exposure_us,
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _exclusive_output_lock(root: Path):
    """Refuse overlapping writers that could interleave source/final manifests."""

    root.mkdir(parents=True, exist_ok=True)
    path = root / ".full-negative.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"full-negative capture is already in progress for {root}") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _require_live_artifacts(root: Path, manifest: FullNegativeProvenanceManifest) -> None:
    evidence = []
    for capture in manifest.captures:
        evidence.extend(capture.artifacts.values())
    evidence.extend(manifest.reconstruction.artifacts.values())
    for artifact in evidence:
        try:
            live = inspect_tiff_payload(root / artifact.path)
        except Exception as exc:
            raise RuntimeError(f"committed full-negative artifact evidence is unreadable: {artifact.path}: {exc}") from exc
        if not (
            live.sha256 == artifact.sha256
            and live.byte_length == artifact.byte_length
            and tuple(live.shape) == artifact.shape
            and live.dtype == artifact.dtype
            and live.page_count == artifact.page_count
            and live.payload_within_file == artifact.payload_within_file
            and live.x_resolution == artifact.x_resolution
            and live.y_resolution == artifact.y_resolution
            and live.resolution_unit == artifact.resolution_unit
        ):
            raise RuntimeError(f"committed full-negative artifact evidence no longer matches: {artifact.path}")


def _require_source_artifacts(root: Path, record: SourceCaptureRecord) -> None:
    for artifact in record.artifacts.values():
        live = inspect_tiff_payload(root / artifact.path)
        if not (
            live.sha256 == artifact.sha256
            and live.byte_length == artifact.byte_length
            and tuple(live.shape) == artifact.shape
            and live.dtype == artifact.dtype
            and live.page_count == artifact.page_count
            and live.payload_within_file == artifact.payload_within_file
            and live.x_resolution == artifact.x_resolution
            and live.y_resolution == artifact.y_resolution
            and live.resolution_unit == artifact.resolution_unit
        ):
            raise RuntimeError(f"checkpointed source artifact evidence no longer matches: {artifact.path}")


def _source_from_checkpoint(root: Path, record: SourceCaptureRecord) -> ScanResult:
    _require_source_artifacts(root, record)
    artifacts = record.artifacts
    rgb4x = tifffile.imread(root / artifacts["rgb4x"].path)
    rgb1x_proxy = tifffile.imread(root / artifacts["rgb1x_proxy"].path)
    ir1x = tifffile.imread(root / artifacts["ir1x"].path)
    aligned_ir = tifffile.imread(root / artifacts["aligned_ir"].path)
    valid = tifffile.imread(root / artifacts["ir_valid_mask"].path) != 0
    state = record.capture_state
    alignment = dict(record.split_alignment)
    return ScanResult(
        rgb=np.asarray(rgb4x),
        ir=np.asarray(aligned_ir),
        dpi=cast(int, record.recipe["dpi"]),
        device_model="checkpointed Coolscan source",
        ir_alignment=SplitIrAlignment(
            mode=cast(str, alignment["mode"]),
            dx_px=float(cast(float, alignment["dx_px"])),
            dy_px=float(cast(float, alignment["dy_px"])),
            phase_responses=tuple(cast(list[float], alignment.get("phase_responses", ()))),
            channel_spread_px=cast(float | None, alignment.get("channel_spread_px")),
            ecc_coefficient=cast(float | None, alignment.get("ecc_coefficient")),
            tile_support_counts=tuple(cast(list[int], alignment.get("tile_support_counts", ()))),
            tile_shift_spread_px=cast(float | None, alignment.get("tile_shift_spread_px")),
            estimator_version=cast(int | None, alignment.get("estimator_version")),
            multiscale_max_dimensions=tuple(int(value) for value in cast(list[int], alignment.get("multiscale_max_dimensions", ()))),
            multiscale_channel_shifts_px=tuple(
                tuple((float(pair[0]), float(pair[1])) for pair in scale)
                for scale in cast(
                    list[list[list[float]]],
                    alignment.get("multiscale_channel_shifts_px", ()),
                )
            ),
            multiscale_responses=tuple(
                tuple(float(value) for value in scale) for scale in cast(list[list[float]], alignment.get("multiscale_responses", ()))
            ),
            multiscale_tile_support_counts=tuple(
                tuple(int(value) for value in scale)
                for scale in cast(
                    list[list[int]],
                    alignment.get("multiscale_tile_support_counts", ()),
                )
            ),
            multiscale_tile_shift_spreads_px=tuple(
                tuple(float(value) for value in scale)
                for scale in cast(
                    list[list[float]],
                    alignment.get("multiscale_tile_shift_spreads_px", ()),
                )
            ),
            multiscale_global_alias_shifts_px=tuple(
                tuple((float(pair[0]), float(pair[1])) for pair in scale)
                for scale in cast(
                    list[list[list[float]]],
                    alignment.get("multiscale_global_alias_shifts_px", ()),
                )
            ),
        ),
        ir_valid_mask=np.asarray(valid, dtype=np.bool_),
        capture_state=ScannerCaptureState(
            focus_position=state.focus_position,
            exposure_multiplier=state.exposure_multiplier,
            red_exposure_us=state.red_exposure_us,
            green_exposure_us=state.green_exposure_us,
            blue_exposure_us=state.blue_exposure_us,
        ),
        split_source=SplitSourceCapture(
            rgb4x=np.asarray(rgb4x),
            rgb1x_proxy=np.asarray(rgb1x_proxy),
            ir1x=np.asarray(ir1x),
        ),
    )


class _CheckpointFirstScanner:
    def __init__(self, scanner: FullNegativeWorkflowScanner, first: ScanResult) -> None:
        self._scanner = scanner
        self._first: ScanResult | None = first

    def run_scan(self, device_id, params, progress, cancel):
        if self._first is not None:
            result = self._first
            self._first = None
            progress(1.0)
            return result
        return self._scanner.run_scan(device_id, params, progress, cancel)


class FullNegativeWorkflow:
    """Capture, preserve raw masters, reconstruct, probe, then commit manifest."""

    def __init__(
        self,
        *,
        scanner: FullNegativeWorkflowScanner,
        output_dir: str | Path,
        identity: PlanIdentity,
        approved_plan_binding_sha256: str,
        pitch_rows: int = 5959,
    ) -> None:
        self._scanner = scanner
        self._root = Path(output_dir).expanduser().resolve()
        self._identity = identity
        if len(approved_plan_binding_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in approved_plan_binding_sha256
        ):
            raise ValueError("approved plan binding must be a lowercase SHA-256 digest")
        self._approved_plan_binding_sha256 = approved_plan_binding_sha256
        self._capturer = FullNegativeCapturer(scanner=scanner, pitch_rows=pitch_rows)
        self._pitch_rows = pitch_rows

    def capture_frame(
        self,
        *,
        device_id: str,
        registration: FrameRegistration,
        recipe: ScanParams,
        stop: threading.Event,
        progress: Callable[[float], None] | None = None,
    ) -> FullNegativeWorkflowResult:
        with _exclusive_output_lock(self._root):
            return self._capture_frame_locked(
                device_id=device_id,
                registration=registration,
                recipe=recipe,
                stop=stop,
                progress=progress,
            )

    def _capture_frame_locked(
        self,
        *,
        device_id: str,
        registration: FrameRegistration,
        recipe: ScanParams,
        stop: threading.Event,
        progress: Callable[[float], None] | None = None,
    ) -> FullNegativeWorkflowResult:
        logical_frame = registration.frame
        common_recipe = {
            "strategy": "coolscan-rgb4-rgbi1-bounded-tail-v1",
            "device_id": device_id,
            "dpi": recipe.dpi,
            "depth": recipe.depth,
            "capture_ir": recipe.capture_ir,
            "rgb_samples": recipe.samples_per_scan,
            "ir_samples": 1,
            "autofocus": recipe.autofocus,
            "auto_exposure": recipe.auto_exposure,
            "area": list(recipe.area) if recipe.area is not None else None,
            # FullNegativeCapturer forces this on for both internal source
            # reservations even though the public recipe keeps area=None.
            "preserve_full_canvas": True,
            "approved_plan_binding_sha256": self._approved_plan_binding_sha256,
        }
        checkpoint_path = self._root / "checkpoints" / f"frame{logical_frame:03d}-first-source.json"
        manifest_path = self._root / "manifests" / f"frame{logical_frame:03d}.json"
        if manifest_path.exists():
            existing = FullNegativeProvenanceManifest.from_json(manifest_path.read_text(encoding="utf-8"))
            if existing.identity != self._identity:
                raise RuntimeError("existing full-negative manifest belongs to another roll insertion/session")
            expected_registration = canonical_semantic_sha256(asdict(registration))
            if (
                existing.source_pair.logical_frame != logical_frame
                or existing.reconstruction.registration_binding_sha256 != expected_registration
                or existing.reconstruction.pitch_rows != self._pitch_rows
                or any(dict(capture.recipe) != common_recipe for capture in existing.captures)
            ):
                raise RuntimeError(f"frame {logical_frame}: existing full-negative manifest has a different request binding")
            _require_live_artifacts(self._root, existing)
            return FullNegativeWorkflowResult(
                manifest_path=manifest_path,
                provenance=existing,
                capture=None,
            )

        records: dict[str, SourceCaptureRecord] = {}
        checkpoint_source: ScanResult | None = None
        if checkpoint_path.exists():
            try:
                checkpoint_record = SourceCaptureRecord.from_dict(json.loads(checkpoint_path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"frame {logical_frame}: first-source checkpoint is invalid: {exc}") from exc
            expected_physical_frame = registration.geometry.frame
            if (
                checkpoint_record.identity != self._identity
                or checkpoint_record.physical_frame != expected_physical_frame
                or dict(checkpoint_record.recipe) != common_recipe
                or checkpoint_record.capture_state.mode != "measured"
                or checkpoint_record.geometry.get("subframe_mm") != 0.0
                or checkpoint_record.geometry.get("br_y_device_px") != self._pitch_rows - 1
            ):
                raise RuntimeError(f"frame {logical_frame}: first-source checkpoint has a different request binding")
            try:
                checkpoint_source = _source_from_checkpoint(self._root, checkpoint_record)
            except (OSError, ValueError, tifffile.TiffFileError) as exc:
                raise RuntimeError(f"frame {logical_frame}: first-source checkpoint artifacts are invalid: {exc}") from exc
            records["first"] = checkpoint_record

        def persist_source(label: str, physical_frame: int, source: ScanResult) -> None:
            split = source.split_source
            if split is None or source.ir is None or source.ir_valid_mask is None:
                raise RuntimeError(f"{label} source is missing immutable split-capture planes")
            bundle = write_split_source_bundle(
                split,
                aligned_ir=source.ir,
                ir_valid_mask=source.ir_valid_mask,
                output_dir=self._root / "sources" / f"frame{logical_frame:03d}",
                dpi=source.dpi,
            )
            raw_artifacts = cast(dict[str, dict[str, object]], bundle["artifacts"])
            bundle_prefix = f"sources/frame{logical_frame:03d}/{cast(str, bundle['bundle_path'])}"
            artifacts = {role: _artifact(raw_artifacts[role], prefix=bundle_prefix) for role in raw_artifacts}
            first = records.get("first")
            replayed_from = first.capture_id if label == "second" and first is not None else None
            rows = int(source.rgb.shape[0])
            geometry = {
                "physical_frame": physical_frame,
                "subframe_mm": 0.0,
                "br_y_device_px": rows - 1,
            }
            alignment = asdict(source.ir_alignment) if source.ir_alignment is not None else {}
            records[label] = SourceCaptureRecord.create(
                identity=self._identity,
                physical_frame=physical_frame,
                geometry=geometry,
                recipe=common_recipe,
                capture_state=_capture_state(
                    source,
                    mode="measured" if label == "first" else "replayed",
                    replayed_from_capture_id=replayed_from,
                ),
                passes=(
                    {
                        "role": "rgb4x",
                        "lines": rows,
                        "depth": source.rgb.dtype.itemsize * 8,
                        "samples_per_scan": recipe.samples_per_scan,
                    },
                    {
                        "role": "rgbi1x",
                        "lines": int(split.rgb1x_proxy.shape[0]),
                        "depth": split.rgb1x_proxy.dtype.itemsize * 8,
                        "samples_per_scan": 1,
                    },
                ),
                artifacts=artifacts,
                split_alignment=alignment,
            )
            if label == "first":
                _atomic_write(checkpoint_path, json.dumps(records[label].to_dict(), indent=2) + "\n")

        capturer = self._capturer
        if checkpoint_source is not None:
            capturer = FullNegativeCapturer(
                scanner=_CheckpointFirstScanner(self._scanner, checkpoint_source),
                pitch_rows=self._pitch_rows,
            )
        capture = capturer.capture(
            device_id=device_id,
            registration=registration,
            recipe=recipe,
            stop=stop,
            progress=progress,
            on_source=persist_source,
        )
        first_record = records.get("first")
        second_record = records.get("second")
        if first_record is None or second_record is None:
            raise RuntimeError("full-negative source persistence did not commit both captures")
        pair = SourcePairRecord.create(
            logical_frame=logical_frame,
            first=first_record,
            second=second_record,
        )

        final_result = ScanResult(
            rgb=capture.result.rgb,
            ir=capture.result.ir,
            dpi=recipe.dpi,
            device_model=capture.first_source.device_model,
            ir_valid_mask=capture.result.ir_valid_mask,
        )
        final_evidence = write_full_negative_tiff(
            final_result,
            ir_valid_mask=capture.result.ir_valid_mask,
            path=self._root / "final" / f"frame{logical_frame:03d}.tif",
        )
        reconstruction_artifacts = {role: _artifact(evidence, prefix="final") for role, evidence in final_evidence.items()}
        registration_binding = canonical_semantic_sha256(asdict(registration))
        reconstruction = ReconstructionBindingRecord.create(
            identity=self._identity,
            logical_frame=logical_frame,
            source_pair=pair,
            registration_binding_sha256=registration_binding,
            algorithm_signature={
                "algorithm": "negpy.adjacent-window-reconstruction",
                "version": 1,
                "rgb_resampling": "none",
                "ir_invalid_fill": 65535,
            },
            pitch_rows=capture.geometry.pitch_rows,
            crop_start_row=capture.geometry.phase_row,
            artifacts=reconstruction_artifacts,
        )

        device = self._scanner.probe_device(device_id)
        if device.id != device_id:
            raise RuntimeError(f"fresh scanner probe returned {device.id!r}, expected {device_id!r}")
        provenance = FullNegativeProvenanceManifest.create(
            identity=self._identity,
            captures=(first_record, second_record),
            source_pair=pair,
            reconstruction=reconstruction,
        )
        _atomic_write(manifest_path, provenance.to_json())
        return FullNegativeWorkflowResult(
            manifest_path=manifest_path,
            provenance=provenance,
            capture=capture,
        )


__all__ = [
    "FullNegativeWorkflow",
    "FullNegativeWorkflowResult",
    "FullNegativeWorkflowScanner",
]
