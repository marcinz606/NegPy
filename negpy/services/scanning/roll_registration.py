"""Preview-derived registration policy for roll-fed film scanners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor, isfinite
from typing import Mapping

import numpy as np

from negpy.infrastructure.scanners.frame_registration import (
    FilmBaseLearningConfig,
    FilmBaseModel,
    TailBoundary,
    TailDetectorConfig,
    TargetEdge,
    learn_film_base,
    measure_leading_margin,
    measure_target_start,
    measure_transport_tails,
    robust_pitch_prediction,
)
from negpy.services.scanning.quality import assess_stopped_transport_smear
from negpy.infrastructure.scanners.params import RegisteredScanGeometry
from negpy.services.scanning.roll_service import (
    AlignmentVerification,
    FrameRegistration,
)


@dataclass(frozen=True)
class RollRegistrationConfig:
    """Geometry calibration shared by the preview and device-pixel grids."""

    preview_dpi: int = 400
    device_dpi: int = 4000
    retained_margin_mm: float = 0.57
    tail_guard_device_px: int = 20
    inclusive_endpoint_adjustment_px: int = 2
    alignment_tolerance_rows: int = 4
    maximum_edge_prediction_error_rows: float = 12.0
    expected_full_preview_rows: int = 595
    # Contiguous LS-5000 feeder-window stride: 5,959 native rows at 4000 dpi.
    # This is a scanner coordinate interval, not nominal 135-film pitch.
    frame_pitch_mm: float = 5959 * 25.4 / 4000
    maximum_subframe_mm: float = 37.82
    maximum_br_y_device_px: int = 5958
    excluded_frames: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        for field, value in (
            ("preview_dpi", self.preview_dpi),
            ("device_dpi", self.device_dpi),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        for field, value in (
            ("tail_guard_device_px", self.tail_guard_device_px),
            ("inclusive_endpoint_adjustment_px", self.inclusive_endpoint_adjustment_px),
            ("alignment_tolerance_rows", self.alignment_tolerance_rows),
            ("expected_full_preview_rows", self.expected_full_preview_rows),
            ("maximum_br_y_device_px", self.maximum_br_y_device_px),
        ):
            minimum = 1 if field == "expected_full_preview_rows" else 0
            if type(value) is not int or value < minimum:
                adjective = "positive" if minimum else "non-negative"
                raise ValueError(f"{field} must be a {adjective} integer")
        for field, value in (
            ("retained_margin_mm", self.retained_margin_mm),
            ("maximum_edge_prediction_error_rows", self.maximum_edge_prediction_error_rows),
            ("frame_pitch_mm", self.frame_pitch_mm),
            ("maximum_subframe_mm", self.maximum_subframe_mm),
        ):
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")
        if type(self.excluded_frames) is not frozenset or any(type(frame) is not int or frame < 1 for frame in self.excluded_frames):
            raise ValueError("excluded_frames must be a frozenset of positive integers")
        if self.maximum_subframe_mm >= self.frame_pitch_mm:
            raise ValueError("maximum_subframe_mm must be smaller than frame_pitch_mm")


def registered_scan_geometry(
    *,
    frame: int,
    target_start_row: int,
    tail_start_row: float,
    config: RollRegistrationConfig,
) -> RegisteredScanGeometry:
    """Convert measured preview bounds into coupled native scanner geometry."""

    if frame < 1 or target_start_row < 0:
        raise ValueError("frame must be positive and target row non-negative")
    if not isfinite(tail_start_row) or tail_start_row <= target_start_row:
        raise ValueError("tail boundary must be finite and follow the target edge")
    if config.preview_dpi <= 0 or config.device_dpi <= 0:
        raise ValueError("preview and device DPI must be positive")

    target_start_mm = target_start_row * 25.4 / config.preview_dpi
    requested_subframe_mm = target_start_mm - config.retained_margin_mm
    physical_frame = frame
    if requested_subframe_mm < 0 and frame > 1:
        # The logical frame origin clipped its leading gap.  Address the same
        # location from the previous feeder frame at the latest safe subframe;
        # this avoids including the preceding exposure while recovering the
        # maximum physically attainable gap prefix.
        physical_frame = frame - 1
        subframe_mm = round(config.maximum_subframe_mm, 2)
        effective_subframe_mm = subframe_mm - config.frame_pitch_mm
    else:
        subframe_mm = round(max(0.0, requested_subframe_mm), 2)
        effective_subframe_mm = subframe_mm
    subframe_device_px = floor(effective_subframe_mm * config.device_dpi / 25.4)
    tail_device_px = floor(tail_start_row * config.device_dpi / config.preview_dpi + 0.5)
    br_y_device_px = min(
        config.maximum_br_y_device_px,
        tail_device_px - subframe_device_px - config.inclusive_endpoint_adjustment_px - config.tail_guard_device_px,
    )
    if br_y_device_px < 0:
        raise ValueError("registered scan window is empty")
    return RegisteredScanGeometry(
        frame=physical_frame,
        subframe_mm=subframe_mm,
        br_y_device_px=br_y_device_px,
    )


def _qc_transport_boundaries(
    previews: Mapping[int, np.ndarray],
    *,
    config: RollRegistrationConfig,
) -> dict[int, TailBoundary]:
    """Conservatively bound complete previews when no tail line can be fit.

    A missing transport-tail observation is not evidence of completeness.
    Exact-height uint16 acquisitions retain their full endpoint unless either
    the whole image or outer bands reports an actual smear.  Indeterminate
    suffixes lower confidence because smooth scene content can produce them,
    but they do not justify shortening an otherwise complete acquisition.
    """

    boundaries: dict[int, TailBoundary] = {}
    for frame, raw_image in previews.items():
        image = np.asarray(raw_image)
        if image.shape[0] != config.expected_full_preview_rows:
            raise ValueError(
                f"position {frame}: expected {config.expected_full_preview_rows} rows in a complete preview; got {image.shape[0]}"
            )
        if image.dtype != np.uint16:
            raise ValueError(f"position {frame}: complete-preview QC requires uint16 data")
        quarter = image.shape[1] // 4
        if quarter < 1:
            raise ValueError(f"position {frame}: preview is too narrow for outer-band QC")
        outer = np.concatenate((image[:, :quarter], image[:, -quarter:]), axis=1)
        assessments = (
            assess_stopped_transport_smear(image, dpi=config.preview_dpi),
            assess_stopped_transport_smear(outer, dpi=config.preview_dpi),
        )
        agreement_rows = TailDetectorConfig().outer_agreement_rows
        smear_starts = [
            assessment.start_row for assessment in assessments if assessment.verdict == "smear" and assessment.start_row is not None
        ]
        if any(assessment.verdict == "smear" for assessment in assessments):
            if not smear_starts:
                raise ValueError(f"position {frame}: smear QC reported a tail without a boundary")
            if len(smear_starts) == 2 and abs(smear_starts[0] - smear_starts[1]) > agreement_rows:
                raise ValueError(f"position {frame}: whole-image and outer-band smear boundaries disagree")
            boundaries[frame] = TailBoundary(
                tail_start_row=float(min(smear_starts)),
                source="qc",
                confidence="high" if len(smear_starts) == 2 else "medium",
            )
            continue
        if any(assessment.verdict == "indeterminate" for assessment in assessments):
            boundaries[frame] = TailBoundary(
                tail_start_row=float(image.shape[0]),
                source="complete",
                confidence="medium",
            )
            continue
        boundaries[frame] = TailBoundary(
            tail_start_row=float(image.shape[0]),
            source="complete",
            confidence="high",
        )
    return boundaries


class PreviewRollRegistration:
    """Jointly calibrate roll previews without assuming leader length or RGB."""

    def __init__(self, config: RollRegistrationConfig) -> None:
        self.config = config
        self._film_base: FilmBaseModel | None = None

    @property
    def preview_dpi(self) -> int:
        return self.config.preview_dpi

    @property
    def minimum_preview_count(self) -> int:
        return max(3, FilmBaseLearningConfig().minimum_preview_support)

    @property
    def registration_signature(self) -> dict[str, object]:
        config = asdict(self.config)
        config["excluded_frames"] = sorted(self.config.excluded_frames)
        return {
            "algorithm": "negpy.preview-roll-registration",
            "version": 3,
            "config": config,
            "film_base_learning": asdict(FilmBaseLearningConfig()),
            "tail_detection": asdict(TailDetectorConfig()),
            "tail_residual_tolerance_rows": 1.5,
            "target_edge_minimum_run_rows": 3,
            "leading_margin_minimum_run_rows": 3,
            "leading_margin_max_initial_outlier_rows": 1,
        }

    def calibrate(self, previews: Mapping[int, np.ndarray]) -> dict[int, FrameRegistration]:
        if not previews:
            return {}
        ordered = {int(frame): np.asarray(image) for frame, image in previews.items()}
        if len(ordered) != len(previews):
            raise ValueError("preview frame positions must be unique integers")
        if len(ordered) < self.minimum_preview_count:
            raise ValueError(f"registration requires at least {self.minimum_preview_count} preview frames; got {len(ordered)}")

        model = learn_film_base(list(ordered.values()))
        self._film_base = model
        edges: dict[int, TargetEdge] = {}
        unresolved: set[int] = set()
        for frame, image in ordered.items():
            try:
                edges[frame] = measure_target_start(image, model)
            except ValueError as exc:
                if str(exc) != "preview contains no film-base run followed by image content":
                    raise
                unresolved.add(frame)

        included_count = len(set(edges) - set(self.config.excluded_frames))
        if included_count >= 3:
            observed = {frame: edge.row for frame, edge in edges.items()}
            refined = {}
            for frame, image in ordered.items():
                if frame in self.config.excluded_frames:
                    if frame not in edges:
                        raise ValueError(f"excluded frame {frame} has no directly measured film-base edge")
                    refined[frame] = edges[frame]
                    continue
                prediction = robust_pitch_prediction(
                    observed,
                    frame=frame,
                    excluded_frames=self.config.excluded_frames,
                )
                try:
                    refined[frame] = measure_target_start(
                        image,
                        model,
                        expected_row=prediction,
                        maximum_prediction_error=(self.config.maximum_edge_prediction_error_rows),
                        allow_short_prefix=frame in unresolved,
                    )
                except ValueError as exc:
                    direct = edges.get(frame)
                    if (
                        str(exc) != "no film-base edge agrees with the supplied prediction"
                        or direct is None
                        or abs(direct.row - prediction) <= self.config.maximum_edge_prediction_error_rows
                    ):
                        raise
                    # A long, smooth scene can look locally like film base.
                    # When the direct candidate is a clear roll-wide outlier,
                    # preserve the robust prediction as medium confidence and
                    # require the normal visual-approval gate downstream.
                    refined[frame] = TargetEdge(
                        row=max(0, round(prediction)),
                        confidence="medium",
                    )
            edges = refined
        elif unresolved:
            raise ValueError("not enough direct film-base edges to recover a clipped transport boundary")

        try:
            tails = measure_transport_tails(ordered)
        except ValueError as exc:
            if str(exc) != "at least three direct tail observations are required":
                raise
            tails = _qc_transport_boundaries(ordered, config=self.config)
        registrations: dict[int, FrameRegistration] = {}
        for frame in ordered:
            edge = edges[frame]
            tail = tails[frame]
            geometry = registered_scan_geometry(
                frame=frame,
                target_start_row=edge.row,
                tail_start_row=tail.tail_start_row,
                config=self.config,
            )
            confidence = (
                "high"
                if edge.confidence == "high" and tail.confidence == "high" and tail.source in {"direct", "complete", "qc"}
                else "medium"
            )
            registrations[frame] = FrameRegistration(
                frame=frame,
                target_start_row=edge.row,
                usable_tail_row=float(tail.tail_start_row),
                confidence=confidence,
                geometry=geometry,
            )
        return registrations

    def verify(
        self,
        frame: int,
        preview: np.ndarray,
        registration: FrameRegistration,
    ) -> AlignmentVerification:
        if frame != registration.frame:
            raise ValueError(f"verification frame mismatch: expected {registration.frame}, got {frame}")
        if self._film_base is None:
            raise RuntimeError("calibrate must run before aligned-preview verification")
        geometry_frame = registration.geometry.frame
        if geometry_frame == frame - 1:
            effective_subframe_mm = registration.geometry.subframe_mm - self.config.frame_pitch_mm
            target_start_mm = registration.target_start_row * 25.4 / self.config.preview_dpi
            target_rows = round((target_start_mm - effective_subframe_mm) * self.config.preview_dpi / 25.4)
        else:
            target_rows = round(self.config.retained_margin_mm * self.config.preview_dpi / 25.4)
        try:
            margin = measure_leading_margin(preview, self._film_base)
        except ValueError:
            return AlignmentVerification(
                leading_margin_rows=None,
                target_margin_rows=target_rows,
                tolerance_rows=self.config.alignment_tolerance_rows,
                confidence="unresolved",
            )
        return AlignmentVerification(
            leading_margin_rows=margin.row,
            target_margin_rows=target_rows,
            tolerance_rows=self.config.alignment_tolerance_rows,
            confidence=margin.confidence,
        )
