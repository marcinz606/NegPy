"""Roll-aware crop calibration built on the single-frame film detector.

The module is deliberately free of Qt, storage, and file-loading concerns.  It
accepts transformed preview buffers, collects deterministic crop evidence, builds
a robust roll template, and resolves only frames that retain film-edge evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from negpy.domain.types import ROI, ImageBuffer
from negpy.features.geometry.logic import (
    BORDER_SIDES,
    _closest_standard_ratio,
    _detection_luma,
    _trim_opaque_border,
    apply_fine_rotation,
    detect_film_bounds_with_confidence,
    enforce_roi_aspect_ratio,
    measure_film_border,
)
from negpy.features.geometry.models import FINE_ROTATION_LIMIT


_TRUSTED_CONFIDENCE = 0.58
_MAX_AUTOMATIC_DESKEW = 8.0
_DEFAULT_SAFETY_BORDER = 0.01
_MIN_PROFILE_CONTRAST = 0.06
# The rebate width is a roll property, so require enough agreeing frames that a few
# bright frame edges cannot move the median.
_MIN_BORDER_SAMPLES = 5
# A fit disagreeing with the consensus by more than this is reading something other than
# the film edge, so it is dropped.
_MAX_EDGE_FIT_DELTA = 0.5
_MIN_FITTED_ANGLES = 3


@dataclass(frozen=True)
class CropEvidence:
    """Single-frame evidence in the final, deskewed display coordinate space."""

    key: str
    canvas_shape: tuple[int, int]
    roi: ROI | None
    correction_angle: float
    confidence: float
    target_ratio: str = "3:2"
    rebate_trim: float = 1.0
    # Set when a line fit confirmed correction_angle. Independent of `confidence`, which
    # scores the box rather than its tilt.
    angle_confident: bool = False
    supported_sides: frozenset[str] = frozenset()
    supported_corners: frozenset[str] = frozenset()
    evidence_sources: tuple[str, ...] = ()
    geometry_score: float = 0.0
    vertical_edge_contrast: float = 0.0
    vertical_edge_profile: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        compare=False,
        repr=False,
    )
    # Per-side thickness in BORDER_SIDES order, as a fraction of the film box side. NaN
    # marks a side with no clean transition, () a frame that was never measured.
    border: tuple[float, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class RollCropTemplate:
    """Robust normalized geometry shared by trustworthy frames in a roll."""

    width: float
    fallback_width: float
    height: float
    center_x: float
    top: float
    correction_angle: float
    width_mad: float
    center_mad: float
    top_mad: float
    angle_mad: float
    confidence: float
    sample_count: int
    # Roll median per side (BORDER_SIDES order); () when too few frames measured one.
    border: tuple[float, ...] = ()
    border_sample_count: int = 0


@dataclass(frozen=True)
class ResolvedCrop:
    """Explicit manual-crop payload ready for controller-side conflict checks."""

    key: str
    manual_crop_rect: tuple[float, float, float, float]
    correction_angle: float
    confidence: float
    calibrated: bool


def _vertical_edge_profile(img: ImageBuffer) -> tuple[np.ndarray, float]:
    """Return a contrast-normalized vertical-edge profile for template fallback."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        gray = arr[..., :3].mean(axis=2)
    else:
        gray = arr
    finite = gray[np.isfinite(gray)]
    if finite.size == 0 or gray.shape[1] < 2:
        return np.zeros(gray.shape[1], dtype=np.float32), 0.0

    low, high = np.percentile(finite, (2.0, 98.0))
    span = max(float(high - low), 1e-6)
    normalized = np.clip((gray - low) / span, 0.0, 1.0).astype(np.float32)
    grad = np.abs(cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3))
    profile = np.percentile(grad, 80, axis=0).astype(np.float32)
    window = max(5, int(round(profile.size * 0.012)))
    if window % 2 == 0:
        window += 1
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (window, 1), 0).ravel()
    peak = float(np.max(profile)) if profile.size else 0.0
    contrast = max(0.0, float(np.percentile(profile, 99.5)) - float(np.percentile(profile, 50.0))) if profile.size else 0.0
    if peak > 0.0:
        profile /= peak
    return profile.astype(np.float32), contrast


def _top_edge_slope(lum: np.ndarray, roi: ROI) -> float | None:
    """Residual tilt of the film box's top edge, in degrees, or None when unreadable.

    Sign matches _correction_angle_from_quad: the measured image-coordinate residual is
    already the additive fix.
    """
    y1, _, x1, x2 = roi
    height = lum.shape[0]
    left, right = x1 + 20, x2 - 20
    columns = np.arange(left, right, 3)
    min_points = max(20, round((right - left) * 0.08))
    if columns.size < min_points:
        return None

    radius = max(12, round(height * 0.035))
    low, high = max(0, y1 - radius), min(height, y1 + radius)
    if high - low < 4:
        return None

    gradient = np.abs(cv2.Sobel(cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 1.5), cv2.CV_32F, 0, 1, ksize=3))
    band = gradient[low:high, columns]
    rows = low + np.argmax(band, axis=0)
    strength = band.max(axis=0)
    # With no edge under the band every column peaks on noise and the fit still comes
    # back clean: a straight line through nothing, reported as a confident zero.
    if float(np.median(strength)) < max(1e-6, 3.0 * float(np.median(band))):
        return None
    points = np.column_stack([columns, rows])[strength >= np.quantile(strength, 0.60)]
    if len(points) < min_points:
        return None

    residuals = np.zeros(len(points))
    slope = 0.0
    for _ in range(5):
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        residuals = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        keep = residuals <= max(2.0, float(np.median(residuals)) * 2.5)
        if keep.all() or keep.sum() < min_points:
            break
        points = points[keep]

    if float(np.median(residuals)) > 2.0:
        return None
    return float(np.degrees(np.arctan(slope)))


def detect_crop_candidate(
    key: str,
    image: ImageBuffer,
    *,
    target_ratio: str = "3:2",
    rebate_trim: float = 1.0,
) -> CropEvidence:
    """Collect crop, deskew, and edge evidence for one transformed preview.

    The caller must apply flat-field correction, coarse rotation, flips, existing
    fine rotation, and distortion first.  Portrait canvases are intentionally left
    unchanged because the first implementation targets the landscape roll workflow.
    """
    h, w = image.shape[:2]
    profile, profile_contrast = _vertical_edge_profile(image)
    if w <= h:
        return CropEvidence(
            key,
            (h, w),
            None,
            0.0,
            0.0,
            target_ratio=target_ratio,
            rebate_trim=rebate_trim,
            vertical_edge_contrast=profile_contrast,
            vertical_edge_profile=profile,
            reason="unsupported_orientation",
        )

    initial = detect_film_bounds_with_confidence(image)
    if initial.roi is None:
        return CropEvidence(
            key,
            (h, w),
            None,
            0.0,
            0.0,
            target_ratio=target_ratio,
            rebate_trim=rebate_trim,
            vertical_edge_contrast=initial.vertical_edge_contrast,
            vertical_edge_profile=np.asarray(initial.vertical_edge_profile, dtype=np.float32),
            reason="no_consensus",
        )

    correction = float(initial.correction_angle)
    if not np.isfinite(correction) or abs(correction) > _MAX_AUTOMATIC_DESKEW:
        correction = 0.0

    # Before rotating, not after: the re-detection below then measures the ROI at the
    # final angle, so no rect has to be remapped.
    delta = _top_edge_slope(_detection_luma(image), initial.roi)
    angle_confident = delta is not None and abs(delta) <= _MAX_EDGE_FIT_DELTA
    if angle_confident:
        correction = float(np.clip(correction + delta, -_MAX_AUTOMATIC_DESKEW, _MAX_AUTOMATIC_DESKEW))

    corrected = apply_fine_rotation(image, correction) if abs(correction) > 1e-4 else image
    final = detect_film_bounds_with_confidence(corrected)
    if final.roi is None:
        return CropEvidence(
            key,
            corrected.shape[:2],
            None,
            correction,
            0.0,
            target_ratio=target_ratio,
            rebate_trim=rebate_trim,
            angle_confident=angle_confident,
            vertical_edge_contrast=final.vertical_edge_contrast,
            vertical_edge_profile=np.asarray(final.vertical_edge_profile, dtype=np.float32),
            reason="deskew_no_consensus",
        )

    # The film box, not the exposed image area: only the measurement is taken here, and
    # resolve_roll_crops decides using the whole roll.
    lum = _detection_luma(corrected)
    roi = _trim_opaque_border(lum, final.roi)
    border = tuple(measure_film_border(lum, roi)[name] for name in BORDER_SIDES)
    y1, y2, x1, x2 = roi
    if y2 <= y1 or x2 <= x1:
        return CropEvidence(
            key,
            corrected.shape[:2],
            None,
            correction,
            0.0,
            target_ratio=target_ratio,
            rebate_trim=rebate_trim,
            angle_confident=angle_confident,
            vertical_edge_contrast=final.vertical_edge_contrast,
            vertical_edge_profile=np.asarray(final.vertical_edge_profile, dtype=np.float32),
            reason="invalid_geometry",
        )

    confidence = float(np.clip(final.confidence, 0.0, 1.0))
    return CropEvidence(
        key=key,
        canvas_shape=corrected.shape[:2],
        roi=roi,
        correction_angle=correction,
        confidence=confidence,
        target_ratio=target_ratio,
        rebate_trim=rebate_trim,
        angle_confident=angle_confident,
        supported_sides=final.supported_sides,
        supported_corners=final.supported_corners,
        evidence_sources=final.evidence_sources,
        geometry_score=float(np.clip(final.geometry_score, 0.0, 1.0)),
        vertical_edge_contrast=final.vertical_edge_contrast,
        vertical_edge_profile=np.asarray(final.vertical_edge_profile, dtype=np.float32),
        border=border,
    )


def _normalized_roi(roi: ROI, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    h, w = shape
    y1, y2, x1, x2 = roi
    return x1 / w, y1 / h, x2 / w, y2 / h


def _pixel_roi(rect: tuple[float, float, float, float], shape: tuple[int, int]) -> ROI:
    h, w = shape
    x1, y1, x2, y2 = rect
    return (
        max(0, min(h, int(round(y1 * h)))),
        max(0, min(h, int(round(y2 * h)))),
        max(0, min(w, int(round(x1 * w)))),
        max(0, min(w, int(round(x2 * w)))),
    )


def _mad(values: np.ndarray, center: float | None = None) -> float:
    if values.size == 0:
        return 0.0
    pivot = float(np.median(values)) if center is None else center
    return float(np.median(np.abs(values - pivot)))


def _roll_border(evidence: Sequence[CropEvidence]) -> tuple[tuple[float, ...], int]:
    """Median border thickness per side across every frame that measured one.

    Pooled over all detections, not the trusted subset: thickness is a property of the
    film gate, so more samples beat stricter ones.
    """
    measured = [item.border for item in evidence if len(item.border) == len(BORDER_SIDES)]
    if not measured:
        return (), 0
    columns = np.asarray(measured, dtype=np.float64)
    medians: list[float] = []
    counts: list[int] = []
    for index in range(len(BORDER_SIDES)):
        values = columns[:, index]
        values = values[np.isfinite(values)]
        counts.append(values.size)
        medians.append(float(np.median(values)) if values.size else 0.0)
    sample_count = min(counts)
    if sample_count < _MIN_BORDER_SAMPLES:
        return (), sample_count
    return tuple(medians), sample_count


def build_roll_template(evidence: Sequence[CropEvidence]) -> RollCropTemplate | None:
    """Build a median/MAD roll template from multi-side, trustworthy detections."""
    trusted = [
        item
        for item in evidence
        if item.roi is not None
        and item.confidence >= _TRUSTED_CONFIDENCE
        and item.geometry_score >= 0.35
        and (len(item.supported_sides) >= 2 or len(item.supported_corners) >= 1)
    ]
    border, border_samples = _roll_border(evidence)
    if not trusted:
        return None

    rects = np.asarray([_normalized_roi(item.roi, item.canvas_shape) for item in trusted], dtype=np.float64)
    widths = rects[:, 2] - rects[:, 0]
    heights = rects[:, 3] - rects[:, 1]
    centers = (rects[:, 0] + rects[:, 2]) * 0.5
    tops = rects[:, 1]
    angles = np.asarray([item.correction_angle for item in trusted], dtype=np.float64)

    width_med = float(np.median(widths))
    angle_med = float(np.median(angles))
    width_tol = max(0.025, 3.5 * _mad(widths, width_med))
    angle_tol = max(0.45, 3.5 * _mad(angles, angle_med))
    keep = (np.abs(widths - width_med) <= width_tol) & (np.abs(angles - angle_med) <= angle_tol)
    if not np.any(keep):
        keep = np.ones(widths.shape, dtype=bool)

    widths = widths[keep]
    heights = heights[keep]
    centers = centers[keep]
    tops = tops[keep]
    angles = angles[keep]
    kept_items = [item for item, accepted in zip(trusted, keep, strict=True) if accepted]

    fitted = np.asarray([item.correction_angle for item in kept_items if item.angle_confident], dtype=np.float64)
    if fitted.size >= _MIN_FITTED_ANGLES:
        angles = fitted

    return RollCropTemplate(
        width=float(np.median(widths)),
        fallback_width=float(np.percentile(widths, 75)),
        height=float(np.median(heights)),
        center_x=float(np.median(centers)),
        top=float(np.median(tops)),
        correction_angle=float(np.median(angles)),
        width_mad=_mad(widths),
        center_mad=_mad(centers),
        top_mad=_mad(tops),
        angle_mad=_mad(angles),
        confidence=float(np.median([item.confidence for item in kept_items])),
        sample_count=len(kept_items),
        border=border,
        border_sample_count=border_samples,
    )


def _clamp_rect(rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = rect
    x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)), float(np.clip(x2, 0.0, 1.0))))
    y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0))))
    return x1, y1, x2, y2


def _map_rect_between_rotations(
    rect: tuple[float, float, float, float],
    shape: tuple[int, int],
    source_angle: float,
    target_angle: float,
) -> tuple[float, float, float, float]:
    """Map a normalized rectangle into a canvas with a different fine rotation.

    ``apply_fine_rotation`` keeps the canvas size fixed and rotates around its
    center. Mapping all four corners and taking their enclosing box preserves the
    original crop without clipping when a weak angle is replaced by roll consensus.
    """
    delta = float(target_angle - source_angle)
    if abs(delta) <= 1e-7:
        return rect
    h, w = shape
    x1, y1, x2, y2 = rect
    points = np.asarray(
        [
            (x1 * w, y1 * h),
            (x2 * w, y1 * h),
            (x2 * w, y2 * h),
            (x1 * w, y2 * h),
        ],
        dtype=np.float64,
    )
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), delta, 1.0)
    mapped = points @ matrix[:, :2].T + matrix[:, 2]
    # Quantize outward before normalizing. Later conversion back to a half-open
    # pixel ROI must not round an extremum inward, especially when one edge touches
    # the canvas and the uniform safety border is consequently limited to zero.
    left = max(0, min(w, int(np.floor(np.min(mapped[:, 0])))))
    top = max(0, min(h, int(np.floor(np.min(mapped[:, 1])))))
    right = max(0, min(w, int(np.ceil(np.max(mapped[:, 0])))))
    bottom = max(0, min(h, int(np.ceil(np.max(mapped[:, 1])))))
    return _clamp_rect(
        (
            left / w,
            top / h,
            right / w,
            bottom / h,
        )
    )


def _calibrate_detected_rect(
    item: CropEvidence,
    template: RollCropTemplate,
) -> tuple[tuple[float, float, float, float], bool]:
    assert item.roi is not None
    x1, y1, x2, y2 = _normalized_roi(item.roi, item.canvas_shape)
    width, height = x2 - x1, y2 - y1
    center = (x1 + x2) * 0.5
    calibrated = False

    width_tol = max(0.035, 4.0 * template.width_mad)
    center_tol = max(0.035, 4.0 * template.center_mad)
    top_tol = max(0.03, 4.0 * template.top_mad)

    target_width = width
    if 0.0 < template.width - width <= max(0.10, width_tol) or (
        item.confidence < _TRUSTED_CONFIDENCE and abs(width - template.width) > width_tol
    ):
        target_width = template.width
        calibrated = True

    if target_width != width:
        if "left" in item.supported_sides and "right" not in item.supported_sides:
            x2 = x1 + target_width
        elif "right" in item.supported_sides and "left" not in item.supported_sides:
            x1 = x2 - target_width
        else:
            x1, x2 = center - target_width * 0.5, center + target_width * 0.5

    if item.confidence < _TRUSTED_CONFIDENCE:
        if abs(center - template.center_x) > center_tol:
            x1, x2 = template.center_x - target_width * 0.5, template.center_x + target_width * 0.5
            calibrated = True
        if abs(y1 - template.top) > top_tol or abs(height - template.height) > max(0.04, 4.0 * template.top_mad):
            y1, y2 = template.top, template.top + template.height
            calibrated = True

    return _clamp_rect((x1, y1, x2, y2)), calibrated


def _peak_near(profile: np.ndarray, expected: float, radius: float) -> tuple[float, float]:
    if profile.size == 0:
        return expected, 0.0
    center = int(round(expected * (profile.size - 1)))
    half = max(2, int(round(radius * profile.size)))
    lo, hi = max(0, center - half), min(profile.size, center + half + 1)
    if hi <= lo:
        return expected, 0.0
    local = profile[lo:hi]
    offset = int(np.argmax(local))
    idx = lo + offset
    return idx / max(profile.size - 1, 1), float(local[offset])


def _rect_from_edge_profile(item: CropEvidence, template: RollCropTemplate) -> tuple[float, float, float, float] | None:
    profile = np.asarray(item.vertical_edge_profile, dtype=np.float32)
    if item.vertical_edge_contrast < _MIN_PROFILE_CONTRAST or profile.size < 8 or not np.any(np.isfinite(profile)):
        return None

    baseline_level = float(np.percentile(profile, 50))
    peak_level = float(np.percentile(profile, 99.5))
    if peak_level - baseline_level < 0.20 or peak_level < 1.35 * max(baseline_level, 0.05):
        return None

    left_expected = template.center_x - template.width * 0.5
    right_expected = template.center_x + template.width * 0.5
    radius = max(0.045, 4.0 * template.center_mad + 2.0 * template.width_mad)
    left, left_strength = _peak_near(profile, left_expected, radius)
    right, right_strength = _peak_near(profile, right_expected, radius)

    baseline = float(np.percentile(profile, 70))
    spread = float(np.percentile(profile, 90) - np.percentile(profile, 50))
    threshold = max(0.24, baseline + 0.55 * spread)
    left_ok, right_ok = left_strength >= threshold, right_strength >= threshold
    if not left_ok and not right_ok:
        return None
    if left_ok and right_ok:
        if right - left < 0.65 * template.width or right - left > 1.35 * template.fallback_width:
            return None
        x1, x2 = left, right
    elif left_ok and left_strength >= threshold + 0.12:
        x1, x2 = left, left + template.fallback_width
    elif right_ok and right_strength >= threshold + 0.12:
        x1, x2 = right - template.fallback_width, right
    else:
        return None

    return _clamp_rect((x1, template.top, x2, template.top + template.height))


def _resolve_border(frame: tuple[float, ...], roll: tuple[float, ...]) -> tuple[float, ...]:
    """Prefer the frame's own border, falling back to the roll median per side.

    How much bed the detector leaves inside the film box varies frame to frame, so
    pooling that measurement across the roll trims the wrong amount. The median is only
    the safety net for sides that abstained (NaN).
    """
    if len(roll) != len(BORDER_SIDES):
        return ()
    if len(frame) != len(BORDER_SIDES):
        return roll
    return tuple(own if np.isfinite(own) else fallback for own, fallback in zip(frame, roll, strict=True))


def _inset_rect_by_border(
    rect: tuple[float, float, float, float],
    border: tuple[float, ...],
) -> tuple[float, float, float, float]:
    """Trim the rebate off a film-box rect. Sides are insets of that rect."""
    if len(border) != len(BORDER_SIDES):
        return rect
    x1, y1, x2, y2 = rect
    width, height = x2 - x1, y2 - y1
    if width <= 0.0 or height <= 0.0:
        return rect
    amounts = dict(zip(BORDER_SIDES, border, strict=True))
    y1 += amounts["top"] * height
    y2 -= amounts["bottom"] * height
    x1 += amounts["left"] * width
    x2 -= amounts["right"] * width
    if y2 - y1 <= 0.0 or x2 - x1 <= 0.0:
        return rect
    return _clamp_rect((x1, y1, x2, y2))


def _apply_target_ratio(roi: ROI, shape: tuple[int, int], target_ratio: str) -> ROI:
    """Re-impose the target ratio, which the border inset breaks by design."""
    h, w = shape
    ratio = target_ratio
    if ratio == "Free":
        ratio = _closest_standard_ratio(roi, (h, w), fallback="3:2").value
    return enforce_roi_aspect_ratio(roi, h, w, ratio)


def add_uniform_safety_border(
    roi: ROI,
    shape: tuple[int, int],
    ratio: float = _DEFAULT_SAFETY_BORDER,
) -> ROI:
    """Expand by one common pixel amount, reduced when any edge lacks room."""
    h, w = shape
    y1, y2, x1, x2 = roi
    requested = max(0, int(round(min(h, w) * ratio)))
    available = max(0, min(y1, x1, h - y2, w - x2))
    pad = min(requested, available)
    return y1 - pad, y2 + pad, x1 - pad, x2 + pad


def resolve_roll_crops(
    evidence: Sequence[CropEvidence],
    *,
    safety_border: float = _DEFAULT_SAFETY_BORDER,
) -> list[ResolvedCrop]:
    """Resolve trustworthy and template-supported frames; ambiguous frames abstain."""
    templates = {
        ratio: build_roll_template([item for item in evidence if item.target_ratio == ratio])
        for ratio in dict.fromkeys(item.target_ratio for item in evidence)
    }
    if not any(template is not None for template in templates.values()):
        return []

    results: list[ResolvedCrop] = []
    for item in evidence:
        if item.reason == "unsupported_orientation":
            continue
        template = templates.get(item.target_ratio)
        if template is None:
            continue

        calibrated = False
        if item.roi is not None:
            rect, calibrated = _calibrate_detected_rect(item, template)
            confidence = item.confidence
        else:
            rect = _rect_from_edge_profile(item, template)
            if rect is None:
                continue
            calibrated = True
            confidence = min(0.55, template.confidence * 0.72)

        angle = item.correction_angle
        angle_tol = max(0.55, 4.0 * template.angle_mad)
        # A fitted angle stands on its own even when the box scored poorly, but still
        # yields to the roll beyond angle_tol, where a mis-detection is likelier.
        angle_trusted = item.angle_confident or item.confidence >= _TRUSTED_CONFIDENCE
        if item.roi is None or not angle_trusted or abs(angle - template.correction_angle) > angle_tol:
            target_angle = template.correction_angle
            rect = _map_rect_between_rotations(rect, item.canvas_shape, angle, target_angle)
            angle = target_angle
            calibrated = True
        angle = float(np.clip(angle, -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT))

        border = _resolve_border(item.border, template.border)
        if item.rebate_trim != 1.0:
            border = tuple(value * item.rebate_trim for value in border)
        inset = _inset_rect_by_border(rect, border)
        trimmed = inset != rect
        roi = _pixel_roi(inset, item.canvas_shape)
        if trimmed:
            # Only the inset breaks the ratio, so only the inset re-imposes it. An
            # untrimmed rect passes through exactly, keeping half-open pixel bounds.
            roi = _apply_target_ratio(roi, item.canvas_shape, item.target_ratio)
        # A measured inset already lands on the rebate edge, so padding it back out by an
        # unrelated fraction would only restore the rebate.
        pad_ratio = 0.0 if trimmed else safety_border
        y1, y2, x1, x2 = add_uniform_safety_border(roi, item.canvas_shape, pad_ratio)
        h, w = item.canvas_shape
        if y2 <= y1 or x2 <= x1:
            continue
        manual_rect = (x1 / w, y1 / h, x2 / w, y2 / h)
        results.append(
            ResolvedCrop(
                key=item.key,
                manual_crop_rect=tuple(float(np.clip(v, 0.0, 1.0)) for v in manual_rect),
                correction_angle=angle,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                calibrated=calibrated,
            )
        )
    return results
