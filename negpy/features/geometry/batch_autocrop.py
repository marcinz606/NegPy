"""Rotation-aware film-frame detection and batch calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.domain.types import ImageBuffer, ROI
from negpy.features.geometry.models import FINE_ROTATION_LIMIT, GeometryConfig

ANALYZE_MAX = 1200
ANGLE_ANALYZE_MAX = 500


@dataclass(frozen=True)
class BatchDetectParams:
    max_frac: float = 0.35
    dark_thresh: float = 0.12
    bright_thresh: float = 0.985
    start_frac: float = 0.60
    strong_frac: float = 0.82
    gap: int = 8
    inset: int = 4
    max_angle: float = 5.0
    min_mask_sides: int = 2
    frame_ratio: float = 1.5
    outward_frac: float = 0.01


@dataclass(frozen=True)
class AutocropCandidate:
    source_id: str
    roi: ROI
    canvas: tuple[int, int]
    angle: float
    plausible_geometry: bool
    top_found: bool
    left_found: bool
    right_found: bool
    left_primary: bool = False
    right_primary: bool = False
    angle_confident: bool = True
    vertical_profile: np.ndarray | None = None


@dataclass(frozen=True)
class AutocropResult:
    roi: ROI
    angle: float
    method: str


@dataclass(frozen=True)
class BatchTemplate:
    width: int
    fallback_width: int
    width_tolerance: int
    center_x: int
    top: int
    angle: float
    angle_tolerance: float


def autocrop_input_signature(config: WorkspaceConfig) -> tuple:
    """Settings that must remain stable between detection and persistence."""
    geometry = config.geometry
    return (
        geometry.rotation,
        geometry.fine_rotation,
        geometry.flip_horizontal,
        geometry.flip_vertical,
        config.process.linear_raw,
        config.rgbscan,
        config.flatfield,
    )


def geometry_from_autocrop_result(
    geometry: GeometryConfig,
    result: AutocropResult,
    canvas: tuple[int, int],
) -> GeometryConfig | None:
    """Convert a calibrated result into persistent NegPy geometry settings."""
    if result.method == "manual":
        return None
    fine_rotation = geometry.fine_rotation + result.angle
    if abs(fine_rotation) > FINE_ROTATION_LIMIT:
        return None
    height, width = canvas
    top, bottom, left, right = result.roi
    manual_rect = (left / width, top / height, right / width, bottom / height)
    return replace(
        geometry,
        fine_rotation=fine_rotation,
        manual_crop_rect=manual_rect,
        auto_crop_enabled=False,
    )


def _validate_params(params: BatchDetectParams) -> None:
    if not 0 < params.max_frac < 0.5:
        raise ValueError("max_frac must be between 0 and 0.5")
    if params.inset < 0:
        raise ValueError("inset must be non-negative")
    if not 0 <= params.max_angle <= 15:
        raise ValueError("max_angle must be between 0 and 15")
    if params.frame_ratio <= 0:
        raise ValueError("frame_ratio must be positive")
    if not 0 <= params.outward_frac <= 0.1:
        raise ValueError("outward_frac must be between 0 and 0.1")


def _analysis_luminance(image: ImageBuffer) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    height, width = image.shape[:2]
    scale = min(1.0, ANALYZE_MAX / max(width, height))
    analysis = image
    if scale < 1:
        analysis = cv2.resize(
            np.ascontiguousarray(image),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    if analysis.ndim == 2:
        gray = analysis
    elif analysis.shape[2] == 4:
        gray = cv2.cvtColor(analysis, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)
    white = float(np.percentile(gray, 99))
    if white <= 0:
        normalized = np.zeros(gray.shape, dtype=np.float32)
    else:
        normalized = np.clip(gray.astype(np.float32) * (255.0 / white), 0, 255)
    return normalized, (height, width), (normalized.shape[0] / height, normalized.shape[1] / width)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.01:
        return image.copy()
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _robust_profile(gradient: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        gradient = gradient[:, int(gradient.shape[1] * 0.08) : int(gradient.shape[1] * 0.92)]
    else:
        gradient = gradient[int(gradient.shape[0] * 0.08) : int(gradient.shape[0] * 0.92)]
    cap = np.quantile(gradient, 0.85)
    return np.quantile(gradient, 0.55, axis=axis) + 0.15 * np.mean(np.minimum(gradient, cap), axis=axis)


def _edge_profiles(gray: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotated = _rotate(gray, angle)
    blurred = cv2.GaussianBlur(rotated, (0, 0), 2.0)
    gradient_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    return (
        rotated,
        _robust_profile(gradient_y, 1),
        _robust_profile(gradient_x, 0),
        gradient_x,
        gradient_y,
    )


def _angle_score(gray: np.ndarray, angle: float, max_frac: float) -> float:
    _, horizontal, vertical, _, _ = _edge_profiles(gray, angle)
    height, width = gray.shape
    bands = (
        horizontal[3 : int(height * max_frac)],
        horizontal[int(height * (1 - max_frac)) : -3],
        vertical[3 : int(width * max_frac)],
        vertical[int(width * (1 - max_frac)) : -3],
    )
    scores = [float(np.max(band)) for band in bands if band.size]
    return sum(scores) + min(scores)


def _estimate_angle(gray: np.ndarray, params: BatchDetectParams) -> float:
    scale = min(1.0, ANGLE_ANALYZE_MAX / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    coarse = np.arange(-params.max_angle, params.max_angle + 0.01, 0.5)
    coarse_scores = np.asarray([_angle_score(small, angle, params.max_frac) for angle in coarse])
    center = float(np.mean(coarse[coarse_scores >= coarse_scores.max() * 0.999]))
    fine = np.arange(
        max(-params.max_angle, center - 0.5),
        min(params.max_angle, center + 0.5) + 0.01,
        0.1,
    )
    fine_scores = np.asarray([_angle_score(small, angle, params.max_frac) for angle in fine])
    return round(float(np.mean(fine[fine_scores >= fine_scores.max() * 0.999])), 2)


def _edge_line_fit(gray: np.ndarray, angle: float, left: int, right: int, edge: int) -> tuple[float, float, int] | None:
    rotated = _rotate(gray, angle)
    blurred = cv2.GaussianBlur(rotated, (0, 0), 1.5)
    gradient = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    radius = max(12, round(gray.shape[0] * 0.035))
    low = max(2, edge - radius)
    high = min(gray.shape[0] - 2, edge + radius)
    xs = np.arange(left + 20, right - 20, 3)
    if len(xs) < 20 or high <= low:
        return None
    ys = np.asarray([low + np.argmax(gradient[low:high, x]) for x in xs])
    scores = gradient[ys, xs]
    points = np.column_stack([xs, ys])[scores >= np.quantile(scores, 0.60)]
    if len(points) < 20:
        return None
    residuals = np.zeros(len(points))
    slope = 0.0
    for _ in range(5):
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        residuals = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        keep = residuals <= max(2.0, float(np.median(residuals)) * 2.5)
        if keep.all() or keep.sum() < 20:
            break
        points = points[keep]
    median_residual = float(np.median(residuals))
    if median_residual > 2.0:
        return None
    return float(np.degrees(np.arctan(slope))), median_residual, len(points)


def _refine_angle(
    gray: np.ndarray,
    angle: float,
    edges: tuple[int, int, int, int],
    params: BatchDetectParams,
) -> tuple[float, bool]:
    left, top, right, _ = edges
    if right - left < gray.shape[1] * 0.4:
        return angle, False
    top_fit = _edge_line_fit(gray, angle, left, right, top)
    min_points = max(20, round((right - left) * 0.08))
    if top_fit is None or abs(top_fit[0]) > 0.5 or top_fit[2] < min_points:
        return angle, False
    return round(min(max(angle + top_fit[0], -params.max_angle), params.max_angle), 2), True


def _mask_depth(fractions: np.ndarray, limit: int, params: BatchDetectParams) -> int | None:
    if limit <= 0 or fractions[: min(8, limit)].max() < params.start_frac:
        return None
    depth = 0
    for index in range(limit):
        if fractions[index] >= params.strong_frac:
            depth = index + 1
        elif index - depth > params.gap:
            break
    if depth == 0 or depth >= limit - 2:
        return None
    return depth


def _strongest_edge(profile: np.ndarray, start: int, end: int) -> int:
    return int(np.argmax(profile[start:end])) + start


def _fallback_is_border(gray: np.ndarray, gradient: np.ndarray, edge: int, side: str) -> bool:
    height, width = gray.shape
    margin = 3
    if side == "top":
        outside = gray[margin : max(margin + 1, edge - 2)]
        texture = gradient[margin : max(margin + 1, edge - 2)]
        span = edge
    elif side == "bottom":
        outside = gray[min(height - 1, edge + 2) : height - margin]
        texture = gradient[min(height - 1, edge + 2) : height - margin]
        span = height - edge
    elif side == "left":
        outside = gray[:, margin : max(margin + 1, edge - 2)]
        texture = gradient[:, margin : max(margin + 1, edge - 2)]
        span = edge
    else:
        outside = gray[:, min(width - 1, edge + 2) : width - margin]
        texture = gradient[:, min(width - 1, edge + 2) : width - margin]
        span = width - edge
    if outside.size == 0 or span < min(height, width) * 0.008:
        return False
    median = float(np.median(outside))
    return (median < 56 or median > 245) and float(np.median(texture)) < 12


def _is_coherent_outer_edge(profile: np.ndarray, edge: int, side: str) -> bool:
    axis_length = len(profile)
    span = edge if side in {"top", "left"} else axis_length - edge
    if span > axis_length * 0.03:
        return False
    outer_limit = max(4, round(axis_length * 0.35))
    band = profile[3:outer_limit] if side in {"top", "left"} else profile[axis_length - outer_limit : axis_length - 3]
    return bool(band.size and float(profile[edge]) >= max(20.0, float(np.quantile(band, 0.95)) * 2.5))


def _analysis_edges(
    gray: np.ndarray,
    angle: float,
    params: BatchDetectParams,
) -> tuple[tuple[int, int, int, int], frozenset[str]] | None:
    rotated, horizontal, vertical, gradient_x, gradient_y = _edge_profiles(gray, angle)
    height, width = rotated.shape
    border = (rotated < params.dark_thresh * 255) | (rotated > params.bright_thresh * 255)
    limit_y = int(height * params.max_frac)
    limit_x = int(width * params.max_frac)
    mask_depths = {
        "top": _mask_depth(border.mean(axis=1), limit_y, params),
        "bottom": _mask_depth(border[::-1].mean(axis=1), limit_y, params),
        "left": _mask_depth(border.mean(axis=0), limit_x, params),
        "right": _mask_depth(border[:, ::-1].mean(axis=0), limit_x, params),
    }
    if sum(depth is not None for depth in mask_depths.values()) < params.min_mask_sides:
        return None
    fallback_edges = {
        "top": _strongest_edge(horizontal, 3, limit_y),
        "bottom": _strongest_edge(horizontal, height - limit_y, height - 3),
        "left": _strongest_edge(vertical, 3, limit_x),
        "right": _strongest_edge(vertical, width - limit_x, width - 3),
    }
    edges: dict[str, int] = {}
    for side in ("top", "bottom", "left", "right"):
        depth = mask_depths[side]
        axis_length = height if side in {"top", "bottom"} else width
        if depth is not None:
            mask_edge = depth if side in {"top", "left"} else axis_length - depth
            gradient_edge = fallback_edges[side]
            inward_delta = gradient_edge - mask_edge if side in {"top", "left"} else mask_edge - gradient_edge
            edges[side] = gradient_edge if 0 < inward_delta <= axis_length * 0.03 else mask_edge
            continue
        candidate = fallback_edges[side]
        gradient = gradient_y if side in {"top", "bottom"} else gradient_x
        profile = horizontal if side in {"top", "bottom"} else vertical
        if _fallback_is_border(rotated, gradient, candidate, side) or _is_coherent_outer_edge(profile, candidate, side):
            edges[side] = candidate
        else:
            edges[side] = 0 if side in {"top", "left"} else axis_length
    inset = params.inset
    return (
        (
            min(edges["left"] + (inset if edges["left"] else 0), width),
            min(edges["top"] + (inset if edges["top"] else 0), height),
            max(edges["right"] - (inset if edges["right"] < width else 0), 0),
            max(edges["bottom"] - (inset if edges["bottom"] < height else 0), 0),
        ),
        frozenset(side for side, depth in mask_depths.items() if depth is not None),
    )


def detect_autocrop_candidate(
    image: ImageBuffer,
    source_id: str,
    params: BatchDetectParams | None = None,
) -> AutocropCandidate:
    selected = params or BatchDetectParams()
    _validate_params(selected)
    gray, (height, width), (scale_y, scale_x) = _analysis_luminance(image)
    angle = _estimate_angle(gray, selected)
    edge_detection = _analysis_edges(gray, angle, selected)
    if edge_detection is None:
        return AutocropCandidate(source_id, (0, height, 0, width), (height, width), angle, False, False, False, False)
    edges, primary_sides = edge_detection
    left, top, right, bottom = edges
    edge_ratio = (right - left) / max(1, bottom - top)
    plausible_geometry = selected.frame_ratio * 0.8 <= edge_ratio <= selected.frame_ratio * 1.2
    angle_confident = False
    if plausible_geometry:
        angle, angle_confident = _refine_angle(gray, angle, edges, selected)
        edge_detection = _analysis_edges(gray, angle, selected)
        if edge_detection is None:
            return AutocropCandidate(source_id, (0, height, 0, width), (height, width), angle, False, False, False, False)
        edges, primary_sides = edge_detection
        left, top, right, bottom = edges
    roi = (
        round(top / scale_y),
        round(bottom / scale_y),
        round(left / scale_x),
        round(right / scale_x),
    )
    if roi[1] <= roi[0] or roi[3] <= roi[2]:
        raise ValueError("detected crop has no usable area")
    return AutocropCandidate(
        source_id=source_id,
        roi=roi,
        canvas=(height, width),
        angle=angle,
        plausible_geometry=plausible_geometry,
        top_found=top > 0,
        left_found=left > 0,
        right_found=right < gray.shape[1],
        left_primary="left" in primary_sides,
        right_primary="right" in primary_sides,
        angle_confident=angle_confident,
        vertical_profile=_edge_profiles(gray, angle)[2],
    )


def _bounded_interval(center: float, length: int, limit: int) -> tuple[int, int]:
    length = min(length, limit)
    start = min(max(0, round(center - length / 2)), limit - length)
    return start, start + length


def prepare_autocrop_image(
    image: ImageBuffer,
    geometry: GeometryConfig,
    distortion_k1: float = 0.0,
) -> ImageBuffer:
    """Apply the geometry that precedes crop detection in the render pipeline."""
    from negpy.features.geometry.logic import apply_fine_rotation, apply_radial_distortion

    prepared = image
    if geometry.rotation:
        prepared = np.rot90(prepared, k=geometry.rotation)
    if geometry.flip_horizontal:
        prepared = np.fliplr(prepared)
    if geometry.flip_vertical:
        prepared = np.flipud(prepared)
    prepared = np.ascontiguousarray(prepared)
    if geometry.fine_rotation:
        prepared = apply_fine_rotation(prepared, geometry.fine_rotation)
    if distortion_k1:
        prepared = apply_radial_distortion(prepared, distortion_k1)
    return prepared


def _build_templates(candidates: list[AutocropCandidate]) -> dict[bool, BatchTemplate]:
    templates: dict[bool, BatchTemplate] = {}
    for landscape in (False, True):
        trusted = [
            candidate
            for candidate in candidates
            if (candidate.canvas[1] >= candidate.canvas[0]) == landscape
            and candidate.plausible_geometry
            and candidate.top_found
            and candidate.left_found
            and candidate.right_found
        ]
        if not trusted:
            continue
        all_widths = np.asarray([item.roi[3] - item.roi[2] for item in trusted])
        median_width = float(np.median(all_widths))
        width_mad = float(np.median(np.abs(all_widths - median_width)))
        width_tolerance = max(median_width * 0.01, width_mad * 3)
        trusted = [item for item in trusted if abs((item.roi[3] - item.roi[2]) - median_width) <= width_tolerance]
        widths = np.asarray([item.roi[3] - item.roi[2] for item in trusted])
        angles = np.asarray([item.angle for item in trusted])
        median_angle = float(np.median(angles))
        angle_mad = float(np.median(np.abs(angles - median_angle)))
        angle_tolerance = max(0.35, angle_mad * 3)
        angle_inliers = angles[np.abs(angles - median_angle) <= angle_tolerance]
        templates[landscape] = BatchTemplate(
            width=round(float(np.median(widths))),
            fallback_width=round(float(np.quantile(widths, 0.9))),
            width_tolerance=round(width_tolerance),
            center_x=round(float(np.median([(item.roi[2] + item.roi[3]) / 2 for item in trusted]))),
            top=round(float(np.median([item.roi[0] for item in trusted]))),
            angle=round(float(np.median(angle_inliers)), 2),
            angle_tolerance=angle_tolerance,
        )
    return templates


def _finalize_candidate(
    candidate: AutocropCandidate,
    template: BatchTemplate | None,
    frame_ratio: float,
) -> AutocropResult:
    height, width = candidate.canvas
    top, _, left, right = candidate.roi
    full = (0, height, 0, width)
    has_two_corners = candidate.plausible_geometry and candidate.top_found and candidate.left_found and candidate.right_found
    measured_width = right - left
    if has_two_corners and template is not None:
        has_two_corners = abs(measured_width - template.width) <= template.width_tolerance
    if has_two_corners:
        adjusted_width = max(measured_width, template.width if template else measured_width)
        if template and measured_width < template.width - template.width_tolerance / 2:
            adjusted_width = template.fallback_width
        adjusted_angle = (
            template.angle
            if template
            and (not candidate.angle_confident or abs(candidate.angle - template.angle) > template.angle_tolerance)
            else candidate.angle
        )
        bottom = top + round(adjusted_width / frame_ratio)
        if left < 0 or top < 0 or left + adjusted_width > width or bottom > height:
            return AutocropResult(full, 0.0, "manual")
        return AutocropResult((top, bottom, left, left + adjusted_width), adjusted_angle, "two-corner")
    if template is None or candidate.roi == full:
        return AutocropResult(full, 0.0, "manual")
    core_width = min(width, template.fallback_width)
    core_height = min(height, round(core_width / frame_ratio))
    corner_count = int(candidate.left_found) + int(candidate.right_found) if candidate.top_found else 0
    anchor_left = candidate.left_found
    anchor_right = candidate.right_found
    if corner_count == 2 and candidate.plausible_geometry and candidate.left_primary != candidate.right_primary:
        corner_count = 1
        anchor_left = candidate.left_primary
        anchor_right = candidate.right_primary
    if corner_count == 1 and anchor_left:
        left, right = left, left + core_width
    elif corner_count == 1 and anchor_right:
        left, right = right - core_width, right
    else:
        left, right = _bounded_interval(template.center_x, core_width, width)
    if left < 0 or right > width:
        left, right = _bounded_interval(template.center_x, core_width, width)
        corner_count = 0
    use_detected_top = corner_count == 1 and abs(top - template.top) <= height * 0.05
    top = top if use_detected_top else template.top
    angle = candidate.angle if corner_count == 1 and candidate.angle_confident else template.angle
    bottom = top + core_height
    if top < 0 or bottom > height:
        return AutocropResult(full, 0.0, "manual")
    return AutocropResult((top, bottom, round(left), round(right)), angle, "one-corner" if corner_count == 1 else "batch-template")


def _add_outward_margin(
    result: AutocropResult,
    canvas: tuple[int, int],
    outward_frac: float,
) -> AutocropResult:
    if result.method == "manual" or outward_frac == 0:
        return result
    height, width = canvas
    top, bottom, left, right = result.roi
    requested = round(min(height, width) * outward_frac)
    margin = max(0, min(requested, top, left, height - bottom, width - right))
    return AutocropResult(
        (top - margin, bottom + margin, left - margin, right + margin),
        result.angle,
        result.method,
    )


def finalize_autocrop_batch(
    candidates: list[AutocropCandidate],
    params: BatchDetectParams | None = None,
) -> dict[str, AutocropResult]:
    selected = params or BatchDetectParams()
    _validate_params(selected)
    templates = _build_templates(candidates)
    return {
        candidate.source_id: _add_outward_margin(
            _finalize_candidate(
                candidate,
                templates.get(candidate.canvas[1] >= candidate.canvas[0]),
                selected.frame_ratio,
            ),
            candidate.canvas,
            selected.outward_frac,
        )
        for candidate in candidates
    }
