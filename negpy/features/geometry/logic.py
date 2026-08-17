import math
from dataclasses import dataclass, field, replace
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from negpy.domain.types import ROI, ImageBuffer
from negpy.features.geometry.models import (
    FILM_FORMAT_RATIOS,
    FINE_ROTATION_LIMIT,
    AspectRatio,
    AutocropMode,
    GeometryConfig,
)
from negpy.kernel.image.logic import get_luminance
from negpy.kernel.image.validation import ensure_image

AUTOCROP_DETECT_RES = 1800
_ADAPTIVE_THRESHOLD_COUNT = 7
_SIDE_NAMES = ("top", "right", "bottom", "left")
_CORNER_SIDES = {
    "top_left": frozenset({"top", "left"}),
    "top_right": frozenset({"top", "right"}),
    "bottom_right": frozenset({"bottom", "right"}),
    "bottom_left": frozenset({"bottom", "left"}),
}
_EVIDENCE_SOURCE_ORDER = ("adaptive-dark", "adaptive-bright", "inverse-threshold", "blackhat", "edges")


@dataclass(frozen=True)
class AutocropDetection:
    """Evidence-backed outer-film detection.

    ``roi`` uses the same half-open ``(y1, y2, x1, x2)`` convention as the rest
    of the geometry pipeline. ``correction_angle`` is a residual deskew angle in
    degrees that can be added directly to ``GeometryConfig.fine_rotation``.
    ``vertical_edge_profile`` contains one normalized Sobel-X value per input
    column and is copied into a read-only array.
    """

    roi: ROI | None
    correction_angle: float
    confidence: float
    supported_sides: frozenset[str]
    supported_corners: frozenset[str]
    evidence_sources: tuple[str, ...]
    geometry_score: float
    vertical_edge_contrast: float
    vertical_edge_profile: np.ndarray = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.roi is not None:
            y1, y2, x1, x2 = (int(value) for value in self.roi)
            if y2 <= y1 or x2 <= x1:
                raise ValueError("AutocropDetection.roi must be a non-empty half-open ROI")
            object.__setattr__(self, "roi", (y1, y2, x1, x2))
        object.__setattr__(self, "correction_angle", float(np.clip(self.correction_angle, -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT)))
        object.__setattr__(self, "confidence", float(np.clip(self.confidence, 0.0, 1.0)))
        object.__setattr__(self, "geometry_score", float(np.clip(self.geometry_score, 0.0, 1.0)))
        object.__setattr__(self, "vertical_edge_contrast", max(0.0, float(self.vertical_edge_contrast)))
        object.__setattr__(self, "supported_sides", frozenset(self.supported_sides))
        object.__setattr__(self, "supported_corners", frozenset(self.supported_corners))
        object.__setattr__(self, "evidence_sources", tuple(dict.fromkeys(self.evidence_sources)))
        profile = np.asarray(self.vertical_edge_profile, dtype=np.float32).reshape(-1).copy()
        profile.setflags(write=False)
        object.__setattr__(self, "vertical_edge_profile", profile)


@dataclass(frozen=True)
class _FilmCandidate:
    roi: ROI
    correction_angle: float
    source: str
    threshold_index: int | None
    polarity: str
    boundary_score: float
    geometry_score: float
    supported_sides: frozenset[str]
    supported_corners: frozenset[str]


@dataclass(frozen=True)
class _FilmConsensus:
    roi: ROI
    correction_angle: float
    confidence: float
    supported_sides: frozenset[str]
    supported_corners: frozenset[str]
    evidence_sources: tuple[str, ...]
    geometry_score: float
    dispersion: float
    polarity: str


def _normalize_detection_input(img: ImageBuffer, detect_res: int) -> tuple[np.ndarray, float]:
    """
    Downsamples to detect_res longest edge (INTER_AREA). Never upscales.
    Returns (det_img, det_scale) with det_scale <= 1.0.
    """
    h, w = img.shape[:2]
    det_scale = min(1.0, detect_res / max(h, w))
    if det_scale >= 1.0:
        return img, 1.0
    d_w, d_h = max(1, round(w * det_scale)), max(1, round(h * det_scale))
    det = cv2.resize(np.ascontiguousarray(img), (d_w, d_h), interpolation=cv2.INTER_AREA)
    return det, det_scale


def _scale_roi(roi: ROI, det_scale: float, h: int, w: int) -> ROI:
    """
    Maps a detection-space ROI back to input coordinates, clamped to bounds.
    """
    if det_scale >= 1.0:
        return roi
    y1, y2, x1, x2 = roi
    return (
        max(0, int(round(y1 / det_scale))),
        min(h, int(round(y2 / det_scale))),
        max(0, int(round(x1 / det_scale))),
        min(w, int(round(x2 / det_scale))),
    )


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image_float = image.astype(np.float32)
    finite_mask = np.isfinite(image_float)
    if not np.any(finite_mask):
        return np.zeros(image.shape, dtype=np.uint8)

    valid = image_float[finite_mask]
    low = float(np.percentile(valid, 1))
    high = float(np.percentile(valid, 99))
    if high <= low:
        high = low + 1.0

    scaled = np.clip((image_float - low) * (255.0 / (high - low)), 0, 255)
    return scaled.astype(np.uint8)


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or signal.size == 0:
        return signal.astype(np.float32)

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def _boundary_candidates(signal: np.ndarray, *, from_start: bool) -> tuple[int, float, int, float]:
    length = signal.size
    if length == 0:
        return 0, 0.0, 0, 0.0

    edge_window = max(int(round(length * 0.08)), 32)
    edge_window = min(edge_window, max(length - 1, 1))
    search_end = max(int(round(length * 0.45)), edge_window + 1)
    search_end = min(search_end, length)
    search_start = min(int(round(length * 0.55)), max(length - edge_window - 1, 0))

    if from_start:
        edge_slice = signal[:edge_window]
        search_slice = signal[edge_window:search_end]
        edge_idx = int(np.argmax(edge_slice)) if edge_slice.size else 0
        edge_value = float(edge_slice[edge_idx]) if edge_slice.size else 0.0
        if search_slice.size == 0:
            return edge_idx, edge_value, edge_idx, edge_value
        inner_offset = int(np.argmax(search_slice))
        inner_idx = edge_window + inner_offset
        inner_value = float(search_slice[inner_offset])
        return edge_idx, edge_value, inner_idx, inner_value

    edge_slice = signal[length - edge_window :]
    search_slice = signal[search_start : length - edge_window]
    edge_offset = int(np.argmax(edge_slice)) if edge_slice.size else 0
    edge_idx = length - edge_window + edge_offset
    edge_value = float(edge_slice[edge_offset]) if edge_slice.size else 0.0
    if search_slice.size == 0:
        return edge_idx, edge_value, edge_idx, edge_value
    inner_offset = int(np.argmax(search_slice))
    inner_idx = search_start + inner_offset
    inner_value = float(search_slice[inner_offset])
    return edge_idx, edge_value, inner_idx, inner_value


def _dark_region_bounds(image: np.ndarray) -> tuple[int, int, int, int] | None:
    preview = _normalize_to_uint8(_ensure_color(image))
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    threshold = float(np.percentile(gray, 55))
    mask = (gray <= threshold).astype(np.uint8) * 255
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(contour)
    image_area = float(gray.shape[0] * gray.shape[1])
    area_ratio = float(cv2.contourArea(contour)) / max(image_area, 1.0)
    if area_ratio < 0.15 or area_ratio > 0.85:
        return None

    min_width = int(round(image.shape[1] * 0.25))
    min_height = int(round(image.shape[0] * 0.25))
    if box_w < min_width or box_h < min_height:
        return None

    pad_x = max(int(round(image.shape[1] * 0.004)), 4)
    pad_y = max(int(round(image.shape[0] * 0.004)), 4)
    left = max(x - pad_x, 0)
    top = max(y - pad_y, 0)
    right = min(x + box_w + pad_x, image.shape[1])
    bottom = min(y + box_h + pad_y, image.shape[0])

    min_inset_x = int(round(image.shape[1] * 0.03))
    min_inset_y = int(round(image.shape[0] * 0.03))
    if left < min_inset_x or top < min_inset_y or (image.shape[1] - right) < min_inset_x or (image.shape[0] - bottom) < min_inset_y:
        return None

    return left, top, right, bottom


def _refine_frame_bounds(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    preview = _normalize_to_uint8(_ensure_color(image))
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    col_signal = _smooth_signal(np.percentile(grad_x, 95, axis=0), 31)
    row_signal = _smooth_signal(np.percentile(grad_y, 95, axis=1), 31)

    left_edge, left_edge_value, left_inner, left_inner_value = _boundary_candidates(col_signal, from_start=True)
    right_edge, right_edge_value, right_inner, right_inner_value = _boundary_candidates(col_signal, from_start=False)
    top_edge, top_edge_value, top_inner, top_inner_value = _boundary_candidates(row_signal, from_start=True)
    bottom_edge, bottom_edge_value, bottom_inner, bottom_inner_value = _boundary_candidates(row_signal, from_start=False)

    col_noise_floor = float(np.percentile(col_signal, 75))
    row_noise_floor = float(np.percentile(row_signal, 75))

    left = left_edge
    right = right_edge + 1
    top = top_edge
    bottom = bottom_edge + 1

    use_inner_pair_x = (
        left_inner >= int(round(image.shape[1] * 0.12))
        and right_inner <= int(round(image.shape[1] * 0.88))
        and left_inner_value >= col_noise_floor * 4.0
        and right_inner_value >= col_noise_floor * 4.0
        and (right_inner - left_inner) >= int(round(image.shape[1] * 0.5))
    )
    if use_inner_pair_x:
        left = left_inner
        right = right_inner + 1
    else:
        if left_inner >= int(round(image.shape[1] * 0.12)) and left_inner_value >= col_noise_floor * 5.0:
            left = left_inner
        if right_inner <= int(round(image.shape[1] * 0.88)) and right_inner_value >= col_noise_floor * 5.0:
            right = right_inner + 1

    use_inner_pair_y = (
        top_inner > top_edge + 20
        and bottom_inner < bottom_edge - 20
        and top_inner_value > max(top_edge_value * 1.2, row_noise_floor + 25.0)
        and bottom_inner_value > max(bottom_edge_value * 1.2, row_noise_floor + 25.0)
        and (bottom_inner - top_inner) >= int(round(image.shape[0] * 0.5))
    )
    if use_inner_pair_y:
        top = top_inner
        bottom = bottom_inner + 1
    else:
        if top_inner > top_edge + 20 and top_inner_value > max(top_edge_value * 1.45, row_noise_floor + 35.0):
            top = top_inner
        if bottom_inner < bottom_edge - 20 and bottom_inner_value > max(bottom_edge_value * 1.45, row_noise_floor + 35.0):
            bottom = bottom_inner + 1

    pad_x = max(int(round(image.shape[1] * 0.004)), 4)
    pad_y = max(int(round(image.shape[0] * 0.004)), 4)
    left = max(left - pad_x, 0)
    right = min(right + pad_x, image.shape[1])
    top = max(top - pad_y, 0)
    bottom = min(bottom + pad_y, image.shape[0])

    min_width = max(int(round(image.shape[1] * 0.5)), 1)
    min_height = max(int(round(image.shape[0] * 0.5)), 1)
    if right - left < min_width:
        left, right = 0, image.shape[1]
    if bottom - top < min_height:
        top, bottom = 0, image.shape[0]

    refined_area_ratio = ((right - left) * (bottom - top)) / max(float(image.shape[0] * image.shape[1]), 1.0)
    dark_bounds = _dark_region_bounds(image)
    if dark_bounds is not None and refined_area_ratio > 0.8:
        dark_left, dark_top, dark_right, dark_bottom = dark_bounds
        dark_area_ratio = ((dark_right - dark_left) * (dark_bottom - dark_top)) / max(float(image.shape[0] * image.shape[1]), 1.0)
        if 0.15 <= dark_area_ratio <= 0.85:
            left, top, right, bottom = dark_left, dark_top, dark_right, dark_bottom

    return image[top:bottom, left:right], (left, top, right, bottom)


def _mask_from_blackhat(gray: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)


def _mask_from_inverse_threshold(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)


def _mask_from_edges(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 160)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, dilate_kernel, iterations=2)
    return cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel, iterations=2)


def _score_contour(contour: np.ndarray, image_area: float) -> tuple[float, np.ndarray] | None:
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    rect_area = float(width * height)
    if rect_area <= 0:
        return None

    contour_area = float(cv2.contourArea(contour))
    area_ratio = rect_area / image_area
    fill_ratio = contour_area / rect_area if rect_area else 0.0
    short_side = min(width, height)
    long_side = max(width, height)
    aspect_ratio = long_side / max(short_side, 1.0)

    if area_ratio < 0.08:
        return None
    if short_side < 40:
        return None
    if aspect_ratio > 8.0:
        return None

    score = area_ratio * 1.5 + min(fill_ratio, 1.0)
    return score, cv2.boxPoints(rect)


def _odd_kernel_size(value: float, lower: int, upper: int) -> int:
    size = int(round(value))
    size = min(max(size, lower), upper)
    if size % 2 == 0:
        size = size + 1 if size < upper else size - 1
    return max(size, 1)


def _outer_ring_values(lum: np.ndarray) -> np.ndarray:
    # 2% not 5%: a tightly framed scan leaves only 1-3% of bed around the film, so a
    # wider ring reads mostly film, fails the ring_delta gate in
    # _oriented_boundary_evidence, and abstains on an otherwise perfect box.
    h, w = lum.shape[:2]
    ring_width = max(2, round(0.02 * min(h, w)))
    ring_width = min(ring_width, max(1, min(h, w) // 2))
    parts = [lum[:ring_width, :].reshape(-1), lum[-ring_width:, :].reshape(-1)]
    if h > 2 * ring_width:
        parts.extend(
            [
                lum[ring_width:-ring_width, :ring_width].reshape(-1),
                lum[ring_width:-ring_width, -ring_width:].reshape(-1),
            ]
        )
    return np.concatenate(parts)


def _threshold_polarities(lum: np.ndarray, robust_span: float, ring_median: float) -> tuple[str, ...]:
    h, w = lum.shape[:2]
    y1, y2 = round(0.2 * h), round(0.8 * h)
    x1, x2 = round(0.2 * w), round(0.8 * w)
    center = lum[y1 : max(y2, y1 + 1), x1 : max(x2, x1 + 1)]
    center_median = float(np.median(center))
    direction_gate = max(0.025, 0.05 * robust_span)
    if ring_median - center_median > direction_gate:
        return ("dark",)
    if center_median - ring_median > direction_gate:
        return ("bright",)
    return ("dark", "bright")


def _adaptive_threshold_levels(lum: np.ndarray, polarity: str) -> tuple[float, ...]:
    lo, hi = (float(value) for value in np.percentile(lum, (2.0, 98.0)))
    if hi - lo < 0.04:
        return ()
    if polarity == "dark":
        # Bias toward the bright surround. On a three-tier negative this gives the
        # outer film/rebate edge at least as many trials as the inner image edge.
        fractions = (0.35, 0.50, 0.62, 0.72, 0.80, 0.87, 0.93)
    else:
        # Mirror the sampling pressure toward a dark holder for positive polarity.
        fractions = (0.07, 0.13, 0.20, 0.28, 0.38, 0.50, 0.65)
    return tuple(lo + fraction * (hi - lo) for fraction in fractions)


def _adaptive_threshold_mask(blurred_lum: np.ndarray, threshold: float, polarity: str) -> np.ndarray:
    selected = blurred_lum <= threshold if polarity == "dark" else blurred_lum >= threshold
    mask = selected.astype(np.uint8) * 255
    min_dim = min(mask.shape[:2])
    close_size = _odd_kernel_size(0.010 * min_dim, 3, 21)
    open_size = _odd_kernel_size(0.003 * min_dim, 3, 9)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)


def _order_quad(quad: np.ndarray) -> np.ndarray:
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[int(np.argmin(sums))]  # top-left
    ordered[1] = points[int(np.argmin(diffs))]  # top-right
    ordered[2] = points[int(np.argmax(sums))]  # bottom-right
    ordered[3] = points[int(np.argmax(diffs))]  # bottom-left
    return ordered


def _polygon_values(lum: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    h, w = lum.shape[:2]
    points = np.asarray(polygon, dtype=np.float32)
    left = max(0, int(math.floor(float(np.min(points[:, 0])))))
    right = min(w, int(math.ceil(float(np.max(points[:, 0])))) + 1)
    top = max(0, int(math.floor(float(np.min(points[:, 1])))))
    bottom = min(h, int(math.ceil(float(np.max(points[:, 1])))) + 1)
    if right <= left or bottom <= top:
        return np.empty(0, dtype=np.float32)
    local_points = np.rint(points - np.array([left, top], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_points, 1)
    values = lum[top:bottom, left:right][mask.astype(bool)]
    return values[np.isfinite(values)].astype(np.float32, copy=False)


def _oriented_boundary_evidence(
    lum: np.ndarray,
    quad: np.ndarray,
    polarity: str,
    ring_median: float,
    robust_span: float,
) -> tuple[float, frozenset[str], frozenset[str]]:
    ordered = _order_quad(quad)
    edge_lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    band = float(max(2, min(24, round(0.012 * max(1.0, float(np.min(edge_lengths)))))))
    contrast_gate = max(0.02, 0.04 * robust_span)
    contrast_scale = max(0.06, 0.20 * robust_span)
    ring_tolerance = max(0.06, 0.16 * robust_span)

    supported: set[str] = set()
    side_scores: list[float] = []
    for index, name in enumerate(_SIDE_NAMES):
        start = ordered[index]
        end = ordered[(index + 1) % 4]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 4.0:
            continue
        # Ignore corners, where interpolation and sprocket holes mix the two populations.
        # Ordered points run clockwise in image coordinates, so (-dy, dx) is the inward normal.
        inner_start = start + 0.08 * vector
        inner_end = end - 0.08 * vector
        inward = np.array([-vector[1], vector[0]], dtype=np.float32) / length
        inside_polygon = np.array([inner_start, inner_end, inner_end + inward * band, inner_start + inward * band])
        outside_polygon = np.array([inner_start, inner_end, inner_end - inward * band, inner_start - inward * band])
        inside = _polygon_values(lum, inside_polygon)
        outside = _polygon_values(lum, outside_polygon)
        expected = max(8.0, 0.20 * 0.84 * length * band)
        if inside.size < expected or outside.size < expected:
            continue

        inside_median = float(np.median(inside))
        outside_median = float(np.median(outside))
        contrast = outside_median - inside_median if polarity == "dark" else inside_median - outside_median
        ring_delta = abs(outside_median - ring_median)
        if contrast < contrast_gate or ring_delta > ring_tolerance:
            continue

        contrast_score = float(np.clip(contrast / contrast_scale, 0.0, 1.0))
        ring_score = float(np.clip(1.0 - ring_delta / ring_tolerance, 0.0, 1.0))
        supported.add(name)
        side_scores.append(0.75 * contrast_score + 0.25 * ring_score)

    supported_sides = frozenset(supported)
    supported_corners = frozenset(name for name, sides in _CORNER_SIDES.items() if sides <= supported_sides)
    boundary_score = float(sum(side_scores) / 4.0)
    return boundary_score, supported_sides, supported_corners


def _correction_angle_from_quad(quad: np.ndarray, aspect_ratio: float) -> float:
    if aspect_ratio < 1.12:
        return 0.0  # Near-square boxes do not have a stable orientation axis.
    points = np.asarray(quad, dtype=np.float32)
    edges = np.roll(points, -1, axis=0) - points
    edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    # OpenCV's positive fine rotation moves a horizontal edge toward a negative
    # image-coordinate angle, so this measured residual is already the additive fix.
    return float(np.clip(angle, -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT))


def _candidate_from_contour(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    lum: np.ndarray,
    ring_median: float,
    robust_span: float,
    source: str,
    threshold_index: int | None,
    polarity: str | None,
) -> _FilmCandidate | None:
    h, w = image_shape
    image_area = float(h * w)
    rect = cv2.minAreaRect(contour)
    rect_width, rect_height = (float(value) for value in rect[1])
    rect_area = rect_width * rect_height
    if rect_area <= 0.0:
        return None
    contour_area = float(cv2.contourArea(contour))
    area_ratio = rect_area / max(image_area, 1.0)
    short_side = min(rect_width, rect_height)
    long_side = max(rect_width, rect_height)
    aspect_ratio = long_side / max(short_side, 1.0)
    rectangularity = contour_area / rect_area
    if area_ratio < 0.08 or area_ratio > 1.02:
        return None
    if short_side < max(8.0, 0.04 * min(h, w)) or aspect_ratio > 8.0:
        return None
    if rectangularity < 0.30:
        return None

    quad = cv2.boxPoints(rect).astype(np.float32)
    x, y, box_width, box_height = cv2.boundingRect(quad)
    roi = (max(0, y), min(h, y + box_height), max(0, x), min(w, x + box_width))
    if roi[1] <= roi[0] or roi[3] <= roi[2]:
        return None

    best: _FilmCandidate | None = None
    for candidate_polarity in (polarity,) if polarity is not None else ("dark", "bright"):
        boundary_score, supported_sides, supported_corners = _oriented_boundary_evidence(
            lum, quad, candidate_polarity, ring_median, robust_span
        )
        if len(supported_sides) < 2:
            continue
        rectangularity_score = float(np.clip((rectangularity - 0.30) / 0.65, 0.0, 1.0))
        geometry_score = 0.70 * rectangularity_score + 0.30 * (len(supported_sides) / 4.0)
        candidate = _FilmCandidate(
            roi=roi,
            correction_angle=_correction_angle_from_quad(quad, aspect_ratio),
            source=source,
            threshold_index=threshold_index,
            polarity=candidate_polarity,
            boundary_score=boundary_score,
            geometry_score=float(np.clip(geometry_score, 0.0, 1.0)),
            supported_sides=supported_sides,
            supported_corners=supported_corners,
        )
        if best is None or (candidate.boundary_score, candidate.geometry_score) > (best.boundary_score, best.geometry_score):
            best = candidate
    return best


def _candidates_from_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    lum: np.ndarray,
    ring_median: float,
    robust_span: float,
    source: str,
    threshold_index: int | None = None,
    polarity: str | None = None,
) -> list[_FilmCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[_FilmCandidate] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        candidate = _candidate_from_contour(
            contour,
            image_shape,
            lum,
            ring_median,
            robust_span,
            source,
            threshold_index,
            polarity,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _roi_iou(first: ROI, second: ROI) -> float:
    y1 = max(first[0], second[0])
    y2 = min(first[1], second[1])
    x1 = max(first[2], second[2])
    x2 = min(first[3], second[3])
    intersection = max(0, y2 - y1) * max(0, x2 - x1)
    first_area = max(0, first[1] - first[0]) * max(0, first[3] - first[2])
    second_area = max(0, second[1] - second[0]) * max(0, second[3] - second[2])
    return intersection / max(float(first_area + second_area - intersection), 1.0)


def _normalized_edge_distance(first: ROI, second: ROI, image_shape: tuple[int, int]) -> float:
    h, w = image_shape
    scales = (max(h, 1), max(h, 1), max(w, 1), max(w, 1))
    return max(abs(a - b) / scale for a, b, scale in zip(first, second, scales))


def _cluster_film_candidates(candidates: list[_FilmCandidate], image_shape: tuple[int, int]) -> list[list[_FilmCandidate]]:
    count = len(candidates)
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(count):
        for second in range(first + 1, count):
            if _roi_iou(candidates[first].roi, candidates[second].roi) < 0.78:
                continue
            if _normalized_edge_distance(candidates[first].roi, candidates[second].roi, image_shape) <= 0.035:
                union(first, second)

    grouped: dict[int, list[_FilmCandidate]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)
    return list(grouped.values())


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def _dedupe_cluster(cluster: list[_FilmCandidate]) -> list[_FilmCandidate]:
    by_trial: dict[tuple[str, int | None, str], _FilmCandidate] = {}
    for candidate in cluster:
        key = (candidate.source, candidate.threshold_index, candidate.polarity)
        current = by_trial.get(key)
        if current is None or (candidate.boundary_score, candidate.geometry_score) > (current.boundary_score, current.geometry_score):
            by_trial[key] = candidate
    return list(by_trial.values())


def _source_tuple(cluster: list[_FilmCandidate]) -> tuple[str, ...]:
    present = {candidate.source for candidate in cluster}
    ordered = [source for source in _EVIDENCE_SOURCE_ORDER if source in present]
    ordered.extend(sorted(present - set(ordered)))
    return tuple(ordered)


def _supported_by_majority(cluster: list[_FilmCandidate], attribute: str) -> frozenset[str]:
    values = {value for candidate in cluster for value in getattr(candidate, attribute)}
    needed = max(1, math.ceil(0.5 * len(cluster)))
    return frozenset(value for value in values if sum(value in getattr(candidate, attribute) for candidate in cluster) >= needed)


def _select_consensus_cluster(candidates: list[_FilmCandidate], image_shape: tuple[int, int]) -> _FilmConsensus | None:
    h, w = image_shape
    accepted: list[_FilmConsensus] = []
    for raw_cluster in _cluster_film_candidates(candidates, image_shape):
        cluster = _dedupe_cluster(raw_cluster)
        adaptive_trials = {
            (candidate.polarity, candidate.threshold_index)
            for candidate in cluster
            if candidate.source.startswith("adaptive-") and candidate.threshold_index is not None
        }
        structural_sources = {candidate.source for candidate in cluster if not candidate.source.startswith("adaptive-")}
        if len(adaptive_trials) < 3 and not (len(adaptive_trials) >= 2 and len(structural_sources) >= 2):
            continue

        weights = [max(0.05, candidate.boundary_score * candidate.geometry_score) for candidate in cluster]
        roi_values = list(zip(*(candidate.roi for candidate in cluster)))
        roi = tuple(int(round(_weighted_median(list(values), weights))) for values in roi_values)
        if roi[1] <= roi[0] or roi[3] <= roi[2]:
            continue

        scales = (max(h, 1), max(h, 1), max(w, 1), max(w, 1))
        edge_mads = []
        for edge_index, values in enumerate(roi_values):
            center = _weighted_median(list(values), weights)
            edge_mads.append(_weighted_median([abs(value - center) / scales[edge_index] for value in values], weights))
        dispersion = float(max(edge_mads, default=1.0))
        stability = float(np.clip(1.0 - dispersion / 0.03, 0.0, 1.0))

        angles = [candidate.correction_angle for candidate in cluster]
        correction_angle = _weighted_median(angles, weights)
        angle_mad = _weighted_median([abs(angle - correction_angle) for angle in angles], weights)
        angle_stability = float(np.clip(1.0 - angle_mad / 3.0, 0.0, 1.0))
        median_geometry = _weighted_median([candidate.geometry_score for candidate in cluster], weights)
        geometry_score = float(np.clip(0.55 * median_geometry + 0.25 * stability + 0.20 * angle_stability, 0.0, 1.0))
        boundary_score = _weighted_median([candidate.boundary_score for candidate in cluster], weights)
        support = min(1.0, len(adaptive_trials) / _ADAPTIVE_THRESHOLD_COUNT)
        corroboration = min(1.0, len(structural_sources) / 3.0)
        confidence = float(
            np.clip(0.40 * support + 0.25 * stability + 0.20 * boundary_score + 0.10 * geometry_score + 0.05 * corroboration, 0.0, 1.0)
        )
        if confidence < 0.55:
            continue

        polarity_weights = {
            polarity: sum(weight for candidate, weight in zip(cluster, weights) if candidate.polarity == polarity)
            for polarity in ("dark", "bright")
        }
        accepted.append(
            _FilmConsensus(
                roi=roi,
                correction_angle=correction_angle,
                confidence=confidence,
                supported_sides=_supported_by_majority(cluster, "supported_sides"),
                supported_corners=_supported_by_majority(cluster, "supported_corners"),
                evidence_sources=_source_tuple(cluster),
                geometry_score=geometry_score,
                dispersion=dispersion,
                polarity=max(polarity_weights, key=polarity_weights.get),
            )
        )

    if not accepted:
        return None
    # Confidence is primary. A small area preference breaks equivalent inner/outer clusters
    # toward the film extent instead of picture content.
    return max(accepted, key=lambda result: (result.confidence, (result.roi[1] - result.roi[0]) * (result.roi[3] - result.roi[2])))


def _vertical_edge_profile(lum: np.ndarray) -> tuple[np.ndarray, float]:
    if lum.size == 0:
        return np.empty(0, dtype=np.float32), 0.0
    gradient = np.abs(cv2.Sobel(lum.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    profile = np.mean(gradient, axis=0).astype(np.float32)
    window = _odd_kernel_size(0.006 * profile.size, 3, 15)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (window, 1), 0).ravel()
    baseline = float(np.percentile(profile, 50.0)) if profile.size else 0.0
    normalizer = float(np.percentile(profile, 99.5)) if profile.size else 0.0
    contrast = max(0.0, normalizer - baseline)
    if normalizer > 1e-8:
        profile = np.clip(profile / normalizer, 0.0, 1.0)
    else:
        profile.fill(0.0)
    return profile, contrast


def _empty_autocrop_detection(
    vertical_edge_profile: np.ndarray,
    vertical_edge_contrast: float = 0.0,
) -> AutocropDetection:
    return AutocropDetection(
        roi=None,
        correction_angle=0.0,
        confidence=0.0,
        supported_sides=frozenset(),
        supported_corners=frozenset(),
        evidence_sources=(),
        geometry_score=0.0,
        vertical_edge_contrast=vertical_edge_contrast,
        vertical_edge_profile=vertical_edge_profile,
    )


def detect_film_bounds_with_confidence(img: ImageBuffer) -> AutocropDetection:
    """Detect the outer film box by agreement across adaptive thresholds.

    Adaptive intensity trials provide the required consensus. The previous
    inverse-threshold, black-hat, and edge masks are retained as independent
    corroborating evidence, but no structural mask can win by itself.
    """
    if img.ndim < 2 or img.shape[0] < 2 or img.shape[1] < 2:
        return _empty_autocrop_detection(np.empty(0, dtype=np.float32))

    lum = _detection_luma(img)
    vertical_edge_profile, vertical_edge_contrast = _vertical_edge_profile(lum)
    lo, hi = (float(value) for value in np.percentile(lum, (2.0, 98.0)))
    robust_span = hi - lo
    if robust_span < 0.04:
        return _empty_autocrop_detection(vertical_edge_profile, vertical_edge_contrast)
    ring_values = _outer_ring_values(lum)
    ring_median = float(np.median(ring_values))
    polarities = _threshold_polarities(lum, robust_span, ring_median)

    min_dim = min(lum.shape[:2])
    blur_size = _odd_kernel_size(0.004 * min_dim, 3, 9)
    blurred_lum = cv2.GaussianBlur(lum.astype(np.float32), (blur_size, blur_size), 0)
    image_shape = lum.shape[:2]
    candidates: list[_FilmCandidate] = []
    for polarity in polarities:
        source = f"adaptive-{polarity}"
        for threshold_index, threshold in enumerate(_adaptive_threshold_levels(lum, polarity)):
            mask = _adaptive_threshold_mask(blurred_lum, threshold, polarity)
            candidates.extend(
                _candidates_from_mask(
                    mask,
                    image_shape,
                    lum,
                    ring_median,
                    robust_span,
                    source,
                    threshold_index=threshold_index,
                    polarity=polarity,
                )
            )

    color = _ensure_color(img)
    preview = _normalize_to_uint8(color)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    structural_masks = (
        ("inverse-threshold", _mask_from_inverse_threshold(gray)),
        ("blackhat", _mask_from_blackhat(gray)),
        ("edges", _mask_from_edges(gray)),
    )
    for source, mask in structural_masks:
        candidates.extend(_candidates_from_mask(mask, image_shape, lum, ring_median, robust_span, source))

    consensus = _select_consensus_cluster(candidates, image_shape)
    if consensus is None or not _film_surround_is_plausible(lum, consensus.roi):
        return _empty_autocrop_detection(vertical_edge_profile, vertical_edge_contrast)

    roi = _snap_film_bounds_to_bed_gradient(consensus.roi, lum)
    return AutocropDetection(
        roi=roi,
        correction_angle=consensus.correction_angle,
        confidence=consensus.confidence,
        supported_sides=consensus.supported_sides,
        supported_corners=consensus.supported_corners,
        evidence_sources=consensus.evidence_sources,
        geometry_score=consensus.geometry_score,
        vertical_edge_contrast=vertical_edge_contrast,
        vertical_edge_profile=vertical_edge_profile,
    )


def _detect_film_bounds(img: ImageBuffer) -> ROI | None:
    """Compatibility wrapper returning only the half-open outer-film ROI."""
    return detect_film_bounds_with_confidence(img).roi


def _film_surround_is_plausible(lum: np.ndarray, roi: ROI) -> bool:
    """
    A real film box sits on a light bed (uniform, near-clipping surround) or in a
    dark holder. A mid-tone textured surround means the contour latched onto image
    content in a borderless scan — reject so detection falls back to full frame.
    """
    h, w = lum.shape[:2]
    y1, y2, x1, x2 = roi
    out_mask = np.ones((h, w), dtype=bool)
    out_mask[y1:y2, x1:x2] = False
    out = lum[out_mask]
    if out.size < 0.005 * lum.size:
        return True  # box covers nearly the whole scan; no surround evidence either way
    out_med = float(np.median(out))
    out_high = float(np.percentile(out, 90))
    box_med = float(np.median(lum[y1:y2, x1:x2]))
    # Near-clipping, not merely bright. On a thin negative the picture itself clears 0.85,
    # so a median alone accepts a blown region as bed and cuts the frame down to whatever is
    # darker beside it. The bed is the light source and sits at the top of the histogram,
    # where _detection_luma's anchor puts it. A highlight does not.
    bed_like = out_med >= 0.85 and out_high >= 0.98
    holder_like = out_med <= 0.30 and out_med <= box_med - 0.15
    return bed_like or holder_like


def _snap_film_bounds_to_bed_gradient(roi: ROI, lum: np.ndarray) -> ROI:
    """
    Refines contour film bounds to the strongest bed->film luminance gradient within
    a +/-2% window per edge (contour morphology kernels inflate/deflate bounds by
    ~10-20px). Each edge moves only on a dominant gradient; otherwise it is kept.
    """
    h, w = lum.shape[:2]
    y1, y2, x1, x2 = roi
    col_profile = lum[y1:y2, :].mean(axis=0)
    row_profile = lum[:, x1:x2].mean(axis=1)
    # 16px floor: contour morphology kernels are fixed-pixel (21-31px), so their
    # inflation doesn't shrink with image size the way the 2% window does.
    snap = dict(window_out=0.02, window_in=0.02, min_dominance=3.0, min_window_px=16)
    nx1 = _snap_edge_to_gradient(col_profile, x1, is_start=True, **snap)
    nx2 = _snap_edge_to_gradient(col_profile, x2, is_start=False, **snap)
    ny1 = _snap_edge_to_gradient(row_profile, y1, is_start=True, **snap)
    ny2 = _snap_edge_to_gradient(row_profile, y2, is_start=False, **snap)
    if ny2 - ny1 <= 0 or nx2 - nx1 <= 0:
        return roi
    # Keep the bed-to-film transition rows inside the box: downstream refinement needs them,
    # and the (2+offset) crop margin re-tightens afterwards.
    return max(0, ny1 - 2), min(h, ny2 + 2), max(0, nx1 - 2), min(w, nx2 + 2)


class _TierLevels(NamedTuple):
    bed: float
    rebate: float
    image: float
    ring_spread: float


def _detection_luma(img: np.ndarray) -> np.ndarray:
    """
    Luminance normalized so the light bed sits near 1.0 (anchored at P99.5).
    Content-stable alternative to _normalize_to_uint8's 1/99 stretch.
    """
    lum = get_luminance(ensure_image(_ensure_color(img)))
    anchor = float(np.percentile(lum, 99.5))
    return np.clip(lum / max(anchor, 1e-6), 0.0, 2.0)


BORDER_SIDES: Tuple[str, ...] = ("top", "bottom", "left", "right")

# Walk cap per side, as a fraction of that side. Real rebate plus bed slop stays well under
# it, so reaching the cap means the profile never came back down and bright picture content
# touches that edge. Those sides report NaN and pooling ignores them.
_BORDER_WALK_CAP = 0.06
_BORDER_MIN_SEPARATION = 0.15


def measure_film_border(lum: np.ndarray, film_roi: ROI) -> dict[str, float]:
    """
    Per-side thickness of the bright border (rebate + bed slop) just inside a film box,
    as a fraction of that side's length.

    Measurement only, no keep/cut decision: a single frame cannot tell a rebate from a
    bright sky reaching the frame edge. Callers pool these across a roll and take the
    median. A side with no clean border->image transition reports NaN.
    """
    y1, y2, x1, x2 = film_roi
    box = lum[y1:y2, x1:x2]
    bh, bw = box.shape[:2]
    if bh < 16 or bw < 16:
        return {name: float("nan") for name in BORDER_SIDES}

    image_level = float(np.percentile(box, 40))
    row_profile = box.mean(axis=1)
    col_profile = box.mean(axis=0)

    result: dict[str, float] = {}
    for name in BORDER_SIDES:
        profile, length = (row_profile, bh) if name in ("top", "bottom") else (col_profile, bw)
        walk = profile[::-1] if name in ("bottom", "right") else profile
        # A 1-2 px border is too few samples for a median; use a high percentile.
        outer = walk[: max(2, round(0.03 * length))]
        base = float(np.percentile(outer, 90))
        if base - image_level < _BORDER_MIN_SEPARATION:
            result[name] = 0.0  # no bright border on this side (full-bleed edge)
            continue
        threshold = 0.5 * (base + image_level)
        cap = max(3, int(_BORDER_WALK_CAP * length))
        head = walk[:cap]
        # From the brightest sample, not index 0. The box edge usually carries a sliver of bed
        # that ramps up into the rebate, and a walk starting there is already under the
        # threshold and reports zero on a side that has a border.
        peak = int(np.argmax(head))
        if head[peak] <= threshold:
            result[name] = 0.0  # no bright border on this side (full-bleed edge)
            continue
        below = np.flatnonzero(head[peak:] <= threshold)
        result[name] = float("nan") if below.size == 0 else float(peak + int(below[0])) / length
    return result


_OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _roi_from_measured_border(lum: np.ndarray, film_roi: ROI) -> ROI | None:
    """
    Inset a film box by its own measured border, for frames the tier path cannot read.

    Requires an opposite pair to agree before trimming, like _find_rebate_level: a lone
    bright side is usually a uniform scene region, and trusting it carves the picture
    down to a dark subject.
    """
    measured = measure_film_border(lum, film_roi)
    y1, y2, x1, x2 = film_roi
    height, width = y2 - y1, x2 - x1

    # Cap against the opposite side, so one over-long walk cannot cut deep against a much
    # smaller reading across the same gate.
    insets: dict[str, float] = {}
    for name in BORDER_SIDES:
        own, opposite = measured[name], measured[_OPPOSITE_SIDE[name]]
        if not np.isfinite(own):
            own = opposite
        if not np.isfinite(own):
            insets[name] = 0.0
            continue
        if own > 0.0 and np.isfinite(opposite) and opposite > 0.0:
            own = min(own, 2.0 * opposite)
        insets[name] = float(own)

    paired = any(insets[first] > 0.0 and insets[second] > 0.0 for first, second in (("top", "bottom"), ("left", "right")))
    if not paired:
        return None

    roi = (
        y1 + round(insets["top"] * height),
        y2 - round(insets["bottom"] * height),
        x1 + round(insets["left"] * width),
        x2 - round(insets["right"] * width),
    )
    if roi[1] <= roi[0] or roi[3] <= roi[2]:
        return None
    return roi


# Camera-scan holder mask against a bed normalized to 1.0. Dense negative bottoms out near
# 0.05, so the cut sits below that. The mask is not always this clean, which is why the walk
# bridges instead of classifying every pixel.
_EDGE_MASK_LEVEL = 0.03
# Clear film base / bare bed. _detection_luma anchors P99.5 at 1.0, so the bed is a stable
# landmark; picture only reaches here when the frame carries no base at all.
_EDGE_BASE_LEVEL = 0.95
# Gap the walk carries across, covering the mid-gray ramp between mask and base.
_EDGE_BRIDGE = 8
_EDGE_TRIM_CAP = 0.03
_EDGE_CORE = 0.70
_EDGE_QUANTILE = 80
_EDGE_MIN_SUPPORT = 0.45
# Borders that no absolute level finds, read against the picture on the same line. A rebate
# on a thin negative sits above the picture but below the bed, because the frame's own
# highlights set the P99.5 anchor. A holder edge sits below the picture but above the mask
# cut, shading from near-black to mid-grey along the side. Both must end in a step, which is
# what separates them from a bright subject or a deep shadow touching the frame edge.
_EDGE_REBATE_MARGIN = 0.10
# The dark side runs a wider margin: dense negative crowds the bottom of the range, so a
# shadow sits nearer the picture median than a highlight does.
_EDGE_SHADE_MARGIN = 0.16
_EDGE_REBATE_STEP = 0.18
_EDGE_REBATE_REF = 60

# Second, deeper walk. _EDGE_TRIM_CAP is the depth a border may reach on trust alone. Past
# it a border must prove itself, because that much of a side is worth real picture. A wide
# holder margin, a sprocket rail or a neighbouring frame all run several times deeper than
# the cap, and clamping there leaves a band the eye reads as an uncropped edge.
_EDGE_DEEP_CAP = 0.10
# Wider than _EDGE_BRIDGE: a border this thick is layered (mask, ramp, base, sometimes a
# second mask segment) and the mid-gray runs between the tiers are longer at this depth.
_EDGE_DEEP_BRIDGE = 12
# Proof one: the step ends a straight edge. A film boundary is a cut line, so the per-line
# depths fit a sloped line to within a fraction of the allowance. Picture content resolved
# by the same rule scatters several times wider.
_EDGE_DEEP_STRAIGHT = 0.025
# Proof two: the walk stopped on its own. Reaching the allowance means no boundary was
# found inside it, and the depth is then an artefact of where the search was cut off.
_EDGE_DEEP_FULL = 0.92
# Proof three: the step carries most of the drop from the outer level to the picture. A
# boundary does that and a subject shading toward the edge does not. Replaces the fixed
# _EDGE_REBATE_STEP at depth, where a dense frame compresses every level.
_EDGE_STEP_SHARE = 0.45
_EDGE_STEP_FLOOR = 0.06
# Proof four: the outermost pixels are one material. Mask, bed and film base each hold their
# level along the whole side, so the outer level barely moves from line to line, while
# picture reaching the frame edge moves with the subject.
_EDGE_DEEP_OUTER_SPREAD = 0.15

_EDGE_ORIENT = {
    "top": lambda box: box,
    "bottom": lambda box: box[::-1],
    "left": lambda box: box.T,
    "right": lambda box: box.T[::-1],
}


def _walk_border_depth(lines: np.ndarray, limit: int, bridge: int, relative_step: bool) -> np.ndarray:
    """
    Per-line depth of the non-image border, walking inward from the frame edge.

    A camera scan puts two tiers outside the picture, either of which may be absent: an
    opaque holder mask at the very edge, then a clear film-base sliver. Neither is
    separable by level alone — the mask overlaps dense negative around 0.05 and the ramp
    between the tiers sits squarely in picture range — but the tiers are adjacent, so the
    walk carries across a short gap and stops at the last border evidence instead of the
    first picture-like pixel.

    relative_step scales the step a soft-ended run must show by that run's own drop,
    instead of the fixed _EDGE_REBATE_STEP; see _EDGE_STEP_SHARE.
    """
    head = lines[: limit + 1]
    # Picture level per line, taken past the deepest cut allowed so a thick border
    # cannot raise its own reference and hide itself.
    picture = np.median(lines[limit : limit + _EDGE_REBATE_REF], axis=0)

    firm = (head < _EDGE_MASK_LEVEL) | (head > _EDGE_BASE_LEVEL)
    # Tiers no absolute level reaches, read against the picture on the same line: a rebate
    # brighter than the picture but under the bed, and a holder edge darker than the picture
    # but over the mask cut. A holder edge shades from near-black to mid-grey along the side,
    # so the absolute cut catches only the dark end.
    reference = picture[None, :]
    border = firm | (head > reference + _EDGE_REBATE_MARGIN) | (head < reference - _EDGE_SHADE_MARGIN)

    depth = np.zeros(head.shape[1], dtype=np.int32)
    last = np.full(head.shape[1], -1, dtype=np.int32)
    soft = np.zeros(head.shape[1], dtype=bool)
    alive = np.ones(head.shape[1], dtype=bool)
    # Tracked alongside: how deep the mask and bed alone reach. Firm evidence is its own
    # proof, so wherever it runs the border runs and the gradient below never arbitrates. It
    # matters when the border stacks two firm tiers, mask outside and film base inside: the
    # outer boundary is the stronger step, and a gradient picks that and leaves the base
    # inside the frame.
    firm_depth = np.zeros(head.shape[1], dtype=np.int32)
    firm_last = np.full(head.shape[1], -1, dtype=np.int32)
    firm_alive = np.ones(head.shape[1], dtype=bool)
    for i in range(head.shape[0]):
        # Evidence must chain back to the frame edge: the border tiers are the outermost
        # things on the film, never something found well inside the picture.
        alive &= (i - last) <= bridge
        firm_alive &= (i - firm_last) <= bridge
        if not alive.any() and not firm_alive.any():
            break
        hit = alive & border[i]
        last = np.where(hit, i, last)
        depth = np.where(hit, i + 1, depth)
        soft = np.where(hit, ~firm[i], soft)

        firm_hit = firm_alive & firm[i]
        firm_last = np.where(firm_hit, i, firm_last)
        firm_depth = np.where(firm_hit, i + 1, firm_depth)

    # Where the walk ended on a relative test alone, the border edge comes from the strongest
    # gradient inside the run, not from where the level test gave out. A level test only
    # nominates candidates: against a bright picture, "darker than the reference" stays true
    # well into the picture's own shadows and the run ends mid-ramp. The gradient is the
    # boundary.
    #
    # That peak must also be a real step. A border stops at the frame edge, a hard boundary,
    # while a bright subject or deep shadow running to the picture edge shades off, and
    # reading that as a border carves the frame down. Magnitude, not sign: a rebate is
    # brighter than the picture it ends against and a holder edge is darker. Mask and bed
    # need no such proof, since nothing else on the film reaches those levels.
    span = np.abs(lines[3 : limit + 3] - lines[:limit])
    reach = np.arange(span.shape[0])[:, None] < depth[None, :]
    masked = np.where(reach, span, -1.0)
    peak = masked.max(axis=0)
    # The outermost span that nearly matches the peak, not the peak itself. Where the picture
    # keeps rising past the boundary, a span straddling the step plus some picture outscores
    # the step alone and drags the cut inward. Ties go outward, keeping picture.
    peak_at = np.argmax(masked >= (0.9 * peak)[None, :], axis=0)
    # span[j] compares j against j+3, so a step between j+2 and j+3 first peaks at j: the
    # boundary is the far end of the winning span, not its middle.
    resolved = np.clip(peak_at + 3, 0, limit)
    if relative_step:
        # Scaled by the run's own drop, not fixed. A dense frame compresses every level toward
        # black, so its borders step by less while still stepping by nearly all there is.
        outer = np.median(head[:3], axis=0)
        gate = np.maximum(_EDGE_STEP_FLOOR, _EDGE_STEP_SHARE * np.abs(outer - picture))
    else:
        gate = np.full(peak.shape, _EDGE_REBATE_STEP, dtype=peak.dtype)
    depth = np.where(soft, np.where(peak >= gate, resolved, 0), depth)
    return np.maximum(depth, firm_depth)


def _pool_border_depth(depth: np.ndarray, limit: int) -> int:
    if float((depth > 0).mean()) < _EDGE_MIN_SUPPORT:
        return 0  # a minority of lines: incidental highlights, not a border
    return int(min(limit, np.percentile(depth, _EDGE_QUANTILE)))


def _border_edge_scatter(depth: np.ndarray) -> float:
    """
    How far the resolved depths sit off a straight line, in pixels (median residual).

    A film boundary is a cut line, so a robust fit through the per-line depths leaves
    almost nothing; the slope absorbs the tilt every frame has. Picture content resolved
    by the same rule follows the subject instead and scatters.
    """
    live = depth > 0
    if int(live.sum()) < 8:
        return float("inf")
    index = np.nonzero(live)[0].astype(np.float64)
    value = depth[live].astype(np.float64)
    step = max(1, index.size // 48)
    a, b = index[::step], value[::step]
    # Theil-Sen over a coarse subsample: the median pairwise slope ignores the runs that
    # follow a subject, which a least-squares fit would let tilt the line.
    da, db = a[:, None] - a[None, :], b[:, None] - b[None, :]
    upper = np.triu(np.ones_like(da, dtype=bool), 1) & (da != 0)
    slope = float(np.median(db[upper] / da[upper])) if upper.any() else 0.0
    intercept = float(np.median(value - slope * index))
    return float(np.median(np.abs(value - (slope * index + intercept))))


def _edge_border_depth(box: np.ndarray, side: str, limit: int, deep_limit: int = 0) -> int:
    """
    Depth of the non-image border on one side of a film box, in pixels.

    Walks each line inward on its own and pools the depths — per line rather than per row
    because these frames are slightly tilted, and one row crosses both border tiers and the
    picture at once, which smears every per-row statistic together.

    Two passes. The first reaches `limit` and is believed as it stands. The second reaches
    `deep_limit`, and is adopted only where it goes deeper and proves itself: it must stop
    inside its allowance, leave a straight edge, and end on a step that carries most of the
    run's drop. Past `limit` a wrong answer costs real picture, so the burden is on the
    deeper reading.
    """
    lines = _EDGE_ORIENT[side](box)
    n = lines.shape[1]
    margin = int(round((1.0 - _EDGE_CORE) * 0.5 * n))
    if n - 2 * margin >= 8:
        lines = lines[:, margin : n - margin]
    if lines.shape[0] < limit + 8 or lines.shape[1] < 8:
        return 0

    near = _pool_border_depth(_walk_border_depth(lines, limit, _EDGE_BRIDGE, False), limit)
    if deep_limit <= limit or lines.shape[0] < deep_limit + 8:
        return near

    deep = _walk_border_depth(lines, deep_limit, _EDGE_DEEP_BRIDGE, True)
    far = _pool_border_depth(deep, deep_limit)
    if far <= near or far >= _EDGE_DEEP_FULL * deep_limit:
        return near
    if _border_edge_scatter(deep) > _EDGE_DEEP_STRAIGHT * deep_limit:
        return near
    outer = np.median(lines[:3], axis=0)[deep > 0]
    if float(np.median(np.abs(outer - np.median(outer)))) > _EDGE_DEEP_OUTER_SPREAD:
        return near
    return far


def measure_film_edges(lum: np.ndarray, film_roi: ROI) -> dict[str, float]:
    """
    Per-side holder-mask + film-base thickness inside a film box, as a fraction of that
    side. 0.0 means the side carries no border of either tier.

    Companion to measure_film_border, which reads a single bright tier through a ring 4%
    of the box wide. A camera scan that fills the sensor compresses mask and base into a
    few pixels, so that ring straddles both tiers plus the picture and abstains; this
    walks the edge itself instead.
    """
    y1, y2, x1, x2 = film_roi
    box = lum[y1:y2, x1:x2]
    bh, bw = box.shape[:2]
    if bh < 16 or bw < 16:
        return {name: 0.0 for name in BORDER_SIDES}

    ly = int(round(_EDGE_TRIM_CAP * bh))
    lx = int(round(_EDGE_TRIM_CAP * bw))
    dy = int(round(_EDGE_DEEP_CAP * bh))
    dx = int(round(_EDGE_DEEP_CAP * bw))
    return {
        "top": _edge_border_depth(box, "top", ly, dy) / bh,
        "bottom": _edge_border_depth(box, "bottom", ly, dy) / bh,
        "left": _edge_border_depth(box, "left", lx, dx) / bw,
        "right": _edge_border_depth(box, "right", lx, dx) / bw,
    }


def _trim_film_edges(lum: np.ndarray, film_roi: ROI) -> ROI:
    """
    Inset a film box past the holder mask and the film-base sliver on each side.

    Sides are independent: a frame sitting off-centre in the holder shows a border on some
    sides and none on others, so requiring an opposite pair (as _roi_from_measured_border
    does) would leave the border on. The caps are what bound the damage instead — picture
    running to the frame edge is trimmed by at most _EDGE_TRIM_CAP of the side, or
    _EDGE_DEEP_CAP where the deeper walk proved a border.
    """
    y1, y2, x1, x2 = film_roi
    height, width = y2 - y1, x2 - x1
    measured = measure_film_edges(lum, film_roi)

    roi = (
        y1 + round(measured["top"] * height),
        y2 - round(measured["bottom"] * height),
        x1 + round(measured["left"] * width),
        x2 - round(measured["right"] * width),
    )
    if roi[1] - roi[0] < 0.5 * height or roi[3] - roi[2] < 0.5 * width:
        return film_roi
    return roi


def _find_rebate_level(lum: np.ndarray, film_roi: ROI) -> Optional[Tuple[float, float]]:
    """
    Searches the four border strips inside the film box for a rebate plateau:
    a uniform strip (low spread, sprocket holes excluded) clearly present on at
    least one side. The rebate can exist on some sides only (cut film strips,
    full-bleed edges), so sides are evaluated independently and the brightest
    qualifying plateau wins. Returns (rebate_level, spread) or None.
    """
    y1, y2, x1, x2 = film_roi
    box = lum[y1:y2, x1:x2]
    bh, bw = box.shape[:2]
    if bh < 16 or bw < 16:
        return None
    bed = float(np.percentile(lum, 99))
    box_median = float(np.percentile(box, 50))
    ring_w = max(3, round(0.04 * min(bh, bw)))
    sides = {
        "top": box[:ring_w, :],
        "bottom": box[-ring_w:, :],
        "left": box[:, :ring_w],
        "right": box[:, -ring_w:],
    }

    qualifying: dict[str, Tuple[float, float]] = {}
    for name, strip in sides.items():
        vals = strip[strip < bed - 0.05]  # exclude sprocket holes / bed slop in the box
        if vals.size < 0.25 * strip.size:
            continue
        spread = float(np.percentile(vals, 80) - np.percentile(vals, 20))
        if spread > 0.10:
            continue  # textured strip = image content reaches this film edge
        p60 = float(np.percentile(vals, 60))
        if p60 < box_median + 0.10:
            continue  # rebate is the lowest-density tier: must sit clearly above the
            # image interior; a plateau near the median is bright image content
            # reaching the film edge (full-bleed) or holder slop, not film base
        qualifying[name] = (p60, spread)

    # A genuine film rebate borders the frame, so it shows up as an opposite pair. A lone
    # qualifying side is almost always a uniform bright scene region in a full-bleed frame,
    # and trusting it carves the picture down to a dark subject. Require a pair.
    has_pair = ("top" in qualifying and "bottom" in qualifying) or ("left" in qualifying and "right" in qualifying)
    if not has_pair:
        return None
    return max(qualifying.values(), key=lambda t: t[0])


def _estimate_tier_levels(lum: np.ndarray, film_roi: ROI) -> Optional[_TierLevels]:
    """
    Estimates the three luminance tiers (bed, rebate, exposed image) for a film box.
    Returns None when the tiers are not reliably separable.
    """
    y1, y2, x1, x2 = film_roi
    box = lum[y1:y2, x1:x2]

    found = _find_rebate_level(lum, film_roi)
    if found is None:
        return None
    rebate, ring_spread = found
    bed = float(np.percentile(lum, 99))

    dark = box[box < rebate - 0.02]
    if dark.size < 0.05 * box.size:
        return None
    image_level = float(np.percentile(dark, 30))

    separation = rebate - image_level
    if separation < max(0.04, 3.0 * ring_spread):
        return None
    return _TierLevels(bed, rebate, image_level, ring_spread)


def _longest_run_above(profile: np.ndarray, threshold: float) -> Optional[Tuple[int, int]]:
    """
    Longest contiguous half-open index run with profile >= threshold.
    """
    idx = np.where(profile >= threshold)[0]
    if idx.size == 0:
        return None
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    k = int(np.argmax(idx[ends] - idx[starts]))
    return int(idx[starts[k]]), int(idx[ends[k]]) + 1


def _snap_edge_to_gradient(
    profile: np.ndarray,
    idx: int,
    is_start: bool,
    window_out: float = 0.06,
    window_in: float = 0.02,
    min_dominance: float = 2.0,
    min_window_px: int = 3,
) -> int:
    """
    Snaps a coarse edge index to the strongest |gradient| of the smoothed profile
    within an asymmetric window biased outward (toward the film border) — recovers
    frame area when bright content touching the edge suppresses occupancy.
    Keeps idx unless the peak clearly dominates the window (>= 2x median).
    """
    n = profile.size
    if n < 8:
        return idx
    sm = _smooth_signal(profile, max(3, round(0.01 * n)))
    grad = np.abs(np.diff(sm))
    out_px = max(min_window_px, round(window_out * n))
    in_px = max(min_window_px, round(window_in * n))
    if is_start:
        lo, hi = max(0, idx - out_px), min(n - 1, idx + in_px)
    else:
        lo, hi = max(0, idx - in_px), min(n - 1, idx + out_px)
    window = grad[lo:hi]
    if window.size == 0:
        return idx
    peak = float(np.max(window))
    if peak >= min_dominance * float(np.median(window)) + 1e-6:
        # Box smoothing turns a sharp step into a nearly flat plateau, so picking the first
        # raw maximum lets sub-ULP OpenCV differences move the snapped edge by several pixels
        # across resolutions and platforms. Treat equivalent maxima as one peak and keep the
        # transition nearest the coarse contour edge.
        near_peak = np.flatnonzero(np.isclose(window, peak, rtol=1e-5, atol=1e-7))
        transitions = lo + near_peak + 1
        return int(transitions[np.argmin(np.abs(transitions - idx))])
    return idx


def _refine_film_roi_by_tiers(lum: np.ndarray, film_roi: ROI) -> Optional[Tuple[ROI, np.ndarray, np.ndarray]]:
    """
    Tier-based image-area refinement: classify pixels inside the film box against the
    rebate/image midpoint, take the longest occupancy runs, then snap each edge to the
    strongest local luminance gradient. Returns (roi, row_occupancy, col_occupancy) in
    detection-image coords (profiles padded to full image length), or None when tiers
    are not separable (caller falls back to the Sobel path).
    """
    levels = _estimate_tier_levels(lum, film_roi)
    if levels is None:
        return None

    y1, y2, x1, x2 = film_roi
    box = lum[y1:y2, x1:x2]
    bh, bw = box.shape[:2]
    # Image pixels are anything meaningfully darker than the rebate plateau, since any
    # exposure adds density over base+fog. This keeps thin, bright frame regions classified as
    # image. The midpoint is the floor for tightly separated tiers.
    threshold = max(
        0.5 * (levels.rebate + levels.image),
        levels.rebate - max(0.04, 3.0 * levels.ring_spread),
    )
    mask = box < threshold

    row_occ = mask.mean(axis=1)
    vrun = _longest_run_above(row_occ, 0.55)
    if vrun is None:
        return None
    vt, vb = vrun
    # Restrict to the vertical run so rebate rows don't dilute column occupancy.
    col_occ = mask[vt:vb].mean(axis=0)
    hrun = _longest_run_above(col_occ, 0.55)
    if hrun is None:
        return None
    hl, hr = hrun

    # Single-frame sanity. Multi-frame strip scans fail here and fall back to Sobel.
    if (vb - vt) < 0.35 * bh or (hr - hl) < 0.35 * bw:
        return None
    area_ratio = ((vb - vt) * (hr - hl)) / float(bh * bw)
    if not (0.15 <= area_ratio <= 0.95):
        return None

    col_profile = box[vt:vb, :].mean(axis=0)
    row_profile = box[:, hl:hr].mean(axis=1)
    hl = _snap_edge_to_gradient(col_profile, hl, is_start=True)
    hr = _snap_edge_to_gradient(col_profile, hr, is_start=False)
    vt = _snap_edge_to_gradient(row_profile, vt, is_start=True)
    vb = _snap_edge_to_gradient(row_profile, vb, is_start=False)

    pad_y = max(2, round(0.004 * bh))
    pad_x = max(2, round(0.004 * bw))
    vt, vb = max(0, vt - pad_y), min(bh, vb + pad_y)
    hl, hr = max(0, hl - pad_x), min(bw, hr + pad_x)
    if vb - vt <= 0 or hr - hl <= 0:
        return None

    h_det, w_det = lum.shape[:2]
    row_occ_full = np.zeros(h_det, dtype=np.float32)
    row_occ_full[y1:y2] = row_occ
    col_occ_full = np.zeros(w_det, dtype=np.float32)
    col_occ_full[x1:x2] = col_occ

    return (y1 + vt, y1 + vb, x1 + hl, x1 + hr), row_occ_full, col_occ_full


def _refine_roi_to_image(img: ImageBuffer, film_roi: ROI) -> Tuple[ROI, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Refines a film-extent ROI inward to the exposed image area (rebate excluded).
    Tier-based refinement first, then the per-side edge walk, then the per-side border
    measurement, then Sobel.
    Returns (roi, row_occupancy | None, col_occupancy | None).
    """
    lum = _detection_luma(img)
    refined = _refine_film_roi_by_tiers(lum, film_roi)
    if refined is not None:
        return refined

    # Ahead of the border measurement, which reads one bright tier through a narrow ring. A
    # camera scan that fills the sensor compresses mask and base into a few pixels, so that
    # ring straddles both plus the picture and abstains.
    walked = _trim_film_edges(lum, film_roi)
    if walked != film_roi:
        return walked, None, None

    measured = _roi_from_measured_border(lum, film_roi)
    if measured is not None:
        return measured, None, None

    if _find_rebate_level(lum, film_roi) is None:
        # No uniform rebate plateau on any side means image content runs to the film edge.
        # There is nothing to refine away, and Sobel would cut into the picture.
        return film_roi, None, None

    y1, y2, x1, x2 = film_roi
    _, (ref_left, ref_top, ref_right, ref_bottom) = _refine_frame_bounds(img[y1:y2, x1:x2])
    roi = (y1 + ref_top, y1 + ref_bottom, x1 + ref_left, x1 + ref_right)

    # Rebate is a small fraction of the film area, so a Sobel cut removing more than a
    # quarter of the box latched onto image content.
    film_area = max(1, (y2 - y1) * (x2 - x1))
    if (roi[1] - roi[0]) * (roi[3] - roi[2]) < 0.75 * film_area:
        return film_roi, None, None
    return roi, None, None


def _find_autocrop_roi_from_contours(img: ImageBuffer) -> ROI | None:
    film_roi = _detect_film_bounds(img)
    if film_roi is None:
        return None
    roi, _, _ = _refine_roi_to_image(img, film_roi)
    return roi


def _get_threshold_autocrop_coords(
    img: ImageBuffer,
    assist_luma: Optional[float],
) -> ROI:
    """
    Luminance-threshold fallback. Expects a detection-resolution image
    (see _normalize_detection_input); returns a det-space ROI.
    """
    h, w = img.shape[:2]
    lum = get_luminance(ensure_image(img))

    threshold = 0.96
    if assist_luma is not None:
        threshold = float(np.clip(assist_luma - 0.02, 0.5, 0.98))

    rows_det = np.where(np.mean(lum, axis=1) < threshold)[0]
    cols_det = np.where(np.mean(lum, axis=0) < threshold)[0]

    if len(rows_det) < 10 or len(cols_det) < 10:
        return 0, h, 0, w

    return int(rows_det[0]), int(rows_det[-1]), int(cols_det[0]), int(cols_det[-1])


def _trim_opaque_border(
    lum: np.ndarray,
    roi: ROI,
    black: float = 0.02,
    frac: float = 0.7,
    max_trim: float = 0.2,
) -> ROI:
    """
    Shrinks each ROI edge inward past a contiguous band of opaque (near-black)
    pixels — a camera-scan negative holder masks frame edges with an opaque stripe
    (lum ~ 0), well below the darkest real negative content (even unexposed film
    base transmits orange light). An edge moves only while its border line is
    dominated (>= `frac`) by sub-`black` pixels, capped at `max_trim` of the side
    so it can never eat into the image. `lum` is detection luminance (bed ~ 1.0).
    """
    y1, y2, x1, x2 = roi
    sub = lum[y1:y2, x1:x2]
    bh, bw = sub.shape[:2]
    if bh < 4 or bw < 4:
        return roi

    row_black = (sub < black).mean(axis=1)
    col_black = (sub < black).mean(axis=0)

    def _run(profile: np.ndarray, limit: int, from_start: bool) -> int:
        n = profile.size
        i = 0
        while i < limit and profile[i if from_start else n - 1 - i] >= frac:
            i += 1
        return i

    ly = int(round(max_trim * bh))
    lx = int(round(max_trim * bw))
    top = _run(row_black, ly, True)
    bottom = _run(row_black, ly, False)
    left = _run(col_black, lx, True)
    right = _run(col_black, lx, False)

    ny1, ny2, nx1, nx2 = y1 + top, y2 - bottom, x1 + left, x2 - right
    if ny2 - ny1 <= 0 or nx2 - nx1 <= 0:
        return roi
    return ny1, ny2, nx1, nx2


def apply_fine_rotation(img: ImageBuffer, angle: float) -> ImageBuffer:
    """
    Sub-degree rotation (bilinear).
    """
    if angle == 0.0:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    res = cv2.warpAffine(
        img,
        m_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return ensure_image(res)


# Radial lens-distortion correction (poly3 / Brown-Conrady k1). The radius is normalized to
# the image half-diagonal (r=1 at the corner), so it is rotation and aspect invariant.
# Forward resample map (corrected pixel -> distorted sample), s = scale-to-fill:
#     P_src = (s * P_out) * (1 + k1 * |s * P_out|^2 / halfdiag^2)
# Mirrored in transform.wgsl (uv/aspect form) and inverted in map_point_radial. Change
# the model in all three.

_DISTORT_EPS = 1e-6


def _radial_center(w: int, h: int) -> Tuple[float, float, float]:
    # Center in pixel-index convention to match the WGSL `(coord+0.5)/dims - 0.5`.
    return (w - 1) * 0.5, (h - 1) * 0.5, 0.5 * math.hypot(w, h)


def compute_distortion_scale(k1: float, w: int, h: int, _samples: int = 128) -> float:
    """Largest scale at which the output frame still maps fully inside the input — fills
    the frame without empty/replicated borders. Numeric, so it's sign-agnostic (the
    binding point is a corner or an edge midpoint depending on barrel vs pincushion)."""
    if abs(k1) < 1e-9:
        return 1.0

    cx, cy, halfdiag = _radial_center(w, h)
    hw, hh = cx, cy
    inv_hd2 = 1.0 / (halfdiag * halfdiag)

    edge = max(1, _samples // 4)
    pts = []
    for i in range(edge):
        t = i / edge
        pts.append((-hw + 2 * hw * t, -hh))
        pts.append((-hw + 2 * hw * t, hh))
        pts.append((-hw, -hh + 2 * hh * t))
        pts.append((hw, -hh + 2 * hh * t))

    def max_ratio(s: float) -> float:
        worst = 0.0
        for px, py in pts:
            pxs, pys = px * s, py * s
            f = 1.0 + k1 * (pxs * pxs + pys * pys) * inv_hd2
            if f <= 0.0:
                return math.inf  # fold-over: scale is too large
            worst = max(worst, abs(pxs * f) / hw, abs(pys * f) / hh)
        return worst

    lo, hi = 1e-3, 1.0
    while max_ratio(hi) < 1.0 and hi < 1e3:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if max_ratio(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _radial_maps(k1: float, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """cv2.remap source-coordinate maps for the radial correction (incl. scale-to-fill)."""
    cx, cy, halfdiag = _radial_center(w, h)
    s = compute_distortion_scale(k1, w, h)
    inv_hd2 = 1.0 / (halfdiag * halfdiag)
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    px = (xs - cx) * s
    py = (ys - cy) * s
    f = 1.0 + k1 * (px * px + py * py) * inv_hd2
    map_x = (cx + px * f).astype(np.float32)
    map_y = (cy + py * f).astype(np.float32)
    return map_x, map_y


def apply_radial_distortion(img: ImageBuffer, k1: float) -> ImageBuffer:
    """Radial lens-distortion correction. Purely geometric — moves pixels via a
    coordinate remap, never scales values (brightness-preserving). No-op for k1≈0."""
    if abs(k1) < _DISTORT_EPS:
        return img
    h, w = img.shape[:2]
    map_x, map_y = _radial_maps(k1, w, h)
    res = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return ensure_image(res)


def map_point_radial(px: float, py: float, k1: float, w: int, h: int) -> Tuple[float, float]:
    """Inverse of the resample map: given a point in the undistorted (pre-correction)
    image, return where it lands in the corrected output. Used by coordinate mappers so
    feature placements (crop corners, retouch spots, dodge/burn masks) stay aligned."""
    if abs(k1) < _DISTORT_EPS:
        return px, py
    cx, cy, halfdiag = _radial_center(w, h)
    s = compute_distortion_scale(k1, w, h)
    ix, iy = px - cx, py - cy
    r_in = math.hypot(ix, iy)
    if r_in < 1e-9:
        return px, py
    # Solve (k1/halfdiag^2)·t^3 + t − r_in = 0 for t = |P_s| ≥ 0.
    roots = np.roots([k1 / (halfdiag * halfdiag), 0.0, 1.0, -r_in])
    real = [rt.real for rt in roots if abs(rt.imag) < 1e-6 and rt.real > 0]
    t = min(real, key=lambda v: abs(v - r_in)) if real else r_in
    scale = (t / s) / r_in
    return cx + ix * scale, cy + iy * scale


# Easel tilt and swing: a plane-to-plane projectivity, the output rect sampling a
# trapezoid of the source. The unit is per-cent of the frame, not tilt degrees, which
# need an enlarger magnification and focal length the pipeline does not model. Mirrored
# in transform.wgsl and in map_coords_to_geometry; change the quad in all three.

_KEYSTONE_EPS = 1e-4  # per-cent


def keystone_matrix_normalized(converge_v: float, converge_h: float) -> np.ndarray:
    """Forward map (source -> corrected) in the GPU's sampling convention,
    u = (index + 0.5) / dims.

    The single definition of the quad: the pixel-space matrix and the shader's inverse
    both derive from it. Convergences are per-cent; positive converge_v stretches the
    top edge, positive converge_h the left.
    """
    a, b = converge_v * 0.005, converge_h * 0.005
    src = np.float32([[a, b], [1.0 - a, -b], [1.0 + a, 1.0 + b], [-a, 1.0 - b]])
    dst = np.float32([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def _norm_to_index(w: int, h: int) -> np.ndarray:
    """Normalized [0,1] -> OpenCV pixel-index coords, where pixel i sits at index i."""
    return np.array([[float(w), 0.0, -0.5], [0.0, float(h), -0.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def keystone_matrix(converge_v: float, converge_h: float, w: int, h: int) -> np.ndarray:
    """The same map in OpenCV pixel-index coords, for warpPerspective."""
    a = _norm_to_index(w, h)
    return a @ keystone_matrix_normalized(converge_v, converge_h) @ np.linalg.inv(a)


def apply_keystone(img: ImageBuffer, converge_v: float, converge_h: float) -> ImageBuffer:
    """Perspective correction. Same canvas size, replicated edges: the wedge stays and
    the user crops it, as fine rotation does. Scaling to fill would change magnification
    and move an existing crop."""
    if abs(converge_v) < _KEYSTONE_EPS and abs(converge_h) < _KEYSTONE_EPS:
        return img
    h, w = img.shape[:2]
    res = cv2.warpPerspective(
        img,
        keystone_matrix(converge_v, converge_h, w, h),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return ensure_image(res)


def map_point_keystone(px: float, py: float, converge_v: float, converge_h: float, w: int, h: int) -> Tuple[float, float]:
    """Where a pre-correction point lands in the corrected output."""
    if abs(converge_v) < _KEYSTONE_EPS and abs(converge_h) < _KEYSTONE_EPS:
        return px, py
    m = keystone_matrix(converge_v, converge_h, w, h)
    den = m[2, 0] * px + m[2, 1] * py + m[2, 2]
    if abs(den) < 1e-12:
        return px, py
    return (
        float((m[0, 0] * px + m[0, 1] * py + m[0, 2]) / den),
        float((m[1, 0] * px + m[1, 1] * py + m[1, 2]) / den),
    )


def keystone_inverse_normalized(converge_v: float, converge_h: float) -> np.ndarray:
    """The keystone's inverse (corrected -> source), which the shader consumes directly
    rather than rebuilding the quad. Identity when both are zero."""
    if abs(converge_v) < _KEYSTONE_EPS and abs(converge_h) < _KEYSTONE_EPS:
        return np.eye(3, dtype=np.float64)
    m = np.linalg.inv(keystone_matrix_normalized(converge_v, converge_h))
    return m / m[2, 2]


def apply_margin_to_roi(
    roi: ROI,
    h: int,
    w: int,
    margin_px: float,
) -> ROI:
    """
    Expands/Contracts ROI.
    """
    y1, y2, x1, x2 = roi
    ny1, ny2, nx1, nx2 = y1 + margin_px, y2 - margin_px, x1 + margin_px, x2 - margin_px
    return int(max(0, ny1)), int(min(h, ny2)), int(max(0, nx1)), int(min(w, nx2))


def scale_roi_inset(film_roi: ROI, roi: ROI, factor: float) -> ROI:
    """
    Scales how far a refined ROI sits inside its film box; 1.0 leaves it untouched.

    Applied to the result of every refinement route so Rebate Trim means one thing.
    Inert when the two boxes are equal (Film mode, and detections never refined).
    """
    if factor == 1.0:
        return roi
    fy1, fy2, fx1, fx2 = film_roi
    y1, y2, x1, x2 = roi
    scaled = (
        fy1 + round((y1 - fy1) * factor),
        fy2 - round((fy2 - y2) * factor),
        fx1 + round((x1 - fx1) * factor),
        fx2 - round((fx2 - x2) * factor),
    )
    if scaled[1] <= scaled[0] or scaled[3] <= scaled[2]:
        return roi
    return scaled


def _resolve_ratio_dims(cw: int, ch: int, target_ratio_str: str) -> Tuple[float, float]:
    """
    Returns (target_w, target_h) <= (cw, ch) for the orientation-corrected ratio.
    """
    try:
        w_r, h_r = map(float, target_ratio_str.split(":"))
        if h_r == 0:
            h_r = 1
        target_aspect = w_r / h_r
    except ValueError:
        target_aspect = 1.5

    is_vertical = ch > cw
    if is_vertical:
        if target_aspect > 1.0:
            target_aspect = 1.0 / target_aspect
    else:
        if target_aspect < 1.0:
            target_aspect = 1.0 / target_aspect

    current_aspect = cw / ch
    if current_aspect > target_aspect:
        return ch * target_aspect, float(ch)
    return float(cw), cw / target_aspect


def enforce_roi_aspect_ratio(
    roi: ROI,
    h: int,
    w: int,
    target_ratio_str: str = "3:2",
) -> ROI:
    """
    Centers ROI within aspect ratio.
    """
    y1, y2, x1, x2 = roi
    cw, ch = x2 - x1, y2 - y1

    if cw <= 0 or ch <= 0:
        return 0, h, 0, w

    if target_ratio_str == "Free":
        return int(max(0, y1)), int(min(h, y2)), int(max(0, x1)), int(min(w, x2))

    target_w, target_h = _resolve_ratio_dims(cw, ch, target_ratio_str)
    if target_w < cw:
        nx1 = x1 + (cw - target_w) / 2
        nx2 = nx1 + target_w
        x1, x2 = int(nx1), int(nx2)
    elif target_h < ch:
        ny1 = y1 + (ch - target_h) / 2
        ny2 = ny1 + target_h
        y1, y2 = int(ny1), int(ny2)

    return int(max(0, y1)), int(min(h, y2)), int(max(0, x1)), int(min(w, x2))


def _place_window_by_occupancy(start: int, end: int, target_len: float, occupancy: np.ndarray, scale: float) -> int:
    """
    Slides a target_len window within [start, end) (input coords) to maximize summed
    occupancy (detection coords; scale = det/input ratio). Ties resolve toward the
    centered position, so uniform occupancy reproduces plain centering.
    Returns the new window start in input coords.
    """
    d_start = max(0, int(round(start * scale)))
    d_end = min(occupancy.size, int(round(end * scale)))
    d_len = max(1, int(round(target_len * scale)))
    if d_len >= d_end - d_start:
        return start

    cs = np.concatenate(([0.0], np.cumsum(occupancy[d_start:d_end], dtype=np.float64)))
    n_pos = (d_end - d_start) - d_len + 1
    scores = cs[d_len : d_len + n_pos] - cs[:n_pos]
    candidates = np.where(scores >= float(np.max(scores)) - 1e-9)[0]
    centered = ((d_end - d_start) - d_len) / 2.0
    k = int(candidates[np.argmin(np.abs(candidates - centered))])

    new_start = round((d_start + k) / scale)
    return int(min(max(start, new_start), end - target_len))


def _enforce_ratio_by_occupancy(
    roi: ROI,
    h: int,
    w: int,
    target_ratio_str: str,
    row_occupancy: np.ndarray,
    col_occupancy: np.ndarray,
    det_scale: float,
) -> ROI:
    """
    Like enforce_roi_aspect_ratio, but places the shrink-axis window where the
    image-class occupancy is highest instead of blindly centering.
    """
    y1, y2, x1, x2 = roi
    cw, ch = x2 - x1, y2 - y1

    if cw <= 0 or ch <= 0:
        return 0, h, 0, w

    if target_ratio_str == "Free":
        return int(max(0, y1)), int(min(h, y2)), int(max(0, x1)), int(min(w, x2))

    target_w, target_h = _resolve_ratio_dims(cw, ch, target_ratio_str)
    if target_w < cw:
        x1 = _place_window_by_occupancy(x1, x2, target_w, col_occupancy, det_scale)
        x2 = int(round(x1 + target_w))
    elif target_h < ch:
        y1 = _place_window_by_occupancy(y1, y2, target_h, row_occupancy, det_scale)
        y2 = int(round(y1 + target_h))

    return int(max(0, y1)), int(min(h, y2)), int(max(0, x1)), int(min(w, x2))


def get_manual_rect_coords(
    img_or_shape: ImageBuffer | Tuple[int, int],
    manual_rect: Tuple[float, float, float, float],
    offset_px: int = 0,
    scale_factor: float = 1.0,
) -> ROI:
    """
    Maps a normalized manual crop rect to a pixel ROI in the TRANSFORMED image.

    The rect is expressed in the coordinate space of the already-transformed image
    (post rotation / fine-rotation / flip / distortion) — the same space the user draws
    it on in the canvas overlay — so it is a plain axis-aligned slice: no corner mapping,
    no bounding-box collapse. Storing it in raw space instead forced the crop through
    `map_coords_to_geometry` + an AABB, which inflated the region as fine rotation tilted
    the mapped rect (the crop grew larger than the drawn box).
    """
    if isinstance(img_or_shape, tuple):
        h_curr, w_curr = img_or_shape
    else:
        h_curr, w_curr = img_or_shape.shape[:2]

    x1_n, y1_n, x2_n, y2_n = manual_rect
    xs = (x1_n * w_curr, x2_n * w_curr)
    ys = (y1_n * h_curr, y2_n * h_curr)

    ix1, ix2 = int(min(xs)), int(max(xs))
    iy1, iy2 = int(min(ys)), int(max(ys))

    roi = (iy1, iy2, ix1, ix2)
    margin = offset_px * scale_factor
    return apply_margin_to_roi(roi, h_curr, w_curr, margin)


def get_manual_crop_coords(
    img: ImageBuffer,
    offset_px: int = 0,
    scale_factor: float = 1.0,
) -> ROI:
    """
    Center crop + offset.
    """
    h, w = img.shape[:2]
    roi = (0, h, 0, w)
    margin = offset_px * scale_factor
    return apply_margin_to_roi(roi, h, w, margin)


def get_autocrop_coords(
    img: ImageBuffer,
    offset_px: int = 0,
    scale_factor: float = 1.0,
    target_ratio_str: str = "3:2",
    detect_res: int = AUTOCROP_DETECT_RES,
    assist_point: Optional[Tuple[float, float]] = None,
    assist_luma: Optional[float] = None,
    mode: str = AutocropMode.IMAGE,
    rebate_trim: float = 1.0,
) -> ROI:
    """
    Detects film border via density thresholding.

    mode="film" crops to the film extent (rebate/sprockets kept);
    mode="image" refines inward to the exposed image area, by rebate_trim of the way
    (1.0 = the detected image edge).
    """
    h, w = img.shape[:2]
    det, det_scale = _normalize_detection_input(img, detect_res)
    film_roi = _detect_film_bounds(det)
    from_contours = film_roi is not None
    if film_roi is None:
        film_roi = _get_threshold_autocrop_coords(det, assist_luma)

    # Trim the opaque holder stripes the film detection left in. They sit at the absolute
    # frame edge and become a white band after inversion.
    lum = _detection_luma(det)
    film_roi = _trim_opaque_border(lum, film_roi)

    row_occ = col_occ = None
    if mode == AutocropMode.FILM:
        roi = film_roi
    elif from_contours:
        roi, row_occ, col_occ = _refine_roi_to_image(det, film_roi)
    else:
        # The threshold fallback's box is too loose to refine inward, but the mask/base walk
        # reads the edges themselves and needs no well-found box.
        roi = _trim_film_edges(lum, film_roi)

    roi = scale_roi_inset(film_roi, roi, rebate_trim)
    roi = _scale_roi(roi, det_scale, h, w)

    ratio_str = target_ratio_str
    if ratio_str == AspectRatio.FREE:
        ratio_str = _closest_standard_ratio(roi, (h, w), fallback="3:2").value

    margin = (2 + offset_px) * scale_factor
    roi = apply_margin_to_roi(roi, h, w, margin)

    if row_occ is None or col_occ is None:
        return enforce_roi_aspect_ratio(roi, h, w, ratio_str)
    return _enforce_ratio_by_occupancy(roi, h, w, ratio_str, row_occ, col_occ, det_scale)


def has_manual_crop(geometry: GeometryConfig) -> bool:
    """True when the crop was drawn or adjusted by hand, so Auto must not overwrite it."""
    return geometry.crop_rect is not None and not geometry.crop_from_auto


def autocrop_detection_key(geometry: GeometryConfig) -> str:
    """Everything border detection reads, as one comparable string.

    A resolved auto crop stays valid while this is unchanged. Crop Offset is absent: it is
    applied to the stored rect on every render, so it must not throw the detection away.
    """
    return "|".join(
        str(part)
        for part in (
            geometry.rotation,
            int(geometry.flip_horizontal),
            int(geometry.flip_vertical),
            round(geometry.fine_rotation, 4),
            round(geometry.converge_v, 4),
            round(geometry.converge_h, 4),
            geometry.autocrop_ratio,
            geometry.autocrop_mode,
            round(geometry.autocrop_rebate_trim, 4),
        )
    )


def resolve_autocrop_rect(
    img: ImageBuffer,
    geometry: GeometryConfig,
    preview_size: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Detect the auto crop and return a normalized rect in transformed-image space, as
    GeometryConfig.crop_rect stores it and the canvas overlay draws it.

    The only border detection a render reaches; the caller freezes the result, so every
    later render of the edit crops identically at any resolution. Runs on a copy normalized
    to AUTOCROP_DETECT_RES, replaying the GeometryProcessor transform order
    (rot90 -> flips -> fine rotation).

    The rect excludes Crop Offset, which get_manual_rect_coords adds back on every render;
    only the 2 px baseline margin is baked in. None when the buffer is too small.
    """
    h, w = img.shape[:2]
    det_s = min(1.0, AUTOCROP_DETECT_RES / max(h, w))
    if det_s < 1.0:
        tmp = cv2.resize(img, (max(1, round(w * det_s)), max(1, round(h * det_s))), interpolation=cv2.INTER_AREA)
    else:
        tmp = img
    if geometry.rotation != 0:
        tmp = np.rot90(tmp, k=geometry.rotation)
    if geometry.flip_horizontal:
        tmp = np.fliplr(tmp)
    if geometry.flip_vertical:
        tmp = np.flipud(tmp)
    tmp = np.ascontiguousarray(tmp.astype(np.float32, copy=False))
    if geometry.fine_rotation != 0.0:
        tmp = apply_fine_rotation(tmp, geometry.fine_rotation)
    # Detection must see the frame the render produces, or the rect is found on a
    # straight rebate and applied to a keystoned one.
    tmp = apply_keystone(tmp, geometry.converge_v, geometry.converge_h)

    rh, rw = tmp.shape[:2]
    if rh < 2 or rw < 2:
        return None
    y1, y2, x1, x2 = get_autocrop_coords(
        tmp,
        offset_px=0,
        # The margin is in detection pixels, so it scales by the detection buffer's own
        # size, not the source's.
        scale_factor=max(rh, rw) / float(preview_size),
        target_ratio_str=geometry.autocrop_ratio,
        mode=geometry.autocrop_mode,
        rebate_trim=geometry.autocrop_rebate_trim,
    )
    if y2 - y1 < 2 or x2 - x1 < 2:
        return None
    return (x1 / rw, y1 / rh, x2 / rw, y2 / rh)


def map_coords_to_geometry(
    nx: float,
    ny: float,
    orig_shape: Tuple[int, int],
    rotation_k: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    roi: Optional[ROI] = None,
    distortion_k1: float = 0.0,
    converge_v: float = 0.0,
    converge_h: float = 0.0,
) -> Tuple[float, float]:
    """
    Maps raw coordinates to geometry-transformed space.
    """
    h_orig, w_orig = orig_shape
    px, py = nx * w_orig, ny * h_orig
    h, w = h_orig, w_orig

    k = rotation_k % 4
    if k == 1:
        px, py = py, w - px
        h, w = w, h
    elif k == 2:
        px, py = w - px, h - py
    elif k == 3:
        px, py = h - py, px
        h, w = w, h

    if flip_horizontal:
        px = w - px
    if flip_vertical:
        py = h - py

    if fine_rotation != 0.0:
        center = (w / 2.0, h / 2.0)
        m_mat = cv2.getRotationMatrix2D(center, fine_rotation, 1.0)
        pt = np.array([px, py, 1.0])
        res_pt = m_mat @ pt
        px, py = float(res_pt[0]), float(res_pt[1])

    # Inverse of the resample map: undistorted feature point -> corrected-image position
    # (last forward op, matching GeometryProcessor / transform.wgsl).
    if distortion_k1 != 0.0:
        px, py = map_point_radial(px, py, distortion_k1, w, h)

    if converge_v != 0.0 or converge_h != 0.0:
        px, py = map_point_keystone(px, py, converge_v, converge_h, w, h)

    if roi:
        y1, y2, x1, x2 = roi
        px -= x1
        py -= y1
        h, w = y2 - y1, x2 - x1

    nx_new = np.clip(px / max(w, 1), 0.0, 1.0)
    ny_new = np.clip(py / max(h, 1), 0.0, 1.0)

    return float(nx_new), float(ny_new)


def smooth_polyline(
    pts: List[Tuple[float, float]],
    closed: bool = True,
    samples_per_seg: int = 16,
) -> List[Tuple[float, float]]:
    """Densify a polyline into a uniform Catmull-Rom curve through its points.

    The curve interpolates every control point (a control point is the t=0 sample
    of its segment). Fewer than 3 points are returned unchanged. `closed` wraps the
    tangents so the loop joins smoothly; open keeps the endpoints fixed.
    """
    n = len(pts)
    if n < 3:
        return [(float(x), float(y)) for x, y in pts]

    p = np.asarray(pts, dtype=np.float64)
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out: List[Tuple[float, float]] = []
    n_seg = n if closed else n - 1
    for i in range(n_seg):
        p0 = p[(i - 1) % n] if closed else p[max(i - 1, 0)]
        p1 = p[i]
        p2 = p[(i + 1) % n] if closed else p[i + 1]
        p3 = p[(i + 2) % n] if closed else p[min(i + 2, n - 1)]
        seg = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
        out.extend((float(x), float(y)) for x, y in seg)
    if not closed:
        out.append((float(p[-1][0]), float(p[-1][1])))
    return out


def translate_normalized_rect(
    rect: Tuple[float, float, float, float],
    dx: float,
    dy: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    max_x1 = max(0.0, 1.0 - w)
    max_y1 = max(0.0, 1.0 - h)
    nx1 = min(max(x1 + dx, 0.0), max_x1)
    ny1 = min(max(y1 + dy, 0.0), max_y1)
    return (nx1, ny1, nx1 + w, ny1 + h)


def mirror_normalized_rect(
    rect: Tuple[float, float, float, float],
    horizontal: bool,
) -> Tuple[float, float, float, float]:
    """
    Mirrors a normalized (x1, y1, x2, y2) rect across the image's vertical
    (horizontal=True) or horizontal (horizontal=False) center line, keeping
    corners ordered.
    """
    x1, y1, x2, y2 = rect
    if horizontal:
        return (1.0 - x2, y1, 1.0 - x1, y2)
    return (x1, 1.0 - y2, x2, 1.0 - y1)


def rotate_normalized_rect(
    rect: Tuple[float, float, float, float],
    quarter_turns_ccw: int,
) -> Tuple[float, float, float, float]:
    """
    Rotates a normalized (x1, y1, x2, y2) rect by whole quarter-turns within the
    display (transformed) image, keeping corners ordered.

    `quarter_turns_ccw` counts 90° counter-clockwise turns (negative = clockwise) —
    the same handedness as the geometry `rotation` field, where k=1 turns the display
    CCW. When the image content rotates one quarter CCW, a feature at (u, v) moves to
    (v, 1 - u); this rotates the crop/analysis rect along with the content it frames so
    it keeps outlining the same area after a 90°/180° rotate.
    """
    x1, y1, x2, y2 = rect
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for _ in range(quarter_turns_ccw % 4):
        corners = [(v, 1.0 - u) for (u, v) in corners]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def toggle_flip(geo: GeometryConfig, horizontal: bool) -> GeometryConfig:
    """
    Toggles a mirror on the geometry so the result is an exact mirror of the
    CURRENTLY rendered image. The pipeline applies flips BEFORE fine rotation,
    and mirror(rotate(+a, img)) == rotate(-a, mirror(img)) — so each single
    mirror must negate the fine-rotation angle, or toggling a flip visibly
    changes the horizon (the tilt doubles instead of mirroring). The crop
    rect lives in transformed space and mirrors along with the content
    it frames.
    """
    if horizontal:
        new_geo = replace(geo, flip_horizontal=not geo.flip_horizontal)
    else:
        new_geo = replace(geo, flip_vertical=not geo.flip_vertical)
    if geo.fine_rotation != 0.0:
        new_geo = replace(new_geo, fine_rotation=-geo.fine_rotation)
    if geo.crop_rect is not None:
        new_geo = replace(new_geo, crop_rect=mirror_normalized_rect(geo.crop_rect, horizontal))
    return new_geo


def straighten_delta_degrees(dx: float, dy: float) -> float:
    """
    Fine-rotation delta (stored convention: positive = CCW on screen) that levels a
    line drawn on the displayed image, snapping to the user's intent: lines closer
    to horizontal straighten to the horizon, lines closer to vertical straighten to
    plumb (a building edge). Direction-agnostic — drawing the same line from either
    end yields the same correction.

    Screen coords have y growing downward, so atan2(dy, dx) measures clockwise from
    east. A line tilted right-end-down (angle +θ) needs the image rotated CCW by θ
    to level — which is +θ in the stored convention — so the deviation from the
    nearest axis is the delta directly. The result is in (-45°, 45°]; deltas are
    additive on top of the current fine rotation because the stored angle rotates
    the *displayed* frame CCW regardless of flips/90° turns (flips apply before
    fine rotation in the pipeline).
    """
    theta = math.degrees(math.atan2(dy, dx)) % 180.0  # fold direction ambiguity
    if theta <= 45.0:
        return theta  # near-horizontal
    if theta < 135.0:
        return theta - 90.0  # near-vertical
    return theta - 180.0  # near-horizontal, other fold


def rotation_drag_angle(
    start_angle_deg: float,
    center: Tuple[float, float],
    press: Tuple[float, float],
    cursor: Tuple[float, float],
    sensitivity: float = 1.0,
    limit: float = FINE_ROTATION_LIMIT,
) -> float:
    """
    Fine-rotation angle for a crop-tool rotation-handle drag: the signed arc the
    cursor swept around `center` (screen coords), scaled by `sensitivity` and added
    to the drag-start angle. Screen y grows downward while positive fine rotation
    is counter-clockwise on screen, so the arc is negated — the image follows the
    cursor like a grabbed wheel. Result is clamped to ±limit degrees.
    """
    a0 = math.atan2(press[1] - center[1], press[0] - center[0])
    a1 = math.atan2(cursor[1] - center[1], cursor[0] - center[0])
    # Shortest signed difference, robust across the ±180° atan2 seam.
    delta = math.degrees(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0)))
    new_angle = start_angle_deg - delta * sensitivity
    return float(np.clip(new_angle, -limit, limit))


def _closest_standard_ratio(roi: ROI, img_shape: Tuple[int, int], fallback: str = "3:2") -> AspectRatio:
    """
    Returns the standard AspectRatio closest to the ROI's aspect (log-space distance),
    sanity-checked against the full image dimensions.
    """
    h_img, w_img = img_shape

    y1, y2, x1, x2 = roi
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return AspectRatio(fallback)

    detected = cw / ch
    is_landscape = cw >= ch

    # FILM_FORMAT_RATIOS, not the full AspectRatio enum. The picker's print and screen sizes
    # are not formats a camera or scanner produces, and they sit close enough to 3:2 and 5:4
    # in log-ratio space that ordinary detection noise misclassifies a real 3:2 frame.
    candidates: list[tuple[AspectRatio, float]] = []
    for ratio in FILM_FORMAT_RATIOS:
        try:
            w_r, h_r = map(float, ratio.value.split(":"))
        except ValueError:
            continue
        target = w_r / h_r
        target_landscape = target >= 1.0
        if is_landscape != target_landscape and target != 1.0:
            continue
        candidates.append((ratio, target))

    if not candidates:
        return AspectRatio(fallback)

    best = min(candidates, key=lambda c: abs(math.log(max(detected, 1e-6)) - math.log(max(c[1], 1e-6))))

    # If the chosen ratio disagrees strongly with the full image dimensions, re-detect against
    # those dimensions. This guards against ROI detection stretching the bounding box, for
    # example reading 2.7:1 on a genuine 3:2 frame and snapping to 65:24.
    #
    # Only when the ROI is the more elongated of the two: a stretched box is longer than the
    # image that holds it, never squarer. Without that condition the guard overrules every
    # film format rounder than the sensor, and a correctly read 1:1 6x6 frame on a 3:2 camera
    # is forced back to 3:2.
    img_ratio = w_img / h_img
    stretched = abs(math.log(max(detected, 1e-6))) > abs(math.log(max(img_ratio, 1e-6)))
    if stretched and abs(math.log(max(img_ratio, 1e-6)) - math.log(max(best[1], 1e-6))) > 0.3:
        best = min(candidates, key=lambda c: abs(math.log(max(img_ratio, 1e-6)) - math.log(max(c[1], 1e-6))))

    return best[0]


def detect_closest_aspect_ratio(img: ImageBuffer, fallback: str = "3:2") -> AspectRatio:
    """
    Detect film frame and return the closest standard AspectRatio enum member.
    Falls back to `fallback` if frame detection fails.
    """
    h_img, w_img = img.shape[:2]

    det, det_scale = _normalize_detection_input(img, AUTOCROP_DETECT_RES)
    roi = _find_autocrop_roi_from_contours(det)
    if roi is None:
        roi = _get_threshold_autocrop_coords(det, None)
    roi = _scale_roi(roi, det_scale, h_img, w_img)

    return _closest_standard_ratio(roi, (h_img, w_img), fallback)
