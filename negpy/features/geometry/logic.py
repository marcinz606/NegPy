import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import negpy_core
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
    det = negpy_core.resize_area(np.ascontiguousarray(img, dtype=np.float32), (d_w, d_h))
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
    return negpy_core.normalize_to_uint8(np.ascontiguousarray(image, dtype=np.float32))


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    return image


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    return negpy_core.smooth_signal(np.ascontiguousarray(signal, dtype=np.float32), int(window))


def _refine_frame_bounds(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """The picture inside a film box by Sobel profiles of its gray preview, and its box
    (left, top, right, bottom); negpy_core holds the rules."""
    left, top, right, bottom = negpy_core.refine_frame_bounds(np.ascontiguousarray(image, dtype=np.float32))
    return image[top:bottom, left:right], (left, top, right, bottom)


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
    """Detect the outer film box by agreement across adaptive thresholds, corroborated by
    structural masks that cannot win alone; negpy_core holds the rules."""
    if img.ndim < 2 or img.shape[0] < 2 or img.shape[1] < 2:
        return _empty_autocrop_detection(np.empty(0, dtype=np.float32))
    consensus, profile, contrast = negpy_core.detect_film_bounds(np.ascontiguousarray(img, dtype=np.float32))
    if consensus is None:
        return _empty_autocrop_detection(profile, contrast)
    roi, angle, confidence, sides, corners, sources, geometry = consensus
    return AutocropDetection(
        roi=roi,
        correction_angle=angle,
        confidence=confidence,
        supported_sides=frozenset(sides),
        supported_corners=frozenset(corners),
        evidence_sources=tuple(sources),
        geometry_score=geometry,
        vertical_edge_contrast=contrast,
        vertical_edge_profile=profile,
    )


def _detect_film_bounds(img: ImageBuffer) -> ROI | None:
    """Compatibility wrapper returning only the half-open outer-film ROI."""
    return detect_film_bounds_with_confidence(img).roi


def _detection_luma(img: np.ndarray) -> np.ndarray:
    """
    Luminance normalized so the light bed sits near 1.0 (anchored at P99.5).
    Content-stable alternative to _normalize_to_uint8's 1/99 stretch.
    """
    return negpy_core.detection_luma(get_luminance(ensure_image(_ensure_color(img))))


BORDER_SIDES: Tuple[str, ...] = ("top", "bottom", "left", "right")


def measure_film_border(lum: np.ndarray, film_roi: ROI) -> dict[str, float]:
    """
    Per-side thickness of the bright border (rebate + bed slop) just inside a film box,
    as a fraction of that side's length.

    Measurement only, no keep/cut decision: a single frame cannot tell a rebate from a
    bright sky reaching the frame edge. Callers pool these across a roll and take the
    median. A side with no clean border->image transition reports NaN.
    """
    sides = negpy_core.measure_film_border(np.ascontiguousarray(lum, dtype=np.float32), tuple(int(v) for v in film_roi))
    return dict(zip(BORDER_SIDES, sides))


def _roi_from_measured_border(lum: np.ndarray, film_roi: ROI) -> ROI | None:
    """
    Inset a film box by its own measured border, for frames the tier path cannot read.

    Requires an opposite pair to agree before trimming, like _find_rebate_level: a lone
    bright side is usually a uniform scene region, and trusting it carves the picture
    down to a dark subject.
    """
    return negpy_core.roi_from_measured_border(np.ascontiguousarray(lum, dtype=np.float32), tuple(int(v) for v in film_roi))


def measure_film_edges(lum: np.ndarray, film_roi: ROI) -> dict[str, float]:
    """
    Per-side holder-mask + film-base thickness inside a film box, as a fraction of that
    side. 0.0 means the side carries no border of either tier.

    Companion to measure_film_border, which reads a single bright tier through a ring 4%
    of the box wide. A camera scan that fills the sensor compresses mask and base into a
    few pixels, so that ring straddles both tiers plus the picture and abstains; this
    walks the edge itself instead.
    """
    sides = negpy_core.measure_film_edges(np.ascontiguousarray(lum, dtype=np.float32), tuple(int(v) for v in film_roi))
    return dict(zip(BORDER_SIDES, sides))


def _trim_film_edges(lum: np.ndarray, film_roi: ROI) -> ROI:
    """
    Inset a film box past the holder mask and the film-base sliver on each side.

    Sides are independent: a frame sitting off-centre in the holder shows a border on some
    sides and none on others, so requiring an opposite pair (as _roi_from_measured_border
    does) would leave the border on. The caps are what bound the damage instead — picture
    running to the frame edge is trimmed by at most 3% of the side, or 10% where the
    deeper walk proved a border (negpy_core holds the walk and its constants).
    """
    return negpy_core.trim_film_edges(np.ascontiguousarray(lum, dtype=np.float32), tuple(int(v) for v in film_roi))


def _find_rebate_level(lum: np.ndarray, film_roi: ROI, bed: Optional[float] = None) -> Optional[Tuple[float, float]]:
    """
    Searches the four border strips inside the film box for a rebate plateau:
    a uniform strip (low spread, sprocket holes excluded) clearly present on at
    least one side. The rebate can exist on some sides only (cut film strips,
    full-bleed edges), so sides are evaluated independently and the brightest
    qualifying plateau wins. Returns (rebate_level, spread) or None. `bed` is the 99th
    percentile of `lum`, for a caller that already has it.
    """
    return negpy_core.find_rebate_level(np.ascontiguousarray(lum, dtype=np.float32), tuple(int(v) for v in film_roi), bed)


def _refine_film_roi_by_tiers(lum: np.ndarray, film_roi: ROI) -> Optional[Tuple[ROI, np.ndarray, np.ndarray]]:
    """
    Tier-based image-area refinement: classify pixels inside the film box against the
    rebate/image midpoint, take the longest occupancy runs, then snap each edge to the
    strongest local luminance gradient. Returns (roi, row_occupancy, col_occupancy) in
    detection-image coords (profiles padded to full image length), or None when tiers
    are not separable (caller falls back to the Sobel path).
    """
    return negpy_core.refine_film_roi_by_tiers(np.ascontiguousarray(lum, dtype=np.float32), tuple(int(v) for v in film_roi))


def film_and_picture(det: np.ndarray) -> Tuple[Optional[ROI], ROI]:
    """The film box (None when the scan shows no film edge) and the picture gate inside it,
    at detection resolution. With no film box the gate is refined over the whole frame; a
    degenerate gate (a blank or dark frame) falls back to the whole frame."""
    return negpy_core.film_and_picture(np.ascontiguousarray(det, dtype=np.float32))


def rebate_mask(shape: Tuple[int, int], film_roi: Optional[ROI], gate: ROI) -> np.ndarray:
    """The clear rebate: inside the film box, outside the picture gate. All False without a
    film box, because the ring would then be whatever surrounds the picture, bare light included."""
    rebate = np.zeros(shape, bool)
    if film_roi is not None:
        fy1, fy2, fx1, fx2 = film_roi
        gy1, gy2, gx1, gx2 = gate
        rebate[fy1:fy2, fx1:fx2] = True
        rebate[gy1:gy2, gx1:gx2] = False
    return rebate


def _refine_roi_to_image(img: ImageBuffer, film_roi: ROI) -> Tuple[ROI, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Refines a film-extent ROI inward to the exposed image area (rebate excluded).
    Tier-based refinement first, then the per-side edge walk, then the per-side border
    measurement, then Sobel.
    Returns (roi, row_occupancy | None, col_occupancy | None).
    """
    return negpy_core.refine_roi_to_image(np.ascontiguousarray(img, dtype=np.float32), tuple(int(v) for v in film_roi))


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

    box = negpy_core.threshold_box(np.ascontiguousarray(lum, dtype=np.float32), threshold)
    return box if box is not None else (0, h, 0, w)


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
    dominated (>= `frac`) by sub-`black` pixels, capped at `max_trim` of the side.

    A run reaching that cap trims nothing. The stripe has a finite width, so a run
    that never ends found no boundary and its depth is only where the search stopped.
    A slide breaks the premise the other way: its shadows reach true black, so dark
    picture running to the frame edge reads exactly like the mask and would be trimmed
    as deep as the cap allows. `lum` is detection luminance (bed ~ 1.0).
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
        return 0 if i >= limit else i

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
    m_mat = negpy_core.rotation_matrix((w / 2.0, h / 2.0), angle, 1.0)
    return negpy_core.warp_affine(np.ascontiguousarray(img, dtype=np.float32), m_mat, (w, h))


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


@lru_cache(maxsize=32)
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


def apply_radial_distortion(img: ImageBuffer, k1: float) -> ImageBuffer:
    """Radial lens-distortion correction. Purely geometric — moves pixels via a
    coordinate remap, never scales values (brightness-preserving). No-op for k1≈0."""
    if abs(k1) < _DISTORT_EPS:
        return img
    h, w = img.shape[:2]
    cx, cy, halfdiag = _radial_center(w, h)
    s = compute_distortion_scale(k1, w, h)
    return negpy_core.remap_radial(np.ascontiguousarray(img, dtype=np.float32), (cx, cy), s, k1, 1.0 / (halfdiag * halfdiag))


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
_KEYSTONE_SOLVE_MAX_ITER = 20
_KEYSTONE_SOLVE_TOL = 1e-7


def keystone_quad(converge_v: float, converge_h: float) -> np.ndarray:
    """Return the normalized source quad for the keystone correction."""
    a, b = converge_v * 0.005, converge_h * 0.005
    return np.asarray([[a, b], [1.0 - a, -b], [1.0 + a, 1.0 + b], [-a, 1.0 - b]], dtype=np.float32)


def keystone_matrix_normalized(converge_v: float, converge_h: float) -> np.ndarray:
    """Forward map (source -> corrected) in the GPU's sampling convention,
    u = (index + 0.5) / dims.

    The single definition of the quad: the pixel-space matrix and the shader's inverse
    both derive from it. Convergences are per-cent; positive converge_v stretches the
    top edge, positive converge_h the left.
    """
    src = keystone_quad(converge_v, converge_h)
    dst = np.float32([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return negpy_core.perspective_transform(src, dst).astype(np.float64)


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
    m = keystone_matrix(converge_v, converge_h, w, h)
    return negpy_core.warp_perspective(np.ascontiguousarray(img, dtype=np.float32), m, (w, h))


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


_CROP_TO_VALID_ITERS = 30
_CROP_TO_VALID_TERNARY_ITERS = 25


@lru_cache(maxsize=32)
def compute_geometry_crop_rect(
    fine_rotation: float, converge_v: float, converge_h: float, w: int, h: int
) -> Tuple[float, float, float, float]:
    """Largest axis-aligned, frame-centered rect, normalized to the frame, that avoids
    the replicated edges fine rotation and keystone (Tilt/Swing) leave behind. Distortion
    correction is excluded: compute_distortion_scale already keeps it void-free by
    scaling to fill.

    Fine rotation and keystone are each a plane projectivity, so the source rect's
    forward-mapped boundary is a convex quadrilateral: a candidate rect is valid iff its
    four corners are (convexity needs no denser a sample), and the largest-area rect at a
    fixed half-width is unimodal in that half-width. Ternary-searches the half-width, with
    a nested binary search for the largest valid half-height at each candidate — mirroring
    compute_distortion_scale's numeric style, one dimension up.
    """
    if abs(fine_rotation) < 1e-9 and abs(converge_v) < _KEYSTONE_EPS and abs(converge_h) < _KEYSTONE_EPS:
        return 0.0, 0.0, 1.0, 1.0

    cx, cy = w / 2.0, h / 2.0
    rot = negpy_core.rotation_matrix((cx, cy), fine_rotation, 1.0)
    m_rot = np.vstack([rot, [0.0, 0.0, 1.0]])
    m_combined = keystone_matrix(converge_v, converge_h, w, h) @ m_rot
    m_inv = np.linalg.inv(m_combined)

    def source_valid(x: float, y: float) -> bool:
        p = m_inv @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-12:
            return False
        sx, sy = p[0] / p[2], p[1] / p[2]
        return -1e-6 <= sx <= (w - 1) + 1e-6 and -1e-6 <= sy <= (h - 1) + 1e-6

    def corners_valid(hw: float, hh: float) -> bool:
        return all(source_valid(cx + dx, cy + dy) for dx in (-hw, hw) for dy in (-hh, hh))

    max_hw, max_hh = cx, cy

    def hh_max(hw: float) -> float:
        lo, hi = 0.0, max_hh
        for _ in range(_CROP_TO_VALID_ITERS):
            mid = 0.5 * (lo + hi)
            if corners_valid(hw, mid):
                lo = mid
            else:
                hi = mid
        return lo

    lo_w, hi_w = 0.0, max_hw
    for _ in range(_CROP_TO_VALID_TERNARY_ITERS):
        m1 = lo_w + (hi_w - lo_w) / 3.0
        m2 = hi_w - (hi_w - lo_w) / 3.0
        if m1 * hh_max(m1) < m2 * hh_max(m2):
            lo_w = m1
        else:
            hi_w = m2
    hw = 0.5 * (lo_w + hi_w)
    hh = hh_max(hw)

    return (cx - hw) / w, (cy - hh) / h, (cx + hw) / w, (cy + hh) / h


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
    # An inset larger than half the crop would turn the ROI inside out.
    margin_px = min(margin_px, (y2 - y1 - 1) / 2, (x2 - x1 - 1) / 2)
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
    return negpy_core.place_window_by_occupancy(
        int(start), int(end), float(target_len), np.ascontiguousarray(occupancy, dtype=np.float32), float(scale)
    )


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
            # Only when set: a frozen rect with no distortion keeps its key.
            *((round(geometry.distortion_k1, 4),) if geometry.distortion_k1 else ()),
            # Only when on: turning Square Before Crop on re-detects, and squares, a cropped frame.
            *(("square",) if geometry.autocrop_square else ()),
        )
    )


def meter_detection_key(geometry: GeometryConfig, shape: Tuple[int, ...]) -> str:
    """autocrop_detection_key plus the source's aspect, since a half-frame split or a stitch puts
    another picture under the same geometry."""
    h, w = shape[:2]
    return f"{autocrop_detection_key(geometry)}|{w / h:.2f}"


def resolve_meter_rect(img: ImageBuffer, geometry: GeometryConfig, preview_size: float) -> Optional[Tuple[float, float, float, float]]:
    """The picture box the meters read on a frame with no crop: the rect Auto Crop would keep, when the film
    sits in an opaque holder or on bare light. None when the film fills the frame, which is metered whole."""
    return resolve_autocrop_rect(img, geometry, preview_size, require_film_box=True)


def resolve_autocrop_rect(
    img: ImageBuffer,
    geometry: GeometryConfig,
    preview_size: float,
    require_film_box: bool = False,
) -> Optional[Tuple[float, float, float, float]]:
    """The auto crop as a normalized rect in transformed-image space; the caller freezes it, so later renders crop
    the same at any resolution. Runs at AUTOCROP_DETECT_RES in GeometryProcessor order (rot90 -> flips -> fine
    rotation -> distortion -> keystone). Excludes Crop Offset, which get_manual_rect_coords adds per render; None when
    too small, or with `require_film_box` when no film box stands out from a holder or bed."""
    tmp, _ = _normalize_detection_input(img, AUTOCROP_DETECT_RES)
    if geometry.rotation != 0:
        tmp = np.rot90(tmp, k=geometry.rotation)
    if geometry.flip_horizontal:
        tmp = np.fliplr(tmp)
    if geometry.flip_vertical:
        tmp = np.flipud(tmp)
    tmp = np.ascontiguousarray(tmp.astype(np.float32, copy=False))
    if geometry.fine_rotation != 0.0:
        tmp = apply_fine_rotation(tmp, geometry.fine_rotation)
    if geometry.distortion_k1 != 0.0:
        tmp = apply_radial_distortion(tmp, geometry.distortion_k1)
    # Detection must see the frame the render produces, or the rect is found on a
    # straight rebate and applied to a keystoned one.
    tmp = apply_keystone(tmp, geometry.converge_v, geometry.converge_h)

    rh, rw = tmp.shape[:2]
    if rh < 2 or rw < 2:
        return None
    if require_film_box and _detect_film_bounds(_normalize_detection_input(tmp, AUTOCROP_DETECT_RES)[0]) is None:
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


def _pre_keystone_point(
    nx: float,
    ny: float,
    orig_shape: Tuple[int, int],
    rotation_k: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    distortion_k1: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Map a raw-normalized point to pixel space before keystone correction."""
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
        m_mat = negpy_core.rotation_matrix(center, fine_rotation, 1.0)
        pt = np.array([px, py, 1.0])
        res_pt = m_mat @ pt
        px, py = float(res_pt[0]), float(res_pt[1])

    if distortion_k1 != 0.0:
        px, py = map_point_radial(px, py, distortion_k1, w, h)

    return px, py, w, h


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
    px, py, w, h = _pre_keystone_point(nx, ny, orig_shape, rotation_k, fine_rotation, flip_horizontal, flip_vertical, distortion_k1)

    if converge_v != 0.0 or converge_h != 0.0:
        px, py = map_point_keystone(px, py, converge_v, converge_h, w, h)

    if roi:
        y1, y2, x1, x2 = roi
        px -= x1
        py -= y1
        h, w = y2 - y1, x2 - x1

    # Not clamped: a vertex off the frame (a Card Edge drawn from outside, a large Oval)
    # must keep its place, as the overlay draws it.
    return float(px / max(w, 1)), float(py / max(h, 1))


def solve_keystone_from_edges(
    edges: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    orig_shape: Tuple[int, int],
    initial_converge_v: float = 0.0,
    initial_converge_h: float = 0.0,
    rotation_k: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    distortion_k1: float = 0.0,
) -> Tuple[float, float]:
    """Solve tilt and swing so opposite marked rebate edges become parallel."""
    required = ("left", "right", "top", "bottom")
    missing = [name for name in required if name not in edges]
    if missing:
        raise ValueError(f"solve_keystone_from_edges needs all four edges; missing {missing}")

    pre = {
        name: tuple(
            _pre_keystone_point(
                point[0],
                point[1],
                orig_shape,
                rotation_k,
                fine_rotation,
                flip_horizontal,
                flip_vertical,
                distortion_k1,
            )
            for point in edges[name]
        )
        for name in required
    }

    def slope_v(name: str, converge_v: float, converge_h: float) -> float:
        (px1, py1, width, height), (px2, py2, _, _) = pre[name]
        mapped1 = map_point_keystone(px1, py1, converge_v, converge_h, width, height)
        mapped2 = map_point_keystone(px2, py2, converge_v, converge_h, width, height)
        dy = mapped2[1] - mapped1[1]
        if abs(dy) < 1e-9:
            raise ValueError(f"the {name} edge is horizontal after correction; mark it along its length")
        return (mapped2[0] - mapped1[0]) / dy

    def slope_h(name: str, converge_v: float, converge_h: float) -> float:
        (px1, py1, width, height), (px2, py2, _, _) = pre[name]
        mapped1 = map_point_keystone(px1, py1, converge_v, converge_h, width, height)
        mapped2 = map_point_keystone(px2, py2, converge_v, converge_h, width, height)
        dx = mapped2[0] - mapped1[0]
        if abs(dx) < 1e-9:
            raise ValueError(f"the {name} edge is vertical after correction; mark it along its length")
        return (mapped2[1] - mapped1[1]) / dx

    def residuals(converge_v: float, converge_h: float) -> np.ndarray:
        return np.array(
            [
                slope_v("left", converge_v, converge_h) - slope_v("right", converge_v, converge_h),
                slope_h("top", converge_v, converge_h) - slope_h("bottom", converge_v, converge_h),
            ]
        )

    converge_v = float(np.clip(initial_converge_v, -15.0, 15.0))
    converge_h = float(np.clip(initial_converge_h, -15.0, 15.0))
    epsilon = 1e-3
    for _ in range(_KEYSTONE_SOLVE_MAX_ITER):
        residual = residuals(converge_v, converge_h)
        if np.max(np.abs(residual)) < _KEYSTONE_SOLVE_TOL:
            break
        jacobian = np.column_stack(
            (
                (residuals(converge_v + epsilon, converge_h) - residual) / epsilon,
                (residuals(converge_v, converge_h + epsilon) - residual) / epsilon,
            )
        )
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        converge_v = float(np.clip(converge_v + delta[0], -15.0, 15.0))
        converge_h = float(np.clip(converge_h + delta[1], -15.0, 15.0))

    return converge_v, converge_h


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


def rotate_geometry_and_analysis(
    geo: GeometryConfig,
    analysis_rect: Optional[Tuple[float, float, float, float]],
    direction: int,
) -> Tuple[GeometryConfig, Optional[Tuple[float, float, float, float]]]:
    """
    Applies a visual quarter-turn (the toolbar's labelled direction) to a frame's
    own geometry, crop rect and analysis rect. `direction` is +1/-1 quarter-turns
    CCW as seen on screen; a mirror inverts pipeline rotation handedness, so the
    stored `rotation` field turns the opposite way under a flip while the crop and
    analysis rects, being display-space, always turn by `direction`. Takes and
    returns the two WorkspaceConfig fields separately, not the config itself: a
    WorkspaceConfig import here would be the only one under features/, and the
    reverse import already runs domain.models -> geometry.models.
    """
    pipeline_direction = -direction if geo.flip_horizontal != geo.flip_vertical else direction
    new_geo = replace(geo, rotation=(geo.rotation + pipeline_direction) % 4)
    if geo.converge_v != 0.0 or geo.converge_h != 0.0:
        converge_v, converge_h = geo.converge_v, geo.converge_h
        for _ in range(direction % 4):
            converge_v, converge_h = 0.0 - converge_h, converge_v
        new_geo = replace(new_geo, converge_v=converge_v, converge_h=converge_h)
    if geo.crop_rect is not None:
        new_geo = replace(new_geo, crop_rect=rotate_normalized_rect(geo.crop_rect, direction))
    new_rect = rotate_normalized_rect(analysis_rect, direction) if analysis_rect is not None else None
    return new_geo, new_rect


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
        new_geo = replace(geo, flip_horizontal=not geo.flip_horizontal, converge_h=0.0 - geo.converge_h)
    else:
        new_geo = replace(geo, flip_vertical=not geo.flip_vertical, converge_v=0.0 - geo.converge_v)
    if geo.fine_rotation != 0.0:
        new_geo = replace(new_geo, fine_rotation=-geo.fine_rotation)
    if geo.crop_rect is not None:
        new_geo = replace(new_geo, crop_rect=mirror_normalized_rect(geo.crop_rect, horizontal))
    return new_geo


def flip_geometry_and_analysis(
    geo: GeometryConfig,
    analysis_rect: Optional[Tuple[float, float, float, float]],
    horizontal: bool,
) -> Tuple[GeometryConfig, Optional[Tuple[float, float, float, float]]]:
    """
    Mirrors a frame's own geometry and analysis rect across its vertical
    (horizontal=True) or horizontal axis. The freehand analysis region is
    transformed-space like the crop rect toggle_flip already mirrors, so it keeps
    reading the same picture content. See rotate_geometry_and_analysis for why
    this takes the two fields separately rather than a WorkspaceConfig.
    """
    new_geo = toggle_flip(geo, horizontal)
    new_rect = mirror_normalized_rect(analysis_rect, horizontal) if analysis_rect is not None else None
    return new_geo, new_rect


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
