"""Frame squaring: rotation and easel tilt/swing measured from the film and gate edges, never the picture."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.geometry.logic import (
    AUTOCROP_DETECT_RES,
    _detection_luma,
    _normalize_detection_input,
    _refine_roi_to_image,
    apply_fine_rotation,
    detect_film_bounds_with_confidence,
    keystone_matrix,
)
from negpy.features.geometry.models import FINE_ROTATION_LIMIT

# Holder skew beyond this is a frame put in sideways or a picture edge, not a tilt.
_MAX_SKEW_DEGREES = 10.0
# No copy stand is this far off square; past it the fit has read a bent or wrong edge.
_MAX_KEYSTONE = 5.0
# Below this the fit reads edge roughness, curl or lens residue.
_MIN_KEYSTONE = 0.3
# Keystone must explain the edges this much better than rotation alone to be kept.
_KEYSTONE_GAIN = 2.5
_KEYSTONE_MIN_RESIDUAL = 0.04
_ANGLE_BIN = 0.05
# Sprocket holes, edge print and most picture detail run shorter than this.
_MIN_SEGMENT = 0.04
_MIN_LINE_SPAN = 0.25
_MAX_LINE_RMS = 1.2
_MIN_INLIER_FRACTION = 0.5
# A line that disagrees with the rest by more than this after the solve is not the frame.
_MAX_LINE_RESIDUAL = 0.15
# Opposite edges this close together cannot separate keystone from noise.
_MIN_PAIR_SEPARATION = 0.3
# A tight scan's border lines sit within this fraction of the cross extent from the canvas edge.
_CANVAS_EDGE_BAND = 0.12
# Film base or bed outside a border never reads this dark; an opaque holder does.
_OPAQUE_RING = 0.04
# Must admit a clean two-line pair: a strip across the canvas shows two film edges, not four.
_SKEW_TRUSTED_CONFIDENCE = 0.5


@dataclass(frozen=True)
class EdgeLine:
    """A fitted frame edge, in pixels of the detection buffer (long edge at most AUTOCROP_DETECT_RES)."""

    p1: tuple[float, float]
    p2: tuple[float, float]
    horizontal: bool
    source: str
    inlier_fraction: float
    rms: float

    @property
    def length(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])


@dataclass(frozen=True)
class FrameSkew:
    """`fine_rotation` adds to the rotation already in the measured image; `converge_v` and
    `converge_h` are None where no edge pair measured them."""

    fine_rotation: float
    converge_v: float | None
    converge_h: float | None
    confidence: float
    lines: tuple[EdgeLine, ...]
    residual: float
    # Trust gates; see trusted_frame_skew.
    from_film_box: bool = False
    has_opposite_pair: bool = False
    at_canvas_edges: bool = False


def _coarse_angles(lum: np.ndarray) -> list[float]:
    """Dominant folded angles of long segments, strongest first. The runner-up is a second
    start: picture lines at their own tilt can outvote the frame."""
    h, w = lum.shape
    lo, hi = np.percentile(lum, (1.0, 99.0))
    u8 = (np.clip((lum - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
    segments = cv2.createLineSegmentDetector().detect(u8)[0]
    if segments is None:
        return []
    seg = segments.reshape(-1, 4).astype(np.float64)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx))
    angle = (angle + 45.0) % 90.0 - 45.0
    keep = (length >= _MIN_SEGMENT * max(h, w)) & (np.abs(angle) <= _MAX_SKEW_DEGREES)
    if not np.any(keep):
        return []
    bins = np.arange(-_MAX_SKEW_DEGREES, _MAX_SKEW_DEGREES + _ANGLE_BIN, _ANGLE_BIN)
    hist, _ = np.histogram(angle[keep], bins=bins, weights=length[keep] ** 2)
    hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), 3.0).ravel()
    first = int(np.argmax(hist))
    modes = [float(bins[first] + _ANGLE_BIN / 2)]
    rest = hist.copy()
    rest[max(0, first - 10) : first + 11] = 0.0
    second = int(np.argmax(rest))
    if rest[second] >= 0.25 * hist[first]:
        modes.append(float(bins[second] + _ANGLE_BIN / 2))
    return modes


def _profile_peaks(profile: np.ndarray) -> np.ndarray:
    """Subpixel positions of the strong gradient steps along one profile."""
    if profile.size < 5:
        return np.empty(0)
    grad = np.abs(np.gradient(cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float32), (1, 0), 1.0).ravel()))
    peak = float(grad.max())
    if peak <= 3.0 * float(np.median(grad)) + 1e-6:
        return np.empty(0)
    i = np.arange(1, grad.size - 1)
    i = i[(grad[i] >= grad[i - 1]) & (grad[i] > grad[i + 1]) & (grad[i] >= 0.4 * peak)]
    a, b, c = grad[i - 1], grad[i], grad[i + 1]
    denom = a - 2 * b + c
    safe = np.where(np.abs(denom) > 1e-9, denom, 1.0)
    return i + np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / safe, 0.0)


def _ransac_line(xs: np.ndarray, ys: np.ndarray, expected: float, half: float, span: float) -> np.ndarray | None:
    """The straight run of peaks with the most distinct stations, preferring one near `expected`."""
    if xs.size < 12:
        return None
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, xs.size, (2, 300))
    ok = np.abs(xs[a] - xs[b]) >= 0.3 * span
    a, b = a[ok], b[ok]
    if a.size == 0:
        return None
    slope = (ys[b] - ys[a]) / (xs[b] - xs[a])
    ok = np.abs(slope) <= math.tan(math.radians(4.0))
    a, b, slope = a[ok], b[ok], slope[ok]
    if a.size == 0:
        return None
    intercept = ys[a] - slope * xs[a]
    res = np.abs(ys[None, :] - (slope[:, None] * xs[None, :] + intercept[:, None]))
    inlier = res <= 1.5
    stations = np.array([np.unique(xs[row]).size for row in inlier])
    mid = slope * (0.5 * (xs.min() + xs.max())) + intercept
    score = stations * (1.0 - 0.5 * np.minimum(1.0, np.abs(mid - expected) / half))
    best = int(np.argmax(score))
    if stations[best] < 12:
        return None
    return np.polyfit(xs[inlier[best]], ys[inlier[best]], 1)


def _fit_side(lum: np.ndarray, horizontal: bool, at: float, start: float, stop: float, source: str) -> EdgeLine | None:
    """Fit the edge running along one axis near `at`, between `start` and `stop`."""
    img = lum if horizontal else lum.T
    h, w = img.shape
    length = stop - start
    if length < _MIN_LINE_SPAN * w or at < 2 or at > h - 3:
        return None
    stations = np.linspace(start + 0.1 * length, stop - 0.1 * length, int(min(240, max(24, length / 4))))

    def sample(center: np.ndarray, half: float, nearest: bool) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for x, c in zip(stations, center):
            x0 = int(round(x))
            lo, hi = int(max(0, math.floor(c - half))), int(min(h, math.ceil(c + half) + 1))
            if hi - lo < 5 or x0 < 1 or x0 > w - 2:
                continue
            peaks = _profile_peaks(img[lo:hi, x0 - 1 : x0 + 2].mean(axis=1))
            if peaks.size == 0:
                continue
            if nearest:
                peaks = peaks[[int(np.argmin(np.abs(peaks - (c - lo))))]]
            xs.extend([x] * peaks.size)
            ys.extend(lo + peaks)
        return np.asarray(xs), np.asarray(ys)

    # Wide enough for an edge that keystone leaves a degree or two off the box.
    wide = max(10.0, 0.035 * max(h, w))
    xs, ys = sample(np.full(stations.size, float(at)), wide, nearest=False)
    fit = _ransac_line(xs, ys, float(at), wide, length)
    if fit is None:
        return None
    xs, ys = sample(np.polyval(fit, stations), max(6.0, 0.3 * wide), nearest=True)
    if xs.size < 12:
        return None
    keep = np.ones(xs.size, bool)
    for _ in range(8):
        fit = np.polyfit(xs[keep], ys[keep], 1)
        res = ys - np.polyval(fit, xs)
        mad = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep]))))
        new = np.abs(res) <= max(1.0, 2.5 * mad)
        if new.sum() < 12 or np.array_equal(new, keep):
            break
        keep = new
    inliers = float(keep.sum()) / stations.size
    rms = float(np.sqrt(np.mean((ys[keep] - np.polyval(fit, xs[keep])) ** 2)))
    if inliers < _MIN_INLIER_FRACTION or rms > _MAX_LINE_RMS:
        return None
    x1, x2 = float(xs[keep].min()), float(xs[keep].max())
    if x2 - x1 < _MIN_LINE_SPAN * w:
        return None
    y1, y2 = float(np.polyval(fit, x1)), float(np.polyval(fit, x2))
    p1, p2 = ((x1, y1), (x2, y2)) if horizontal else ((y1, x1), (y2, x2))
    return EdgeLine(p1, p2, horizontal, source, inliers, rms)


def _unrotate(line: EdgeLine, angle: float, w: int, h: int) -> EdgeLine:
    """Map a line found on the image turned by `angle` back to the unturned image."""
    m = np.vstack([cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0), [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(m)
    p1 = inv @ [line.p1[0], line.p1[1], 1.0]
    p2 = inv @ [line.p2[0], line.p2[1], 1.0]
    return EdgeLine(
        (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), line.horizontal, line.source, line.inlier_fraction, line.rms
    )


def _frame_lines(lum: np.ndarray, rotated: np.ndarray) -> list[EdgeLine]:
    """Film-edge and gate-edge lines on an image already turned near square."""
    film = detect_film_bounds_with_confidence(rotated)
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    if film.roi is not None:
        boxes.append(("film", film.roi))
        gate, _, _ = _refine_roi_to_image(rotated, film.roi)
        if gate != film.roi:
            boxes.append(("gate", gate))
    lines: list[EdgeLine] = []
    seen: list[tuple[bool, float]] = []
    for source, (y1, y2, x1, x2) in boxes:
        for horizontal, at, start, stop, side in (
            (True, y1, x1, x2, "top"),
            (True, y2, x1, x2, "bottom"),
            (False, x1, y1, y2, "left"),
            (False, x2, y1, y2, "right"),
        ):
            if any(o == horizontal and abs(p - at) < 4 for o, p in seen):
                continue
            line = _fit_side(lum, horizontal, float(at), float(start), float(stop), f"{source}-{side}")
            if line is None:
                continue
            # A film and a gate side can land on the same edge; count it once.
            placed = _line_position(line)
            if any(o == horizontal and abs(p - placed) < 4 for o, p in seen):
                continue
            lines.append(line)
            seen.append((horizontal, placed))
    return lines


def _line_position(line: EdgeLine) -> float:
    idx = 1 if line.horizontal else 0
    return 0.5 * (line.p1[idx] + line.p2[idx])


def _outer_peaks(coverage: np.ndarray, floor: float) -> tuple[int, int] | None:
    """First and last local maxima of `coverage` at or above `floor`, off the canvas border."""
    n = coverage.size
    i = np.arange(3, n - 3)
    peaks = i[(coverage[i] >= floor) & (coverage[i] >= coverage[i - 1]) & (coverage[i] >= coverage[i + 1])]
    if peaks.size < 2 or peaks[-1] - peaks[0] < _MIN_PAIR_SEPARATION * n:
        return None
    return int(peaks[0]), int(peaks[-1])


def _profile_lines(lum: np.ndarray) -> list[EdgeLine]:
    """Outermost long edges by row and column coverage, for frames the film detector cannot box.
    The spread keeps an edge a degree off axis inside one band."""
    h, w = lum.shape
    lines: list[EdgeLine] = []
    spread = max(3, int(0.02 * max(h, w)))
    gy = np.abs(cv2.Sobel(lum.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    strong = (gy >= 0.25 * float(np.percentile(gy, 99.5))).astype(np.uint8)
    strong = cv2.dilate(strong, np.ones((2 * spread + 1, 1), np.uint8))
    rows = _outer_peaks(strong[:, int(0.05 * w) : int(0.95 * w)].mean(axis=1), 0.35)
    top, bottom = rows if rows is not None else (0, h)
    if rows is not None:
        for y in rows:
            line = _fit_side(lum, True, float(y), 0.0, float(w), "profile-horizontal")
            if line is not None:
                lines.append(line)

    gx = np.abs(cv2.Sobel(lum.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    strong = (gx >= 0.25 * float(np.percentile(gx, 99.5))).astype(np.uint8)
    strong = cv2.dilate(strong, np.ones((1, 2 * spread + 1), np.uint8))
    lo, hi = top + int(0.05 * (bottom - top)), bottom - int(0.05 * (bottom - top))
    cols = _outer_peaks(strong[lo:hi].mean(axis=0), 0.35) if hi - lo > 10 else None
    if cols is not None:
        for x in cols:
            line = _fit_side(lum, False, float(x), float(top), float(bottom), "profile-vertical")
            if line is not None:
                lines.append(line)
    # A frame shows its edges in opposite pairs; a lone long line is picture content.
    return [line for line in lines if sum(other.horizontal == line.horizontal for other in lines) >= 2]


def _map(points: np.ndarray, angle: float, cv: float, ch: float, w: int, h: int) -> np.ndarray:
    """Where input points land after fine rotation then keystone, as the pipeline applies them."""
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = points @ m[:, :2].T + m[:, 2]
    if abs(cv) < 1e-6 and abs(ch) < 1e-6:
        return out
    k = keystone_matrix(cv, ch, w, h)
    hom = np.column_stack([out, np.ones(len(out))]) @ k.T
    return hom[:, :2] / hom[:, 2:3]


def _line_angles(lines: list[EdgeLine], params: np.ndarray, w: int, h: int) -> np.ndarray:
    """Each line's angle off its axis after (fine rotation, converge_v, converge_h) = `params`."""
    pts = np.array([p for line in lines for p in (line.p1, line.p2)], np.float64)
    mapped = _map(pts, float(params[0]), float(params[1]), float(params[2]), w, h).reshape(-1, 2, 2)
    d = mapped[:, 1] - mapped[:, 0]
    horizontal = np.array([line.horizontal for line in lines])
    return np.degrees(np.where(horizontal, np.arctan2(d[:, 1], d[:, 0]), -np.arctan2(d[:, 0], d[:, 1])))


def _solve(lines: list[EdgeLine], w: int, h: int, fit_v: bool, fit_h: bool) -> tuple[float, float, float, np.ndarray]:
    """Least-squares rotation and keystone that level every line; Gauss-Newton."""
    weights = np.sqrt(np.array([line.length * line.inlier_fraction for line in lines]))
    params = np.zeros(3)
    free = [0] + ([1] if fit_v else []) + ([2] if fit_h else [])
    for _ in range(12):
        r = _line_angles(lines, params, w, h) * weights
        jac = np.empty((len(lines), len(free)))
        for j, idx in enumerate(free):
            step = np.zeros(3)
            step[idx] = 1e-3
            jac[:, j] = (_line_angles(lines, params + step, w, h) * weights - r) / 1e-3
        delta, *_ = np.linalg.lstsq(jac, -r, rcond=None)
        params[free] += delta
        params[0] = float(np.clip(params[0], -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT))
        params[1:] = np.clip(params[1:], -2 * _MAX_KEYSTONE, 2 * _MAX_KEYSTONE)
        if np.max(np.abs(delta)) < 1e-6:
            break
    return float(params[0]), float(params[1]), float(params[2]), _line_angles(lines, params, w, h)


def _at_canvas_edges(lines: list[EdgeLine], lum: np.ndarray) -> bool:
    """Whether every line hugs the canvas edge, paired on both axes, with no opaque holder outside."""
    if not lines:
        return False
    h, w = lum.shape
    for line in lines:
        extent = h if line.horizontal else w
        position = _line_position(line)
        if min(position, extent - position) > _CANVAS_EDGE_BAND * extent:
            return False
        index = int(position)
        outside = (
            (lum[: max(0, index - 2)] if position < extent / 2 else lum[index + 3 :])
            if line.horizontal
            else (lum[:, : max(0, index - 2)] if position < extent / 2 else lum[:, index + 3 :])
        )
        if outside.size and float(np.median(outside)) < _OPAQUE_RING:
            return False
    return _pair_spread(lines, True, h) >= _MIN_PAIR_SEPARATION and _pair_spread(lines, False, w) >= _MIN_PAIR_SEPARATION


def _pair_spread(lines: list[EdgeLine], horizontal: bool, extent: int) -> float:
    """How far apart the outermost two lines of one axis sit, as a fraction of `extent`."""
    pos = [_line_position(line) for line in lines if line.horizontal == horizontal]
    return (max(pos) - min(pos)) / extent if len(pos) >= 2 else 0.0


def _measure_from(det: np.ndarray, coarse: float, fit_keystone: bool) -> tuple[FrameSkew | None, bool]:
    """The fit starting from the image turned by `coarse`, and whether a film box led it."""
    h, w = det.shape[:2]
    rotated = apply_fine_rotation(det, coarse) if abs(coarse) > 1e-4 else det
    rot_lum = _detection_luma(rotated)
    found = _frame_lines(rot_lum, rotated)
    boxed = bool(found)
    if not found:
        found = _profile_lines(rot_lum)
    lines = [_unrotate(line, coarse, w, h) for line in found]
    if not lines:
        return None, False

    while True:
        fit_v = fit_keystone and _pair_spread(lines, False, w) >= _MIN_PAIR_SEPARATION
        fit_h = fit_keystone and _pair_spread(lines, True, h) >= _MIN_PAIR_SEPARATION
        angle, cv, ch, residual = _solve(lines, w, h, fit_v, fit_h)
        worst = int(np.argmax(np.abs(residual)))
        if len(lines) <= 2 or abs(residual[worst]) <= _MAX_LINE_RESIDUAL:
            break
        lines = lines[:worst] + lines[worst + 1 :]

    # Every gate re-runs after each re-solve: dropping one axis moves the survivor.
    # The solver clips past _MAX_KEYSTONE, so an out-of-range solution is rejected here, not clamped.
    if fit_v or fit_h:
        _, _, _, rotation_only = _solve(lines, w, h, False, False)
        rot_rms = float(np.sqrt(np.mean(rotation_only**2)))
        while fit_v or fit_h:
            full_rms = float(np.sqrt(np.mean(residual**2)))
            if rot_rms < max(_KEYSTONE_MIN_RESIDUAL, _KEYSTONE_GAIN * full_rms):
                fit_v = fit_h = False
            else:
                keep_v = fit_v and _MIN_KEYSTONE <= abs(cv) <= _MAX_KEYSTONE
                keep_h = fit_h and _MIN_KEYSTONE <= abs(ch) <= _MAX_KEYSTONE
                if keep_v == fit_v and keep_h == fit_h:
                    break
                fit_v, fit_h = keep_v, keep_h
            angle, cv, ch, residual = _solve(lines, w, h, fit_v, fit_h)

    if abs(angle) > _MAX_SKEW_DEGREES:
        return None, False
    rms = float(np.sqrt(np.mean(residual**2)))
    support = min(1.0, sum(line.length for line in lines) / (2.0 * (w + h)))
    agreement = float(np.clip(1.0 - rms / _MAX_LINE_RESIDUAL, 0.0, 1.0))
    count = min(1.0, len(lines) / 4.0)
    confidence = float(np.clip(0.45 * support + 0.35 * agreement + 0.20 * count, 0.0, 1.0))
    skew = FrameSkew(
        fine_rotation=angle,
        converge_v=cv if fit_v else None,
        converge_h=ch if fit_h else None,
        confidence=confidence,
        lines=tuple(lines),
        residual=rms,
        from_film_box=boxed,
        has_opposite_pair=_pair_spread(lines, True, h) >= _MIN_PAIR_SEPARATION or _pair_spread(lines, False, w) >= _MIN_PAIR_SEPARATION,
        at_canvas_edges=not boxed and _at_canvas_edges(lines, _detection_luma(det)),
    )
    return skew, boxed


def measure_frame_skew(img: ImageBuffer, fit_keystone: bool = True) -> FrameSkew | None:
    """Rotation and keystone that square the frame in `img`, or None when no frame edge is found.
    `img` is the pipeline buffer before fine rotation and keystone. The rotation adds to any
    already in `img`; the keystone is absolute only when `img` carries none."""
    det, _ = _normalize_detection_input(img, AUTOCROP_DETECT_RES)
    det = np.ascontiguousarray(det, dtype=np.float32)
    modes = _coarse_angles(_detection_luma(det))
    starts: list[float] = []
    for angle in modes[:1] + [0.0] + modes[1:]:
        if all(abs(angle - other) > 0.5 for other in starts):
            starts.append(angle)

    # The fitted edges set the angle; a start only has to be near enough to find the box.
    best: tuple[bool, float] | None = None
    result = None
    for start in starts:
        skew, boxed = _measure_from(det, start, fit_keystone)
        if skew is None:
            continue
        rank = (boxed, skew.confidence)
        if best is None or rank > best:
            best, result = rank, skew
        if boxed and skew.confidence >= 0.8:
            break
    return result


def trusted_frame_skew(img: ImageBuffer, fit_keystone: bool = True) -> FrameSkew | None:
    """measure_frame_skew, or None when the fit may not be applied unwatched. The gates are
    structural: a profile-led fit or a lone edge can fit picture content perfectly. Lines that
    frame the canvas on both axes pass without a film box: they are a tight scan's border."""
    skew = measure_frame_skew(img, fit_keystone)
    if skew is None or not skew.has_opposite_pair:
        return None
    if not skew.from_film_box and not skew.at_canvas_edges:
        return None
    return skew if skew.confidence >= _SKEW_TRUSTED_CONFIDENCE else None
