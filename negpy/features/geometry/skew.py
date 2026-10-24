"""Frame squaring: rotation and easel tilt/swing measured from the film and gate edges, never the picture."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import negpy_core
import numpy as np

from negpy.domain.interfaces import PipelineContext
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
from negpy.features.geometry.models import FINE_ROTATION_LIMIT, GeometryConfig
from negpy.features.geometry.processor import GeometryProcessor
from negpy.features.process.models import ProcessMode

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
# A line that disagrees with the rest by more than this after the solve is not the frame.
_MAX_LINE_RESIDUAL = 0.15
# Opposite edges this close together cannot separate keystone from noise.
_MIN_PAIR_SEPARATION = 0.3
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
    return negpy_core.coarse_angles(np.ascontiguousarray(lum, dtype=np.float32), _MIN_SEGMENT, _MAX_SKEW_DEGREES, _ANGLE_BIN)


def _fit_side(lum: np.ndarray, horizontal: bool, at: float, start: float, stop: float, source: str, outward: int = 0) -> EdgeLine | None:
    """Fit the edge running along one axis near `at`, between `start` and `stop`. `outward` is
    the sign of the box's outside across the axis, or 0 when `at` is the edge itself; negpy_core
    holds the sampling and the fit."""
    found = negpy_core.skew_fit_side(np.ascontiguousarray(lum, dtype=np.float32), horizontal, at, start, stop, outward)
    return None if found is None else EdgeLine(found[0], found[1], found[2], source, found[3], found[4])


def _off_border(lo: float, hi: float, extent: int) -> bool:
    """Whether the sides at `lo` and `hi` both sit inside the canvas, off its border."""
    return lo >= 2 and hi <= extent - 3


def _unrotate(line: EdgeLine, angle: float, w: int, h: int) -> EdgeLine:
    """Map a line found on the image turned by `angle` back to the unturned image."""
    m = np.vstack([negpy_core.rotation_matrix((w / 2.0, h / 2.0), angle, 1.0), [0.0, 0.0, 1.0]])
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
    # A slide shows no film edge to the detector (black rebate on a dark bed), but its picture
    # box is still found from the canvas.
    outer = film.roi if film.roi is not None else (0, rotated.shape[0], 0, rotated.shape[1])
    gate, _, _ = _refine_roi_to_image(rotated, outer)
    if gate != outer:
        boxes.append(("gate", gate))
    h, w = lum.shape
    lines: list[EdgeLine] = []
    seen: list[tuple[bool, float]] = []
    for source, (y1, y2, x1, x2) in boxes:
        # Film edges come in pairs. A side on the canvas border is a strip that runs past the
        # capture, and the holder or the next frame bounds the other side of that axis.
        rows = source != "film" or _off_border(y1, y2, h)
        cols = source != "film" or _off_border(x1, x2, w)
        for horizontal, at, start, stop, side, outward, fit in (
            (True, y1, x1, x2, "top", -1, rows),
            (True, y2, x1, x2, "bottom", 1, rows),
            (False, x1, y1, y2, "left", -1, cols),
            (False, x2, y1, y2, "right", 1, cols),
        ):
            if not fit or any(o == horizontal and abs(p - at) < 4 for o, p in seen):
                continue
            line = _fit_side(lum, horizontal, float(at), float(start), float(stop), f"{source}-{side}", outward)
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


def _profile_lines(lum: np.ndarray) -> list[EdgeLine]:
    """Outermost long edges by row and column coverage, for frames the film detector cannot box."""
    found = negpy_core.skew_profile_lines(np.ascontiguousarray(lum, dtype=np.float32))
    return [
        EdgeLine(p1, p2, horizontal, f"profile-{'horizontal' if horizontal else 'vertical'}", inliers, rms)
        for p1, p2, horizontal, inliers, rms in found
    ]


def _map(points: np.ndarray, angle: float, cv: float, ch: float, w: int, h: int) -> np.ndarray:
    """Where input points land after fine rotation then keystone, as the pipeline applies them."""
    m = negpy_core.rotation_matrix((w / 2.0, h / 2.0), angle, 1.0)
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
    placed = [(line.horizontal, _line_position(line)) for line in lines]
    return negpy_core.skew_at_canvas_edges(np.ascontiguousarray(lum, dtype=np.float32), placed)


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


def squared_geometry(source: ImageBuffer, geometry: GeometryConfig, process_mode: ProcessMode) -> tuple[GeometryConfig, FrameSkew] | None:
    """*geometry* with Fine Rotation, and Tilt and Swing where measured, read off the frame's own
    edges in *source* (the flat-fielded linear frame before geometry), and the fit itself; None
    when the edges are not trusted. The fit runs on the frame with its skew zeroed, so the
    values are absolute."""
    small, _ = _normalize_detection_input(source, AUTOCROP_DETECT_RES)
    base = replace(geometry, fine_rotation=0.0, converge_v=0.0, converge_h=0.0, crop_rect=None, crop_from_auto=False, autocrop_offset=0)
    context = PipelineContext(original_size=(small.shape[1], small.shape[0]), scale_factor=1.0, process_mode=process_mode)
    skew = trusted_frame_skew(GeometryProcessor(base).process(small, context))
    if skew is None:
        return None
    squared = replace(
        geometry,
        fine_rotation=float(np.clip(skew.fine_rotation, -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT)),
        converge_v=geometry.converge_v if skew.converge_v is None else float(skew.converge_v),
        converge_h=geometry.converge_h if skew.converge_h is None else float(skew.converge_h),
    )
    return squared, skew
