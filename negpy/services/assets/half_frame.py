"""Half-frame scans: one file holds two frames side by side.

A half asset is a normal asset dict plus ``half`` (1 = left, 2 = right) and
``split_x`` (normalized gutter position). Its identity is the file hash
suffixed with ``#<half>``, so every hash-keyed store (edits, history, marks,
thumbnails) is per-frame automatically. Decode caches key on the unsuffixed
hash so both halves share one decode.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from negpy.kernel.system.logging import get_logger

if TYPE_CHECKING:
    from negpy.domain.models import WorkspaceConfig

logger = get_logger(__name__)

_SEP = "#"

SPLIT_SCANS_KEY = "half_frame_scans"

# The inter-frame gap in a joined diptych, in output space. Black is what the gap between
# two exposures looks like once rendered; swap for the finish border colour if wanted.
_GAP_FILL = 0.0

_MAX_GUTTER_THICKNESS = 0.1  # matches the split editor's Cut thickness slider range


def is_composite(file_info: Dict[str, Any]) -> bool:
    """Whether an asset is assembled from more than one file (triplet, stitch, HDR).

    Half frame does not apply to these: a composite carries its primary's plain hash, so
    without this test a stale ``#1``/``#2`` edit left on that primary by an earlier
    half-frame session makes the assembled frame render as a diptych. Single source of
    truth for the rule — discovery refuses to split them and the diptych readers refuse
    to claim them, so the two cannot drift.
    """
    return bool(file_info.get("green_path") or file_info.get("stitch_paths") or file_info.get("hdr_paths"))


def split_scans(repo: Any) -> set:
    """Base hashes the user split with Half Frame on — the scans allowed to be diptychs.

    A saved ``#1``/``#2`` edit is not on its own evidence that a scan is half-frame: the row
    is keyed by content hash, so it outlives the folder, the session and the mode being on.
    Only a scan the user actually split may come back as a diptych.
    """
    return set(repo.get_global_setting(SPLIT_SCANS_KEY, default=None) or ())


def remember_split_scans(repo: Any, hashes: Any) -> None:
    """Add base hashes to the split-scan set; no write when they are all known already."""
    known = split_scans(repo)
    new = known | {h for h in hashes if h}
    if new != known:
        repo.save_global_setting(SPLIT_SCANS_KEY, sorted(new))


def forget_split_scan(repo: Any, file_hash: Optional[str]) -> None:
    """Drop a base hash from the split-scan set, so the scan is no longer a diptych."""
    known = split_scans(repo)
    if file_hash in known:
        repo.save_global_setting(SPLIT_SCANS_KEY, sorted(known - {file_hash}))


def half_hash(file_hash: str, half: int) -> str:
    return f"{file_hash}{_SEP}{half}"


def half_of(file_hash: Optional[str]) -> Optional[int]:
    """The half index of a ``#``-suffixed hash, or None when it is not a half.

    Composite hashes (``#stitch``, ``#hdr``) share the separator by design, so the
    suffix — not its presence — decides.
    """
    _, sep, suffix = (file_hash or "").rpartition(_SEP)
    return int(suffix) if sep and suffix.isdigit() else None


def base_hash(file_hash: Optional[str]) -> Optional[str]:
    """The unsuffixed file hash — the decode-cache identity shared by both halves."""
    return file_hash.split(_SEP, 1)[0] if file_hash else file_hash


def half_name(name: str, half: int) -> str:
    return f"{name} [{half}]"


def saved_crop_rect(value: Any) -> Optional[tuple[float, float, float, float]]:
    """A persisted half-frame ``crop_rect`` as four floats, or None when absent or malformed.

    Settings are stored as JSON with ``default=str``, so a crop saved from numpy values
    reads back as strings.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2)


def _slice_half_bounds(
    height: int,
    width: int,
    half: int,
    split_x: float,
    crop_rect: Optional[tuple[float, float, float, float]] = None,
    gutter_thickness: float = 0.0,
) -> tuple[int, int, int, int]:
    """Pixel bounds read by ``slice_half``."""
    x1, y1, x2, y2 = 0, 0, width, height
    if crop_rect is not None:
        rx1, ry1, rx2, ry2 = crop_rect
        cx1 = min(max(int(round(width * rx1)), 0), width)
        cx2 = min(max(int(round(width * rx2)), 0), width)
        cy1 = min(max(int(round(height * ry1)), 0), height)
        cy2 = min(max(int(round(height * ry2)), 0), height)
        if cx2 <= cx1 or cy2 <= cy1:
            return (0, height, 0, min(1, width))
        x1, y1, x2, y2 = cx1, cy1, cx2, cy2
    if not half:
        return (y1, y2, x1, x2)

    cropped_width = x2 - x1
    split = min(max(int(round(cropped_width * split_x)), 1), cropped_width - 1)
    if gutter_thickness > 0:
        gutter = max(1, int(round(cropped_width * gutter_thickness)))
        lo = max(1, split - gutter // 2)
        hi = min(cropped_width - 1, lo + gutter)
        lo = max(1, hi - gutter)
        return (y1, y2, x1, x1 + lo) if half == 1 else (y1, y2, x1 + hi, x2)
    return (y1, y2, x1, x1 + split) if half == 1 else (y1, y2, x1 + split, x2)


def slice_half_dimensions(
    dimensions: tuple[int, int],
    half: int,
    split_x: float,
    crop_rect: Optional[tuple[float, float, float, float]] = None,
    gutter_thickness: float = 0.0,
) -> tuple[int, int]:
    """Return the full-resolution image-space dimensions of a half slice."""
    height, width = dimensions
    y1, y2, x1, x2 = _slice_half_bounds(height, width, half, split_x, crop_rect, gutter_thickness)
    return (y2 - y1, x2 - x1)


def slice_half(
    buf: np.ndarray,
    half: int,
    split_x: float,
    crop_rect: Optional[tuple[float, float, float, float]] = None,
    gutter_thickness: float = 0.0,
) -> np.ndarray:
    """View of one half of a decoded buffer; ``half=0`` crops only, without splitting.

    The scan is first cropped to ``crop_rect`` (normalized x1,y1,x2,y2; None =
    full frame), then split at the normalized gutter ``split_x`` relative to the
    cropped width. A ``gutter_thickness`` (normalized fraction of the cropped
    width) discards a band centered on the split so the physical black separator
    between the two exposures does not bleed into either half.
    """
    h, w = buf.shape[:2]
    y1, y2, x1, x2 = _slice_half_bounds(h, w, half, split_x, crop_rect, gutter_thickness)
    return buf[y1:y2, x1:x2]


def slice_for_asset(buf: np.ndarray, file_info: Dict[str, Any]) -> np.ndarray:
    """Apply the asset's half slice; no-op for whole-frame assets without a crop rect.

    Recognizes an optional ``crop_rect`` (normalized x1,y1,x2,y2) and
    ``gutter_thickness`` (normalized fraction of the cropped width) set by the
    half-frame rectangle editor. A whole-frame asset that carries a crop rect is a
    diptych: cropped to the rect, still whole, split later per half.
    """
    half = int(file_info.get("half") or 0)
    if not half and not file_info.get("crop_rect"):
        return buf
    raw_rect = file_info.get("crop_rect")
    crop_rect = tuple(float(v) for v in raw_rect) if isinstance(raw_rect, (tuple, list)) else None
    return slice_half(
        buf,
        half,
        float(file_info.get("split_x") or 0.5),
        crop_rect=crop_rect,
        gutter_thickness=float(file_info.get("gutter_thickness") or 0.0),
    )


@dataclass(frozen=True)
class HalfGeometry:
    """A half-frame asset's crop/split, as ``slice_half`` reads it. What a manual
    annotation's normalized coordinates are relative to — two of these (old, new)
    are enough to re-anchor a point to the same physical film location across a
    split/crop change."""

    crop_rect: Optional[tuple[float, float, float, float]] = None
    split_x: float = 0.5
    gutter_thickness: float = 0.0


def _to_scan(x: float, y: float, half: int, geom: HalfGeometry) -> tuple[float, float]:
    """Half-local normalized point -> full-scan normalized point.

    The continuous counterpart of ``slice_half``'s pixel-rounded crop and split:
    close enough to re-anchor a stroke, since the rounding is sub-pixel.
    """
    x1, y1, x2, y2 = geom.crop_rect if geom.crop_rect is not None else (0.0, 0.0, 1.0, 1.0)
    cw, ch = max(1e-9, x2 - x1), max(1e-9, y2 - y1)
    if not half:
        cx, cy = x, y
    else:
        lo = max(0.0, geom.split_x - geom.gutter_thickness / 2.0)
        hi = min(1.0, geom.split_x + geom.gutter_thickness / 2.0)
        cx, cy = (x * lo, y) if half == 1 else (hi + x * (1.0 - hi), y)
    return x1 + cx * cw, y1 + cy * ch


def _from_scan(fx: float, fy: float, half: int, geom: HalfGeometry) -> tuple[float, float]:
    """Full-scan normalized point -> half-local normalized point, the inverse of ``_to_scan``."""
    x1, y1, x2, y2 = geom.crop_rect if geom.crop_rect is not None else (0.0, 0.0, 1.0, 1.0)
    cw, ch = max(1e-9, x2 - x1), max(1e-9, y2 - y1)
    cx, cy = (fx - x1) / cw, (fy - y1) / ch
    if not half:
        return cx, cy
    lo = max(0.0, geom.split_x - geom.gutter_thickness / 2.0)
    hi = min(1.0, geom.split_x + geom.gutter_thickness / 2.0)
    x = cx / max(1e-9, lo) if half == 1 else (cx - hi) / max(1e-9, 1.0 - hi)
    return x, cy


def remap_point(x: float, y: float, half: int, old_geom: HalfGeometry, new_geom: HalfGeometry) -> tuple[float, float]:
    """Re-anchor a half-local normalized point to the same physical film location
    under a changed crop/split, so a manual edit stays put when the frame is later
    recropped or resplit."""
    return _from_scan(*_to_scan(x, y, half, old_geom), half, new_geom)


def remap_workspace_config(config: "WorkspaceConfig", half: int, old_geom: HalfGeometry, new_geom: HalfGeometry) -> "WorkspaceConfig":
    """Re-anchor every position-based manual edit in a half's saved config, so a
    heal stroke, dust spot, scratch line or dodge/burn mask stays on the same
    physical film location after that half's crop/split changes.

    ``geometry.crop_rect`` is cleared instead: unlike these, it lives in
    transformed-image space (after rotation/flip/keystone/distortion), not raw
    space, so the same point-remap does not apply to it, and a rect drawn
    against the old half's frame boundary has no correct position in the new
    one. Clearing (not remapping) leaves an auto-detected crop armed to
    re-detect against the new boundary, and drops a manual one back to none —
    either beats silently keeping a crop that no longer lines up with the frame.
    """
    if old_geom == new_geom:
        return config

    def pt(p: tuple[float, float]) -> list[float]:
        return list(remap_point(p[0], p[1], half, old_geom, new_geom))

    def stroke(points, size, src_dx, src_dy):
        new_points = [pt(p) for p in points]
        if not points:
            return (new_points, size, src_dx, src_dy)
        ox, oy = points[0]
        sx, sy = remap_point(ox + src_dx, oy + src_dy, half, old_geom, new_geom)
        nx, ny = new_points[0]
        return (new_points, size, sx - nx, sy - ny)

    retouch = config.retouch
    heal_strokes = [stroke(*s) for s in retouch.manual_heal_strokes]
    dust_spots = [(*pt((x, y)), size) for (x, y, size) in retouch.manual_dust_spots]
    scratch_lines = [(*pt((x0, y0)), *pt((x1, y1)), width) for (x0, y0, x1, y1, width) in retouch.scratch_lines]
    masks = tuple(replace(m, vertices=tuple(tuple(pt(v)) for v in m.vertices)) for m in config.local.masks)

    geometry = config.geometry
    return replace(
        config,
        retouch=replace(
            retouch,
            manual_heal_strokes=heal_strokes,
            manual_dust_spots=dust_spots,
            scratch_lines=scratch_lines,
        ),
        local=replace(config.local, masks=masks),
        geometry=replace(geometry, crop_rect=None) if geometry.crop_rect is not None else geometry,
    )


def gap_px(left_width: int, right_width: int, gutter_thickness: float) -> int:
    """Width of the diptych gap, from the two rendered halves.

    Derived from the halves rather than from the source: an export can resize, so the
    discarded band's pixel count on the scan is not the gap's pixel count on the output.
    ``gutter_thickness`` is a fraction of the cropped scan width, of which the two halves
    hold the remaining ``1 - gutter_thickness``.
    """
    if gutter_thickness <= 0 or gutter_thickness >= 1:
        return 0
    return int(round((left_width + right_width) * gutter_thickness / (1.0 - gutter_thickness)))


def _pad_height(a: np.ndarray, height: int) -> np.ndarray:
    pad = height - a.shape[0]
    if pad <= 0:
        return a
    top = pad // 2
    return np.pad(a, ((top, pad - top), (0, 0)) + ((0, 0),) * (a.ndim - 2), constant_values=_GAP_FILL)


def join_halves(left: np.ndarray, right: np.ndarray, gap: int = 0) -> np.ndarray:
    """Join two rendered halves side by side, with a ``gap``-wide band where the gutter was.

    The gap is filled rather than copied from the scan: the source gutter is
    scene-linear negative data, so pasting it in gives a bright bar, and running the
    pipeline on a thin dark strip renormalizes it into noise.

    Unequal heights (a per-half crop aspect or border) are centre-padded, never
    resampled — an export must not resize pixels the pipeline already sized.
    """
    height = max(left.shape[0], right.shape[0])
    parts = [_pad_height(left, height)]
    if gap > 0:
        parts.append(np.full((height, gap) + left.shape[2:], _GAP_FILL, dtype=left.dtype))
    parts.append(_pad_height(right, height))
    return np.ascontiguousarray(np.concatenate(parts, axis=1))


def diptych_configs(repo: Any, file_hash: Optional[str]) -> Optional[tuple[Any, Any]]:
    """The two halves' saved edits for a whole-frame scan, or None when it has none.

    A half with no saved edit takes its sibling's, so a scan where only one side was
    worked on still renders as a consistent pair. Any ``#``-suffixed hash is rejected:
    a half is not a diptych, and a composite's base is its reference frame, whose half
    edits are not the composite's. A scan the user never split is rejected too — see
    `split_scans`.
    """
    if not file_hash or _SEP in file_hash or file_hash not in split_scans(repo):
        return None
    keys = [half_hash(file_hash, 1), half_hash(file_hash, 2)]
    found = repo.load_file_settings_many(keys)
    first, second = found.get(keys[0]), found.get(keys[1])
    if first is None and second is None:
        return None
    return (first or second, second or first)


def detect_gutter(buf: np.ndarray) -> tuple[float, float]:
    """Normalized (split_x, gutter_thickness) of the unexposed band between the two frames.

    The gutter is a narrow column extremal against its surroundings in either polarity,
    bright film base on a negative and dark on a positive, so the pick is the column whose
    smoothed luma deviates most from a local running-median background, over a window much
    wider than the gutter. Its edges are the steepest slope on each side of that peak,
    searched in a window sized to the smoothing rather than to the deviation band, since an
    in-scene gradient blending into the gutter widens that band on one side and drags the
    center with it. Returns ``(0.5, 0.0)`` when no gutter stands out.
    """
    a = np.asarray(buf)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32, copy=False)
    h, w = a.shape[:2]
    if w < 64 or h < 8:
        return 0.5, 0.0
    peak_val = float(a.max())
    if peak_val <= 0:
        return 0.5, 0.0
    sub = a[:: max(1, h // 512)] / peak_val
    col = sub.mean(axis=0)
    k = max(3, w // 150)
    sm = np.convolve(col, np.ones(k, np.float32) / k, mode="same")
    win = max(9, (w // 8) | 1)
    padded = np.pad(sm, win // 2, mode="edge")
    bg = np.median(np.lib.stride_tricks.sliding_window_view(padded, win), axis=1)
    dev = np.abs(sm - bg)
    lo, hi = int(w * 0.35), int(w * 0.65)
    peak = lo + int(np.argmax(dev[lo:hi]))

    rad = max(int(w * 0.05), 6 * k)
    s0, s1 = max(0, peak - rad), min(w - 1, peak + rad)
    grad = np.gradient(sm[s0 : s1 + 1])
    peak_rel = peak - s0
    bright = sm[peak] >= bg[peak]
    rising, falling = (True, False) if bright else (False, True)
    left_edge = _subpixel_extreme(grad[: peak_rel + 1], s0, rising)
    right_edge = _subpixel_extreme(grad[peak_rel:], peak, falling)
    if right_edge < left_edge:
        left_edge, right_edge = right_edge, left_edge
    center_f = 0.5 * (left_edge + right_edge)
    center = max(0, min(w - 1, int(round(center_f))))

    # A gutter is extremal against BOTH sides. A step edge, up one side and down the other,
    # is in-scene, so reject it.
    delta = max(3, int(w * 0.05))
    d1 = float(sm[center] - sm[max(0, center - delta)])
    d2 = float(sm[center] - sm[min(w - 1, center + delta)])
    if min(abs(d1), abs(d2)) < 0.04 or d1 * d2 <= 0:
        return 0.5, 0.0
    # Unexposed film is uniform top to bottom; a bright/dark in-scene feature isn't.
    if float(sub[:, center].std()) > 0.10:
        return 0.5, 0.0
    thickness = min(_MAX_GUTTER_THICKNESS, max(0.0, (right_edge - left_edge) / w))
    return float(center_f / w), float(thickness)


def _subpixel_extreme(seg: np.ndarray, offset: int, want_max: bool) -> float:
    """Sub-pixel index of ``seg``'s max (or min) via a parabola through its neighbors."""
    idx = int(np.argmax(seg)) if want_max else int(np.argmin(seg))
    if 0 < idx < len(seg) - 1:
        y0, y1, y2 = seg[idx - 1], seg[idx], seg[idx + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
    else:
        frac = 0.0
    return offset + idx + max(-0.5, min(0.5, frac))


def detect_split_x(buf: np.ndarray) -> float:
    """Normalized x of the unexposed gutter between the two frames; see ``detect_gutter``."""
    return detect_gutter(buf)[0]


def detect_split_x_for_file(file_path: str) -> float:
    """Gutter position from a small decode of the file; 0.5 on any failure."""
    try:
        from negpy.services.assets.thumbnails import decode_source_image

        img = decode_source_image(file_path)
        if img is None:
            return 0.5
        img.thumbnail((1024, 1024))
        return detect_split_x(np.asarray(img))
    except Exception as e:
        logger.warning("Half-frame split detection failed for %s: %s", file_path, e)
        return 0.5


_MAX_FILM_MARGIN = 0.15  # bounds a plausible rebate/sprocket margin; wider is read as picture content
_EDGE_MIN_CONTRAST = 0.08
_EDGE_MAX_UNIFORMITY = 0.02  # stricter than the gutter's 0.10: a wrong trim here deletes picture, not just a split line
_EDGE_EXTREME_TOL = 0.20  # how far the band's level may sit from the frame's own darkest/brightest tone


def _edge_band(profile: np.ndarray) -> Optional[tuple[float, float]]:
    """Sub-pixel inward extent (pixels) and level of a uniform band anchored at index 0
    of ``profile``, or None when nothing within ``_MAX_FILM_MARGIN`` stands apart from
    the interior. Unlike the gutter's peak search, the band's outer edge is the array
    boundary itself, so smoothing pads with the edge value rather than zeros.
    """
    n = len(profile)
    k = max(3, n // 150)
    cap = int(n * _MAX_FILM_MARGIN)
    if cap < 4:
        return None
    pad = k // 2
    sm = np.convolve(np.pad(profile, pad, mode="edge"), np.ones(k, np.float32) / k, mode="same")[pad : pad + n]
    edge_level = float(np.median(sm[:k]))
    lo, hi = cap, min(n, 2 * cap)
    if hi <= lo:
        return None
    interior_level = float(np.median(sm[lo:hi]))
    if abs(edge_level - interior_level) < _EDGE_MIN_CONTRAST:
        return None
    mid = 0.5 * (edge_level + interior_level)
    rising = interior_level > edge_level
    i = 0
    while i < cap and ((sm[i] < mid) if rising else (sm[i] > mid)):
        i += 1
    if i == 0 or i >= cap:
        return None
    y0, y1 = sm[i - 1], sm[i]
    frac = (mid - y0) / (y1 - y0) if y1 != y0 else 0.0
    return (i - 1) + max(0.0, min(1.0, frac)), edge_level


def _side_margin(norm: np.ndarray, axis: int, from_end: bool, gmin: float, gmax: float) -> float:
    """Normalized inward film-edge margin on one side of ``norm`` (collapsed along
    ``axis``: 0 for left/right, 1 for top/bottom; ``from_end`` picks the far side),
    or 0.0 when that edge does not read as unexposed film.

    A band whose level sits well inside the frame's own tonal range, rather than near
    its darkest or brightest tone, is picture content even where it is locally uniform
    (a calm sea, an overcast sky) -- rejected here rather than by contrast or
    uniformity alone, since either can look like film base by coincidence.
    """
    profile = norm.mean(axis=axis)
    n = len(profile)
    if from_end:
        profile = profile[::-1]
    band = _edge_band(profile)
    if band is None:
        return 0.0
    edge_px, edge_level = band
    span = max(1e-6, gmax - gmin)
    if min(abs(edge_level - gmin), abs(edge_level - gmax)) / span > _EDGE_EXTREME_TOL:
        return 0.0
    # Uniformity is checked over the band's inner half, short of the transition itself:
    # a real edge slopes there even when the frame is tilted or the object rounds off.
    depth = max(2, int(round(edge_px * 0.6)))
    if axis == 0:
        region = norm[:, :depth] if not from_end else norm[:, n - depth :]
    else:
        region = norm[:depth, :] if not from_end else norm[n - depth :, :]
    if float(region.std()) > _EDGE_MAX_UNIFORMITY:
        return 0.0
    return float(edge_px / n)


def detect_film_crop(buf: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """Normalized (x1, y1, x2, y2) outer film extent: the unexposed rebate alongside a
    diptych's own edges. Both frames and the gutter between them must stay inside it, so
    this runs once on the whole scan, not per half, and each of the four sides is
    searched independently, inward from the scan's own boundary, for the same
    uniform-and-extremal signature ``detect_gutter`` finds in the middle. A side with no
    rebate -- common under tight framing -- is left uncropped rather than guessed at;
    None when every side reads as picture content, and on too small a buffer.
    """
    a = np.asarray(buf)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32, copy=False)
    h, w = a.shape[:2]
    if w < 64 or h < 8:
        return None
    peak_val = float(a.max())
    if peak_val <= 0:
        return None
    norm = a / peak_val
    gmin, gmax = float(norm.min()), float(norm.max())
    left = _side_margin(norm, 0, False, gmin, gmax)
    right = _side_margin(norm, 0, True, gmin, gmax)
    top = _side_margin(norm, 1, False, gmin, gmax)
    bottom = _side_margin(norm, 1, True, gmin, gmax)
    if left == 0.0 and right == 0.0 and top == 0.0 and bottom == 0.0:
        return None
    return (left, top, 1.0 - right, 1.0 - bottom)


def detect_split_and_crop_for_file(
    file_path: str,
) -> tuple[float, float, Optional[tuple[float, float, float, float]]]:
    """Gutter position, gutter thickness and outer film crop from one decode of the
    file -- the triple Auto-detect All Splits saves as a roll's half-frame profile.
    (0.5, 0.0, None) on any failure, matching detect_split_x_for_file's fallback."""
    try:
        from negpy.services.assets.thumbnails import decode_source_image

        img = decode_source_image(file_path)
        if img is None:
            return 0.5, 0.0, None
        img.thumbnail((1024, 1024))
        buf = np.asarray(img)
        crop_rect = detect_film_crop(buf)
        # split_x is relative to the cropped width (slice_half's own convention), so the
        # gutter search has to run inside the new crop, not the full, uncropped scan.
        detect_buf = slice_half(buf, 0, 0.5, crop_rect=crop_rect) if crop_rect is not None else buf
        split_x, thickness = detect_gutter(detect_buf)
        return split_x, thickness, crop_rect
    except Exception as e:
        logger.warning("Half-frame split/crop detection failed for %s: %s", file_path, e)
        return 0.5, 0.0, None
