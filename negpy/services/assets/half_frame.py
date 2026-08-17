"""Half-frame scans: one file holds two frames side by side.

A half asset is a normal asset dict plus ``half`` (1 = left, 2 = right) and
``split_x`` (normalized gutter position). Its identity is the file hash
suffixed with ``#<half>``, so every hash-keyed store (edits, history, marks,
thumbnails) is per-frame automatically. Decode caches key on the unsuffixed
hash so both halves share one decode.
"""

from typing import Any, Dict, Optional

import numpy as np

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_SEP = "#"

SPLIT_SCANS_KEY = "half_frame_scans"

# The inter-frame gap in a joined diptych, in output space. Black is what the gap between
# two exposures looks like once rendered; swap for the finish border colour if wanted.
_GAP_FILL = 0.0


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
    a = buf
    if crop_rect is not None:
        h, w = a.shape[:2]
        x1, y1, x2, y2 = crop_rect
        cx1 = min(max(int(round(w * x1)), 0), w)
        cx2 = min(max(int(round(w * x2)), 0), w)
        cy1 = min(max(int(round(h * y1)), 0), h)
        cy2 = min(max(int(round(h * y2)), 0), h)
        if cx2 <= cx1 or cy2 <= cy1:
            return a[:, :1]
        a = a[cy1:cy2, cx1:cx2]
    if not half:
        return a
    w = a.shape[1]
    xs = min(max(int(round(w * split_x)), 1), w - 1)
    if gutter_thickness > 0:
        gw = max(1, int(round(w * gutter_thickness)))
        lo = max(1, xs - gw // 2)
        hi = min(w - 1, lo + gw)
        lo = max(1, hi - gw)
        left = a[:, :lo]
        right = a[:, hi:]
        return left if half == 1 else right
    return a[:, :xs] if half == 1 else a[:, xs:]


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


def detect_split_x(buf: np.ndarray) -> float:
    """Normalized x of the unexposed gutter between the two frames.

    The gutter is a narrow column extremal against its surroundings in either
    polarity (bright film base on negatives, dark on positives), so pick the
    column whose smoothed luma deviates most from a local running-median
    background — a window much wider than the gutter, so broad brightness
    differences between the two frames don't register. Returns 0.5 when no
    clear gutter stands out in the central band.
    """
    # ponytail: 1-D local-deviation heuristic; upgrade to variance+edge profile if it misses
    a = np.asarray(buf)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32, copy=False)
    h, w = a.shape[:2]
    if w < 64 or h < 8:
        return 0.5
    peak_val = float(a.max())
    if peak_val <= 0:
        return 0.5
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
    # Take the deviating band's center so the ±delta taps below land outside the gutter.
    thr = 0.5 * dev[peak]
    i0 = peak
    while i0 > 0 and dev[i0 - 1] >= thr:
        i0 -= 1
    i1 = peak
    while i1 < w - 1 and dev[i1 + 1] >= thr:
        i1 += 1
    center = (i0 + i1) // 2
    # A gutter is extremal against BOTH sides. A step edge, up one side and down the other,
    # is in-scene, so reject it.
    delta = max(3, int(w * 0.05))
    d1 = float(sm[center] - sm[max(0, center - delta)])
    d2 = float(sm[center] - sm[min(w - 1, center + delta)])
    if min(abs(d1), abs(d2)) < 0.04 or d1 * d2 <= 0:
        return 0.5
    # Unexposed film is uniform top to bottom; a bright/dark in-scene feature isn't.
    if float(sub[:, center].std()) > 0.10:
        return 0.5
    return center / w


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
