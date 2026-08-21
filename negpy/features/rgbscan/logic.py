import os
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from negpy.domain.tokens import composite_token
from negpy.features.rgbscan.models import RgbScanConfig, is_rgb_triplet

# Channel indices, matching the demosaiced RGB axis order.
RED, GREEN, BLUE = 0, 1, 2

# Estimate translation at this width then scale up: cheaper and denoises the FFT peak.
_EST_WIDTH = 1024

# Scene signature: block-averaged mosaic, this many blocks wide.
_SIG_WIDTH = 96
# Score only this fraction of the frame. The holder, the rebate and the copy-stand
# background outside it are identical in every exposure of every frame, so scoring
# the whole capture correlates two unrelated scenes on their shared surround.
_SIG_INTERIOR = 0.6
# Floor a triplet's three exposures must clear. A backstop, not the discriminator:
# how well one scene scores across channels varies with how much texture it has, so
# no single value separates a real triplet from a confusable one on every roll. The
# ranking test does that. This only has to reject the unrelated, which is what a
# folder too small for a ranking to mean anything falls back on.
MIN_TRIPLET_AFFINITY = 0.35

# Strongest channel over the next, above which a file was lit by one narrowband color.
# Well clear of both: white light leaves the three channels within a small factor,
# while a red, green or blue exposure puts one of them several times ahead.
_NARROWBAND_DOMINANCE = 2.0


@dataclass(frozen=True)
class Triplet:
    """One RGB-scan frame: the three exposures assigned to R/G/B channels."""

    red: str
    green: str
    blue: str
    ok: bool  # False when the chunk didn't classify to one of each channel, or didn't agree.


@dataclass(frozen=True)
class FrameProbe:
    """One file's classification evidence, from a single raw read."""

    means: Tuple[float, float, float]
    signature: np.ndarray


def classify_channel(means: Sequence[float]) -> int:
    """Dominant channel of an (R, G, B) mean triple. Narrowband light makes this unambiguous."""
    return int(np.argmax(means[:3]))


def _channel_means(mosaic: np.ndarray, colors: np.ndarray, black: float) -> Tuple[float, float, float]:
    def mean_of(*idx: int) -> float:
        mask = np.isin(colors, idx)
        return float(mosaic[mask].mean()) - black if mask.any() else 0.0

    # color_desc is RGBG: 0=R, 1=G, 2=B, 3=second green.
    return mean_of(0), mean_of(1, 3), mean_of(2)


def _scene_signature(mosaic: np.ndarray) -> np.ndarray:
    """Edge structure of the frame interior, as a z-scored gradient map.

    Block-averaged straight off the mosaic: at this scale a whole CFA cell per block
    reads as luminance, so no demosaic is needed. Structure rather than tone, because
    two exposures of one frame are the same scene through different narrowband light
    and share their edges, not their levels.
    """
    h, w = mosaic.shape[:2]
    block = max(2, (w // _SIG_WIDTH) & ~1)  # even, so every block covers whole CFA cells
    hh, ww = (h // block) * block, (w // block) * block
    if hh < block or ww < block:
        return np.zeros((1, 1), dtype=np.float32)
    small = mosaic[:hh, :ww].reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))

    sh, sw = small.shape
    dh, dw = int(sh * (1.0 - _SIG_INTERIOR) / 2), int(sw * (1.0 - _SIG_INTERIOR) / 2)
    interior = small[dh : sh - dh, dw : sw - dw]

    gy, gx = np.gradient(interior)
    return _zscore(np.hypot(gx, gy)).astype(np.float32)


def _zscore(a: np.ndarray) -> np.ndarray:
    centered = a - a.mean()
    sd = float(centered.std())
    return centered / sd if sd > 0 else centered


def frame_affinity(a: np.ndarray, b: np.ndarray) -> float:
    """How much two signatures look like the same frame. 1 is identical, 0 unrelated."""
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return 0.0
    return float((a * b).sum() / a.size)


def probe_frame(path: str) -> FrameProbe:
    """Classification means and scene signature, from one read of the raw."""
    import rawpy

    with rawpy.imread(path) as raw:
        mosaic = raw.raw_image_visible.astype(np.float32)
        colors = raw.raw_colors_visible
        black = float(np.mean(raw.black_level_per_channel))
        return FrameProbe(means=_channel_means(mosaic, colors, black), signature=_scene_signature(mosaic))


def probe_channel_means(path: str) -> Tuple[float, float, float]:
    """Black-subtracted per-Bayer-color means, without demosaicing (cheap classification probe)."""
    return probe_frame(path).means


def channel_dominance(means: Sequence[float]) -> float:
    """How far the strongest channel stands above the next. Narrowband light puts one
    channel far ahead; light that carries all three leaves them close together."""
    ordered = sorted(means[:3], reverse=True)
    return ordered[0] / ordered[1] if ordered[1] > 0 else float("inf")


def looks_narrowband(all_means: Sequence[Sequence[float]]) -> bool:
    """Whether a folder was shot one color at a time.

    Separates a trichrome folder that failed to group from a folder of ordinary scans
    that RGB Scan should never have been applied to — a distinction worth making,
    because the first needs explaining and the second needs turning off. The median,
    so a few odd frames do not decide it.
    """
    if not all_means:
        return False
    ratios = sorted(channel_dominance(means) for means in all_means)
    return ratios[len(ratios) // 2] >= _NARROWBAND_DOMINANCE


def capture_timestamp(path: str) -> str:
    """The file's stated capture time, or empty when it states none."""
    from negpy.features.metadata.exif_read import extract_scan_from_exif
    from negpy.infrastructure.loaders.helpers import read_exif_from_file

    exif = read_exif_from_file(path)
    return extract_scan_from_exif(exif).datetime_original if exif else ""


def capture_ordered(paths: Sequence[str], times: Dict[str, str]) -> List[str]:
    """``paths`` in capture order, or unchanged when any file states no capture time.

    Chunking assumes the list runs in the order the exposures were shot. A filename
    only carries that order by convention, and a convention that sorts the color word
    above the frame number groups a roll by color instead. The clock is the thing that
    actually records it. All or nothing: ordering some files by time and the rest by
    name interleaves two sequences and chunks worse than either alone. The filename
    breaks ties, since a whole-second stamp cannot separate one burst.
    """
    if not paths or any(not times.get(path) for path in paths):
        return list(paths)
    return sorted(paths, key=lambda path: (times[path], os.path.basename(path).lower()))


def _affinity_lookup(paths: Sequence[str], signatures: Dict[str, np.ndarray]) -> Callable[[str, str], float]:
    """An ``(a, b) -> affinity`` callable over ``paths``.

    Signatures are z-scored, so their affinity is a dot product over the element
    count, and the whole pairwise matrix is one product. Mixed signature shapes (two
    cameras in one folder) have no common matrix, so those fall back to pairwise.
    """
    if paths and len({signatures[p].shape for p in paths}) == 1:
        index = {p: i for i, p in enumerate(paths)}
        rows = np.stack([signatures[p].ravel() for p in paths]).astype(np.float32)
        matrix = (rows / np.sqrt(rows.shape[1])) @ (rows / np.sqrt(rows.shape[1])).T
        return lambda a, b: float(matrix[index[a], index[b]])
    return lambda a, b: frame_affinity(signatures[a], signatures[b])


def _mutual_best(triplet: Triplet, pools: Dict[int, List[str]], affinity: Callable[[str, str], float]) -> bool:
    """Whether each exposure finds the other two when it searches the whole folder for
    its best partner in each other channel.

    A ranking, not a score, so it is unaffected by a roll whose affinities all run low.
    It needs candidates to rank: with one file per channel every member is trivially
    its own best match, and only the floor stands between that and any three files.
    """
    members = (triplet.red, triplet.green, triplet.blue)
    for channel, member in enumerate(members):
        for want in (RED, GREEN, BLUE):
            if want == channel:
                continue
            pool = pools.get(want)
            if not pool or max(pool, key=lambda q: affinity(member, q)) != members[want]:
                return False
    return True


def triplet_affinity(triplet: Triplet, signatures: Dict[str, np.ndarray]) -> Optional[float]:
    """Weakest pairwise agreement among a triplet's three exposures, or None when any
    signature is missing. The weakest pair, because one intruder must sink the chunk."""
    sigs = [signatures.get(p) for p in (triplet.red, triplet.green, triplet.blue)]
    if any(sig is None for sig in sigs):
        return None
    return min(
        frame_affinity(sigs[0], sigs[1]),
        frame_affinity(sigs[0], sigs[2]),
        frame_affinity(sigs[1], sigs[2]),
    )


def group_triplets(
    items: Sequence[Tuple[str, int]],
    signatures: Optional[Dict[str, np.ndarray]] = None,
    min_affinity: float = MIN_TRIPLET_AFFINITY,
) -> List[Triplet]:
    """Group classified files into consecutive triplets.

    ``items`` is ``[(path, channel), ...]`` already sorted into capture order. Files
    are chunked in threes; within a chunk each file is placed by its dominant channel,
    so the order inside a chunk does not matter. A chunk that doesn't yield exactly one
    of each channel, or a short trailing chunk, is returned best-effort with ``ok=False``.

    With ``signatures``, a chunk must also agree on what it shows. One of each channel
    is too weak a test on its own: a list off by one file still yields one of each, from
    two different frames, and would otherwise merge silently. Two tests apply, because
    they fail in different places — the three must each be the others' best match in the
    folder, and must clear ``min_affinity``. A chunk with a missing signature keeps the
    membership-only verdict rather than being rejected unseen.
    """
    known = [path for path, _ in items if signatures and path in signatures]
    affinity = _affinity_lookup(known, signatures) if known else None
    pools: Dict[int, List[str]] = {}
    if signatures:
        for path, channel in items:
            if path in signatures:
                pools.setdefault(channel, []).append(path)

    triplets: List[Triplet] = []
    for i in range(0, len(items), 3):
        chunk = items[i : i + 3]
        by_channel = {ch: path for path, ch in chunk}
        paths = [path for path, _ in chunk]
        triplet = Triplet(
            red=by_channel.get(RED, paths[0] if paths else ""),
            green=by_channel.get(GREEN, paths[1] if len(paths) > 1 else ""),
            blue=by_channel.get(BLUE, paths[2] if len(paths) > 2 else ""),
            ok=len(chunk) == 3 and set(by_channel) == {RED, GREEN, BLUE},
        )
        if triplet.ok and signatures and affinity is not None:
            score = triplet_affinity(triplet, signatures)
            if score is not None and (score < min_affinity or not _mutual_best(triplet, pools, affinity)):
                triplet = replace(triplet, ok=False)
        triplets.append(triplet)
    return triplets


def estimate_shift(ref_gray: np.ndarray, mov_gray: np.ndarray) -> Tuple[float, float]:
    """Sub-pixel translation of ``mov_gray`` relative to ``ref_gray`` (phase correlation)."""
    h, w = ref_gray.shape[:2]
    scale = 1.0
    r, m = ref_gray, mov_gray
    if w > _EST_WIDTH:
        scale = w / _EST_WIDTH
        sz = (_EST_WIDTH, max(1, round(h / scale)))
        r = cv2.resize(ref_gray, sz, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mov_gray, sz, interpolation=cv2.INTER_AREA)
    r = np.ascontiguousarray(r, dtype=np.float32)
    m = np.ascontiguousarray(m, dtype=np.float32)
    win = cv2.createHanningWindow((r.shape[1], r.shape[0]), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(r, m, win)
    return dx * scale, dy * scale


def _align_to(ref_gray: np.ndarray, mov: np.ndarray, mov_ch: int, max_shift: float) -> np.ndarray:
    """Shift ``mov`` so its scene content lines up with ``ref_gray``. No-op if the
    estimate is implausibly large (correlation failed)."""
    dx, dy = estimate_shift(ref_gray, mov[..., mov_ch])
    if max(abs(dx), abs(dy)) > max_shift:
        return mov
    h, w = mov.shape[:2]
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    return cv2.warpAffine(mov, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def assemble_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, align: bool = True) -> np.ndarray:
    """Assemble one HxWx3 image: red channel from the red shot, green from green, blue from blue.

    With ``align``, green/blue are registered to the red exposure first (sub-pixel
    translation) to remove fringing from frame-to-frame drift during capture.
    """
    if not (r.shape == g.shape == b.shape):
        raise ValueError(f"RGB-scan exposures differ in shape: {r.shape}, {g.shape}, {b.shape}")
    out = np.empty_like(r)
    out[..., RED] = r[..., RED]
    if align:
        ref = r[..., RED].astype(np.float32)
        max_shift = max(16.0, 0.02 * r.shape[1])
        g = _align_to(ref, g, GREEN, max_shift)
        b = _align_to(ref, b, BLUE, max_shift)
    out[..., GREEN] = g[..., GREEN]
    out[..., BLUE] = b[..., BLUE]
    return out


def merge_rgb_triplet(
    decode_fn: Callable[[str], np.ndarray],
    red_path: str,
    green_path: str,
    blue_path: str,
    align: bool = True,
) -> np.ndarray:
    """Decode the three exposures via ``decode_fn`` and assemble them into one frame."""
    return assemble_rgb(decode_fn(red_path), decode_fn(green_path), decode_fn(blue_path), align=align)


def rgbscan_token(config: RgbScanConfig) -> str:
    """Identity of the active triplet, folded into the render source hash. Empty when inactive."""
    if not is_rgb_triplet(config):
        return ""
    return composite_token("rgb", config, (config.green_path, config.blue_path))
