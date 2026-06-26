import os
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

from negpy.features.rgbscan.models import RgbScanConfig

# Channel indices, matching the demosaiced RGB axis order.
RED, GREEN, BLUE = 0, 1, 2


@dataclass(frozen=True)
class Triplet:
    """One RGB-scan frame: the three exposures assigned to R/G/B channels."""

    red: str
    green: str
    blue: str
    ok: bool  # False when the source chunk didn't classify to one of each channel.


def classify_channel(means: Sequence[float]) -> int:
    """Dominant channel of an (R, G, B) mean triple. Narrowband light makes this unambiguous."""
    return int(np.argmax(means[:3]))


def probe_channel_means(path: str) -> Tuple[float, float, float]:
    """Black-subtracted per-Bayer-colour means, without demosaicing (cheap classification probe)."""
    import rawpy

    with rawpy.imread(path) as raw:
        mosaic = raw.raw_image_visible.astype(np.float32)
        colors = raw.raw_colors_visible
        black = float(np.mean(raw.black_level_per_channel))

        def mean_of(*idx: int) -> float:
            mask = np.isin(colors, idx)
            return float(mosaic[mask].mean()) - black if mask.any() else 0.0

        # color_desc is RGBG: 0=R, 1=G, 2=B, 3=second green.
        return mean_of(0), mean_of(1, 3), mean_of(2)


def group_triplets(items: Sequence[Tuple[str, int]]) -> List[Triplet]:
    """Group classified files into consecutive triplets.

    ``items`` is ``[(path, channel), ...]`` already sorted by filename. Files are
    chunked in threes (sequential capture order); within a chunk each file is placed
    by its dominant channel. A chunk that doesn't yield exactly one of each channel,
    or a short trailing chunk, is returned best-effort with ``ok=False``.
    """
    triplets: List[Triplet] = []
    for i in range(0, len(items), 3):
        chunk = items[i : i + 3]
        by_channel = {ch: path for path, ch in chunk}
        ok = len(chunk) == 3 and set(by_channel) == {RED, GREEN, BLUE}
        paths = [path for path, _ in chunk]
        triplets.append(
            Triplet(
                red=by_channel.get(RED, paths[0] if paths else ""),
                green=by_channel.get(GREEN, paths[1] if len(paths) > 1 else ""),
                blue=by_channel.get(BLUE, paths[2] if len(paths) > 2 else ""),
                ok=ok,
            )
        )
    return triplets


def merge_rgb_triplet(
    decode_fn: Callable[[str], np.ndarray],
    red_path: str,
    green_path: str,
    blue_path: str,
) -> np.ndarray:
    """Assemble one HxWx3 image: red channel from the red shot, green from green, blue from blue.

    ``decode_fn`` decodes a path to an HxWx3 array using the caller's normal RAW path.
    """
    r = decode_fn(red_path)
    g = decode_fn(green_path)
    b = decode_fn(blue_path)
    if not (r.shape == g.shape == b.shape):
        raise ValueError(f"RGB-scan exposures differ in shape: {r.shape}, {g.shape}, {b.shape}")
    out = np.empty_like(r)
    out[..., RED] = r[..., RED]
    out[..., GREEN] = g[..., GREEN]
    out[..., BLUE] = b[..., BLUE]
    return out


def rgbscan_token(config: RgbScanConfig) -> str:
    """Identity of the active triplet, folded into the render source hash. Empty when inactive."""
    if not config.enabled or not config.green_path or not config.blue_path:
        return ""
    try:
        g_mtime = os.path.getmtime(config.green_path)
        b_mtime = os.path.getmtime(config.blue_path)
    except OSError:
        return ""
    return f"|rgb:{config.green_path}:{g_mtime}:{config.blue_path}:{b_mtime}"
