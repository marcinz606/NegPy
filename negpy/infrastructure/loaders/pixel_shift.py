"""Pentax Pixel Shift: the four shots of one RAW combined into measured red, green and blue at
every pixel, and a check that the shots agree.

The shot mapping follows RawTherapee (rtengine/pixelshift.cc) without its motion correction:
shot k shows at (i + dy, j + dx) what the first shot shows at (i, j), with (dy, dx) = (0, 0),
(1, 0), (1, 1), (0, 1). The four samples of pixel (i, j) come from the 2x2 CFA block at (i, j),
so each pixel gets one red, two green and one blue sample, and no demosaic runs.
"""

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from negpy.infrastructure.capture.raw_demosaic import _user_sat
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, open_shot
from negpy.kernel.image.logic import uint16_to_float32
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

SHOT_SHIFT: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (1, 1), (0, 1))

# A green difference above this many noise sigmas counts as a mismatch.
_MISMATCH_SIGMA = 6.0
# Above this share of mismatched pixels, the camera or the subject moved between the shots.
MISMATCH_WARNING = 0.005
# Above this change of the mean level from the first shot, the light changed during the sequence.
LIGHT_WARNING = 0.01

_NOISE_BINS = 32
_NOISE_BIN_MIN = 256
_FLAT_SHARE = 0.25
_CACHE_BYTES = 640 * 1024 * 1024


@dataclass(frozen=True)
class PixelShiftStats:
    """How well the four shots of one Pixel Shift RAW agree, measured while they are combined."""

    mismatch: float  # share of pixels whose two green samples, from two shots, differ beyond noise
    pair_mismatch: tuple[float, float]  # the same share for shots 1 and 3, and for shots 2 and 4
    levels: tuple[float, ...]  # mean level of each shot over the first shot
    noise: tuple[float, float]  # var(green difference) = gain * level + floor, in sensor counts
    seconds: float  # read and combine time

    @property
    def light_change(self) -> float:
        return max(abs(level - 1.0) for level in self.levels)

    @property
    def disagreement(self) -> float:
        """The larger of mismatch and light change, each over its warning threshold."""
        return max(self.mismatch / MISMATCH_WARNING, self.light_change / LIGHT_WARNING)

    @property
    def shots_disagree(self) -> bool:
        return self.disagreement > 1.0


class _GreenCheck:
    """Collects the green sample pairs of the four output parities and measures their agreement.

    In a still, aligned scene the two green samples of a pixel differ only by noise. The noise
    model comes from the flattest pixels of each level: on an edge, misaligned shots differ by
    the edge itself, so a fit over all pixels grows with the misalignment and hides it.
    """

    def __init__(self) -> None:
        self._pairs: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
        self._fit: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(self, shots: tuple[int, int], first: np.ndarray, second: np.ndarray, total: np.ndarray) -> None:
        diff = first - second
        self._pairs.append(((min(shots), max(shots)), diff[::2, ::2].copy(), total[::2, ::2].copy()))
        green = total.astype(np.float32)
        grad = np.abs(green[1:-1, 2:] - green[1:-1, :-2]) + np.abs(green[2:, 1:-1] - green[:-2, 1:-1])
        self._fit.append((diff[1:-1:4, 1:-1:4].ravel(), total[1:-1:4, 1:-1:4].ravel(), grad[::4, ::4].ravel()))

    def _noise_model(self) -> tuple[float, float]:
        if not self._fit:
            return 0.0, 1.0
        diff = np.concatenate([d for d, _t, _g in self._fit]).astype(np.float64)
        level = 0.5 * np.concatenate([t for _d, t, _g in self._fit]).astype(np.float64)
        grad = np.concatenate([g for _d, _t, g in self._fit])
        if diff.size == 0:
            return 0.0, 1.0
        order = np.argsort(level, kind="stable")
        diff, level, grad = diff[order], level[order], grad[order]
        bins = max(1, min(_NOISE_BINS, diff.size // _NOISE_BIN_MIN))
        centers, variances = [], []
        for d, lv, gr in zip(np.array_split(diff, bins), np.array_split(level, bins), np.array_split(grad, bins)):
            keep = gr <= np.quantile(gr, _FLAT_SHARE)
            flat = d[keep]
            mad = np.median(np.abs(flat - np.median(flat)))
            centers.append(float(np.median(lv[keep])))
            variances.append(float((1.4826 * mad) ** 2))
        if len(centers) < 2 or max(centers) == min(centers):
            return 0.0, max(float(np.mean(variances)), 1.0)
        gain, floor = np.polyfit(np.array(centers), np.array(variances), 1)
        return max(float(gain), 0.0), max(float(floor), 1.0)

    def measure(self) -> tuple[float, tuple[float, float], tuple[float, float]]:
        """(share of mismatched pixels, share per shot pair, noise model)."""
        gain, floor = self._noise_model()
        hits = {(0, 2): [0, 0], (1, 3): [0, 0]}
        limit = np.float32(_MISMATCH_SIGMA * _MISMATCH_SIGMA)
        for pair, diff, total in self._pairs:
            allowed = (np.float32(0.5 * gain) * total.astype(np.float32) + np.float32(floor)) * limit
            d = diff.astype(np.float32)
            count = hits.setdefault(pair, [0, 0])
            count[0] += int(np.count_nonzero(d * d > allowed))
            count[1] += d.size
        moving = sum(h[0] for h in hits.values())
        total_px = sum(h[1] for h in hits.values())
        share_13, share_24 = (hits[p][0] / hits[p][1] if hits[p][1] else 0.0 for p in ((0, 2), (1, 3)))
        return (moving / total_px if total_px else 0.0), (share_13, share_24), (gain, floor)


def combine_shots(shots: np.ndarray, cfa: np.ndarray, black: Sequence[int]) -> tuple[np.ndarray, _GreenCheck]:
    """Combine four (H, W) shots into HxWx3 uint16 sensor counts above black.

    R and B hold one count each, G the sum of its two counts, so the result is exact in 16 bits
    for a sensor up to 15 bits. `cfa` is the 2x2 color-index block of the first shot
    (raw_colors_visible: 0 R, 1 G, 2 B, 3 second G), `black` the black level per color index.
    The last row and column take their neighbors, which the mapping leaves without samples.
    """
    count, h, w = shots.shape
    if count != len(SHOT_SHIFT) or h < 3 or w < 3:
        raise ValueError(f"Pixel Shift needs {len(SHOT_SHIFT)} shots of at least 3x3 photosites, got {shots.shape}")
    counts = np.empty((h, w, 3), dtype=np.uint16)
    check = _GreenCheck()
    for p in (0, 1):
        for q in (0, 1):
            rows, cols = len(range(p, h - 1, 2)), len(range(q, w - 1, 2))
            dst = counts[p : h - 1 : 2, q : w - 1 : 2]
            greens = []
            for k, (dy, dx) in enumerate(SHOT_SHIFT):
                r0, c0 = p + dy, q + dx
                site = int(cfa[r0 % 2, c0 % 2])
                sample = shots[k, r0 : r0 + 2 * rows : 2, c0 : c0 + 2 * cols : 2].astype(np.int32)
                sample -= int(black[site])
                np.maximum(sample, 0, out=sample)
                if site in (1, 3):
                    greens.append((k, sample))
                else:
                    dst[..., site] = sample
            if len(greens) != 2:
                raise ValueError(f"Pixel Shift needs a Bayer CFA with two green sites, got {cfa.tolist()}")
            (ka, ga), (kb, gb) = greens
            total = ga + gb
            np.minimum(total, 65535, out=total)
            dst[..., 1] = total
            check.add((ka, kb), ga, gb, total)
    counts[h - 1] = counts[h - 2]
    counts[:, w - 1] = counts[:, w - 2]
    return counts, check


def shot_levels(shots: np.ndarray, black: Sequence[int]) -> tuple[float, ...]:
    """Mean level of each shot above black, over the first shot."""
    offset = float(np.mean(black))
    means = [float(shots[k].mean(dtype=np.float64)) - offset for k in range(shots.shape[0])]
    return tuple(m / means[0] if means[0] > 0 else 1.0 for m in means)


@dataclass(frozen=True)
class PixelShiftImage:
    """The four shots of one Pixel Shift RAW combined, in sensor counts above black, plus the
    first shot's color data. It holds no scale: each decode path asks for its own."""

    counts: np.ndarray  # HxWx3 uint16; G holds the sum of its two samples
    render_white: int  # full scale of the render and preview decodes: _user_sat of the first shot
    export_white: int  # full scale of the linear export: LibRaw's white level minus black
    rgb_xyz_matrix: np.ndarray
    camera_whitebalance: tuple[float, ...]
    daylight_whitebalance: tuple[float, ...]
    sizes: Any  # the first shot's LibRaw sizes: its visible area is the combined image
    other: Any  # the first shot's LibRaw capture data (timestamp, exposure)
    stats: PixelShiftStats

    @property
    def nbytes(self) -> int:
        return int(self.counts.nbytes)

    def to_u16(self, white: int, gains: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """16-bit linear camera RGB on `white` full scale, the way LibRaw scales one shot:
        trunc(float32(counts) * float32(gain * 65535 / white)), clipped at 65535."""
        out = np.empty(self.counts.shape, dtype=np.uint16)
        for c, samples in ((0, 1), (1, 2), (2, 1)):
            channel = self.counts[..., c].astype(np.float32)
            channel *= np.float32(gains[c] * 65535.0 / (samples * white))
            np.minimum(channel, np.float32(65535.0), out=channel)
            out[..., c] = channel
        return out


def libraw_wb_gains(multipliers: Optional[Sequence[float]]) -> tuple[float, float, float]:
    """(R, G, B) gains as LibRaw applies [R, G, B, G2] multipliers with highlight clipping: divided
    by the smallest of the four. The two greens of a pixel share the mean of the two green gains."""
    if multipliers is None or len(multipliers) < 3:
        return (1.0, 1.0, 1.0)
    r, g, b = (float(v) for v in multipliers[:3])
    g2 = float(multipliers[3]) if len(multipliers) > 3 and float(multipliers[3]) > 0 else g
    values = (r, g, b, g2)
    if not all(np.isfinite(values)) or min(values) <= 0:
        return (1.0, 1.0, 1.0)
    low = min(values)
    return (r / low, 0.5 * (g + g2) / low, b / low)


class PixelShiftWrapper(NonStandardFileWrapper):
    """A combined Pixel Shift image behind the rawpy calls NegPy makes on a camera RAW.

    It carries the first shot's matrix, white balance, sizes and capture data, so the render
    reads it like one shot. postprocess() honors use_camera_wb and user_wb as LibRaw does, which
    keeps a bracket sibling on the reference white balance. It ignores highlight_mode and bright.
    """

    def __init__(self, image: PixelShiftImage, white: int) -> None:
        super().__init__(wb_gains=None)
        self.image = image
        self.white = white
        self.rgb_xyz_matrix = image.rgb_xyz_matrix
        self.camera_whitebalance = image.camera_whitebalance
        self.daylight_whitebalance = image.daylight_whitebalance
        self.other = image.other

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = uint16_to_float32(self.image.to_u16(self.white))
        return self._data

    @property
    def sizes(self) -> Any:
        return self.image.sizes

    def close(self) -> None:
        self._data = None

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        if kwargs.get("use_camera_wb"):
            gains = libraw_wb_gains(self.camera_whitebalance)
        else:
            gains = libraw_wb_gains(kwargs.get("user_wb"))
        u16 = self.image.to_u16(self.white, gains)
        if kwargs.get("half_size"):
            u16 = np.ascontiguousarray(u16[::2, ::2])
        bps = kwargs.get("output_bps", 8)
        gamma = kwargs.get("gamma")
        if bps == 16 and gamma is not None and tuple(gamma) == (1, 1):
            return u16
        return NonStandardFileWrapper(source_u16=u16).postprocess(output_bps=bps, gamma=gamma)


class _ImageCache:
    """Combined images by file identity, least recently used out beyond a byte budget. One lock
    per file, so decodes of one file that start together wait for a single combine."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._items: OrderedDict[tuple, PixelShiftImage] = OrderedDict()
        self._lock = threading.Lock()
        self._file_locks: dict[tuple, threading.Lock] = {}

    def _get(self, key: tuple) -> Optional[PixelShiftImage]:
        with self._lock:
            image = self._items.get(key)
            if image is not None:
                self._items.move_to_end(key)
            return image

    def get_or_build(self, key: tuple, build: Callable[[], PixelShiftImage]) -> PixelShiftImage:
        image = self._get(key)
        if image is not None:
            return image
        with self._lock:
            file_lock = self._file_locks.setdefault(key, threading.Lock())
        with file_lock:
            image = self._get(key)
            if image is not None:
                return image
            # A failed or cancelled build keeps the lock, so a waiter and a newcomer still share it.
            image = build()
            with self._lock:
                self._items[key] = image
                self._file_locks.pop(key, None)
                total = sum(item.nbytes for item in self._items.values())
                while total > self._max_bytes and len(self._items) > 1:
                    _key, old = self._items.popitem(last=False)
                    total -= old.nbytes
            return image

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_CACHE = _ImageCache(_CACHE_BYTES)


def clear_cache() -> None:
    """Drop every combined image held for this session."""
    _CACHE.clear()


def _build(file_path: str, should_cancel: Optional[Callable[[], bool]]) -> PixelShiftImage:
    start = time.perf_counter()
    shots: Optional[np.ndarray] = None
    first: dict = {}
    for k in range(len(SHOT_SHIFT)):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("pixel shift combine cancelled")
        with open_shot(file_path, k) as raw:
            visible = raw.raw_image_visible
            if shots is None:
                shots = np.empty((len(SHOT_SHIFT), *visible.shape), dtype=np.uint16)
                black = [int(v) for v in raw.black_level_per_channel]
                export_white = int(raw.white_level) - min(black)
                first = {
                    "cfa": np.array(raw.raw_colors_visible[:2, :2]),
                    "black": black,
                    "render_white": int(_user_sat(raw) or export_white),
                    "export_white": export_white,
                    # Copied: the arrays LibRaw hands back die with the decoder.
                    "rgb_xyz_matrix": np.array(raw.rgb_xyz_matrix, dtype=np.float64),
                    "camera_whitebalance": tuple(float(v) for v in raw.camera_whitebalance),
                    "daylight_whitebalance": tuple(float(v) for v in raw.daylight_whitebalance),
                    "sizes": raw.sizes,
                    "other": raw.other,
                }
            elif visible.shape != shots.shape[1:]:
                raise ValueError(f"Pixel Shift shots differ in size: {visible.shape} and {shots.shape[1:]}")
            shots[k] = visible
    assert shots is not None
    if should_cancel is not None and should_cancel():
        raise InterruptedError("pixel shift combine cancelled")
    counts, check = combine_shots(shots, first["cfa"], first["black"])
    levels = shot_levels(shots, first["black"])
    del shots
    mismatch, pair_mismatch, noise = check.measure()
    stats = PixelShiftStats(mismatch, pair_mismatch, levels, noise, time.perf_counter() - start)
    logger.info(
        "pixel shift %s: mismatch %.4f%% (shots 1/3 %.4f%%, 2/4 %.4f%%), levels %s, noise var %.3f*L + %.1f, %.2fs",
        os.path.basename(file_path),
        100 * stats.mismatch,
        100 * pair_mismatch[0],
        100 * pair_mismatch[1],
        " ".join(f"{v:.4f}" for v in levels),
        noise[0],
        noise[1],
        stats.seconds,
    )
    return PixelShiftImage(
        counts=counts,
        render_white=first["render_white"],
        export_white=first["export_white"],
        rgb_xyz_matrix=first["rgb_xyz_matrix"],
        camera_whitebalance=first["camera_whitebalance"],
        daylight_whitebalance=first["daylight_whitebalance"],
        sizes=first["sizes"],
        other=first["other"],
        stats=stats,
    )


def combined_image(file_path: str, should_cancel: Optional[Callable[[], bool]] = None) -> PixelShiftImage:
    """The four shots of `file_path` combined, from the session cache when this file was already
    combined. The cache key is the path, size and modification time."""
    try:
        st = os.stat(file_path)
        key: tuple = (os.path.abspath(file_path), st.st_size, st.st_mtime_ns)
    except OSError:
        key = (os.path.abspath(file_path), -1, -1)
    return _CACHE.get_or_build(key, lambda: _build(file_path, should_cancel))
