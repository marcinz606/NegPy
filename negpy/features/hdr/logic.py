"""
HDR merge of bracketed captures of one frame.

A single camera-scan exposure cannot hold a slide's density range: the densest parts sit
in sensor noise, and the longer exposure that reaches them blows the thin areas. This
merges a bracket into one linear source that carries the lot.

Source assembly, not a pipeline stage — it runs at decode alongside the RGB-scan triplet
merge, so no stage changes and there is no GPU mirror to keep in parity.

Everything is computed in the **reference** frame's exposure units. The reference is the
longest exposure that does not clip, so the result stays inside [0, 1] (the pipeline
clips at entry, see features/process/logic.py) and the recovered range arrives as real
shadow detail rather than as values above white. Which exposure it then *renders* at is a
separate choice, nominated by the user rather than measured — see `output_scale`.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from numba import prange  # type: ignore

from negpy.features.hdr.models import ANCHOR_EV_UNSET, HdrConfig
from negpy.features.rgbscan.logic import estimate_shift
from negpy.kernel.system.parallel import parallel_njit

#: A sample at or above this is saturated and carries no information.
SATURATION = 0.995
#: Weight rolls off linearly from here to SATURATION, so the frame a pixel is taken from
#: changes gradually. A hard switch prints as banding across a smooth gradient.
ROLLOFF_START = 0.90
#: A frame with more than this fraction of saturated pixels cannot be the exposure
#: reference — its own white level is no longer a measurement of anything.
MAX_REFERENCE_CLIPPING = 0.001
#: Samples below this are noise in both frames of a pair and would bias a ratio fit.
RATIO_FLOOR = 0.02
#: Pixel budget for the measurements that only read a distribution — the level percentile
#: and the ratio median. Moves the solved ratios by <0.1%, below one 16-bit step in the
#: merge. The clipped fraction is not subsampled: it picks the reference against a hard
#: threshold, and a different reference is a different white.
MEASURE_PIXELS = 1_500_000


@dataclass(frozen=True)
class ExposureStats:
    """One bracket frame's measured exposure, used to order the bracket and pick the reference."""

    path: str
    level: float  # robust brightness; monotonic in exposure time
    clipped: float  # fraction of samples at or above SATURATION


def to_float(frame: np.ndarray) -> np.ndarray:
    """uint16 decode -> float32 [0, 1], RGB only. Already-float input keeps its buffer."""
    out = frame if frame.dtype == np.float32 else frame.astype(np.float32) / np.float32(65535.0)
    # A carrier channel (IR, alpha) has no place in a radiometric merge, and dropping it
    # from only one of the two input dtypes would make the shape checks below dtype-
    # dependent.
    return out[..., :3] if out.ndim == 3 and out.shape[2] > 3 else out


def subsample(frame: np.ndarray) -> np.ndarray:
    """Strided view of about MEASURE_PIXELS pixels; the frame itself when it is smaller.

    A stride, not a resize: averaging neighbours would blend back in the very samples
    these measurements exclude — saturated ones, noise-floor ones.
    """
    step = int(np.sqrt(frame.shape[0] * frame.shape[1] / MEASURE_PIXELS))
    return frame[::step, ::step] if step > 1 else frame


@parallel_njit(cache=True, fastmath=True)
def _count_clipped_rows(values: np.ndarray, out: np.ndarray) -> None:
    """Saturated-pixel count per row. Per row rather than one total: numba's parfor pass
    cannot analyse a reduction whose addend is itself computed by a loop."""
    height, width, channels = values.shape
    for y in prange(height):
        n = 0
        for x in range(width):
            hit = 0
            for c in range(channels):
                if values[y, x, c] >= SATURATION:
                    hit = 1
            n += hit
        out[y] = n


def clipped_fraction(values: np.ndarray) -> float:
    """Fraction of pixels saturated in any channel. Whole frame, no subsample — see
    MEASURE_PIXELS."""
    rows = np.empty(values.shape[0], dtype=np.int64)
    _count_clipped_rows(values, rows)
    return float(rows.sum()) / float(values.shape[0] * values.shape[1])


def measure_exposure(frame: np.ndarray) -> Tuple[float, float]:
    """(level, clipped fraction) for one decoded frame.

    Level is the 99th percentile rather than the mean or the max: the mean moves with
    scene content across a bracket of the same frame only through exposure, but so does
    a stray specular highlight, and the max is pinned at 1.0 the moment anything clips.

    It saturates too, only later: a frame clipping more than 1% has level exactly 1.0,
    so level alone cannot order the long end of a bracket. Use `exposure_order_key`.
    """
    f = to_float(frame)
    return float(np.percentile(subsample(f), 99.0)), float(clipped_fraction(f))


def exposure_order_key(level: float, clipped: float) -> Tuple[float, float]:
    """Sort key that orders a bracket by exposure, shortest first.

    `level` alone does not: it is a high percentile, so every frame clipping more than
    1% reports exactly 1.0 and they all tie. On a 5-frame bracket that left the four
    longest frames in input order, and `solve_ratios` then chained the reference
    straight to the *longest* frame and walked back down the ramp -- fitting pairs
    4 stops apart, with `step()` receiving the longer frame as its `short` argument.
    The ratios came out up to 7% wrong, which prints as a visible seam where a frame
    drops out of the merge.

    `clipped` is what orders the long end: over one scene, a longer exposure always
    clips at least as much. It is flat at zero below that, where `level` still moves,
    so the pair covers the whole bracket.
    """
    return (clipped, level)


def probe_exposures(decode_fn: Callable[[str], np.ndarray], paths: Sequence[str]) -> List[ExposureStats]:
    """Measure every frame in the bracket. `decode_fn` may decode at reduced size."""
    return [ExposureStats(path, *measure_exposure(decode_fn(path))) for path in paths]


def choose_reference(stats: Sequence[ExposureStats]) -> int:
    """Index of the frame that should define white: the longest exposure that does not clip.

    Longest, not "correctly exposed": it uses the most of [0, 1] and so renders brightest,
    and every shorter frame still contributes through its ratio. If they all clip, the
    least-clipped one is the best available and the caller is told by its `clipped`.
    """
    if not stats:
        raise ValueError("no frames to choose a reference from")
    usable = [i for i, s in enumerate(stats) if s.clipped <= MAX_REFERENCE_CLIPPING]
    if usable:
        return max(usable, key=lambda i: exposure_order_key(stats[i].level, stats[i].clipped))
    return min(range(len(stats)), key=lambda i: stats[i].clipped)


def pair_ratio(short: np.ndarray, long_: np.ndarray) -> Optional[float]:
    """Exposure ratio long_/short from the images themselves, or None if they do not overlap.

    Measured rather than read from EXIF: several supported loaders are scanner formats
    carrying no exposure tags, and measured transmitted light beats a nominal shutter
    speed regardless. The median of per-sample ratios, over samples that are above the
    noise floor in the short frame and unsaturated in the long one — robust to the dust,
    clipping and per-channel differences a least-squares fit would chase. Taken on a
    subsample: a median needs a distribution, not every pixel.
    """
    s, lo = to_float(short), to_float(long_)
    if s.shape != lo.shape:
        raise ValueError(f"bracket frames differ in shape: {s.shape}, {lo.shape}")
    s, lo = subsample(s), subsample(lo)
    valid = (s >= RATIO_FLOOR) & (lo < SATURATION)
    if int(valid.sum()) < 1000:
        return None
    ratio = float(np.median(lo[valid] / s[valid]))
    return ratio if np.isfinite(ratio) and ratio > 0.0 else None


def solve_ratios(frames: Sequence[np.ndarray], reference: int) -> Tuple[float, ...]:
    """Per-frame exposure relative to `frames[reference]`, which is 1.0 by definition.

    Solved pairwise between exposure-adjacent frames and chained outward from the
    reference: adjacent frames overlap most, so each fit uses the most valid samples.
    A pair that cannot be fitted falls back to the ratio of measured levels, which keeps
    one unusable frame from breaking the chain past it.
    """
    measured = [measure_exposure(f) for f in frames]
    levels = [m[0] for m in measured]
    order = sorted(range(len(frames)), key=lambda i: exposure_order_key(*measured[i]))
    ratios = [1.0] * len(frames)
    pos = order.index(reference)

    def step(a: int, b: int) -> float:
        """Ratio of frame b to frame a, a being the shorter exposure."""
        r = pair_ratio(frames[a], frames[b])
        if r is not None:
            return r
        return (levels[b] / levels[a]) if levels[a] > 1e-9 else 1.0

    for k in range(pos + 1, len(order)):  # longer than the reference
        ratios[order[k]] = ratios[order[k - 1]] * step(order[k - 1], order[k])
    for k in range(pos - 1, -1, -1):  # shorter than the reference
        ratios[order[k]] = ratios[order[k + 1]] / step(order[k], order[k + 1])
    return tuple(ratios)


def output_scale(ratios: Sequence[float], anchor: Optional[float] = None) -> float:
    """Gain that decides which exposure the merged result renders at.

    The merge is computed in the reference frame's units, and the reference is the longest
    unclipped exposure — the best choice radiometrically, but a stop or more brighter than
    the frame the photographer metered for. Rendering there stacks "brightest frame in the
    bracket" on top of the transfer curve's fixed baseline (calibrated for a typical single
    capture), and the result opens washed out: measured against the same slide's correctly
    exposed single frame, p25 came out 0.436 against 0.286. Reference and render exposure
    are separate choices, and this is the second one.

    `anchor` is the ratio of a frame the user nominated: render as *that* exposure. Which
    frame looks right is intent, and no measurement recovers it — a slide's own brightest
    point is denser than clear film, so the longest unclipped capture is not the one that
    renders as the photographer meant.

    Without one it falls back to the bracket's middle exposure, in log space so an even
    count lands between its two middle frames in EV rather than in linear light. That is
    only a guess, and a poor one on a bracket ramped *upward* from the metered frame:
    every frame is then at or above the reference, the median is >= 1, and the clamp
    below makes it inert.

    Never above 1.0: the merge's [0, 1] guarantee comes from the reference defining white,
    and an exposure *longer* than the reference would otherwise scale straight past it.
    """
    if anchor is not None and anchor > 0.0:
        return float(min(anchor, 1.0))
    r = np.asarray([x for x in ratios if x > 0.0], dtype=np.float64)
    if r.size == 0:
        return 1.0
    return float(min(np.exp(np.median(np.log(r))), 1.0))


def resolve_anchor(paths: Sequence[str], ratios: Sequence[float], config: "HdrConfig") -> Optional[float]:
    """The exposure a merge renders at, as a ratio, however the user asked for it.

    Takes the whole config rather than the fields, so a way of asking added later reaches
    every merge path at once. There are four of them — the render decode, the export decode,
    the preview service and the seeded shadow lift — and they have to agree.

    A value set in stops wins over a named frame: the slider is the finer control, and a
    frame name is what the menu writes. None means the bracket's middle exposure.
    """
    ev = float(config.hdr_anchor_ev)
    if ev < ANCHOR_EV_UNSET:
        return float(2.0**ev)
    return anchor_ratio(paths, ratios, config.hdr_anchor)


def anchor_ratio(paths: Sequence[str], ratios: Sequence[float], anchor_path: str) -> Optional[float]:
    """Ratio of the frame the merge should render at, or None for the bracket middle.

    None for an anchor that names no frame in the bracket, so an edit that outlives its
    files degrades to the default rather than raising mid-decode.
    """
    if not anchor_path:
        return None
    for path, ratio in zip(paths, ratios):
        if path == anchor_path:
            return float(ratio) if ratio > 0.0 else None
    return None


#: Shadows Density seeded per stop of recovered shadow reach, and its cap. Calibrated by
#: measuring rendered noise: a merge is only worth anything if you open the shadows far
#: enough to see the precision it bought, and the honest budget is the noise the single
#: metered frame already had. On two real brackets (~2 stops of reach) the merged render
#: stayed below that budget out to -0.6 and crossed it by -0.8 — hence 0.30/stop. The cap
#: is conservative: a wider bracket has more budget, but that is untested here.
SEED_SHADOW_PER_STOP = 0.30
SEED_SHADOW_LIMIT = 0.70


def anchor_choices(paths: Sequence[str], ratios: Sequence[float]) -> List[Tuple[str, float]]:
    """(path, stops from the reference) for every frame that can be the render exposure.

    Longest first, so the reference heads the list. A frame *longer* than the reference is
    left out: `output_scale` clamps at 1.0, so picking one renders exactly as the reference
    does. On a five-frame bracket that made four of the six menu entries land on the same
    picture. Those frames are in the bracket to buy shadows, not to be rendered at -- the
    reference is by definition the brightest exposure that does not clip, so there is no
    legitimate brighter render to ask for.
    """
    usable = sorted(((p, float(r)) for p, r in zip(paths, ratios) if 0.0 < float(r) <= 1.0), key=lambda pr: -pr[1])
    return [(path, float(np.log2(ratio))) for path, ratio in usable]


def shadow_reach_stops(ratios: Sequence[float], anchor: Optional[float] = None) -> float:
    """Stops of shadow the merge reaches below the exposure it renders at.

    The longest frame relative to the render exposure — that is the frame supplying the
    extra photons where the render frame had only noise. Zero when nothing in the bracket
    is longer than the render exposure, which is a bracket that cannot help.
    """
    r = [float(x) for x in ratios if x > 0.0]
    if not r:
        return 0.0
    return max(0.0, float(np.log2(max(r) / output_scale(ratios, anchor))))


def seed_shadow_density(ratios: Sequence[float], anchor: Optional[float] = None) -> float:
    """Shadows Density a fresh merge opens with — negative lifts (the Density convention).

    A merge of well-exposed frames recovers *precision*, not range: the tones were all
    recorded, just with few levels and much noise (a slide's density-2.9 region gets ~72 of
    65535). Precision is invisible at the same tone — cleaner noise looks like the same
    picture — so a merge left at neutral renders indistinguishable from the frame it was
    metered on, which is not what anyone merged for.

    So it opens with the shadows already lifted, by an amount derived from the reach the
    bracket actually recovered. Seeded rather than baked in: the slider shows the value, so
    it is visible and reversible, and zeroing it returns the faithful render.
    """
    stops = shadow_reach_stops(ratios, anchor)
    if stops <= 0.0:
        return 0.0
    return -min(SEED_SHADOW_PER_STOP * stops, SEED_SHADOW_LIMIT)


@parallel_njit(cache=True, fastmath=True)
def _accumulate(num: np.ndarray, den: np.ndarray, values: np.ndarray, ratio: float) -> None:
    """Add one frame to the weighted sums, in reference units. One fused pass: the numpy
    form cost more than the RAW decode it follows, mostly in full-size temporaries.

    Weight is the **square** of the exposure ratio. Converting a sample to reference units
    divides by the ratio, which divides its noise by the ratio too, so a frame's variance
    in those units goes as 1/ratio² and the inverse-variance weight goes as ratio². That
    is what makes the merge provably at least as good as the best single frame; weighting
    by the ratio itself lets a short exposure's amplified quantization noise back in, and
    measurably makes the deep shadows *worse* than not merging at all.

    The roll-off toward saturation is per channel — one channel can clip while the others
    still carry signal — and is gradual so the frame a pixel is drawn from changes
    smoothly. A hard switch prints as banding across a gradient.
    """
    inv_span = np.float32(1.0) / np.float32(SATURATION - ROLLOFF_START)
    weight = np.float32(ratio) ** 2
    scale = np.float32(max(ratio, 1e-9))
    height, width, channels = values.shape
    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                v = values[y, x, c]
                rolloff = (np.float32(SATURATION) - v) * inv_span
                if rolloff < np.float32(0.0):
                    rolloff = np.float32(0.0)
                elif rolloff > np.float32(1.0):
                    rolloff = np.float32(1.0)
                w = weight * rolloff
                num[y, x, c] += w * (v / scale)
                den[y, x, c] += w


def merge_providers(
    providers: Sequence[Callable[[], np.ndarray]],
    ratios: Sequence[float],
    reference: int = 0,
    align: bool = True,
) -> np.ndarray:
    """Merge a bracket into one float32 image in the reference frame's units, **unscaled**.

    The render exposure is deliberately *not* applied here. It is a single multiply, while
    this is seconds of decoding and accumulation, so keeping them apart lets the expensive
    part be cached once and re-used while the exposure is dragged. `apply_render_exposure`
    is the other half; every caller needs both.

    Values above 1.0 survive: they are radiance the shorter frames recorded above the
    reference's white, and clipping here would throw it away before the exposure that might
    have brought it back into range.

    Frames arrive as callables and are pulled one at a time: a full-res bracket is several
    GB, and only the reference plus the two accumulators need to stay resident.

    float32 rather than uint16: the recovered detail sits below the shortest exposure's
    quantization step, so rounding back to 16 bits would discard exactly what the merge
    was for.
    """
    if len(providers) != len(ratios):
        raise ValueError(f"{len(providers)} frames but {len(ratios)} ratios")
    if not providers:
        raise ValueError("no frames to merge")

    ref = to_float(providers[reference]())
    ref_gray = ref.mean(axis=2)
    max_shift = max(16.0, 0.02 * ref.shape[1])
    num = np.zeros_like(ref)
    den = np.zeros_like(ref)

    for i, (provider, ratio) in enumerate(zip(providers, ratios)):
        f = ref if i == reference else to_float(provider())
        if f.shape != ref.shape:
            raise ValueError(f"bracket frames differ in shape: {f.shape}, {ref.shape}")
        if align and i != reference:
            f = _align(ref_gray, f, max_shift)
        _accumulate(num, den, f, float(ratio))
        if i != reference:
            del f

    # Every frame saturated in a channel leaves no measurement there. It is a genuine
    # highlight, so take the reference's own (clipped) value rather than a hole.
    empty = den <= 0.0
    if empty.any():
        num[empty] = ref[empty]
        den[empty] = 1.0
    return np.ascontiguousarray(num / den, dtype=np.float32)


def apply_render_exposure(merged: np.ndarray, ratios: Sequence[float], anchor: Optional[float] = None) -> np.ndarray:
    """Scale an unscaled merge to the exposure it renders at, and clip to [0, 1].

    Split from the merge because it is cheap and the merge is not: one full-res multiply
    against seconds of decode. That is what lets the render exposure be dragged.
    """
    scale = np.float32(output_scale(ratios, anchor))
    return np.clip(merged * scale, 0.0, 1.0).astype(np.float32)


def _align(ref_gray: np.ndarray, mov: np.ndarray, max_shift: float) -> np.ndarray:
    """Register `mov` to the reference by sub-pixel translation; no-op on an implausible
    estimate. Translation only: the camera does not move between bracket frames, so a
    richer model would only fit noise."""
    dx, dy = estimate_shift(ref_gray, mov.mean(axis=2))
    if max(abs(dx), abs(dy)) > max_shift:
        return mov
    h, w = mov.shape[:2]
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    return cv2.warpAffine(mov, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def merge_frames(
    frames: Sequence[np.ndarray],
    ratios: Sequence[float],
    reference: int = 0,
    align: bool = True,
    anchor: Optional[float] = None,
) -> np.ndarray:
    """merge_providers over already-decoded frames. For callers holding the whole bracket
    (tests, ratio solving); the decode paths use merge_bracket so frames stay lazy."""
    merged = merge_providers([lambda f=f: f for f in frames], ratios, reference=reference, align=align)  # type: ignore[misc]
    return apply_render_exposure(merged, ratios, anchor)


def merge_bracket(decode_fn: Callable[[str], np.ndarray], reference_path: str, config: HdrConfig) -> np.ndarray:
    """Decode a bracket via `decode_fn` and merge it.

    Takes the config whole rather than its fields: the ratios, the alignment and the render
    exposure all come from it, and a field added later then reaches every caller without
    touching one of them. `config.hdr_ratios` is per frame in (reference, *hdr_paths) order.
    """
    paths = [reference_path, *config.hdr_paths]
    ratios = list(config.hdr_ratios)
    if len(ratios) != len(paths):
        raise ValueError(f"{len(paths)} frames but {len(ratios)} ratios")
    merged = merge_providers(  # type: ignore[misc]
        [lambda p=p: decode_fn(p) for p in paths], ratios, reference=0, align=config.hdr_align
    )
    return apply_render_exposure(merged, ratios, resolve_anchor(paths, ratios, config))
