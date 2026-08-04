"""OpenICE IR dust removal — Digital ICE ported from openICE (GPL-3.0).

Works in log density, where absorber densities add (Beer-Lambert): a neutral defect is a
fixed subtraction in every channel and the red dye's IR absorption is one scalar, not a
per-channel power law. § numbers track ``../openICE/docs/pipeline.md``.

Shares no code with the ``logic.py`` IR chain, deliberately — see CLAUDE.md.

Two departures from the original: the level constants are measured per frame rather than
fixed (see ``calibrate``), and the §8 dither is drawn from a coordinate hash rather than
ICE's frame-global LCG, which is the algorithm's only serial dependency.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_M = 65535.0
_K = _M / (16.0 * math.log(2.0))

# [γ, a_hi1, a_lo1, a_hi2, a_lo2, a_hi3, a_lo3] per channel, openICE kind 8 (LS-5000).
# γ converts an IR deficit to added visible density; a_* scale the IR contrast a detail
# band must beat to count as picture. ≈1 because a defect is neutral, which is why these
# transfer between scanners where the level constants below do not. Kinds 7/9 differ only
# here (openICE pipeline.md Appendix B); not exposed, this set covers both.
_COEF = np.array(
    [
        [1.10, 1.21, 1.09, 1.17, 1.08, 1.04, 0.96],
        [1.10, 1.23, 1.13, 1.14, 1.05, 0.93, 0.84],
        [1.10, 1.13, 1.04, 1.08, 1.02, 0.97, 0.89],
    ],
    dtype=np.float32,
)
_STAGE_GAIN = 1.25  # ICE Normal. Fine (1.0, no output floor) rewrites the whole frame.
# ICE's gate bias θ (kind 8). Cancels inside the weight, so it only keeps the gate
# numerically equal to the reference for the pyrdiff harness.
_GATE_BIAS = 1.0
_WEIGHT_FLOOR = 0.02
_CONF_GAIN_B0 = 2.0  # the coarse band reaches full trust at half confidence
_WEIGHT_BIAS = float(_K * math.log1p(math.floor(0.98 * _M)) - _M)  # D(0.98M) − D(M) = −119.4
_ICE_RAMP = float(_M - _K * math.log1p(math.floor(0.85 * _M)))  # ICE's fixed ramp, 15% light loss

# --- the scanner bridge -------------------------------------------------------------
# ICE's level constants are absolute, tuned to a Coolscan under Nikon Scan, and do not
# transfer: on a SilverFast LS-9000 scan every pixel clears its clear-film gate and dust
# never reaches its ramp. Measured per frame instead, in σ of the gate distribution.
# Anchored on σ, not on observed dip depth — σ is a property of the scanner, dip depth of
# how dirty this frame is, and a clean frame would collapse the ramp onto its own noise.
# ICE's fixed 960.4 measures 11.2σ on that LS-9000 and 5.6σ on a 4800 dpi flatbed, which
# is the range the slider spans.
_RAMP_SIGMA_LO = 2.0
_RAMP_SIGMA_HI = 11.0
# Without a σ-scaled margin, half of a noisy clean frame lands on the ramp and mottles.
_MARGIN_SIGMA = 2.5
_RAMP_MIN = 180.0  # ≈3% light loss; a quantized plane can measure σ ≈ 0
_RAMP_MAX = 4100.0  # ≈50%
_CLEAR_SIGMA = 3.0  # calibration sees only pixels this close to the median
# ICE's Cfg_DustFloor is the absolute D(0.065·M). Carried over as the transmittance
# fraction it encodes, measured against this frame's clear-film reference — the same
# substitution the gate and ramp above need, for the same reason. Anchoring it to σ
# instead put it ~10x shallower, which fires the give-up trigger on ordinary deep dust
# and routes it away from the reconstruction that should have repaired it.
_DUST_FLOOR_TRANSMITTANCE = 0.065
_DUST_FLOOR_DROP = float(_K * math.log(1.0 / _DUST_FLOOR_TRANSMITTANCE))
_DEAD_FLOOR = 0.05  # below this the beam is blocked outright: holder, not film
# |pearson(gate, green density)| above this: the IR mirrors the picture (B&W, Kodachrome).
# Dust is uncorrelated with the image; a silver image is not.
_DEGENERATE_CORR = 0.7
# ICE calibrates on a ~281 px prescan. Kept: at detection scale the per-tile dye deviations
# shrink and the δIR/δR slope goes to noise.
_CALIB_WIDTH = 281
_CALIB_MIN_PIXELS = 4096
_CALIB_ROWS = 256  # whole rows, at working resolution — striding would break the 3-tap min

_HALO = 8  # pyramid reach (4) + the band-range cross (1); bands overlap by this

# --- §8 dither ----------------------------------------------------------------------
# ICE's synthetic grain. Not optional decoration: the ℓ=3 band restores real grain from the
# pixel itself, and a pixel fully covered by dust has none left, so without this the repair
# comes back glassy against the film around it (issue #732). Amplitudes and band anchors are
# ICE's own and do transfer — unlike the level constants above, they scale with the density
# of the pixel being written, not with the scanner's absolute IR level.
_DITHER_AMP = np.array([0.015, 0.015, 0.025], dtype=np.float32)  # Cfg_DitherAmt{R,G,B}
_DITHER_LO = float(_K * math.log1p(math.floor(0.01 * _M)))
_DITHER_HI = float(_K * math.log1p(math.floor(0.99 * _M)))
_DITHER_ENV = 4.0 / (_DITHER_HI - _DITHER_LO) ** 2  # parabola peaking at 1 mid-band


def _min3(a: np.ndarray) -> np.ndarray:
    """Horizontal 3-tap minimum, as ICE applies to the gate above 550 dpi — film scans
    always qualify. Widens a defect's reach by a pixel so a sub-pixel hairline still
    drives the weight. Horizontal only, which is how the original does it."""
    out = np.minimum(a, np.minimum(np.roll(a, 1, 1), np.roll(a, -1, 1)))
    out[:, 0] = a[:, 0]
    out[:, -1] = a[:, -1]
    return out


def _octagon(radius: int, limit: int) -> np.ndarray:
    a = np.arange(-radius, radius + 1)
    di, dj = np.meshgrid(a, a, indexing="ij")
    k = ((np.abs(di) + np.abs(dj)) <= limit).astype(np.float32)
    return k / float(k.sum())


# The four normalized-convolution scales. Unit-weight octagons at 9×9 (69 cells) and
# 5×5 (21), a 1-2-1 binomial tent at 3×3 (16); level 3 is the pixel itself.
_K9 = _octagon(4, 6)
_K5 = _octagon(2, 3)
_K3 = np.outer([1.0, 2.0, 1.0], [1.0, 2.0, 1.0]).astype(np.float32) / 16.0
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


@dataclass(frozen=True)
class IceCalibration:
    """Per-frame scalars held constant across the reconstruction (§2 + the bridge)."""

    crosstalk: float
    ir_ref: float
    ramp: float
    margin: float
    dust_floor: float
    degenerate: bool


def density(v: np.ndarray) -> np.ndarray:
    """Linear [0,1] → log density [0, 65535]. ``D(v) = (M/16·ln2)·ln(v·M + 1)``."""
    return (_K * np.log1p(np.clip(v, 0.0, 1.0) * _M)).astype(np.float32)


def density_inv(d: np.ndarray) -> np.ndarray:
    """Log density → linear [0,1]; exact inverse of :func:`density`."""
    return (np.expm1(np.clip(d, 0.0, _M) / _K) / _M).astype(np.float32)


def _filter(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convolve with zero fill — zero-weight pads drop out of a normalized convolution."""
    return cv2.filter2D(a, -1, k, borderType=cv2.BORDER_CONSTANT)


def _crosstalk(d_r: np.ndarray, d_ir: np.ndarray, ir: np.ndarray, clear: np.ndarray) -> float:
    """Dye→IR crosstalk ``c`` from 8×8 tiles that are *entirely* clear film (§2).

    Within such a tile any variation must be dye, not dust, so the slope of IR deviation
    on red deviation across the four 4×4 quadrants measures the leak directly.
    """
    h, w = clear.shape
    nrow, ncol = h // 8, w // 8
    if nrow < 1 or ncol < 1:
        return 0.0

    def tiles(a: np.ndarray) -> np.ndarray:
        return a[: nrow * 8, : ncol * 8].reshape(nrow, 8, ncol, 8)

    def quads(a: np.ndarray) -> np.ndarray:
        return tiles(a).reshape(nrow, 2, 4, ncol, 2, 4).mean(axis=(2, 5))

    all_clear = np.all(tiles(clear), axis=(1, 3))
    if not all_clear.any():
        return 0.0
    q_r, q_ir = quads(d_r.astype(np.float64)), quads(d_ir.astype(np.float64))
    dev_r = q_r - q_r.mean(axis=(1, 3), keepdims=True)
    dev_ir = q_ir - q_ir.mean(axis=(1, 3), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dev_ir / dev_r
    keep = all_clear[:, None, :, None] & np.isfinite(ratio) & (np.abs(ratio) <= 0.2)
    # A brighter tile is a more reliable slope. The scale cancels in the weighted mean.
    tile_w = tiles(ir.astype(np.float64)).sum(axis=(1, 3))[:, None, :, None]
    term = np.where(keep, dev_r * dev_r, 0.0) * tile_w * tile_w
    denom = float(term.sum())
    if denom <= 0.0:
        return 0.0
    return float((np.where(keep, ratio, 0.0) * term).sum() / denom)


def calibrate(rgb: np.ndarray, ir: np.ndarray, threshold: float) -> IceCalibration:
    """Per-frame crosstalk, clear-film IR reference, ramp and floor.

    ``threshold`` is the IR Threshold slider (0..1, higher = conservative), biasing the ramp.
    """
    full_ir = np.ascontiguousarray(ir if ir.ndim == 2 else ir[:, :, 0], dtype=np.float32)
    full_rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    if full_ir[full_ir >= _DEAD_FLOOR].size < _CALIB_MIN_PIXELS:
        logger.warning("OpenICE: too little live IR — correction skipped")
        return IceCalibration(0.0, 0.0, _ICE_RAMP, _WEIGHT_BIAS, 0.0, True)

    h, w = full_ir.shape[:2]
    ir, rgb = full_ir, full_rgb
    if w > _CALIB_WIDTH:
        dims = (_CALIB_WIDTH, max(1, round(h * _CALIB_WIDTH / w)))
        # Area mean, not the min-preserving detection downsample: the calibration wants
        # dust averaged away, so its clear-film statistics describe film.
        ir = cv2.resize(full_ir, dims, interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(full_rgb, dims, interpolation=cv2.INTER_AREA)

    live = ir >= _DEAD_FLOOR
    sample = ir[live]
    if sample.size < 64:
        return IceCalibration(0.0, 0.0, _ICE_RAMP, _WEIGHT_BIAS, 0.0, True)
    med = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - med)))
    d_rgb, d_ir = density(rgb), density(ir)
    clear = live & (ir >= med - _CLEAR_SIGMA * sigma)

    c = _crosstalk(d_rgb[:, :, 0], d_ir, ir, clear)
    ir_sq = np.where(clear, ir.astype(np.float64) ** 2, 0.0)
    total = float(ir_sq.sum())
    if total <= 0.0:
        return IceCalibration(0.0, 0.0, _ICE_RAMP, _WEIGHT_BIAS, 0.0, True)
    # Inverse-variance (IR²) weighting leans both references toward the clearest film.
    r_ref = float((ir_sq * d_rgb[:, :, 0]).sum() / total)
    ir_raw = float((ir_sq * d_ir).sum() / total)
    ir_ref = (ir_raw - c * r_ref) / (1.0 - c)

    degenerate = _correlates_with_image((d_ir - c * d_rgb[:, :, 0]) / (1.0 - c), d_rgb[:, :, 1], live)
    if degenerate:
        logger.info("OpenICE: IR plane tracks the image (B&W/Kodachrome) — correction skipped")

    g_med, g_sigma = _gate_stats(full_rgb, full_ir, c)
    n = _RAMP_SIGMA_LO + float(np.clip(threshold, 0.0, 1.0)) * (_RAMP_SIGMA_HI - _RAMP_SIGMA_LO)
    ramp = float(np.clip(n * g_sigma, _RAMP_MIN, _RAMP_MAX))
    # `g_med - ir_ref` absorbs the systematic offset the 3-tap min puts between the two.
    margin = min(_WEIGHT_BIAS, g_med - ir_ref - _MARGIN_SIGMA * g_sigma)
    dust_floor = ir_ref - _DUST_FLOOR_DROP
    return IceCalibration(c, ir_ref, ramp, margin, dust_floor, degenerate)


def _gate_stats(rgb: np.ndarray, ir: np.ndarray, c: float) -> Tuple[float, float]:
    """Clean-film median and MAD σ of the gate that feeds the weight, in density units.

    Measured on the gate, not inferred from the IR plane: the crosstalk subtraction, the
    3-tap min and the working resolution each move the distribution.
    """
    h = ir.shape[0]
    rows = np.linspace(0, h - 1, min(h, _CALIB_ROWS)).astype(np.intp)
    d_ir = density(ir[rows])
    d_r = density(rgb[rows, :, 0])
    gate = _min3((d_ir - c * d_r) / (1.0 - c) - _GATE_BIAS)
    sample = gate[ir[rows] >= _DEAD_FLOOR]
    if sample.size < 64:
        return 0.0, _ICE_RAMP
    med = float(np.median(sample))
    return med, max(1.4826 * float(np.median(np.abs(sample - med))), 1e-3)


def _correlates_with_image(gate: np.ndarray, d_g: np.ndarray, live: np.ndarray) -> bool:
    x, y = gate[live].astype(np.float64), d_g[live].astype(np.float64)
    if x.size < _CALIB_MIN_PIXELS:
        return False
    xs, ys = x.std(), y.std()
    if xs < 1e-6 or ys < 1e-6:
        return False
    return abs(float(((x - x.mean()) * (y - y.mean())).mean() / (xs * ys))) > _DEGENERATE_CORR


def _gate_and_weight(d_rgb: np.ndarray, d_ir: np.ndarray, live: np.ndarray, cal: IceCalibration) -> Tuple[np.ndarray, np.ndarray]:
    """IR gate ``g`` (§3) and clean-confidence weight ``w`` (§4).

    ``g`` sits at ``IR_ref`` on clear film and drops by the light a defect stole; ``w``
    maps that to 1 = intact, floor = fully occluded.
    """
    gate = ((d_ir - cal.crosstalk * d_rgb[:, :, 0]) / (1.0 - cal.crosstalk) - _GATE_BIAS).astype(np.float32)
    # Lifting the holder to clear-film level, not just forcing its weight, keeps it out of
    # the give-up trigger (a strip margin would swallow the whole inpaint budget) and out
    # of the pyramid at the frame edge.
    gate[~live] = cal.ir_ref
    w = 1.0 - (cal.ir_ref + cal.margin - _min3(gate)) / cal.ramp
    w = np.clip(w, _WEIGHT_FLOOR, 1.0).astype(np.float32)
    w[~live] = 1.0
    return gate, w


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate by (dy, dx), replicating the edge rather than wrapping."""
    out = np.roll(a, (dy, dx), axis=(0, 1))
    if dy > 0:
        out[:dy] = out[dy : dy + 1]
    elif dy < 0:
        out[dy:] = out[dy - 1 : dy]
    if dx > 0:
        out[:, :dx] = out[:, dx : dx + 1]
    elif dx < 0:
        out[:, dx:] = out[:, dx - 1 : dx]
    return out


def _giveup_trigger(gate: np.ndarray, floor: float) -> np.ndarray:
    """Pixels whose defect is wider than the reconstruction window (§5).

    Four 9-sample probes on the perimeter of the 9×9 box, the vertical edges at ±4 columns
    and the horizontal ones at ±4 rows. Any probe entirely below the dust floor means the
    defect continues past the window, so there is no intact film to rebuild from.
    """
    below = (gate < floor).astype(np.uint8)
    # Eroding by a 9-long line = "all nine below". Replicate, or the frame edge reads as an
    # all-dust probe (openICE edge-replicates its gate history for this).
    col = cv2.erode(below, np.ones((9, 1), np.uint8), borderType=cv2.BORDER_REPLICATE)
    row = cv2.erode(below, np.ones((1, 9), np.uint8), borderType=cv2.BORDER_REPLICATE)
    hit = _shift(col, 0, 4) | _shift(col, 0, -4) | _shift(row, 4, 0) | _shift(row, -4, 0)
    return hit.astype(bool)


def _normconv(num: np.ndarray, conf: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Local average where each sample carries its own reliability, so defect pixels drop
    out and the hole fills from intact neighbours."""
    if num.ndim == 3:
        return _filter(num, k) / conf[..., None]
    return _filter(num, k) / conf


def _reconstruct_tile(d_rgb: np.ndarray, gate: np.ndarray, w: np.ndarray, cal: IceCalibration) -> np.ndarray:
    """Reconstruction core (§6-§7): start from the coarsest confidence-weighted average
    with the stolen light added back, then restore each detail band that beats the local
    IR contrast at its own scale."""
    conf = [np.maximum(_filter(w, k), 1e-6) for k in (_K9, _K5, _K3)]
    p = [_normconv(w * gate, c, k) for c, k in zip(conf, (_K9, _K5, _K3))] + [gate]
    wd = d_rgb * w[..., None]

    lo_prev = _normconv(wd, conf[0], _K9)
    # §7a: the local IR deficit is the light the defect stole; γ converts it to density.
    acc = lo_prev + _COEF[:, 0] * (cal.ir_ref - p[0])[..., None]

    band_conf = (
        np.clip(_CONF_GAIN_B0 * conf[1], 0.0, 1.0),
        np.maximum(conf[2], 0.0),
        w * w,
    )
    for b in range(3):
        l_cur = d_rgb if b == 2 else _normconv(wd, conf[b + 1], (_K5, _K3)[b])
        detail = (l_cur - lo_prev) * _STAGE_GAIN
        lo_prev = l_cur
        # §7b: the IR's own contrast at this scale is what dust would look like here.
        delta = p[b + 1] - p[b]
        hi, lo = cv2.dilate(delta, _CROSS), cv2.erode(delta, _CROSS)
        a_hi, a_lo = _COEF[:, 1 + 2 * b], _COEF[:, 2 + 2 * b]
        # On an entirely-negative band the coefficients swap, so the larger one always
        # scales whichever edge sits further from zero.
        neg = ((hi < 0.0) & (lo < 0.0))[..., None]
        hi_t = np.where(neg, a_lo, a_hi) * hi[..., None]
        lo_t = np.where(neg, a_hi, a_lo) * lo[..., None]
        # Dead zone: detail within the IR contrast is dust, detail beyond it is picture.
        resid = np.where(detail > hi_t, detail - hi_t, np.where(detail < lo_t, detail - lo_t, 0.0))
        acc += resid * band_conf[b][..., None]
    return acc


def _uniform(row0: int, shape: Tuple[int, int]) -> np.ndarray:
    """Per-pixel, per-channel draw in [-0.5, 0.5), hashed from absolute image coordinates.

    ICE advances one frame-global LCG per draw, which serializes the whole reconstruction;
    hashing the coordinate instead also keeps a pixel's grain independent of which band it
    lands in, so the banded and single-pass paths stay identical.
    """
    h, w = shape
    iy = (np.arange(row0, row0 + h, dtype=np.uint32) * np.uint32(0x165667B1))[:, None, None]
    ix = (np.arange(w, dtype=np.uint32) * np.uint32(0x27D4EB2D))[None, :, None]
    ic = (np.arange(3, dtype=np.uint32) * np.uint32(0x9E3779B9))[None, None, :]
    k = iy ^ ix ^ ic
    k ^= k >> np.uint32(15)
    k *= np.uint32(0x2C1B3C6D)
    k ^= k >> np.uint32(13)
    k *= np.uint32(0x297A2D39)
    k ^= k >> np.uint32(16)
    return (k >> np.uint32(8)).astype(np.float32) * np.float32(2.0**-24) - np.float32(0.5)


def _dither(acc: np.ndarray, row0: int) -> np.ndarray:
    """§8: zero-mean grain, parabolic across the density band and zero outside it, scaled by
    the reconstructed density itself. Suppressed unless ``acc + dither`` also stays in band,
    as ICE does; ICE's second draw for the never-darken comparison is not reproduced (its
    own docs call that a bug — it can store a value below the scan)."""
    env = _DITHER_ENV * (_DITHER_HI - acc) * (acc - _DITHER_LO)
    d = env * _uniform(row0, acc.shape[:2]) * (_DITHER_AMP * acc)
    in_band = (acc > _DITHER_LO) & (acc < _DITHER_HI)
    return np.where(in_band & (acc + d > _DITHER_LO) & (acc + d < _DITHER_HI), d, 0.0).astype(np.float32)


_BAND_ROWS = 256


def reconstruct(img: np.ndarray, ir: np.ndarray, cal: IceCalibration) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean the linear source, at the buffer's own resolution.

    ``trigger`` marks defects too wide to rebuild (for the inpaint router); ``weight`` is 1
    on intact film and drops under a defect (for the overlay). Banded because the pyramid
    holds a dozen full planes live at once; bands overlap by _HALO so seams carry nothing.
    """
    src = np.ascontiguousarray(img, dtype=np.float32)
    h, w_px = src.shape[:2]
    ir = np.ascontiguousarray(ir, dtype=np.float32)
    if ir.ndim == 3:
        ir = ir[:, :, 0]
    if ir.shape != (h, w_px):
        ir = cv2.resize(ir, (w_px, h), interpolation=cv2.INTER_LINEAR)

    out = np.empty_like(src)
    trigger = np.empty((h, w_px), dtype=bool)
    weight = np.empty((h, w_px), dtype=np.float32)
    for y0 in range(0, h, _BAND_ROWS):
        y1 = min(h, y0 + _BAND_ROWS)
        a, b = max(0, y0 - _HALO), min(h, y1 + _HALO)
        ir_t = ir[a:b]
        d_rgb = density(src[a:b])
        gate, w = _gate_and_weight(d_rgb, density(ir_t), ir_t >= _DEAD_FLOOR, cal)
        hopeless = _giveup_trigger(gate, cal.dust_floor)
        acc = _reconstruct_tile(d_rgb, gate, w, cal)
        # Already clean, wider than the window, or a non-positive reconstruction. Tested on
        # the bare accumulator: ICE's positivity guard runs before the dither is added.
        keep = (w >= 1.0) | hopeless | (acc <= 0.0).any(axis=-1)
        acc = acc + _dither(acc, a)
        # "Only fill, never darken": a defect steals light, so a repair may only lighten.
        # Written at full strength wherever w < 1, as ICE does (max(L3, acc + dither)). A
        # confidence ramp here looks like cheap insurance against mottling and is not: dust
        # sits at the weight floor only once it is deep, so the ramp cost shallow specks most of their
        # repair (measured on samples/scans: 81% of the lift at a 15-25% dip).
        lift = np.where(keep[..., None], 0.0, np.maximum(acc - d_rgb, 0.0))
        s, e = y0 - a, y1 - a
        # Verbatim where nothing was lifted: D/D⁻¹ is not bit-exact in float32, and an
        # untouched pixel must come out byte-identical.
        touched = lift[s:e] > 0.0
        out[y0:y1] = np.where(touched, density_inv(d_rgb[s:e] + lift[s:e]), src[y0:y1])
        trigger[y0:y1] = hopeless[s:e]
        weight[y0:y1] = w[s:e]
    return out, trigger, weight


_ROUTE_DILATE = 2
_ROUTE_BUDGET = 0.02  # fraction of the frame
# No single defect covers this much. The film rebate does: above the dead floor on some
# scanners, it arrives as one component of 1.2% of a flatbed frame (two full-height strips
# down both edges) and blows the budget, taking the real dust with it.
_ROUTE_MAX_COMPONENT = 0.002


def route(trigger: np.ndarray) -> Optional[np.ndarray]:
    """Given-up defects as an inpaint mask, or ``None`` when there are none or too many.

    ICE copies these through untouched; NegPy sends them to its own inpaint. Over budget
    means a misregistered or garbage IR plane, not a filthy frame.
    """
    if not trigger.any():
        return None
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(trigger.astype(np.uint8), connectivity=8)
    cap = _ROUTE_MAX_COMPONENT * trigger.size
    keep = np.zeros(n_lbl, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] <= cap
    if not keep.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _ROUTE_DILATE + 1,) * 2)
    mask = cv2.dilate(keep[labels].astype(np.uint8), k)
    frac = float(mask.mean())
    if frac > _ROUTE_BUDGET:
        logger.warning("OpenICE: unreconstructable defects cover %.1f%% of the frame — inpaint skipped", frac * 100.0)
        return None
    return mask


def _to_detect(mask: np.ndarray, dims: Optional[Tuple[int, int]], keep: float) -> np.ndarray:
    """Boolean mask down to the detection scale, kept where coverage reaches ``keep``."""
    if dims is None or mask.shape[::-1] == dims:
        return mask
    return cv2.resize(mask.astype(np.float32), dims, interpolation=cv2.INTER_AREA) >= keep


def run(
    img: np.ndarray,
    ir: np.ndarray,
    threshold: float,
    detect_dims: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], bool, Optional[np.ndarray]]:
    """Clean ``img`` (linear source transmittance) using ``ir``.

    Returns the same shape ``_ir_bake`` does: ``(image, corrected_mask, degenerate,
    routed_mask)``, the two masks at ``detect_dims`` for the overlay and the inpaint pass.
    """
    cal = calibrate(img, ir, threshold)
    if cal.degenerate:
        return img, None, True, None
    out, trigger, weight = reconstruct(img, ir, cal)
    corrected = _to_detect(weight < 1.0, detect_dims, 0.001)
    return out, corrected, False, route(_to_detect(trigger, detect_dims, 0.5))
