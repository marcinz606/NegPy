import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R, ImageBuffer
from negpy.features.geometry.logic import map_coords_to_geometry, smooth_polyline
from negpy.features.retouch.models import HEAL_SIZE_REF
from negpy.kernel.image.logic import get_luminance, working_oetf_decode, working_oetf_encode
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# Golden-angle fallback used when a heal has no scored source offset
# (legacy spots, or no preview buffer at click time).
_GOLDEN_ANGLE = 2.39996322972865332
_FALLBACK_OFFSET_FACTOR = 2.6
# Clone-sample dust guard: a sample whose luma exceeds its 3×3 luma-median
# neighbour by this much is treated as dust and replaced by the median pixel,
# so dust in the source patch is never recloned. Mirrored in retouch.wgsl.
_CLONE_GUARD_LUMA = 0.06
# Destination dust gate: a brushed pixel is healed only when its luma exceeds
# the membrane-predicted clean value by this ramp (encoded domain) — the brush
# marks a search area, only the bright dust inside it gets replaced.
_HEAL_GATE_LO = 0.04
_HEAL_GATE_HI = 0.12
# Spread floor: stops noise on low-contrast sources (fog, flat frames) from
# being amplified to full range; dust sits ≥ ~1 density unit above surroundings.
_PROXY_MIN_SPREAD = 0.8
# Pad heals past the detected bright core — an unhealed soft skirt reads as a halo.
_DETECT_PAD_PX = 2.5
# Membrane boundary ring sits this far outside the blend radius (preview-scale px):
# a ring on the defect's PSF skirt biases every boundary diff bright and the whole
# clone renders as a soft ghost. The blend footprint stays at the blend radius.
_MEMBRANE_RIM_PX = 2.0
# Rim feather fraction of the blend radius (1.5px floor). Ungated auto-luma regions
# widen it via the gate lane. Mirrored in retouch.wgsl.
_RIM_FEATHER_FRAC = 0.25
_RIM_FEATHER_UNGATED = 0.15

# IR ratio-normalization base window (px at detection scale, pinned like HEAL_SIZE_REF).
# Defects wider than ~half of it depress their own base (max-area/Scratch territory).
_IR_BASE_WIN = 25
_IR_GAIN_IDENTITY = 0.97  # gain is identity at/above this ratio
_IR_GAIN_CLAMP = 2.0  # caps misregistration halos
# Per-channel refraction γ, fitted per frame (patent 1.03–1.10 under-correct file IR).
_IR_GAMMA_LO = 1.0
_IR_GAMMA_HI = 2.2
_IR_GAMMA_FALLBACK = 1.5
# Below this the beam is blocked outright — holder, not film. Coolscan rolls: margin 98% under
# it, in-frame dust bottoms at 0.17. ponytail: absolute; a low-IR-gain scanner would want a percentile.
_IR_DEAD_FLOOR = 0.05
# Crosstalk unmixing: dye/silver absorbs some IR, so the IR plane carries a ghost of
# the image that normalize_ir's spatial high-pass can't see (a sharp edge survives it).
_IR_XTALK_MAX = 0.8  # per-channel exponent cap; ≥0 only — density can only block IR
_IR_XTALK_MIN = 0.02  # |b| sum below this is a noise-level fit → exact no-op
_IR_DEGENERATE_GHOST = 0.5  # fitted exponent sum above this: IR mirrors the image (B&W/Kodachrome)
_IR_XTALK_TRIM = 5.0  # fit drops this bottom-ratio percentile (the dust minority)
# γ fit sample: keep this flattest fraction of the band by visible Laplacian, dropping the
# restriction below _IR_FIT_MIN_PX rather than fitting a handful of pixels. See _fit_refraction_gammas.
_IR_FIT_FLAT_PCT = 40
_IR_FIT_MIN_PX = 200
# Clean-base cap window (detection-scale px, odd). The bake may never lift a pixel above its
# own local clean base — past that it invents signal rather than recovering it. Needed because
# downsample_ir is min-preserving while the visible arrives area-averaged, so at detection scale
# the ratio's dip runs deeper and ~1 px wider than the defect the visible carries (0.816 against
# 0.892); uncapped, that skirt lifts clean film and every speck and hair renders with a dark
# outline. Reaches ±4 px, past _DETECT_PAD_PX's skirt. Base = defect-excluded local mean −
# _IR_CAP_SIGMA·σ, not blur(dilate): the dilate is a local max, ~2σ of grain high, and re-admits
# the ring on grainy film (#563). Under _IR_CAP_MIN_SUPPORT clean pixels in the window → the max
# estimate returns (deep inside wide defects, where _IR_GAIN_CLAMP binds first).
_IR_CAP_WIN = 9
_IR_CAP_SIGMA = 1.0
_IR_CAP_MIN_SUPPORT = 0.1  # fraction of the window (~8 px at 9×9)

# IR reconstruction: concepts ported from digital-fauxice (MIT, © 2026 Rohan
# Pandula, see NOTICE.md) — continuous score, score-weighted fill, original-floor rule.
# Score: 1 = clean (ratio ≥ _IR_GAIN_IDENTITY), floor at/below the slider's cutoff.
# Never thresholded — no mask edge to halo, no coverage fraction to abort on.
_IR_SCORE_FLOOR = 0.02
# Fill supports (detection-scale px, × the buffer's upsample factor). Candidate per
# support: Σ(rgb·score·win)/Σ(score·win) — low-score neighbours self-exclude. A finer
# support wins once its clean fraction reaches _IR_FILL_TAU (edges continue through).
_IR_FILL_SCALES = (9, 5, 3)
_IR_FILL_TAU = 0.15
# Write ramp: untouched above HI (grain survives), full fill at/below LO.
_IR_WRITE_HI = 0.85
_IR_WRITE_LO = 0.40
# Route to inpaint only components with a core the fill can't see across (chebyshev
# radius ≥ 5 ⇔ solid 9×9 interior). Thin hairs stay with the fill: every pixel is
# within reach of clean film, and NS inpaint would smear structure the fill keeps.
# The budget bounds only this heavy path; the fill always runs.
_IR_ROUTE_RADIUS = 5
_IR_ROUTE_DILATE = 2
_IR_ROUTE_BUDGET = 0.02  # fraction of the frame

# Strong hairs/scratches (auto-luma) route to structure-following inpaint instead of
# the membrane clone: a long twist crosses varied background, and one clone-source
# offset can't match it. Bar sits above the mild-elongation capsule test so ordinary
# dust clusters stay on the membrane. Detection-scale px. See _is_hair.
_HAIR_MIN_AREA = 20
_HAIR_MIN_ELONG = 8.0  # area/thickness² ≈ length/thickness; round specks measure 1–3
# cv2.inpaint fill: dilate covers the PSF skirt at 1:1 (apply_hair_inpaint widens it to
# track the mask's upsample); NS radius; gamma gives the 8-bit encode a perceptual
# spread (cv2.inpaint is 8-bit only). Navier-Stokes only propagates outward from the mask
# boundary, so each defect is filled in its own bbox + _HAIR_INPAINT_PAD (>= the radius):
# same pixels, without gamma-encoding the whole frame to serve a hairline.
_HAIR_DILATE_PX = 1
_HAIR_INPAINT_RADIUS = 3
_HAIR_INPAINT_GAMMA = 2.2
_HAIR_INPAINT_PAD = 16


@njit(cache=True, fastmath=True)
def _dist_to_chain(px: float, py: float, pts: np.ndarray) -> float:
    """Distance from (px, py) to the polyline ``pts`` ((M, 2) pixel coords)."""
    m = pts.shape[0]
    if m == 1:
        dx = px - pts[0, 0]
        dy = py - pts[0, 1]
        return math.sqrt(dx * dx + dy * dy)
    best = 1e18
    for s in range(m - 1):
        ax, ay = pts[s, 0], pts[s, 1]
        bx, by = pts[s + 1, 0], pts[s + 1, 1]
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            t = 0.0
        else:
            t = ((px - ax) * abx + (py - ay) * aby) / ab2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        cx = ax + t * abx
        cy = ay + t * aby
        dx = px - cx
        dy = py - cy
        d = math.sqrt(dx * dx + dy * dy)
        if d < best:
            best = d
    return best


@njit(cache=True, fastmath=True)
def _sample_clean_jit(img: np.ndarray, ix: int, iy: int, out: np.ndarray) -> None:
    """Dust-guarded clone sample: the pixel at (ix, iy), or its 3×3 luma-median
    neighbour when the pixel is a strong bright outlier (a dust speck).

    Keeps grain (a real neighbouring pixel is returned, never an average).
    Ceiling: specks wider than ~2px fill the 3×3 window and pass through —
    the source-scoring penalty in select_source_offset avoids those upfront.
    """
    h, w, _ = img.shape
    lums = np.empty(9, dtype=np.float64)
    sxs = np.empty(9, dtype=np.int64)
    sys_ = np.empty(9, dtype=np.int64)
    n = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            sx = max(0, min(w - 1, ix + dx))
            sy = max(0, min(h - 1, iy + dy))
            lums[n] = LUMA_R * img[sy, sx, 0] + LUMA_G * img[sy, sx, 1] + LUMA_B * img[sy, sx, 2]
            sxs[n] = sx
            sys_[n] = sy
            n += 1

    order = np.argsort(lums)
    mi = order[4]
    lv = LUMA_R * img[iy, ix, 0] + LUMA_G * img[iy, ix, 1] + LUMA_B * img[iy, ix, 2]
    if lv - lums[mi] > _CLONE_GUARD_LUMA:
        out[0] = img[sys_[mi], sxs[mi], 0]
        out[1] = img[sys_[mi], sxs[mi], 1]
        out[2] = img[sys_[mi], sxs[mi], 2]
    else:
        out[0] = img[iy, ix, 0]
        out[1] = img[iy, ix, 1]
        out[2] = img[iy, ix, 2]


@njit(cache=True, fastmath=True)
def _sample_clean5_jit(img: np.ndarray, ix: int, iy: int, out: np.ndarray) -> None:
    """5×5 variant of `_sample_clean_jit` for the directly-cloned source sample —
    catches specks up to ~4px that slip through the 3×3 window."""
    h, w, _ = img.shape
    lums = np.empty(25, dtype=np.float64)
    sxs = np.empty(25, dtype=np.int64)
    sys_ = np.empty(25, dtype=np.int64)
    n = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sx = max(0, min(w - 1, ix + dx))
            sy = max(0, min(h - 1, iy + dy))
            lums[n] = LUMA_R * img[sy, sx, 0] + LUMA_G * img[sy, sx, 1] + LUMA_B * img[sy, sx, 2]
            sxs[n] = sx
            sys_[n] = sy
            n += 1

    order = np.argsort(lums)
    mi = order[12]
    lv = LUMA_R * img[iy, ix, 0] + LUMA_G * img[iy, ix, 1] + LUMA_B * img[iy, ix, 2]
    if lv - lums[mi] > _CLONE_GUARD_LUMA:
        out[0] = img[sys_[mi], sxs[mi], 0]
        out[1] = img[sys_[mi], sxs[mi], 1]
        out[2] = img[sys_[mi], sxs[mi], 2]
    else:
        out[0] = img[iy, ix, 0]
        out[1] = img[iy, ix, 1]
        out[2] = img[iy, ix, 2]


@njit(cache=True, fastmath=True)
def _membrane_heal_jit(
    buf: np.ndarray,
    reg_i: np.ndarray,
    reg_f: np.ndarray,
    pts: np.ndarray,
) -> None:
    """Mean-value-coordinates membrane clone (Georgiev healing brush), in place.

    ``reg_i``: (R, 4) int32 — pt_start, pt_count, bnd_start, bnd_count into ``pts``.
    ``reg_f``: (R, 4) float32 — radius_px, src_off_x, src_off_y (pixels), gate
    (1 = bright-only dust gate, 0 = unconditional clone).
    ``pts``: (P, 2) float32 pixel coords (continuous, +0.5 = pixel center).

    out(p) = img(p + off) + Σ ŵ_i (img(b_i) − img(b_i + off)) — the copied
    source patch carries real grain; the MVC-weighted boundary-difference field
    is the smooth membrane that matches the destination at the rim. All clone
    samples go through the `_sample_clean_jit` dust guard so specks in the
    source patch or on the boundary are never recloned, and a destination
    dust gate limits the heal to pixels brighter than the membrane-predicted
    clean value — the brush marks a search area, clean pixels stay untouched.
    Heal values sample the immutable stage input (matching the GPU's
    single-pass ``input_tex`` reads); only the blend base evolves in ``buf``.
    """
    img = buf.copy()
    h, w, _ = buf.shape
    n_reg = reg_i.shape[0]
    diffs = np.empty((64, 3), dtype=np.float32)
    tans = np.empty(64, dtype=np.float64)
    vlen = np.empty(64, dtype=np.float64)
    vx = np.empty(64, dtype=np.float64)
    vy = np.empty(64, dtype=np.float64)
    smp_a = np.empty(3, dtype=np.float32)
    smp_b = np.empty(3, dtype=np.float32)

    for r in range(n_reg):
        ps, pc, bs, bc = reg_i[r, 0], reg_i[r, 1], reg_i[r, 2], reg_i[r, 3]
        rad = reg_f[r, 0]
        ox = reg_f[r, 1]
        oy = reg_f[r, 2]
        gate = reg_f[r, 3]
        if bc < 3 or bc > 64 or pc < 1:
            continue

        for i in range(bc):
            bxf = pts[bs + i, 0]
            byf = pts[bs + i, 1]
            bx = max(0, min(w - 1, int(bxf)))
            by = max(0, min(h - 1, int(byf)))
            sx = max(0, min(w - 1, int(bxf + ox)))
            sy = max(0, min(h - 1, int(byf + oy)))
            _sample_clean_jit(img, bx, by, smp_a)
            _sample_clean_jit(img, sx, sy, smp_b)
            for c in range(3):
                diffs[i, c] = smp_a[c] - smp_b[c]

        x0 = int(pts[ps, 0])
        x1 = x0
        y0 = int(pts[ps, 1])
        y1 = y0
        for i in range(pc):
            x0 = min(x0, int(pts[ps + i, 0]))
            x1 = max(x1, int(pts[ps + i, 0]))
            y0 = min(y0, int(pts[ps + i, 1]))
            y1 = max(y1, int(pts[ps + i, 1]))
        pad = int(rad) + 2
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w - 1, x1 + pad)
        y1 = min(h - 1, y1 + pad)

        chain = pts[ps : ps + pc]

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                px = float(x) + 0.5
                py = float(y) + 0.5
                d = _dist_to_chain(px, py, chain)
                if d >= rad:
                    continue

                on_sample = -1
                for i in range(bc):
                    vix = pts[bs + i, 0] - px
                    viy = pts[bs + i, 1] - py
                    li = math.sqrt(vix * vix + viy * viy)
                    vx[i] = vix
                    vy[i] = viy
                    vlen[i] = li
                    if li < 1e-4:
                        on_sample = i

                mr = 0.0
                mg = 0.0
                mb = 0.0
                if on_sample >= 0:
                    mr = diffs[on_sample, 0]
                    mg = diffs[on_sample, 1]
                    mb = diffs[on_sample, 2]
                else:
                    for i in range(bc):
                        j = i + 1
                        if j == bc:
                            j = 0
                        cross = vx[i] * vy[j] - vy[i] * vx[j]
                        if -1e-9 < cross < 1e-9:
                            cross = 1e-9
                        tans[i] = (vlen[i] * vlen[j] - (vx[i] * vx[j] + vy[i] * vy[j])) / cross
                    wsum = 0.0
                    for i in range(bc):
                        prev = i - 1
                        if prev < 0:
                            prev = bc - 1
                        wi = (tans[prev] + tans[i]) / vlen[i]
                        wsum += wi
                        mr += wi * diffs[i, 0]
                        mg += wi * diffs[i, 1]
                        mb += wi * diffs[i, 2]
                    if -1e-12 < wsum < 1e-12:
                        continue
                    mr /= wsum
                    mg /= wsum
                    mb /= wsum

                sx = max(0, min(w - 1, int(px + ox)))
                sy = max(0, min(h - 1, int(py + oy)))

                fth = (_RIM_FEATHER_FRAC + _RIM_FEATHER_UNGATED * (1.0 - gate)) * rad
                if fth < 1.5:
                    fth = 1.5
                t = (d - (rad - fth)) / fth
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                alpha = 1.0 - t * t * (3.0 - 2.0 * t)
                if alpha <= 0.0:
                    continue

                _sample_clean5_jit(img, sx, sy, smp_a)
                hr = smp_a[0] + mr
                hg = smp_a[1] + mg
                hb = smp_a[2] + mb

                # Dust gate: heal only pixels brighter than the membrane-predicted
                # clean value; gate=0 regions clone unconditionally.
                dest_l = LUMA_R * buf[y, x, 0] + LUMA_G * buf[y, x, 1] + LUMA_B * buf[y, x, 2]
                heal_l = LUMA_R * hr + LUMA_G * hg + LUMA_B * hb
                g = (dest_l - heal_l - _HEAL_GATE_LO) / (_HEAL_GATE_HI - _HEAL_GATE_LO)
                if g < 0.0:
                    g = 0.0
                elif g > 1.0:
                    g = 1.0
                alpha *= 1.0 - gate * (1.0 - g * g * (3.0 - 2.0 * g))
                if alpha <= 0.0:
                    continue

                buf[y, x, 0] = buf[y, x, 0] * (1.0 - alpha) + hr * alpha
                buf[y, x, 1] = buf[y, x, 1] * (1.0 - alpha) + hg * alpha
                buf[y, x, 2] = buf[y, x, 2] * (1.0 - alpha) + hb * alpha


def _capsule_boundary(pts_px: np.ndarray, radius: float, n: int) -> np.ndarray:
    """Ordered closed loop of ``n`` samples on the capsule outline around a polyline.

    Left side → end cap → right side (reversed) → start cap, so the loop is a
    simple polygon suitable for mean-value coordinates.
    """
    m = pts_px.shape[0]
    if m == 1:
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([pts_px[0, 0] + radius * np.cos(ang), pts_px[0, 1] + radius * np.sin(ang)], axis=1).astype(np.float32)

    seg = np.diff(pts_px, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = float(seg_len.sum())
    n_cap = max(3, int(round(n * (np.pi * radius) / (2.0 * total + 2.0 * np.pi * radius))))
    n_side = max(2, (n - 2 * n_cap) // 2)

    # Resample chain at n_side points; normals from central-difference tangents.
    t_targets = np.linspace(0.0, total, n_side)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    samples = np.empty((n_side, 2), dtype=np.float64)
    normals = np.empty((n_side, 2), dtype=np.float64)
    for i, t in enumerate(t_targets):
        k = int(np.searchsorted(cum, t, side="right") - 1)
        k = min(max(k, 0), m - 2)
        f = 0.0 if seg_len[k] < 1e-9 else (t - cum[k]) / seg_len[k]
        samples[i] = pts_px[k] + f * seg[k]
        tx, ty = seg[k]
        ln = math.hypot(tx, ty)
        if ln < 1e-9:
            tx, ty = 1.0, 0.0
        else:
            tx, ty = tx / ln, ty / ln
        normals[i] = (-ty, tx)

    left = samples + radius * normals
    right = samples - radius * normals

    def _cap(center: np.ndarray, from_pt: np.ndarray) -> np.ndarray:
        # Half-circle from the loop's current end, swept clockwise — that side
        # bulges outward past the chain end (the CCW side crosses the chain).
        a0 = math.atan2(from_pt[1] - center[1], from_pt[0] - center[0])
        ang = np.linspace(a0, a0 - np.pi, n_cap + 2)[1:-1]
        return np.stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)], axis=1)

    end_cap = _cap(samples[-1], left[-1])
    start_cap = _cap(samples[0], right[0])
    loop = np.concatenate([left, end_cap, right[::-1], start_cap], axis=0)
    return loop.astype(np.float32)


def fallback_source_offset(index: int, size_px: float, orig_shape: Tuple[int, int]) -> Tuple[float, float]:
    ang = _GOLDEN_ANGLE * float(index)
    dist = _FALLBACK_OFFSET_FACTOR * max(1.0, size_px)
    h, w = orig_shape
    return (math.cos(ang) * dist / max(1, w), math.sin(ang) * dist / max(1, h))


@njit(cache=True, fastmath=True)
def _detect_dust_mask_jit(
    luma: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    w_std: np.ndarray,
    dust_threshold: float,
) -> np.ndarray:
    """Local-contrast dust detector on the normalized-density plane; the
    wide-window texture penalty protects rocks/foliage."""
    h, w = luma.shape
    hit_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            l_curr = luma[y, x]
            l_mean = mean[y, x]
            local_s = max(0.005, std[y, x])

            w_s = max(0.0, w_std[y, x] - 0.02)
            wide_penalty = (w_s * w_s * w_s) * 800.0
            thresh = (dust_threshold * 0.4) + (local_s * 1.0) + wide_penalty

            if (l_curr - l_mean) > thresh and l_curr > 0.15 and (l_curr - l_mean) / local_s > 3.0:
                is_strong = (l_curr - l_mean) > (thresh * 2.5) or (l_curr - l_mean) > 0.25
                if 0 < y < h - 1 and 0 < x < w - 1:
                    is_max = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                            if luma[y + dy, x + dx] >= l_curr:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if is_max or is_strong:
                        hit_mask[y, x] = 1
                else:
                    hit_mask[y, x] = 1
    return hit_mask


def _proxy_norm(img: ImageBuffer) -> Tuple[float, float]:
    """(lo, spread) percentile normalization shared by the luma and per-channel proxies."""
    dens = -np.log10(np.clip(get_luminance(img), 1e-6, None))
    lo, hi = np.percentile(dens, (0.5, 99.5))
    return float(lo), max(float(hi - lo), _PROXY_MIN_SPREAD)


def _detection_proxy(img: ImageBuffer) -> np.ndarray:
    """Percentile-normalized source density: grade-independent, dust is bright
    in every process mode, and a defect's step stays proportional to its
    physical density excess — a print-like tone mapping would compress it
    below the detector threshold on wide-spread scans."""
    dens = -np.log10(np.clip(get_luminance(img), 1e-6, None))
    lo, spread = _proxy_norm(img)
    return np.clip((dens - lo) / spread, 0.0, 1.0).astype(np.float32)


def _detection_proxy_rgb(img: ImageBuffer, lo: float, spread: float) -> np.ndarray:
    """Per-channel density on the luma proxy's scale, for wrong-colour source scoring."""
    dens = -np.log10(np.clip(img, 1e-6, None))
    return np.clip((dens - lo) / spread, 0.0, 1.0).astype(np.float32)


def _is_hair(labels_sub: np.ndarray, area: int) -> bool:
    """Hair/scratch (thin) rather than speck: ``2*max(distanceTransform)`` is the widest
    the defect ever gets, so ``area/thickness²`` reads as length/thickness for a ribbon.

    Thin, not straight — bending moves no interior pixel further from its edge, so a
    twist scores like a straight hair, where PCA extent/width (the obvious measure)
    calls it compact and hands it to the membrane clone. The real hair on
    samples/ir/18.tiff: PCA aspect 2.45 = "speck", thinness 26.3 = hair.
    """
    if area < _HAIR_MIN_AREA:
        return False
    # Pad, or a component touching the sub-image border reads as thin along that edge.
    dist = cv2.distanceTransform(np.pad(labels_sub.astype(np.uint8), 1), cv2.DIST_L2, 5)
    thickness = 2.0 * float(dist.max())
    return area / max(thickness * thickness, 1e-6) >= _HAIR_MIN_ELONG


def _mask_to_strokes(
    mask: np.ndarray,
    pad_px: float,
    max_n: int,
) -> Tuple[List[Tuple[np.ndarray, float, float]], Optional[np.ndarray]]:
    """Connected defect components → ``(compact_comps, hair_mask)``. Compact specks
    become ``(chain_px, radius_px, area)`` tuples (largest first, truncated to
    ``max_n``); mildly elongated ones a ≤8-point membrane capsule. Strongly
    elongated defects (long twisted hairs/scratches) are painted into ``hair_mask``
    instead — they route to structure-following inpaint, which a single-offset
    membrane clone can't track."""
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8)
    comps = []
    hair_mask: Optional[np.ndarray] = None
    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        labels_sub = labels[y0 : y0 + bh, x0 : x0 + bw] == i
        if _is_hair(labels_sub, area):
            if hair_mask is None:
                hair_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            hair_mask[y0 : y0 + bh, x0 : x0 + bw][labels_sub] = 1
            continue

        ys, xs = np.nonzero(labels_sub)
        xs = xs.astype(np.float64) + x0 + 0.5
        ys = ys.astype(np.float64) + y0 + 0.5

        chain = None
        radius = math.sqrt(area / math.pi) + pad_px
        if area >= 6:
            mx, my = xs.mean(), ys.mean()
            cov = np.cov(np.stack([xs - mx, ys - my]))
            evals, evecs = np.linalg.eigh(cov)
            ax, ay = evecs[0, 1], evecs[1, 1]
            proj = (xs - mx) * ax + (ys - my) * ay
            perp = -(xs - mx) * ay + (ys - my) * ax
            ext = float(proj.max() - proj.min())
            half_w = max(0.5, float(np.percentile(np.abs(perp), 95)))
            if ext >= 2.5 * (2.0 * half_w) and ext >= 8.0:
                n_bins = int(min(8, max(2, ext / max(4.0 * half_w, 4.0))))
                edges = np.linspace(proj.min(), proj.max(), n_bins + 1)
                idx = np.clip(np.digitize(proj, edges) - 1, 0, n_bins - 1)
                pts = []
                for b in range(n_bins):
                    sel = idx == b
                    if np.any(sel):
                        pts.append([float(xs[sel].mean()), float(ys[sel].mean())])
                if len(pts) >= 2:
                    chain = np.array(pts, dtype=np.float64)
                    radius = half_w + pad_px
        if chain is None:
            chain = np.array([[float(centroids[i, 0]) + 0.5, float(centroids[i, 1]) + 0.5]], dtype=np.float64)
        comps.append((chain, float(radius), float(area)))

    comps.sort(key=lambda c: -c[2])
    return comps[:max_n], hair_mask


_PICK_RINGS = (2.6, 3.6, 4.6, 6.2)
_PICK_RINGS_FAR = (2.6, 3.6, 4.6, 6.2, 8.0, 11.0)


def _pick_candidate_dirs(chain: np.ndarray, index: int) -> List[Tuple[float, float]]:
    """Search directions: perp/along/perp-diagonal for a capsule, 8 golden-angle for a speck."""
    if len(chain) >= 2:
        d = chain[-1] - chain[0]
        ln = math.hypot(d[0], d[1])
        tx, ty = (d[0] / ln, d[1] / ln) if ln > 1e-6 else (1.0, 0.0)
        nx, ny = -ty, tx
        dirs = [(nx, ny), (-nx, -ny), (tx, ty), (-tx, -ty)]
        for ax, ay in ((nx + tx, ny + ty), (nx - tx, ny - ty)):
            ll = math.hypot(ax, ay)
            if ll > 1e-6:
                dirs.append((ax / ll, ay / ll))
                dirs.append((-ax / ll, -ay / ll))
        return dirs
    return [
        (math.cos(_GOLDEN_ANGLE * (index + 1) + k * math.pi / 4.0), math.sin(_GOLDEN_ANGLE * (index + 1) + k * math.pi / 4.0))
        for k in range(8)
    ]


def _pick_source_offsets(
    mask: np.ndarray,
    comps: List[Tuple[np.ndarray, float, float]],
    guide: np.ndarray,
) -> List[Tuple[float, float]]:
    """Best clone-source offset per component by content match on ``guide`` (box stats
    via integral images; mask-freedom alone would clone interior detail through the
    membrane). ``guide`` may be single- or per-channel (RGB density) — the per-channel
    |Δmean| term rejects a wrong-colour source. Pass 2 scores with an overlap penalty
    instead of rejecting, so a ringed-in defect still gets a content pick, not a blind one."""
    h, w = mask.shape
    m8 = np.ascontiguousarray(mask, dtype=np.uint8)
    integ = cv2.integral(m8)
    g3 = guide.astype(np.float32)
    if g3.ndim == 2:
        g3 = g3[:, :, None]
    ch = g3.shape[2]
    keep = (1.0 - m8)[:, :, None]
    s1, s2 = [], []
    for c in range(ch):
        a1, a2 = cv2.integral2(np.ascontiguousarray(g3[:, :, c] * keep[:, :, 0]))
        s1.append(a1)
        s2.append(a2)

    def box(ii, x0, y0, x1, y1):
        return float(ii[y1 + 1, x1 + 1] - ii[y0, x1 + 1] - ii[y1 + 1, x0] + ii[y0, x0])

    offsets = []
    for index, (chain, radius, _area) in enumerate(comps):
        dirs = _pick_candidate_dirs(chain, index)
        b = int(math.ceil(1.2 * radius)) + 1
        area = float((2 * b + 1) ** 2)

        d_n = 0.0
        d_s = [0.0] * ch
        d_ss = [0.0] * ch
        for px, py in chain:
            x0, x1 = max(int(px) - b, 0), min(int(px) + b, w - 1)
            y0, y1 = max(int(py) - b, 0), min(int(py) + b, h - 1)
            d_n += (x1 - x0 + 1) * (y1 - y0 + 1) - box(integ, x0, y0, x1, y1)
            for c in range(ch):
                d_s[c] += box(s1[c], x0, y0, x1, y1)
                d_ss[c] += box(s2[c], x0, y0, x1, y1)
        dn = max(d_n, 1.0)
        d_mean = [d_s[c] / dn for c in range(ch)]
        d_std = [math.sqrt(max(d_ss[c] / dn - d_mean[c] * d_mean[c], 0.0)) for c in range(ch)]

        def evaluate(rings, allow_overlap):
            best, best_score = None, math.inf
            for ring in rings:
                dist = ring * radius
                for dx, dy in dirs:
                    ox, oy = dx * dist, dy * dist
                    n = 0.0
                    overlap = 0.0
                    s = [0.0] * ch
                    ss = [0.0] * ch
                    in_bounds = True
                    for px, py in chain:
                        x0, x1 = int(px + ox) - b, int(px + ox) + b
                        y0, y1 = int(py + oy) - b, int(py + oy) + b
                        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
                            in_bounds = False
                            break
                        overlap += box(integ, x0, y0, x1, y1)
                        n += area
                        for c in range(ch):
                            s[c] += box(s1[c], x0, y0, x1, y1)
                            ss[c] += box(s2[c], x0, y0, x1, y1)
                    if not in_bounds or (not allow_overlap and overlap > 0):
                        continue
                    score = 0.0
                    for c in range(ch):
                        mean = s[c] / n
                        std = math.sqrt(max(ss[c] / n - mean * mean, 0.0))
                        score += abs(mean - d_mean[c]) + max(0.0, std - d_std[c])
                    if allow_overlap:
                        score += 10.0 * (overlap / max(n, 1.0))
                    if score < best_score:
                        best_score, best = score, (ox, oy)
            return best

        found = evaluate(_PICK_RINGS, allow_overlap=False)
        if found is None:
            found = evaluate(_PICK_RINGS_FAR, allow_overlap=True)
        if found is None:
            fdx, fdy = fallback_source_offset(index, 2.0 * radius, (h, w))
            found = (fdx * w, fdy * h)
        offsets.append(found)
    return offsets


def _finalize_strokes(
    comps: List[Tuple[np.ndarray, float, float]],
    offsets: List[Tuple[float, float]],
    det_dims: Tuple[int, int],
    gate: float,
) -> List[Tuple]:
    """Detection-space components → stroke tuples (source-normalized, plain
    rounded floats — numpy scalars would make the config hash repr-dependent)."""
    h, w = det_dims
    strokes = []
    for (chain, radius, _area), (ox, oy) in zip(comps, offsets):
        points = [[round(float(px) / w, 6), round(float(py) / h, 6)] for px, py in chain]
        size = round(2.0 * float(radius) * HEAL_SIZE_REF / max(w, h), 6)
        strokes.append((points, size, round(float(ox) / w, 6), round(float(oy) / h, 6), float(gate)))
    return strokes


def compute_dust_stats(img: ImageBuffer, dust_size: int) -> Tuple[np.ndarray, ...]:
    """Threshold-independent detection stat maps (proxy + blur windows + per-channel
    proxy) — the expensive ~2/3 of a detection pass, cacheable across threshold
    changes. The 5th element is the RGB density proxy used to score clone sources."""
    lo, spread = _proxy_norm(img)
    dens = -np.log10(np.clip(get_luminance(img), 1e-6, None))
    proxy = np.clip((dens - lo) / spread, 0.0, 1.0).astype(np.float32)
    proxy_rgb = _detection_proxy_rgb(img, lo, spread)
    base_size = max(1.0, float(dust_size))
    v_win = int(max(3, base_size * 3.0)) * 2 + 1
    w_win = int(max(7, base_size * 4.0)) * 2 + 1
    mean = cv2.blur(proxy, (v_win, v_win))
    std = np.sqrt(np.clip(cv2.blur(proxy**2, (v_win, v_win)) - mean**2, 0, None))
    w_std = np.sqrt(np.clip(cv2.blur(proxy**2, (w_win, w_win)) - cv2.blur(proxy, (w_win, w_win)) ** 2, 0, None))
    return (
        np.ascontiguousarray(proxy.astype(np.float32)),
        np.ascontiguousarray(mean.astype(np.float32)),
        np.ascontiguousarray(std.astype(np.float32)),
        np.ascontiguousarray(w_std.astype(np.float32)),
        np.ascontiguousarray(proxy_rgb),
    )


def detect_luma_regions(
    img: ImageBuffer,
    dust_threshold: float,
    dust_size: int,
    gate: float = 1.0,
    max_n: int = 512,
    stats: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[List[Tuple], Optional[np.ndarray]]:
    """Statistical dust detection on the linear source → ``(strokes, hair_mask)``:
    compact specks become membrane strokes, strong hairs a detection-scale mask
    for structure-following inpaint."""
    if stats is None:
        stats = compute_dust_stats(img, dust_size)
    proxy, mean, std, w_std = stats[:4]
    hit = _detect_dust_mask_jit(proxy, mean, std, w_std, float(dust_threshold))
    if not np.any(hit):
        return [], None
    comps, hair_mask = _mask_to_strokes(hit, _DETECT_PAD_PX, max_n)
    if not comps:
        return [], hair_mask
    offsets = _pick_source_offsets(hit, comps, stats[4])
    return _finalize_strokes(comps, offsets, hit.shape, gate), hair_mask


def downsample_ir(plane: np.ndarray, target_long_edge: int, dims: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Min-preserving IR downsample to ``target_long_edge`` (no-op if already smaller).
    ``dims`` (w, h) overrides the computed target for callers that must land on an
    existing buffer's exact shape.

    A defect is a *minimum* in IR transmittance and INTER_AREA averages sub-pixel minima
    away: a ~4 px hair downsampled 4.5x lost its dip from 0.22 to 0.31 and shattered into
    stray pixels. Eroding by the resample footprint first carries the dip through;
    ``normalize_ir``'s ``blur(dilate(ir))`` base tracks the eroded plane back up, so clean
    film still sits at ~1.0. Every IR consumer routes through here or preview and export
    detect different region sets.
    """
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    h, w = plane.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target_long_edge and dims is None:
        return plane
    if dims is None:
        s = target_long_edge / long_edge
        dims = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    if dims == (w, h):
        return plane
    # Erode by the resample footprint — a 1.25x downsample must not fatten by a 4.5x kernel.
    k = max(1, int(round(long_edge / target_long_edge)) | 1)
    if k > 1:
        plane = cv2.erode(plane, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.resize(plane, dims, interpolation=cv2.INTER_AREA).astype(np.float32)


def normalize_ir(plane: np.ndarray) -> np.ndarray:
    """Locally-normalized IR: ``ir / blur(dilate(ir))`` — ~1.0 on clean film, dips on
    defects, illumination-independent. Separates dust from content that raw-IR
    thresholding conflated (dilate→max estimates the clean base, blur smooths it)."""
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_IR_BASE_WIN, _IR_BASE_WIN))
    base = cv2.blur(cv2.dilate(plane, kernel), (_IR_BASE_WIN, _IR_BASE_WIN))
    return plane / np.maximum(base, 1e-4)


def ir_detect_cutoff(slider: float, attenuation: bool) -> float:
    """UI IR sensitivity (higher = conservative) → ratio cutoff; lower slider catches
    more. Attenuation-on band sits lower (division handles the rest, only cores need cloning)."""
    s = float(np.clip(slider, 0.0, 1.0))
    return (0.85 - 0.40 * s) if attenuation else (0.95 - 0.20 * s)


def ir_defect_score(ratio: np.ndarray, cutoff: float) -> np.ndarray:
    """Continuous defect score in ``[_IR_SCORE_FLOOR, 1]``: 1 = clean film, floor
    at/below ``cutoff`` (from ir_detect_cutoff). The 3×3 erode bleeds a defect's score
    one pixel outward, covering sub-pixel hairs and the min-pool skirt."""
    span = max(_IR_GAIN_IDENTITY - cutoff, 1e-4)
    t = (np.ascontiguousarray(ratio, dtype=np.float32) - cutoff) / span
    score = np.clip(t * (1.0 - _IR_SCORE_FLOOR) + _IR_SCORE_FLOOR, _IR_SCORE_FLOOR, 1.0)
    return cv2.erode(score, np.ones((3, 3), np.uint8))


def score_weighted_fill(img: np.ndarray, score: np.ndarray, scales: Tuple[int, ...] = _IR_FILL_SCALES) -> np.ndarray:
    """Multiscale score-normalized average, blended coarse→fine by clean fraction.
    Where no support holds clean film the quotient tends to zero and the original-floor
    rule in apply_ir_reconstruction keeps the source pixel."""
    s3 = score[..., None]
    weighted = img * s3
    fill: Optional[np.ndarray] = None
    for i, k in enumerate(scales):
        if i == len(scales) - 1:
            num = cv2.GaussianBlur(weighted, (k, k), 0)
            den = cv2.GaussianBlur(score, (k, k), 0)
        else:
            num = cv2.boxFilter(weighted, -1, (k, k))
            den = cv2.boxFilter(score, -1, (k, k))
        cand = num / np.maximum(den, 1e-6)[..., None]
        if fill is None:
            fill = cand
        else:
            conf = np.clip(den / _IR_FILL_TAU, 0.0, 1.0)[..., None]
            fill = fill * (1.0 - conf) + cand * conf
    assert fill is not None  # scales is never empty
    return fill


def apply_ir_reconstruction(img: ImageBuffer, score_det: np.ndarray) -> ImageBuffer:
    """Bake the score-weighted fill into the linear source (new array). The detection-scale
    score is upsampled; the fill convolutions rerun at the buffer's own resolution with
    rescaled supports — filled pixels are never upsampled."""
    h, w = img.shape[:2]
    src = np.ascontiguousarray(img, dtype=np.float32)
    if score_det.shape[:2] == (h, w):
        score, factor = np.ascontiguousarray(score_det, dtype=np.float32), 1.0
    else:
        factor = max(h / score_det.shape[0], w / score_det.shape[1])
        score = cv2.resize(score_det, (w, h), interpolation=cv2.INTER_LINEAR)
    scales = tuple(int(round(k * factor)) | 1 for k in _IR_FILL_SCALES)
    fill = score_weighted_fill(src, score, scales)
    a = np.clip((_IR_WRITE_HI - score) / (_IR_WRITE_HI - _IR_WRITE_LO), 0.0, 1.0)
    a = (a * a * (3.0 - 2.0 * a))[..., None]
    out = src * (1.0 - a) + fill * a
    # Original-floor rule: dust is dark in negative transmittance — repairs only lighten.
    np.maximum(out, src, out=out)
    return ensure_image(out)


def route_ir_defects(score: np.ndarray) -> Optional[np.ndarray]:
    """Detection-scale mask of at-floor components past the fill's reach, for
    apply_hair_inpaint. Over budget (misregistered/garbage IR) → None + warning."""
    at_floor = (score <= _IR_SCORE_FLOOR + 1e-6).astype(np.uint8)
    if not at_floor.any():
        return None
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(at_floor, connectivity=8)
    routed = np.zeros_like(at_floor)
    side = 2 * _IR_ROUTE_RADIUS - 1
    hit = False
    for i in range(1, n_lbl):
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if min(bw, bh) < side:  # can't contain a side² solid → radius under the bar
            continue
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        sub = np.pad((labels[y0 : y0 + bh, x0 : x0 + bw] == i).astype(np.uint8), 1)
        if float(cv2.distanceTransform(sub, cv2.DIST_C, 3).max()) >= _IR_ROUTE_RADIUS:
            routed[labels == i] = 1
            hit = True
    if not hit:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _IR_ROUTE_DILATE + 1,) * 2)
    routed = cv2.dilate(routed, k)
    frac = float(routed.mean())
    if frac > _IR_ROUTE_BUDGET:
        logger.warning("IR dust: routed defects cover %.1f%% of the frame — inpaint skipped, fill only", frac * 100.0)
        return None
    return routed


def _ir_decontaminate(ratio: np.ndarray, vis_log: np.ndarray) -> Tuple[np.ndarray, float]:
    """Divide the visible-image ghost out of the normalized IR: robust LS fit of
    log(ir) on log(vis) over clean film, then ``ratio / Π vis_c^b_c``. Exponents clamp
    to ≥0 (density can only block IR) and fit to ~0 on a clean scanner (→ no-op). Also
    returns the exponent sum — ghost strength, which is how ``ir_ratio_and_gain`` bails."""
    if ratio.size < 500:
        return ratio, 0.0
    # Fit on clean film only. Dust dips *both* planes, so a fit that sees it explains
    # the defect away as ghost and the division stops lifting it. Trim by ratio
    # percentile, not a fixed cutoff (a strong ghost drags clean film below any fixed
    # one) and not by residual (the dust fits itself perfectly — residual can't see it).
    keep = ratio >= np.percentile(ratio, _IR_XTALK_TRIM)
    y = np.log(np.clip(ratio[keep], 1e-4, 1.0))
    x = vis_log[keep].reshape(-1, vis_log.shape[-1])
    if y.size < 500:
        return ratio, 0.0
    # Intercept column, dropped from the result: both logs sit below their own dilate+blur
    # envelope, so origin-forced least squares reads that shared negative offset as slope
    # and fits b≈0.6 on two *independent* noisy planes.
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    b = np.clip(np.linalg.lstsq(x, y, rcond=None)[0][:3], 0.0, _IR_XTALK_MAX)
    ghost = float(np.abs(b).sum())
    if ghost < _IR_XTALK_MIN:
        return ratio, ghost
    return np.clip(ratio / np.exp((vis_log * b).sum(-1)), 0.0, 1.5).astype(np.float32), ghost


def _fit_refraction_gammas(ratio: np.ndarray, vis_log: np.ndarray, img_det: np.ndarray) -> Tuple[float, ...]:
    """Per-channel refraction γ: the slope of log(vis_norm) on log(ratio) over the
    shallow-dust band, as the median of the per-pixel slopes over locally flat film.

    Median and flat restriction are both load-bearing. The band selects on the IR ratio
    alone, so besides dust it collects ``_ir_decontaminate``'s residue at hard image edges,
    and least squares through the origin is x²-weighted — that deep non-dust minority
    dominated it, reading γ 1.9/2.2/2.2 for dust measuring ~1.0/1.1/1.2 and over-correcting
    every speck into a dark cyan blob. Median alone reads 1.3/1.8/1.8, flat-only least
    squares 1.4/2.1/2.0, and γ 1.5 already tints."""
    band = (ratio > 0.70) & (ratio < 0.92)
    if int(band.sum()) < 500:
        return (_IR_GAMMA_FALLBACK,) * 3
    # ksize=5 carries its own smoothing, so no separate blur.
    edge = np.abs(cv2.Laplacian(img_det[:, :, 1], cv2.CV_32F, ksize=5))
    flat = band & (edge < np.percentile(edge[band], _IR_FIT_FLAT_PCT))
    fit = flat if int(flat.sum()) >= _IR_FIT_MIN_PX else band
    # The band bounds ratio away from 1, so the per-pixel slope needs no guard.
    xb = np.log(ratio[fit])
    return tuple(float(np.clip(np.median(vis_log[:, :, c][fit] / xb), _IR_GAMMA_LO, _IR_GAMMA_HI)) for c in range(3))


def _ir_clean_base(img_det: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Local clean-film level per channel over ``_IR_CAP_WIN``: mean of the pixels the
    IR ratio calls clean, minus ``_IR_CAP_SIGMA`` of their σ (see the constants block)."""
    win = (_IR_CAP_WIN, _IR_CAP_WIN)
    w_clean = (ratio >= _IR_GAIN_IDENTITY).astype(np.float32)
    den = np.maximum(cv2.blur(w_clean, win), 1e-6)[..., None]
    mean = cv2.blur(img_det * w_clean[..., None], win) / den
    var = cv2.blur(img_det * img_det * w_clean[..., None], win) / den - mean * mean
    base = mean - _IR_CAP_SIGMA * np.sqrt(np.clip(var, 0.0, None))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_IR_CAP_WIN, _IR_CAP_WIN))
    dil = cv2.blur(cv2.dilate(img_det, kernel), win)
    return np.where(den > _IR_CAP_MIN_SUPPORT, base, dil)


def ir_ratio_and_gain(ir_det: np.ndarray, img_det: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Tuple[float, ...]]:
    """Detection-scale ``(ratio, gain HxWx3, degenerate, gammas)`` for IR-division
    attenuation: semi-transparent dust recovered by ``RGB / ratio^γ``, γ per channel from
    ``_fit_refraction_gammas``. ``degenerate`` = IR carrying image content
    (B&W/Kodachrome) → caller skips the whole IR bake."""
    plane = ir_det[:, :, 0] if ir_det.ndim == 3 else ir_det
    ratio = normalize_ir(plane)
    # No film under the head is not a defect; left as a dip the holder margin would score
    # as one giant routed component and swamp the routing budget.
    ratio[plane < _IR_DEAD_FLOOR] = 1.0
    img_det = np.ascontiguousarray(img_det, dtype=np.float32)
    if img_det.shape[:2] != ratio.shape[:2]:
        img_det = cv2.resize(img_det, (ratio.shape[1], ratio.shape[0]), interpolation=cv2.INTER_AREA)

    vis_log = np.stack([np.log(np.clip(normalize_ir(img_det[:, :, c]), 1e-4, 1.0)) for c in range(3)], axis=-1)
    ratio, ghost = _ir_decontaminate(ratio, vis_log)
    # On the fitted exponent, not on how far the ratio dips: a few percent of IR noise
    # (deepened by the min-preserving downsample) read as silver on clean C41 rolls.
    degenerate = ghost > _IR_DEGENERATE_GHOST

    gammas = _fit_refraction_gammas(ratio, vis_log, img_det)
    base = np.clip(ratio / _IR_GAIN_IDENTITY, 1e-4, 1.0)
    gain = np.empty(ratio.shape + (3,), dtype=np.float32)
    for c in range(3):
        gain[:, :, c] = np.minimum(_IR_GAIN_CLAMP, base ** (-gammas[c]))
    # Never lift a pixel past its own local clean base (see _IR_CAP_WIN); floored at 1 so the
    # cap only ever holds the bake back, never darkens a pixel itself.
    clean = _ir_clean_base(img_det, ratio)
    np.minimum(gain, np.maximum(clean / np.maximum(img_det, 1e-5), 1.0), out=gain)
    return ratio, gain, degenerate, gammas


def apply_ir_attenuation(img: ImageBuffer, gain_det: np.ndarray) -> ImageBuffer:
    """Visible buffer × upsampled per-channel IR gain map (new array — buffers are read-only)."""
    h, w = img.shape[:2]
    gain = gain_det if gain_det.shape[:2] == (h, w) else cv2.resize(gain_det, (w, h), interpolation=cv2.INTER_LINEAR)
    # cv2.multiply, not `a * b`: the product of two float32 buffers is already float32,
    # so the astype numpy needs here would copy the whole frame a second time.
    return ensure_image(cv2.multiply(np.ascontiguousarray(img, dtype=np.float32), gain))


def ir_bake_token(retouch, has_ir: bool) -> str:
    """Config-identity token for the IR bake (mirrors ``flatfield_token``); folded into
    source_hash so a toggle or threshold drag invalidates the engine cache."""
    if not (retouch.ir_dust_remove and has_ir):
        return ""
    return f"|ir{int(retouch.ir_attenuation)}r{round(float(retouch.ir_threshold), 3)}"


def apply_hair_inpaint(
    img: ImageBuffer,
    hair_masks: List[np.ndarray],
    radius: int = _HAIR_INPAINT_RADIUS,
    dilate_px: Optional[int] = None,
) -> ImageBuffer:
    """Structure-following fill of long/twisted defects (``cv2.inpaint``, Navier–Stokes)
    baked into the linear source. Each detection-scale mask is upsampled to the buffer,
    unioned and dilated to cover the PSF skirt; only masked pixels are overwritten (the
    rest stay byte-identical — the 8-bit encode cv2.inpaint requires touches only the
    fabricated hairline). Returns a new array (buffers are read-only)."""
    h, w = img.shape[:2]
    masks = [hm for hm in hair_masks if hm is not None]
    if not masks:
        return img
    factor = max(1.0, h / masks[0].shape[0], w / masks[0].shape[1])
    if dilate_px is None:
        # A detection-scale mask knows its boundary only to within the upsample factor,
        # so the dilate tracks it; at 1:1 that's _HAIR_DILATE_PX, the PSF skirt alone.
        dilate_px = max(_HAIR_DILATE_PX, round(factor))
    m = np.zeros((h, w), dtype=np.uint8)
    for hm in masks:
        r = hm if hm.shape[:2] == (h, w) else cv2.resize(hm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        m |= (np.asarray(r) > 0.5).astype(np.uint8)
    if not m.any():
        return img
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    src = np.ascontiguousarray(img, dtype=np.float32)
    out = src.copy()
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n_lbl):
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        x0, y0 = max(0, bx - _HAIR_INPAINT_PAD), max(0, by - _HAIR_INPAINT_PAD)
        x1 = min(w, bx + int(stats[i, cv2.CC_STAT_WIDTH]) + _HAIR_INPAINT_PAD)
        y1 = min(h, by + int(stats[i, cv2.CC_STAT_HEIGHT]) + _HAIR_INPAINT_PAD)
        # Mask the whole crop, not just this component: a neighbour reaching into the
        # bbox must stay unknown or it becomes clone source and its dust is filled back in.
        sub_m = np.ascontiguousarray(m[y0:y1, x0:x1])
        crop = src[y0:y1, x0:x1]
        # Encode against the crop's clean range — clip(0,1) posterizes fills in dark regions.
        ctx = crop[sub_m == 0]
        lo = float(np.percentile(ctx, 0.5)) if ctx.size else 0.0
        hi = float(np.percentile(ctx, 99.5)) if ctx.size else 1.0
        span = max(hi - lo, 1e-4)
        enc = np.clip((crop - lo) / span, 0.0, 1.0) ** (1.0 / _HAIR_INPAINT_GAMMA)
        filled = cv2.inpaint((enc * 255.0 + 0.5).astype(np.uint8), sub_m, radius, cv2.INPAINT_NS)
        dec = ((filled.astype(np.float32) / 255.0) ** _HAIR_INPAINT_GAMMA) * span + lo
        # ...but only keep this component (a neighbour clipped by the bbox fills badly here,
        # and gets its own correctly-padded crop anyway), alpha-feathered across the dilate
        # band: full fill on the detected defect, ramp over the skirt. dilate_px=0 → no feather.
        mb = labels[y0:y1, x0:x1] == i
        d = cv2.distanceTransform(mb.astype(np.uint8), cv2.DIST_C, 3)
        a = np.minimum(d / float(dilate_px + 1), 1.0)[..., None]
        blended = crop * (1.0 - a) + dec * a
        out[y0:y1, x0:x1][mb] = blended[mb]
    return out


def hair_bake_token(retouch) -> str:
    """Detection-param identity for the hair inpaint (folded into source_hash when a
    hair is actually detected). Distinct params → distinct inpainted source."""
    r = retouch
    return f"|hair{int(r.dust_remove)}_{round(float(r.dust_threshold), 3)}_{int(r.dust_size)}_{int(r.ir_dust_remove)}_{round(float(r.ir_threshold), 3)}"


def build_heal_regions(
    strokes: List[Tuple],
    legacy_spots: List[Tuple[float, float, float]],
    orig_shape: Tuple[int, int],
    rotation: int,
    fine_rotation: float,
    flip_h: bool,
    flip_v: bool,
    distortion_k1: float,
    full_dims: Tuple[int, int],
    max_regions: int = 512,
    max_points: int = 32768,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maps manual heals into the geometry frame as capsule regions.

    Returns ``(reg_i, reg_f, pts)`` in the layout `_membrane_heal_jit` consumes;
    ``pts`` are continuous pixel coords in the post-geometry frame at ``full_dims``.
    Shared by the CPU processor and the GPU storage upload so both paths heal
    from identical geometry.
    """
    fw, fh = float(full_dims[0]), float(full_dims[1])

    def _map(nx: float, ny: float) -> Tuple[float, float]:
        mx, my = map_coords_to_geometry(nx, ny, orig_shape, rotation, fine_rotation, flip_h, flip_v, distortion_k1=distortion_k1)
        return mx * fw, my * fh

    entries: List[Tuple[List, float, float, float, float]] = []
    for stroke in strokes:
        points, size, sdx, sdy = stroke[:4]
        # 5th element = gate flag (synthesized regions); 4-tuple user strokes stay gated.
        gate = float(stroke[4]) if len(stroke) > 4 else 1.0
        entries.append((list(points), float(size), float(sdx), float(sdy), gate))
    for i, (nx, ny, size) in enumerate(legacy_spots):
        fdx, fdy = fallback_source_offset(i, float(size), orig_shape)
        entries.append(([[nx, ny]], float(size), fdx, fdy, 1.0))

    reg_i_list = []
    reg_f_list = []
    pts_list: List[np.ndarray] = []
    n_pts = 0

    for points, size, sdx, sdy, gate in entries[:max_regions]:
        chain = np.array([_map(p[0], p[1]) for p in points], dtype=np.float32)
        # Curve the heal band through its waypoints (spots/2-point strokes unaffected).
        if len(chain) >= 3:
            chain = np.array(smooth_polyline([(float(x), float(y)) for x, y in chain], closed=False), dtype=np.float32)
        # Brush size is a DIAMETER at HEAL_SIZE_REF scale: the footprint must match
        # the cursor (overlay._brush_screen_radius draws size/(2·HEAL_SIZE_REF)).
        radius = max(1.0, float(size) * (max(fw, fh) / HEAL_SIZE_REF) * 0.5)

        cx = float(np.mean([p[0] for p in points]))
        cy = float(np.mean([p[1] for p in points]))
        c_px = _map(cx, cy)
        s_px = _map(cx + sdx, cy + sdy)
        off_x, off_y = s_px[0] - c_px[0], s_px[1] - c_px[1]

        rim_rad = radius + _MEMBRANE_RIM_PX * (max(fw, fh) / HEAL_SIZE_REF)
        seg = np.diff(chain, axis=0)
        perimeter = 2.0 * float(np.hypot(seg[:, 0], seg[:, 1]).sum()) + 2.0 * np.pi * rim_rad
        n_bnd = int(min(64, max(16, perimeter / 4.0)))
        boundary = _capsule_boundary(chain.astype(np.float64), rim_rad, n_bnd)

        if n_pts + len(chain) + len(boundary) > max_points:
            break
        reg_i_list.append((n_pts, len(chain), n_pts + len(chain), len(boundary)))
        reg_f_list.append((radius, off_x, off_y, gate))
        pts_list.append(chain)
        pts_list.append(boundary)
        n_pts += len(chain) + len(boundary)

    if not reg_i_list:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    return (
        np.array(reg_i_list, dtype=np.int32),
        np.array(reg_f_list, dtype=np.float32),
        np.concatenate(pts_list, axis=0).astype(np.float32),
    )


def select_source_offset(
    preview_img: np.ndarray,
    pts_norm: List[Tuple[float, float]],
    radius_px: float,
    index: int,
) -> Tuple[float, float]:
    """Lightroom-style automatic clone-source pick, scored on the source-frame preview.

    Candidates sit perpendicular to the stroke (ring for spots) at 2.6r/3.6r;
    each is scored by RGB SSD between a clean rim band around the defect and
    the same band shifted by the candidate. Returns a source-normalized offset.
    """
    h, w = preview_img.shape[:2]
    orig_shape = (h, w)
    pts_px = np.array([[p[0] * w, p[1] * h] for p in pts_norm], dtype=np.float64)
    r = max(1.5, float(radius_px))

    if len(pts_px) >= 2:
        d = pts_px[-1] - pts_px[0]
        ln = math.hypot(d[0], d[1])
        tx, ty = (d[0] / ln, d[1] / ln) if ln > 1e-6 else (1.0, 0.0)
    else:
        tx, ty = 1.0, 0.0
    nx_, ny_ = -ty, tx

    candidates = []
    for dist in (_FALLBACK_OFFSET_FACTOR * r, (_FALLBACK_OFFSET_FACTOR + 1.0) * r):
        candidates.append((nx_ * dist, ny_ * dist))
        candidates.append((-nx_ * dist, -ny_ * dist))
    if len(pts_px) == 1:
        for k in range(4):
            ang = np.pi / 4.0 + k * np.pi / 2.0
            dist = _FALLBACK_OFFSET_FACTOR * r
            candidates.append((math.cos(ang) * dist, math.sin(ang) * dist))
    else:
        # Along-stroke candidates must clear the whole stroke length.
        seg = np.diff(pts_px, axis=0)
        length = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        for sgn in (1.0, -1.0):
            candidates.append((sgn * tx * (length + _FALLBACK_OFFSET_FACTOR * r), sgn * ty * (length + _FALLBACK_OFFSET_FACTOR * r)))

    # Clean rim band just outside the defect.
    boundary = _capsule_boundary(pts_px, 1.6 * r, 32)
    # Chain samples (vertices + midpoints) for the shifted-defect overlap test.
    chain_samples = [tuple(p) for p in pts_px]
    for a, b in zip(pts_px[:-1], pts_px[1:]):
        chain_samples.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
    # Interior probes of the candidate patch (dust check inside, not just the rim).
    interior = chain_samples + [tuple(p) for p in _capsule_boundary(pts_px, 0.6 * r, 16)]
    luma_w = np.array([LUMA_R, LUMA_G, LUMA_B], dtype=np.float64)

    best = None
    best_score = np.inf
    for cdx, cdy in candidates:
        # The shifted defect must clear the original defect entirely.
        if any(_dist_to_chain(cx + cdx, cy + cdy, pts_px) < 2.2 * r for cx, cy in chain_samples):
            continue
        score = 0.0
        valid = True
        band_lums = []
        for bx, by in boundary:
            sx, sy = bx + cdx, by + cdy
            if not (0 <= sx < w - 1 and 0 <= sy < h - 1):
                valid = False
                break
            src_px = preview_img[int(sy), int(sx)]
            diff = src_px - preview_img[int(by), int(bx)]
            score += float(np.dot(diff, diff))
            band_lums.append(float(np.dot(src_px[:3], luma_w)))
        if not valid:
            continue
        # Heavy penalty for structure inside the candidate patch: interior lumas
        # far from the candidate band's median mean the patch contains a speck
        # (bright) or real detail (dark) that would be cloned into the heal.
        med = float(np.median(band_lums))
        for cx_, cy_ in interior:
            sx, sy = cx_ + cdx, cy_ + cdy
            if not (0 <= sx < w - 1 and 0 <= sy < h - 1):
                valid = False
                break
            excess = abs(float(np.dot(preview_img[int(sy), int(sx)][:3], luma_w)) - med) - _CLONE_GUARD_LUMA
            if excess > 0.0:
                score += excess * excess * 100.0 * len(boundary)
        if valid and score < best_score:
            best_score = score
            best = (cdx, cdy)

    if best is None:
        return fallback_source_offset(index, r, orig_shape)
    return (best[0] / w, best[1] / h)


def apply_manual_heals(
    img: ImageBuffer,
    reg_i: np.ndarray,
    reg_f: np.ndarray,
    pts: np.ndarray,
) -> ImageBuffer:
    """Membrane-clones all manual heal regions. Perceptual op — brackets the linear buffer."""
    if len(reg_i) == 0:
        return img
    buf = np.ascontiguousarray(working_oetf_encode(img).astype(np.float32))
    _membrane_heal_jit(
        buf,
        np.ascontiguousarray(reg_i),
        np.ascontiguousarray(reg_f),
        np.ascontiguousarray(pts),
    )
    return ensure_image(working_oetf_decode(buf))
