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
# Rim feather fraction of the blend radius (1.5px floor — a fixed 1.5px edge is a
# hard seam at export scale). Mirrored in retouch.wgsl.
_RIM_FEATHER_FRAC = 0.25


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

                fth = _RIM_FEATHER_FRAC * rad
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


def _detection_proxy(img: ImageBuffer) -> np.ndarray:
    """Percentile-normalized source density: grade-independent, dust is bright
    in every process mode, and a defect's step stays proportional to its
    physical density excess — a print-like tone mapping would compress it
    below the detector threshold on wide-spread scans."""
    luma = get_luminance(img)
    dens = -np.log10(np.clip(luma, 1e-6, None))
    lo, hi = np.percentile(dens, (0.5, 99.5))
    spread = max(float(hi - lo), _PROXY_MIN_SPREAD)
    return np.clip((dens - lo) / spread, 0.0, 1.0).astype(np.float32)


def _mask_to_strokes(
    mask: np.ndarray,
    pad_px: float,
    max_n: int,
) -> List[Tuple[np.ndarray, float, float]]:
    """Connected defect components → ``(chain_px, radius_px, area)``, largest
    first, truncated to ``max_n``. Elongated components (hairs) become ≤8-point
    polylines along the principal axis — a circle either misses the hair or
    over-heals its bounding disk."""
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8)
    comps = []
    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        ys, xs = np.nonzero(labels[y0 : y0 + bh, x0 : x0 + bw] == i)
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
    return comps[:max_n]


def _pick_source_offsets(
    mask: np.ndarray,
    comps: List[Tuple[np.ndarray, float, float]],
    guide: np.ndarray,
) -> List[Tuple[float, float]]:
    """Best mask-free candidate by content match on ``guide`` (integral-image
    box stats — the click-time SSD scorer costs ~ms/region, too slow for
    hundreds of detected regions). Mask-freedom alone let a patch full of real
    detail win: the membrane corrects only the boundary offset, so interior
    structure gets cloned into the heal. Score = |Δmean vs the destination
    background| + texture in excess of the destination's."""
    h, w = mask.shape
    m8 = np.ascontiguousarray(mask, dtype=np.uint8)
    integ = cv2.integral(m8)
    bg = guide.astype(np.float32) * (1.0 - m8)
    s1, s2 = cv2.integral2(bg)

    def box(ii, x0, y0, x1, y1):
        return float(ii[y1 + 1, x1 + 1] - ii[y0, x1 + 1] - ii[y1 + 1, x0] + ii[y0, x0])

    offsets = []
    for index, (chain, radius, _area) in enumerate(comps):
        if len(chain) >= 2:
            d = chain[-1] - chain[0]
            ln = math.hypot(d[0], d[1])
            tx, ty = (d[0] / ln, d[1] / ln) if ln > 1e-6 else (1.0, 0.0)
            dirs = [(-ty, tx), (ty, -tx), (tx, ty), (-tx, -ty)]
        else:
            dirs = []
            for k in range(6):
                ang = _GOLDEN_ANGLE * (index + 1) + k * math.pi / 3.0
                dirs.append((math.cos(ang), math.sin(ang)))

        b = int(math.ceil(1.2 * radius)) + 1
        area = float((2 * b + 1) ** 2)

        d_n = d_s = d_ss = 0.0
        for px, py in chain:
            x0, x1 = max(int(px) - b, 0), min(int(px) + b, w - 1)
            y0, y1 = max(int(py) - b, 0), min(int(py) + b, h - 1)
            d_n += (x1 - x0 + 1) * (y1 - y0 + 1) - box(integ, x0, y0, x1, y1)
            d_s += box(s1, x0, y0, x1, y1)
            d_ss += box(s2, x0, y0, x1, y1)
        d_mean = d_s / max(d_n, 1.0)
        d_std = math.sqrt(max(d_ss / max(d_n, 1.0) - d_mean * d_mean, 0.0))

        found, best_score = None, math.inf
        for ring in range(3):
            dist = (_FALLBACK_OFFSET_FACTOR + ring) * radius
            for dx, dy in dirs:
                ox, oy = dx * dist, dy * dist
                n = s = ss = 0.0
                clean = True
                for px, py in chain:
                    sx, sy = px + ox, py + oy
                    x0, x1 = int(sx) - b, int(sx) + b
                    y0, y1 = int(sy) - b, int(sy) + b
                    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
                        clean = False
                        break
                    if box(integ, x0, y0, x1, y1) > 0:
                        clean = False
                        break
                    n += area
                    s += box(s1, x0, y0, x1, y1)
                    ss += box(s2, x0, y0, x1, y1)
                if not clean:
                    continue
                mean = s / n
                std = math.sqrt(max(ss / n - mean * mean, 0.0))
                score = abs(mean - d_mean) + max(0.0, std - d_std)
                if score < best_score:
                    best_score, found = score, (ox, oy)
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
    """Threshold-independent detection stat maps (proxy + blur windows) — the
    expensive ~2/3 of a detection pass, cacheable across threshold changes."""
    proxy = _detection_proxy(img)
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
    )


def detect_luma_regions(
    img: ImageBuffer,
    dust_threshold: float,
    dust_size: int,
    gate: float = 1.0,
    max_n: int = 512,
    stats: Optional[Tuple[np.ndarray, ...]] = None,
) -> List[Tuple]:
    """Statistical dust detection on the linear source → synthesized heal strokes."""
    proxy, mean, std, w_std = stats if stats is not None else compute_dust_stats(img, dust_size)
    hit = _detect_dust_mask_jit(proxy, mean, std, w_std, float(dust_threshold))
    if not np.any(hit):
        return []
    comps = _mask_to_strokes(hit, _DETECT_PAD_PX, max_n)
    offsets = _pick_source_offsets(hit, comps, proxy)
    return _finalize_strokes(comps, offsets, hit.shape, gate)


def detect_ir_regions(
    ir: np.ndarray,
    threshold: float,
    pad_px: float = 3.0,
    max_n: int = 512,
    guide: Optional[np.ndarray] = None,
) -> List[Tuple]:
    """IR defects → synthesized heal strokes. Dye = high IR transmittance,
    defects = low, so ``ir < threshold`` marks them (caller passes
    1 − ir_threshold). Ungated: IR-confirmed defects clone unconditionally.
    ``guide`` (the visible source's detection proxy, same dims) scores clone
    sources by content; without it the IR plane itself is the guide."""
    mask = (ir < threshold).astype(np.uint8)
    if not np.any(mask):
        return []
    if guide is None or guide.shape[:2] != ir.shape[:2]:
        guide = np.ascontiguousarray(ir, dtype=np.float32)
    comps = _mask_to_strokes(mask, pad_px, max_n)
    offsets = _pick_source_offsets(mask, comps, guide)
    return _finalize_strokes(comps, offsets, mask.shape, gate=0.0)


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
