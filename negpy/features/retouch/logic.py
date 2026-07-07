import math

import numpy as np
import cv2
from numba import njit  # type: ignore
from typing import List, Optional, Tuple
from negpy.domain.types import ImageBuffer, LUMA_R, LUMA_G, LUMA_B
from negpy.features.geometry.logic import map_coords_to_geometry
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.image.logic import get_luminance, working_oetf_decode, working_oetf_encode

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


@njit(cache=True, fastmath=True)
def _hash2(x: float, y: float) -> float:
    """Port of the WGSL hash() so degenerate-direction picks match the GPU."""
    px = (x * 0.1031) % 1.0
    py = (y * 0.1031) % 1.0
    pz = (x * 0.1031) % 1.0
    d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33)
    px += d
    py += d
    pz += d
    return ((px + py) * pz) % 1.0


@njit(cache=True, fastmath=True)
def _heal_with_mask_jit(
    img: np.ndarray,
    hit_mask: np.ndarray,
    exp_rad: int,
    p_rad: int,
) -> np.ndarray:
    """Guarded reflection-copy heal with cubic-smoothstep feather.

    For each pixel, finds the nearest masked pixel within ``exp_rad`` and the
    centroid of masked pixels in the window, then copies the pixel at
    ``p + normalize(p - centroid) * p_rad`` — a coherent outward copy that
    preserves grain/texture. If the source lands on another defect the
    direction is rotated (±45°, ±90°); if all rotations fail it falls back to
    the old 8-point trimmed mean, so the worst case equals the previous fill.
    """
    h, w, _ = img.shape
    res = img.copy()

    cos_a = np.empty(5, dtype=np.float64)
    sin_a = np.empty(5, dtype=np.float64)
    angles = (0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0, -math.pi / 2.0)
    for i in range(5):
        cos_a[i] = math.cos(angles[i])
        sin_a[i] = math.sin(angles[i])

    for y in range(h):
        for x in range(w):
            min_d2 = 1e6
            c_x = 0.0
            c_y = 0.0
            c_n = 0.0
            for dy in range(-exp_rad, exp_rad + 1):
                for dx in range(-exp_rad, exp_rad + 1):
                    ry, rx = y + dy, x + dx
                    if 0 <= ry < h and 0 <= rx < w and hit_mask[ry, rx] > 0.5:
                        d2 = float(dy * dy + dx * dx)
                        if d2 < min_d2:
                            min_d2 = d2
                        c_x += float(rx)
                        c_y += float(ry)
                        c_n += 1.0

            if min_d2 < float(exp_rad * exp_rad + 1):
                dist = np.sqrt(min_d2)
                feather = 1.0 - (dist / float(exp_rad + 1.0))
                if feather < 0.0:
                    feather = 0.0
                feather = feather * feather * (3.0 - 2.0 * feather)

                if feather > 0.001:
                    ux = float(x) - c_x / c_n
                    uy = float(y) - c_y / c_n
                    ul = math.sqrt(ux * ux + uy * uy)
                    if ul < 1e-3:
                        ang = _hash2(float(x), float(y)) * 6.28318530718
                        ux = math.cos(ang)
                        uy = math.sin(ang)
                    else:
                        ux /= ul
                        uy /= ul

                    found = False
                    bg_r = 0.0
                    bg_g = 0.0
                    bg_b = 0.0
                    for k in range(5):
                        rx_dir = ux * cos_a[k] - uy * sin_a[k]
                        ry_dir = ux * sin_a[k] + uy * cos_a[k]
                        sx = int(round(float(x) + rx_dir * float(p_rad)))
                        sy = int(round(float(y) + ry_dir * float(p_rad)))
                        sx = max(0, min(w - 1, sx))
                        sy = max(0, min(h - 1, sy))
                        if hit_mask[sy, sx] < 0.5:
                            bg_r = img[sy, sx, 0]
                            bg_g = img[sy, sx, 1]
                            bg_b = img[sy, sx, 2]
                            found = True
                            break

                    if not found:
                        s_r = np.zeros(8)
                        s_g = np.zeros(8)
                        s_b = np.zeros(8)
                        s_l = np.zeros(8)

                        dy_off = np.array([-p_rad, p_rad, 0, 0, -p_rad, -p_rad, p_rad, p_rad])
                        dx_off = np.array([0, 0, -p_rad, p_rad, -p_rad, p_rad, -p_rad, p_rad])

                        for i in range(8):
                            sy2, sx2 = y + dy_off[i], x + dx_off[i]
                            sy2, sx2 = max(0, min(h - 1, sy2)), max(0, min(w - 1, sx2))
                            r, g, b = img[sy2, sx2, 0], img[sy2, sx2, 1], img[sy2, sx2, 2]
                            s_r[i], s_g[i], s_b[i] = r, g, b
                            s_l[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b

                        for i in range(8):
                            for j in range(i + 1, 8):
                                if s_l[i] > s_l[j]:
                                    s_l[i], s_l[j] = s_l[j], s_l[i]
                                    s_r[i], s_r[j] = s_r[j], s_r[i]
                                    s_g[i], s_g[j] = s_g[j], s_g[i]
                                    s_b[i], s_b[j] = s_b[j], s_b[i]

                        bg_r = (s_r[2] + s_r[3] + s_r[4] + s_r[5]) / 4.0
                        bg_g = (s_g[2] + s_g[3] + s_g[4] + s_g[5]) / 4.0
                        bg_b = (s_b[2] + s_b[3] + s_b[4] + s_b[5]) / 4.0

                    res[y, x, 0] = img[y, x, 0] * (1.0 - feather) + bg_r * feather
                    res[y, x, 1] = img[y, x, 1] * (1.0 - feather) + bg_g * feather
                    res[y, x, 2] = img[y, x, 2] * (1.0 - feather) + bg_b * feather

    return res


@njit(cache=True, fastmath=True)
def _apply_auto_retouch_jit(
    img_det: np.ndarray,
    img_heal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    w_std: np.ndarray,
    dust_threshold: float,
    dust_size: float,
    scale_factor: float,
) -> np.ndarray:
    """Detect on the display-encoded ``img_det`` (perceptual), heal ``img_heal`` (linear)."""
    h, w, _ = img_det.shape
    hit_mask = np.zeros((h, w), dtype=np.float32)

    # 1. Detection Pass
    for y in range(h):
        for x in range(w):
            l_curr = LUMA_R * img_det[y, x, 0] + LUMA_G * img_det[y, x, 1] + LUMA_B * img_det[y, x, 2]
            l_mean = mean[y, x]
            local_s = max(0.005, std[y, x])

            # Wide-area penalty for textures (rocks, foliage)
            w_s = max(0.0, w_std[y, x] - 0.02)
            wide_penalty = (w_s * w_s * w_s) * 800.0
            thresh = (dust_threshold * 0.4) + (local_s * 1.0) + wide_penalty

            # Multi-stage validation: Contrast, Luminance, and Z-Score
            if (l_curr - l_mean) > thresh and l_curr > 0.15 and (l_curr - l_mean) / local_s > 3.0:
                is_strong = (l_curr - l_mean) > (thresh * 2.5) or (l_curr - l_mean) > 0.25

                if 0 < y < h - 1 and 0 < x < w - 1:
                    is_max = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                            nl = (
                                LUMA_R * img_det[y + dy, x + dx, 0]
                                + LUMA_G * img_det[y + dy, x + dx, 1]
                                + LUMA_B * img_det[y + dy, x + dx, 2]
                            )
                            if nl >= l_curr:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if is_max or is_strong:
                        hit_mask[y, x] = 1.0
                else:
                    hit_mask[y, x] = 1.0

    exp_rad = int(max(1.0, dust_size * 0.4 * scale_factor))
    if exp_rad > 16:
        exp_rad = 16
    p_rad = exp_rad + int(3 * scale_factor)

    return _heal_with_mask_jit(img_heal, hit_mask, exp_rad, p_rad)


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
    ``reg_f``: (R, 3) float32 — radius_px, src_off_x, src_off_y (pixels).
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

                # 1.5px feather at the rim hides boundary-sampling aliasing.
                t = (d - (rad - 1.5)) / 1.5
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
                # clean value — the brush is a search area, not a clone stamp.
                dest_l = LUMA_R * buf[y, x, 0] + LUMA_G * buf[y, x, 1] + LUMA_B * buf[y, x, 2]
                heal_l = LUMA_R * hr + LUMA_G * hg + LUMA_B * hb
                g = (dest_l - heal_l - _HEAL_GATE_LO) / (_HEAL_GATE_HI - _HEAL_GATE_LO)
                if g < 0.0:
                    g = 0.0
                elif g > 1.0:
                    g = 1.0
                alpha *= g * g * (3.0 - 2.0 * g)
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


def build_heal_regions(
    strokes: List[Tuple],
    legacy_spots: List[Tuple[float, float, float]],
    orig_shape: Tuple[int, int],
    rotation: int,
    fine_rotation: float,
    flip_h: bool,
    flip_v: bool,
    distortion_k1: float,
    scale_factor: float,
    full_dims: Tuple[int, int],
    max_regions: int = 512,
    max_points: int = 16384,
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

    entries: List[Tuple[List, float, float, float]] = []
    for points, size, sdx, sdy in strokes:
        entries.append((list(points), float(size), float(sdx), float(sdy)))
    for i, (nx, ny, size) in enumerate(legacy_spots):
        fdx, fdy = fallback_source_offset(i, float(size), orig_shape)
        entries.append(([[nx, ny]], float(size), fdx, fdy))

    reg_i_list = []
    reg_f_list = []
    pts_list: List[np.ndarray] = []
    n_pts = 0

    for points, size, sdx, sdy in entries[:max_regions]:
        chain = np.array([_map(p[0], p[1]) for p in points], dtype=np.float32)
        # Brush size is a DIAMETER: the healed footprint must match the on-screen
        # cursor circle (overlay._brush_screen_radius draws size/2 at preview scale).
        radius = max(1.0, float(size) * float(scale_factor) * 0.5)

        cx = float(np.mean([p[0] for p in points]))
        cy = float(np.mean([p[1] for p in points]))
        c_px = _map(cx, cy)
        s_px = _map(cx + sdx, cy + sdy)
        off_x, off_y = s_px[0] - c_px[0], s_px[1] - c_px[1]

        seg = np.diff(chain, axis=0)
        perimeter = 2.0 * float(np.hypot(seg[:, 0], seg[:, 1]).sum()) + 2.0 * np.pi * radius
        n_bnd = int(min(64, max(16, perimeter / 4.0)))
        boundary = _capsule_boundary(chain.astype(np.float64), radius, n_bnd)

        if n_pts + len(chain) + len(boundary) > max_points:
            break
        reg_i_list.append((n_pts, len(chain), n_pts + len(chain), len(boundary)))
        reg_f_list.append((radius, off_x, off_y))
        pts_list.append(chain)
        pts_list.append(boundary)
        n_pts += len(chain) + len(boundary)

    if not reg_i_list:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
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
        # Heavy penalty for dust inside the candidate patch: interior lumas that
        # pop above the candidate band's median mean the patch contains a speck.
        med = float(np.median(band_lums))
        for cx_, cy_ in interior:
            sx, sy = cx_ + cdx, cy_ + cdy
            if not (0 <= sx < w - 1 and 0 <= sy < h - 1):
                valid = False
                break
            excess = float(np.dot(preview_img[int(sy), int(sx)][:3], luma_w)) - med - _CLONE_GUARD_LUMA
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


def apply_ir_dust_removal(
    img: ImageBuffer,
    ir: np.ndarray,
    threshold: float,
    inpaint_radius: int,
    scale_factor: float,
) -> Tuple[ImageBuffer, np.ndarray]:
    """Threshold IR → guarded reflection-copy heal with cubic-smoothstep feather.

    Returns (img_out, mask_u8). IR convention: dye = high IR transmittance,
    physical defects = low transmittance, so `ir < threshold` marks defects.
    Mask must be in the same frame as `img` (i.e. post-geometry).
    """
    if ir.shape[:2] != img.shape[:2]:
        return img, np.zeros(img.shape[:2], dtype=np.uint8)

    hit_mask = (ir < threshold).astype(np.float32)
    mask_u8 = (hit_mask * 255).astype(np.uint8)

    if not np.any(hit_mask):
        return img, mask_u8

    scale = max(1.0, float(scale_factor))
    exp_rad = int(max(1.0, float(inpaint_radius) * scale))
    if exp_rad > 16:
        exp_rad = 16
    p_rad = exp_rad + int(max(2.0, 3.0 * scale))

    out = _heal_with_mask_jit(
        np.ascontiguousarray(img.astype(np.float32)),
        np.ascontiguousarray(hit_mask),
        exp_rad,
        p_rad,
    )
    return ensure_image(out), mask_u8


def apply_dust_removal(
    img: ImageBuffer,
    dust_remove: bool,
    dust_threshold: float,
    dust_size: int,
    heal_regions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scale_factor: float,
    ir_buffer: Optional[np.ndarray] = None,
    ir_dust_remove: bool = False,
    ir_threshold: float = 0.55,
    ir_inpaint_radius: int = 3,
) -> ImageBuffer:
    """Composite dust removal: luminance-auto → IR → manual heals."""
    do_ir = ir_dust_remove and ir_buffer is not None
    has_manual = heal_regions is not None and len(heal_regions[0]) > 0
    if not (dust_remove or has_manual or do_ir):
        return img

    if dust_remove:
        base_size, scale = max(1.0, float(dust_size)), max(1.0, float(scale_factor))
        v_win = int(max(3, base_size * 3.0 * scale)) * 2 + 1
        w_win = int(max(7, base_size * 4.0 * scale)) * 2 + 1

        # Detection is perceptual: run it on a display-encoded copy, heal in linear.
        img_enc = ensure_image(working_oetf_encode(img))
        gray = get_luminance(img_enc)
        mean_gray = cv2.blur(gray, (v_win, v_win))
        std_gray = np.sqrt(np.clip(cv2.blur(gray**2, (v_win, v_win)) - mean_gray**2, 0, None))
        w_mean_gray = cv2.blur(gray, (w_win, w_win))
        w_std_gray = np.sqrt(np.clip(cv2.blur(gray**2, (w_win, w_win)) - w_mean_gray**2, 0, None))

        img = _apply_auto_retouch_jit(
            np.ascontiguousarray(img_enc.astype(np.float32)),
            np.ascontiguousarray(img.astype(np.float32)),
            np.ascontiguousarray(mean_gray.astype(np.float32)),
            np.ascontiguousarray(std_gray.astype(np.float32)),
            np.ascontiguousarray(w_std_gray.astype(np.float32)),
            float(dust_threshold),
            float(dust_size),
            float(scale_factor),
        )

    if do_ir and ir_buffer is not None:
        img, _ = apply_ir_dust_removal(
            img,
            ir_buffer,
            ir_threshold,
            ir_inpaint_radius,
            scale_factor,
        )

    if has_manual and heal_regions is not None:
        img = apply_manual_heals(img, *heal_regions)

    return ensure_image(img)
