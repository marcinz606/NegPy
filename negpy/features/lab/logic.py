import math

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.logic import lab_to_rgb_working, rgb_to_lab_working, working_oetf_encode
from negpy.kernel.image.validation import ensure_image


CLAHE_GRID = 8
CLAHE_BINS = 256


def _clahe_cdfs(bins: np.ndarray, clip_limit: float) -> np.ndarray:
    """
    Per-tile clipped CDFs, (64, 256) float32. Mirrors clahe_hist.wgsl /
    clahe_cdf.wgsl exactly (integer counts, f32-truncated limit, excess
    redistributed evenly with the remainder going to the first bins).
    """
    h, w = bins.shape
    tsy, tsx = (h + CLAHE_GRID - 1) // CLAHE_GRID, (w + CLAHE_GRID - 1) // CLAHE_GRID
    ty = (np.arange(h) // tsy).astype(np.int32)
    tx = (np.arange(w) // tsx).astype(np.int32)
    comb = (ty[:, None] * CLAHE_GRID + tx[None, :]) * CLAHE_BINS + bins
    hist = np.bincount(comb.ravel(), minlength=CLAHE_GRID * CLAHE_GRID * CLAHE_BINS)
    hist = hist.reshape(CLAHE_GRID * CLAHE_GRID, CLAHE_BINS)

    total = hist.sum(axis=1)
    limit = np.maximum(1, (np.float32(clip_limit) * total.astype(np.float32) / np.float32(CLAHE_BINS)).astype(np.int64))
    clipped = np.minimum(hist, limit[:, None])
    excess = (hist - clipped).sum(axis=1)
    inc, rem = excess // CLAHE_BINS, excess % CLAHE_BINS
    counts = clipped + inc[:, None] + (np.arange(CLAHE_BINS)[None, :] < rem[:, None])
    return np.cumsum(counts, axis=1).astype(np.float32) / np.maximum(total, 1)[:, None].astype(np.float32)


def _clahe_axis(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # tile_pos = pos/dims*8 - 0.5, smoothstep frac; true floor (can be -1), clamp 0..7.
    tp = np.arange(n, dtype=np.float32) / np.float32(n) * np.float32(CLAHE_GRID) - np.float32(0.5)
    tf = np.floor(tp)
    rf = tp - tf
    fr = rf * rf * (np.float32(3.0) - np.float32(2.0) * rf)
    lo = np.maximum(tf.astype(np.int32), 0)
    hi = np.minimum(tf.astype(np.int32) + 1, CLAHE_GRID - 1)
    return lo, hi, fr


def apply_clahe(img: ImageBuffer, strength: float) -> ImageBuffer:
    """
    L-channel Contrast Limited Adaptive Histogram Equalization.
    Fixed 8x8 tile grid over the full frame at every scale; mirrors the
    clahe_*.wgsl shaders bin-for-bin so CPU and GPU stay in parity.
    """
    if strength <= 0:
        return img

    lab = rgb_to_lab_working(img)
    l_chan = lab[..., 0]
    h, w = l_chan.shape
    bins = np.clip(l_chan / np.float32(100.0) * np.float32(255.0), 0.0, 255.0).astype(np.int32)
    cdfs = _clahe_cdfs(bins, strength * 2.5).reshape(-1)

    y0, y1, fy = _clahe_axis(h)
    x0, x1, fx = _clahe_axis(w)
    v00 = cdfs[(y0[:, None] * CLAHE_GRID + x0[None, :]) * CLAHE_BINS + bins]
    v10 = cdfs[(y0[:, None] * CLAHE_GRID + x1[None, :]) * CLAHE_BINS + bins]
    v01 = cdfs[(y1[:, None] * CLAHE_GRID + x0[None, :]) * CLAHE_BINS + bins]
    v11 = cdfs[(y1[:, None] * CLAHE_GRID + x1[None, :]) * CLAHE_BINS + bins]
    top = v00 + (v10 - v00) * fx[None, :]
    bot = v01 + (v11 - v01) * fx[None, :]
    cdf_l = top + (bot - top) * fy[:, None]

    lab[..., 0] = l_chan + (cdf_l * np.float32(100.0) - l_chan) * np.float32(strength)
    return ensure_image(np.clip(lab_to_rgb_working(lab), 0.0, 1.0))


# Sharpen constants — mirrored as WGSL consts in shaders/lab.wgsl.
SHARPEN_GATE_LO = 1.5
SHARPEN_GATE_HI = 2.0
# L*-domain USM exaggerates light halos, so overshoot above the local max is
# clamped tighter than undershoot below the local min.
SHARPEN_OVERSHOOT_LIGHT = 1.0
SHARPEN_OVERSHOOT_DARK = 2.0
SHARPEN_MASK_T_HI = 10.0


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """
    Single source of truth for the sharpen blur taps: the CPU path convolves
    with this array and the GPU path uploads it verbatim (sharpen_k buffer),
    so kernel support and weights match bit-for-bit on both sides.
    """
    r = max(1, min(255, int(math.ceil(2.5 * sigma))))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / np.float32(2.0 * sigma * sigma)).astype(np.float32)
    return k / np.float32(k.sum())


# Richardson-Lucy noise floor (linear luminance) — mirrors RL_EPS in the WGSL shaders.
RL_EPS = 1e-6
# Adobe RGB (1998) -> XYZ D65 luminance row (Yn = 1). Mirrors lab_sharpen_h.wgsl / rl_init.wgsl.
LUM_R, LUM_G, LUM_B = 0.2973769, 0.6273491, 0.0752741


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - np.float32(e0)) / np.float32(e1 - e0), 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def _lab_l_from_y(y: np.ndarray) -> np.ndarray:
    """CIELAB L* from linear luminance Y (D65, Yn=1) — mirrors lab_l_from_y in the shaders."""
    y = np.maximum(y, 0.0)
    f = np.where(y > 0.008856, np.cbrt(y), np.float32(7.787) * y + np.float32(16.0 / 116.0))
    return (np.float32(116.0) * f - np.float32(16.0)).astype(np.float32)


def _edge_mask(l_chan: np.ndarray, masking: float, scale_factor: float) -> np.ndarray:
    """Boxed |∇L*| edge mask (smoothstep over 0.5t..t, t=10·masking); shared by
    both sharpen methods. Mirrors the WGSL boxed-gradient loop."""
    lp = np.pad(l_chan, 1, mode="edge")
    gx = (lp[1:-1, 2:] - lp[1:-1, :-2]) * np.float32(0.5)
    gy = (lp[2:, 1:-1] - lp[:-2, 1:-1]) * np.float32(0.5)
    grad = cv2.blur(np.hypot(gx, gy).astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)
    t = SHARPEN_MASK_T_HI * masking
    return _smoothstep(0.5 * t, t, grad * np.float32(scale_factor))


def rl_iterations(radius: float) -> int:
    """Deterministic RL iteration count from the user radius (not the scaled σ),
    so preview and export run identical counts. Shared by CPU and GPU."""
    return int(np.clip(int(round(10.0 * radius)), 5, 20))


def apply_output_sharpening(
    img: ImageBuffer,
    amount: float,
    scale_factor: float = 1.0,
    radius: float = 1.0,
    masking: float = 0.0,
) -> ImageBuffer:
    """
    L-channel unsharp mask; mirrors lab_sharpen_h/v.wgsl + the lab.wgsl sharpen
    block. Soft-gated USM with an overshoot clamp to the local 3x3 range (halo
    suppression) and an optional edge mask (boxed |∇L|) protecting flat areas.
    """
    if amount <= 0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    l_chan, a, b = cv2.split(lab)

    k = gaussian_kernel_1d(radius * scale_factor)
    l_blur = cv2.sepFilter2D(l_chan, -1, k, k, borderType=cv2.BORDER_REFLECT_101)

    diff = l_chan - l_blur
    gain = np.float32(amount * 2.5) * _smoothstep(SHARPEN_GATE_LO, SHARPEN_GATE_HI, np.abs(diff))

    if masking > 0.0:
        gain = gain * _edge_mask(l_chan, masking, scale_factor)

    kern3 = np.ones((3, 3), np.uint8)
    l_min = cv2.erode(l_chan, kern3, borderType=cv2.BORDER_REPLICATE)
    l_max = cv2.dilate(l_chan, kern3, borderType=cv2.BORDER_REPLICATE)

    l_new = l_chan + diff * gain
    l_new = np.clip(l_new, l_min - np.float32(SHARPEN_OVERSHOOT_DARK), l_max + np.float32(SHARPEN_OVERSHOOT_LIGHT))
    l_new = np.clip(l_new, 0.0, 100.0)

    res_lab = cv2.merge([l_new.astype(np.float32), a, b])
    res_rgb = lab_to_rgb_working(res_lab)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))


def apply_rl_sharpening(
    img: ImageBuffer,
    amount: float,
    scale_factor: float = 1.0,
    radius: float = 1.0,
    masking: float = 0.0,
) -> ImageBuffer:
    """
    Richardson-Lucy deconvolution on linear luminance Y (Gaussian PSF), applied
    as an RGB ratio so chroma is preserved. Mirrors rl_*.wgsl. Iterations are
    fixed by radius (rl_iterations); no per-pixel early stop or damping — the
    edge mask governs grain, matching RawTherapee's shipped configuration.
    """
    if amount <= 0:
        return img

    rgb = img.astype(np.float32)
    obs = (
        np.maximum(rgb[..., 0], 0.0) * np.float32(LUM_R)
        + np.maximum(rgb[..., 1], 0.0) * np.float32(LUM_G)
        + np.maximum(rgb[..., 2], 0.0) * np.float32(LUM_B)
    ).astype(np.float32)

    k = gaussian_kernel_1d(radius * scale_factor)
    est = obs.copy()
    for _ in range(rl_iterations(radius)):
        blurred = cv2.sepFilter2D(est, -1, k, k, borderType=cv2.BORDER_REFLECT_101)
        corr = cv2.sepFilter2D(obs / np.maximum(blurred, np.float32(RL_EPS)), -1, k, k, borderType=cv2.BORDER_REFLECT_101)
        est = est * corr

    ratio = est / np.maximum(obs, np.float32(RL_EPS))
    gain = np.float32(amount)
    if masking > 0.0:
        gain = gain * _edge_mask(_lab_l_from_y(obs), masking, scale_factor)

    factor = np.maximum(np.float32(1.0) + (ratio - np.float32(1.0)) * gain, 0.0)
    out = rgb * factor[..., np.newaxis]
    return ensure_image(np.clip(out, 0.0, 1.0))


def apply_saturation(img: ImageBuffer, saturation: float) -> ImageBuffer:
    """
    Adjusts saturation by scaling chroma (a*, b*) in CIELAB.
    Preserves perceived lightness, unlike HSV S-scaling which darkens
    already-saturated colors when S clips to 1.0.
    """
    if saturation == 1.0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    l_chan, a, b = cv2.split(lab)
    a_new = a * saturation
    b_new = b * saturation
    res_lab = cv2.merge([l_chan, a_new, b_new])
    res_rgb = lab_to_rgb_working(res_lab)
    return ensure_image(np.clip(res_rgb, 0.0, 1.0))


def apply_chroma_denoise(img: ImageBuffer, radius: float, scale_factor: float = 1.0) -> ImageBuffer:
    """
    Smooths A and B channels in LAB space to reduce color noise.
    """
    if radius <= 0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    l_chan, a, b = cv2.split(lab)

    k_radius = radius * scale_factor
    k_size = max(3, int(k_radius * 2 + 1) | 1)
    sigma = k_radius

    a_blur = cv2.GaussianBlur(a, (k_size, k_size), sigma)
    b_blur = cv2.GaussianBlur(b, (k_size, k_size), sigma)

    res_lab = cv2.merge([l_chan, a_blur, b_blur])
    res_rgb = lab_to_rgb_working(res_lab)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))


# Halation mask threshold in LINEAR reflectance: regions the negative rendered
# dense (near paper white on the print). Thresholding linear light keeps the
# halation footprint fixed by scene exposure instead of moving with grade/density.
HALATION_THRESHOLD_LINEAR = 0.65


def apply_glow_and_halation(
    img: ImageBuffer,
    glow_amount: float,
    halation_strength: float,
    scale_factor: float = 1.0,
) -> ImageBuffer:
    """
    Glow: all-channel Gaussian bloom of highlights (lens diffusion, a print-side
    effect — its mask stays in the display domain).
    Halation: red-dominant scatter of highlights (light reflecting off the film
    base at capture — masked in linear light, composited additively: scattered
    light is added exposure, not an opacity composite).
    """
    if glow_amount == 0.0 and halation_strength == 0.0:
        return img

    result = img.copy().astype(np.float32)

    if glow_amount > 0.0:
        # Highlight mask in the display domain (keeps the 0.5 threshold); bloom is linear.
        enc = working_oetf_encode(img)
        luma = enc[:, :, 0] * 0.2126 + enc[:, :, 1] * 0.7152 + enc[:, :, 2] * 0.0722
        threshold = 0.5
        glow_mask = np.clip((luma - threshold) / (1.0 - threshold), 0.0, 1.0) ** 2
        base_r = max(3, int(15 * scale_factor))
        k = min((base_r * 2 + 1) | 1, 201)
        sigma = base_r * 0.5
        highlights = (img * glow_mask[:, :, np.newaxis]).astype(np.float32)
        glow_blur = cv2.GaussianBlur(highlights, (k, k), sigma)
        result = result + glow_blur * glow_amount

    if halation_strength > 0.0:
        lin_luma = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
        t = HALATION_THRESHOLD_LINEAR
        hal_mask = np.clip((lin_luma - t) / (1.0 - t), 0.0, 1.0) ** 2
        base_r = max(5, int(25 * scale_factor))
        k = min((base_r * 2 + 1) | 1, 301)
        sigma = base_r * 0.5
        red_hl = np.zeros_like(img, dtype=np.float32)
        red_hl[:, :, 0] = img[:, :, 0] * hal_mask
        red_hl[:, :, 1] = img[:, :, 0] * hal_mask * 0.3
        red_hl[:, :, 2] = img[:, :, 0] * hal_mask * 0.05
        hal_blur = cv2.GaussianBlur(red_hl, (k, k), sigma)
        result = result + hal_blur * halation_strength

    return ensure_image(np.clip(result, 0.0, 1.0))


def apply_vibrance(img: ImageBuffer, strength: float) -> ImageBuffer:
    """
    Selectively boosts saturation of muted colors in LAB space.
    """
    if strength == 1.0:
        return img

    lab = rgb_to_lab_working(img.astype(np.float32))
    l_chan, a, b = cv2.split(lab)

    chroma = np.sqrt(a**2 + b**2)
    muted_mask = np.clip(1.0 - (chroma / 60.0), 0.0, 1.0)

    boost = (strength - 1.0) * muted_mask
    a_new = a * (1.0 + boost)
    b_new = b * (1.0 + boost)

    res_lab = cv2.merge([l_chan, a_new, b_new])
    res_rgb = lab_to_rgb_working(res_lab)

    return ensure_image(np.clip(res_rgb, 0.0, 1.0))
