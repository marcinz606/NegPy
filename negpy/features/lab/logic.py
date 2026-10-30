import math
from typing import Callable, Optional

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.lab.models import LabConfig, SharpenMethod
from negpy.features.process.models import ProcessMode
from negpy.kernel.image.logic import (
    WORKING_INV_GAMMA,
    WORKING_LAB,
    gamut_aware_chroma_scale,
    lab_to_rgb_working,
    rgb_to_lab_working,
    skin_chroma_rein,
    working_oetf_decode,
    working_oetf_encode,
)
from negpy.kernel.image.validation import ensure_image


def apply_clahe(img: ImageBuffer, strength: float) -> ImageBuffer:
    """
    L-channel Contrast Limited Adaptive Histogram Equalization.
    Fixed 8x8 tile grid over the full frame at every scale; mirrors the
    clahe_*.wgsl shaders bin-for-bin so CPU and GPU stay in parity.
    """
    if strength <= 0:
        return img
    return negpy_core.clahe(np.ascontiguousarray(img, dtype=np.float32), strength, *WORKING_LAB)


# Sharpen constants, mirrored as WGSL consts in shaders/lab.wgsl; the USM gate, overshoot and
# edge-mask limits live in negpy_core/src/lab.rs.
# Gain rolls off toward paper black, where the thin negative's grain lives: floor at
# L* 0, full gain from SHARPEN_SHADOW_L_HI up (Gallagher & Gindele, US 7,228,004).
SHARPEN_SHADOW_FLOOR = 1.0 / 3.0
SHARPEN_SHADOW_L_HI = 35.0


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


# Richardson-Lucy noise floor (linear luminance), mirroring RL_EPS in the WGSL shaders.
RL_EPS = 1e-6
# Adobe RGB (1998) -> XYZ D65 luminance row (Yn = 1). Mirrors lab_sharpen_h.wgsl / rl_init.wgsl.
LUM_R, LUM_G, LUM_B = 0.2973769, 0.6273491, 0.0752741


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - np.float32(e0)) / np.float32(e1 - e0), 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def sharpen_shadow_gain(l_chan: np.ndarray) -> np.ndarray:
    """Per-pixel sharpen gain multiplier from L*; mirrors sharpen_shadow_gain in lab.wgsl."""
    f = np.float32(SHARPEN_SHADOW_FLOOR)
    return f + (np.float32(1.0) - f) * _smoothstep(0.0, SHARPEN_SHADOW_L_HI, l_chan)


def rl_iterations(radius: float) -> int:
    """Deterministic RL iteration count from the user radius (not the scaled σ),
    so preview and export run identical counts. Shared by CPU and GPU."""
    return int(np.clip(int(round(10.0 * radius)), 5, 20))


def _map_row_blocks(img: ImageBuffer, fn: Callable[[ImageBuffer], ImageBuffer], halo: int = 0) -> ImageBuffer:
    rows = max(1, 1048576 // img.shape[1])
    if img.shape[0] <= rows:
        return fn(img)
    output = np.empty(img.shape, dtype=np.float32)
    for start in range(0, img.shape[0], rows):
        end = min(start + rows, img.shape[0])
        low, high = max(0, start - halo), min(img.shape[0], end + halo)
        block = fn(img[low:high])
        output[start:end] = block[start - low : end - low]
    return output


def apply_usm_sharpening(img: ImageBuffer, amount: float, radius: float = 1.0, masking: float = 0.0) -> ImageBuffer:
    if amount <= 0:
        return img
    # Gaussian support, local range, and Sobel-plus-box support must cross block edges.
    halo = max(len(gaussian_kernel_1d(radius)) // 2, 2)
    return _map_row_blocks(img, lambda block: _usm_sharpening_block(block, amount, radius, masking), halo)


# Auto Masking: the masking that holds back the grain is this gain times the measured grain
# gradient. sharpen_grain_pick.wgsl mirrors both.
AUTO_MASKING_GAIN = 0.27
NEUTRAL_MASKING = LabConfig().sharpen_masking


def auto_masking(img: ImageBuffer, config: LabConfig) -> Optional[float]:
    """The masking Auto Masking picks for this buffer, or None when it is off. Without a measure it
    picks the neutral masking."""
    if not config.sharpen_auto_masking:
        return None
    lum = (np.float32(LUM_R), np.float32(LUM_G), np.float32(LUM_B))
    grain = negpy_core.sharpen_grain(np.ascontiguousarray(img, dtype=np.float32), lum)
    return NEUTRAL_MASKING if grain is None else min(AUTO_MASKING_GAIN * grain, 1.0)


def effective_masking(config: LabConfig, auto: Optional[float]) -> float:
    if auto is None:
        return config.sharpen_masking
    return min(max(auto + config.sharpen_masking - NEUTRAL_MASKING, 0.0), 1.0)


def apply_lab_sharpening(img: ImageBuffer, config: LabConfig, auto: Optional[float] = None) -> ImageBuffer:
    """The Lab Sharpening of `config` on linear working RGB, at this buffer's pixel scale. `auto`
    is auto_masking of this buffer."""
    sharpen = apply_rl_sharpening if config.sharpen_method == SharpenMethod.RL else apply_usm_sharpening
    return sharpen(img, config.sharpen, radius=config.sharpen_radius, masking=effective_masking(config, auto))


def apply_resize_sharpening(encoded: np.ndarray, config: LabConfig) -> np.ndarray:
    """
    The Lab Sharpening on export content already scaled to its final size, so its radius is in
    exported pixels. Takes and returns working-OETF code values; a channel after RGB passes through.
    """
    if config.sharpen <= 0:
        return encoded
    linear = working_oetf_decode(np.ascontiguousarray(encoded[..., :3]))
    sharpened = working_oetf_encode(apply_lab_sharpening(linear, config, auto_masking(linear, config)))
    if encoded.shape[2] == 3:
        return sharpened
    return np.concatenate([sharpened, encoded[..., 3:]], axis=2)


def _usm_sharpening_block(
    img: ImageBuffer,
    amount: float,
    radius: float = 1.0,
    masking: float = 0.0,
) -> ImageBuffer:
    """L-channel soft-gated USM, its overshoot clamped to the local 3x3 range, with an optional edge mask for flat
    areas. Radius is in pixels of this buffer, before any export resize; judge sharpening at 1:1.
    lab_sharpen_h/v.wgsl and lab.wgsl mirror this."""
    if amount <= 0:
        return img
    k = gaussian_kernel_1d(radius)
    return negpy_core.usm_sharpen(np.ascontiguousarray(img, dtype=np.float32), np.float32(amount * 2.5), masking, k, *WORKING_LAB)


def apply_rl_sharpening(
    img: ImageBuffer,
    amount: float,
    radius: float = 1.0,
    masking: float = 0.0,
) -> ImageBuffer:
    """Richardson-Lucy deconvolution of linear Y (Gaussian PSF of width radius), applied as an RGB ratio so chroma
    holds. Iterations are fixed by radius and the edge mask governs grain, so there is no damping.
    rl_*.wgsl mirrors this."""
    if amount <= 0:
        return img
    lum = (np.float32(LUM_R), np.float32(LUM_G), np.float32(LUM_B))
    k = gaussian_kernel_1d(radius)
    return negpy_core.rl_sharpen(
        np.ascontiguousarray(img, dtype=np.float32), np.float32(amount), masking, k, rl_iterations(radius), lum, np.float32(RL_EPS)
    )


def apply_saturation(img: ImageBuffer, saturation: float, skin_protection: float = 0.0) -> ImageBuffer:
    if saturation == 1.0 and skin_protection <= 0.0:
        return img
    return _map_row_blocks(img, lambda block: _saturation_block(block, saturation, skin_protection))


def _saturation_block(img: ImageBuffer, saturation: float, skin_protection: float = 0.0) -> ImageBuffer:
    """
    Adjusts saturation by scaling chroma (a*, b*) in CIELAB.
    Preserves perceived lightness, unlike HSV S-scaling which darkens
    already-saturated colors when S clips to 1.0.

    Gamut-aware above 1.0 (see gamut_aware_chroma_scale): a flat a*/b* scale
    preserves hue mathematically, but pixels that overshoot the display gamut
    get hard-clamped per channel afterward, which shifts the *actual* hue
    since clamping only the overshooting channel(s) changes the R:G:B ratio.
    A soft per-pixel knee toward each pixel's real in-gamut headroom avoids
    that instead of clamping after the fact.

    Skin protection (see skin_chroma_rein) then reins skin-hued chroma in
    against a ceiling. It shares this one Lab round trip but is independent of
    the scale: it runs at saturation 1.0 too, since skin can arrive
    over-chromatic from the print curve without Chroma pushing it there.
    """
    if saturation == 1.0 and skin_protection <= 0.0:
        return img

    lab = rgb_to_lab_working(np.ascontiguousarray(img, dtype=np.float32))
    if saturation != 1.0:
        lab = gamut_aware_chroma_scale(lab, saturation)
    if skin_protection > 0.0:
        lab = skin_chroma_rein(lab, skin_protection)
    res_rgb = np.ascontiguousarray(lab_to_rgb_working(lab), dtype=np.float32)
    negpy_core.clip_unit(res_rgb)
    return ensure_image(res_rgb)


# Below this blur sigma (px) both engines skip the stage.
CHROMA_DENOISE_MIN_SIGMA = 0.25


def chroma_denoise_radius(radius: float, process_mode: ProcessMode) -> float:
    """B&W has no chroma to denoise, so the stage never runs there."""
    return 0.0 if process_mode == ProcessMode.BW else radius


def chroma_denoise_sigma(radius: float, scale_factor: float) -> float:
    """The low-pass sigma in render pixels for both engines; 0.0 when the stage does not run."""
    sigma = radius * scale_factor
    return sigma if sigma >= CHROMA_DENOISE_MIN_SIGMA else 0.0


def apply_chroma_denoise(img: ImageBuffer, radius: float, scale_factor: float = 1.0) -> ImageBuffer:
    """
    Band split of CIELAB a*/b* (Fuji, US 5,739,922): chroma above a Gaussian cutoff is noise or
    grain, except where it belongs to a color edge. L* is untouched, so grain stays as luminance.
    """
    sigma = chroma_denoise_sigma(radius, scale_factor)
    if sigma == 0.0:
        return img

    k = gaussian_kernel_1d(sigma)
    return negpy_core.chroma_denoise(np.ascontiguousarray(img, dtype=np.float32), sigma, k, *WORKING_LAB)


# Base-reflection bounces, (weight, sigma over HALATION_SIGMA). Mirrored in shaders/lab.wgsl.
HALATION_SIGMA = 0.0027  # of the short side: 65 um on 35 mm
HALATION_BOUNCES = ((4.0 / 7.0, 1.0), (2.0 / 7.0, math.sqrt(2.0)), (1.0 / 7.0, math.sqrt(3.0)))
HALATION_AMOUNT_MAX = 0.2
HALATION_TINT = (1.0, 0.3, 0.05)


def halation_amounts(strength: float, bw: bool) -> tuple[float, float, float]:
    """Per-channel mix at a Halation value; neutral, same luminance, on B&W."""
    r, g, b = (HALATION_AMOUNT_MAX * float(strength) * t for t in HALATION_TINT)
    if bw:
        y = r * LUM_R + g * LUM_G + b * LUM_B
        return (y, y, y)
    return (r, g, b)


def halation_sigma_px(shape: tuple[int, ...]) -> float:
    return HALATION_SIGMA * min(shape[:2])


def halation_reach_px(shape: tuple[int, ...]) -> int:
    return int(math.ceil(3.0 * HALATION_BOUNCES[-1][1] * halation_sigma_px(shape)))


# Glow: a mist filter's scatter, Gaussian fit of spektrafilm's Black Pro-Mist halo and bloom
# without its core, as (weight, sigma over the short side). Mirrored in shaders/lab.wgsl.
GLOW_PSF = ((0.1104, 0.001582), (0.3559, 0.004122), (0.4014, 0.010181), (0.1323, 0.024798))
GLOW_AMOUNT_MAX = 0.25
GLOW_THRESHOLD = 0.5  # display-encoded luma where the highlights start to scatter


def glow_reach_px(shape: tuple[int, ...]) -> int:
    return int(math.ceil(3.0 * GLOW_PSF[-1][1] * min(shape[:2])))


def apply_glow_and_halation(img: ImageBuffer, glow_amount: float, halation_strength: float, bw: bool = False) -> ImageBuffer:
    """Glow spreads the highlights, halation all light, each I + a·(K*L − L), so a flat area keeps its value."""
    if glow_amount == 0.0 and halation_strength == 0.0:
        return img

    glow = halation = None
    if glow_amount > 0.0:
        a = GLOW_AMOUNT_MAX * float(glow_amount)
        glow = (((a, a, a), [(w, s * min(img.shape[:2])) for w, s in GLOW_PSF]), GLOW_THRESHOLD)
    if halation_strength > 0.0:
        sigma = halation_sigma_px(img.shape)
        halation = (halation_amounts(halation_strength, bw), [(w, k * sigma) for w, k in HALATION_BOUNCES])
    lum = (np.float32(LUM_R), np.float32(LUM_G), np.float32(LUM_B))
    return negpy_core.glow_and_halation(np.ascontiguousarray(img, dtype=np.float32), glow, halation, lum, WORKING_INV_GAMMA)
