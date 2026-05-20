import numpy as np
import cv2
from numba import njit  # type: ignore
from typing import List, Optional, Tuple
from negpy.domain.types import ImageBuffer, LUMA_R, LUMA_G, LUMA_B
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.image.logic import get_luminance


@njit(cache=True, fastmath=True)
def _heal_with_mask_jit(
    img: np.ndarray,
    hit_mask: np.ndarray,
    exp_rad: int,
    p_rad: int,
) -> np.ndarray:
    """Stochastic perimeter sampling with cubic-smoothstep feather.

    For each pixel, finds nearest masked pixel within ``exp_rad``, builds a
    feather weight from the distance, samples 8 perimeter points at ``p_rad``,
    trim-means them, and blends into the original.
    """
    h, w, _ = img.shape
    res = img.copy()

    for y in range(h):
        for x in range(w):
            min_d2 = 1e6
            for dy in range(-exp_rad, exp_rad + 1):
                for dx in range(-exp_rad, exp_rad + 1):
                    ry, rx = y + dy, x + dx
                    if 0 <= ry < h and 0 <= rx < w and hit_mask[ry, rx] > 0.5:
                        d2 = float(dy * dy + dx * dx)
                        if d2 < min_d2:
                            min_d2 = d2

            if min_d2 < float(exp_rad * exp_rad + 1):
                dist = np.sqrt(min_d2)
                feather = 1.0 - (dist / float(exp_rad + 1.0))
                if feather < 0.0:
                    feather = 0.0
                feather = feather * feather * (3.0 - 2.0 * feather)

                if feather > 0.001:
                    s_r = np.zeros(8)
                    s_g = np.zeros(8)
                    s_b = np.zeros(8)
                    s_l = np.zeros(8)

                    dy_off = np.array([-p_rad, p_rad, 0, 0, -p_rad, -p_rad, p_rad, p_rad])
                    dx_off = np.array([0, 0, -p_rad, p_rad, -p_rad, p_rad, -p_rad, p_rad])

                    for i in range(8):
                        sy, sx = y + dy_off[i], x + dx_off[i]
                        sy, sx = max(0, min(h - 1, sy)), max(0, min(w - 1, sx))
                        r, g, b = img[sy, sx, 0], img[sy, sx, 1], img[sy, sx, 2]
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
    img: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    w_std: np.ndarray,
    dust_threshold: float,
    dust_size: float,
    scale_factor: float,
) -> np.ndarray:
    h, w, _ = img.shape
    hit_mask = np.zeros((h, w), dtype=np.float32)

    # 1. Detection Pass
    for y in range(h):
        for x in range(w):
            l_curr = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
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
                            nl = LUMA_R * img[y + dy, x + dx, 0] + LUMA_G * img[y + dy, x + dx, 1] + LUMA_B * img[y + dy, x + dx, 2]
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

    return _heal_with_mask_jit(img, hit_mask, exp_rad, p_rad)


@njit(cache=True, fastmath=True)
def _apply_inpainting_grain_jit(
    img: np.ndarray,
    img_inpainted: np.ndarray,
    mask_final: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    h, w, c = img_inpainted.shape
    res = np.empty_like(img_inpainted)

    for y in range(h):
        for x in range(w):
            lum = (LUMA_R * img_inpainted[y, x, 0] + LUMA_G * img_inpainted[y, x, 1] + LUMA_B * img_inpainted[y, x, 2]) / 255.0
            mod = 3.0 * lum * (1.0 - lum)
            m = mask_final[y, x, 0]

            orig_luma = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
            heal_luma = (LUMA_R * img_inpainted[y, x, 0] + LUMA_G * img_inpainted[y, x, 1] + LUMA_B * img_inpainted[y, x, 2]) / 255.0

            luma_key = (orig_luma - heal_luma - 0.04) / 0.08
            if luma_key < 0.0:
                luma_key = 0.0
            if luma_key > 1.0:
                luma_key = 1.0

            final_m = m * luma_key

            for ch in range(3):
                val = img_inpainted[y, x, ch] + noise[y, x, ch] * 0.4 * mod * final_m
                res[y, x, ch] = img[y, x, ch] * (1.0 - final_m) + (val / 255.0) * final_m

    return res


def _inpaint_with_mask(img: ImageBuffer, mask_u8: np.ndarray, inpaint_rad: int) -> ImageBuffer:
    """Telea inpaint + grain restoration. Shared by manual_spots and IR paths."""
    rad = max(1, inpaint_rad) | 1
    img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
    img_inpainted_u8 = ensure_image(cv2.inpaint(img_u8, mask_u8, rad, cv2.INPAINT_TELEA))

    noise_arr = np.random.normal(0, 3.5, img_inpainted_u8.shape).astype(np.float32)
    mask_base = mask_u8.astype(np.float32) / 255.0
    mask_blur = cv2.GaussianBlur(mask_base, (rad, rad), 0)
    if mask_blur.ndim == 2:
        mask_final = mask_blur[:, :, None].astype(np.float32)
    else:
        mask_final = mask_blur.astype(np.float32)

    return ensure_image(
        _apply_inpainting_grain_jit(
            np.ascontiguousarray(img.astype(np.float32)),
            np.ascontiguousarray(img_inpainted_u8.astype(np.float32)),
            np.ascontiguousarray(mask_final.astype(np.float32)),
            np.ascontiguousarray(noise_arr.astype(np.float32)),
        )
    )


def apply_ir_dust_removal(
    img: ImageBuffer,
    ir: np.ndarray,
    threshold: float,
    inpaint_radius: int,
    scale_factor: float,
) -> Tuple[ImageBuffer, np.ndarray]:
    """Threshold IR → perimeter-sample inpaint with cubic-smoothstep feather.

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
    manual_spots: List[Tuple[float, float, float]],
    scale_factor: float,
    ir_buffer: Optional[np.ndarray] = None,
    ir_dust_remove: bool = False,
    ir_threshold: float = 0.55,
    ir_inpaint_radius: int = 3,
) -> ImageBuffer:
    """Composite dust removal: luminance-auto → IR → manual spots."""
    do_ir = ir_dust_remove and ir_buffer is not None
    if not (dust_remove or manual_spots or do_ir):
        return img

    if dust_remove:
        base_size, scale = max(1.0, float(dust_size)), max(1.0, float(scale_factor))
        v_win = int(max(3, base_size * 3.0 * scale)) * 2 + 1
        w_win = int(max(7, base_size * 4.0 * scale)) * 2 + 1

        gray = get_luminance(img)
        mean_gray = cv2.blur(gray, (v_win, v_win))
        std_gray = np.sqrt(np.clip(cv2.blur(gray**2, (v_win, v_win)) - mean_gray**2, 0, None))
        w_mean_gray = cv2.blur(gray, (w_win, w_win))
        w_std_gray = np.sqrt(np.clip(cv2.blur(gray**2, (w_win, w_win)) - w_mean_gray**2, 0, None))

        img = _apply_auto_retouch_jit(
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

    if manual_spots:
        h_img, w_img = img.shape[:2]
        manual_mask_u8 = np.zeros((h_img, w_img), dtype=np.uint8)
        for spot in manual_spots:
            nx, ny, s_size = spot
            radius = int(max(1, s_size * scale_factor))
            cv2.circle(manual_mask_u8, (int(nx * w_img), int(ny * h_img)), radius, 255, -1)

        inpaint_rad = int(3 * scale_factor)
        img = _inpaint_with_mask(img, manual_mask_u8, inpaint_rad)

    return ensure_image(img)
