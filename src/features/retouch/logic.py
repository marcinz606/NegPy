import numpy as np
import cv2
from numba import njit, prange  # type: ignore
from typing import List, Tuple
from src.domain.types import ImageBuffer, LUMA_R, LUMA_G, LUMA_B
from src.kernel.image.validation import ensure_image
from src.kernel.image.logic import get_luminance


@njit(parallel=True, cache=True, fastmath=True)
def _compute_dust_masks_jit(
    img: np.ndarray,
    img_ref: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    w_std: np.ndarray,
    dust_threshold: float,
    dust_size: float,
    scale_factor: float,
) -> np.ndarray:
    h, w, c = img.shape
    hit_mask = np.zeros((h, w), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            max_pos_diff = 0.0
            for ch in range(3):
                d = img[y, x, ch] - img_ref[y, x, ch]
                if d > max_pos_diff:
                    max_pos_diff = d

            l_curr = (
                0.2126 * img[y, x, 0] + 0.7152 * img[y, x, 1] + 0.0722 * img[y, x, 2]
            )

            # Match GPU: thresh = (threshold * 0.4) + (local_s * 1.0) + wide_penalty
            w_s = max(0.0, w_std[y, x] - 0.02)
            wide_penalty = (w_s * w_s * w_s) * 800.0

            local_s = max(0.005, std[y, x])
            z_score = (l_curr - mean[y, x]) / local_s
            thresh = (dust_threshold * 0.4) + (local_s * 1.0) + wide_penalty

            if max_pos_diff > thresh and l_curr > 0.15 and z_score > 3.0:
                # Strong Signal Bypass: catch hairs/plateaus that aren't single-pixel peaks
                # but are definitively brighter than the background.
                is_strong = max_pos_diff > (thresh * 2.5) or max_pos_diff > 0.25

                if y > 0 and y < h - 1 and x > 0 and x < w - 1:
                    # Strict 3x3 local maximum check
                    is_max = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                            neighbor_l = (
                                0.2126 * img[y + dy, x + dx, 0]
                                + 0.7152 * img[y + dy, x + dx, 1]
                                + 0.0722 * img[y + dy, x + dx, 2]
                            )
                            if neighbor_l >= l_curr:
                                is_max = False
                                break
                        if not is_max:
                            break

                    if is_max or is_strong:
                        hit_mask[y, x] = 1.0
                else:
                    hit_mask[y, x] = 1.0

    # 2. Expansion radius scales with dust_size to cover larger footprints
    exp_rad = int(max(1.0, dust_size * 0.25 * scale_factor))
    if exp_rad > 6:
        exp_rad = 6
    res_mask = np.zeros((h, w), dtype=np.float32)
    for y in prange(exp_rad, h - exp_rad):
        for x in range(exp_rad, w - exp_rad):
            # Check neighbors for a detection hit
            has_hit = False
            for dy in range(-exp_rad, exp_rad + 1):
                for dx in range(-exp_rad, exp_rad + 1):
                    if hit_mask[y + dy, x + dx] > 0.5:
                        has_hit = True
                        break
                if has_hit:
                    break

            if has_hit:
                res_mask[y, x] = 1.0

    return res_mask


@njit(parallel=True, cache=True, fastmath=True)
def _apply_inpainting_grain_jit(
    img: np.ndarray,
    img_inpainted: np.ndarray,
    mask_final: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    h, w, c = img_inpainted.shape
    res = np.empty_like(img_inpainted)

    for y in prange(h):
        for x in range(w):
            lum = (
                LUMA_R * img_inpainted[y, x, 0]
                + LUMA_G * img_inpainted[y, x, 1]
                + LUMA_B * img_inpainted[y, x, 2]
            ) / 255.0

            mod = 3.0 * lum * (1.0 - lum)
            m = mask_final[y, x, 0]

            orig_luma = (
                LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
            )
            heal_luma = (
                LUMA_R * img_inpainted[y, x, 0]
                + LUMA_G * img_inpainted[y, x, 1]
                + LUMA_B * img_inpainted[y, x, 2]
            ) / 255.0

            diff = orig_luma - heal_luma
            luma_key = (diff - 0.04) / 0.08
            if luma_key < 0.0:
                luma_key = 0.0
            elif luma_key > 1.0:
                luma_key = 1.0

            final_m = m * luma_key

            for ch in range(3):
                val = img_inpainted[y, x, ch] + noise[y, x, ch] * 0.4 * mod * final_m
                res[y, x, ch] = (
                    img[y, x, ch] * (1.0 - final_m) + (val / 255.0) * final_m
                )

    return res


def apply_dust_removal(
    img: ImageBuffer,
    dust_remove: bool,
    dust_threshold: float,
    dust_size: int,
    manual_spots: List[Tuple[float, float, float]],
    scale_factor: float,
) -> ImageBuffer:
    if not (dust_remove or manual_spots):
        return img

    if dust_remove:
        base_size = max(1.0, float(dust_size))
        scale = max(1.0, float(scale_factor))

        v_win = int(max(3, base_size * 3.0 * scale)) * 2 + 1
        w_win = int(max(7, base_size * 4.0 * scale)) * 2 + 1

        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        if scale > 1.5:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            img_ref_u8 = cv2.erode(img_uint8, kernel)
            img_ref = img_ref_u8.astype(np.float32) / 255.0

            r_off = int(base_size * 3.0 * scale)
            for dy, dx in [(-r_off, -r_off), (r_off, r_off)]:
                shifted = np.roll(img, (dy, dx), axis=(0, 1))
                img_ref = np.minimum(img_ref, shifted)
        else:
            d_rad = int(base_size * 3.0 * scale) | 1
            img_ref_u8 = cv2.medianBlur(img_uint8, d_rad)
            img_ref = img_ref_u8.astype(np.float32) / 255.0

        gray = get_luminance(img)
        mean_gray = cv2.blur(gray, (v_win, v_win))
        sq_mean_gray = cv2.blur(gray**2, (v_win, v_win))
        std_gray = np.sqrt(np.clip(sq_mean_gray - mean_gray**2, 0, None))

        w_mean_gray = cv2.blur(gray, (w_win, w_win))
        w_sq_mean_gray = cv2.blur(gray**2, (w_win, w_win))
        w_std_gray = np.sqrt(np.clip(w_sq_mean_gray - w_mean_gray**2, 0, None))

        mask = _compute_dust_masks_jit(
            np.ascontiguousarray(img.astype(np.float32)),
            np.ascontiguousarray(img_ref.astype(np.float32)),
            np.ascontiguousarray(mean_gray.astype(np.float32)),
            np.ascontiguousarray(std_gray.astype(np.float32)),
            np.ascontiguousarray(w_std_gray.astype(np.float32)),
            float(dust_threshold),
            float(dust_size),
            float(scale_factor),
        )

        if np.any(mask > 0):
            feather = int(base_size * 2.0 * scale) | 1
            mask_soft = cv2.GaussianBlur(mask, (feather, feather), 0)
            img = img * (1.0 - mask_soft[:, :, None]) + img_ref * mask_soft[:, :, None]

    if manual_spots:
        h_img, w_img = img.shape[:2]

        manual_mask_u8 = np.zeros((h_img, w_img), dtype=np.uint8)
        for spot in manual_spots:
            nx, ny, s_size = spot
            radius = int(s_size * scale_factor)

            if radius < 1:
                radius = 1

            px = int(nx * w_img)
            py = int(ny * h_img)
            cv2.circle(manual_mask_u8, (px, py), radius, 255, -1)

        img_u8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        inpaint_rad = int(3 * scale_factor) | 1
        img_inpainted_u8 = ensure_image(
            cv2.inpaint(img_u8, manual_mask_u8, inpaint_rad, cv2.INPAINT_TELEA)
        )

        noise_arr = np.random.normal(0, 3.5, img_inpainted_u8.shape).astype(np.float32)

        mask_base = manual_mask_u8.astype(np.float32) / 255.0
        mask_3d = mask_base[:, :, None]

        feather_size = inpaint_rad | 1
        mask_blur = cv2.GaussianBlur(mask_3d, (feather_size, feather_size), 0)
        mask_final = (
            mask_blur[:, :, None] if mask_blur.ndim == 2 else mask_blur
        ).astype(np.float32)

        img = ensure_image(
            _apply_inpainting_grain_jit(
                np.ascontiguousarray(img.astype(np.float32)),
                np.ascontiguousarray(img_inpainted_u8.astype(np.float32)),
                np.ascontiguousarray(mask_final.astype(np.float32)),
                np.ascontiguousarray(noise_arr.astype(np.float32)),
            )
        )

    return ensure_image(img)
