import numpy as np
import cv2
from src.config import ImageSettings
from src.helpers import get_luminance, ensure_array
from src.logging_config import get_logger

logger = get_logger(__name__)


def apply_dust_removal(
    img: np.ndarray, params: ImageSettings, scale_factor: float
) -> np.ndarray:
    """
    Applies both automatic and manual dust removal (healing).
    Automatic uses median blur replacement, manual uses inpainting with grain matching.
    """
    manual_spots = params.manual_dust_spots
    if not (params.dust_remove or manual_spots):
        return img

    # --- Automatic Detection & Healing ---
    if params.dust_remove:
        d_size = int(params.dust_size * 2.0 * scale_factor) | 1
        img_uint8 = np.clip(np.nan_to_num(img * 255), 0, 255).astype(np.uint8)
        img_median_u8: np.ndarray = cv2.medianBlur(img_uint8, d_size)
        img_median = img_median_u8.astype(np.float32) / 255.0

        gray = get_luminance(img)
        blur_win = int(15 * scale_factor) | 1
        mean = cv2.blur(gray, (blur_win, blur_win))
        sq_mean = cv2.blur(gray**2, (blur_win, blur_win))
        std = np.sqrt(np.clip(sq_mean - mean**2, 0, None))
        flatness = np.clip(1.0 - (std / 0.08), 0, 1)
        flatness_weight = np.sqrt(flatness)
        brightness = np.clip(gray, 0, 1)
        highlight_sens = np.clip((brightness - 0.4) * 1.5, 0, 1)
        detail_boost = (1.0 - flatness) * 0.05
        sens_factor = (1.0 - 0.98 * flatness_weight) * (1.0 - 0.5 * highlight_sens)
        adaptive_thresh = params.dust_threshold * sens_factor + detail_boost

        diff = np.max(np.abs(img - img_median), axis=2)
        raw_mask = (diff > adaptive_thresh).astype(np.float32)
        exclusion_mask = (std > 0.2).astype(np.float32)
        raw_mask = raw_mask * (1.0 - exclusion_mask)

        if np.any(raw_mask > 0):
            m_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, m_kernel_close)
            m_kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, m_kernel_dilate, iterations=2)
            feather = d_size | 1
            mask = cv2.GaussianBlur(mask, (feather, feather), 0)
            img = img * (1.0 - mask[:, :, None]) + img_median * mask[:, :, None]

    # --- Manual Healing (using Inpainting) ---
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
        img_inpainted_u8 = cv2.inpaint(
            img_u8, manual_mask_u8, inpaint_rad, cv2.INPAINT_TELEA
        )

        # Grain Matching
        noise_arr = ensure_array(np.random.normal(0, 3.5, img_inpainted_u8.shape))
        noise_f32 = noise_arr.astype(np.float32)
        lum_arr = get_luminance(img_inpainted_u8) / 255.0
        mod_arr = 5.0 * lum_arr * (1.0 - lum_arr)
        final_noise = noise_f32 * mod_arr[:, :, None]

        mask_base = manual_mask_u8.astype(np.float32) / 255.0
        mask_3d = ensure_array(mask_base)[:, :, None]
        mask_blur = ensure_array(cv2.GaussianBlur(mask_3d, (3, 3), 0)).astype(
            np.float32
        )
        if mask_blur.ndim == 2:
            mask_final = mask_blur[:, :, None]
        else:
            mask_final = mask_blur

        img_inpainted_f = img_inpainted_u8.astype(np.float32) + final_noise * mask_final
        img = np.clip(img_inpainted_f, 0, 255) / 255.0

    return img
