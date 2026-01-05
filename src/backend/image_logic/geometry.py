import numpy as np
import cv2
from typing import Tuple
from src.config import APP_CONFIG
from src.helpers import ensure_rgb, get_luminance, ensure_array
from src.logging_config import get_logger

logger = get_logger(__name__)


def apply_fine_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates the image by a specific angle (in degrees).
    Keeps the original image dimensions, filling new areas with black.

    Args:
        img (np.ndarray): Input image array (float [0, 1]).
        angle (float): Rotation angle in degrees.

    Returns:
        np.ndarray: Rotated image.
    """
    if angle == 0.0:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return ensure_array(
        cv2.warpAffine(
            img,
            m_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    )


def get_autocrop_coords(
    img: np.ndarray,
    offset_px: int = 0,
    scale_factor: float = 1.0,
    target_ratio_str: str = "3:2",
) -> Tuple[int, int, int, int]:
    """
    Center-Out Valley Stop Solver.

    Starts from the image center and expands outwards until it hits the "Valley"
    (the black film borders). This implicitly ignores the "White Scanner Bed"
    which lies beyond the black borders.
    """
    img = ensure_rgb(img)
    h_orig, w_orig, _ = img.shape

    # 1. Preparation: Detect on a resized proxy for speed
    detect_res = APP_CONFIG.autocrop_detect_res
    det_scale = detect_res / max(h_orig, w_orig)
    img_small = ensure_array(
        cv2.resize(
            img,
            (int(w_orig * det_scale), int(h_orig * det_scale)),
            interpolation=cv2.INTER_AREA,
        )
    )

    # Ensure float32 0.0-1.0
    if img_small.dtype != np.float32:
        img_small = img_small.astype(np.float32) / (
            65535.0 if img_small.dtype == np.uint16 else 255.0
        )

    lum = get_luminance(img_small)
    h_det, w_det = lum.shape

    # Compute 1D Projection Profiles
    row_means = np.mean(lum, axis=1)
    col_means = np.mean(lum, axis=0)

    # 2. Define the "Border Floor"
    # We look for the 'Valley' (black rebate).
    # If the darkest part of the profile is still bright, no borders exist.
    global_min = min(np.min(row_means), np.min(col_means))

    if global_min > 0.20:
        # Image is likely already cropped to content
        return 0, h_orig, 0, w_orig

    # The threshold to stop: slightly above the deepest black of the border
    border_threshold = global_min + 0.12

    # 3. Center-Out Search with Patience
    # Patience prevents stopping on thin scratches or dust inside the image.
    y_patience = max(3, int(h_det * 0.005))
    x_patience = max(3, int(w_det * 0.005))

    def walk_to_valley(
        profile: np.ndarray,
        start: int,
        direction: int,
        threshold: float,
        patience: int,
    ) -> int:
        count = 0
        curr = start
        limit = len(profile)
        while 0 <= curr < limit:
            if profile[curr] < threshold:
                count += 1
            else:
                count = 0  # Reset patience if we hit a bright pixel (image content)

            if count >= patience:
                # Return the index where we first crossed the threshold
                return curr - (direction * (patience - 1))
            curr += direction
        return 0 if direction == -1 else limit - 1

    mid_y, mid_x = h_det // 2, w_det // 2

    gy1 = walk_to_valley(row_means, mid_y, -1, border_threshold, y_patience)
    gy2 = walk_to_valley(row_means, mid_y, 1, border_threshold, y_patience)
    gx1 = walk_to_valley(col_means, mid_x, -1, border_threshold, x_patience)
    gx2 = walk_to_valley(col_means, mid_x, 1, border_threshold, x_patience)

    # Apply manual user offset (scaled)
    final_margin = 2 + offset_px
    gy1, gy2 = gy1 + final_margin, gy2 - final_margin
    gx1, gx2 = gx1 + final_margin, gx2 - final_margin

    # 4. Aspect Ratio Fit
    # Translate Small-Image coordinates back to Original Image pixels
    y1, y2, x1, x2 = (
        gy1 / det_scale,
        gy2 / det_scale,
        gx1 / det_scale,
        gx2 / det_scale,
    )
    cw, ch = x2 - x1, y2 - y1

    if cw <= 0 or ch <= 0:
        return 0, h_orig, 0, w_orig

    # Parse target ratio
    try:
        w_r, h_r = map(float, target_ratio_str.split(":"))
        target_aspect = w_r / h_r
    except Exception:
        target_aspect = 1.5  # Default 3:2

    # Orientation Matching
    if (ch > cw) != (h_r > w_r):
        target_aspect = 1.0 / target_aspect

    # Center-aligned fit maximally inside the detected valley bounds
    cx, cy = x1 + cw / 2.0, y1 + ch / 2.0
    current_aspect = cw / ch

    if current_aspect > target_aspect:
        # Detected area is wider than target: limit by height
        final_h = ch
        final_w = ch * target_aspect
    else:
        # Detected area is taller than target: limit by width
        final_w = cw
        final_h = cw / target_aspect

    # Final integer pixel coordinates constrained to original image bounds
    res_x1 = int(max(0, cx - final_w / 2.0))
    res_x2 = int(min(w_orig, cx + final_w / 2.0))
    res_y1 = int(max(0, cy - final_h / 2.0))
    res_y2 = int(min(h_orig, cy + final_h / 2.0))

    return res_y1, res_y2, res_x1, res_x2


def apply_autocrop(
    img: np.ndarray, offset_px: int = 0, scale_factor: float = 1.0, ratio: str = "3:2"
) -> np.ndarray:
    """
    Detects film edges and automatically crops the image to the specified aspect ratio frame.

    Args:
        img (np.ndarray): Input image array.
        offset_px (int): Additional margin to add to the crop.
        scale_factor (float): Scaling factor for the current processing resolution.
        ratio (str): Target aspect ratio string (e.g., "3:2", "6:7").

    Returns:
        np.ndarray: Cropped image array.
    """
    y1, y2, x1, x2 = get_autocrop_coords(img, offset_px, scale_factor, ratio)
    return img[y1:y2, x1:x2]
