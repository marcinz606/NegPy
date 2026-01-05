import numpy as np
import cv2
from typing import Tuple, Optional
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
    Calculates the autocrop coordinates.
    Returns (y1, y2, x1, x2) in pixels relative to input img.
    """
    img = ensure_rgb(img)
    h, w, _ = img.shape
    detect_res = APP_CONFIG.autocrop_detect_res
    det_scale = detect_res / max(h, w)
    img_small = ensure_array(
        cv2.resize(
            img, (int(w * det_scale), int(h * det_scale)), interpolation=cv2.INTER_AREA
        )
    )
    lum = get_luminance(img_small)

    # Threshold for film base detection (detecting darker frame)
    rows_det = np.where(np.mean(lum, axis=1) < 0.96)[0]
    cols_det = np.where(np.mean(lum, axis=0) < 0.96)[0]

    if len(rows_det) < 10 or len(cols_det) < 10:
        return 0, h, 0, w

    y1, y2 = rows_det[0] / det_scale, rows_det[-1] / det_scale
    x1, x2 = cols_det[0] / det_scale, cols_det[-1] / det_scale

    margin = (2 + offset_px) * scale_factor
    y1, y2, x1, x2 = y1 + margin, y2 - margin, x1 + margin, x2 - margin
    cw, ch = x2 - x1, y2 - y1

    if cw <= 0 or ch <= 0:
        return 0, h, 0, w

    # Parse target ratio
    try:
        w_r, h_r = map(float, target_ratio_str.split(":"))
        target_aspect = w_r / h_r
    except Exception as e:
        logger.error(
            f"Invalid aspect ratio: {target_ratio_str}, defaulting to 3:2. Error: {e}"
        )
        target_aspect = 1.5  # Default 3:2

    # Handle Orientation: Automatically flip ratio to match image orientation
    is_vertical = ch > cw
    if is_vertical:
        if target_aspect > 1.0:
            target_aspect = 1.0 / target_aspect
    else:
        if target_aspect < 1.0:
            target_aspect = 1.0 / target_aspect

    # Enforce Ratio
    current_aspect = cw / ch

    if current_aspect > target_aspect:
        # Too wide, crop width
        target_w = ch * target_aspect
        x1 = x1 + (cw - target_w) // 2
        x2 = x1 + target_w
    else:
        # Too tall, crop height
        target_h = cw / target_aspect
        y1 = y1 + (ch - target_h) // 2
        y2 = y1 + target_h

    return int(max(0, y1)), int(min(h, y2)), int(max(0, x1)), int(min(w, x2))


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


def map_coords_to_geometry(
    nx: float,
    ny: float,
    orig_shape: Tuple[int, int],
    rotation_k: int = 0,
    fine_rotation: float = 0.0,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[float, float]:
    """


    Maps normalized coordinates (0-1) from the original image space


    to the current geometric state (rotated and optionally cropped).


    """

    h_orig, w_orig = orig_shape

    # 1. Start with absolute pixels in original space

    px, py = nx * w_orig, ny * h_orig

    h, w = h_orig, w_orig

    # 2. Apply 90-degree rotations (match np.rot90 which is CCW)

    k = rotation_k % 4

    if k == 1:  # 90 CCW
        # Original (x, y) in (W, H) -> (y, W-x) in (H, W)

        px, py = py, w - px

        h, w = w, h

    elif k == 2:  # 180
        # Original (x, y) in (W, H) -> (W-x, H-y) in (W, H)

        px, py = w - px, h - py

    elif k == 3:  # 270 CCW (90 CW)
        # Original (x, y) in (W, H) -> (H-y, x) in (H, W)

        px, py = h - py, px

        h, w = w, h

    # 3. Apply Fine Rotation

    if fine_rotation != 0.0:
        center = (w / 2.0, h / 2.0)

        m_mat = cv2.getRotationMatrix2D(center, fine_rotation, 1.0)

        # To map a point from src -> dst, we use M @ P

        pt = np.array([px, py, 1.0])

        res_pt = m_mat @ pt

        px, py = float(res_pt[0]), float(res_pt[1])

    # 4. Apply ROI Offset (if we want coordinates relative to the crop)

    if roi:
        y1, y2, x1, x2 = roi

        px -= x1

        py -= y1

        h, w = y2 - y1, x2 - x1

    # 5. Return normalized to the NEW shape

    # Clip to [0, 1] to ensure we stay within image bounds

    nx_new = np.clip(px / max(w, 1), 0.0, 1.0)

    ny_new = np.clip(py / max(h, 1), 0.0, 1.0)

    logger.debug(
        f"MapCoords: ({nx:.3f}, {ny:.3f}) -> ({nx_new:.3f}, {ny_new:.3f}) [k={k}, fine={fine_rotation:.1f}, roi={roi}]"
    )

    return float(nx_new), float(ny_new)
