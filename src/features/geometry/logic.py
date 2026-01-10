import numpy as np
import cv2
from typing import Tuple, Optional
from src.domain.types import ImageBuffer, ROI
from src.kernel.image.validation import ensure_image
from src.kernel.system.performance import time_function
from src.kernel.image.logic import get_luminance


@time_function
def apply_fine_rotation(img: ImageBuffer, angle: float) -> ImageBuffer:
    """
    Rotates the image by a specific angle (in degrees).

    Used for horizon leveling and precise alignment. Uses bilinear interpolation
    to preserve fine photographic detail during the transformation.
    """
    if angle == 0.0:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    res = cv2.warpAffine(
        img,
        m_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return ensure_image(res)


def apply_margin_to_roi(
    roi: ROI,
    h: int,
    w: int,
    margin_px: float,
) -> ROI:
    """
    Applies a uniform margin (in pixels) to an ROI, ensuring it stays within bounds.
    Positive margin crops IN, negative margin expands OUT.
    """
    y1, y2, x1, x2 = roi
    ny1, ny2, nx1, nx2 = y1 + margin_px, y2 - margin_px, x1 + margin_px, x2 - margin_px
    return int(max(0, ny1)), int(min(h, ny2)), int(max(0, nx1)), int(min(w, nx2))


def enforce_roi_aspect_ratio(
    roi: ROI,
    h: int,
    w: int,
    target_ratio_str: str = "3:2",
) -> ROI:
    """
    Adjusts the ROI to match a specific aspect ratio, centering the crop.
    """
    y1, y2, x1, x2 = roi
    cw, ch = x2 - x1, y2 - y1

    if cw <= 0 or ch <= 0:
        return 0, h, 0, w

    try:
        w_r, h_r = map(float, target_ratio_str.split(":"))
        target_aspect = w_r / h_r
    except Exception:
        target_aspect = 1.5

    is_vertical = ch > cw
    if is_vertical:
        if target_aspect > 1.0:
            target_aspect = 1.0 / target_aspect
    else:
        if target_aspect < 1.0:
            target_aspect = 1.0 / target_aspect

    current_aspect = cw / ch

    if current_aspect > target_aspect:
        target_w = ch * target_aspect
        nx1 = x1 + (cw - target_w) / 2
        nx2 = nx1 + target_w
        x1, x2 = int(nx1), int(nx2)
    else:
        target_h = cw / target_aspect
        ny1 = y1 + (ch - target_h) / 2
        ny2 = ny1 + target_h
        y1, y2 = int(ny1), int(ny2)

    return int(max(0, y1)), int(min(h, y2)), int(max(0, x1)), int(min(w, x2))


@time_function
def get_manual_crop_coords(
    img: ImageBuffer,
    offset_px: int = 0,
    scale_factor: float = 1.0,
) -> ROI:
    """
    Calculates crop coordinates based on image center and a manual offset.
    Used when autocrop is disabled but the user still wants to crop in/out.
    """
    h, w = img.shape[:2]
    roi = (0, h, 0, w)
    margin = offset_px * scale_factor
    return apply_margin_to_roi(roi, h, w, margin)


@time_function
def get_autocrop_coords(
    img: ImageBuffer,
    offset_px: int = 0,
    scale_factor: float = 1.0,
    target_ratio_str: str = "3:2",
    detect_res: int = 1800,
    assist_point: Optional[Tuple[float, float]] = None,
    assist_luma: Optional[float] = None,
) -> ROI:
    """
    Autonomously detects the image boundaries of a scanned negative.

    This function identifies the frame edges (rebates) by analyzing the
    intensity transitions between the latent image and the unexposed film base.
    It then enforces a specific aspect ratio (e.g., 3:2, 6:7) for the final crop.
    """
    h, w = img.shape[:2]
    det_scale = detect_res / max(h, w)

    d_h, d_w = int(h * det_scale), int(w * det_scale)
    img_small = cv2.resize(img, (d_w, d_h), interpolation=cv2.INTER_AREA)

    lum = get_luminance(ensure_image(img_small))

    threshold = 0.96
    if assist_luma is not None:
        threshold = float(np.clip(assist_luma - 0.02, 0.5, 0.98))

    rows_det = np.where(np.mean(lum, axis=1) < threshold)[0]
    cols_det = np.where(np.mean(lum, axis=0) < threshold)[0]

    if len(rows_det) < 10 or len(cols_det) < 10:
        return 0, h, 0, w

    y1, y2 = rows_det[0] / det_scale, rows_det[-1] / det_scale
    x1, x2 = cols_det[0] / det_scale, cols_det[-1] / det_scale

    margin = (2 + offset_px) * scale_factor
    roi = (y1, y2, x1, x2)
    roi = apply_margin_to_roi(roi, h, w, margin)

    return enforce_roi_aspect_ratio(roi, h, w, target_ratio_str)


@time_function
def map_coords_to_geometry(
    nx: float,
    ny: float,
    orig_shape: Tuple[int, int],
    rotation_k: int = 0,
    fine_rotation: float = 0.0,
    roi: Optional[ROI] = None,
) -> Tuple[float, float]:
    """
    Maps normalized coordinates (0-1) from the original image space
    to the current geometric state (rotated and optionally cropped).
    """
    h_orig, w_orig = orig_shape
    px, py = nx * w_orig, ny * h_orig
    h, w = h_orig, w_orig

    k = rotation_k % 4
    if k == 1:  # 90 CCW
        px, py = py, w - px
        h, w = w, h
    elif k == 2:  # 180
        px, py = w - px, h - py
    elif k == 3:  # 270 CCW (90 CW)
        px, py = h - py, px
        h, w = w, h

    if fine_rotation != 0.0:
        center = (w / 2.0, h / 2.0)
        m_mat = cv2.getRotationMatrix2D(center, fine_rotation, 1.0)
        pt = np.array([px, py, 1.0])
        res_pt = m_mat @ pt
        px, py = float(res_pt[0]), float(res_pt[1])

    if roi:
        y1, y2, x1, x2 = roi
        px -= x1
        py -= y1
        h, w = y2 - y1, x2 - x1

    nx_new = np.clip(px / max(w, 1), 0.0, 1.0)
    ny_new = np.clip(py / max(h, 1), 0.0, 1.0)

    return float(nx_new), float(ny_new)
