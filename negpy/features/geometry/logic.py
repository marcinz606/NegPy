import numpy as np
import cv2
from typing import Tuple, Optional
from negpy.domain.types import ImageBuffer, ROI
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.image.logic import get_luminance


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image_float = image.astype(np.float32)
    finite_mask = np.isfinite(image_float)
    if not np.any(finite_mask):
        return np.zeros(image.shape, dtype=np.uint8)

    valid = image_float[finite_mask]
    low = float(np.percentile(valid, 1))
    high = float(np.percentile(valid, 99))
    if high <= low:
        high = low + 1.0

    scaled = np.clip((image_float - low) * (255.0 / (high - low)), 0, 255)
    return scaled.astype(np.uint8)


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or signal.size == 0:
        return signal.astype(np.float32)

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def _boundary_candidates(signal: np.ndarray, *, from_start: bool) -> tuple[int, float, int, float]:
    length = signal.size
    if length == 0:
        return 0, 0.0, 0, 0.0

    edge_window = max(int(round(length * 0.08)), 32)
    edge_window = min(edge_window, max(length - 1, 1))
    search_end = max(int(round(length * 0.45)), edge_window + 1)
    search_end = min(search_end, length)
    search_start = min(int(round(length * 0.55)), max(length - edge_window - 1, 0))

    if from_start:
        edge_slice = signal[:edge_window]
        search_slice = signal[edge_window:search_end]
        edge_idx = int(np.argmax(edge_slice)) if edge_slice.size else 0
        edge_value = float(edge_slice[edge_idx]) if edge_slice.size else 0.0
        if search_slice.size == 0:
            return edge_idx, edge_value, edge_idx, edge_value
        inner_offset = int(np.argmax(search_slice))
        inner_idx = edge_window + inner_offset
        inner_value = float(search_slice[inner_offset])
        return edge_idx, edge_value, inner_idx, inner_value

    edge_slice = signal[length - edge_window :]
    search_slice = signal[search_start : length - edge_window]
    edge_offset = int(np.argmax(edge_slice)) if edge_slice.size else 0
    edge_idx = length - edge_window + edge_offset
    edge_value = float(edge_slice[edge_offset]) if edge_slice.size else 0.0
    if search_slice.size == 0:
        return edge_idx, edge_value, edge_idx, edge_value
    inner_offset = int(np.argmax(search_slice))
    inner_idx = search_start + inner_offset
    inner_value = float(search_slice[inner_offset])
    return edge_idx, edge_value, inner_idx, inner_value


def _dark_region_bounds(image: np.ndarray) -> tuple[int, int, int, int] | None:
    preview = _normalize_to_uint8(_ensure_color(image))
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    threshold = float(np.percentile(gray, 55))
    mask = (gray <= threshold).astype(np.uint8) * 255
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(contour)
    image_area = float(gray.shape[0] * gray.shape[1])
    area_ratio = float(cv2.contourArea(contour)) / max(image_area, 1.0)
    if area_ratio < 0.15 or area_ratio > 0.85:
        return None

    min_width = int(round(image.shape[1] * 0.25))
    min_height = int(round(image.shape[0] * 0.25))
    if box_w < min_width or box_h < min_height:
        return None

    pad_x = max(int(round(image.shape[1] * 0.004)), 4)
    pad_y = max(int(round(image.shape[0] * 0.004)), 4)
    left = max(x - pad_x, 0)
    top = max(y - pad_y, 0)
    right = min(x + box_w + pad_x, image.shape[1])
    bottom = min(y + box_h + pad_y, image.shape[0])

    min_inset_x = int(round(image.shape[1] * 0.03))
    min_inset_y = int(round(image.shape[0] * 0.03))
    if left < min_inset_x or top < min_inset_y or (image.shape[1] - right) < min_inset_x or (image.shape[0] - bottom) < min_inset_y:
        return None

    return left, top, right, bottom


def _refine_frame_bounds(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    preview = _normalize_to_uint8(_ensure_color(image))
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    col_signal = _smooth_signal(np.percentile(grad_x, 95, axis=0), 31)
    row_signal = _smooth_signal(np.percentile(grad_y, 95, axis=1), 31)

    left_edge, left_edge_value, left_inner, left_inner_value = _boundary_candidates(col_signal, from_start=True)
    right_edge, right_edge_value, right_inner, right_inner_value = _boundary_candidates(col_signal, from_start=False)
    top_edge, top_edge_value, top_inner, top_inner_value = _boundary_candidates(row_signal, from_start=True)
    bottom_edge, bottom_edge_value, bottom_inner, bottom_inner_value = _boundary_candidates(row_signal, from_start=False)

    col_noise_floor = float(np.percentile(col_signal, 75))
    row_noise_floor = float(np.percentile(row_signal, 75))

    left = left_edge
    right = right_edge + 1
    top = top_edge
    bottom = bottom_edge + 1

    use_inner_pair_x = (
        left_inner >= int(round(image.shape[1] * 0.12))
        and right_inner <= int(round(image.shape[1] * 0.88))
        and left_inner_value >= col_noise_floor * 4.0
        and right_inner_value >= col_noise_floor * 4.0
        and (right_inner - left_inner) >= int(round(image.shape[1] * 0.5))
    )
    if use_inner_pair_x:
        left = left_inner
        right = right_inner + 1
    else:
        if left_inner >= int(round(image.shape[1] * 0.12)) and left_inner_value >= col_noise_floor * 5.0:
            left = left_inner
        if right_inner <= int(round(image.shape[1] * 0.88)) and right_inner_value >= col_noise_floor * 5.0:
            right = right_inner + 1

    use_inner_pair_y = (
        top_inner > top_edge + 20
        and bottom_inner < bottom_edge - 20
        and top_inner_value > max(top_edge_value * 1.2, row_noise_floor + 25.0)
        and bottom_inner_value > max(bottom_edge_value * 1.2, row_noise_floor + 25.0)
        and (bottom_inner - top_inner) >= int(round(image.shape[0] * 0.5))
    )
    if use_inner_pair_y:
        top = top_inner
        bottom = bottom_inner + 1
    else:
        if top_inner > top_edge + 20 and top_inner_value > max(top_edge_value * 1.45, row_noise_floor + 35.0):
            top = top_inner
        if bottom_inner < bottom_edge - 20 and bottom_inner_value > max(bottom_edge_value * 1.45, row_noise_floor + 35.0):
            bottom = bottom_inner + 1

    pad_x = max(int(round(image.shape[1] * 0.004)), 4)
    pad_y = max(int(round(image.shape[0] * 0.004)), 4)
    left = max(left - pad_x, 0)
    right = min(right + pad_x, image.shape[1])
    top = max(top - pad_y, 0)
    bottom = min(bottom + pad_y, image.shape[0])

    min_width = max(int(round(image.shape[1] * 0.5)), 1)
    min_height = max(int(round(image.shape[0] * 0.5)), 1)
    if right - left < min_width:
        left, right = 0, image.shape[1]
    if bottom - top < min_height:
        top, bottom = 0, image.shape[0]

    refined_area_ratio = ((right - left) * (bottom - top)) / max(float(image.shape[0] * image.shape[1]), 1.0)
    dark_bounds = _dark_region_bounds(image)
    if dark_bounds is not None and refined_area_ratio > 0.8:
        dark_left, dark_top, dark_right, dark_bottom = dark_bounds
        dark_area_ratio = ((dark_right - dark_left) * (dark_bottom - dark_top)) / max(float(image.shape[0] * image.shape[1]), 1.0)
        if 0.15 <= dark_area_ratio <= 0.85:
            left, top, right, bottom = dark_left, dark_top, dark_right, dark_bottom

    return image[top:bottom, left:right], (left, top, right, bottom)


def _mask_from_blackhat(gray: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)


def _mask_from_inverse_threshold(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)


def _mask_from_edges(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 160)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, dilate_kernel, iterations=2)
    return cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel, iterations=2)


def _score_contour(contour: np.ndarray, image_area: float) -> tuple[float, np.ndarray] | None:
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    rect_area = float(width * height)
    if rect_area <= 0:
        return None

    contour_area = float(cv2.contourArea(contour))
    area_ratio = rect_area / image_area
    fill_ratio = contour_area / rect_area if rect_area else 0.0
    short_side = min(width, height)
    long_side = max(width, height)
    aspect_ratio = long_side / max(short_side, 1.0)

    if area_ratio < 0.08:
        return None
    if short_side < 40:
        return None
    if aspect_ratio > 8.0:
        return None

    score = area_ratio * 1.5 + min(fill_ratio, 1.0)
    return score, cv2.boxPoints(rect)


def _find_autocrop_roi_from_contours(img: ImageBuffer) -> ROI | None:
    color = _ensure_color(img)
    preview = _normalize_to_uint8(color)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    image_area = float(gray.shape[0] * gray.shape[1])

    best_score = -1.0
    best_quad: np.ndarray | None = None

    for mask in (_mask_from_blackhat(gray), _mask_from_inverse_threshold(gray), _mask_from_edges(gray)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            scored = _score_contour(contour, image_area)
            if scored is None:
                continue
            score, quad = scored
            if score > best_score:
                best_score = score
                best_quad = quad

    if best_quad is None:
        return None

    x, y, box_w, box_h = cv2.boundingRect(best_quad.astype(np.float32))
    left = max(int(x), 0)
    top = max(int(y), 0)
    right = min(int(x + box_w), img.shape[1])
    bottom = min(int(y + box_h), img.shape[0])
    if right - left <= 0 or bottom - top <= 0:
        return None

    _, (ref_left, ref_top, ref_right, ref_bottom) = _refine_frame_bounds(img[top:bottom, left:right])
    return top + ref_top, top + ref_bottom, left + ref_left, left + ref_right


def _get_threshold_autocrop_coords(
    img: ImageBuffer,
    target_ratio_str: str,
    detect_res: int,
    assist_luma: Optional[float],
) -> ROI:
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
    return int(y1), int(y2), int(x1), int(x2)


def apply_fine_rotation(img: ImageBuffer, angle: float) -> ImageBuffer:
    """
    Sub-degree rotation (bilinear).
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
        borderMode=cv2.BORDER_REPLICATE,
    )
    return ensure_image(res)


def apply_margin_to_roi(
    roi: ROI,
    h: int,
    w: int,
    margin_px: float,
) -> ROI:
    """
    Expands/Contracts ROI.
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
    Centers ROI within aspect ratio.
    """
    y1, y2, x1, x2 = roi
    cw, ch = x2 - x1, y2 - y1

    if cw <= 0 or ch <= 0:
        return 0, h, 0, w

    if target_ratio_str == "Free":
        return int(max(0, y1)), int(min(h, y2)), int(max(0, x1)), int(min(w, x2))

    try:
        w_r, h_r = map(float, target_ratio_str.split(":"))
        target_aspect = w_r / h_r
    except ValueError:
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


def get_manual_rect_coords(
    img_or_shape: ImageBuffer | Tuple[int, int],
    manual_rect: Tuple[float, float, float, float],
    orig_shape: Tuple[int, int],
    rotation_k: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    offset_px: int = 0,
    scale_factor: float = 1.0,
) -> ROI:
    """
    Maps normalized manual crop rect (RAW coords) to pixel ROI in TRANSFORMED image space.
    """
    if isinstance(img_or_shape, tuple):
        h_curr, w_curr = img_or_shape
    else:
        h_curr, w_curr = img_or_shape.shape[:2]

    x1_n, y1_n, x2_n, y2_n = manual_rect

    corners = [(x1_n, y1_n), (x2_n, y1_n), (x2_n, y2_n), (x1_n, y2_n)]
    mapped_corners = []

    for nx, ny in corners:
        mx, my = map_coords_to_geometry(
            nx,
            ny,
            orig_shape,
            rotation_k,
            fine_rotation,
            flip_horizontal,
            flip_vertical,
            roi=None,
        )
        mapped_corners.append((mx, my))

    xs = [p[0] * w_curr for p in mapped_corners]
    ys = [p[1] * h_curr for p in mapped_corners]

    ix1, ix2 = int(min(xs)), int(max(xs))
    iy1, iy2 = int(min(ys)), int(max(ys))

    roi = (iy1, iy2, ix1, ix2)
    margin = offset_px * scale_factor
    return apply_margin_to_roi(roi, h_curr, w_curr, margin)


def get_manual_crop_coords(
    img: ImageBuffer,
    offset_px: int = 0,
    scale_factor: float = 1.0,
) -> ROI:
    """
    Center crop + offset.
    """
    h, w = img.shape[:2]
    roi = (0, h, 0, w)
    margin = offset_px * scale_factor
    return apply_margin_to_roi(roi, h, w, margin)


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
    Detects film border via density thresholding.
    """
    h, w = img.shape[:2]
    roi = _find_autocrop_roi_from_contours(img)
    if roi is None:
        roi = _get_threshold_autocrop_coords(img, target_ratio_str, detect_res, assist_luma)

    margin = (2 + offset_px) * scale_factor
    roi = apply_margin_to_roi(roi, h, w, margin)

    return enforce_roi_aspect_ratio(roi, h, w, target_ratio_str)


def map_coords_to_geometry(
    nx: float,
    ny: float,
    orig_shape: Tuple[int, int],
    rotation_k: int = 0,
    fine_rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    roi: Optional[ROI] = None,
) -> Tuple[float, float]:
    """
    Maps raw coordinates to geometry-transformed space.
    """
    h_orig, w_orig = orig_shape
    px, py = nx * w_orig, ny * h_orig
    h, w = h_orig, w_orig

    k = rotation_k % 4
    if k == 1:
        px, py = py, w - px
        h, w = w, h
    elif k == 2:
        px, py = w - px, h - py
    elif k == 3:
        px, py = h - py, px
        h, w = w, h

    if flip_horizontal:
        px = w - px
    if flip_vertical:
        py = h - py

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


def translate_manual_crop_rect(
    rect: Tuple[float, float, float, float],
    dx: float,
    dy: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    max_x1 = max(0.0, 1.0 - w)
    max_y1 = max(0.0, 1.0 - h)
    nx1 = min(max(x1 + dx, 0.0), max_x1)
    ny1 = min(max(y1 + dy, 0.0), max_y1)
    return (nx1, ny1, nx1 + w, ny1 + h)
