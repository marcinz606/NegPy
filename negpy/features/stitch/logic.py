"""Registration and blending for multi-part scan stitching.

All functions operate on scene-linear float32 buffers that are already
EXIF-oriented and flat-fielded — transforms estimated here are only valid on
buffers decoded the same way.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from negpy.features.stitch.models import StitchConfig

PROXY_MAX_EDGE = 1600
MIN_INLIERS = 100

_NO_OVERLAP_MSG = "Could not find overlap between the selected frames"


class StitchError(RuntimeError):
    """Registration failure; the message is shown to the user verbatim."""


class StitchCancelled(StitchError):
    pass


def build_proxy(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Registration proxy: green channel, <=1600px, contrast-stretched uint8.

    Returns (proxy, scale) with proxy_coords = full_coords * scale.
    """
    gray = rgb[..., 1] if rgb.ndim == 3 else rgb
    scale = min(1.0, PROXY_MAX_EDGE / max(gray.shape))
    if scale < 1.0:
        size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(gray, (1.0, 99.0))
    gray = np.clip((gray - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return (gray ** (1.0 / 2.2) * 255.0).astype(np.uint8), scale


def register_pair(ref_proxy: np.ndarray, mov_proxy: np.ndarray, min_inliers: int = MIN_INLIERS) -> Tuple[np.ndarray, int]:
    """Similarity transform (2x3) mapping ``mov_proxy`` coords to ``ref_proxy`` coords.

    SIFT + ratio test + RANSAC; ORB and phase correlation both fail on film
    negatives (feature-poor orange-mask content, sub-degree rotations).
    """
    sift = cv2.SIFT_create(nfeatures=6000)
    ref_kp, ref_desc = sift.detectAndCompute(ref_proxy, None)
    mov_kp, mov_desc = sift.detectAndCompute(mov_proxy, None)
    if ref_desc is None or mov_desc is None or len(ref_kp) < 2 or len(mov_kp) < 2:
        raise StitchError(_NO_OVERLAP_MSG)

    pairs = cv2.BFMatcher().knnMatch(mov_desc, ref_desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        raise StitchError(_NO_OVERLAP_MSG)

    mov_pts = np.float32([mov_kp[m.queryIdx].pt for m in good])
    ref_pts = np.float32([ref_kp[m.trainIdx].pt for m in good])
    matrix, inlier_mask = cv2.estimateAffinePartial2D(mov_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if matrix is None or inliers < min_inliers:
        raise StitchError(_NO_OVERLAP_MSG)
    return matrix.astype(np.float64), inliers


def scale_affine(matrix: np.ndarray, ref_scale: float, mov_scale: float) -> np.ndarray:
    """Proxy-coordinate affine -> full-resolution affine."""
    out = matrix.astype(np.float64).copy()
    out[:, :2] *= mov_scale / ref_scale
    out[:, 2] /= ref_scale
    return out


def chain_affine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose 2x3 affines: (a ∘ b)(x) = a(b(x))."""
    a3 = np.vstack([a, (0.0, 0.0, 1.0)])
    b3 = np.vstack([b, (0.0, 0.0, 1.0)])
    return (a3 @ b3)[:2]


def compute_canvas(sizes: Sequence[Tuple[int, int]], transforms: Sequence[np.ndarray]) -> Tuple[Tuple[int, int], List[np.ndarray]]:
    """Union bounding box of the warped parts. Returns ((W, H), transforms shifted to it)."""
    corners = []
    for (w, h), matrix in zip(sizes, transforms):
        box = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float64)
        corners.append(cv2.transform(box, matrix)[0])
    points = np.vstack(corners)
    lo = np.floor(points.min(axis=0))
    hi = np.ceil(points.max(axis=0))
    shift = np.array([[1.0, 0.0, -lo[0]], [0.0, 1.0, -lo[1]]], np.float64)
    shifted = [chain_affine(shift, m) for m in transforms]
    return (int(hi[0] - lo[0]), int(hi[1] - lo[1])), shifted


def register_parts(
    parts: Sequence[np.ndarray],
    is_cancelled: Callable[[], bool] = lambda: False,
    on_progress: Callable[[int, int, str], None] = lambda *a: None,
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Chained pairwise registration (each part vs the previous — adjacent shots overlap).

    Returns full-res part->canvas transforms (incl. the identity-based primary)
    and the canvas size. Raises StitchError when a pair has no usable overlap.
    """
    proxies = [build_proxy(p) for p in parts]
    transforms = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64)]
    for i in range(1, len(parts)):
        if is_cancelled():
            raise StitchCancelled("Cancelled")
        on_progress(i, len(parts) - 1, f"Registering frame {i + 1}")
        matrix, _ = register_pair(proxies[i - 1][0], proxies[i][0])
        full = scale_affine(matrix, proxies[i - 1][1], proxies[i][1])
        transforms.append(chain_affine(transforms[i - 1], full))
    sizes = [(p.shape[1], p.shape[0]) for p in parts]
    canvas, shifted = compute_canvas(sizes, transforms)
    return shifted, canvas


def scale_transforms(config: StitchConfig, decoded_sizes: Sequence[Tuple[int, int]]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Stored full-res transforms -> the scale the parts were actually decoded at.

    Per-part scale comes from ``stitch_sizes`` (preview decodes can round each
    part differently); canvas coordinates scale with the primary part.
    """
    ref_scale = decoded_sizes[0][0] / config.stitch_sizes[0][0]
    out = []
    for flat, (dec_w, _), (full_w, _) in zip(config.stitch_transforms, decoded_sizes, config.stitch_sizes):
        matrix = np.array(flat, np.float64).reshape(2, 3)
        part_scale = dec_w / full_w
        matrix[:, :2] *= ref_scale / part_scale
        matrix[:, 2] *= ref_scale
        out.append(matrix)
    canvas = (round(config.stitch_canvas[0] * ref_scale), round(config.stitch_canvas[1] * ref_scale))
    return out, canvas


def warp_into_canvas(
    img: np.ndarray,
    transform: np.ndarray,
    canvas_wh: Tuple[int, int],
    interpolation: int = cv2.INTER_CUBIC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a part and its validity mask into the canvas. The mask is eroded by the
    >0.999 threshold so resampled border pixels never bleed into the blend."""
    warped = cv2.warpAffine(img, transform.astype(np.float32), canvas_wh, flags=interpolation)
    ones = np.ones(img.shape[:2], np.float32)
    mask = cv2.warpAffine(ones, transform.astype(np.float32), canvas_wh, flags=cv2.INTER_LINEAR)
    return warped, mask > 0.999


_MIN_OVERLAP_PX = 1000


def gain_compensate(ref: np.ndarray, ref_mask: np.ndarray, mov: np.ndarray, mov_mask: np.ndarray) -> np.ndarray:
    """Scale ``mov`` (in place) so its overlap with ``ref`` matches per channel —
    removes light-source drift between shots. No-op on tiny overlap."""
    overlap = ref_mask & mov_mask
    if int(overlap.sum()) < _MIN_OVERLAP_PX:
        return mov
    ref_mean = ref[overlap].mean(axis=0)
    mov_mean = mov[overlap].mean(axis=0)
    if np.any(mov_mean < 1e-6):
        return mov
    mov *= (ref_mean / mov_mean).astype(np.float32)
    return mov


def feather_blend(warped: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> np.ndarray:
    """Distance-transform-weighted average: seams fade over the full overlap width."""
    acc = np.zeros_like(warped[0])
    weight_sum = np.zeros(warped[0].shape[:2], np.float32)
    for img, mask in zip(warped, masks):
        weight = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        acc += img * weight[..., None]
        weight_sum += weight
    return acc / np.maximum(weight_sum, 1e-6)[..., None]


def blend_ir(irs: Sequence[np.ndarray], transforms: Sequence[np.ndarray], canvas_wh: Tuple[int, int]) -> np.ndarray:
    """Per-pixel max over valid contributions: film dust appears in every part and
    survives; single-part (sensor-side) specks are erased. Uncovered pixels are 1.0
    (loader convention: clean)."""
    w, h = canvas_wh
    acc = np.full((h, w), -1.0, np.float32)
    for ir, transform in zip(irs, transforms):
        warped, mask = warp_into_canvas(ir, transform, canvas_wh, interpolation=cv2.INTER_LINEAR)
        acc[mask] = np.maximum(acc[mask], warped[mask])
    acc[acc < 0.0] = 1.0
    return acc


def stitch_composite(
    parts: List[np.ndarray],
    irs: Sequence[Optional[np.ndarray]],
    config: StitchConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Warp + gain-compensate + feather-blend the decoded parts into one frame.

    ``parts`` is consumed (slots dropped after warping) to cap peak memory.
    IR is carried only when every part has one.
    """
    transforms, canvas = scale_transforms(config, [(p.shape[1], p.shape[0]) for p in parts])
    warped: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for i in range(len(parts)):
        img, mask = warp_into_canvas(parts[i], transforms[i], canvas)
        parts[i] = None  # type: ignore[call-overload]
        for j in range(len(warped)):
            if (masks[j] & mask).sum() >= _MIN_OVERLAP_PX:
                gain_compensate(warped[j], masks[j], img, mask)
                break
        warped.append(img)
        masks.append(mask)
    rgb = np.clip(feather_blend(warped, masks), 0.0, 1.0)
    ir = None
    if irs and all(x is not None for x in irs):
        ir = blend_ir(irs, transforms, canvas)  # type: ignore[arg-type]
    return rgb, ir
