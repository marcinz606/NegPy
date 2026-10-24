"""Registration and blending for multi-part scan stitching.

All functions operate on scene-linear float32 buffers that are already
EXIF-oriented and flat-fielded — transforms estimated here are only valid on
buffers decoded the same way.
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple

import negpy_core
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
        gray = negpy_core.resize_area(np.ascontiguousarray(gray, dtype=np.float32), size)
    return negpy_core.proxy_stretch(np.ascontiguousarray(gray, dtype=np.float32)), scale


def register_pair(ref_proxy: np.ndarray, mov_proxy: np.ndarray, min_inliers: int = MIN_INLIERS) -> Tuple[np.ndarray, int]:
    """Similarity transform (2x3) mapping ``mov_proxy`` coords to ``ref_proxy`` coords.

    SIFT + ratio test + RANSAC; ORB and phase correlation both fail on film
    negatives (feature-poor orange-mask content, sub-degree rotations).
    """
    found = negpy_core.register_pair(
        np.ascontiguousarray(ref_proxy, dtype=np.uint8), np.ascontiguousarray(mov_proxy, dtype=np.uint8), min_inliers
    )
    if found is None:
        raise StitchError(_NO_OVERLAP_MSG)
    matrix, inliers = found
    return np.array(matrix, np.float64).reshape(2, 3), inliers


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
        # cv2.transform's order: its dispatched build fuses the first product.
        m = matrix.tolist()
        corners.append([[math.fma(r[0], x, r[1] * y) + r[2] for r in m] for x, y in ((0, 0), (w, 0), (w, h), (0, h))])
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
    interpolation: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a part and its validity mask into the canvas. The mask is eroded by the
    >0.999 threshold so resampled border pixels never bleed into the blend."""
    m = transform.astype(np.float32).astype(np.float64)
    warped = negpy_core.warp_affine(
        np.ascontiguousarray(img, dtype=np.float32), m, canvas_wh, interpolation=interpolation, border="constant"
    )
    ones = np.ones(img.shape[:2], np.float32)
    mask = negpy_core.warp_affine(ones, m, canvas_wh, border="constant")
    return warped, mask > 0.999


_MIN_OVERLAP_PX = 1000


def gain_compensate(ref: np.ndarray, ref_mask: np.ndarray, mov: np.ndarray, mov_mask: np.ndarray) -> np.ndarray:
    """Scale ``mov`` (in place) so its overlap with ``ref`` matches per channel —
    removes light-source drift between shots. No-op on tiny overlap."""
    # The means are float64: a full-res overlap is tens of millions of rows, and a float32 sum
    # drifts per channel, so export gains would differ from the preview's.
    gains = negpy_core.compensation_gains(
        np.ascontiguousarray(ref, dtype=np.float32),
        np.ascontiguousarray(ref_mask),
        np.ascontiguousarray(mov, dtype=np.float32),
        np.ascontiguousarray(mov_mask),
        _MIN_OVERLAP_PX,
    )
    if gains is not None:
        negpy_core.scale_channels(mov, gains)
    return mov


def feather_blend(warped: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> np.ndarray:
    """Distance-transform-weighted average over the full overlap width.

    Averaging draws misregistered detail twice, so this is the fallback for when a
    seam cannot be cut, not the normal path. See ``seam_blend``.
    """
    return negpy_core.feather_blend(_contiguous(warped, np.float32), _contiguous(masks, bool))


def _contiguous(arrays: Sequence[np.ndarray], dtype) -> List[np.ndarray]:
    return [np.ascontiguousarray(a, dtype=dtype) for a in arrays]


_SEAM_MAX_EDGE = 1024
# Blend band around the cut, in full-res canvas px, scaled to whatever resolution the
# parts decoded at. Wide enough to hide the residual step gain compensation leaves,
# narrow enough that no edge the registration misplaced is drawn twice across it.
_SEAM_FEATHER_PX = 48.0


def _seam_cost_image(img: np.ndarray) -> np.ndarray:
    """Perceptual copy for the cut's color cost. Flat-field gain can push linear
    values past white, which would otherwise decide the seam on its own."""
    return (np.clip(img, 0.0, 1.0) ** (1.0 / 2.2)).astype(np.float32)


def find_seams(warped: Sequence[np.ndarray], masks: Sequence[np.ndarray], canvas_wh: Tuple[int, int]) -> Optional[List[np.ndarray]]:
    """Partition the covered canvas between the parts along a minimum-error path.

    The cut is searched on a downscaled copy: where a seam runs is a low-frequency
    decision and does not repay canvas resolution. Returns None when no cut is
    possible, leaving the caller on ``feather_blend``.

    Dynamic programming rather than a graph cut: the graph cut costs superlinearly in
    the overlap, for no better seam.
    """
    if len(warped) < 2:
        return None
    w, h = canvas_wh
    scale = min(1.0, _SEAM_MAX_EDGE / max(w, h))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # Everything through the upscale is guarded: the masks come back at canvas size, so
    # this is where an export tight on memory fails, and it has to degrade to the
    # average rather than end the export.
    try:
        small = [_seam_cost_image(negpy_core.resize_area(np.ascontiguousarray(img, dtype=np.float32), size)) for img in warped]
        small_masks = [negpy_core.resize_nearest(m.astype(np.uint8), size) for m in masks]
        cut = negpy_core.dp_seams(small, small_masks)
        seams = [(negpy_core.resize_nearest(m, (w, h)) > 0) & valid for m, valid in zip(cut, masks)]
        # Nearest-neighbour upscaling can leave a covered pixel assigned to no part, and
        # an unassigned pixel has no weight to normalize against. Give each one back to a
        # part that actually covers it.
        orphan = np.zeros((h, w), bool)
        for mask in masks:
            orphan |= mask
        for seam in seams:
            orphan &= ~seam
        if orphan.any():
            for seam, mask in zip(seams, masks):
                claim = orphan & mask
                seam |= claim
                orphan &= ~claim
    except (ValueError, MemoryError):
        return None
    return seams


def seam_blend(
    warped: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    seams: Sequence[np.ndarray],
    feather_px: float,
) -> np.ndarray:
    """Cross-fade within ``feather_px`` of the cut; beyond it one part wins outright.

    Weight ramps on the *signed* distance to the cut, so it reaches across into the
    neighbour's side and the two parts actually mix there. A weight that only ramped
    inside a part's own side would leave the halves disjoint and the cut hard.

    Each part's weight is damped to zero at its own coverage edge, so an overlap narrower
    than the band does not end on a step in the weight sum.
    """
    return negpy_core.seam_blend(_contiguous(warped, np.float32), _contiguous(masks, bool), _contiguous(seams, bool), max(feather_px, 1.0))


def blend_ir(
    irs: Sequence[np.ndarray],
    transforms: Sequence[np.ndarray],
    canvas_wh: Tuple[int, int],
    seams: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Defect mask for the assembled canvas. Uncovered pixels are 1.0 (loader
    convention: clean).

    With a seam, each pixel's mask comes from the part the pixel itself came from, so
    the mask describes what is actually on the canvas. Without one the blend is an
    average of every part, and the mask is a per-pixel max: film dust lands at the same
    canvas place in every part and survives, while a speck on the camera side lands in
    only one and is diluted rather than flagged.
    """
    w, h = canvas_wh
    acc = np.full((h, w), -1.0, np.float32)
    for i, (ir, transform) in enumerate(zip(irs, transforms)):
        warped, mask = warp_into_canvas(ir, transform, canvas_wh, interpolation="linear")
        sel = mask if seams is None else seams[i]
        negpy_core.ir_blend_into(acc, np.ascontiguousarray(warped, dtype=np.float32), np.ascontiguousarray(sel, dtype=bool), seams is None)
    negpy_core.ir_finish(acc)
    return acc


def stitch_composite(
    parts: List[np.ndarray],
    irs: Sequence[Optional[np.ndarray]],
    config: StitchConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Warp, gain-compensate, then seam-cut and blend the decoded parts into one frame.

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
            if negpy_core.mask_overlap(masks[j], mask) >= _MIN_OVERLAP_PX:
                gain_compensate(warped[j], masks[j], img, mask)
                break
        warped.append(img)
        masks.append(mask)
    seams = find_seams(warped, masks, canvas)
    if seams is None:
        blended = feather_blend(warped, masks)
    else:
        decoded_scale = canvas[0] / max(config.stitch_canvas[0], 1)
        blended = seam_blend(warped, masks, seams, _SEAM_FEATHER_PX * decoded_scale)
    rgb = np.ascontiguousarray(blended, dtype=np.float32)
    negpy_core.clip_unit(rgb)
    ir = None
    if irs and all(x is not None for x in irs):
        ir = blend_ir(irs, transforms, canvas, seams)  # type: ignore[arg-type]
    return rgb, ir
