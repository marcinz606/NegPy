"""IR-specific dust repair with scanner-rail exclusion.

The LS-5000 full aperture includes opaque bands outside the supported film
area.  Those bands are low in the IR plane just like dust, so applying a plain
``ir < threshold`` mask to the whole canvas smears the rails into the image.

This module keeps the proven pre-v0.37 pixel-mask healer for IR defects and
adds a conservative, edge-seeded support detector.  The newer membrane healer
remains the implementation for manual and luminance-detected retouch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional, Tuple

from numba import njit
import numpy as np

from negpy.domain.types import ImageBuffer, ROI
from negpy.kernel.image.validation import ensure_image


class IRSupportError(RuntimeError):
    """The IR plane cannot provide a safe exposed-film support region."""


@dataclass(frozen=True)
class IRRepairSupport:
    """Rectangular film support and the evidence used to derive it."""

    y0: int
    y1: int
    x0: int
    x1: int
    height: int
    width: int
    detected_edges: tuple[str, ...]
    threshold: float
    edge_seed_fraction: float
    interior_fraction_limit: float
    stable_run: int
    masked_pixels_total: int
    masked_pixels_inside: int

    @property
    def bounds(self) -> ROI:
        return self.y0, self.y1, self.x0, self.x1

    @property
    def masked_pixels_outside(self) -> int:
        return self.masked_pixels_total - self.masked_pixels_inside

    @property
    def inside_pixels(self) -> int:
        return (self.y1 - self.y0) * (self.x1 - self.x0)

    @property
    def total_pixels(self) -> int:
        return self.height * self.width

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "bounds": list(self.bounds),
                "inside_pixels": self.inside_pixels,
                "total_pixels": self.total_pixels,
                "masked_fraction_full": self.masked_pixels_total / self.total_pixels,
                "masked_fraction_inside": self.masked_pixels_inside / self.inside_pixels,
                "outside_share_of_mask": (
                    self.masked_pixels_outside / self.masked_pixels_total if self.masked_pixels_total else 0.0
                ),
            }
        )
        return result


def normalize_ir(ir: np.ndarray) -> np.ndarray:
    """Return one finite float32 IR plane in the scanner's [0, 1] range."""

    array = np.asarray(ir)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2 or min(array.shape) < 1:
        raise IRSupportError(f"IR must be a non-empty 2-D plane, got {array.shape}")
    if array.dtype == np.uint16:
        result = array.astype(np.float32) * np.float32(1.0 / 65535.0)
    elif array.dtype == np.uint8:
        result = array.astype(np.float32) * np.float32(1.0 / 255.0)
    elif np.issubdtype(array.dtype, np.floating):
        result = array.astype(np.float32, copy=False)
    else:
        raise IRSupportError(f"IR dtype must be uint8, uint16, or float, got {array.dtype}")
    if not np.all(np.isfinite(result)):
        raise IRSupportError("IR contains non-finite values")
    low = float(result.min())
    high = float(result.max())
    if low < 0.0 or high > 1.0:
        raise IRSupportError(f"float IR must remain in [0, 1], got [{low}, {high}]")
    return np.ascontiguousarray(result)


def _edge_boundary(
    profile: np.ndarray,
    *,
    edge_seed_fraction: float,
    interior_fraction_limit: float,
    stable_run: int,
    edge_name: str,
) -> tuple[int, bool]:
    """Return the first stable interior line measured inward from one edge."""

    seed_lines = min(4, profile.size)
    if float(profile[:seed_lines].mean()) < edge_seed_fraction:
        return 0, False

    last_start = profile.size - stable_run
    for start in range(1, last_start + 1):
        if bool(np.all(profile[start : start + stable_run] < interior_fraction_limit)):
            return start, True
    raise IRSupportError(
        f"{edge_name} edge is opaque but never reaches {stable_run} stable interior lines; "
        "refusing IR repair (wrong threshold or IR-opaque film)"
    )


def detect_ir_repair_support(
    ir: np.ndarray,
    *,
    threshold: float = 0.5,
    edge_seed_fraction: float = 0.5,
    interior_fraction_limit: float = 0.01,
    stable_run: int = 8,
    minimum_support_fraction: float = 0.25,
) -> IRRepairSupport:
    """Detect a conservative rectangular support for IR repair.

    An edge is removed only when at least half of its first four lines are
    below the dust threshold.  The boundary is the first run of stable lines
    with fewer than one percent threshold hits.  Ordinary edge dust therefore
    remains repairable, while full-height/full-width scanner rails are gated.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 < edge_seed_fraction <= 1.0:
        raise ValueError("edge_seed_fraction must be in (0, 1]")
    if not 0.0 < interior_fraction_limit < edge_seed_fraction:
        raise ValueError("interior_fraction_limit must be positive and below edge_seed_fraction")
    if stable_run < 2:
        raise ValueError("stable_run must be at least 2")
    if not 0.0 < minimum_support_fraction <= 1.0:
        raise ValueError("minimum_support_fraction must be in (0, 1]")

    plane = normalize_ir(ir)
    height, width = plane.shape
    if min(height, width) < stable_run + 2:
        raise IRSupportError(f"IR plane {plane.shape} is too small for a {stable_run}-line stability test")

    defects = plane < np.float32(threshold)
    column_fraction = defects.mean(axis=0)
    row_fraction = defects.mean(axis=1)

    x0, left = _edge_boundary(
        column_fraction,
        edge_seed_fraction=edge_seed_fraction,
        interior_fraction_limit=interior_fraction_limit,
        stable_run=stable_run,
        edge_name="left",
    )
    right_depth, right = _edge_boundary(
        column_fraction[::-1],
        edge_seed_fraction=edge_seed_fraction,
        interior_fraction_limit=interior_fraction_limit,
        stable_run=stable_run,
        edge_name="right",
    )
    y0, top = _edge_boundary(
        row_fraction,
        edge_seed_fraction=edge_seed_fraction,
        interior_fraction_limit=interior_fraction_limit,
        stable_run=stable_run,
        edge_name="top",
    )
    bottom_depth, bottom = _edge_boundary(
        row_fraction[::-1],
        edge_seed_fraction=edge_seed_fraction,
        interior_fraction_limit=interior_fraction_limit,
        stable_run=stable_run,
        edge_name="bottom",
    )
    x1 = width - right_depth
    y1 = height - bottom_depth

    if x1 <= x0 or y1 <= y0:
        raise IRSupportError(f"opposing opaque edges overlap: {(y0, y1, x0, x1)}")
    support_fraction = ((x1 - x0) * (y1 - y0)) / float(width * height)
    if support_fraction < minimum_support_fraction:
        raise IRSupportError(f"detected support is only {support_fraction:.3%} of the frame; refusing repair")

    edges = tuple(
        name
        for name, present in (
            ("left", left),
            ("right", right),
            ("top", top),
            ("bottom", bottom),
        )
        if present
    )
    masked_total = int(defects.sum())
    masked_inside = int(defects[y0:y1, x0:x1].sum())
    return IRRepairSupport(
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        height=height,
        width=width,
        detected_edges=edges,
        threshold=threshold,
        edge_seed_fraction=edge_seed_fraction,
        interior_fraction_limit=interior_fraction_limit,
        stable_run=stable_run,
        masked_pixels_total=masked_total,
        masked_pixels_inside=masked_inside,
    )


@njit(cache=True, fastmath=True)
def _hash2(x: float, y: float) -> float:
    """Port of the retouch shader hash used for degenerate directions."""

    px = (x * 0.1031) % 1.0
    py = (y * 0.1031) % 1.0
    pz = (x * 0.1031) % 1.0
    d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33)
    px += d
    py += d
    pz += d
    return ((px + py) * pz) % 1.0


@njit(cache=True, fastmath=True)
def _heal_with_mask_jit(
    img: np.ndarray,
    hit_mask: np.ndarray,
    exp_rad: int,
    p_rad: int,
) -> np.ndarray:
    """Guarded reflection-copy heal with cubic-smoothstep feather.

    This is the proven pre-v0.37 IR implementation, kept byte-for-byte in its
    numerical path so a full-canvas call remains golden-equivalent.
    """

    h, w, _ = img.shape
    res = img.copy()

    cos_a = np.empty(5, dtype=np.float64)
    sin_a = np.empty(5, dtype=np.float64)
    angles = (0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0, -math.pi / 2.0)
    for i in range(5):
        cos_a[i] = math.cos(angles[i])
        sin_a[i] = math.sin(angles[i])

    for y in range(h):
        for x in range(w):
            min_d2 = 1e6
            c_x = 0.0
            c_y = 0.0
            c_n = 0.0
            for dy in range(-exp_rad, exp_rad + 1):
                for dx in range(-exp_rad, exp_rad + 1):
                    ry, rx = y + dy, x + dx
                    if 0 <= ry < h and 0 <= rx < w and hit_mask[ry, rx] > 0.5:
                        d2 = float(dy * dy + dx * dx)
                        if d2 < min_d2:
                            min_d2 = d2
                        c_x += float(rx)
                        c_y += float(ry)
                        c_n += 1.0

            if min_d2 < float(exp_rad * exp_rad + 1):
                dist = np.sqrt(min_d2)
                feather = 1.0 - (dist / float(exp_rad + 1.0))
                if feather < 0.0:
                    feather = 0.0
                feather = feather * feather * (3.0 - 2.0 * feather)

                if feather > 0.001:
                    ux = float(x) - c_x / c_n
                    uy = float(y) - c_y / c_n
                    ul = math.sqrt(ux * ux + uy * uy)
                    if ul < 1e-3:
                        ang = _hash2(float(x), float(y)) * 6.28318530718
                        ux = math.cos(ang)
                        uy = math.sin(ang)
                    else:
                        ux /= ul
                        uy /= ul

                    found = False
                    bg_r = 0.0
                    bg_g = 0.0
                    bg_b = 0.0
                    for k in range(5):
                        rx_dir = ux * cos_a[k] - uy * sin_a[k]
                        ry_dir = ux * sin_a[k] + uy * cos_a[k]
                        sx = int(round(float(x) + rx_dir * float(p_rad)))
                        sy = int(round(float(y) + ry_dir * float(p_rad)))
                        sx = max(0, min(w - 1, sx))
                        sy = max(0, min(h - 1, sy))
                        if hit_mask[sy, sx] < 0.5:
                            bg_r = img[sy, sx, 0]
                            bg_g = img[sy, sx, 1]
                            bg_b = img[sy, sx, 2]
                            found = True
                            break

                    if not found:
                        s_r = np.zeros(8)
                        s_g = np.zeros(8)
                        s_b = np.zeros(8)
                        s_l = np.zeros(8)

                        dy_off = np.array([-p_rad, p_rad, 0, 0, -p_rad, -p_rad, p_rad, p_rad])
                        dx_off = np.array([0, 0, -p_rad, p_rad, -p_rad, p_rad, -p_rad, p_rad])

                        for i in range(8):
                            sy2, sx2 = y + dy_off[i], x + dx_off[i]
                            sy2, sx2 = max(0, min(h - 1, sy2)), max(0, min(w - 1, sx2))
                            r, g, b = img[sy2, sx2, 0], img[sy2, sx2, 1], img[sy2, sx2, 2]
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


def _validate_bounds(bounds: ROI, shape: tuple[int, int]) -> ROI:
    y0, y1, x0, x1 = (int(v) for v in bounds)
    height, width = shape
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError(f"support_bounds {(y0, y1, x0, x1)} fall outside IR shape {shape}")
    return y0, y1, x0, x1


def apply_ir_dust_removal(
    img: ImageBuffer,
    ir: np.ndarray,
    threshold: float,
    inpaint_radius: int,
    scale_factor: float,
    *,
    support_bounds: Optional[ROI] = None,
    support_mask: Optional[np.ndarray] = None,
) -> Tuple[ImageBuffer, np.ndarray]:
    """Threshold IR and repair the exact defect mask.

    The five positional arguments preserve the pre-v0.37 public API.  New
    callers can provide either rectangular ``support_bounds`` in NegPy ROI
    order ``(y0, y1, x0, x1)`` or a boolean ``support_mask``.  Pixels outside
    that support are byte-exact copies of the float32 input and the returned
    defect mask is zero there.
    """

    image = ensure_image(np.asarray(img))
    if np.asarray(ir).shape[:2] != image.shape[:2]:
        return image, np.zeros(image.shape[:2], dtype=np.uint8)
    if support_bounds is not None and support_mask is not None:
        raise ValueError("provide support_bounds or support_mask, not both")
    if inpaint_radius < 1:
        raise ValueError("inpaint_radius must be at least 1")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    plane = normalize_ir(ir)
    # Reuse an already-contiguous float32 pipeline buffer.  A full LS-5000
    # frame is ~282 MB as float RGB, so an unconditional cast-copy would add
    # substantial peak memory without changing a byte.
    base = np.ascontiguousarray(image, dtype=np.float32)
    height, width = plane.shape
    bounds: ROI = (0, height, 0, width)

    if support_bounds is not None:
        bounds = _validate_bounds(support_bounds, plane.shape)
        y0, y1, x0, x1 = bounds
        crop_ir = plane[y0:y1, x0:x1]
        crop_hit = (crop_ir < np.float32(threshold)).astype(np.float32)
        mask_u8 = np.zeros(plane.shape, dtype=np.uint8)
        mask_u8[y0:y1, x0:x1] = (crop_hit * 255).astype(np.uint8)
        if not np.any(crop_hit):
            return base, mask_u8
        heal_img = np.ascontiguousarray(base[y0:y1, x0:x1])
        hit_mask = np.ascontiguousarray(crop_hit)
    else:
        hit_mask = plane < np.float32(threshold)
        if support_mask is not None:
            support = np.asarray(support_mask, dtype=bool)
            if support.shape != plane.shape:
                raise ValueError(f"support_mask shape {support.shape} does not match IR shape {plane.shape}")
            hit_mask &= support
        else:
            support = None
        hit_mask = np.ascontiguousarray(hit_mask.astype(np.float32))
        mask_u8 = (hit_mask * 255).astype(np.uint8)
        if not np.any(hit_mask):
            return base, mask_u8
        heal_img = base

    scale = max(1.0, float(scale_factor))
    exp_rad = min(16, int(max(1.0, float(inpaint_radius) * scale)))
    p_rad = exp_rad + int(max(2.0, 3.0 * scale))
    healed = _heal_with_mask_jit(heal_img, hit_mask, exp_rad, p_rad)

    if support_bounds is not None:
        output = base.copy()
        y0, y1, x0, x1 = bounds
        output[y0:y1, x0:x1] = healed
    elif support_mask is not None:
        assert support is not None
        output = healed
        output[~support] = base[~support]
    else:
        output = healed
    return ensure_image(output), mask_u8
