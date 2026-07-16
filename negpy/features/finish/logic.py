import numpy as np
from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image

# One filed carrier per darkroom: a fixed seed keeps the rebate edge identical
# across every frame and across the CPU/GPU paths (the GPU samples the same array).
_CARRIER_SEED = 1898
CARRIER_SAMPLES = 512
# Roughness slider at 1.0 jitters the rebate width by ±24%.
CARRIER_JITTER = 0.24
# Penumbra: the carrier sits above the negative, so its edge prints soft —
# transition width as a fraction of the rebate width. Mirrored in finish.wgsl.
CARRIER_SOFT = 0.35
_carrier_cache: np.ndarray | None = None


def carrier_profiles() -> np.ndarray:
    """(4, CARRIER_SAMPLES) float32 jitter profiles in [-1, 1], one per edge."""
    global _carrier_cache
    if _carrier_cache is None:
        rng = np.random.default_rng(_CARRIER_SEED)
        raw = rng.standard_normal((4, CARRIER_SAMPLES)).astype(np.float32)
        k = np.exp(-0.5 * (np.arange(-12, 13, dtype=np.float32) / 4.0) ** 2)
        k /= k.sum()
        # Circular smoothing so a profile has no seam where an edge wraps.
        tiled = np.concatenate([raw, raw, raw], axis=1)
        sm = np.stack([np.convolve(row, k, mode="same") for row in tiled])[:, CARRIER_SAMPLES : 2 * CARRIER_SAMPLES]
        sm /= np.max(np.abs(sm), axis=1, keepdims=True)
        _carrier_cache = np.ascontiguousarray(sm, dtype=np.float32)
    return _carrier_cache


def apply_carrier(img: ImageBuffer, width_px: float, rough: float) -> ImageBuffer:
    """
    Filed-out negative carrier: a black rebate frame with a rough inner edge,
    multiplied into the scene-linear image (unexposed film base prints max black).
    """
    if width_px <= 0.0:
        return img

    h, w = img.shape[:2]
    profiles = carrier_profiles()
    soft = max(1.0, width_px * CARRIER_SOFT)
    band = min(int(np.ceil(width_px * (1.0 + CARRIER_JITTER) + soft)) + 1, h, w)
    d = np.arange(band, dtype=np.float32)

    def edge_width(edge: int, count: int) -> np.ndarray:
        s = ((np.arange(count, dtype=np.float32) + 0.5) / np.float32(count)).astype(np.float32)
        idx = np.minimum((s * CARRIER_SAMPLES).astype(np.int32), CARRIER_SAMPLES - 1)
        return width_px * (1.0 + CARRIER_JITTER * rough * profiles[edge, idx])

    out = img.copy()
    a_top = np.clip((d[:, None] - edge_width(0, w)[None, :]) / soft + 0.5, 0.0, 1.0)
    a_bot = np.clip((d[:, None] - edge_width(1, w)[None, :]) / soft + 0.5, 0.0, 1.0)
    a_left = np.clip((d[None, :] - edge_width(2, h)[:, None]) / soft + 0.5, 0.0, 1.0)
    a_right = np.clip((d[None, :] - edge_width(3, h)[:, None]) / soft + 0.5, 0.0, 1.0)
    out[:band] *= a_top[..., None]
    out[h - band :] *= a_bot[::-1][..., None]
    out[:, :band] *= a_left[..., None]
    out[:, w - band :] *= a_right[:, ::-1][..., None]
    return ensure_image(out)


def apply_vignette(img: ImageBuffer, stops: float, size: float, roundness: float = 0.0) -> ImageBuffer:
    """
    Edge burn / hold-back as a true exposure change in scene-linear.

    Args:
        img: Float32 RGB image [0, 1], scene-linear.
        stops: [-2, 2]. Positive = burn (darken edges, more exposure),
            negative = dodge (hold back, lighten edges), 0 = no effect.
        size: [0, 1]. 0 = effect barely visible at extreme corners, 1 = covers entire image from center.
        roundness: [0, 1]. 0 = radial falloff (lens-like), 1 = rectangular
            following the print edges (card-like burn).

    Returns:
        Modified ImageBuffer with the burn applied.
    """
    if stops == 0.0:
        return img

    h, w = img.shape[:2]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5

    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    max_dist = float(np.sqrt(cx * cx + cy * cy))
    d_radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(max_dist, 1.0)
    d_rect = np.maximum(np.abs(xx - cx) / max(cx, 1.0), np.abs(yy - cy) / max(cy, 1.0))
    dist = d_radial * (1.0 - roundness) + d_rect * roundness

    # Remap: size=0 → vignette barely at corners, size=1 → covers entire image
    midpoint = 1.0 - size
    t = (dist - midpoint) / max(1.0 - midpoint, 1e-6)
    t = np.clip(t, 0.0, 1.0)

    # Smooth cosine falloff
    factor = 0.5 * (1.0 - np.cos(t * np.pi))

    result = img * np.exp2(-stops * factor[:, :, np.newaxis])

    return ensure_image(np.clip(result, 0.0, 1.0))
