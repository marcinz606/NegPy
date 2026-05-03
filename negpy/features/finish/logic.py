import numpy as np
from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image


def apply_vignette(img: ImageBuffer, strength: float, size: float) -> ImageBuffer:
    """
    Radial vignette overlay using cosine falloff.

    Args:
        img: Float32 RGB image [0, 1].
        strength: [-1, 1]. Negative = darken edges, positive = brighten edges, 0 = no effect.
        size: [0, 1]. 0 = vignette barely visible at extreme corners, 1 = covers entire image from center.

    Returns:
        Modified ImageBuffer with vignette applied.
    """
    if strength == 0.0:
        return img

    h, w = img.shape[:2]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5

    # Euclidean distance from center, normalized so corners = 1.0
    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    dy = (yy - cy) / max(cy, 1.0)
    dx = (xx - cx) / max(cx, 1.0)
    dist = np.sqrt(dx**2 + dy**2)  # range [0, 1]

    # Remap: size=0 → vignette barely at corners, size=1 → covers entire image
    midpoint = 1.0 - size
    t = (dist - midpoint) / max(1.0 - midpoint, 1e-6)
    t = np.clip(t, 0.0, 1.0)

    # Smooth cosine falloff
    factor = 0.5 * (1.0 - np.cos(t * np.pi))

    strength_abs = abs(strength)

    if strength < 0.0:
        # Darken: multiply toward black
        result = img * (1.0 - factor[:, :, np.newaxis] * strength_abs)
    else:
        # Brighten: blend toward white
        result = img + (1.0 - img) * factor[:, :, np.newaxis] * strength_abs

    return ensure_image(np.clip(result, 0.0, 1.0))
