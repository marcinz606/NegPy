from dataclasses import dataclass


@dataclass(frozen=True)
class VignetteConfig:
    """
    Lab-section luminance vignette, applied after crop and before finish.
    A scalar gain on every channel moves luma, not color, so the edge
    change is a luminance vignette, not a tint.
    """

    strength: float = 0.0  # [-1.0, 1.0] 0 = off, positive = lighten edges, negative = darken edges
    feather: float = 0.5  # [0.0, 1.0] falloff width from the crop edge inward; 0 = a hard edge
    roundness: float = 0.0  # [0.0, 1.0] 0 = circle, 1 = ellipse following the crop aspect
