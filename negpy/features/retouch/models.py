from dataclasses import dataclass, field
from typing import List, Tuple

# Heal `size` is a diameter at this reference long edge (px). Pinned so stored
# heals keep their footprint when preview_render_size changes.
HEAL_SIZE_REF = 1600


@dataclass(frozen=True)
class RetouchConfig:
    dust_remove: bool = False
    dust_threshold: float = 0.66
    dust_size: int = 4
    manual_dust_spots: List[Tuple[float, float, float]] = field(default_factory=list)
    # Each stroke: (points, size, src_dx, src_dy); points = [[nx, ny], ...] source-normalized,
    # size = diameter at HEAL_SIZE_REF scale, (src_dx, src_dy) = source-normalized offset to
    # the clone source. A single-point stroke is a spot. manual_dust_spots is the legacy
    # pre-stroke format.
    manual_heal_strokes: List[Tuple] = field(default_factory=list)
    manual_dust_size: int = 6
    ir_dust_remove: bool = False
    # Sensitivity on the normalized IR ratio (higher = conservative). Default 0.66 →
    # cutoff 0.59 with attenuation on: division fixes shallow dust, the fill rebuilds cores.
    ir_threshold: float = 0.66
    # IR-division tier: recover the image under semi-transparent dust (no cloning).
    # Tracks ir_dust_remove from the single "IR Removal" control; B&W/Kodachrome
    # frames are auto-skipped by the degenerate guard, not this flag.
    ir_attenuation: bool = True
