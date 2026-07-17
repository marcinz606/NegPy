from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from negpy.features.exposure.models import EXPOSURE_CONSTANTS


class ProcessMode(StrEnum):
    C41 = "C41"
    BW = "B&W"
    E6 = "E-6"


# Built-in fallback crosstalk matrix (row-major 3x3) used when no profile is baked.
DEFAULT_CROSSTALK_MATRIX = (1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0)


@dataclass(frozen=True)
class ProcessConfig:
    """
    Core film/sensor processing parameters.
    """

    process_mode: ProcessMode = ProcessMode.C41
    linear_raw: bool = True
    # Correct narrowband RGB camera scans via the bundled RGBScan input profile
    # (applied at preview soft-proof / export; an explicit Input ICC overrides it).
    narrowband_scan: bool = False
    analysis_buffer: float = 0.05
    # Optional freehand analysis region, normalized in the transformed (display) image —
    # the same space as the manual crop rect. When set it defines the exact area the
    # black/white-point meters read (the centered analysis_buffer inset is bypassed);
    # None falls back to the analysis_buffer slider.
    analysis_rect: Optional[tuple] = None
    # Two independent normalization clip axes: luma drives black/white-point span
    # (dynamic range), colour is the per-channel-balance clip percentile (orange-mask
    # cast removal), defaulting to the robust base_color_clip neutral.
    luma_range_clip: float = 0.0
    color_range_clip: float = float(EXPOSURE_CONSTANTS["base_color_clip"])
    e6_normalize: bool = True
    # Roll-wide baseline applied independently per axis: luma (span) and colour (cast).
    use_luma_average: bool = False
    use_colour_average: bool = False
    locked_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    locked_ceils: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_ceils: tuple[float, float, float] = (0.0, 0.0, 0.0)

    white_point_offset: float = 0.0
    black_point_offset: float = 0.0
    # Per-layer trims on top of the global white/black point (per-dye-layer
    # film-base / Dmax correction — scanner-style per-channel levels).
    white_point_trim_red: float = 0.0
    white_point_trim_green: float = 0.0
    white_point_trim_blue: float = 0.0
    black_point_trim_red: float = 0.0
    black_point_trim_green: float = 0.0
    black_point_trim_blue: float = 0.0

    # Spectral crosstalk (dye unmix), applied to the raw NEGATIVE densities
    # before bounds analysis and the stretch — the physically correct domain
    # (Beer–Lambert: secondary dye absorptions are linear in negative dye
    # density, and the bundled matrices were derived from negative spectral
    # dye-density curves). Matrix is 9 floats (row-major), baked from a
    # crosstalk profile. Replaces the old Lab-stage positive-domain op; legacy
    # `color_separation` is migrated in WorkspaceConfig.from_flat_dict.
    crosstalk_strength: float = 0.0
    crosstalk_matrix: Optional[tuple] = None
    crosstalk_profile: str = "Default"

    lock_bounds: bool = False

    roll_name: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Ensure JSON-loaded lists are converted back to tuples.
        """
        object.__setattr__(self, "locked_floors", tuple(self.locked_floors))
        object.__setattr__(self, "locked_ceils", tuple(self.locked_ceils))
        object.__setattr__(self, "local_floors", tuple(self.local_floors))
        object.__setattr__(self, "local_ceils", tuple(self.local_ceils))
        if self.crosstalk_matrix is not None:
            object.__setattr__(self, "crosstalk_matrix", tuple(self.crosstalk_matrix))
        if self.analysis_rect is not None:
            object.__setattr__(self, "analysis_rect", tuple(self.analysis_rect))

    @property
    def is_local_initialized(self) -> bool:
        """Checks if per-file auto-exposure has been performed."""
        return any(v != 0.0 for v in self.local_floors)

    @property
    def is_locked_initialized(self) -> bool:
        """Checks if a roll-wide baseline is available."""
        return any(v != 0.0 for v in self.locked_floors)


def invalidate_local_bounds(process: ProcessConfig) -> dict:
    """Returns kwargs for dataclasses.replace that clear local bounds; no-op when lock_bounds=True."""
    if process.lock_bounds:
        return {}
    return {"local_floors": (0.0, 0.0, 0.0), "local_ceils": (0.0, 0.0, 0.0)}


def per_channel_point_offsets(process: ProcessConfig, e6: bool) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Signed per-channel white/black point offsets: global + per-layer trim.
    E6 negates (positive film reverses the floor/ceil roles). Single source of
    truth for the CPU normalization and the GPU uniform pack.
    """
    sign = -1.0 if e6 else 1.0
    wp3 = (
        sign * (process.white_point_offset + process.white_point_trim_red),
        sign * (process.white_point_offset + process.white_point_trim_green),
        sign * (process.white_point_offset + process.white_point_trim_blue),
    )
    bp3 = (
        sign * (process.black_point_offset + process.black_point_trim_red),
        sign * (process.black_point_offset + process.black_point_trim_green),
        sign * (process.black_point_offset + process.black_point_trim_blue),
    )
    return wp3, bp3
