from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from negpy.features.exposure.models import EXPOSURE_CONSTANTS


class ProcessMode(StrEnum):
    C41 = "Color Negative"
    BW = "B&W Negative"
    E6 = "Transparency"

    @classmethod
    def _missing_(cls, value: object) -> "ProcessMode":
        """Legacy chemistry codes (the values before the rename), and anything stale.

        An unrecognised mode has always rendered as color negative — every branch in
        the pipeline reads `if BW / elif E6 / else` — so a corrupt saved value stays
        non-fatal here rather than raising on load.
        """
        return _LEGACY_MODES.get(str(value), cls.C41)


_LEGACY_MODES = {"C41": ProcessMode.C41, "B&W": ProcessMode.BW, "E-6": ProcessMode.E6}


# Built-in fallback crosstalk matrix (row-major 3x3) used when no profile is baked.
DEFAULT_CROSSTALK_MATRIX = (1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0)


@dataclass(frozen=True)
class ProcessConfig:
    """
    Core film/sensor processing parameters.
    """

    process_mode: ProcessMode = ProcessMode.C41
    linear_raw: bool = False
    # Correct narrowband RGB camera scans via the bundled RGBScan input profile
    # (applied at preview soft-proof / export; an explicit Input ICC overrides it).
    narrowband_scan: bool = False
    analysis_buffer: float = 0.05
    # Optional freehand analysis region, normalized in the transformed (display)
    # image, the same space as the manual crop rect. When set it is the exact area the
    # black/white-point meters read. None falls back to the analysis_buffer slider.
    analysis_rect: Optional[tuple] = None
    # Two normalization clip axes: luma sets the black/white-point span, color sets
    # the per-channel balance clip (orange-mask cast removal).
    luma_range_clip: float = 0.0
    color_range_clip: float = float(EXPOSURE_CONSTANTS["base_color_clip"])
    # Off by default. A slide's density runs to Dmax but only its top decades carry
    # picture, so a per-frame stretch crushes the picture into the top of the print
    # curve. Off renders the capture as shot; turn it on for faded film.
    e6_normalize: bool = False
    # Roll-wide baseline applied independently per axis: luma (span) and color (cast).
    use_luma_average: bool = False
    use_color_average: bool = False
    locked_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    locked_ceils: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_ceils: tuple[float, float, float] = (0.0, 0.0, 0.0)

    white_point_offset: float = 0.0
    black_point_offset: float = 0.0
    # Per-layer trims on the global white/black point: scanner-style per-channel levels.
    white_point_trim_red: float = 0.0
    white_point_trim_green: float = 0.0
    white_point_trim_blue: float = 0.0
    black_point_trim_red: float = 0.0
    black_point_trim_green: float = 0.0
    black_point_trim_blue: float = 0.0

    # Spectral crosstalk (dye unmix) on the raw NEGATIVE densities, before bounds
    # analysis and the stretch. That is the correct domain: by Beer-Lambert the
    # secondary dye absorptions are linear in negative dye density. Matrix is 9
    # floats row-major, baked from a crosstalk profile. Legacy `color_separation`
    # is migrated in WorkspaceConfig.from_flat_dict.
    crosstalk_strength: float = 0.0
    crosstalk_matrix: Optional[tuple] = None
    crosstalk_profile: str = "Generic C41"
    # Film process the crosstalk profile was derived for, baked at selection so the
    # render can gate on it without disk access. The dye set differs between C-41 and
    # E-6, so a mismatch resolves to identity instead of mixing in the wrong matrix.
    crosstalk_process: str = ProcessMode.C41

    # Sensor (CFA) crosstalk unmix on the LINEAR capture, before inversion. A
    # per-setup property calibrated from three bare-light R/G/B exposures
    # (features/process/sensor.py). 9 floats row-major; None = off.
    sensor_matrix: Optional[tuple] = None
    sensor_profile: str = "None"

    # Light-source hue rotation in degrees, applied to the print in CIELAB a*b*
    # (features/process/hue.py); 0.0 = off.
    hue_trim: float = 0.0

    lock_bounds: bool = False

    roll_name: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Ensure JSON-loaded lists are converted back to tuples.
        """
        # Not a MIGRATIONS entry: the old mode names also reach us from sticky settings
        # and asset dicts, not only a loaded flat config, so this runs on every build.
        object.__setattr__(self, "process_mode", ProcessMode(self.process_mode))
        object.__setattr__(self, "locked_floors", tuple(self.locked_floors))
        object.__setattr__(self, "locked_ceils", tuple(self.locked_ceils))
        object.__setattr__(self, "local_floors", tuple(self.local_floors))
        object.__setattr__(self, "local_ceils", tuple(self.local_ceils))
        if self.crosstalk_matrix is not None:
            object.__setattr__(self, "crosstalk_matrix", tuple(self.crosstalk_matrix))
        if self.sensor_matrix is not None:
            object.__setattr__(self, "sensor_matrix", tuple(self.sensor_matrix))
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


def scan_setup_values(capture: str, light: str) -> tuple[bool, bool]:
    """(linear_raw, narrowband_scan) for a scanning rig — capture is "camera"/"scanner",
    light is "white"/"narrowband". A camera under white light is the only combination that
    wants the as-shot-WB decode."""
    return not (capture == "camera" and light == "white"), light == "narrowband"


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
