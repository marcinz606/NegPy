from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Optional

from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig

if TYPE_CHECKING:
    from negpy.domain.models import WorkspaceConfig


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


class DemosaicMode(StrEnum):
    """CFA interpolation, mapped to rawpy in loaders/helpers.py. AUTO keeps NegPy's own
    choice per path. Availability is asked of rawpy at runtime, never assumed here."""

    AUTO = "Auto"
    LINEAR = "Linear"
    VNG = "VNG"
    PPG = "PPG"
    AHD = "AHD"
    DCB = "DCB"
    DHT = "DHT"
    AAHD = "AAHD"

    @classmethod
    def _missing_(cls, value: object) -> "DemosaicMode":
        """A value this build cannot run, or a stale one, decodes as AUTO rather than raising."""
        return cls.AUTO


def cast_removal_for_mode(mode: str, strength: float) -> float:
    """The Cast Removal strength `mode` starts at, given the strength carried in.

    A transparency starts at 0 and a negative at the shipped default: on a negative the
    control defeats the orange mask, which is never part of the picture, while on a slide
    it corrects a faded original's crossover — a deliberate act, since a slide's cast can
    be the photograph. Only the other mode's default is rewritten, so a strength the user
    chose survives a mode switch.
    """
    default = float(ExposureConfig.cast_removal_strength)
    if mode == ProcessMode.E6:
        return 0.0 if strength == default else strength
    return default if strength == 0.0 else strength


def auto_meter_for_mode(mode: str, current: bool) -> bool:
    """The value Auto Density/Auto Grade's toggle carries after a switch into Film Mode *mode*.

    A negative starts metered (True): its exposure has no meaning until printed, so a meter
    is what makes it printable at all. A slide starts unmetered (False): it was exposed
    deliberately, and a meter would make a bracket converge. Only the other mode's default is
    rewritten, so a toggle the user chose survives the switch (mirrors cast_removal_for_mode).
    """
    negative_default, slide_default = True, False
    if mode == ProcessMode.E6:
        return slide_default if current == negative_default else current
    return negative_default if current == slide_default else current


def mode_aware_exposure_reset(mode: str, base: ExposureConfig) -> ExposureConfig:
    """`base` (typically the shipped default exposure section) with Cast Removal and Auto
    Density/Auto Grade at *mode*'s own defaults instead of the flat values `base` carries.
    Single source for every reset path that resets a whole exposure section rather than one
    field at a time."""
    return replace(
        base,
        cast_removal_strength=cast_removal_for_mode(mode, base.cast_removal_strength),
        auto_exposure=auto_meter_for_mode(mode, base.auto_exposure),
        auto_normalize_contrast=auto_meter_for_mode(mode, base.auto_normalize_contrast),
    )


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
    # The source is a finished positive (a scanned print, an export from other software, a
    # scanner's own positive) rather than a raw capture. Slide only, held in __post_init__.
    # It decodes on its embedded profile, sRGB when untagged, instead of as literal linear
    # data, and skips metering, negative inversion, the baseline lift and the filmic curve.
    # See effective_linear_raw and path.render_path.
    positive_source: bool = False
    # See loaders/helpers.get_best_demosaic_algorithm for what AUTO resolves to on each path.
    demosaic_preview: DemosaicMode = DemosaicMode.AUTO
    demosaic_export: DemosaicMode = DemosaicMode.AUTO
    # libraw HighlightMode: 0=Clip (current behaviour), 2=Blend, 3-9=Reconstruct(level).
    # 1 (Ignore) is reserved for a separate decode-correctness fix and is never valid here.
    # Only meaningful on the positive path; see effective_highlight_reconstruction.
    highlight_reconstruction: int = 0
    analysis_buffer: float = 0.05
    # Optional freehand analysis region, normalized in the transformed (display)
    # image, the same space as the manual crop rect. When set it is the exact area the
    # black/white-point meters read. None falls back to the analysis_buffer slider.
    analysis_rect: Optional[tuple] = None
    # Two normalization clip axes: luma sets the black/white-point span, color sets
    # the per-channel balance clip (orange-mask cast removal).
    luma_range_clip: float = 0.0
    color_range_clip: float = float(EXPOSURE_CONSTANTS["base_color_clip"])
    # Roll-wide baseline applied independently per axis: luma (span) and color (cast).
    use_luma_average: bool = False
    use_color_average: bool = False
    locked_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    locked_ceils: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_floors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_ceils: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Roll/scene Cast Removal: the pooled neutral axis in raw log, in the meter's own shape
    # (midtone, shadow, highlight or None, confidence) plus an offset weight
    # (pool_neutral_axis). Color Negative only.
    use_cast_average: bool = False
    locked_neutral_axis: Optional[tuple] = None

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
    # Where locked_floors/ceils came from: "roll:<roll id>", "scene:<scene id>" or "frame:<file name>".
    baseline_source: str = ""

    def __post_init__(self) -> None:
        """
        Ensure JSON-loaded lists are converted back to tuples.
        """
        # Not a MIGRATIONS entry: the old mode names also reach us from sticky settings
        # and asset dicts, not only a loaded flat config, so this runs on every build.
        object.__setattr__(self, "process_mode", ProcessMode(self.process_mode))
        # Slide-only, and every path into a config -- saved row, sticky settings, roll
        # default, asset dict -- has to land where the panel does.
        if self.positive_source and self.process_mode != ProcessMode.E6:
            object.__setattr__(self, "positive_source", False)
        object.__setattr__(self, "locked_floors", tuple(self.locked_floors))
        object.__setattr__(self, "locked_ceils", tuple(self.locked_ceils))
        object.__setattr__(self, "local_floors", tuple(self.local_floors))
        object.__setattr__(self, "local_ceils", tuple(self.local_ceils))
        if self.locked_neutral_axis is not None:
            object.__setattr__(self, "locked_neutral_axis", neutral_axis_tuple(self.locked_neutral_axis))
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


def neutral_axis_tuple(axis) -> tuple:
    """A neutral axis read back from JSON as nested tuples; a pooled one carries a fifth
    element, its offset weight."""
    mid, shadow, highlight, *scalars = axis
    return (tuple(mid), tuple(shadow), tuple(highlight) if highlight is not None else None, *(float(v) for v in scalars))


def pooled_neutral_axis(process: ProcessConfig) -> Optional[tuple]:
    """The roll/scene neutral axis this frame renders Cast Removal with, or None to meter its own."""
    if process.use_cast_average and process.process_mode == ProcessMode.C41:
        return process.locked_neutral_axis
    return None


def invalidate_local_bounds(process: ProcessConfig) -> dict:
    """Returns kwargs for dataclasses.replace that clear local bounds; no-op when lock_bounds=True."""
    if process.lock_bounds:
        return {}
    return {"local_floors": (0.0, 0.0, 0.0), "local_ceils": (0.0, 0.0, 0.0)}


def with_process_mode(config: "WorkspaceConfig", mode: str) -> "WorkspaceConfig":
    """*config* switched to Film Mode *mode*: the Cast Removal and Auto Density/Auto Grade
    defaults move with it, and leaving Slide drops Positive. Every path that sets a frame's mode
    goes through here. The same mode returns *config* unchanged, since the default rewrites
    would read a slide's deliberate negative-default value as untouched."""
    proc, exp = config.process, config.exposure
    mode = ProcessMode(mode)
    if mode == proc.process_mode:
        return config
    exp = replace(
        exp,
        cast_removal_strength=cast_removal_for_mode(mode, exp.cast_removal_strength),
        auto_exposure=auto_meter_for_mode(mode, exp.auto_exposure),
        auto_normalize_contrast=auto_meter_for_mode(mode, exp.auto_normalize_contrast),
    )
    proc = replace(
        proc, process_mode=mode, positive_source=proc.positive_source and mode == ProcessMode.E6, **invalidate_local_bounds(proc)
    )
    return replace(config, process=proc, exposure=exp)


def with_positive_source(config: "WorkspaceConfig", checked: bool) -> "WorkspaceConfig":
    """*config* with Positive set to *checked*. Unchanged outside Slide, where Positive does not apply."""
    proc = config.process
    if proc.process_mode != ProcessMode.E6 or proc.positive_source == checked:
        return config
    return replace(config, process=replace(proc, positive_source=checked, **invalidate_local_bounds(proc)))


def with_film_fields(config: "WorkspaceConfig", fields: dict) -> "WorkspaceConfig":
    """*config* with the Film card's fields in *fields* applied through with_process_mode and
    with_positive_source, popping them from *fields* so the caller overlays only the rest."""
    if "process_mode" in fields:
        config = with_process_mode(config, fields.pop("process_mode"))
    if "positive_source" in fields:
        config = with_positive_source(config, fields.pop("positive_source"))
    return config


def scan_setup_values(capture: str, light: str) -> tuple[bool, bool]:
    """(linear_raw, narrowband_scan) for a scanning rig — capture is "camera"/"scanner",
    light is "white"/"narrowband". A camera under white light is the only combination that
    wants the as-shot-WB decode."""
    return not (capture == "camera" and light == "white"), light == "narrowband"


def per_channel_point_offsets(process: ProcessConfig) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Per-channel white/black point offsets: global + per-layer trim. Single source of truth
    for the CPU normalization and the GPU uniform pack; a slide negates them
    (transparency.logic.transfer_point_offsets).
    """
    wp3 = (
        process.white_point_offset + process.white_point_trim_red,
        process.white_point_offset + process.white_point_trim_green,
        process.white_point_offset + process.white_point_trim_blue,
    )
    bp3 = (
        process.black_point_offset + process.black_point_trim_red,
        process.black_point_offset + process.black_point_trim_green,
        process.black_point_offset + process.black_point_trim_blue,
    )
    return wp3, bp3
