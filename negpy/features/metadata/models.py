from dataclasses import dataclass
from typing import Optional


# "" is a format nobody set, and it is a row of its own: it is not "Other".
FORMAT_UNSET = "—"
FORMAT_OPTIONS: tuple[str, ...] = (FORMAT_UNSET, "35mm", "120", "4×5", "8×10", "110", "Other")


def format_label(value: str) -> str:
    """The dropdown row for a stored format."""
    return value if value in FORMAT_OPTIONS else FORMAT_UNSET


def format_value(label: str) -> str:
    """The stored format for a dropdown row."""
    return "" if label == FORMAT_UNSET else label


# Panel order: strongest push first.
PUSH_PULL_VALUES: tuple[int, ...] = (3, 2, 1, 0, -1, -2, -3)

PUSH_PULL_LABELS = {
    -3: "Pull -3",
    -2: "Pull -2",
    -1: "Pull -1",
    0: "Normal",
    1: "Push +1",
    2: "Push +2",
    3: "Push +3",
}

# Ordered keys for EXIF ImageDescription; only non-empty values are joined.
DESCRIPTION_FIELD_ORDER: tuple[str, ...] = (
    "camera",
    "lens",
    "film",
    "iso",
    "format",
    "developer",
    "push_pull",
    "scanning",
)
DESCRIPTION_FIELD_LABELS: dict[str, str] = {
    "camera": "Camera",
    "lens": "Lens",
    "film": "Film stock",
    "iso": "Film ISO",
    "format": "Format",
    "developer": "Developer",
    "push_pull": "Push / Pull",
    "scanning": "Scanning",
}
# Preserve pre-selector behaviour: gear only.
DEFAULT_DESCRIPTION_FIELDS: tuple[str, ...] = ("camera", "lens", "film", "iso")
_DESCRIPTION_FIELD_SET = frozenset(DESCRIPTION_FIELD_ORDER)

# One library pick and every value read from it. Copied and stored as a unit — split,
# a pasted camera would sit under another frame's lens in the panel.
PROCESS_FIELDS: tuple[str, ...] = (
    "developer",
    "process_dilution",
    "push_pull",
    "process_time_seconds",
    "process_temperature_c",
    "process_id",
)
SCANNING_FIELDS: tuple[str, ...] = ("scanning", "scanning_id")

GEAR_FIELDS: tuple[str, ...] = (
    "camera_id",
    "lens_id",
    "film_stock_id",
    "camera_make",
    "camera_model",
    "lens_make",
    "lens_model",
    "focal_length_mm",
    "max_aperture",
    "film",
    "film_manufacturer",
    "film_iso",
    "film_color_type",
    # The stock carries the film's format, so it travels with the pick: split off, a preset
    # for a 120 stock would leave a 35mm frame claiming 35mm while naming the 120 stock.
    "format",
    "format_other",
)


def normalize_description_fields(fields: object) -> tuple[str, ...]:
    """Keep known keys in canonical order (JSON lists round-trip cleanly)."""
    if fields is None:
        return DEFAULT_DESCRIPTION_FIELDS
    if isinstance(fields, str):
        raw = {fields}
    else:
        try:
            raw = set(fields)
        except TypeError:
            return DEFAULT_DESCRIPTION_FIELDS
    return tuple(k for k in DESCRIPTION_FIELD_ORDER if k in raw and k in _DESCRIPTION_FIELD_SET)


def resolve_description_fields(fields: object, sticky: object = None) -> tuple[str, ...]:
    """Per-frame fields if set; otherwise sticky (or gear-only defaults)."""
    if fields is not None:
        return normalize_description_fields(fields)
    if sticky is not None:
        return normalize_description_fields(sticky)
    return DEFAULT_DESCRIPTION_FIELDS


@dataclass(frozen=True)
class MetadataConfig:
    """
    Custom analog photography metadata written to exported files.
    Empty strings = field not set (nothing written to export).
    """

    # Gear library references (empty = manual entry / not linked)
    camera_id: str = ""
    lens_id: str = ""
    film_stock_id: str = ""

    # Structured gear fields (resolved from library or manual)
    camera_make: str = ""
    camera_model: str = ""
    lens_make: str = ""
    lens_model: str = ""
    focal_length_mm: Optional[float] = None
    max_aperture: Optional[float] = None
    film_iso: Optional[int] = None
    film_manufacturer: str = ""
    film_color_type: str = ""

    film: str = ""
    format: str = ""  # "35mm" | "120" | "4×5" | "8×10" | "110" | "Other" | ""
    format_other: str = ""  # shown when format == "Other"
    # Library references for the development recipe and the digitizing setup
    process_id: str = ""
    scanning_id: str = ""

    developer: str = ""
    process_dilution: str = ""  # free text: "1+50", "1+1", "stock"
    push_pull: int = 0  # -3..+3, 0 = Normal
    process_time_seconds: Optional[int] = None
    process_temperature_c: Optional[float] = None
    scanning: str = ""

    # Original capture instant, ISO-8601 truncated to the precision the user knows.
    capture_date: str = ""

    # Capture place: WGS-84 position and the place names for it.
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    location_city: str = ""
    location_state: str = ""
    location_country: str = ""

    # Scanlight capture identity (not process.roll_name / Roll Analysis)
    capture_roll: str = ""
    capture_frame: Optional[int] = None

    # When True, export copies the source EXIF and XMP unchanged and NegPy writes no metadata.
    protect_original_metadata: bool = False

    exposure_override: str = ""  # free-text e.g. "1/125s f/2.8 ISO 400"; empty = use source EXIF

    # EXIF ImageDescription field set. None inherits the sticky roll choice on open. An
    # explicit tuple is per-frame and is not overwritten by sticky.
    description_fields: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.description_fields is None:
            return
        normalized = normalize_description_fields(self.description_fields)
        if normalized != self.description_fields:
            object.__setattr__(self, "description_fields", normalized)
