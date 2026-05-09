from dataclasses import dataclass


@dataclass(frozen=True)
class MetadataConfig:
    """
    Custom analog photography metadata written to exported files.
    Empty strings = field not set (nothing written to export).
    """

    film: str = ""
    format: str = ""  # "35mm" | "120" | "4x5" | "8x10" | "Other" | ""
    format_other: str = ""  # shown when format == "Other"
    developer: str = ""
    push_pull: int = 0  # -3..+3, 0 = Normal
    scanning: str = ""
    sync_to_batch: bool = False

    # EXIF field overrides (editable by user, empty = use source EXIF)
    camera_override: str = ""
    lens_override: str = ""
    exposure_override: str = ""
