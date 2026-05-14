from dataclasses import dataclass


@dataclass(frozen=True)
class ScannerSettings:
    """Persisted scanner preferences, stored as JSON blob."""

    last_device_id: str = ""
    dpi: int = 3600
    depth: int = 16
    capture_ir: bool = False
    output_folder: str = ""
    output_format: str = "TIFF"
    filename_pattern: str = "scan_{date}_{seq:03d}"

    @classmethod
    def defaults(cls) -> "ScannerSettings":
        return cls()
