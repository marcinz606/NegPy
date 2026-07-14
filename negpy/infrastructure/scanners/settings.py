from dataclasses import dataclass


@dataclass(frozen=True)
class ScannerSettings:
    """Persisted scanner preferences, stored as JSON blob."""

    last_device_id: str = ""
    dpi: int = 3600
    depth: int = 16
    capture_ir: bool = False
    autofocus: bool = True
    samples_per_scan: int = 1
    # Hardware auto-exposure (SANE `ae`). A device-level style preference like
    # autofocus/samples_per_scan above, so it persists the same way.
    auto_exposure: bool = False
    # Validated archival recipe: capture_ir + samples_per_scan=4 together.
    # Persisted like the other device-profile toggles above. Frame number and
    # registered geometry (subframe_mm/br_y_device_px) are deliberately NOT
    # persisted here — they are specific to one physical frame position and
    # silently replaying them onto a different frame/session would be wrong.
    archival_split_capture: bool = False
    output_folder: str = ""
    output_format: str = "TIFF"
    filename_pattern: str = '{{ date }}_{{ "%03d" % seq }}'

    @classmethod
    def defaults(cls) -> "ScannerSettings":
        return cls()
