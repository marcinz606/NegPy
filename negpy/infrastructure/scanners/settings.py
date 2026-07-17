from dataclasses import dataclass


@dataclass(frozen=True)
class ScannerSettings:
    """Persisted scanner preferences, stored as JSON blob."""

    last_device_id: str = ""
    dpi: int = 3600
    depth: int = 16
    capture_ir: bool = False
    autofocus: bool = True
    auto_exposure: bool = False
    frame_from: int = 1
    frame_to: int = 1
    output_folder: str = ""
    output_format: str = "TIFF"
    filename_pattern: str = '{{ date }}_{{ "%03d" % seq }}'
    scan_window: tuple[float, float, float, float] | None = None
    frame_offset_mm: float = 0.0

    def __post_init__(self) -> None:
        # JSON round-trips the tuple as a list; coerce back to keep the frozen dataclass hashable.
        if isinstance(self.scan_window, list):
            object.__setattr__(self, "scan_window", tuple(self.scan_window))

    @classmethod
    def defaults(cls) -> "ScannerSettings":
        return cls()
