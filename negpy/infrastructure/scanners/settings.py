from dataclasses import dataclass, field

Rect = tuple[float, float, float, float]


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
    scan_window: Rect | None = None
    frame_offset_mm: float = 0.0
    # Feed-axis drift (mm/frame): frame N gets frame_offset_mm + (N-1) * modifier,
    # floored at 0. Corrects progressive frame-gap drift along a strip.
    frame_offset_modifier_mm: float = 0.0
    # Per-frame crop windows (absent key = full frame) and the strip-dialog frame
    # selection. ponytail: dict field means ScannerSettings is unhashable; nothing
    # hashes it — switch to a sorted tuple of pairs if that ever changes.
    frame_windows: dict[int, Rect] = field(default_factory=dict)
    selected_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # JSON round-trips tuples as lists and dict keys as strings; coerce back.
        if isinstance(self.scan_window, list):
            object.__setattr__(self, "scan_window", tuple(self.scan_window))
        if isinstance(self.frame_windows, dict):
            object.__setattr__(
                self,
                "frame_windows",
                {int(k): tuple(v) for k, v in self.frame_windows.items()},
            )
        if isinstance(self.selected_frames, list):
            object.__setattr__(self, "selected_frames", tuple(self.selected_frames))

    @classmethod
    def defaults(cls) -> "ScannerSettings":
        return cls()


def resolve_batch_selection(
    settings: ScannerSettings, frame_from: int, frame_to: int
) -> tuple[tuple[int, ...], dict[int, Rect], Rect | None]:
    """(frames, per-frame windows, base window) for a BatchRequest.

    The strip-dialog selection wins when present; otherwise fall back to the
    sidebar spinbox range with the single reused scan_window.
    """
    if settings.selected_frames:
        frames = tuple(sorted(settings.selected_frames))
        windows = {f: settings.frame_windows[f] for f in frames if f in settings.frame_windows}
        return frames, windows, None
    return tuple(range(frame_from, frame_to + 1)), {}, settings.scan_window
