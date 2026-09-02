from collections.abc import Iterable
from dataclasses import dataclass, field, fields

from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID

Rect = tuple[float, float, float, float]

#: Written as one 16-bit grey plane instead of three. Film that carries a single record
#: (a B&W negative) reads the same off one plane, at a third of the size.
MONO_TIFF = "TIFF (mono)"
#: What the Format combo offers, in order.
OUTPUT_FORMATS = ("TIFF", MONO_TIFF)


@dataclass(frozen=True)
class ScannerSettings:
    """Persisted scanner preferences, stored as JSON blob."""

    last_device_id: str = ""
    backend: str = DEFAULT_BACKEND_ID

    dpi: int = 3600
    depth: int = 16
    capture_ir: bool = False
    multi_exposure: bool = False
    autofocus: bool = True
    auto_exposure: bool = False
    # Hardware scan exposure time in microseconds (SANE `scan-exposure-time`). None is the
    # scanner default. Only meaningful when the device exposes the option.
    exposure_time_us: int | None = None
    # Let the transport remove dust itself, baked into the file it writes.
    clean: bool = False
    samples: int = 1
    superfine: bool = False
    # Frame length for a transport that measures the strip; None lets it decide.
    film_format: str | None = None
    film_type: str = "negative"
    output_folder: str = ""
    output_format: str = "TIFF"
    filename_pattern: str = '{{ date }}_{{ "%03d" % seq }}'
    scan_window: Rect | None = None
    frame_offset_mm: float = 0.0
    # Feed-axis drift (mm/frame): frame N gets frame_offset_mm + (N-1) * modifier, floored at
    # 0. Corrects progressive frame-gap drift along a strip.
    frame_offset_modifier_mm: float = 0.0
    # Per-frame crop windows (an absent key means the full frame) and the strip-dialog frame
    # selection. ponytail: a dict field makes ScannerSettings unhashable. Nothing hashes it;
    # switch to a sorted tuple of pairs if that ever changes.
    frame_windows: dict[int, Rect] = field(default_factory=dict)
    selected_frames: tuple[int, ...] = ()
    # Per-frame feed-axis correction (mm), added on top of frame_offset_mm + drift. An absent
    # key means no correction for that frame.
    frame_offsets: dict[int, float] = field(default_factory=dict)

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
        if isinstance(self.frame_offsets, dict):
            object.__setattr__(self, "frame_offsets", {int(k): float(v) for k, v in self.frame_offsets.items()})

    @classmethod
    def defaults(cls) -> "ScannerSettings":
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "ScannerSettings":
        """Build from a persisted blob, migrating and dropping keys this version dropped.

        An unknown key must not throw: the whole blob would fall back to defaults and take
        every unrelated preference with it.
        """
        data = dict(data)
        # DNG output is retired; a saved one lands on TIFF, the other 16-bit master.
        if str(data.get("output_format", "")).upper() == "DNG":
            data["output_format"] = "TIFF"
        first, last = data.pop("frame_from", None), data.pop("frame_to", None)
        if not data.get("selected_frames") and isinstance(first, int) and isinstance(last, int) and (first, last) != (1, 1):
            data["selected_frames"] = tuple(range(first, last + 1))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def resolve_batch_selection(
    settings: ScannerSettings, *, capacity: int | None = None, whole_strip: bool = False
) -> tuple[tuple[int, ...], dict[int, Rect], Rect | None]:
    """(frames, per-frame windows, base window) for a BatchRequest.

    A named selection wins. With none, a transport that measures the film gets an empty tuple,
    meaning every frame it finds; a feeder gets every slot it holds, because its own frame
    count reads 0 and an empty tuple there would scan nothing.
    """
    if settings.selected_frames:
        frames = tuple(sorted(settings.selected_frames))
        windows = {f: settings.frame_windows[f] for f in frames if f in settings.frame_windows}
        return frames, windows, None
    if whole_strip or capacity is None:
        return (), {}, settings.scan_window
    return tuple(range(1, capacity + 1)), {}, settings.scan_window


def parse_frame_spec(text: str) -> tuple[int, ...]:
    """Frame numbers from an operator's list: "1-6", "1,2,5", "1-3, 5". Empty means every frame.

    Raises ValueError on anything else, so a typo cannot quietly scan the wrong film.
    """
    frames: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        first, sep, last = part.partition("-")
        try:
            lo = int(first)
            hi = int(last) if sep else lo
        except ValueError:
            raise ValueError(f"Cannot read {part!r} as a frame number") from None
        if lo < 1 or hi < lo:
            raise ValueError(f"{part!r} is not a frame range")
        frames.update(range(lo, hi + 1))
    return tuple(sorted(frames))


def format_frame_spec(frames: Iterable[int]) -> str:
    """The compact form of a frame selection, contiguous runs collapsed: (1,2,3,5) → "1-3,5"."""
    ordered = sorted(set(frames))
    if not ordered:
        return ""
    runs: list[list[int]] = [[ordered[0], ordered[0]]]
    for frame in ordered[1:]:
        if frame == runs[-1][1] + 1:
            runs[-1][1] = frame
        else:
            runs.append([frame, frame])
    return ",".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in runs)
