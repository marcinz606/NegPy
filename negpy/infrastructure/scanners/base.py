import threading
from dataclasses import dataclass
from typing import Callable, Protocol

from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from negpy.infrastructure.scanners.result import ScanResult


class ScannerUnavailable(RuntimeError):
    """The backend's driver or library is not installed.

    The message is shown to the user verbatim, so it must carry an install hint.
    """


class TransientScanError(RuntimeError):
    """A transport glitch worth retrying — a USB link hiccup, a busy device.

    ScannerService retries only this type. A real error (bad option, missing frame,
    no film) must be a plain exception so it fails fast.
    """


@dataclass(frozen=True)
class ScannerCapabilities:
    ir_channel: bool
    supported_dpi: tuple[int, ...]
    supported_depths: tuple[int, ...]
    sources: tuple[ScanMode, ...]
    max_area_mm: tuple[float, float]  # (width, height)
    auto_exposure: bool = False
    adapter_frame_capacity: int | None = None  # transport capacity bound, not an exposure count
    adapter_frame_control: bool = False
    can_eject: bool = False
    frame_pitch_mm: float = 0.0  # feed-axis distance between frame positions; 0.0 = unknown


@dataclass(frozen=True)
class ScannerDevice:
    id: str  # backend-native device address, e.g. "plustek:libusb:001:008"
    vendor: str
    model: str
    capabilities: ScannerCapabilities


class ScannerSession(Protocol):
    """An exclusive hold on one device: opened once, N scans, released once.

    The handover seam for batch/roll workflows that must own the transport for a
    whole strip. Both close() and eject() are terminal and idempotent.
    """

    device_id: str

    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult: ...
    def eject(self) -> bool: ...
    def close(self) -> None: ...
    def __enter__(self) -> "ScannerSession": ...
    def __exit__(self, *exc: object) -> None: ...


class ScannerBackend(Protocol):
    """One scanner transport. Implementations live outside NegPy where the device warrants it.

    Obligations beyond the signatures:

    - `list_devices` drops devices whose `capabilities.sources` is empty; a film
      scanner with no selectable source must still populate it or it never appears.
    - `scan` raises `TransientScanError` for retryable transport failures and a plain
      exception for everything else — that choice is the backend's alone.
    - `eject` returns False for a device with no eject action; it raises only when a
      present eject genuinely fails.
    - The constructor raises `ScannerUnavailable` when the driver is missing, with an
      install hint in the message.
    """

    def list_devices(self) -> list[ScannerDevice]: ...
    def refresh_devices(self) -> list[ScannerDevice]:
        """Re-enumerate, bypassing any cache. `return self.list_devices()` is a valid answer."""
        ...

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult: ...
    def open_session(self, device_id: str) -> ScannerSession: ...
    def eject(self, device_id: str) -> bool: ...
