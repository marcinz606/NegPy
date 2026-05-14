import threading
from dataclasses import dataclass
from typing import Callable, Protocol

from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from negpy.infrastructure.scanners.result import ScanResult


@dataclass(frozen=True)
class ScannerCapabilities:
    ir_channel: bool
    supported_dpi: tuple[int, ...]
    supported_depths: tuple[int, ...]
    sources: tuple[ScanMode, ...]
    max_area_mm: tuple[float, float]  # (width, height)


@dataclass(frozen=True)
class ScannerDevice:
    id: str  # SANE device name e.g. "plustek:libusb:001:008"
    vendor: str
    model: str
    capabilities: ScannerCapabilities


class ScannerBackend(Protocol):
    def list_devices(self) -> list[ScannerDevice]: ...
    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult: ...
