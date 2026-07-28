from negpy.infrastructure.scanners.base import ScannerDevice, ScannerSession, ScannerUnavailable
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult

from collections.abc import Callable

import numpy
import threading

class PieusbSession:
    device_id: str

    def __init__(self) -> None:
        pass
    
    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult: ...
    def eject(self) -> bool:
        return False
    def close(self) -> None:
        pass
    def __enter__(self) -> "ScannerSession":
        return self
    def __exit__(self, *exc: object) -> None:
        pass

class PieusbBackend:
    def __init__(self) -> None:
        try:
            import pieusb
        except ImportError:
            raise ScannerUnavailable('Could not import module pieusb')
        self._pieusb = pieusb
        self._devices_cache: list[ScannerDevice] | None = None

    def list_devices(self) -> list[ScannerDevice]:
        return []
    
    def refresh_devices(self) -> list[ScannerDevice]:
        return []

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        return ScanResult(
            rgb=numpy.array([]),
            ir=numpy.array([]),
            dpi=0,
            device_model='Unknown'
        )
    def open_session(self, device_id: str) -> ScannerSession:
        return PieusbSession()
    def eject(self, device_id: str) -> bool:
        return False