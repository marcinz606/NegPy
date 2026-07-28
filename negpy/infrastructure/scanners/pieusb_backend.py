from negpy.infrastructure.scanners.base import (
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    ScannerCapabilities
)
from negpy.infrastructure.scanners.params import ScanParams, ScanMode
from negpy.infrastructure.scanners.result import ScanResult

from pieusb.types import DeviceInfo
from pieusb.scanner import Scanner

from collections.abc import Callable

import numpy
import threading

class PieusbSession:
    device_id: str

    def __init__(self, info: DeviceInfo) -> None:
        self.dev = Scanner(info)
    
    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult: ...
    def eject(self) -> bool:
        return False
    def close(self) -> None:
        self.dev.dev.close()
    def __enter__(self) -> "ScannerSession":
        return self
    def __exit__(self, *exc: object) -> None:
        self.close()

class PieusbBackend:
    def __init__(self) -> None:
        try:
            import pieusb
        except ImportError:
            raise ScannerUnavailable('Could not import module pieusb')
        self._pieusb = pieusb
        self._devices_cache: list[ScannerDevice] | None = None
        self._devices_map: dict[str, DeviceInfo] = {}

    def list_devices(self) -> list[ScannerDevice]:
        if self._devices_cache is not None:
            return self._devices_cache

        return self.refresh_devices()
    
    def refresh_devices(self) -> list[ScannerDevice]:
        self._devices_cache = []
        self._devices_map = {}
        devices = self._pieusb.get_devices()
        for dev in devices:
            native_res = dev.inquiry.max_resolution_x
            max_w = dev.inquiry.max_scan_w / native_res * 25.4
            max_h = dev.inquiry.max_scan_h / native_res * 25.4
            caps = ScannerCapabilities(
                ir_channel=self._pieusb.types.Filter.INFRARED in dev.inquiry.filters,
                supported_dpi=(300, 500, 1000, 2500, 5000, 10000),
                supported_depths=(8, 16),
                sources=(ScanMode.POSITIVE),
                max_area_mm=(max_w, max_h),
                auto_exposure=True
            )
            device_str = f'pieusb:{dev.dev.bus}:{dev.dev.address}'
            self._devices_map[device_str] = dev.dev
            self._devices_cache.append(ScannerDevice(
                id='something',
                vendor=dev.inquiry.vendor,
                model=dev.inquiry.model_str,
                capabilities=caps
            ))

        return self._devices_cache


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
        return PieusbSession(self._devices_map[device_id])

    def eject(self, device_id: str) -> bool:
        return False