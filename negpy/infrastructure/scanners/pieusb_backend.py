from __future__ import annotations

from typing import TYPE_CHECKING


from negpy.infrastructure.scanners.base import (
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
)
from negpy.infrastructure.scanners.params import ScanParams, ScanMode
from negpy.infrastructure.scanners.result import ScanResult

if TYPE_CHECKING:
    from pieusb.types import DeviceInfo

from collections.abc import Callable

import threading


def _require_pieusb() -> None:
    try:
        # An actual import, not find_spec: a resolvable spec still fails to load
        # if pyusb or libusb_package's bundled library is missing or ABI-broken,
        # which is the failure this is here to catch.
        import pieusb  # noqa: F401
    except ImportError:
        raise ScannerUnavailable("pieusb not importable. Install: uv sync --group pieusb") from None


class PieusbSession:
    device_id: str


    def __init__(self, backend: PieusbBackend) -> None:
        raise NotImplementedError('PieusbSession not yet implemented')

    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        raise NotImplementedError("PieusbSession is a stub; use PieusbBackend.scan")

    def eject(self) -> bool:
        return False

    def close(self) -> None:
        self.dev.close()

    def __enter__(self) -> "ScannerSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class PieusbBackend:
    def __init__(self) -> None:
        _require_pieusb()

        self._devices_cache: list[ScannerDevice] | None = None
        self._devices_map: dict[str, DeviceInfo] = {}

    def list_devices(self) -> list[ScannerDevice]:
        if self._devices_cache is not None:
            return self._devices_cache

        return self.refresh_devices()

    def refresh_devices(self) -> list[ScannerDevice]:
        from pieusb import get_devices
        from pieusb.types import Filter

        self._devices_cache = []
        self._devices_map = {}
        devices = get_devices()
        for dev in devices:
            native_res = dev.inquiry.max_resolution_x
            max_w = dev.inquiry.max_scan_w / native_res * 25.4
            max_h = dev.inquiry.max_scan_h / native_res * 25.4
            supported_dpi = tuple([int(native_res / d) for d in [1, 2, 4, 5, 8, 10, 20]])
            supported_depths = tuple([d for d in [8, 16] if d in dev.inquiry.color_depths])
            caps = ScannerCapabilities(
                ir_channel=Filter.INFRARED in dev.inquiry.filters,
                supported_dpi=supported_dpi,
                supported_depths=supported_depths,
                sources=(ScanMode.POSITIVE,),
                max_area_mm=(max_w, max_h),
                auto_exposure=True,
                autofocus=False,
            )
            device_str = f"pieusb:{dev.dev.bus}:{dev.dev.address}"
            self._devices_map[device_str] = dev
            self._devices_cache.append(
                ScannerDevice(id=device_str, vendor=dev.inquiry.vendor, model=dev.inquiry.model_str, capabilities=caps)
            )

        return self._devices_cache

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        from pieusb.scanner import Scanner

        if cancel.is_set():
            raise Exception("Scan was cancelled")

        dev = self._devices_map[device_id]

        with Scanner(dev) as s:
            if params.capture_ir:
                s.mode = "rgbi"
            else:
                s.mode = "rgb"

            s.color_depth = params.depth
            s.resolution = params.dpi
            s.auto_exp = params.auto_exposure

            if params.window is not None:
                tl_x, tl_y, br_x, br_y = params.window
                tl_x *= dev.inquiry.max_scan_w
                br_x *= dev.inquiry.max_scan_w
                tl_y *= dev.inquiry.max_scan_h
                br_y *= dev.inquiry.max_scan_h
                s.tl_x = int(tl_x)
                s.tl_y = int(tl_y)
                s.br_x = int(br_x)
                s.br_y = int(br_y)

            result = None
            scan_error = None

            def on_update(update):
                progress(update.progress, update.phase)

            def on_complete(scan_result):
                nonlocal result, scan_error
                scan_error = scan_result.error
                result = ScanResult(
                    rgb=scan_result.rgb,
                    ir=scan_result.ir,
                    dpi=params.dpi,
                    device_model=dev.inquiry.model_str
                )
            
            s.scan(on_update, on_complete)

            while not s.wait(0.2):
                if cancel.is_set():
                    s.cancel()
                    raise Exception("Scan was cancelled")

            if result is None or result.rgb is None:
                raise Exception('Error while assembling the scan result')

            return result

    def open_session(self, device_id: str) -> ScannerSession:
        raise NotImplementedError("open_session not yet implemented in PieusbBackend")

    def eject(self, device_id: str) -> bool:
        return False
