from __future__ import annotations

from typing import TYPE_CHECKING


from negpy.infrastructure.scanners.base import (
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import ScanParams, ScanMode
from negpy.infrastructure.scanners.result import ScanResult

if TYPE_CHECKING:
    from pieusb.types import DeviceInfo

from collections.abc import Callable

import errno
import threading

# libusb failures worth another attempt: a fresh `with Scanner(dev)` re-claims the
# interface, which clears a stall or a busy handle. pyusb maps these from
# LIBUSB_ERROR_BUSY/TIMEOUT/PIPE (usb.backend.libusb1._libusb_errno).
_RETRYABLE_USB_ERRNOS = frozenset({errno.EBUSY, errno.ETIMEDOUT, errno.EPIPE})

# ...and the ones a retry cannot help, because _devices_map still holds the
# vanished device and every attempt would reuse a dead handle.
_GONE_USB_ERRNOS = frozenset({errno.ENODEV, errno.ENOENT})


def _require_pieusb() -> None:
    try:
        # An actual import, not find_spec: a resolvable spec still fails to load
        # if pyusb or libusb_package's bundled library is missing or ABI-broken,
        # which is the failure this is here to catch.
        import pieusb  # noqa: F401
    except ImportError:
        raise ScannerUnavailable("pieusb not importable. Install: uv sync --group pieusb") from None


def _as_scan_error(exc: BaseException, params: ScanParams) -> Exception:
    """Re-type a failed scan so ScannerService can decide on type alone.

    `TransientScanError` earns a retry (ScannerService gives it three attempts
    with a settle delay); anything else fails fast with a message the sidebar
    shows verbatim, so each one has to say what the user can do about it.

    pieusb reports failures by exception type, unlike SANE's message markers in
    `sane_backend._as_scan_error` — with one exception, noted below. Note that
    pyusb's USBError is NOT wrapped by pieusb's transport layer, so it arrives
    raw and must be handled here.
    """
    import usb.core
    from pieusb.exceptions import (
        CheckCondition,
        DeviceNotReady,
        PieusbError,
        Timeout,
        TransportError,
        WarmingUp,
    )

    # --- retryable ---------------------------------------------------------
    if isinstance(exc, (TransportError, Timeout)):
        # TransportError already reset the transport before raising, so the
        # device is deliberately left in a state a fresh open can use.
        return TransientScanError(f"Scanner transport failure: {exc}")
    if isinstance(exc, WarmingUp):
        # Retryable by nature, but pieusb has already waited out its own budget
        # (~150s at START SCAN), so each further attempt costs that again. The
        # WARMING_UP phase reaches the progress bar, so the wait is at least visible.
        return TransientScanError(f"Scanner lamp is still warming up: {exc}")
    if isinstance(exc, usb.core.USBError) and getattr(exc, "errno", None) in _RETRYABLE_USB_ERRNOS:
        return TransientScanError(f"USB error during the scan: {exc}")

    # --- fatal, most specific first ----------------------------------------
    if isinstance(exc, usb.core.USBError):
        if getattr(exc, "errno", None) in _GONE_USB_ERRNOS:
            return RuntimeError(f"Lost contact with the scanner mid-scan; refresh the device list and try again ({exc})")
        # No errno, or one not worth a claim either way: say what happened and
        # stop, rather than guessing at a cause the message cannot support.
        return RuntimeError(f"USB error during the scan: {exc}")
    if isinstance(exc, MemoryError):
        return RuntimeError(
            f"Not enough memory to hold a {params.dpi} dpi {params.depth}-bit scan"
            f"{' with infrared' if params.capture_ir else ''}. Lower the resolution, "
            f"scan 8-bit, or select a smaller area."
        )
    if isinstance(exc, DeviceNotReady):
        # WarmingUp, its one retryable subclass, was taken above.
        return RuntimeError(f"The scanner is not ready to scan — check the holder and the cover ({exc})")
    if isinstance(exc, CheckCondition):
        # str() carries the sense key/code/qualifier, which is the only thing that
        # makes an unrecognised refusal diagnosable after the fact.
        return RuntimeError(f"The scanner refused a command: {exc}")
    if isinstance(exc, PieusbError):
        # The one message match, because pieusb has no distinct type for it and it
        # is the likeliest failure with a user action attached. Worth an upstream
        # exception type; until then, matching beats showing raw geometry.
        if "empty image" in str(exc):
            return RuntimeError("The scanner reported an empty image — check that film is loaded and the area is in range")
        return RuntimeError(f"Scan failed: {exc}")
    return RuntimeError(f"Scan failed: {exc}")


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

        self._devices_cache = None
        devices = get_devices()
        self._devices_map = {}
        self._devices_cache = []
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
                exposure_time_us=(4100, 65518), # Empirical findings, not really matching reality, either
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
            if not params.auto_exposure and params.exposure_time_us is not None:
                rel = int(100 * params.exposure_time_us / 4100)
                s.exp_rel_r = rel
                s.exp_rel_g = rel
                s.exp_rel_b = rel

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
            scan_cancelled = False

            def on_update(update):
                progress(update.progress, update.phase)

            def on_complete(scan_result):
                nonlocal result, scan_error, scan_cancelled
                scan_error = scan_result.error
                scan_cancelled = scan_result.cancelled
                result = ScanResult(
                    rgb=scan_result.rgb,
                    ir=scan_result.ir,
                    dpi=params.dpi,
                    device_model=dev.inquiry.model_str
                )

            s.scan(on_update, on_complete)

            done = False
            while not done:
                if cancel.is_set():
                    s.cancel()
                    raise Exception("Scan was cancelled")
                done = s.wait(0.2)

            # A pieusb scan reports exactly one outcome, and the three below are
            # not interchangeable: only `error` may be retried, and a cancelled
            # scan is not a failure at all.
            if scan_cancelled:
                # The worker noticed the cancel at a chunk boundary and stopped the
                # device itself, so the poll above saw it finish rather than cancel.
                raise Exception("Scan was cancelled")
            if scan_error is not None:
                raise _as_scan_error(scan_error, params) from scan_error
            if result is None:
                # on_complete never ran: the worker thread died on something that
                # is not an Exception, so the device state is unknown, not failed.
                raise RuntimeError("The scan worker did not report an outcome")
            if result.rgb is None:
                raise RuntimeError("The scan completed but returned no image data")

            return result

    def open_session(self, device_id: str) -> ScannerSession:
        raise NotImplementedError("open_session not yet implemented in PieusbBackend")

    def eject(self, device_id: str) -> bool:
        return False
