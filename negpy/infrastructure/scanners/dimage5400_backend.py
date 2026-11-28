# SPDX-License-Identifier: GPL-3.0-or-later
"""NegPy ``ScannerBackend`` adapter for the Minolta DiMAGE Scan Elite 5400 and 5400 II.

The driver returns film transmission (1.0 = clear) after its own shading and film-path
correction, so the adapter only maps ScanParams onto a holder slot and a window, and the
result onto uint16. Whichever model is on the bus is the one device it lists.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Any

import numpy as np

from negpy.infrastructure.scanners.base import (
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import FilmType, ScanMode, ScanParams, film_passes_infrared
from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_INSTALL_HINT = (
    "The dimage5400 driver is not installed. Install it with: uv sync --group dimage5400. On Linux "
    "the scanner needs a udev rule for 132b:0012 (5400 II) or 0686:400e (5400) (see docs/DIMAGE5400.md)."
)

DEVICE_ID = "dimage5400:usb"
_SUPPORTED_DPI = (675, 1350, 2700, 5400)
_DEFAULT_SLOTS = 6
_FRAME_PITCH_MM = 38.0


def _require_driver() -> None:
    try:
        import dimage5400.workflow  # noqa: F401
    except ImportError as exc:
        raise ScannerUnavailable(_INSTALL_HINT) from exc


def _capabilities(slots: int, model: Any) -> ScannerCapabilities:
    # The sensor line and the feed travel of one holder slot, at 5400 dpi.
    area_mm = (model.sensor_pixels / 5400 * 25.4, model.max_lines / 5400 * 25.4)
    return ScannerCapabilities(
        ir_channel=True,
        supported_dpi=_SUPPORTED_DPI,
        supported_depths=(8, 16),
        sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE),
        max_area_mm=area_mm,
        auto_exposure=True,
        autofocus=True,
        adapter_frame_capacity=slots,
        adapter_frame_control=True,
        can_eject=True,
        frame_pitch_mm=_FRAME_PITCH_MM,
        film_types=(FilmType.NEGATIVE.value, FilmType.MONO.value, FilmType.POSITIVE.value),
        grain_dissolver=model.grain_dissolver,
    )


def _nearest_dpi(dpi: int) -> int:
    return min(_SUPPORTED_DPI, key=lambda d: (abs(d - dpi), -d))


def _to_uint16(transmission: np.ndarray, depth: int) -> np.ndarray:
    from dimage5400.process import to_uint16

    out = to_uint16(transmission)
    if depth == 8:
        return (out >> 8).astype(np.uint8)
    return out


def _bridge(progress: Callable[..., None] | None) -> Callable[..., None]:
    def report(fraction: float, phase: str = "Scanning") -> None:
        if progress is None:
            return
        with suppress(Exception):
            progress(max(0.0, min(1.0, float(fraction))), phase)

    return report


class Dimage5400Session:
    """An exclusive hold on the scanner: one open USB handle, N scans, one release."""

    def __init__(self, backend: Dimage5400Backend, scanner: Any) -> None:
        self.device_id = DEVICE_ID
        self._backend = backend
        self._scanner = scanner
        self._closed = False

    def scan(self, params: ScanParams, progress: Callable[..., None], cancel: threading.Event) -> ScanResult:
        self._require_open()
        return self._backend._scan_with(self._scanner, params, progress, cancel)

    def eject(self) -> bool:
        self._require_open()
        with self._backend._mapped_errors():
            self._scanner.eject()
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            with suppress(Exception):
                self._scanner.close()
        finally:
            self._backend._release(self)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Scanner session is closed")

    def __enter__(self) -> Dimage5400Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Dimage5400Backend:
    """ScannerBackend for the DiMAGE Scan Elite 5400 and 5400 II over the dimage5400 driver."""

    def __init__(self) -> None:
        _require_driver()
        self._devices: list[ScannerDevice] | None = None
        self._session: Dimage5400Session | None = None
        self._lock = threading.Lock()

    # ── enumeration ───────────────────────────────────────────────────

    def list_devices(self) -> list[ScannerDevice]:
        if self._devices is None:
            self._devices = self._probe()
        return self._devices

    def refresh_devices(self) -> list[ScannerDevice]:
        self._devices = None
        return self.list_devices()

    def _probe(self) -> list[ScannerDevice]:
        from dimage5400.transport import find_device

        found = find_device()
        if found is None:
            return []
        model = found[1]
        slots = _DEFAULT_SLOTS
        if self._session is None:
            # Opening runs the start-up cycle when the slot is empty, which the unit needs
            # before it takes a holder; with a holder in, the slot count comes from it.
            try:
                with self._open() as scanner:
                    holder = scanner.holder()
                    if holder.loaded:
                        slots = holder.frames
            except Exception as exc:
                logger.warning("Could not probe the %s: %s", model.name, exc)
        return [
            ScannerDevice(
                id=DEVICE_ID,
                vendor=model.vendor,
                model=model.name,
                capabilities=_capabilities(slots, model),
            )
        ]

    @contextmanager
    def _open(self) -> Iterator[Any]:
        from dimage5400.scanner import Scanner

        with self._mapped_errors():
            scanner = Scanner.open()
        try:
            yield scanner
        finally:
            with suppress(Exception):
                scanner.close()

    # ── sessions ──────────────────────────────────────────────────────

    def open_session(self, device_id: str) -> ScannerSession:
        with self._lock:
            if self._session is not None:
                raise RuntimeError(f"Device already held in a session: {device_id}")
        from dimage5400.scanner import Scanner

        with self._mapped_errors():
            scanner = Scanner.open()
        session = Dimage5400Session(self, scanner)
        with self._lock:
            self._session = session
        return session

    def _release(self, session: Dimage5400Session) -> None:
        with self._lock:
            if self._session is session:
                self._session = None

    # ── scanning ──────────────────────────────────────────────────────

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[..., None],
        cancel: threading.Event,
    ) -> ScanResult:
        with self._lock:
            held = self._session
        if held is not None:
            return held.scan(params, progress, cancel)
        with self._open() as scanner:
            return self._scan_with(scanner, params, progress, cancel)

    def _scan_with(
        self,
        scanner: Any,
        params: ScanParams,
        progress: Callable[..., None],
        cancel: threading.Event,
    ) -> ScanResult:
        from dimage5400.scanner import Exposure
        from dimage5400.workflow import load_profile, scan_frame

        if cancel.is_set():
            raise RuntimeError("Scan cancelled")
        if params.capture_ir and not film_passes_infrared(params.film_type):
            raise RuntimeError("This film blocks infrared, so an IR pass has nothing to read on it")
        if params.n_passes > 1:
            raise RuntimeError("Multi-pass scanning is not supported on this scanner")
        if params.grain_dissolver and not scanner.model.grain_dissolver:
            raise RuntimeError(f"The {scanner.model.name} has no Grain Dissolver")
        exposure = None if params.auto_exposure else Exposure()
        dpi = _nearest_dpi(params.dpi)
        with self._mapped_errors():
            result = scan_frame(
                scanner,
                params.frame,
                dpi=dpi,
                window=params.window,
                feed_offset_mm=params.frame_offset_mm,
                autofocus=params.autofocus,
                exposure=exposure,
                profile=load_profile(scanner.model, params.grain_dissolver),
                grain_dissolver=params.grain_dissolver,
                progress=_bridge(progress),
                cancel=cancel,
            )
        ir = None
        mask = None
        if params.capture_ir and result.ir is not None:
            ir = _to_uint16(result.ir, params.depth)
            mask = np.ones(ir.shape, dtype=np.bool_)
        exp = result.exposure
        return ScanResult(
            rgb=_to_uint16(result.rgb, params.depth),
            ir=ir,
            dpi=dpi,
            device_model=scanner.model.name,
            ir_valid_mask=mask,
            exposures={"red": exp.r, "green": exp.g, "blue": exp.b, "ir": exp.ir},
        )

    # ── eject ─────────────────────────────────────────────────────────

    def eject(self, device_id: str) -> bool:
        with self._lock:
            held = self._session
        if held is not None:
            return held.eject()
        with self._open() as scanner, self._mapped_errors():
            scanner.eject()
        return True

    # ── errors ────────────────────────────────────────────────────────

    @contextmanager
    def _mapped_errors(self) -> Iterator[None]:
        """The driver's errors onto NegPy's, by type."""
        import usb.core
        from dimage5400.scanner import HolderPresent, HolderRequired, HolderWaitStopped
        from dimage5400.transport import DeviceBusy, ScannerError

        try:
            yield
        except HolderWaitStopped as exc:
            raise RuntimeError("Scan cancelled") from exc
        except InterruptedError as exc:
            # The driver ejects and recalibrates after a stop, so the holder can go straight back in.
            raise RuntimeError("Scan cancelled. The holder was ejected and the scanner recalibrated: insert it again.") from exc
        except HolderRequired as exc:
            raise RuntimeError("No film holder in the scanner. Insert one and try again.") from exc
        except HolderPresent as exc:
            raise RuntimeError(
                "The scanner must calibrate with the slot empty. Eject the holder, wait for the lamp to settle, then insert it again."
            ) from exc
        except DeviceBusy as exc:
            raise TransientScanError(str(exc)) from exc
        except usb.core.USBTimeoutError as exc:
            raise RuntimeError("The scanner stopped answering. Switch it off and on with the holder out, then try again.") from exc
        except usb.core.USBError as exc:
            raise TransientScanError(f"USB error: {exc}") from exc
        except ScannerError as exc:
            raise RuntimeError(str(exc)) from exc
