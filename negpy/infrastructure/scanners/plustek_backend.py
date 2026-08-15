# SPDX-License-Identifier: GPL-3.0-or-later
"""NegPy ``ScannerBackend`` adapter for the pyopticfilm Plustek USB driver."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from negpy.infrastructure.scanners.base import (
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from pyopticfilm.device.select import model_for_device, model_is_scan_ready
from pyopticfilm.exceptions import (
    DeviceNotFoundError,
    DriverBindingError,
    PlustekError,
    ScanCancelled,
    UsbError,
)
from pyopticfilm.logging import get_logger
from pyopticfilm.scanner import Scanner
from pyopticfilm.usb.device import UsbDeviceInfo, find_devices, list_devices
from negpy.infrastructure.scanners.result import ScanResult

logger = get_logger(__name__)


def _caps_for(model: Any) -> ScannerCapabilities:
    prescan_ready = bool(getattr(model, "scan_ready", False))
    from pyopticfilm.scan.bringup import PRESCAN_DPI, default_frame_crop_norm

    return ScannerCapabilities(
        ir_channel=model.supports_infrared,
        supported_dpi=tuple(sorted(model.resolutions_dpi)),
        supported_depths=model.bpp_color,
        sources=(ScanMode.TRANSPARENCY,),
        max_area_mm=model.max_area_mm,
        auto_exposure=False,
        autofocus=False,
        prescan=prescan_ready,
        prescan_dpi=PRESCAN_DPI if prescan_ready else 0,
        prescan_mirror_x=bool(getattr(model, "mirror_x", False)) if prescan_ready else False,
        prescan_default_crop=default_frame_crop_norm(model) if prescan_ready else None,
        adapter_frame_capacity=None,
        adapter_frame_control=False,
        can_eject=False,
        frame_pitch_mm=0.0,
        exposure_time_us=None,
    )


def _to_scanner_device(info: UsbDeviceInfo) -> ScannerDevice:
    model = model_for_device(info.product_id, getattr(info, "bcd_device", 0))
    return ScannerDevice(
        id=info.device_id,
        vendor=model.vendor,
        model=model.model,
        capabilities=_caps_for(model),
    )


def _safe_progress(
    progress: Callable[[float, str], None] | None,
    value: float,
    phase: str = "Scanning",
) -> None:
    if progress is None:
        return
    with suppress(Exception):
        progress(max(0.0, min(1.0, float(value))), phase)


def _validate_params(params: ScanParams, *, model: Any | None = None) -> None:
    from pyopticfilm.device.model_8200i import MODEL_8200I

    dpi = int(params.dpi)
    depth = int(params.depth)
    resolutions = getattr(model, "resolutions_dpi", None) if model is not None else None
    bpp = getattr(model, "bpp_color", None) if model is not None else None
    if not isinstance(resolutions, (tuple, list)) or not resolutions:
        resolutions = MODEL_8200I.resolutions_dpi
    if not isinstance(bpp, (tuple, list)) or not bpp:
        bpp = MODEL_8200I.bpp_color
    if dpi not in resolutions:
        raise RuntimeError(f"Unsupported dpi={dpi}; supported={sorted(resolutions)}")
    if depth not in bpp:
        raise RuntimeError(f"Unsupported depth={depth}; supported={sorted(bpp)}")

    if params.frame is not None:
        raise RuntimeError(f"Frame {params.frame} requested but the device has no frame-selection option")
    if params.auto_exposure:
        raise RuntimeError("Auto-exposure requested but the device has no 'ae' option")
    if params.autofocus:
        raise RuntimeError("Autofocus requested but the device has no autofocus option")
    if params.capture_ir and model is not None and getattr(model, "supports_infrared", None) is False:
        raise RuntimeError(f"{getattr(model, 'model', 'device')} does not support infrared")


class PlustekSession:
    """Exclusive device hold for batch workflows."""

    def __init__(
        self,
        backend: PlustekBackend,
        device_id: str,
        scanner: Scanner,
    ) -> None:
        self.device_id = device_id
        self._backend = backend
        self._scanner = scanner
        self._closed = False

    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        if self._closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        return self._backend._scan_on_scanner(self._scanner, params, progress, cancel)

    def eject(self) -> bool:
        if self._closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._scanner.close()
        finally:
            self._backend._release_session(self)

    def __enter__(self) -> PlustekSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class PlustekBackend:
    """ScannerBackend for OpticFilm 8200i SE."""

    def __init__(self, *, calib_cache: Path | None = None) -> None:
        try:
            import usb.core  # noqa: F401
        except ImportError as exc:
            raise ScannerUnavailable(
                "Plustek USB needs pyopticfilm (uv sync --group plustek). "
                "On Windows, bind WinUSB with Zadig for OpticFilm 8200i SE (07b3:1825) — "
                "see docs/PLUSTEK_WINDOWS.md."
            ) from exc
        self._calib_cache = calib_cache
        self._sessions: dict[str, PlustekSession] = {}

    def list_devices(self) -> list[ScannerDevice]:
        try:
            devices = [_to_scanner_device(d) for d in find_devices(supported_only=True)]
        except DeviceNotFoundError:
            return []
        except DriverBindingError as exc:
            raise ScannerUnavailable(str(exc)) from exc
        return [d for d in devices if d.capabilities.sources]

    def refresh_devices(self) -> list[ScannerDevice]:
        return self.list_devices()

    def open_session(self, device_id: str) -> ScannerSession:
        if device_id in self._sessions:
            raise RuntimeError(f"Device already held in a session: {device_id}")
        self._ensure_known_device(device_id)
        try:
            scanner = Scanner.open(device_id, calib_cache=self._calib_cache)
            if not scanner.asic._initialized:
                scanner.asic.init()
            if not scanner.asic.is_at_home():
                scanner.home()
        except DriverBindingError as exc:
            raise ScannerUnavailable(str(exc)) from exc
        except UsbError as exc:
            raise TransientScanError(str(exc)) from exc
        except DeviceNotFoundError as exc:
            raise RuntimeError(f"Unknown or disconnected device: {device_id}") from exc

        session = PlustekSession(self, device_id, scanner)
        self._sessions[device_id] = session
        return session

    def _release_session(self, session: PlustekSession) -> None:
        self._sessions.pop(session.device_id, None)

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        if device_id in self._sessions:
            raise RuntimeError(f"Device {device_id} is held by an open session; use session.scan()")
        self._ensure_known_device(device_id)
        try:
            with Scanner.open(device_id, calib_cache=self._calib_cache) as scanner:
                if not scanner.asic._initialized:
                    scanner.asic.init()
                if not scanner.asic.is_at_home():
                    scanner.home()
                return self._scan_on_scanner(scanner, params, progress, cancel)
        except TransientScanError:
            raise
        except ScannerUnavailable:
            raise
        except DriverBindingError as exc:
            raise TransientScanError(str(exc)) from exc
        except UsbError as exc:
            raise TransientScanError(str(exc)) from exc
        except DeviceNotFoundError as exc:
            raise RuntimeError(f"Unknown or disconnected device: {device_id}") from exc

    def eject(self, device_id: str) -> bool:
        del device_id
        return False

    def _ensure_known_device(self, device_id: str) -> None:
        ids = {d.device_id for d in find_devices(supported_only=True)}
        if device_id in ids:
            return
        known = {d.device_id: d for d in list_devices()}
        if device_id in known:
            raise RuntimeError(f"{device_id} is not a supported Plustek film scanner (need 07b3:130d or 07b3:1825).")
        raise RuntimeError(f"Unknown or disconnected device: {device_id}")

    def _scan_on_scanner(
        self,
        scanner: Scanner,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        if not model_is_scan_ready(scanner.model):
            msg = (
                f"{scanner.model.model} ({scanner.model.asic}) cannot scan with "
                "pyOpticfilm in this release — only OpticFilm 8200i SE is validated."
            )
            if sys.platform != "win32":
                msg += " Try Backend → SANE if that backend lists this scanner."
            raise RuntimeError(msg)
        _validate_params(params, model=scanner.model)
        dpi = int(params.dpi)
        capture_ir = bool(params.capture_ir)
        window = params.window
        geometry = self._default_scan_geometry(scanner, dpi=dpi, window=window)

        _safe_progress(progress, 0.0)
        if cancel.is_set():
            raise RuntimeError("Scan cancelled before start")

        try:
            self._ensure_colour_calib(
                scanner,
                dpi=dpi,
                window=window,
                geometry=geometry,
                progress=progress,
                cancel=cancel,
            )
            if capture_ir:
                rgb_image, ir_plane = self._scan_color_and_ir(
                    scanner,
                    dpi=dpi,
                    window=window,
                    geometry=geometry,
                    progress=progress,
                    cancel=cancel,
                    progress_start=0.1,
                )
            else:

                def color_only_progress(p: float) -> None:
                    _safe_progress(progress, 0.1 + 0.9 * p)

                rgb_image = scanner.scan(
                    resolution=dpi,
                    mode="color",
                    area=None if geometry is not None else window,
                    geometry=geometry,
                    progress=color_only_progress,
                    cancel=cancel,
                )
                ir_plane = None
        except ScanCancelled as exc:
            raise RuntimeError("Scan cancelled") from exc
        except UsbError as exc:
            raise TransientScanError(str(exc)) from exc
        except PlustekError as exc:
            raise RuntimeError(str(exc)) from exc

        _safe_progress(progress, 1.0)
        return ScanResult(
            rgb=np.asarray(rgb_image.rgb),
            ir=ir_plane,
            dpi=dpi,
            device_model=rgb_image.device_model or f"{scanner.model.vendor} {scanner.model.model}",
            ir_valid_mask=None,
        )

    @staticmethod
    def _ensure_colour_calib(
        scanner: Scanner,
        *,
        dpi: int,
        window: tuple[float, float, float, float] | None,
        geometry: object | None,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> None:
        """Home ASIC shading before the colour feed (apply cache or measure).

        Progress uses 0–0.1 so a cold measure does not look hung. Film may stay
        loaded — shading runs at home before position feeds, like SilverFast.
        """
        if cancel.is_set():
            raise RuntimeError("Scan cancelled before start")
        _safe_progress(progress, 0.02, "Calibrating")
        from pyopticfilm.exceptions import CalibrationError
        from pyopticfilm.scan.geometry import ScanGeometry

        geo = geometry if isinstance(geometry, ScanGeometry) else None
        if geo is None and window is not None:
            from pyopticfilm.scan.geometry import compute_geometry

            geo = compute_geometry(dpi, model=scanner.model, area=window)
        if geo is None:
            from pyopticfilm.scan.bringup import (
                bringup_scan_geometry,
                is_opticfilm_8200i_se,
            )

            if is_opticfilm_8200i_se(scanner.model):
                geo, _ = bringup_scan_geometry(scanner.model, dpi, profile="preview_safe")
        if geo is None:
            _safe_progress(progress, 0.1, "Calibrating")
            return

        asic = scanner.asic
        calibrator = scanner.calibrator
        hit = calibrator.find_for_scan(method="transparency", geometry=geo)
        if hit is not None and hit.has_asic_blob and getattr(asic, "asic_shading_ready", False):
            _safe_progress(progress, 0.1, "Calibrating")
            return

        logger.info("Ensuring colour ASIC shading for dpi=%d", dpi)
        try:
            # Prefer ensure (apply or measure); Scanner.calibrate also parks motor.
            was_armed = scanner._bringup_motor_armed
            try:
                if getattr(scanner.model, "asic", "") == "GL128":
                    scanner.disarm_bringup_motor()
                calibrator.ensure_colour_asic_shading(geo)
            finally:
                if was_armed:
                    scanner.arm_bringup_motor()
        except CalibrationError as exc:
            raise RuntimeError(str(exc)) from exc
        if cancel.is_set():
            raise RuntimeError("Scan cancelled")
        _safe_progress(progress, 0.1, "Calibrating")

    @staticmethod
    def _default_scan_geometry(
        scanner: Scanner,
        *,
        dpi: int,
        window: tuple[float, float, float, float] | None,
    ) -> object | None:
        """Lab Full-window or crop geometry for SE (forced geometry into scan)."""
        from pyopticfilm.scan.bringup import (
            bringup_scan_geometry,
            crop_scan_geometry,
            is_opticfilm_8200i_se,
        )

        if not is_opticfilm_8200i_se(scanner.model):
            return None
        if window is not None:
            geometry, _meta = crop_scan_geometry(scanner.model, dpi, window)
            return geometry
        geometry, _meta = bringup_scan_geometry(scanner.model, dpi, profile="preview_safe")
        return geometry

    def _scan_color_and_ir(
        self,
        scanner: Scanner,
        *,
        dpi: int,
        window: tuple[float, float, float, float] | None,
        geometry: object | None,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
        progress_start: float = 0.0,
    ) -> tuple[Any, np.ndarray]:
        span = max(0.0, 1.0 - progress_start)

        def color_progress(p: float) -> None:
            _safe_progress(progress, progress_start + 0.5 * span * p)

        def ir_progress(p: float) -> None:
            _safe_progress(progress, progress_start + 0.5 * span * (1.0 + p), "Infrared")

        if cancel.is_set():
            raise RuntimeError("Scan cancelled before start")

        scan_area = None if geometry is not None else window
        color = scanner.scan(
            resolution=dpi,
            mode="color",
            area=scan_area,
            geometry=geometry,
            progress=color_progress,
            cancel=cancel,
        )
        if cancel.is_set():
            raise RuntimeError("Scan cancelled")

        ir_img = scanner.scan(
            resolution=dpi,
            mode="infrared",
            area=scan_area,
            geometry=geometry,
            progress=ir_progress,
            cancel=cancel,
        )
        ir_plane = ir_img.ir if ir_img.ir is not None else ir_img.rgb[:, :, 1].copy()
        from pyopticfilm.ir_align import align_ir_to_rgb

        ir_plane = align_ir_to_rgb(np.asarray(color.rgb), np.asarray(ir_plane))
        return color, ir_plane
