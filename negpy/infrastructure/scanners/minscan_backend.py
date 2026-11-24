# SPDX-License-Identifier: GPL-3.0-or-later
"""NegPy adapter for the Minolta DiMAGE ``minscan`` package.

The C ABI, shared-library loading and frame buffers live in the ``minscan``
Python package; this module only maps NegPy's ``ScanParams`` onto that API and
turns its arrays into ``ScanResult``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
import minscan

from negpy.infrastructure.scanners.base import (
    ScanMode,
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import MultiExposureMode, ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.scanners.roll import RollPreview

_INSTALL_HINT = (
    "minscan is not installed. Install the Python package with `uv sync --group minscan`, "
    "or `uv pip install -e ~/source/minscan`."
)

_MODEL = "Minolta DiMAGE Scan Multi PRO"
_DEVICE_PREFIX = "minscan:"

# MsMedia / MsFilm in the minscan package
_MEDIA_35MM = int(minscan.Media.FORMAT_35MM)
_MEDIA_120 = int(minscan.Media.FORMAT_120)
_FILM_NEG = int(minscan.Film.COLOR_NEG)
_FILM_POS = int(minscan.Film.COLOR_POS)
_FILM_MONO = int(minscan.Film.MONO)

_PHASES = {
    int(minscan.Phase.CALIBRATE): "Calibrating",
    int(minscan.Phase.FOCUS): "Focusing",
    int(minscan.Phase.SCAN): "Scanning",
}

_FORMAT_MAP = {
    "135": "35mm",
    "35": "35mm",
    "35mm": "35mm",
    "645": "6x45",
    "6x4.5": "6x45",
    "6x45": "6x45",
    "66": "6x6",
    "6x6": "6x6",
    "67": "6x7",
    "6x7": "6x7",
    "68": "6x8",
    "6x8": "6x8",
    "69": "6x9",
    "6x9": "6x9",
}

_FORMAT_CODE = {
    "35mm": 1,
    "6x45": 2,
    "6x6": 4,
    "6x7": 8,
    "6x8": 0x10,
    "6x9": 0x20,
}

_FILM_MAP = {
    "negative": _FILM_NEG,
    "positive": _FILM_POS,
    "mono": _FILM_MONO,
    "kodachrome": _FILM_POS,
}

_DPI_35 = (4800, 2400, 1200, 600, 300)
_DPI_120 = (3200, 1600, 800, 400, 200)
_DPI_ALL = (4800, 3200, 2400, 1600, 1200, 800, 600, 400, 300, 200)
_FORMATS_35 = ("135",)
_FORMATS_120 = ("645", "66", "67", "68", "69")
_FORMATS_ALL = ("135", *_FORMATS_120)


def _crop_result(result: ScanResult, window: tuple[float, float, float, float] | None) -> ScanResult:
    """Apply a normalized scan window to the returned raster.

    The minscan descriptor has a native crop path, but it is not wired yet; cropping
    here makes the Preview dialog's window do something useful without changing the
    acquisition path. Preview scans always keep the full frame so the overlay has
    room to move.
    """
    if not window or len(window) != 4:
        return result
    x1, y1, x2, y2 = (float(v) for v in window)
    h, w = result.rgb.shape[:2]
    left = max(0, min(w - 1, int(round(x1 * w))))
    right = max(left + 1, min(w, int(round(x2 * w))))
    top = max(0, min(h - 1, int(round(y1 * h))))
    bottom = max(top + 1, min(h, int(round(y2 * h))))
    rgb = result.rgb[top:bottom, left:right]
    ir = result.ir[top:bottom, left:right] if result.ir is not None else None
    return replace(result, rgb=rgb, ir=ir)


def _capabilities(media: int | None) -> ScannerCapabilities:
    if media == _MEDIA_35MM:
        dpi = _DPI_35
        area = (24.0, 36.0)
        capacity = 6
        formats = _FORMATS_35
        glassless = False
    elif media == _MEDIA_120:
        dpi = _DPI_120
        area = (56.0, 84.0)
        capacity = None
        formats = _FORMATS_120
        glassless = True
    else:
        dpi = _DPI_ALL
        area = (56.0, 84.0)
        capacity = None
        formats = _FORMATS_ALL
        # The carrier is unknown until a session opens; allow the 120 insert
        # choice, and let the format combo hide it for 35 mm.
        glassless = True

    return ScannerCapabilities(
        ir_channel=True,
        supported_dpi=dpi,
        supported_depths=(16,),
        sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE),
        max_area_mm=area,
        dpi_fixed=True,
        auto_exposure=False,
        autofocus=True,
        prescan=False,
        multi_exposure=False,
        max_n_passes=1,
        adapter_frame_capacity=capacity,
        can_eject=True,
        frame_pitch_mm=38.0 if media == _MEDIA_35MM else 0.0,
        film_formats=formats,
        film_types=("negative", "positive", "mono", "kodachrome"),
        hw_clean=False,
        roll_discovery=False,
        glassless_option=glassless,
    )


def _format_for(params: ScanParams, default_media: int | None = None) -> str:
    raw = params.film_format
    if raw is None:
        raw = "135" if default_media == _MEDIA_35MM else "69"
    raw = raw.strip().lower()
    if raw not in _FORMAT_MAP:
        raise RuntimeError(f"Unsupported film format {params.film_format!r} for minscan")
    return _FORMAT_MAP[raw]


def _carrier_for(
    params: ScanParams, ev: float, default_media: int | None = None
) -> tuple[tuple[int, int, int, int, float], str]:
    size = _format_for(params, default_media)
    media = _MEDIA_35MM if size == "35mm" else _MEDIA_120
    film = _FILM_MAP.get(params.film_type)
    if film is None:
        raise RuntimeError(f"Unsupported film type {params.film_type!r} for minscan")
    code = _FORMAT_CODE.get(size)
    if code is None:
        raise RuntimeError(f"Unsupported minscan format {size!r}")
    return (media, film, code, 0, ev), size


def _frame_group_slot(size: str, frame: int | None) -> tuple[int, int]:
    if size == "35mm":
        if frame is None:
            return 1, 0
        if not 1 <= frame <= 6:
            raise RuntimeError(f"35mm frame must be 1..6, not {frame}")
        # same mapping as minscan's parseFrame: even frames are A (slot 0), odd are B (slot 1)
        return (frame + 1) // 2, 0 if frame % 2 == 0 else 1
    if frame not in (None, 1):
        raise RuntimeError("120 carrier has one frame")
    return 1, 0


def _validate(params: ScanParams, default_media: int | None = None) -> None:
    if params.depth != 16:
        raise RuntimeError("minscan always returns 16-bit data; request depth=16")
    if params.glassless and _format_for(params, default_media) == "35mm":
        raise RuntimeError("glassless is only available on the 120 carrier")
    if params.n_passes > 1:
        raise RuntimeError("minscan does not support multi-pass stacking")
    if params.samples != 1:
        raise RuntimeError("minscan does not support per-line multi-sampling")
    if params.superfine:
        raise RuntimeError("minscan does not support superfine mode")
    if params.multi_exposure_mode != MultiExposureMode.OFF:
        raise RuntimeError("minscan does not support multi-exposure")
    if params.exposure_time_us is not None:
        raise RuntimeError("minscan does not expose a fixed exposure time")
    if any(ev != 0.0 for ev in params.channel_gain_ev):
        raise RuntimeError("minscan does not support per-channel analog gain")


class MinscanSession:
    """Open device session: one minscan transport, one calibration, N frames."""

    def __init__(self, device_id: str, default_media: int | None = None) -> None:
        self.device_id = device_id
        self._default_media = default_media
        self._closed = False
        self._carrier: tuple[int, int, int, int, float] | None = None
        self._identity: tuple[str, int, int, int, int, float] | None = None
        self._size: str | None = None

        path = device_id.split(":", 1)[1] if device_id.startswith(_DEVICE_PREFIX) else device_id
        try:
            self._session = minscan.Session(None if path in ("", "auto") else path)
        except minscan.MinscanUnavailable as exc:
            raise ScannerUnavailable(_INSTALL_HINT) from exc
        except minscan.MinscanError as exc:
            raise RuntimeError(f"minscan open {path or 'auto'}: {exc}") from exc

        resolved = getattr(self._session, "device", "")
        if resolved:
            self.device_id = f"{_DEVICE_PREFIX}{resolved}"

    # ── configuration ────────────────────────────────────────────────
    def _configure(self, params: ScanParams) -> None:
        carrier, size = _carrier_for(params, 0.0, self._default_media)
        identity = (size, *carrier)

        if self._identity == identity:
            return

        try:
            self._session.configure(
                minscan.Media(carrier[0]),
                minscan.Film(carrier[1]),
                format_code=carrier[2],
                glassless=bool(carrier[3]),
                ev=carrier[4],
            )
        except minscan.MinscanError as exc:
            raise RuntimeError(f"minscan configure: {exc}") from exc

        self._carrier = carrier
        self._identity = identity
        self._size = size

    # ── scan ─────────────────────────────────────────────────────────
    def scan(
        self,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        return self._acquire(params, prescan=False, progress=progress, cancel=cancel)

    def preview(
        self,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        """One hardware prescan. For 35 mm this is what the index scan runs once per frame."""
        return self._acquire(params, prescan=True, progress=progress, cancel=cancel)

    def _acquire(
        self,
        params: ScanParams,
        *,
        prescan: bool,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        if self._closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        if cancel.is_set():
            raise RuntimeError("Scan cancelled")
        _validate(params, self._default_media)
        self._configure(params)
        assert self._size is not None

        group, slot = _frame_group_slot(self._size, params.frame)

        def report(fraction: float, phase: minscan.Phase) -> None:
            label = _PHASES.get(int(phase), "Scanning")
            try:
                progress(fraction, label)
            except Exception:
                pass

        try:
            frame = self._session.scan(
                dpi=int(params.dpi),
                frame_group=group,
                slot=slot,
                autofocus=params.autofocus,
                capture_ir=params.capture_ir or params.clean,
                ice=params.clean,
                prescan=prescan,
                glassless=params.glassless,
                ev=params.analog_gain_ev,
                progress=report,
                cancel=cancel,
            )
        except minscan.MinscanCancelled as exc:
            raise RuntimeError("Scan cancelled") from exc
        except minscan.MinscanTransientError as exc:
            raise TransientScanError(str(exc)) from exc
        except minscan.MinscanError as exc:
            raise RuntimeError(str(exc)) from exc

        ir = frame.ir if params.capture_ir else None
        result = ScanResult(rgb=frame.rgb, ir=ir, dpi=frame.dpi, device_model=_MODEL)
        return result if prescan else _crop_result(result, params.window)

    # ── lifecycle ────────────────────────────────────────────────────
    def eject(self) -> bool:
        if self._closed:
            return False
        try:
            return self._session.eject()
        except minscan.MinscanError as exc:
            raise RuntimeError(f"minscan eject: {exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.close()

    def __enter__(self) -> MinscanSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MinscanRollSession:
    """Whole-strip preview over one minscan session.

    The scanner has a real index/prescan job, so previewing is not a low-DPI full
    scan: 35 mm walks the three two-frame groups and reads both slots as the
    vendor index scan does, while 120 has one frame. The session stays open
    across slots, so calibration and the lens stage are paid once.
    """

    def __init__(
        self,
        session: MinscanSession,
        device: ScannerDevice,
        *,
        dpi: int,
        film_format: str | None,
        film_type: str,
    ) -> None:
        self._session = session
        self._device = device
        self._dpi = dpi
        self._film_format = film_format
        self._film_type = film_type
        self.slot_count = max(1, device.capabilities.adapter_frame_capacity or 1)
        # The carrier addresses whole frames; it cannot trim a boundary in software.
        self.offset_range = (0.0, 0.0)
        self.supports_single_slot_preview = True

    def preview(
        self,
        slots: Iterable[int],
        *,
        cancel: threading.Event,
        progress: Callable[..., None] | None = None,
    ) -> Iterator[RollPreview]:
        report = progress or (lambda *_args, **_kwargs: None)
        for slot in slots:
            if cancel.is_set():
                return
            params = ScanParams(
                dpi=self._dpi,
                depth=16,
                capture_ir=False,
                autofocus=False,
                auto_exposure=False,
                frame=slot,
                film_format=self._film_format,
                film_type=self._film_type,
            )
            try:
                result = self._session.preview(params, report, cancel)
            except Exception as error:
                if cancel.is_set():
                    return
                yield RollPreview(slot=slot, error=str(error))
                continue
            yield RollPreview(slot=slot, rgb=result.rgb)

    def set_offset(self, slot: int, offset: float) -> None:
        del slot, offset  # the feeder addresses frames, it cannot back up or trim

    def approve(self, slot: int) -> None:
        del slot  # boundaries are the holder's, not inferred

    def close(self) -> None:
        self._session.close()


class MinscanBackend:
    """ScannerBackend over the ``minscan`` package (CFFI ABI)."""

    def __init__(self) -> None:
        try:
            minscan.load()
        except minscan.MinscanUnavailable as exc:
            raise ScannerUnavailable(_INSTALL_HINT) from exc
        self._media_by_device: dict[str, int | None] = {}

    def _probe(self, path: str) -> int | None:
        """None when the device did not answer; -1 when it answered without a carrier."""
        try:
            media = minscan.probe(None if path in ("", "auto") else path)
        except minscan.MinscanError:
            return None
        return -1 if media is None else int(media)

    def list_devices(self) -> list[ScannerDevice]:
        devices: list[ScannerDevice] = []
        self._media_by_device.clear()

        try:
            found = minscan.devices()
        except minscan.MinscanError as exc:
            raise RuntimeError(f"minscan detect: {exc}") from exc

        for path, media in found:
            value = int(media) if media is not None else None
            self._media_by_device[path] = value if value in (_MEDIA_35MM, _MEDIA_120) else None
            devices.append(
                ScannerDevice(
                    id=f"{_DEVICE_PREFIX}{path}",
                    vendor="Minolta",
                    model=_MODEL,
                    capabilities=_capabilities(value),
                )
            )
        return devices

    def refresh_devices(self) -> list[ScannerDevice]:
        return self.list_devices()

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float, str], None],
        cancel: threading.Event,
    ) -> ScanResult:
        with self.open_session(device_id) as session:
            return session.scan(params, progress, cancel)

    def _resolve_path(self, device_id: str) -> str:
        path = device_id.split(":", 1)[1] if device_id.startswith(_DEVICE_PREFIX) else device_id
        if path not in ("", "auto"):
            return path
        try:
            detected = minscan.detect()
        except minscan.MinscanError as exc:
            raise RuntimeError(f"minscan detect: {exc}") from exc
        if detected is None:
            raise ScannerUnavailable("No Minolta scanner found")
        return detected

    def open_session(self, device_id: str) -> ScannerSession:
        path = self._resolve_path(device_id)
        if path not in self._media_by_device:
            media = self._probe(path)
            self._media_by_device[path] = media if media in (_MEDIA_35MM, _MEDIA_120) else None
        return MinscanSession(f"{_DEVICE_PREFIX}{path}", self._media_by_device.get(path))

    def open_roll(
        self,
        device: ScannerDevice,
        *,
        dpi: int,
        film_format: str | None = None,
        film_type: str = "negative",
    ) -> MinscanRollSession:
        session = self.open_session(device.id)
        return MinscanRollSession(
            session,
            device,
            dpi=dpi,
            film_format=film_format,
            film_type=film_type,
        )

    def eject(self, device_id: str) -> bool:
        with self.open_session(device_id) as session:
            return session.eject()
