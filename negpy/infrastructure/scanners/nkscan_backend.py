# SPDX-License-Identifier: GPL-3.0-or-later
"""NegPy ``ScannerBackend`` adapter for the nkscan Nikon Coolscan driver.

nkscan addresses a frame by its rectangle on the film, measured per strip, where the rest of
NegPy addresses one by index. The index is resolved against the rectangles the last discovery
found, cached per device: they are absolute stage addresses, so they stay valid across session
closes and until the film moves.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager, suppress
from typing import Any, Iterator

import numpy as np

from negpy.infrastructure.scanners.base import (
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import (
    FILM_TYPES,
    ScanMode,
    ScanParams,
    dpi_stops_in_range,
    film_passes_infrared,
    film_reads_positive,
)
from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_INSTALL_HINT = (
    "nkscan is not installed. Install it with: uv sync --group nkscan. On Linux a Coolscan on "
    "USB needs a udev rule for Nikon (04b0); on FireWire/SCSI it needs the sg kernel module."
)

# Frame lengths nkscan cannot measure for itself, as it spells them.
FILM_FORMATS = ("135", "half", "IX240", "16", "645", "66", "67", "68", "69")

_MAX_SAMPLES = 16  # the protocol's own ceiling; a unit's own limit comes from its capabilities

# 135, across the film by along the feed: the raster is portrait. Only the tile aspect and the
# window mm readout use it, and no adapter reports its opening through the bindings.
_DEFAULT_AREA_MM = (24.0, 36.0)

_PHASES = {"discover": "Detecting frames", "meter": "Metering", "scan": "Scanning"}

_MM_PER_INCH = 25.4

# The two of nkscan's four framing mechanisms that take a thumbnail pass, and so the two that
# can be told a frame length. The others read the holder's own table or an address.
_MEASURED_FRAMING = ("thumbnail", "perforation")


def _caps_for(caps: Any) -> ScannerCapabilities:
    lo, hi = (int(v) for v in caps.x_dpi_range)
    optical = int(caps.optical_dpi)
    dpi = tuple(sorted({*dpi_stops_in_range(lo, hi), optical}))
    return ScannerCapabilities(
        ir_channel=True,
        supported_dpi=dpi,
        # Every plane comes back stretched to a 16-bit ceiling, whatever the unit read at.
        supported_depths=(16,),
        sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE),
        max_area_mm=_DEFAULT_AREA_MM,
        # nkscan meters every frame itself and focuses itself, and offers no parameter for
        # either, so neither has a control to carry — `hardware_metering` and `autofocus` say
        # what the unit does, not what a caller may ask for.
        auto_exposure=False,
        autofocus=False,
        # Frames come from `discover_frames`, never from an index, so `max_frames` is not a
        # capacity to range over: the strip dialog picks from what the film turned out to hold.
        adapter_frame_capacity=None,
        can_eject=bool(caps.eject),
        exposure_time_us=None,
        hw_clean=True,
        roll_discovery=True,
        film_formats=FILM_FORMATS if str(caps.framing) in _MEASURED_FRAMING else (),
        film_types=tuple(FILM_TYPES),
        max_samples=max(1, int(caps.max_samples)),
        # One read mode means no Superfine to switch: every pass on the unit is already one
        # line at a time.
        superfine=bool(caps.multi_line),
    )


def _safe_progress(
    progress: Callable[..., None] | None,
    value: float,
    phase: str = "Scanning",
) -> None:
    if progress is None:
        return
    with suppress(Exception):
        progress(max(0.0, min(1.0, float(value))), phase)


def _progress_bridge(progress: Callable[..., None] | None, cancel: threading.Event) -> Callable[..., bool]:
    """nkscan's (phase, pass, done, total) callback over NegPy's (fraction, phase) one.

    Returning False is how nkscan is cancelled, so the event is polled here rather than
    between passes.
    """

    def report(phase: str, _pass: int, done: int, total: int) -> bool:
        _safe_progress(progress, (done / total) if total else 0.0, _PHASES.get(phase, "Scanning"))
        return not cancel.is_set()

    return report


def _validate_params(params: ScanParams) -> None:
    samples = int(params.samples)
    if not 1 <= samples <= _MAX_SAMPLES:
        raise RuntimeError(f"Samples must be 1..{_MAX_SAMPLES}, not {samples}")
    if params.auto_exposure:
        raise RuntimeError("Auto-exposure requested but nkscan meters every frame itself")
    if params.film_format is not None and params.film_format not in FILM_FORMATS:
        raise RuntimeError(f"Unknown film format {params.film_format!r}; expected one of {', '.join(FILM_FORMATS)}")
    if params.film_type not in FILM_TYPES:
        raise RuntimeError(f"Unknown film type {params.film_type!r}; expected one of {', '.join(FILM_TYPES)}")
    if (params.capture_ir or params.clean) and not film_passes_infrared(params.film_type):
        label = FILM_TYPES[params.film_type][0]
        raise RuntimeError(f"{label} blocks infrared, so IR and ICE have nothing to read on it")
    # `depth` and `autofocus` carry no request here: every scan comes back 16-bit and focuses
    # itself, so honouring the defaults silently is the truth, not a skipped option.


def _shift_frame(rect: tuple[int, int, int, int], units: int) -> tuple[int, int, int, int]:
    """Slide a frame along the feed axis, floored at the start of the stage's range."""
    top, left, bottom, right = rect
    units = max(units, -top)
    return (top + units, left, bottom + units, right)


def _offset_units(offset_mm: float, optical_dpi: int) -> int:
    """Feed-axis mm as stage addresses: one address is one pixel at the optical resolution."""
    if not offset_mm or optical_dpi <= 0:
        return 0
    return int(round(offset_mm * optical_dpi / _MM_PER_INCH))


def _crop_frame(
    rect: tuple[int, int, int, int],
    window: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int]:
    """Apply a normalized window inside a frame rect. x is across the film, y along the feed."""
    if window is None:
        return rect
    top, left, bottom, right = rect
    x1, y1, x2, y2 = window
    height = bottom - top
    width = right - left
    new_top = top + int(round(min(y1, y2) * height))
    new_bottom = top + int(round(max(y1, y2) * height))
    new_left = left + int(round(min(x1, x2) * width))
    new_right = left + int(round(max(x1, x2) * width))
    return (new_top, new_left, max(new_bottom, new_top + 1), max(new_right, new_left + 1))


def _stack_rgb(colors: dict[str, np.ndarray]) -> np.ndarray:
    """One (rows, cols, 3) array from nkscan's per-channel planes.

    A unit or adapter with no colour components delivers one plane, which every downstream
    stage still expects three of.
    """
    planes = [colors[name] for name in ("red", "green", "blue") if name in colors]
    if len(planes) != 3:
        if not colors:
            raise RuntimeError("The scan returned no image planes")
        planes = [next(iter(colors.values()))] * 3
    return np.stack(planes, axis=-1)


class NkscanSession:
    """Exclusive hold on one Coolscan: one nkscan session, N scans, one release."""

    def __init__(self, backend: NkscanBackend, device_id: str, session: Any, model: str) -> None:
        self.device_id = device_id
        self._backend = backend
        self._session = session
        self._model = model
        self._closed = False

    def scan(
        self,
        params: ScanParams,
        progress: Callable[..., None],
        cancel: threading.Event,
    ) -> ScanResult:
        self._require_open()
        return self._backend._scan_on_session(self._session, self.device_id, self._model, params, progress, cancel)

    def eject(self) -> bool:
        self._require_open()
        with self._backend._mapped_errors():
            ejected = bool(self._session.eject())
        self._backend.forget_frames(self.device_id)
        return ejected

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._session.close()
        finally:
            self._backend._release_session(self)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")

    def __enter__(self) -> NkscanSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NkscanBackend:
    """ScannerBackend for Nikon Coolscans over nkscan."""

    def __init__(self) -> None:
        try:
            import nkscan
        except ImportError as exc:
            raise ScannerUnavailable(_INSTALL_HINT) from exc
        self._nk = nkscan
        self._devices_cache: list[ScannerDevice] | None = None
        self._sessions: dict[str, NkscanSession] = {}
        # Frame rects from the last discovery, per device. Absolute stage addresses, so they
        # outlive the session that measured them, and so does the pass they were measured on:
        # re-previewing a strip after a nudge must not cost another read of the film.
        self._frames: dict[str, list[tuple[int, int, int, int]]] = {}
        self._strips: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # ── enumeration ───────────────────────────────────────────────────

    def list_devices(self) -> list[ScannerDevice]:
        if self._devices_cache is None:
            self._devices_cache = self._probe_devices()
        return [d for d in self._devices_cache if d.capabilities.sources]

    def refresh_devices(self) -> list[ScannerDevice]:
        self._devices_cache = None
        return self.list_devices()

    def _probe_devices(self) -> list[ScannerDevice]:
        with self._mapped_errors():
            found = list(self._nk.list_devices())
        cached = {d.id: d for d in self._devices_cache or []}
        devices: list[ScannerDevice] = []
        for device in found:
            location = str(device.location)
            # Probing opens the unit, and a held one refuses. Reuse what the hold was built from.
            if location in self._sessions and location in cached:
                devices.append(cached[location])
                continue
            probed = self._probe_one(device, location)
            if probed is not None:
                devices.append(probed)
        return devices

    def _probe_one(self, device: Any, location: str) -> ScannerDevice | None:
        try:
            session = self._nk.Session.open(device)
        except Exception as exc:
            logger.warning("Could not probe %s: %s", location, exc)
            return None
        try:
            caps = session.capabilities
            return ScannerDevice(
                id=location,
                vendor=str(caps.vendor).strip(),
                model=str(caps.model or caps.product).strip(),
                capabilities=_caps_for(caps),
            )
        except Exception as exc:
            logger.warning("Could not read capabilities of %s: %s", location, exc)
            return None
        finally:
            with suppress(Exception):
                session.close()

    # ── sessions ──────────────────────────────────────────────────────

    def open_session(self, device_id: str) -> ScannerSession:
        with self._lock:
            if device_id in self._sessions:
                raise RuntimeError(f"Device already held in a session: {device_id}")
        session, model = self._open(device_id)
        held = NkscanSession(self, device_id, session, model)
        with self._lock:
            self._sessions[device_id] = held
        return held

    def _release_session(self, session: NkscanSession) -> None:
        with self._lock:
            self._sessions.pop(session.device_id, None)

    def _open(self, device_id: str) -> tuple[Any, str]:
        """Open the unit at `device_id` and stage it for a scan."""
        model = next((d.model for d in self.list_devices() if d.id == device_id), "")
        with self._mapped_errors():
            session = self._nk.Session(device_id)
        try:
            with self._mapped_errors():
                if not session.media_loaded():
                    session.load()
                session.stage()
        except Exception:
            with suppress(Exception):
                session.close()
            raise
        return session, model

    # ── scanning ──────────────────────────────────────────────────────

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[..., None],
        cancel: threading.Event,
    ) -> ScanResult:
        with self._lock:
            if device_id in self._sessions:
                raise RuntimeError(f"Device {device_id} is held by an open session; use session.scan()")
        # Validate before opening: a refused option must never leave the unit staged.
        _validate_params(params)
        session, model = self._open(device_id)
        try:
            return self._scan_on_session(session, device_id, model, params, progress, cancel)
        finally:
            with suppress(Exception):
                session.close()

    def _scan_on_session(
        self,
        session: Any,
        device_id: str,
        model: str,
        params: ScanParams,
        progress: Callable[..., None],
        cancel: threading.Event,
        *,
        exposures: dict[str, int] | None = None,
    ) -> ScanResult:
        _validate_params(params)
        if cancel.is_set():
            raise RuntimeError("Scan cancelled before start")
        report = _progress_bridge(progress, cancel)
        rect = self._resolve_frame(session, device_id, params, report)
        optical = int(session.capabilities.optical_dpi)
        detected = rect
        rect = _shift_frame(rect, _offset_units(params.frame_offset_mm, optical))
        rect = _crop_frame(rect, params.window)
        logger.info("Frame %s detected %s, scanning %s (%+0.2f mm)", params.frame, detected, rect, params.frame_offset_mm)
        with self._mapped_errors():
            result = self.scan_frame(
                session,
                rect,
                dpi=int(params.dpi),
                samples=int(params.samples),
                superfine=bool(params.superfine),
                infrared=bool(params.capture_ir),
                clean=bool(params.clean),
                lock_white_balance=self.locks_white_balance(params.film_type),
                exposures=exposures,
                progress=report,
            )
        if cancel.is_set():
            raise RuntimeError("Scan cancelled")
        if result.cleaned:
            logger.info("Dust removal rebuilt %d pixels", result.cleaned)
        return self._to_result(result, model)

    def scan_frame(
        self,
        session: Any,
        rect: tuple[int, int, int, int],
        *,
        superfine: bool = False,
        **options: Any,
    ) -> Any:
        """One pass over `rect`. Every scan goes through here, previews included.

        A unit whose CCD cannot read its lines at once — the LS-50 cannot — has only the
        superfine ordering, and asking for the fast one is refused before the stage moves.
        """
        want = bool(superfine) or not bool(session.capabilities.multi_line)
        return session.scan_frame(rect, superfine=want, **options)

    def locks_white_balance(self, film_type: str) -> bool:
        """nkscan's own metering default for this film.

        A colour negative is metered per channel, which takes the orange mask off before the
        ADC instead of quantising the blue record through it; everything else keeps the factory
        balance, because there the cast is the picture.
        """
        return bool(self._nk.Capabilities.locks_white_balance(film_type))

    def _to_result(self, result: Any, model: str) -> ScanResult:
        ir = result.ir
        return ScanResult(
            rgb=_stack_rgb(result.colors),
            ir=ir,
            dpi=int(result.dpi),
            device_model=model,
            ir_valid_mask=np.ones(ir.shape[:2], dtype=np.bool_) if ir is not None else None,
        )

    # ── frames ────────────────────────────────────────────────────────

    def discover_frames(
        self,
        session: Any,
        device_id: str,
        *,
        film_format: str | None,
        film_type: str = "negative",
        progress: Callable[..., bool] | None = None,
    ) -> Any:
        """Measure the loaded film, cache the rects, and return nkscan's Discovery."""
        with self._mapped_errors():
            discovery = session.discover_frames(
                format=film_format,
                positive=film_reads_positive(film_type),
                progress=progress,
            )
        self._frames[device_id] = [tuple(int(v) for v in rect) for rect in discovery.frames]
        thumbnail = getattr(discovery, "thumbnail", None)
        if thumbnail:
            self._strips[device_id] = _stack_rgb(thumbnail)
        logger.info("Detected %d frames on %s", len(self._frames[device_id]), device_id)
        return discovery

    def detect_frames(self, device_id: str, *, film_format: str | None = None, film_type: str = "negative") -> int:
        """How many frames the loaded film carries, measuring it only if that is not known.

        A strip previewed a moment ago is already measured, so this usually costs nothing.
        """
        known = self._frames.get(device_id)
        if known:
            return len(known)
        with self._lock:
            held = self._sessions.get(device_id)
        if held is not None:
            with self._mapped_errors():
                self.discover_frames(held._session, device_id, film_format=film_format, film_type=film_type)
            return len(self._frames.get(device_id, ()))
        session, _model = self._open(device_id)
        try:
            with self._mapped_errors():
                self.discover_frames(session, device_id, film_format=film_format, film_type=film_type)
        finally:
            with suppress(Exception):
                session.close()
        return len(self._frames.get(device_id, ()))

    def frames(self, device_id: str) -> list[tuple[int, int, int, int]]:
        return list(self._frames.get(device_id, ()))

    def strip_pass(self, device_id: str) -> np.ndarray | None:
        """The whole-strip read the frames were measured on, where the mechanism took one."""
        return self._strips.get(device_id)

    def set_frame(self, device_id: str, slot: int, rect: tuple[int, int, int, int]) -> None:
        """Replace one detected rect, so a nudge in the preview reaches the fine scan."""
        frames = self._frames.get(device_id)
        if frames and 1 <= slot <= len(frames):
            frames[slot - 1] = rect

    def forget_frames(self, device_id: str) -> None:
        """Drop the cached rects and the pass they came from: that film has moved."""
        self._frames.pop(device_id, None)
        self._strips.pop(device_id, None)

    def _resolve_frame(
        self,
        session: Any,
        device_id: str,
        params: ScanParams,
        report: Callable[..., bool],
    ) -> tuple[int, int, int, int]:
        frames = self._frames.get(device_id)
        if not frames:
            self.discover_frames(
                session,
                device_id,
                film_format=params.film_format,
                film_type=params.film_type,
                progress=report,
            )
            frames = self._frames.get(device_id)
        if not frames:
            raise RuntimeError("No frames were detected on the loaded film")
        # No frame requested means the first one on the strip, never whatever the stage happens
        # to be over.
        index = 1 if params.frame is None else int(params.frame)
        if not 1 <= index <= len(frames):
            raise RuntimeError(f"Frame {index} was not detected on this strip ({len(frames)} found)")
        return frames[index - 1]

    # ── strip preview ─────────────────────────────────────────────────

    def open_roll(
        self,
        device: ScannerDevice,
        *,
        dpi: int,
        film_format: str | None = None,
        film_type: str = "negative",
    ) -> Any:
        from negpy.infrastructure.scanners.nkscan_roll import NkscanRollSession

        session, model = self._open(device.id)
        return NkscanRollSession(self, device, session, model, dpi=dpi, film_format=film_format, film_type=film_type)

    # ── eject ─────────────────────────────────────────────────────────

    def eject(self, device_id: str) -> bool:
        with self._lock:
            held = self._sessions.get(device_id)
        if held is not None:
            return held.eject()
        session, _model = self._open(device_id)
        try:
            with self._mapped_errors():
                return bool(session.eject())
        finally:
            self.forget_frames(device_id)
            with suppress(Exception):
                session.close()

    # ── errors ────────────────────────────────────────────────────────

    @contextmanager
    def _mapped_errors(self) -> Iterator[None]:
        """nkscan's exception tree onto NegPy's, by type — never by message."""
        nk = self._nk
        try:
            yield
        except nk.ScanCancelled as exc:
            raise RuntimeError("Scan cancelled") from exc
        except nk.TransientError as exc:
            raise TransientScanError(str(exc)) from exc
        except nk.UnsupportedError as exc:
            raise RuntimeError(f"{getattr(exc, 'op', 'operation')}: {getattr(exc, 'reason', exc)}") from exc
        except nk.ScannerError as exc:
            raise RuntimeError(str(exc)) from exc
