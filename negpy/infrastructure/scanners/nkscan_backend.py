# SPDX-License-Identifier: GPL-3.0-or-later
"""NegPy ``ScannerBackend`` adapter for the nkscan Nikon Coolscan driver.

nkscan addresses a frame by its rectangle on the film, measured per strip, where the rest of
NegPy addresses one by index. The index is resolved against the rectangles the last discovery
found, cached per device: they are absolute stage addresses, so they stay valid until the film moves.

The backend keeps one nkscan session open per unit, as Nikon Scan does. Each operation uses that
session. An error closes it, and the next operation opens it again.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable
from contextlib import contextmanager, suppress
from typing import Any, Iterator

import numpy as np

from negpy.infrastructure.scanners.base import (
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    StripReturned,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import (
    FILM_TYPES,
    ScanMode,
    ScanParams,
    dpi_stops_in_range,
    film_passes_infrared,
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

# Idle time after which a unit returns a loaded strip without notice; a strip put back in reads as loaded.
# Set under the unit's own timeout: a false return costs only a re-measure.
_IDLE_RETURN_S = 9 * 60

_MM_PER_INCH = 25.4

# The maximum number of thumbnail lines that a frame moves in each direction. This is one frame,
# as in Nikon Scan (`CSAThumbCtrl::BaseLineOffset`).
_MAX_NUDGE_LINES = 144

# The two nkscan framing mechanisms that measure the frames. Only a strip feeder that measures
# the frames can return a strip without a command.
_MEASURED_FRAMING = ("thumbnail", "perforation")


def _close_units(units: dict[str, Any]) -> None:
    for session in list(units.values()):
        with suppress(Exception):
            session.close()
    units.clear()


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
        strip_pass=bool(caps.thumbnail),
        # Only when the holder or adapter does not set the format.
        film_formats=tuple(caps.film_formats),
        film_types=tuple(FILM_TYPES),
        max_samples=max(1, int(caps.max_samples)),
        # One read mode means no Superfine to switch: every pass on the unit is already one
        # line at a time.
        superfine=bool(caps.multi_line),
        exposure_lock=True,
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
    if params.capture_ir and params.clean:
        # ICE changes the image with the infrared plane, thus an IR file next to it has no use.
        raise RuntimeError("IR and ICE cannot be used together: ICE changes the image that the IR file is for")
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
        if ejected:
            self._backend._returned.discard(self.device_id)
        return ejected

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # The unit stays open in the backend. This only ends the hold.
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
        # The open nkscan session of each unit. The finalizer closes them when NegPy stops.
        self._units: dict[str, Any] = {}
        self._finalizer = weakref.finalize(self, _close_units, self._units)
        # Frame rects from the last discovery, per device. Absolute stage addresses, so they
        # outlive the session that measured them, and so does the pass they were measured on:
        # re-previewing a strip after a nudge must not cost another read of the film.
        self._frames: dict[str, list[tuple[int, int, int, int]]] = {}
        self._strips: dict[str, np.ndarray] = {}
        self._columns: dict[str, np.ndarray] = {}  # The feed address of each column of the strip pass
        self._densities: dict[str, tuple[float, float, float]] = {}  # The film base density from the strip pass
        self._optical: dict[str, int] = {}  # The optical dpi of the rect addresses
        # The film format and the film type of the measurement. The two change the frame search.
        self._measured_for: dict[str, tuple[str | None, str]] = {}
        # Unreported strip returns, kept apart from the rects so a failed load or eject keeps the return.
        self._returned: set[str] = set()
        self._last_contact = time.monotonic()  # end of the last command to any unit
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
        locations = {str(device.location) for device in found}
        # A unit that is gone does not come back on the same session.
        for location in [loc for loc in self._units if loc not in locations]:
            self._drop(location)
        devices: list[ScannerDevice] = []
        for device in found:
            probed = self._probe_one(device, str(device.location))
            if probed is not None:
                devices.append(probed)
        return devices

    def _probe_one(self, device: Any, location: str) -> ScannerDevice | None:
        """The unit and its capabilities. This opens the unit and keeps it open."""
        with self._lock:
            session = self._units.get(location)
        try:
            if session is None:
                session = self._nk.Session.open(device)
                with self._lock:
                    self._units[location] = session
            else:
                # A command reads the capabilities again if the adapter changed
                session.media_loaded()
        except Exception as exc:
            logger.warning("Could not probe %s: %s", location, exc)
            self._drop(location)
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
            self._drop(location)
            return None

    def close(self) -> None:
        """Close every open unit."""
        _close_units(self._units)

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

    def _unit(self, device_id: str) -> Any:
        """The open nkscan session of the unit. This opens the unit if it is not open."""
        with self._lock:
            session = self._units.get(device_id)
        if session is None:
            with self._mapped_errors():
                session = self._nk.Session(device_id)
            with self._lock:
                self._units[device_id] = session
        return session

    def _drop(self, device_id: str) -> None:
        """Close the session of the unit. The next operation opens the unit again."""
        with self._lock:
            session = self._units.pop(device_id, None)
        if session is not None:
            with suppress(Exception):
                session.close()

    @contextmanager
    def _dropping(self, device_id: str) -> Iterator[None]:
        """Close the unit if the operation fails, because nkscan may not know the state of the unit."""
        try:
            yield
        except StripReturned:
            raise
        except Exception:
            self._drop(device_id)
            raise

    @contextmanager
    def _using(self, device_id: str, *, notice_return: bool = True) -> Iterator[tuple[Any, str]]:
        """The staged unit and its model, for one operation. An error closes the unit."""
        session, model = self._open(device_id, notice_return=notice_return)
        with self._dropping(device_id):
            yield session, model

    def _open(self, device_id: str, *, notice_return: bool = True) -> tuple[Any, str]:
        """Stage the unit at `device_id` for a scan. This opens the unit if it is not open.

        Raises StripReturned when the unit returned a measured strip, unless `notice_return=False` (an eject).
        Only a strip feeder that measures its frames (SA-21, SA-30) can return a strip without a command.
        """
        device = next((d for d in self.list_devices() if d.id == device_id), None)
        model = device.model if device is not None else ""
        idle = time.monotonic() - self._last_contact
        session = self._unit(device_id)
        try:
            returns = str(session.capabilities.framing) in _MEASURED_FRAMING
            returned: StripReturned | None = None
            with self._mapped_errors():
                loaded = True
                if not session.media_loaded():
                    # An Eject already dropped the rects, so rects still held mean the unit returned the strip.
                    # load() takes in only a strip that waits in the adapter.
                    if returns and device_id in self._frames:
                        self._returned.add(device_id)
                    self.forget_frames(device_id)
                    loaded = bool(session.load())
                elif returns and idle >= _IDLE_RETURN_S and device_id in self._frames:
                    # Long enough for the unit to return the strip and for it to go back in unseen.
                    self._returned.add(device_id)
                    self.forget_frames(device_id)
                if notice_return and device_id in self._returned:
                    self._returned.discard(device_id)
                    returned = StripReturned(loaded=loaded)
                else:
                    session.stage()
            if returned is not None:
                raise returned
        except StripReturned:
            raise
        except Exception:
            self._drop(device_id)
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
        with self._using(device_id) as (session, model):
            return self._scan_on_session(session, device_id, model, params, progress, cancel)

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
        rect = self.nudged(device_id, rect, params.frame_offset_mm, optical)
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
                exposures=exposures if exposures is not None else params.exposures,
                progress=report,
                **self.metering(device_id, params.film_type, params.film_format),
            )
        if cancel.is_set():
            raise RuntimeError("Scan cancelled")
        if result.cleaned:
            logger.info("Dust removal rebuilt %d pixels", result.cleaned)
        # ICE reads the infrared plane to clean the frame. Keep it only when the scan asks for IR.
        return self._to_result(result, model, self._profile(session, params.film_type), keep_ir=bool(params.capture_ir))

    def meter(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[..., None],
        cancel: threading.Event,
    ) -> dict[str, int]:
        """Exposures for `params.frame`, to hand later scans through `ScanParams.exposures`.

        Metered on the whole detected frame, never the window: nkscan meters Nikon Scan's crop
        of the frame for its format. Infrared is metered too, so the result also holds for an
        IR or ICE scan.
        """
        with self._lock:
            if device_id in self._sessions:
                raise RuntimeError(f"Device {device_id} is held by an open session")
        _validate_params(params)
        with self._using(device_id) as (session, _model):
            report = _progress_bridge(progress, cancel)
            rect = self._resolve_frame(session, device_id, params, report)
            rect = self.nudged(device_id, rect, params.frame_offset_mm, int(session.capabilities.optical_dpi))
            metering = self.metering(device_id, params.film_type, params.film_format)
            with self._mapped_errors():
                exposures = session.meter_frame(rect, infrared=True, progress=report, **metering)
        if cancel.is_set():
            raise RuntimeError("Metering cancelled")
        return {str(k): int(v) for k, v in exposures.items()}

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
        result = session.scan_frame(rect, superfine=want, **options)
        if not result.complete:
            raise TransientScanError(f"The pass ended early: {result.blocks} blocks arrived")
        return result

    def metering(self, device_id: str, film_type: str, film_format: str | None) -> dict[str, Any]:
        """The values that nkscan uses to meter a frame: the film, the format and the film base density.

        nkscan picks the white balance lock for the film, so a colour negative is metered per channel.
        """
        return {"film": film_type, "format": film_format, "base_density": self._densities.get(device_id)}

    def _profile(self, session: Any, film_type: str) -> bytes | None:
        """The Nikon Scan profile for this unit and film, as the nkscan CLI embeds it. None if there is no profile."""
        try:
            profile = session.nikon_profile(film_type)
        except Exception as exc:
            logger.warning("No Nikon profile for %s: %s", film_type, exc)
            return None
        return bytes(profile) if profile else None

    def _to_result(self, result: Any, model: str, icc_profile: bytes | None = None, *, keep_ir: bool = True) -> ScanResult:
        ir = result.ir if keep_ir else None
        return ScanResult(
            rgb=_stack_rgb(result.colors),
            ir=ir,
            dpi=int(result.dpi),
            device_model=model,
            ir_valid_mask=np.ones(ir.shape[:2], dtype=np.bool_) if ir is not None else None,
            exposures={str(k): int(v) for k, v in result.exposures.items()},
            icc_profile=icc_profile,
        )

    # ── frames ────────────────────────────────────────────────────────

    def discover_frames(
        self,
        session: Any,
        device_id: str,
        *,
        film_format: str | None,
        film_type: str,
        progress: Callable[..., bool] | None = None,
    ) -> Any:
        """Measure the loaded film, cache the rects, and return nkscan's Discovery."""
        with self._mapped_errors():
            discovery = session.discover_frames(format=film_format, film=film_type, progress=progress)
        # A short thumbnail pass reads as film ending early, so its fit miscounts the frames.
        if discovery.thumbnail_complete is False:
            raise TransientScanError(f"The thumbnail pass ended early: {discovery.thumbnail_blocks} blocks arrived")
        self.forget_frames(device_id)
        self._frames[device_id] = [tuple(int(v) for v in rect) for rect in discovery.frames]
        self._optical[device_id] = int(session.capabilities.optical_dpi)
        self._measured_for[device_id] = (film_format, film_type)
        if discovery.thumbnail and discovery.column_addresses:
            self._strips[device_id] = _stack_rgb(discovery.thumbnail)
            self._columns[device_id] = np.asarray(discovery.column_addresses, dtype=np.int64)
        if discovery.base_density is not None:
            self._densities[device_id] = tuple(float(d) for d in discovery.base_density)
        logger.info("Detected %d frames on %s", len(self._frames[device_id]), device_id)
        return discovery

    def detect_frames(self, device_id: str, *, film_format: str | None = None, film_type: str = "negative") -> int:
        """How many frames the loaded film carries, measuring it only if that is not known.

        A strip previewed a moment ago is already measured, so this usually costs nothing.
        """
        known = self.measured_frames(device_id, film_format, film_type)
        if known:
            return len(known)
        with self._lock:
            held = self._sessions.get(device_id)
        if held is not None:
            with self._mapped_errors():
                self.discover_frames(held._session, device_id, film_format=film_format, film_type=film_type)
            return len(self._frames.get(device_id, ()))
        with self._using(device_id) as (session, _model), self._mapped_errors():
            self.discover_frames(session, device_id, film_format=film_format, film_type=film_type)
        return len(self._frames.get(device_id, ()))

    def frames(self, device_id: str) -> list[tuple[int, int, int, int]]:
        return list(self._frames.get(device_id, ()))

    def measured_frames(self, device_id: str, film_format: str | None, film_type: str) -> list[tuple[int, int, int, int]]:
        """The cached rects for this film. If the rects are for a different film, this deletes them."""
        if device_id in self._frames and self._measured_for.get(device_id) != (film_format, film_type):
            self.forget_frames(device_id)
        return self.frames(device_id)

    def strip_pass(self, device_id: str) -> np.ndarray | None:
        """The whole-strip read the frames were measured on, where the mechanism took one."""
        return self._strips.get(device_id)

    def optical_dpi(self, device_id: str) -> int | None:
        """The optical dpi of the cached rect addresses."""
        return self._optical.get(device_id)

    def nudged(self, device_id: str, rect: tuple[int, int, int, int], offset_mm: float, optical_dpi: int) -> tuple[int, int, int, int]:
        """Move `rect` along the feed by approximately `offset_mm`, as Nikon Scan moves a frame.

        Nikon Scan moves a frame by full thumbnail lines. It moves a frame by one frame or less,
        and it keeps the frame on the strip pass. The top is then the address of a line.
        Without the column addresses of the strip pass, this moves the rect by `offset_mm`.
        """
        units = _offset_units(offset_mm, optical_dpi)
        columns = self._columns.get(device_id)
        if columns is None or not len(columns) or not units:
            return _shift_frame(rect, units)
        top, _left, bottom, _right = rect
        first = int(np.searchsorted(columns, top))
        last = int(np.searchsorted(columns, bottom))
        wanted = int(np.searchsorted(columns, top + units))
        if 0 < wanted < len(columns) and top + units - columns[wanted - 1] < columns[wanted] - (top + units):
            wanted -= 1
        back = -min(first, _MAX_NUDGE_LINES)
        ahead = max(0, min(_MAX_NUDGE_LINES, len(columns) - 1 - last))
        lines = max(back, min(wanted - first, ahead))
        if lines == 0:
            return rect
        return _shift_frame(rect, int(columns[first + lines]) - top)

    def column_addresses(self, device_id: str) -> np.ndarray | None:
        """The feed address of each column of the strip pass. The addresses do not decrease."""
        return self._columns.get(device_id)

    def set_frame(self, device_id: str, slot: int, rect: tuple[int, int, int, int]) -> None:
        """Replace one detected rect, so a nudge in the preview reaches the fine scan."""
        frames = self._frames.get(device_id)
        if frames and 1 <= slot <= len(frames):
            frames[slot - 1] = rect

    def forget_frames(self, device_id: str) -> None:
        """Drop the cached rects and the pass they came from: that film has moved."""
        self._frames.pop(device_id, None)
        self._strips.pop(device_id, None)
        self._columns.pop(device_id, None)
        self._densities.pop(device_id, None)
        self._optical.pop(device_id, None)
        self._measured_for.pop(device_id, None)

    def _resolve_frame(
        self,
        session: Any,
        device_id: str,
        params: ScanParams,
        report: Callable[..., bool],
    ) -> tuple[int, int, int, int]:
        frames = self.measured_frames(device_id, params.film_format, params.film_type)
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

        # A measured strip gives its tiles from the strip pass. Open the unit only to measure or scan.
        cut = self.measured_frames(device.id, film_format, film_type) and self.strip_pass(device.id) is not None
        session, model = (None, device.model) if cut else self._open(device.id)
        return NkscanRollSession(self, device, session, model, dpi=dpi, film_format=film_format, film_type=film_type)

    # ── film ──────────────────────────────────────────────────────────

    def media_loaded(self, device_id: str) -> bool:
        """Whether film or a holder is in the unit. If the film went out, this drops the measured frames."""
        with self._dropping(device_id), self._mapped_errors():
            loaded = bool(self._unit(device_id).media_loaded())
        if not loaded:
            self.forget_frames(device_id)
        return loaded

    # ── eject ─────────────────────────────────────────────────────────

    def eject(self, device_id: str) -> bool:
        with self._lock:
            held = self._sessions.get(device_id)
        if held is not None:
            return held.eject()
        try:
            with self._using(device_id, notice_return=False) as (session, _model), self._mapped_errors():
                if device_id in self._returned and not session.media_loaded():
                    ejected = True
                else:
                    ejected = bool(session.eject())
        finally:
            self.forget_frames(device_id)
        if ejected:
            self._returned.discard(device_id)
        return ejected

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
        finally:
            self._last_contact = time.monotonic()
