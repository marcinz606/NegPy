# SPDX-License-Identifier: GPL-3.0-or-later
"""NegPy ``ScannerBackend`` adapter for plusteck (Plustek OpticFilm 135i).

This backend never touches USB. It drives plusteck's own local FastAPI service
(plusteck/backend/src/plusteck/api/main.py, default http://127.0.0.1:48213) over HTTP and
reads the resulting files off the local disk, both processes being on the same machine.
plusteck owns the hardware-validated protocol logic: the golden-sequence replay, the
resolution tiers, the IR toggle and per-slot boundary detection. This adapter only
translates the ScannerBackend, ScannerSession and RollSession protocols into calls
against it, so it stays in step with that sequence and inherits its fixes. Start the
service with plusteck's `backend/run_backend.sh` first.

An in-process USB driver, the shape pyopticfilm, nkscan and pieusb take, would fight
plusteck's own device lock over the one physical device.

One physical pass covers a whole tray rather than one frame: the golden sequence replays
a captured USB trace of a full carriage sweep, and the driver cannot address one frame.
PlusteckBackend._scan_via_cache reconciles that with the one scan() per frame
ScanWorker.run_batch expects, and says why the cache lives on the backend.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import tifffile

from negpy.infrastructure.scanners.base import (
    ScannerCapabilities,
    ScannerDevice,
    ScannerSession,
    ScannerUnavailable,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.system.logging import get_logger

if TYPE_CHECKING:
    # Deferred: plusteck_roll imports helpers back out of this module (same
    # pattern as nkscan_roll.py <-> nkscan_backend.py), so importing it at
    # module level here would be circular. See open_roll's own local import.
    from negpy.infrastructure.scanners.roll import RollSession

logger = get_logger(__name__)

ProgressCb = Callable[[float, str], None]

#: Fixed port, mirroring plusteck/backend/src/plusteck/api/main.py's own PORT
#: constant. One instance runs locally, so there is nothing to discover.
DEFAULT_BASE_URL = "http://127.0.0.1:48213"

#: This adapter's one synthetic device id. plusteck exposes no serial number or
#: bus address over HTTP, so enumeration is "reachable or not".
DEVICE_ID = "plusteck:opticfilm135i:local"

_HTTP_TIMEOUT_S = 10.0
_POLL_INTERVAL_S = 1.0
#: Hard ceiling on one physical pass, several times plusteck's own documented
#: worst case for a full tray. A ceiling, not a tight bound.
_SCAN_MAX_WAIT_S = 1800.0

_TRANSIENT_ERROR_HINTS = ("lost connection", "wedged", "timed out", "timeout", "usb error")

#: NegPy dpi -> plusteck's own named resolution tiers (ResolutionTier in
#: backend/src/plusteck/usb/base.py). Only Color + STANDARD_1800 is validated
#: end to end there; plusteck's "Known limitations" panel holds that caveat.
_DPI_TO_TIER = {
    600: "low_600",
    1200: "lower_1200",
    1800: "standard_1800",
    2400: "better_2400",
    3600: "best_3600",
    7200: "max_7200",
}


def _looks_transient(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _TRANSIENT_ERROR_HINTS)


def _http_json(method: str, base_url: str, path: str, body: dict[str, Any] | None = None) -> Any:
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"plusteck API {method} {path} failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise ScannerUnavailable(
            f"plusteck backend not reachable at {base_url} ({exc.reason}). Start it with plusteck's backend/run_backend.sh, then retry."
        ) from exc
    except TimeoutError as exc:
        # urllib wraps a timeout as URLError only while sending, so a slow answer arrives
        # here instead: the process is up and its single device lock is held by another
        # scan or register access. That clears on its own, so it is worth a retry.
        raise TransientScanError(f"plusteck API {method} {path} did not respond within {_HTTP_TIMEOUT_S:.0f}s (busy?)") from exc


def _resolve_dpi_tier(dpi: int) -> str:
    tier = _DPI_TO_TIER.get(int(dpi))
    if tier is None:
        raise RuntimeError(f"Unsupported dpi={dpi}; plusteck's own tiers are {sorted(_DPI_TO_TIER)}")
    return tier


def _caps() -> ScannerCapabilities:
    return ScannerCapabilities(
        ir_channel=True,
        supported_dpi=tuple(sorted(_DPI_TO_TIER)),
        supported_depths=(16,),
        sources=(ScanMode.TRANSPARENCY,),
        # Width comes from Standard 1800 dpi's pixel count: a 35mm frame with its
        # sprocket margin. Height is a whole tray sweep, so the width is reused as a
        # conservative square bound rather than asserting an unmeasured number.
        max_area_mm=(36.6, 36.6),
        auto_exposure=False,
        autofocus=False,
        multi_exposure=False,  # plusteck's own UI: "not available yet"
        can_eject=False,  # tray-release handshake is real but not reliable yet -- see plusteck's docs/HARDWARE.md
        # Slot count is only known once a pass has actually measured the
        # loaded tray (0-6 slots, each possibly split into 2 half-frames) --
        # never addressed ahead of time by index alone.
        roll_discovery=True,
    )


#: How long a completed pass stays usable for a later scan() at the same
#: (dpi, capture_ir). Long enough for one batch, short enough that a different
#: roll is never served the previous one. See _scan_via_cache for the risk.
_CACHE_TTL_S = 600.0


class PlusteckSession:
    """Thin `ScannerSession` wrapper: real one-pass caching lives on the
    `PlusteckBackend` this session was opened from, not here -- see
    `PlusteckBackend._scan_via_cache`'s own docstring for why a
    session-scoped cache would silently do nothing against NegPy's real
    batch-scan codepath."""

    device_id = DEVICE_ID

    def __init__(self, backend: "PlusteckBackend") -> None:
        self._backend = backend
        self._closed = False

    def scan(
        self,
        params: ScanParams,
        progress: ProgressCb | None,
        cancel: threading.Event,
    ) -> ScanResult:
        if self._closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        return self._backend.scan(DEVICE_ID, params, progress, cancel)

    def eject(self) -> bool:
        if self._closed:
            raise RuntimeError(f"Scanner session for {self.device_id} is closed")
        self.close()
        return False  # see ScannerCapabilities.can_eject's own comment

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "PlusteckSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _safe_progress(progress: ProgressCb | None, value: float, phase: str = "Scanning") -> None:
    if progress is None:
        return
    try:
        progress(max(0.0, min(1.0, float(value))), phase)
    except Exception:
        logger.debug("progress callback raised", exc_info=True)


def _wait_for_scan(base_url: str, session_id: str, progress: ProgressCb | None, cancel: threading.Event) -> None:
    """Poll /api/status until this session's scan leaves 'scanning'.

    plusteck's per-byte percent lives on its WebSocket event stream (main.py's
    /api/events). Polling keeps a websocket-client dependency out of NegPy, at the cost of
    a coarse phase indicator instead of fractional progress.
    """
    start = time.monotonic()
    _safe_progress(progress, 0.02, "Scanning")
    while True:
        if cancel.is_set():
            raise RuntimeError("Scan cancelled")
        elapsed = time.monotonic() - start
        if elapsed > _SCAN_MAX_WAIT_S:
            raise TransientScanError(f"plusteck scan {session_id} did not finish within {_SCAN_MAX_WAIT_S:.0f}s")
        status = _http_json("GET", base_url, "/api/status")
        if status.get("scan_session_id") != session_id:
            time.sleep(_POLL_INTERVAL_S)
            continue
        state = status.get("scan_state")
        if state == "scanning":
            # No real fraction available over REST (see docstring) -- creep
            # slowly toward "nearly done" so a progress bar still moves
            # rather than sitting frozen at one value for minutes.
            _safe_progress(progress, min(0.9, 0.1 + elapsed / _SCAN_MAX_WAIT_S), "Scanning")
            time.sleep(_POLL_INTERVAL_S)
            continue
        if state == "done":
            return
        if state == "error":
            message = status.get("scan_error") or "unknown plusteck scan error"
            if _looks_transient(message):
                raise TransientScanError(message)
            raise RuntimeError(message)
        time.sleep(_POLL_INTERVAL_S)


def _ensure_all_split(base_url: str, session_id: str) -> list[Path]:
    """Every tray-slot scan in this session, split into its two half-frame
    TIFFs (POST .../split is documented idempotent -- safe even if a slot
    was already split by someone using plusteck's own UI concurrently).
    Flat, slot-ordered: each scan_id contributes its "_a" then its "_b"."""
    status = _http_json("GET", base_url, "/api/status")
    split_dir = Path(status["split_output_dir"]) / session_id
    scans = _http_json("GET", base_url, f"/api/sessions/{session_id}/scans")
    frames: list[Path] = []
    for entry in sorted(scans, key=lambda e: e["scan_id"]):
        scan_id = entry["scan_id"]
        _http_json("POST", base_url, f"/api/sessions/{session_id}/scans/{scan_id}/split")
        frames.append(split_dir / f"{scan_id}_a.tiff")
        frames.append(split_dir / f"{scan_id}_b.tiff")
    return frames


def _read_half_frame(path: Path) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"expected split output missing: {path}")
    return tifffile.imread(str(path))


class PlusteckBackend:
    """ScannerBackend for the Plustek OpticFilm 135i, via plusteck's own local service.

    Owns the one fine-scan cache (see `_scan_via_cache`). ScannerService._get_backend()
    builds one instance and reuses it for the whole app session, and ScanWorker.run_batch
    reaches it through service.run_scan -> backend.scan rather than a held-open session.
    """

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL) -> None:
        self._base_url = base_url
        # Fail fast with the real install/start hint rather than a generic
        # connection error the first time list_devices() is actually called.
        _http_json("GET", self._base_url, "/api/status")
        # The most recent completed physical pass, if still fresh -- see
        # _scan_via_cache and _CACHE_TTL_S.
        self._cache_key: tuple[int, bool] | None = None
        self._cache_frames: list[Path] | None = None
        self._cache_at: float = 0.0

    def list_devices(self) -> list[ScannerDevice]:
        status = _http_json("GET", self._base_url, "/api/status")
        if not status.get("connected"):
            return []
        return [
            ScannerDevice(
                id=DEVICE_ID,
                vendor="Plustek",
                model="OpticFilm 135i",
                capabilities=_caps(),
            )
        ]

    def refresh_devices(self) -> list[ScannerDevice]:
        # The closest thing to an "I just changed what's loaded" signal this Protocol
        # offers, so the cache drops here even inside the TTL. See _scan_via_cache.
        self._cache_key = None
        self._cache_frames = None
        return self.list_devices()

    def open_session(self, device_id: str) -> ScannerSession:
        self._ensure_known_device(device_id)
        return PlusteckSession(self)

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: ProgressCb | None,
        cancel: threading.Event,
    ) -> ScanResult:
        self._ensure_known_device(device_id)
        if params.frame is None:
            raise RuntimeError(
                "plusteck addresses frames by ScanParams.frame (1-based) -- "
                "no free-standing window crop exists on a fixed tray-sweep transport."
            )
        frames = self._scan_via_cache(params, progress, cancel)
        index = params.frame - 1
        if not 0 <= index < len(frames):
            raise RuntimeError(f"frame {params.frame} out of range -- this tray load has {len(frames)} half-frame slots")
        rgb = _read_half_frame(frames[index])
        _safe_progress(progress, 1.0)
        return ScanResult(
            rgb=rgb,
            ir=None,
            dpi=int(params.dpi),
            device_model="Plustek OpticFilm 135i (plusteck)",
        )

    def eject(self, device_id: str) -> bool:
        del device_id
        return False

    def open_roll(self, device: ScannerDevice, *, dpi: int, film_format: str | None, film_type: str) -> "RollSession":
        del film_format, film_type  # not modeled yet -- see PlusteckRollSession's own docstring
        # Starting a fresh roll-preview pass is another real "the user is
        # (re)approaching this tray now" signal -- same reasoning as
        # refresh_devices(), see _scan_via_cache's own docstring.
        self._cache_key = None
        self._cache_frames = None
        from negpy.infrastructure.scanners.plusteck_roll import PlusteckRollSession

        return PlusteckRollSession(self._base_url, device, dpi=dpi)

    def detect_frames(self, device_id: str, *, film_format: str | None = None, film_type: str = "negative") -> int:
        """Half-frame slot count on the loaded tray.

        The other half of `roll_discovery`: ScanWorker.run_batch's whole-strip path calls
        this before it can call scan() per frame, and ScannerService.detect_frames
        otherwise reports 0 frames, failing the batch on a full tray.

        It runs plusteck's cheap Preview pass (PlusteckRollSession.discover) so a request
        with no frames named never pays for two physical tray passes. A fresh real-scan
        cache answers it instead, the same cache the per-frame scans then hit.
        """
        del film_format, film_type  # plusteck measures the physical layout itself; not modeled here (see open_roll)
        self._ensure_known_device(device_id)
        if self._cache_frames is not None and time.monotonic() - self._cache_at < _CACHE_TTL_S:
            return len(self._cache_frames)

        from negpy.infrastructure.scanners.plusteck_roll import PlusteckRollSession

        device = ScannerDevice(id=device_id, vendor="Plustek", model="OpticFilm 135i", capabilities=_caps())
        session = PlusteckRollSession(self._base_url, device, dpi=0)
        try:
            return session.discover(threading.Event())
        finally:
            session.close()

    def _scan_via_cache(self, params: ScanParams, progress: ProgressCb | None, cancel: threading.Event) -> list[Path]:
        """One physical pass serves every frame asked for at the same (dpi, capture_ir),
        for _CACHE_TTL_S.

        The cache belongs to the backend, not to PlusteckSession: ScanWorker.run_batch
        calls service.run_scan once per frame, straight through to backend.scan, and holds
        no session open across the loop. The backend is what lives for the whole batch, so
        caching here is what keeps a batch from repeating the physical tray pass per frame.

        Nothing here can tell "same settings, same roll" from "same settings, a different
        roll loaded since". The TTL and refresh_devices() dropping the cache narrow that
        window without closing it.
        """
        key = (int(params.dpi), bool(params.capture_ir))
        fresh = self._cache_frames is not None and time.monotonic() - self._cache_at < _CACHE_TTL_S
        if self._cache_key == key and fresh:
            assert self._cache_frames is not None
            _safe_progress(progress, 0.99, "Scanning")
            return self._cache_frames

        tier = _resolve_dpi_tier(params.dpi)
        # plusteck's mode is the capture colorspace, a different axis from NegPy's
        # ScanMode (the holder type, see params.py). ScanParams has no grayscale
        # capture, so this asks for Color, plus IR when requested.
        mode_value = "ir" if params.capture_ir else "color"
        session_id = f"negpy_{int(time.time() * 1000)}"
        response = _http_json(
            "POST",
            self._base_url,
            "/api/scanner/scan",
            {"session_id": session_id, "mode": mode_value, "resolution": tier},
        )
        if response.get("scan_state") == "error":
            raise RuntimeError(response.get("scan_error") or "scan request rejected")
        _wait_for_scan(self._base_url, session_id, progress, cancel)
        frames = _ensure_all_split(self._base_url, session_id)
        if not frames:
            raise RuntimeError("plusteck reported a completed scan with no slot images -- is film loaded?")
        self._cache_key = key
        self._cache_frames = frames
        self._cache_at = time.monotonic()
        return frames

    def _ensure_known_device(self, device_id: str) -> None:
        if device_id != DEVICE_ID:
            raise RuntimeError(f"Unknown or disconnected device: {device_id}")
