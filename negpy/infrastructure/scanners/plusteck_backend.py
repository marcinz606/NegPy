# SPDX-License-Identifier: GPL-3.0-or-later
"""NegPy ``ScannerBackend`` adapter for plusteck (Plustek OpticFilm 135i).

Unlike pyopticfilm/nkscan/pieusb, this backend never touches USB itself. It
drives plusteck's own already-running local FastAPI service (see
plusteck/backend/src/plusteck/api/main.py, default http://127.0.0.1:48213)
over plain HTTP and then reads the resulting files straight off the local
disk -- both processes run on the same machine, so there's no need to
proxy pixel data through a request body. plusteck owns all the real
hardware-validated protocol logic (the golden-sequence replay, resolution
tiers, IR toggle, per-slot boundary auto-detection); this adapter only
translates NegPy's ScannerBackend/ScannerSession/RollSession protocols
into calls against it. Start it with plusteck's own `backend/run_backend.sh`
before using this backend.

Why not a real USB driver in-process, the way pyopticfilm/nkscan/pieusb
are: plusteck's backend already serializes all device access behind its
own lock, and there is exactly one physical device -- two processes both
trying to hold the raw USB handle would just fight each other. Going
through its existing HTTP API instead means this adapter can never be out
of sync with the actual validated scan sequence, and gets every future
fix to it for free.

Why one physical pass covers a whole tray, not one frame at a time (the
shape every other ScannerBackend here assumes): the OpticFilm 135i driver
has no way to address "just frame 3" -- the golden sequence is a replay of
one fixed captured USB trace covering a full physical carriage sweep
across every loaded tray slot. See PlusteckBackend._scan_via_cache's own
docstring for how this is reconciled with the one-scan()-per-frame shape
ScanWorker.run_batch expects, without re-scanning the whole tray once per
frame -- and specifically why that cache has to live on PlusteckBackend,
not on PlusteckSession.
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

#: Fixed port -- mirrors plusteck/backend/src/plusteck/api/main.py's own PORT
#: constant. There is exactly one of these locally, not a discoverable fleet
#: of them, so a fixed default (rather than mDNS/config-file discovery) is
#: the honest amount of engineering for what this actually is.
DEFAULT_BASE_URL = "http://127.0.0.1:48213"

#: This adapter's one synthetic device id -- plusteck exposes no serial
#: number/bus address of its own over HTTP, and there is only ever one
#: locally reachable instance, so "reachable or not" is the whole of device
#: enumeration here.
DEVICE_ID = "plusteck:opticfilm135i:local"

_HTTP_TIMEOUT_S = 10.0
_POLL_INTERVAL_S = 1.0
#: Generous hard ceiling on how long one physical pass may run before this
#: adapter gives up waiting -- plusteck's own UI documents "3+ minutes for a
#: full tray at Standard 1800 dpi; other tiers vary", so this is several
#: times the slowest case seen, not a tight bound.
_SCAN_MAX_WAIT_S = 1800.0

_TRANSIENT_ERROR_HINTS = ("lost connection", "wedged", "timed out", "timeout", "usb error")

#: NegPy dpi -> plusteck's own named resolution tiers (ResolutionTier in
#: backend/src/plusteck/usb/base.py). Only Color + STANDARD_1800 is fully
#: validated end-to-end on the plusteck side as of this writing; every other
#: tier is either a real capture not yet visually confirmed, or a disclosed
#: inference -- plusteck's own "Known limitations" panel is the place that
#: caveat lives, deliberately not re-litigated a second time here.
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
        # Distinct from URLError above: the connection was accepted (the process is up),
        # it just didn't answer within _HTTP_TIMEOUT_S -- urllib only wraps a timeout as
        # URLError while sending the request, not while waiting for the response, so this
        # would otherwise escape as a bare, unclassified TimeoutError. Most likely cause on
        # plusteck's own model: its single device lock is held by something else's real
        # device I/O for the whole duration (a scan, a register read/write -- see
        # scanner_service.py's own comment), which resolves on its own; worth ScannerService's
        # existing retry, not a hard failure.
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
        # Standard 1800 dpi's own pixel width (2592px, confirmed real) implies
        # ~36.6mm -- a standard 35mm frame including sprocket margin. Height
        # is a full tray sweep, not one frame's, and isn't independently
        # confirmed here -- reusing the width as a conservative square bound
        # rather than asserting an unverified number for it.
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


#: How long a completed physical pass stays usable for a later scan() call at
#: the same (dpi, capture_ir) before this adapter insists on a fresh one.
#: Bounds the blast radius of the staleness risk described on
#: PlusteckBackend._scan_via_cache's own docstring: long enough to cover one
#: realistic batch (plusteck's own UI: "3+ minutes for a full tray at
#: Standard 1800 dpi"), short enough that walking away and coming back with a
#: different roll doesn't silently serve the old one for the rest of the day.
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

    plusteck's live per-byte percent only exists on its WebSocket event
    stream (see main.py's /api/events); this adapter deliberately stays on
    plain polling HTTP to avoid adding a websocket-client dependency to
    NegPy for what is, at this stage, a coarse phase indicator anyway --
    genuinely fractional progress is a reasonable follow-up, not a
    correctness requirement.
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

    Owns the one real fine-scan cache (see `_scan_via_cache`) -- it, not
    PlusteckSession, is what NegPy's own ScannerService keeps alive for a
    whole app session (`ScannerService._get_backend()` builds one instance
    and reuses it), and what ScanWorker.run_batch's real per-frame loop
    actually calls into (via `service.run_scan` -> `backend.scan`, never
    through a held-open ScannerSession -- confirmed by reading
    negpy/desktop/workers/scan_worker.py's run_batch, 2026-09-03).
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
        # A natural point to drop a stale cache even inside the TTL: a
        # refresh is the closest thing to an explicit "I just changed what's
        # loaded" signal this Protocol offers (e.g. the scanner panel being
        # reopened between rolls). Not a guarantee by itself -- see
        # _scan_via_cache's own docstring -- but a real, principled signal,
        # not just a timer.
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

        The other half of `roll_discovery` (see `ScannerCapabilities.roll_discovery`'s
        own docstring): `ScanWorker.run_batch`'s whole-strip path (no frames named)
        calls this via `ScannerService.detect_frames` before it can call `scan()` once
        per frame at all -- without it, `ScannerService.detect_frames`'s own
        `getattr(..., None)` fallback silently reports 0 frames and the batch fails
        immediately with "No frames were detected on the loaded film", however full the
        tray actually is.

        Runs plusteck's own cheap Preview pass (`PlusteckRollSession.discover`) rather
        than a full-quality `_scan_via_cache` one: a request with no frames named must
        not pay for two real physical passes (one to count them, one to read them) on a
        transport that can only sweep the whole tray at a time. An already-fresh
        real-scan cache answers this for free instead -- the batch's own per-frame
        scans right after this call would hit that same cache anyway.
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
        """One physical pass serves every frame NegPy asks for at the same
        (dpi, capture_ir), for _CACHE_TTL_S, without a second physical scan.

        This has to live here rather than on PlusteckSession: NegPy's real
        batch-scan path (ScanWorker.run_batch) calls `service.run_scan`
        once PER FRAME, and that goes straight to `backend.scan` -- it never
        holds one ScannerSession open across the loop the way a session-
        scoped cache would need. PlusteckBackend, by contrast, genuinely is
        the thing that stays alive for the whole batch (see this class's
        own docstring), so caching here is what actually avoids re-running
        the whole ~3+ minute physical tray pass once per frame -- the
        transport still can't address a single frame (see module
        docstring), a batch just no longer notices.

        Known limitation, disclosed rather than silently assumed away: nothing
        here can tell "same settings, still the same roll" apart from "same
        settings, a DIFFERENT roll loaded since." refresh_devices() dropping
        the cache and the TTL both narrow this, neither closes it completely
        -- this genuinely needs exercising against a real batch run + a real
        roll change to know how much it matters in practice.
        """
        key = (int(params.dpi), bool(params.capture_ir))
        fresh = self._cache_frames is not None and time.monotonic() - self._cache_at < _CACHE_TTL_S
        if self._cache_key == key and fresh:
            assert self._cache_frames is not None
            _safe_progress(progress, 0.99, "Scanning")
            return self._cache_frames

        tier = _resolve_dpi_tier(params.dpi)
        # plusteck's mode is its capture colorspace (color/gray/ir), a
        # different axis entirely from NegPy's own ScanMode (the film
        # holder type -- negative/positive/transparency, see params.py).
        # This adapter only ever requests Color, +IR when asked -- NegPy's
        # ScanParams has no "grayscale capture" concept to map Gray from.
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
