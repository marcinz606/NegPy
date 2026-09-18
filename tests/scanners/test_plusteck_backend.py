"""How PlusteckBackend talks to plusteck's own local HTTP service.

plusteck itself already validates the real scan protocol against real hardware
(this repo has no opinion on that, see plusteck's own docs/HARDWARE.md) -- what's
tested here is this adapter's own logic: dpi-tier mapping, transient-vs-real error
classification, and above all PlusteckBackend._scan_via_cache -- the one physical
pass that has to serve every frame in a batch. That cache HAS to live on the
backend, not on PlusteckSession: NegPy's real batch-scan path (ScanWorker.run_batch
-> ScannerService.run_scan) calls backend.scan() once per frame directly, never
through one held-open ScannerSession (confirmed by reading scan_worker.py's
run_batch). The tests below therefore drive PlusteckBackend.scan() the same way --
separate calls, no shared session -- specifically to catch a regression back to a
session-scoped cache that would silently do nothing there.

Everything is mocked at the _http_json boundary -- no real HTTP, no real plusteck
process required to run these.
"""

import threading

import numpy as np
import pytest
import tifffile

from negpy.infrastructure.scanners.base import ScannerUnavailable, TransientScanError
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.plusteck_backend import (
    DEVICE_ID,
    PlusteckBackend,
    _http_json,
    _looks_transient,
    _resolve_dpi_tier,
)

_BASE = "http://127.0.0.1:48213"


def _params(**overrides):
    base = dict(dpi=1800, depth=16, capture_ir=False, frame=1)
    base.update(overrides)
    return ScanParams(**base)


def _make_backend(monkeypatch, fake_http_json) -> PlusteckBackend:
    """A PlusteckBackend whose _http_json is fully mocked, including the
    constructor's own reachability check."""
    monkeypatch.setattr("negpy.infrastructure.scanners.plusteck_backend._http_json", fake_http_json)
    return PlusteckBackend()


class _FakePlusteck:
    """A minimal in-memory stand-in for plusteck's own HTTP API: one POST
    /api/scanner/scan writes a fresh session's slot files (one scan0001 with
    an _a/_b pair) and reports done immediately on the first status poll --
    enough to drive PlusteckBackend's own orchestration without a real
    plusteck process or real hardware."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.session_id: str | None = None
        self.calls: list[tuple[str, str]] = []
        self.pixels = np.zeros((4, 4, 3), dtype=np.uint16)

    def __call__(self, method, base_url, path, body=None):
        self.calls.append((method, path))
        if path == "/api/status":
            return {
                "connected": True,
                "scan_state": "done" if self.session_id else "idle",
                "scan_session_id": self.session_id,
                "split_output_dir": str(self.tmp_path),
            }
        if path in ("/api/scanner/scan", "/api/scanner/scan/preview"):
            self.session_id = body["session_id"]
            session_dir = self.tmp_path / self.session_id
            # exist_ok: the session_id is millisecond-based (see
            # PlusteckBackend._scan_via_cache) -- two mocked scans in one
            # test can legitimately land in the same millisecond, unlike a
            # real multi-minute physical scan.
            session_dir.mkdir(exist_ok=True)
            tifffile.imwrite(session_dir / "scan0001_a.tiff", self.pixels)
            tifffile.imwrite(session_dir / "scan0001_b.tiff", self.pixels)
            return {"scan_state": "scanning", "scan_session_id": self.session_id}
        if path.endswith("/scans"):
            return [{"scan_id": "scan0001", "src": "/media/raw/x/scan0001.tiff"}]
        if path.endswith("/split"):
            return {}
        raise AssertionError(f"unexpected call: {method} {path}")

    @property
    def scan_trigger_count(self) -> int:
        return len([c for c in self.calls if c == ("POST", "/api/scanner/scan")])

    @property
    def preview_trigger_count(self) -> int:
        return len([c for c in self.calls if c == ("POST", "/api/scanner/scan/preview")])


# ── dpi mapping ──────────────────────────────────────────────────────────


def test_known_dpi_maps_to_plusteck_tier():
    assert _resolve_dpi_tier(1800) == "standard_1800"
    assert _resolve_dpi_tier(600) == "low_600"


def test_unknown_dpi_fails_fast():
    with pytest.raises(RuntimeError, match="Unsupported dpi"):
        _resolve_dpi_tier(4800)


# ── transient classification ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    ["lost connection mid-scan (event: ctrl)", "device wedged", "USB error: timed out"],
)
def test_transport_trouble_is_transient(message):
    assert _looks_transient(message)


@pytest.mark.parametrize("message", ["no film loaded", "unsupported dpi=9999"])
def test_real_errors_are_not_transient(message):
    assert not _looks_transient(message)


# ── construction / device listing ────────────────────────────────────────


def test_unreachable_service_raises_scanner_unavailable(monkeypatch):
    def fake_http_json(method, base_url, path, body=None):
        raise ScannerUnavailable(f"plusteck backend not reachable at {base_url}")

    monkeypatch.setattr("negpy.infrastructure.scanners.plusteck_backend._http_json", fake_http_json)
    with pytest.raises(ScannerUnavailable):
        PlusteckBackend()


def test_response_timeout_is_transient_not_a_bare_crash(monkeypatch):
    """Reached (the connection succeeds) but no answer within _HTTP_TIMEOUT_S -- most
    likely plusteck's own device lock held by someone else's real I/O (see
    scanner_service.py's own comment on why that can take a while). urllib only wraps a
    timeout as URLError while sending the request, not while waiting for the response, so
    without this the caller would see a bare, unclassified TimeoutError instead of
    something ScannerService's existing retry logic knows to act on."""

    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(TransientScanError, match="did not respond"):
        _http_json("GET", _BASE, "/api/status")


def test_list_devices_empty_when_scanner_not_connected(monkeypatch):
    backend = _make_backend(monkeypatch, lambda *a, **k: {"connected": False})
    assert backend.list_devices() == []


def test_list_devices_one_device_when_connected(monkeypatch):
    backend = _make_backend(monkeypatch, lambda *a, **k: {"connected": True})
    devices = backend.list_devices()
    assert len(devices) == 1
    assert devices[0].id == DEVICE_ID
    assert devices[0].capabilities.sources  # never an empty-sources device, see base.py's own contract


# ── one physical pass per batch (the real ScanWorker.run_batch shape) ────


def test_backend_scan_only_triggers_one_physical_pass_across_separate_calls(tmp_path, monkeypatch):
    """The regression this whole test module exists to catch: NegPy's real
    batch-scan path calls PlusteckBackend.scan() once per frame with NO
    shared session object (see module docstring) -- so it's this, not a
    session-held cache, that has to avoid one physical tray pass per frame."""
    fake = _FakePlusteck(tmp_path)
    backend = _make_backend(monkeypatch, fake)
    cancel = threading.Event()

    result1 = backend.scan(DEVICE_ID, _params(frame=1), progress=None, cancel=cancel)
    result2 = backend.scan(DEVICE_ID, _params(frame=2), progress=None, cancel=cancel)

    assert fake.scan_trigger_count == 1, f"expected exactly one physical scan, got {fake.calls}"
    assert result1.rgb.shape == (4, 4, 3)
    assert result2.rgb.shape == (4, 4, 3)
    assert result1.device_model == "Plustek OpticFilm 135i (plusteck)"


def test_open_session_shares_the_same_backend_cache(tmp_path, monkeypatch):
    """A caller that DOES hold a session open must see the same one-pass
    behavior as the direct backend.scan() path -- PlusteckSession is a thin
    delegator, not a second cache."""
    fake = _FakePlusteck(tmp_path)
    backend = _make_backend(monkeypatch, fake)
    cancel = threading.Event()

    with backend.open_session(DEVICE_ID) as session:
        session.scan(_params(frame=1), progress=None, cancel=cancel)
    # A later direct backend.scan() call (the real run_batch shape) after the
    # session closed must still hit the cache from the session's own pass.
    backend.scan(DEVICE_ID, _params(frame=2), progress=None, cancel=cancel)

    assert fake.scan_trigger_count == 1, f"expected exactly one physical scan, got {fake.calls}"


def test_different_dpi_forces_a_fresh_pass(tmp_path, monkeypatch):
    """The cache key is (dpi, capture_ir) -- a genuinely different request
    must not be served from a stale pass at another resolution."""
    fake = _FakePlusteck(tmp_path)
    backend = _make_backend(monkeypatch, fake)
    cancel = threading.Event()

    backend.scan(DEVICE_ID, _params(dpi=1800, frame=1), progress=None, cancel=cancel)
    backend.scan(DEVICE_ID, _params(dpi=600, frame=1), progress=None, cancel=cancel)

    assert fake.scan_trigger_count == 2, f"expected two physical scans (different dpi), got {fake.calls}"


def test_refresh_devices_drops_the_cache(tmp_path, monkeypatch):
    """refresh_devices() is treated as a real 'user is looking at this
    device again' signal (e.g. reopening the scanner panel between rolls) --
    see PlusteckBackend.refresh_devices's own comment."""
    fake = _FakePlusteck(tmp_path)
    backend = _make_backend(monkeypatch, fake)
    cancel = threading.Event()

    backend.scan(DEVICE_ID, _params(frame=1), progress=None, cancel=cancel)
    backend.refresh_devices()
    backend.scan(DEVICE_ID, _params(frame=1), progress=None, cancel=cancel)

    assert fake.scan_trigger_count == 2, f"expected a fresh pass after refresh_devices(), got {fake.calls}"


# ── error handling ────────────────────────────────────────────────────────


# ── whole-strip frame count (ScanWorker.run_batch's "no frames named" path) ─


def test_detect_frames_runs_a_preview_pass_and_reports_the_slot_count(tmp_path, monkeypatch):
    """The regression this test exists to catch: PlusteckBackend previously had no
    detect_frames at all, so ScannerService.detect_frames's getattr(..., None)
    fallback silently returned 0 and every whole-strip Scan (no frames named,
    the default) failed immediately with "No frames were detected on the loaded
    film" -- see ScanWorker._whole_strip."""
    fake = _FakePlusteck(tmp_path)
    backend = _make_backend(monkeypatch, fake)

    count = backend.detect_frames(DEVICE_ID)

    assert count == 2  # one tray slot, split into its "_a"/"_b" half-frames
    assert fake.preview_trigger_count == 1
    assert fake.scan_trigger_count == 0  # counting frames must not pay for a full-quality pass


def test_detect_frames_reuses_a_fresh_real_scan_cache_without_a_preview_pass(tmp_path, monkeypatch):
    """A real scan just run (e.g. a prior batch, still within _CACHE_TTL_S)
    already answers "how many frames" for free -- no need to run Preview too."""
    fake = _FakePlusteck(tmp_path)
    backend = _make_backend(monkeypatch, fake)
    backend.scan(DEVICE_ID, _params(frame=1), progress=None, cancel=threading.Event())

    count = backend.detect_frames(DEVICE_ID)

    assert count == 2
    assert fake.scan_trigger_count == 1
    assert fake.preview_trigger_count == 0


def test_scan_without_frame_fails_fast(monkeypatch):
    """No free-standing window crop exists on this transport -- a caller
    must address a frame by index."""
    backend = _make_backend(monkeypatch, lambda *a, **k: {"connected": True})
    with pytest.raises(RuntimeError, match="ScanParams.frame"):
        backend.scan(DEVICE_ID, _params(frame=None), progress=None, cancel=threading.Event())


def test_scan_error_state_is_reported(monkeypatch):
    def fake_http_json(method, base_url, path, body=None):
        if path == "/api/status":
            return {"connected": True}
        if path == "/api/scanner/scan":
            return {"scan_state": "error", "scan_error": "no film loaded"}
        raise AssertionError(f"unexpected call: {method} {path}")

    backend = _make_backend(monkeypatch, fake_http_json)
    with pytest.raises(RuntimeError, match="no film loaded"):
        backend.scan(DEVICE_ID, _params(frame=1), progress=None, cancel=threading.Event())


def test_transient_scan_error_is_typed(monkeypatch):
    """A failure that only shows up once polling starts (not on the initial
    202-Accepted response) must still be classified, not just raised bare --
    and _wait_for_scan's own session_id check (skip a status snapshot that
    isn't this scan's yet) must be satisfied, or this would poll for the
    full _SCAN_MAX_WAIT_S before ever seeing the error."""
    state: dict[str, str] = {}

    def fake_http_json(method, base_url, path, body=None):
        if path == "/api/status" and "session_id" not in state:
            return {"connected": True}
        if path == "/api/scanner/scan":
            state["session_id"] = body["session_id"]
            return {"scan_state": "scanning", "scan_session_id": state["session_id"]}
        if path == "/api/status":
            return {
                "scan_state": "error",
                "scan_session_id": state["session_id"],
                "scan_error": "lost connection mid-scan (event: ctrl)",
            }
        raise AssertionError(f"unexpected call: {method} {path}")

    backend = _make_backend(monkeypatch, fake_http_json)
    with pytest.raises(TransientScanError):
        backend.scan(DEVICE_ID, _params(frame=1), progress=None, cancel=threading.Event())
