"""The NegPy adapter over the minscan package: request mapping and result shape."""

from __future__ import annotations

# The package import is intentionally guarded by importorskip below.
# ruff: noqa: E402

import threading

import numpy as np
import pytest

minscan = pytest.importorskip("minscan")

from negpy.infrastructure.scanners import minscan_backend
from negpy.infrastructure.scanners.base import TransientScanError
from negpy.infrastructure.scanners.minscan_backend import MinscanSession
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult


def _params(**overrides) -> ScanParams:
    values = {
        "dpi": 3200,
        "depth": 16,
        "capture_ir": False,
        "autofocus": False,
        "auto_exposure": False,
        "film_format": "69",
        "film_type": "negative",
    }
    values.update(overrides)
    return ScanParams(**values)


class _FakeSession:
    def __init__(self, path: str) -> None:
        self.path = path
        self.configured: list = []
        self.scans: list = []
        self.closed = False
        self.raise_on_scan: Exception | None = None

    def configure(self, media, film, *, format_code=0, glassless=False, ev=0.0) -> None:
        self.configured.append((media, film, format_code, glassless, ev))

    def scan(self, **kwargs) -> minscan.Frame:
        self.scans.append(kwargs)
        if self.raise_on_scan is not None:
            raise self.raise_on_scan
        return minscan.Frame(
            rgb=np.zeros((4, 6, 3), dtype=np.uint16),
            ir=None,
            dpi=kwargs["dpi"],
            planes=3,
            rows=4,
            cols=6,
        )

    def eject(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(minscan_backend.minscan, "Session", _FakeSession)
    return _FakeSession


def _session(default_media: int = 1) -> tuple[MinscanSession, _FakeSession]:
    session = MinscanSession("minscan:/dev/sg1", default_media)
    return session, session._session


def test_configure_maps_carrier_and_is_idempotent(fake) -> None:
    session, raw = _session(1)
    params = _params()

    session.scan(params, lambda *_a: None, threading.Event())
    session.scan(params, lambda *_a: None, threading.Event())

    assert raw.configured == [(minscan.Media.FORMAT_120, minscan.Film.COLOR_NEG, 0x20, False, 0.0)]


def test_35mm_frame_maps_to_group_and_slot(fake) -> None:
    session, raw = _session(0)
    params = _params(film_format="135", frame=1)

    session.scan(params, lambda *_a: None, threading.Event())

    call = raw.scans[0]
    assert (call["frame_group"], call["slot"]) == (1, 1)


def test_preview_is_a_prescan_without_focus_or_ir(fake) -> None:
    session, raw = _session(0)
    params = _params(film_format="135", frame=2)

    session.preview(params, lambda *_a: None, threading.Event())

    call = raw.scans[0]
    assert call["prescan"] is True
    assert call["autofocus"] is False
    assert call["capture_ir"] is False
    assert call["ice"] is False
    assert (call["frame_group"], call["slot"]) == (1, 0)


def test_scan_returns_a_result_and_applies_the_window(fake) -> None:
    session, _raw = _session(1)
    params = _params(window=(1 / 6, 0.25, 4 / 6, 0.75))

    result = session.scan(params, lambda *_a: None, threading.Event())

    assert isinstance(result, ScanResult)
    assert result.rgb.shape == (2, 3, 3)


def test_transient_and_cancelled_errors_map_to_the_backend_contract(fake) -> None:
    session, raw = _session(1)
    raw.raise_on_scan = minscan.MinscanTransientError(-6, "busy")
    with pytest.raises(TransientScanError):
        session.scan(_params(), lambda *_a: None, threading.Event())

    raw.raise_on_scan = minscan.MinscanCancelled(-1, "cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        session.scan(_params(), lambda *_a: None, threading.Event())


def test_probe_maps_media_and_failures(monkeypatch) -> None:
    backend = minscan_backend.MinscanBackend.__new__(minscan_backend.MinscanBackend)

    monkeypatch.setattr(minscan_backend.minscan, "probe", lambda _path: minscan.Media.FORMAT_120)
    assert backend._probe("/dev/sg1") == 1

    monkeypatch.setattr(minscan_backend.minscan, "probe", lambda _path: None)
    assert backend._probe("/dev/sg1") == -1

    def fail(_path):
        raise minscan.MinscanError(-2, "I/O")

    monkeypatch.setattr(minscan_backend.minscan, "probe", fail)
    assert backend._probe("/dev/sg1") is None


def test_list_devices_uses_autodetect(monkeypatch) -> None:
    backend = minscan_backend.MinscanBackend.__new__(minscan_backend.MinscanBackend)
    backend._media_by_device = {}
    monkeypatch.setattr(
        minscan_backend.minscan, "devices", lambda: [("/dev/sg9", minscan.Media.FORMAT_35MM)]
    )

    devices = backend.list_devices()

    assert [d.id for d in devices] == ["minscan:/dev/sg9"]
    assert devices[0].capabilities.ir_channel is True


def test_open_session_resolves_auto(monkeypatch) -> None:
    backend = minscan_backend.MinscanBackend.__new__(minscan_backend.MinscanBackend)
    backend._media_by_device = {}
    monkeypatch.setattr(minscan_backend.minscan, "detect", lambda: "/dev/sg9")
    monkeypatch.setattr(minscan_backend.minscan, "probe", lambda _path: minscan.Media.FORMAT_35MM)

    class _FakeSession:
        def __init__(self, device_id, media):
            self.device_id = device_id
            self.media = media

    monkeypatch.setattr(minscan_backend, "MinscanSession", _FakeSession)

    session = backend.open_session("minscan:auto")

    assert session.device_id == "minscan:/dev/sg9"
    assert session.media == 0
