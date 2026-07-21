"""Exclusive scanner sessions: the batch/roll handover seam.

A roll workflow must own the scanner for a whole strip — one continuous SANE
open, per-frame scans, one release at the end (SANE hardware is single-open,
and the Coolscan feeder auto-parks after any session closes mid-roll). While a
session is open the backend must get out of the way: scan()/eject() on the
held device are refused and list_devices() reuses the cached entry instead of
probing (a probe would open the held device).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest

from negpy.infrastructure.scanners import sane_backend
from negpy.infrastructure.scanners.base import ScanMode, ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.sane_backend import SaneBackend, SaneSession
from negpy.services.scanning.service import ScannerService

_DEV_ID = "coolscan3:usb:libusb:003:006"
_FRESH_ID = "coolscan3:usb:libusb:003:007"
_PARAMS = ScanParams(dpi=1000, depth=16, capture_ir=False)


@dataclass
class FakeOption:
    constraint: Any = None
    active: bool = True
    settable: bool = True

    def is_active(self) -> bool:
        return self.active

    def is_settable(self) -> bool:
        return self.settable


def _opt_map(*, eject: bool = False) -> dict[str, FakeOption]:
    opt = {
        "depth": FakeOption(constraint=[8, 16]),
        "resolution": FakeOption(constraint=[1000, 2000]),
        "frame": FakeOption(constraint=(1, 40, 1)),
    }
    if eject:
        opt["eject"] = FakeOption()
    return opt


class FakeDev:
    """Minimal python-sane SaneDev stand-in for plain RGB scans."""

    _INTERNAL = ("opt_map", "recorded", "frame_data", "cancel_calls", "close_calls")

    def __init__(self, opt_map: dict[str, FakeOption] | None = None) -> None:
        object.__setattr__(self, "opt_map", _opt_map() if opt_map is None else opt_map)
        object.__setattr__(self, "recorded", {})
        object.__setattr__(self, "frame_data", np.zeros((6, 5, 3), dtype=np.uint16))
        object.__setattr__(self, "cancel_calls", 0)
        object.__setattr__(self, "close_calls", 0)

    @property
    def opt(self) -> dict[str, FakeOption]:
        return self.opt_map

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.opt_map:
            raise AttributeError(f"No such SANE option: {name}")
        self.recorded[name] = value

    def start(self) -> None:
        pass

    def get_parameters(self):
        h, w, _ = self.frame_data.shape
        return ("color", 1, (w, h), 16, w * 3 * 2)

    def arr_snap(self, progress=None) -> np.ndarray:
        return self.frame_data

    def cancel(self) -> None:
        object.__setattr__(self, "cancel_calls", self.cancel_calls + 1)

    def close(self) -> None:
        object.__setattr__(self, "close_calls", self.close_calls + 1)


@dataclass
class FakeSaneModule:
    dev: FakeDev
    fail_ids: tuple[str, ...] = ()
    opened: list[str] = field(default_factory=list)

    def init(self) -> None:
        pass

    def open(self, device_id: str) -> FakeDev:
        self.opened.append(device_id)
        if device_id in self.fail_ids:
            raise RuntimeError("Invalid argument")
        return self.dev

    def get_devices(self) -> list[tuple[str, str, str]]:
        return [(_DEV_ID, "Nikon", "LS-50")]


def _sd(device_id: str) -> ScannerDevice:
    caps = ScannerCapabilities(
        ir_channel=False,
        supported_dpi=(4000,),
        supported_depths=(8, 16),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(25.0, 38.0),
        can_eject=True,
    )
    return ScannerDevice(id=device_id, vendor="Nikon", model="LS-50", capabilities=caps)


def _make_backend(module: FakeSaneModule, cache: list[ScannerDevice] | None = None) -> SaneBackend:
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = module
    backend._sane_initialized = True
    backend._devices_cache = cache
    backend._id_remap = {}
    backend._active_sessions = {}
    backend._session_lock = threading.Lock()
    return backend


class FakeRun:
    def __init__(self, *, on_call: Callable[[], None] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._on_call = on_call

    def __call__(self, args: list[str], **kwargs: Any):
        self.calls.append(args)
        if self._on_call is not None:
            self._on_call()

        class _Completed:
            returncode = 0
            stderr = ""

        return _Completed()


class TestSaneSession:
    def test_scans_share_one_open_and_release_once(self) -> None:
        module = FakeSaneModule(FakeDev())
        backend = _make_backend(module)

        session = backend.open_session(_DEV_ID)
        r1 = session.scan(_PARAMS, None, threading.Event())
        r2 = session.scan(_PARAMS, None, threading.Event())

        assert module.opened == [_DEV_ID]  # opened once for both frames
        assert module.dev.close_calls == 0  # held between frames
        assert module.dev.cancel_calls == 2  # sane_cancel per completed frame
        assert r1.rgb.shape == (6, 5, 3) and r2.rgb.shape == (6, 5, 3)

        session.close()
        assert module.dev.close_calls == 1

    def test_scan_after_close_raises(self) -> None:
        backend = _make_backend(FakeSaneModule(FakeDev()))
        session = backend.open_session(_DEV_ID)
        session.close()

        with pytest.raises(RuntimeError, match="closed"):
            session.scan(_PARAMS, None, threading.Event())

    def test_close_is_idempotent(self) -> None:
        module = FakeSaneModule(FakeDev())
        backend = _make_backend(module)
        session = backend.open_session(_DEV_ID)

        session.close()
        session.close()

        assert module.dev.close_calls == 1

    def test_context_manager_releases_the_device(self) -> None:
        module = FakeSaneModule(FakeDev())
        backend = _make_backend(module)

        with backend.open_session(_DEV_ID) as session:
            assert isinstance(session, SaneSession)

        # Released: a one-shot scan may open the device again.
        result = backend.scan(_DEV_ID, _PARAMS, None, threading.Event())
        assert result.rgb.shape == (6, 5, 3)

    def test_backend_scan_refuses_a_held_device(self) -> None:
        module = FakeSaneModule(FakeDev())
        backend = _make_backend(module)
        backend.open_session(_DEV_ID)

        with pytest.raises(RuntimeError, match="held by an active session"):
            backend.scan(_DEV_ID, _PARAMS, None, threading.Event())

        assert module.opened == [_DEV_ID]  # never tried a second open

    def test_backend_eject_refuses_a_held_device(self) -> None:
        module = FakeSaneModule(FakeDev())
        backend = _make_backend(module)
        backend.open_session(_DEV_ID)

        with pytest.raises(RuntimeError, match="held by an active session"):
            backend.eject(_DEV_ID)

    def test_second_session_on_the_same_device_raises(self) -> None:
        backend = _make_backend(FakeSaneModule(FakeDev()))
        backend.open_session(_DEV_ID)

        with pytest.raises(RuntimeError, match="already held"):
            backend.open_session(_DEV_ID)

    def test_guard_matches_the_remapped_id_after_reenumeration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The stale id fails to open (USB re-enumeration); _open_device remaps
        # to the fresh id. The guard must hold for BOTH ids afterwards.
        module = FakeSaneModule(FakeDev(), fail_ids=(_DEV_ID,))
        backend = _make_backend(module, cache=[_sd(_DEV_ID)])
        monkeypatch.setattr(backend, "refresh_devices", lambda: [_sd(_FRESH_ID)])

        session = backend.open_session(_DEV_ID)

        assert session.opened_id == _FRESH_ID
        with pytest.raises(RuntimeError, match="held by an active session"):
            backend.scan(_FRESH_ID, _PARAMS, None, threading.Event())

    def test_list_devices_reuses_the_cached_entry_for_a_held_device(self) -> None:
        cached = _sd(_DEV_ID)
        module = FakeSaneModule(FakeDev())
        backend = _make_backend(module, cache=[cached])
        backend.open_session(_DEV_ID)

        devices = backend.refresh_devices()

        assert devices == [cached]
        assert module.opened == [_DEV_ID]  # only the session's open — no probe


class TestSessionEject:
    def _held(self, monkeypatch: pytest.MonkeyPatch, *, eject: bool):
        module = FakeSaneModule(FakeDev(_opt_map(eject=eject)))
        backend = _make_backend(module)
        session = backend.open_session(_DEV_ID)
        observed: dict[str, int] = {}
        run = FakeRun(on_call=lambda: observed.__setitem__("close_calls_at_press", module.dev.close_calls))
        monkeypatch.setattr(sane_backend.subprocess, "run", run)
        return backend, module, session, run, observed

    def test_eject_closes_the_handle_before_scanimage_and_ends_the_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, module, session, run, observed = self._held(monkeypatch, eject=True)

        assert session.eject() is True
        assert run.calls == [["scanimage", "-d", _DEV_ID, "--eject"]]
        assert observed["close_calls_at_press"] == 1  # single-open hardware
        assert session.closed
        assert backend._active_sessions == {}

    def test_eject_is_capability_gated_but_still_releases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, module, session, run, _ = self._held(monkeypatch, eject=False)

        assert session.eject() is False
        assert run.calls == []
        assert session.closed
        assert backend._active_sessions == {}

    def test_eject_on_a_closed_session_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, session, _, _ = self._held(monkeypatch, eject=True)
        session.close()

        with pytest.raises(RuntimeError, match="closed"):
            session.eject()


class TestServiceOpenSession:
    def test_delegates_to_a_backend_with_session_support(self) -> None:
        sentinel = object()

        class FakeBackend:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def open_session(self, device_id: str):
                self.calls.append(device_id)
                return sentinel

        service = ScannerService()
        backend = FakeBackend()
        service._backend = backend

        assert service.open_session(_DEV_ID) is sentinel
        assert backend.calls == [_DEV_ID]
