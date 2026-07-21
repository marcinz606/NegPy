"""Behaviours every ScannerBackend must have, independent of transport.

ScannerService and the Scan sidebar are written against these, not against SANE.
A backend that lives outside NegPy (a direct-USB Coolscan driver, say) can import
this module, append its own entry to BACKENDS, and run the suite against itself.

The harness contract is `_Factory`: given a scenario, hand back a backend and the
device id to address it with. Everything else here is transport-neutral.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Protocol

import numpy as np
import pytest

from negpy.infrastructure.scanners.base import ScannerDevice, TransientScanError
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from tests.scanners.test_sane_session import _DEV_ID, FakeDev, FakeSaneModule, _make_backend, _opt_map

_PARAMS = ScanParams(dpi=1000, depth=16, capture_ir=False)


class _Factory(Protocol):
    def __call__(
        self,
        *,
        scan_error: Exception | None = None,
        with_eject: bool = False,
        film: bool = True,
        progress_steps: int = 0,
    ) -> tuple[Any, str]:
        """(backend, device_id) for one scenario.

        `film=False` means the device advertises no film source — it must not
        survive list_devices(). `progress_steps` drives the read callback that
        many times. `scan_error` is raised by the transport mid-read.
        """
        ...


# ── the SANE entry ────────────────────────────────────────────────────────


class _ContractDev(FakeDev):
    """FakeDev that can drive the read callback and fail the read on cue."""

    def __init__(self, opt_map: dict[str, Any], scan_error: Exception | None, progress_steps: int) -> None:
        super().__init__(opt_map)
        object.__setattr__(self, "scan_error", scan_error)
        object.__setattr__(self, "progress_steps", progress_steps)

    def arr_snap(self, progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        if progress is not None:
            for i in range(1, self.progress_steps + 1):
                progress(i, self.progress_steps)
        if self.scan_error is not None:
            raise self.scan_error
        return self.frame_data


class _ModuleWithDevice(FakeSaneModule):
    def __init__(self, dev: FakeDev, device_id: str) -> None:
        super().__init__(dev=dev)
        self.device_id = device_id

    def get_devices(self) -> list[tuple[str, str, str]]:
        return [(self.device_id, "Nikon", "LS-50")]


def _sane_backend(
    *,
    scan_error: Exception | None = None,
    with_eject: bool = False,
    film: bool = True,
    progress_steps: int = 0,
) -> tuple[Any, str]:
    # A non-film transport is spelled as a flatbed id with no `source` option, so
    # nothing infers film sources for it.
    device_id = _DEV_ID if film else "epson2:libusb:001:002"
    dev = _ContractDev(_opt_map(eject=with_eject), scan_error, progress_steps)
    return _make_backend(_ModuleWithDevice(dev, device_id)), device_id


BACKENDS: list[tuple[str, _Factory]] = [("sane", _sane_backend)]

pytestmark = pytest.mark.parametrize("name,make_backend", BACKENDS)


# ── enumeration ───────────────────────────────────────────────────────────


def test_list_devices_returns_scanner_devices(name: str, make_backend: _Factory) -> None:
    backend, device_id = make_backend()
    devices = backend.list_devices()

    assert devices, f"{name}: expected the fake device to enumerate"
    assert all(isinstance(d, ScannerDevice) for d in devices)
    assert all(isinstance(d.id, str) and d.id for d in devices)


def test_devices_without_film_sources_are_dropped(name: str, make_backend: _Factory) -> None:
    """The sidebar disables scanning on an empty `sources`, so such devices never list."""
    backend, _ = make_backend(film=False)
    assert backend.list_devices() == []


def test_refresh_devices_re_enumerates(name: str, make_backend: _Factory) -> None:
    backend, _ = make_backend()
    assert backend.refresh_devices() == backend.list_devices()


# ── scanning ──────────────────────────────────────────────────────────────


def test_scan_returns_a_well_formed_result(name: str, make_backend: _Factory) -> None:
    backend, device_id = make_backend()
    result = backend.scan(device_id, _PARAMS, lambda _: None, threading.Event())

    assert isinstance(result, ScanResult)
    assert result.rgb.ndim == 3 and result.rgb.shape[2] == 3
    assert result.dpi == _PARAMS.dpi
    assert result.device_model


def test_scan_honours_a_pre_set_cancel(name: str, make_backend: _Factory) -> None:
    backend, device_id = make_backend()
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(Exception, match="[Cc]ancel"):
        backend.scan(device_id, _PARAMS, lambda _: None, cancel)


def test_progress_stays_within_the_unit_range(name: str, make_backend: _Factory) -> None:
    backend, device_id = make_backend(progress_steps=4)
    seen: list[float] = []

    backend.scan(device_id, _PARAMS, seen.append, threading.Event())

    assert seen, f"{name}: expected progress during the read"
    assert all(0.0 <= v <= 1.0 for v in seen), seen


def test_transport_glitches_are_typed_transient(name: str, make_backend: _Factory) -> None:
    """The service retries on type alone — it must not have to read messages."""
    backend, device_id = make_backend(scan_error=RuntimeError("Error during device I/O"))

    with pytest.raises(TransientScanError):
        backend.scan(device_id, _PARAMS, lambda _: None, threading.Event())


def test_real_errors_are_not_transient(name: str, make_backend: _Factory) -> None:
    backend, device_id = make_backend(scan_error=RuntimeError("Could not set frame=3"))

    with pytest.raises(Exception) as excinfo:
        backend.scan(device_id, _PARAMS, lambda _: None, threading.Event())
    assert not isinstance(excinfo.value, TransientScanError)


# ── eject ─────────────────────────────────────────────────────────────────


def test_eject_returns_false_when_the_device_has_no_eject_action(name: str, make_backend: _Factory) -> None:
    """A no-op, not an error — run_batch ejects unconditionally after a clean batch."""
    backend, device_id = make_backend(with_eject=False)
    assert backend.eject(device_id) is False


# ── sessions ──────────────────────────────────────────────────────────────


def test_open_session_yields_a_session_shaped_object(name: str, make_backend: _Factory) -> None:
    backend, device_id = make_backend()
    session = backend.open_session(device_id)
    try:
        assert session.device_id == device_id
        for method in ("scan", "eject", "close", "__enter__", "__exit__"):
            assert callable(getattr(session, method, None)), f"{name}: session lacks {method}"
    finally:
        session.close()


def test_session_scans_on_the_held_handle(name: str, make_backend: _Factory) -> None:
    backend, device_id = make_backend()
    with backend.open_session(device_id) as session:
        result = session.scan(_PARAMS, lambda _: None, threading.Event())
    assert isinstance(result, ScanResult)


def test_session_close_is_idempotent(name: str, make_backend: _Factory) -> None:
    backend, device_id = make_backend()
    session = backend.open_session(device_id)
    session.close()
    session.close()
