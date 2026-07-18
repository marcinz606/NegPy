"""Self-healing open across USB re-enumeration.

A mid-session USB re-enumeration changes the libusb address embedded in the SANE
device id (observed on an LS-50: ...:003:006 -> ...:003:007), so the cached id
goes stale and sane.open() raises "Invalid argument". SaneBackend._open_device
re-lists, remaps to the same physical scanner, retries once, and remembers the
remap so later opens skip straight to the fresh id.
"""

from __future__ import annotations

import threading

import pytest

from negpy.infrastructure.scanners.base import ScanMode, ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.sane_backend import SaneBackend

_STALE = "coolscan3:usb:libusb:003:006"
_FRESH = "coolscan3:usb:libusb:003:007"


class FakeSaneModule:
    """open() succeeds only for ids in `ok`; anything else raises like a stale id."""

    def __init__(self, ok: dict[str, object]) -> None:
        self.ok = ok
        self.opened: list[str] = []

    def init(self) -> None:
        pass

    def open(self, device_id: str) -> object:
        self.opened.append(device_id)
        if device_id in self.ok:
            return self.ok[device_id]
        raise RuntimeError("Invalid argument")


def _sd(device_id: str, *, vendor: str = "Nikon", model: str = "LS-50") -> ScannerDevice:
    caps = ScannerCapabilities(
        ir_channel=False,
        supported_dpi=(4000,),
        supported_depths=(8,),
        sources=(ScanMode.NEGATIVE,),
        max_area_mm=(25.0, 38.0),
        adapter_frame_capacity=6,
        can_eject=True,
    )
    return ScannerDevice(id=device_id, vendor=vendor, model=model, capabilities=caps)


def _backend(module: FakeSaneModule, cache: list[ScannerDevice]) -> SaneBackend:
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = module
    backend._sane_initialized = True
    backend._devices_cache = cache
    backend._id_remap = {}
    backend._active_sessions = {}
    backend._session_lock = threading.Lock()
    return backend


def test_opens_directly_when_the_id_is_still_valid() -> None:
    dev = object()
    module = FakeSaneModule({_STALE: dev})
    backend = _backend(module, [_sd(_STALE)])

    got, opened_id = backend._open_device(_STALE)

    assert got is dev
    assert opened_id == _STALE
    assert backend._id_remap == {}  # no re-list needed


def test_remaps_by_vendor_and_model_after_reenumeration(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = object()
    module = FakeSaneModule({_FRESH: dev})  # stale id now raises
    backend = _backend(module, [_sd(_STALE)])
    monkeypatch.setattr(backend, "refresh_devices", lambda: [_sd(_FRESH)])

    got, opened_id = backend._open_device(_STALE)

    assert got is dev
    assert opened_id == _FRESH
    assert backend._id_remap[_STALE] == _FRESH


def test_remaps_by_prefix_when_the_stale_device_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = object()
    module = FakeSaneModule({_FRESH: dev})
    backend = _backend(module, [])  # empty cache -> no vendor/model to match
    monkeypatch.setattr(backend, "refresh_devices", lambda: [_sd(_FRESH)])

    got, opened_id = backend._open_device(_STALE)

    assert got is dev
    assert opened_id == _FRESH


def test_remembered_remap_skips_relisting(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = object()
    module = FakeSaneModule({_FRESH: dev})
    backend = _backend(module, [_sd(_STALE)])
    backend._id_remap = {_STALE: _FRESH}
    calls: list[int] = []
    monkeypatch.setattr(backend, "refresh_devices", lambda: calls.append(1) or [])

    got, opened_id = backend._open_device(_STALE)

    assert got is dev
    assert opened_id == _FRESH
    assert calls == []  # went straight to the fresh id


def test_reraises_when_no_unambiguous_match(monkeypatch: pytest.MonkeyPatch) -> None:
    module = FakeSaneModule({_FRESH: object()})
    backend = _backend(module, [_sd(_STALE)])
    # Two devices share the prefix and neither matches vendor/model exactly enough.
    monkeypatch.setattr(
        backend,
        "refresh_devices",
        lambda: [_sd("coolscan3:usb:libusb:003:007", model="LS-50"), _sd("coolscan3:usb:libusb:003:009", model="LS-50")],
    )

    with pytest.raises(RuntimeError, match="Invalid argument"):
        backend._open_device(_STALE)


def test_reraises_when_device_is_truly_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    module = FakeSaneModule({})  # nothing opens
    backend = _backend(module, [_sd(_STALE)])
    monkeypatch.setattr(backend, "refresh_devices", lambda: [])

    with pytest.raises(RuntimeError, match="Invalid argument"):
        backend._open_device(_STALE)
