"""Dimage5400Backend: capability projection, parameter mapping, result shape, error typing."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("dimage5400")

import dimage5400.scanner as driver_scanner  # noqa: E402
import dimage5400.workflow as workflow  # noqa: E402
import usb.core  # noqa: E402
from dimage5400.models import ELITE_5400, ELITE_5400_II  # noqa: E402
from dimage5400.scanner import Exposure, HolderRequired  # noqa: E402
from dimage5400.transport import DeviceBusy  # noqa: E402

from negpy.infrastructure.scanners.base import TransientScanError  # noqa: E402
from negpy.infrastructure.scanners.dimage5400_backend import DEVICE_ID, Dimage5400Backend  # noqa: E402
from negpy.infrastructure.scanners.params import ScanMode, ScanParams  # noqa: E402


@dataclass
class _Holder:
    loaded: bool = True
    frames: int = 6


class _FakeScanner:
    instances: list["_FakeScanner"] = []
    model = ELITE_5400_II

    def __init__(self, holder: _Holder) -> None:
        self._holder = holder
        self.ejected = False
        self.closed = False
        _FakeScanner.instances.append(self)

    def holder(self) -> _Holder:
        return self._holder

    def eject(self) -> None:
        self.ejected = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake(monkeypatch):
    _FakeScanner.instances = []
    holder = _Holder()
    calls: list[dict] = []
    outcome: dict = {"error": None}

    monkeypatch.setattr(usb.core, "find", lambda **_: object())
    monkeypatch.setattr(driver_scanner.Scanner, "open", classmethod(lambda cls, **_: _FakeScanner(holder)))

    def scan_frame(scanner, frame, **kwargs):
        calls.append({"frame": frame, **kwargs})
        if outcome["error"] is not None:
            raise outcome["error"]
        rgb = np.full((4, 6, 3), 0.5, np.float32)
        ir = np.full((4, 6), 0.8, np.float32)
        exp = kwargs["exposure"] or Exposure(180, 200, 200, 100)
        return workflow.FrameScan(rgb=rgb, ir=ir, dpi=kwargs["dpi"], exposure=exp, focus=12, focus_curve=None)

    monkeypatch.setattr(workflow, "scan_frame", scan_frame)
    monkeypatch.setattr(workflow, "load_profile", lambda model, grain_dissolver=False: None)
    return holder, calls, outcome


def _scan(backend: Dimage5400Backend, params: ScanParams):
    return backend.scan(DEVICE_ID, params, lambda *_: None, threading.Event())


def test_capabilities_take_the_slot_count_from_the_holder(fake) -> None:
    holder, _calls, _ = fake
    holder.frames = 4
    caps = Dimage5400Backend().list_devices()[0].capabilities
    assert caps.adapter_frame_capacity == 4
    assert caps.supported_dpi == (675, 1350, 2700, 5400)
    assert caps.sources == (ScanMode.NEGATIVE, ScanMode.POSITIVE)
    assert caps.ir_channel and caps.autofocus and caps.auto_exposure and caps.can_eject


def test_no_device_on_the_bus_lists_nothing(fake, monkeypatch) -> None:
    monkeypatch.setattr(usb.core, "find", lambda **_: None)
    assert Dimage5400Backend().list_devices() == []


def test_params_map_onto_the_frame_scan(fake) -> None:
    _holder, calls, _ = fake
    params = ScanParams(dpi=3000, depth=16, capture_ir=True, frame=3, window=(0.1, 0.2, 0.9, 0.8), frame_offset_mm=1.5, autofocus=False)
    result = _scan(Dimage5400Backend(), params)
    call = calls[-1]
    assert call["frame"] == 3 and call["dpi"] == 2700
    assert call["window"] == (0.1, 0.2, 0.9, 0.8) and call["feed_offset_mm"] == 1.5
    assert call["autofocus"] is False and call["exposure"] == Exposure()
    assert result.rgb.dtype == np.uint16 and result.rgb.shape == (4, 6, 3)
    assert result.ir is not None and result.ir.shape == (4, 6) and result.ir_valid_mask.all()
    assert result.dpi == 2700


def test_auto_exposure_lets_the_driver_meter(fake) -> None:
    _holder, calls, _ = fake
    result = _scan(Dimage5400Backend(), ScanParams(dpi=675, depth=16, capture_ir=False, auto_exposure=True))
    assert calls[-1]["exposure"] is None
    assert result.exposures == {"red": 180, "green": 200, "blue": 200, "ir": 100}
    assert result.ir is None


def test_preview_depth_comes_back_8_bit(fake) -> None:
    result = _scan(Dimage5400Backend(), ScanParams(dpi=675, depth=8, capture_ir=False))
    assert result.rgb.dtype == np.uint8


def test_infrared_on_silver_film_is_refused(fake) -> None:
    with pytest.raises(RuntimeError, match="infrared"):
        _scan(Dimage5400Backend(), ScanParams(dpi=675, depth=16, capture_ir=True, film_type="mono"))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DeviceBusy("busy"), TransientScanError),
        (usb.core.USBError("pipe"), TransientScanError),
        (HolderRequired("no holder"), RuntimeError),
    ],
)
def test_driver_errors_are_typed_for_retry(fake, error, expected) -> None:
    _holder, _calls, outcome = fake
    outcome["error"] = error
    with pytest.raises(expected) as info:
        _scan(Dimage5400Backend(), ScanParams(dpi=675, depth=16, capture_ir=False))
    if expected is RuntimeError:
        assert not isinstance(info.value, TransientScanError)


def test_a_session_holds_one_handle_and_ejects(fake) -> None:
    backend = Dimage5400Backend()
    with backend.open_session(DEVICE_ID) as session:
        session.scan(ScanParams(dpi=675, depth=16, capture_ir=False), lambda *_: None, threading.Event())
        session.scan(ScanParams(dpi=675, depth=16, capture_ir=False), lambda *_: None, threading.Event())
        assert session.eject() is True
        held = _FakeScanner.instances[-1]
        with pytest.raises(RuntimeError, match="already held"):
            backend.open_session(DEVICE_ID)
    assert held.ejected and held.closed


def test_the_5400_ii_offers_no_grain_dissolver(fake) -> None:
    caps = Dimage5400Backend().list_devices()[0].capabilities
    assert caps.grain_dissolver is False
    with pytest.raises(RuntimeError, match="Grain Dissolver"):
        _scan(Dimage5400Backend(), ScanParams(dpi=675, depth=16, capture_ir=False, grain_dissolver=True))


def test_the_elite_5400_lists_under_its_own_name_and_passes_the_grain_dissolver(fake, monkeypatch) -> None:
    _holder, calls, _ = fake
    monkeypatch.setattr(_FakeScanner, "model", ELITE_5400)
    monkeypatch.setattr(
        "dimage5400.transport.usb.core.find",
        lambda idVendor, idProduct, **_: object() if (idVendor, idProduct) == (0x0686, 0x400E) else None,
    )
    device = Dimage5400Backend().list_devices()[0]
    assert (device.vendor, device.model) == ("Minolta", "DiMAGE Scan Elite 5400")
    assert device.capabilities.grain_dissolver
    assert device.capabilities.max_area_mm[0] < 25.0
    result = _scan(Dimage5400Backend(), ScanParams(dpi=1350, depth=16, capture_ir=False, grain_dissolver=True))
    assert calls[-1]["grain_dissolver"] is True
    assert result.device_model == "DiMAGE Scan Elite 5400"
