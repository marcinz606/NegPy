"""Regression coverage for the two-pass IR progress-bar fix: the 'source' IR
strategy (Plustek/genesys) runs a second, separate arr_snap() for IR. It used to
report no progress at all, so the bar hit 100% after the RGB pass and sat still.
Each physical pass now reports its own independent 0->1 (no phase-splitting, no
conditionality on whether IR was requested)."""

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.sane_backend import SaneBackend


@dataclass
class FakeOption:
    constraint: Any = None
    desc: str = ""
    active: bool = True
    settable: bool = True

    def is_active(self) -> bool:
        return self.active

    def is_settable(self) -> bool:
        return self.settable


_SOURCE_OPT = {
    "source": FakeOption(constraint=["Negative", "Negative (IR)"]),
    "depth": FakeOption(constraint=[8, 16]),
    "resolution": FakeOption(constraint=[300, 600, 1200]),
    "tl_x": FakeOption(constraint=(0.0, 36.33, 0.0)),
    "tl_y": FakeOption(constraint=(0.0, 25.0, 0.0)),
    "br_x": FakeOption(constraint=(0.0, 36.33, 0.0)),
    "br_y": FakeOption(constraint=(0.0, 25.0, 0.0)),
}

_GEOMETRY_NAMES = ("tl_x", "tl_y", "br_x", "br_y")


class TwoPassFakeSaneDev:
    """Plustek/genesys-style: switches `source` for a second full arr_snap() pass.

    Also models a real backend quirk: switching `source` can reset/re-range the
    device's geometry options, so a scan window applied only before the RGB pass
    would silently vanish for the IR pass. reset_geometry_on_source_switch=True
    reproduces that, to prove the window gets re-applied for both passes.
    """

    _INTERNAL = (
        "recorded",
        "true_frame",
        "closed",
        "cancelled",
        "rgb_frame",
        "ir_frame",
        "reset_geometry_on_source_switch",
        "geometry_at_snap",
    )

    def __init__(self, rgb_frame: np.ndarray, ir_frame: np.ndarray, *, reset_geometry_on_source_switch: bool = False) -> None:
        object.__setattr__(self, "recorded", {"source": "Negative"})
        object.__setattr__(self, "rgb_frame", rgb_frame)
        object.__setattr__(self, "ir_frame", ir_frame)
        object.__setattr__(self, "true_frame", rgb_frame)
        object.__setattr__(self, "closed", False)
        object.__setattr__(self, "cancelled", False)
        object.__setattr__(self, "reset_geometry_on_source_switch", reset_geometry_on_source_switch)
        # One geometry snapshot per arr_snap() call — what a real backend would
        # actually have scanned with for that pass, independent of what a later
        # source-switch-back (RGB->IR->RGB) subsequently wipes.
        object.__setattr__(self, "geometry_at_snap", [])

    @property
    def opt(self):
        return _SOURCE_OPT

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._INTERNAL:
            object.__setattr__(self, name, value)
            return
        if name not in _SOURCE_OPT:
            raise AttributeError(f"No such SANE option: {name}")
        if name == "source" and self.reset_geometry_on_source_switch:
            for geo in _GEOMETRY_NAMES:
                self.recorded.pop(geo, None)
        self.recorded[name] = value

    def __getattr__(self, name: str) -> Any:
        if name in _SOURCE_OPT:
            return self.recorded.get(name)
        raise AttributeError(f"No readable SANE option: {name}")

    def start(self) -> None:
        pass

    def get_parameters(self):
        h, w = self.rgb_frame.shape[:2]
        return ("color", 1, (w, h), 16, w * 3 * 2)

    def arr_snap(self, progress=None) -> np.ndarray:
        self.geometry_at_snap.append({geo: self.recorded.get(geo) for geo in _GEOMETRY_NAMES})
        frame = self.ir_frame if self.recorded.get("source") == "Negative (IR)" else self.rgb_frame
        if progress is not None:
            h = frame.shape[0]
            for i in range(1, h + 1):
                progress(i, h)
        return frame

    def cancel(self) -> None:
        object.__setattr__(self, "cancelled", True)

    def close(self) -> None:
        object.__setattr__(self, "closed", True)


@dataclass
class FakeSaneModule:
    dev: TwoPassFakeSaneDev
    opened: list = field(default_factory=list)

    def open(self, device_id: str) -> TwoPassFakeSaneDev:
        self.opened.append(device_id)
        return self.dev


def _make_backend(dev: TwoPassFakeSaneDev) -> SaneBackend:
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = FakeSaneModule(dev)
    backend._sane_initialized = True
    backend._devices_cache = None
    backend._id_remap = {}
    backend._active_sessions = {}
    backend._session_lock = threading.Lock()
    return backend


def test_source_strategy_scan_reports_0_to_1_on_each_pass_independently():
    h, w = 20, 8
    rgb = np.zeros((h, w, 3), dtype=np.uint16)
    ir = np.zeros((h, w), dtype=np.uint16)
    dev = TwoPassFakeSaneDev(rgb, ir)
    backend = _make_backend(dev)

    calls: list[float] = []
    backend.scan(
        "plustek:libusb:001:008",
        ScanParams(dpi=300, depth=16, capture_ir=True),
        calls.append,
        threading.Event(),
    )

    # The RGB pass reaches 1.0 on its own...
    rgb_end = calls.index(1.0)
    assert rgb_end > 0
    # ...then the IR pass explicitly resets to 0.0 rather than continuing from
    # wherever the RGB pass left off (that was the whole point: no shared bar).
    assert calls[rgb_end + 1] == 0.0
    # ...and climbs back to 1.0 on its own by the end (a harmless duplicate 1.0
    # from the scan's own final "done" call may follow — not asserted on here).
    assert calls[-1] == 1.0
    assert 1.0 in calls[rgb_end + 1 :]
    # Each pass individually is monotonically non-decreasing.
    first_pass, second_pass = calls[: rgb_end + 1], calls[rgb_end + 1 :]
    assert first_pass == sorted(first_pass)
    assert second_pass == sorted(second_pass)


def test_no_ir_scan_still_reaches_100_percent():
    h, w = 10, 6
    rgb = np.zeros((h, w, 3), dtype=np.uint16)
    dev = TwoPassFakeSaneDev(rgb, rgb)
    backend = _make_backend(dev)

    calls: list[float] = []
    backend.scan(
        "plustek:libusb:001:008",
        ScanParams(dpi=300, depth=16, capture_ir=False),
        calls.append,
        threading.Event(),
    )

    assert calls[0] == 0.0
    assert calls[-1] == 1.0
    assert calls.count(0.0) == 1  # single pass — never resets back to 0 mid-scan
    assert calls == sorted(calls)


def test_scan_window_survives_a_source_switch_that_resets_geometry():
    """A device that clears tl_x/tl_y/br_x/br_y on `source =` (plausible on real
    hardware — different sources can report different scannable areas) must not
    silently lose the crop window for the IR pass: RGB and IR would then come
    back different pixel sizes, and the loader drops the "mismatched" IR plane."""
    h, w = 10, 6
    rgb = np.zeros((h, w, 3), dtype=np.uint16)
    ir = np.zeros((h, w), dtype=np.uint16)
    dev = TwoPassFakeSaneDev(rgb, ir, reset_geometry_on_source_switch=True)
    backend = _make_backend(dev)

    backend.scan(
        "plustek:libusb:001:008",
        ScanParams(dpi=300, depth=16, capture_ir=True, window=(0.1, 0.2, 0.9, 0.8)),
        None,
        threading.Event(),
    )

    # Two arr_snap() calls: RGB, then IR after the source switch wiped geometry.
    # The IR-pass snapshot must show the window re-applied, not the wiped state.
    rgb_snap, ir_snap = dev.geometry_at_snap
    expected = {"tl_x": 0.1 * 36.33, "tl_y": 0.2 * 25.0, "br_x": 0.9 * 36.33, "br_y": 0.8 * 25.0}
    for geo, value in expected.items():
        assert rgb_snap[geo] == pytest.approx(value)
        assert ir_snap[geo] == pytest.approx(value)
