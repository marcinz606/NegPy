"""Regression coverage for the two-pass IR progress-bar fix: the 'source' IR
strategy (Plustek/genesys) runs a second, separate arr_snap() for IR. It used to
report no progress at all, so the bar hit 100% after the RGB pass and sat still.
Each physical pass now reports its own independent 0->1 (no phase-splitting, no
conditionality on whether IR was requested)."""

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
}


class TwoPassFakeSaneDev:
    """Plustek/genesys-style: switches `source` for a second full arr_snap() pass."""

    _INTERNAL = ("recorded", "true_frame", "closed", "cancelled", "rgb_frame", "ir_frame")

    def __init__(self, rgb_frame: np.ndarray, ir_frame: np.ndarray) -> None:
        object.__setattr__(self, "recorded", {"source": "Negative"})
        object.__setattr__(self, "rgb_frame", rgb_frame)
        object.__setattr__(self, "ir_frame", ir_frame)
        object.__setattr__(self, "true_frame", rgb_frame)
        object.__setattr__(self, "closed", False)
        object.__setattr__(self, "cancelled", False)

    @property
    def opt(self):
        return _SOURCE_OPT

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._INTERNAL:
            object.__setattr__(self, name, value)
            return
        if name not in _SOURCE_OPT:
            raise AttributeError(f"No such SANE option: {name}")
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
