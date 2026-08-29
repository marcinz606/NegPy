"""Regression coverage for a flatbed+TPU SANE device (Epson V-series etc.) that defaults to
reflective Flatbed scanning and never switches to its Transparency Unit on its own.

Three bugs, one root cause: `caps.sources` (Negative/Positive/Transparency) only ever gated
the Scan button — nothing wrote `source`/`film_type` to the device, and geometry was probed
before any such switch, so both the reported scan area and the actual scan ran against the
wrong (Flatbed) reference. Fixed by:

1. `_scan_on_device` resolving and setting `source`/`film_type` before every scan.
2. `_detect_caps` doing the same switch before reading geometry, so `max_area_mm` (and the
   crop-window coordinate space built from it) reflects the source that will actually be used.
3. Setting `mode`/`depth`/`resolution` *after* the source switch, since switching source can
   re-range or reset those options on this backend (confirmed for geometry; resolution is the
   same class of bug and silently discarded a valid depth/DPI request if set first).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.sane_backend import SaneBackend, _resolve_transparency_source

_DEV_ID = "epkowa:interpreter:001:006"


@dataclass
class FakeOption:
    constraint: Any = None
    desc: str = ""
    unit: Any = None
    active: bool = True
    settable: bool = True

    def is_active(self) -> bool:
        return self.active

    def is_settable(self) -> bool:
        return self.settable


# Two distinct option sets, keyed by which `source` is currently selected — the real epkowa
# backend re-ranges tl_x/br_x/tl_y/br_y (and, per the bug this guards against, resolution) the
# moment `source` changes, so a faithful fake must do the same rather than share one static map.
def _flatbed_opts() -> dict[str, FakeOption]:
    return {
        "source": FakeOption(constraint=["Flatbed", "Transparency Unit"]),
        "film_type": FakeOption(constraint=["Positive Film", "Negative Film"], active=False),
        "depth": FakeOption(constraint=[8, 16]),
        "resolution": FakeOption(constraint=[75, 150, 300, 600, 1200, 1600]),
        "tl_x": FakeOption(constraint=(0.0, 215.9, 0.0), unit=3),
        "tl_y": FakeOption(constraint=(0.0, 297.18, 0.0), unit=3),
        "br_x": FakeOption(constraint=(0.0, 215.9, 0.0), unit=3),
        "br_y": FakeOption(constraint=(0.0, 297.18, 0.0), unit=3),
    }


def _transparency_opts() -> dict[str, FakeOption]:
    opts = _flatbed_opts()
    opts["source"] = FakeOption(constraint=["Flatbed", "Transparency Unit"])
    opts["film_type"] = FakeOption(constraint=["Positive Film", "Negative Film"], active=True)
    # The real bug: the Transparency Unit's usable area is a different physical size, not a
    # simple crop of the flatbed's — 68.6x237.0mm vs 215.9x297.2mm on the reporting device.
    opts["tl_x"] = FakeOption(constraint=(0.0, 68.58, 0.0), unit=3)
    opts["br_x"] = FakeOption(constraint=(0.0, 68.58, 0.0), unit=3)
    opts["tl_y"] = FakeOption(constraint=(0.0, 236.98, 0.0), unit=3)
    opts["br_y"] = FakeOption(constraint=(0.0, 236.98, 0.0), unit=3)
    return opts


class FakeDev:
    """python-sane SaneDev stand-in that models source-dependent option re-ranging and
    records every option write in order, so tests can assert both value and sequence."""

    _INTERNAL = ("opt_map", "writes", "frame_data", "cancel_calls", "close_calls")

    def __init__(self) -> None:
        object.__setattr__(self, "opt_map", _flatbed_opts())
        object.__setattr__(self, "writes", [])
        object.__setattr__(self, "frame_data", np.zeros((4, 4, 3), dtype=np.uint8))
        object.__setattr__(self, "cancel_calls", 0)
        object.__setattr__(self, "close_calls", 0)

    @property
    def opt(self) -> dict[str, FakeOption]:
        return self.opt_map

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.opt_map:
            raise AttributeError(f"No such SANE option: {name}")
        self.writes.append((name, value))
        if name == "source":
            # Re-range: swap in the option set the real device would report post-switch.
            object.__setattr__(self, "opt_map", _transparency_opts() if value == "Transparency Unit" else _flatbed_opts())
            # New option objects mean any previously-set resolution is gone until re-applied —
            # this is exactly the reset the ordering fix guards against.

    def start(self) -> None:
        pass

    def get_parameters(self):
        h, w, _ = self.frame_data.shape
        return ("color", 1, (w, h), 8, w * 3)

    def arr_snap(self, progress=None) -> np.ndarray:
        return self.frame_data

    def cancel(self) -> None:
        object.__setattr__(self, "cancel_calls", self.cancel_calls + 1)

    def close(self) -> None:
        object.__setattr__(self, "close_calls", self.close_calls + 1)


@dataclass
class FakeSaneModule:
    dev: FakeDev
    opened: list[str] = field(default_factory=list)

    def init(self) -> None:
        pass

    def open(self, device_id: str) -> FakeDev:
        self.opened.append(device_id)
        return self.dev

    def get_devices(self) -> list[tuple[str, str, str]]:
        return [(_DEV_ID, "Epson", "Perfection V500")]


def _make_backend(module: FakeSaneModule) -> SaneBackend:
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = module
    backend._sane_initialized = True
    backend._devices_cache = None
    backend._id_remap = {}
    backend._active_sessions = {}
    backend._session_lock = threading.Lock()
    return backend


class TestCapabilityDetectionUsesTransparencyGeometry:
    """_detect_caps must switch source before reading tl_x/br_x/etc., or max_area_mm (and the
    crop-window coordinate space the UI builds from it) reports the wrong physical area."""

    def test_max_area_reflects_transparency_unit_not_flatbed(self) -> None:
        dev = FakeDev()
        backend = _make_backend(FakeSaneModule(dev))

        caps = backend._detect_caps(dev, _DEV_ID)

        assert caps.max_area_mm == (68.58, 236.98)

    def test_switches_source_as_part_of_the_probe(self) -> None:
        dev = FakeDev()
        backend = _make_backend(FakeSaneModule(dev))

        backend._detect_caps(dev, _DEV_ID)

        assert ("source", "Transparency Unit") in dev.writes


class TestScanOrdersSourceBeforeResolution:
    """The real bug: source switch re-ranges resolution's option, so setting resolution first
    (the original code order) applied it against an option object the switch then discarded."""

    def test_source_is_set_before_resolution(self) -> None:
        dev = FakeDev()
        backend = _make_backend(FakeSaneModule(dev))
        params = ScanParams(dpi=1600, depth=8, capture_ir=False)

        backend.scan(_DEV_ID, params, None, threading.Event())

        names = [name for name, _ in dev.writes]
        assert names.index("source") < names.index("resolution")

    def test_requested_resolution_survives_the_source_switch(self) -> None:
        dev = FakeDev()
        backend = _make_backend(FakeSaneModule(dev))
        params = ScanParams(dpi=1600, depth=8, capture_ir=False)

        backend.scan(_DEV_ID, params, None, threading.Event())

        # The last recorded resolution write is what the scan actually ran with.
        resolution_writes = [value for name, value in dev.writes if name == "resolution"]
        assert resolution_writes[-1] == 1600

    def test_film_type_is_set_after_source_when_it_only_activates_post_switch(self) -> None:
        """film_type is `active=False` under Flatbed in this fixture (mirrors the real device,
        where it reads back <unreadable> until the Transparency Unit is selected)."""
        dev = FakeDev()
        backend = _make_backend(FakeSaneModule(dev))
        params = ScanParams(dpi=1600, depth=8, capture_ir=False, film_type="negative")

        backend.scan(_DEV_ID, params, None, threading.Event())

        assert ("film_type", "Negative Film") in dev.writes
        names = [name for name, _ in dev.writes]
        assert names.index("source") < names.index("film_type")


class TestResolveTransparencySourceIntegration:
    """Sanity check against the fixture's realistic constraint list, complementing the pure
    unit tests in test_capabilities.py."""

    def test_resolves_against_flatbed_and_transparency_fixture(self) -> None:
        dev = FakeDev()
        assert _resolve_transparency_source(dev.opt) == "Transparency Unit"
