"""Shared fixtures for tests/roll/: a fake `coolscanpy` module.

`negpy.infrastructure.roll.coolscanpy_roll` and `negpy.services.roll.service`
treat coolscanpy as optional and import it lazily, so these tests never need
the real package installed -- `fake_coolscanpy` injects a minimal stand-in
module into `sys.modules` instead, exercising the same `import coolscanpy`
path the production code takes.
"""

from __future__ import annotations

import dataclasses
import importlib.machinery
import sys
import types
from typing import Any

import pytest


class FakePyCoolscanError(Exception):
    """Stand-in for coolscanpy.PyCoolscanError."""


class FakeDeviceNotFound(FakePyCoolscanError):
    pass


class FakeManualReviewRequired(FakePyCoolscanError):
    def __init__(self, message: str, *, slot: int) -> None:
        super().__init__(message)
        self.slot = slot


class FakeSafeStopRequested(FakePyCoolscanError):
    pass


class FakeMaterial:
    COLOR_NEGATIVE = "color-negative"
    BLACK_AND_WHITE_NEGATIVE = "black-and-white-negative"


@dataclasses.dataclass(frozen=True)
class FakeThumbnail:
    slot: int
    image: Any
    boundary_rows: tuple
    spacing_offset: int
    needs_approval: bool
    warnings: tuple = ()


@dataclasses.dataclass(frozen=True)
class FakeReceipt:
    """Deliberately smaller than the real coolscanpy Receipt -- write_frame()
    only needs *a* dataclass instance to round-trip through dataclasses.asdict()."""

    version: int
    slot: int
    dpi: int
    depth: int
    device_id: str
    transport_smear_verdict: str


@dataclasses.dataclass(frozen=True)
class FakeFrame:
    slot: int
    rgb: Any
    ir: Any
    ir_validity: Any
    receipt: Any


@dataclasses.dataclass(frozen=True)
class FakeProgress:
    """Stand-in for coolscanpy.types.Progress, same field names."""

    stage: str
    slot: Any
    index: int
    total: int
    fraction: float
    message: str


class FakeRoll:
    """Scripted in-memory stand-in for `coolscanpy.Roll`."""

    def __init__(self, thumbnails=None, frames=None, raise_on=None) -> None:
        self._thumbnails = list(thumbnails or [])
        self._frames = {frame.slot: frame for frame in (frames or [])}
        self._raise_on = raise_on or {}
        self.approved: list[int] = []
        self.spacing_offsets: dict[int, int] = {}
        self.safe_stop_called = False
        self.closed = False

    def preview(self, slots=None, *, on_progress=None):
        if "preview" in self._raise_on:
            raise self._raise_on["preview"]
        if on_progress is not None:
            on_progress(FakeProgress(stage="preview", slot=None, index=0, total=1, fraction=0.0, message="reading whole-roll transport index"))
        wanted = None if slots is None else set(slots)
        result = [t for t in self._thumbnails if wanted is None or t.slot in wanted]
        if on_progress is not None:
            on_progress(FakeProgress(stage="preview", slot=None, index=1, total=1, fraction=1.0, message="preview complete"))
        return result

    def set_spacing_offset(self, slot, offset_rows) -> None:
        self.spacing_offsets[slot] = offset_rows

    def approve(self, slot) -> None:
        if "approve" in self._raise_on:
            raise self._raise_on["approve"]
        self.approved.append(slot)

    def needs_approval(self, slot) -> bool:
        return slot in self._raise_on.get("needs_approval_slots", ())

    def scan_many(self, slots, *, on_progress=None):
        ordered = list(slots)
        for i, slot in enumerate(ordered):
            error = self._raise_on.get("scan_many_slots", {}).get(slot)
            if error is not None:
                raise error
            if on_progress is not None:
                on_progress(FakeProgress(stage="fine-scan", slot=slot, index=i, total=len(ordered), fraction=1.0, message=f"slot {slot} complete"))
            yield self._frames[slot]

    def safe_stop(self) -> None:
        self.safe_stop_called = True

    def close(self) -> None:
        self.closed = True


class FakeDevice:
    def __init__(self, roll) -> None:
        self._roll = roll
        self.closed = False
        self.roll_called_with = None

    def roll(self, *, material=None):
        self.roll_called_with = material
        return self._roll

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_coolscanpy(monkeypatch):
    """Install a fake `coolscanpy` module; returns an object to script it.

    Usage::

        fake_coolscanpy.state["open_device"] = fake_coolscanpy.Device(fake_coolscanpy.Roll())
        handle = coolscanpy_roll.open_roll()
    """
    module = types.ModuleType("coolscanpy")
    # importlib.util.find_spec() raises ValueError on a sys.modules entry
    # with no __spec__, so give the fake one like a real imported module has.
    module.__spec__ = importlib.machinery.ModuleSpec("coolscanpy", loader=None)

    state: dict[str, Any] = {"devices": [], "open_device": None, "open_error": None}

    def fake_get_devices(local_only: bool = False):
        return state["devices"]

    def fake_open(devname: str):
        if state["open_error"] is not None:
            raise state["open_error"]
        return state["open_device"]

    module.get_devices = fake_get_devices
    module.open = fake_open
    module.Material = FakeMaterial
    module.PyCoolscanError = FakePyCoolscanError
    module.DeviceNotFound = FakeDeviceNotFound
    module.ManualReviewRequired = FakeManualReviewRequired
    module.SafeStopRequested = FakeSafeStopRequested

    monkeypatch.setitem(sys.modules, "coolscanpy", module)

    return types.SimpleNamespace(
        module=module,
        state=state,
        Device=FakeDevice,
        Roll=FakeRoll,
        Thumbnail=FakeThumbnail,
        Frame=FakeFrame,
        Receipt=FakeReceipt,
    )
