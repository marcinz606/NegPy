"""Tests for the vendor eject-at-completion primitive.

Nikon Scan ejects film at batch completion instead of leaving it parked
(the LS-5000 feeder auto-parks a few minutes after any session closes, and
a parked feeder reports frames: 0 until a power-cycle). These tests mock
the SANE device/module boundary — no hardware, no python-sane required —
and prove the eject primitive is capability-gated (skips cleanly when the
device has no usable 'eject' option) and fails loud when a *present*
option cannot actually be triggered. The runner-level fail-soft wrapping
(swallow-and-log so a failed eject never corrupts a completed archive) is
covered separately in tests/scanners/test_roll_scan_runner.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from negpy.infrastructure.scanners.sane_backend import SaneBackend, _find_eject_option
from negpy.services.scanning.service import ScannerService


@dataclass
class FakeOption:
    """Stand-in for python-sane's Option (only the fields the module reads)."""

    constraint: Any = None
    active: bool = True
    settable: bool = True

    def is_active(self) -> bool:
        return self.active

    def is_settable(self) -> bool:
        return self.settable


class FakeSaneDev:
    """Mimics python-sane's SaneDev for a coolscan3-like device.

    Setting an attribute that is not a known SANE option raises
    AttributeError, like python-sane does — so eject() must consult
    dev.opt before writing, exactly like the codebase's other option
    writers (dev.frame, dev.autofocus, ...).
    """

    def __init__(self, opt_map: dict[str, FakeOption], *, reject_trigger: bool = False) -> None:
        self._opt_map = opt_map
        self._reject_trigger = reject_trigger
        self.recorded: dict[str, object] = {}
        self.closed = False
        self.close_calls = 0

    @property
    def opt(self) -> dict[str, FakeOption]:
        return self._opt_map

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("_opt_map", "_reject_trigger", "recorded", "closed", "close_calls"):
            object.__setattr__(self, name, value)
            return
        if name not in self._opt_map:
            raise AttributeError(f"No such SANE option: {name}")
        if self._reject_trigger:
            raise OSError("device rejected eject")
        self.recorded[name] = value

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1


@dataclass
class FakeSaneModule:
    dev: FakeSaneDev | None = None
    open_error: Exception | None = None
    opened: list[str] = field(default_factory=list)

    def init(self) -> None:
        pass

    def open(self, device_id: str) -> FakeSaneDev:
        self.opened.append(device_id)
        if self.open_error is not None:
            raise self.open_error
        assert self.dev is not None
        return self.dev


def _make_backend(sane_module: FakeSaneModule) -> SaneBackend:
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = sane_module
    backend._sane_initialized = True
    backend._devices_cache = None
    return backend


COOLSCAN3_OPT_WITH_EJECT = {
    "frame": FakeOption(constraint=(1, 40, 1)),
    "eject": FakeOption(),
}
COOLSCAN3_OPT_NO_EJECT = {
    "frame": FakeOption(constraint=(1, 40, 1)),
}


class TestFindEjectOption:
    def test_finds_exact_eject_option(self) -> None:
        assert _find_eject_option(COOLSCAN3_OPT_WITH_EJECT) == "eject"

    def test_absent_when_no_eject_option(self) -> None:
        assert _find_eject_option(COOLSCAN3_OPT_NO_EJECT) is None

    def test_matches_hyphenated_spelling(self) -> None:
        assert _find_eject_option({"eject": FakeOption()}) == "eject"


class TestSaneBackendEject:
    def test_triggers_the_eject_option_and_returns_true(self) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT))
        backend = _make_backend(FakeSaneModule(dev))

        result = backend.eject("coolscan3:usb:libusb:001:007")

        assert result is True
        assert dev.recorded.get("eject") is True
        assert dev.closed is True

    def test_capability_gated_skip_when_device_has_no_eject_option(self) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_NO_EJECT))
        backend = _make_backend(FakeSaneModule(dev))

        result = backend.eject("coolscan3:usb:libusb:001:007")

        assert result is False
        assert dev.recorded == {}
        assert dev.closed is True  # session hygiene: still opened and closed cleanly

    @pytest.mark.parametrize(
        "broken_option",
        [FakeOption(active=False), FakeOption(settable=False)],
    )
    def test_capability_gated_skip_when_eject_option_is_inactive_or_unsettable(
        self,
        broken_option: FakeOption,
    ) -> None:
        opt = dict(COOLSCAN3_OPT_NO_EJECT)
        opt["eject"] = broken_option
        dev = FakeSaneDev(opt)
        backend = _make_backend(FakeSaneModule(dev))

        result = backend.eject("coolscan3:usb:libusb:001:007")

        assert result is False
        assert dev.recorded == {}
        assert dev.closed is True

    def test_raises_when_a_present_eject_option_rejects_the_trigger(self) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT), reject_trigger=True)
        backend = _make_backend(FakeSaneModule(dev))

        with pytest.raises(RuntimeError, match="Could not trigger eject.*device rejected eject"):
            backend.eject("coolscan3:usb:libusb:001:007")

        # A failed trigger must not leak an open device handle.
        assert dev.closed is True

    def test_raises_when_the_device_cannot_be_opened(self) -> None:
        module = FakeSaneModule(open_error=PermissionError("scanner is busy"))
        backend = _make_backend(module)

        with pytest.raises(RuntimeError, match="Failed to open scanner.*scanner is busy"):
            backend.eject("coolscan3:usb:libusb:001:007")

    def test_only_the_requested_device_is_opened(self) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT))
        module = FakeSaneModule(dev)
        backend = _make_backend(module)

        backend.eject("coolscan3:usb:libusb:001:007")

        assert module.opened == ["coolscan3:usb:libusb:001:007"]


class TestScannerServiceEject:
    def test_delegates_to_a_backend_that_supports_eject(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def eject(self, device_id: str) -> bool:
                self.calls.append(device_id)
                return True

        service = ScannerService()
        backend = FakeBackend()
        service._backend = backend

        assert service.eject("coolscan3:usb:test") is True
        assert backend.calls == ["coolscan3:usb:test"]

    def test_returns_false_cleanly_when_the_backend_has_no_eject_method(self) -> None:
        class BackendWithoutEject:
            pass

        service = ScannerService()
        service._backend = BackendWithoutEject()

        assert service.eject("coolscan3:usb:test") is False

    def test_propagates_a_genuine_eject_failure_from_the_backend(self) -> None:
        class FailingBackend:
            def eject(self, device_id: str) -> bool:
                raise RuntimeError("could not trigger eject")

        service = ScannerService()
        service._backend = FailingBackend()

        with pytest.raises(RuntimeError, match="could not trigger eject"):
            service.eject("coolscan3:usb:test")
