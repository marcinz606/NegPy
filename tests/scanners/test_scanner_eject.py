"""Tests for the vendor eject-at-completion primitive.

Nikon Scan ejects film at completion instead of leaving it parked (the LS-5000
feeder auto-parks a few minutes after any session closes, and a parked feeder
reports frames: 0 until a power-cycle). python-sane cannot activate a
SANE_TYPE_BUTTON (setattr, set_option and set_auto_option all raise on the real
library — verified on an LS-50), so eject is pressed by shelling out to
`scanimage --eject`, the C-level path that actually works. These tests mock the
SANE device/module boundary and the scanimage subprocess — no hardware — and
prove the primitive is capability-gated (skips cleanly when the device has no
usable 'eject' option), presses via scanimage once capability is confirmed,
tolerates scanimage's spurious post-eject "out of documents" exit, and fails
loud on a genuine scanimage error.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from negpy.infrastructure.scanners import sane_backend
from negpy.infrastructure.scanners.sane_backend import SaneBackend, _find_eject_option
from negpy.services.scanning.service import ScannerService


@dataclass
class FakeOption:
    """Stand-in for python-sane's Option (only the fields the module reads)."""

    constraint: Any = None
    active: bool = True
    settable: bool = True
    index: int = 0
    is_button: bool = False

    def is_active(self) -> bool:
        return self.active

    def is_settable(self) -> bool:
        return self.settable


class FakeCDev:
    """Stand-in for the C object python-sane exposes as SaneDev.dev.

    set_option is recorded only so tests can assert eject() never routes the
    button press through it — the real python-sane rejects button set_option.
    """

    def __init__(self) -> None:
        self.set_option_calls: list[tuple[int, object]] = []

    def set_option(self, index: int, value: object) -> None:
        self.set_option_calls.append((index, value))


class FakeSaneDev:
    """Mimics python-sane's SaneDev for a coolscan3-like device (detection only)."""

    def __init__(self, opt_map: dict[str, FakeOption]) -> None:
        self._opt_map = opt_map
        self.recorded: dict[str, object] = {}
        self.closed = False
        self.close_calls = 0
        self.dev = FakeCDev()

    @property
    def opt(self) -> dict[str, FakeOption]:
        return self._opt_map

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("_opt_map", "recorded", "closed", "close_calls", "dev"):
            object.__setattr__(self, name, value)
            return
        if name not in self._opt_map:
            raise AttributeError(f"No such SANE option: {name}")
        if self._opt_map[name].is_button:
            raise AttributeError(f"Buttons don't have values: {name}")
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


@dataclass
class _FakeCompleted:
    returncode: int = 0
    stderr: str = ""
    stdout: str = ""


class FakeRun:
    """Records subprocess.run calls; replays a configured result or exception."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stderr: str = "",
        raises: Exception | None = None,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self._returncode = returncode
        self._stderr = stderr
        self._raises = raises
        self._on_call = on_call

    def __call__(self, args: list[str], **kwargs: Any) -> _FakeCompleted:
        self.calls.append((args, kwargs))
        if self._on_call is not None:
            self._on_call()
        if self._raises is not None:
            raise self._raises
        return _FakeCompleted(self._returncode, self._stderr)


def _patch_run(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> FakeRun:
    fake = FakeRun(**kwargs)
    monkeypatch.setattr(sane_backend.subprocess, "run", fake)
    return fake


def _make_backend(sane_module: FakeSaneModule) -> SaneBackend:
    backend = SaneBackend.__new__(SaneBackend)
    backend._sane = sane_module
    backend._sane_initialized = True
    backend._devices_cache = None
    backend._id_remap = {}
    backend._active_sessions = {}
    backend._session_lock = threading.Lock()
    return backend


_EJECT_INDEX = 7
_DEV = "coolscan3:usb:libusb:001:007"
COOLSCAN3_OPT_WITH_EJECT = {
    "frame": FakeOption(constraint=(1, 40, 1)),
    "eject": FakeOption(index=_EJECT_INDEX, is_button=True),
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
    def test_presses_eject_via_scanimage_and_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT))
        backend = _make_backend(FakeSaneModule(dev))
        run = _patch_run(monkeypatch, returncode=0)

        result = backend.eject(_DEV)

        assert result is True
        assert run.calls[0][0] == ["scanimage", "-d", _DEV, "--eject"]
        assert dev.dev.set_option_calls == []  # never routed through the broken button path
        assert dev.recorded == {}
        assert dev.closed is True  # detection handle closed before scanimage runs

    def test_detection_handle_is_closed_before_scanimage_opens_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Single-open hardware: the python-sane handle MUST be closed before
        # scanimage tries to open the same device, or scanimage hits "device busy".
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT))
        backend = _make_backend(FakeSaneModule(dev))
        observed: dict[str, bool] = {}
        _patch_run(monkeypatch, on_call=lambda: observed.__setitem__("closed_at_call", dev.closed))

        backend.eject(_DEV)

        assert observed["closed_at_call"] is True

    def test_spurious_out_of_documents_exit_is_treated_as_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT))
        backend = _make_backend(FakeSaneModule(dev))
        _patch_run(
            monkeypatch,
            returncode=7,
            stderr="scanimage: sane_start: Document feeder out of documents\n",
        )

        assert backend.eject(_DEV) is True

    def test_capability_gated_skip_when_device_has_no_eject_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_NO_EJECT))
        backend = _make_backend(FakeSaneModule(dev))
        run = _patch_run(monkeypatch, returncode=0)

        result = backend.eject(_DEV)

        assert result is False
        assert run.calls == []  # scanimage never invoked
        assert dev.closed is True

    @pytest.mark.parametrize(
        "broken_option",
        [
            FakeOption(active=False, is_button=True),
            FakeOption(settable=False, is_button=True),
        ],
    )
    def test_capability_gated_skip_when_eject_option_is_inactive_or_unsettable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        broken_option: FakeOption,
    ) -> None:
        opt = dict(COOLSCAN3_OPT_NO_EJECT)
        opt["eject"] = broken_option
        dev = FakeSaneDev(opt)
        backend = _make_backend(FakeSaneModule(dev))
        run = _patch_run(monkeypatch, returncode=0)

        result = backend.eject(_DEV)

        assert result is False
        assert run.calls == []
        assert dev.closed is True

    def test_raises_when_scanimage_reports_a_real_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT))
        backend = _make_backend(FakeSaneModule(dev))
        _patch_run(monkeypatch, returncode=1, stderr="scanimage: open of device failed: Device busy\n")

        with pytest.raises(RuntimeError, match="scanimage --eject failed.*[Dd]evice busy"):
            backend.eject(_DEV)

    def test_raises_when_scanimage_is_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT))
        backend = _make_backend(FakeSaneModule(dev))
        _patch_run(monkeypatch, raises=FileNotFoundError("scanimage"))

        with pytest.raises(RuntimeError, match="not installed"):
            backend.eject(_DEV)

    def test_raises_when_the_device_cannot_be_opened(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = FakeSaneModule(open_error=PermissionError("scanner is busy"))
        backend = _make_backend(module)
        run = _patch_run(monkeypatch, returncode=0)

        with pytest.raises(RuntimeError, match="Failed to open scanner.*scanner is busy"):
            backend.eject(_DEV)

        assert run.calls == []  # never reached the press

    def test_only_the_requested_device_is_opened_and_ejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dev = FakeSaneDev(dict(COOLSCAN3_OPT_WITH_EJECT))
        module = FakeSaneModule(dev)
        backend = _make_backend(module)
        run = _patch_run(monkeypatch, returncode=0)

        backend.eject(_DEV)

        assert module.opened == [_DEV]
        assert run.calls[0][0][2] == _DEV


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

    def test_propagates_a_genuine_eject_failure_from_the_backend(self) -> None:
        class FailingBackend:
            def eject(self, device_id: str) -> bool:
                raise RuntimeError("could not trigger eject")

        service = ScannerService()
        service._backend = FailingBackend()

        with pytest.raises(RuntimeError, match="could not trigger eject"):
            service.eject("coolscan3:usb:test")
