"""Tests for cross-platform monitor ICC profile detection."""

import PyQt6.QtDBus as qtdbus
import pytest

from negpy.infrastructure.display import monitor_profile as mp
from negpy.infrastructure.display.monitor_profile import detect_monitor_icc


class _Screen:
    def __init__(self, model: str = "LG HDR 4K", manufacturer: str = "LG Electronics") -> None:
        self._model = model
        self._mfr = manufacturer

    def model(self) -> str:
        return self._model

    def manufacturer(self) -> str:
        return self._mfr


# --- platform dispatch -------------------------------------------------------


@pytest.mark.parametrize(
    "platform,helper",
    [("linux", "_detect_colord"), ("darwin", "_detect_macos"), ("win32", "_detect_windows")],
)
def test_dispatch_per_platform(monkeypatch, platform, helper) -> None:
    monkeypatch.setattr(mp.sys, "platform", platform)
    called = {}
    monkeypatch.setattr(mp, helper, lambda *a, **k: called.setdefault("hit", b"icc") or b"icc")
    assert detect_monitor_icc(_Screen()) == b"icc"
    assert called["hit"] == b"icc"


def test_unknown_platform_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(mp.sys, "platform", "sunos5")
    assert detect_monitor_icc(_Screen()) is None


def test_detection_errors_are_swallowed(monkeypatch) -> None:
    monkeypatch.setattr(mp.sys, "platform", "linux")

    def boom(*a, **k):
        raise RuntimeError("dbus down")

    monkeypatch.setattr(mp, "_detect_colord", boom)
    assert detect_monitor_icc(_Screen()) is None


# --- colord D-Bus parsing ----------------------------------------------------


class _Reply:
    def __init__(self, args) -> None:
        self._args = args

    def arguments(self):
        return self._args


class _Bus:
    def isConnected(self) -> bool:
        return True


class _Manager:
    def __init__(self, device_paths) -> None:
        self._device_paths = device_paths

    def isValid(self) -> bool:
        return True

    def call(self, method, kind):
        assert method == "GetDevicesByKind" and kind == "display"
        return _Reply([self._device_paths])


def _patch_manager(monkeypatch, device_paths) -> None:
    monkeypatch.setattr(qtdbus.QDBusConnection, "systemBus", staticmethod(lambda: _Bus()))
    monkeypatch.setattr(qtdbus, "QDBusInterface", lambda *a: _Manager(device_paths))


def test_colord_reads_profile_bytes(monkeypatch, tmp_path) -> None:
    icc = tmp_path / "monitor.icc"
    icc.write_bytes(b"FAKE-ICC-DATA")
    _patch_manager(monkeypatch, ["/dev/display0"])

    props = {
        ("/dev/display0", "Profiles"): ["/prof/0"],
        ("/prof/0", "Filename"): str(icc),
    }
    monkeypatch.setattr(mp, "_get_property", lambda bus, path, iface, name: props.get((path, name)))
    assert mp._detect_colord(_Screen()) == b"FAKE-ICC-DATA"


def test_colord_no_display_device_returns_none(monkeypatch) -> None:
    _patch_manager(monkeypatch, [])
    assert mp._detect_colord(_Screen()) is None


def test_colord_device_without_profile_returns_none(monkeypatch) -> None:
    _patch_manager(monkeypatch, ["/dev/display0"])
    monkeypatch.setattr(mp, "_get_property", lambda *a: None)
    assert mp._detect_colord(_Screen()) is None


# --- macOS P3 fallback -------------------------------------------------------


def test_macos_uses_colorsync_when_available(monkeypatch) -> None:
    monkeypatch.setattr(mp, "_colorsync_icc", lambda: b"REAL-DISPLAY-ICC")
    assert mp._detect_macos() == b"REAL-DISPLAY-ICC"


def test_macos_falls_back_to_p3(monkeypatch) -> None:
    from negpy.domain.models import ColorSpace
    from negpy.infrastructure.display.color_mgmt import icc_bytes_for_space

    monkeypatch.setattr(mp, "_colorsync_icc", lambda: None)
    assert mp._detect_macos() == icc_bytes_for_space(ColorSpace.P3_D65.value)


def test_macos_falls_back_to_p3_on_error(monkeypatch) -> None:
    from negpy.domain.models import ColorSpace
    from negpy.infrastructure.display.color_mgmt import icc_bytes_for_space

    def boom():
        raise RuntimeError("no CoreGraphics")

    monkeypatch.setattr(mp, "_colorsync_icc", boom)
    assert mp._detect_macos() == icc_bytes_for_space(ColorSpace.P3_D65.value)
