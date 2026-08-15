import sys

import pytest

from negpy.infrastructure.scanners import registry
from negpy.infrastructure.scanners.base import ScannerUnavailable


def test_backend_choices_include_plustek():
    assert ("plustek", "pyOpticfilm (Plustek)") in registry.backend_choices()


def test_backend_choices_include_pieusb():
    assert ("pieusb", "PIEUSB") in registry.backend_choices()


def test_backend_choices_sane_only_off_windows():
    ids = {bid for bid, _ in registry.backend_choices()}
    if sys.platform == "win32":
        assert "sane" not in ids
    else:
        assert "sane" in ids


def test_default_backend_id_is_platform_aware():
    if sys.platform == "win32":
        assert registry.DEFAULT_BACKEND_ID == "plustek"
    else:
        assert registry.DEFAULT_BACKEND_ID == "sane"


def test_create_backend_resolves_and_falls_back(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(registry, "BACKENDS", {"sane": ("SANE", lambda: sentinel)})
    monkeypatch.setattr(registry, "DEFAULT_BACKEND_ID", "sane")

    assert registry.create_backend("sane") is sentinel
    assert registry.create_backend("bogus") is sentinel


def test_create_backend_falls_back_from_sane_on_windows(monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows-only fallback")
    sentinel = object()
    monkeypatch.setattr(
        registry,
        "BACKENDS",
        {"plustek": ("pyOpticfilm (Plustek)", lambda: sentinel)},
    )
    assert registry.create_backend("sane") is sentinel


def test_create_plustek_backend_when_usb_available(monkeypatch):
    class _FakeBackend:
        pass

    fake = _FakeBackend()
    monkeypatch.setattr(
        "negpy.infrastructure.scanners.plustek_backend.PlustekBackend",
        lambda calib_cache=None: fake,
    )
    monkeypatch.setattr(
        "negpy.kernel.system.paths.get_default_user_dir",
        lambda: "/tmp/negpy-test-user",
    )
    assert registry.create_backend("plustek") is fake


def test_create_plustek_unavailable_without_pyusb(monkeypatch):
    def _boom(*, calib_cache=None):
        raise ScannerUnavailable("PyUSB is required")

    monkeypatch.setattr(
        "negpy.infrastructure.scanners.plustek_backend.PlustekBackend",
        _boom,
    )
    with pytest.raises(ScannerUnavailable, match="PyUSB"):
        registry.create_backend("plustek")
