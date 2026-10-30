"""Which light the Scan panel connects to: a Scanlight first, else a Lento Dock."""

import pytest

from negpy.infrastructure.capture import lights
from negpy.infrastructure.capture.lento_dock import LentoDock


class _Light:
    def __init__(self, port=None):
        self.port = port


@pytest.fixture
def no_hardware(monkeypatch):
    monkeypatch.delenv("NEGPY_SIMULATE_HARDWARE", raising=False)
    monkeypatch.setattr(lights, "Scanlight", _Light)


def test_a_scanlight_on_the_bus_is_used_first(no_hardware, monkeypatch):
    monkeypatch.setattr(lights.scanlight, "discover_port", lambda: "/dev/cu.usbmodem1101")
    monkeypatch.setattr(lights.lento_dock, "connect", lambda port=None: pytest.fail("the dock must not be probed"))
    light = lights.open_light()
    assert isinstance(light, _Light) and light.port == "/dev/cu.usbmodem1101"


def test_the_dock_is_used_when_no_scanlight_is_found(no_hardware, monkeypatch):
    def no_scanlight():
        raise RuntimeError("Could not auto-discover the Scanlight serial port.")

    monkeypatch.setattr(lights.scanlight, "discover_port", no_scanlight)
    monkeypatch.setattr(lights.lento_dock, "connect", lambda port=None: "dock")
    assert lights.open_light() == "dock"


def test_with_no_light_the_scanlight_error_is_kept(no_hardware, monkeypatch):
    # The Scan panel keeps this error out of the status line by its wording.
    def no_scanlight():
        raise RuntimeError("No serial ports found. Is the Scanlight plugged in?")

    def no_dock(port=None):
        raise RuntimeError("No Lento Dock found.")

    monkeypatch.setattr(lights.scanlight, "discover_port", no_scanlight)
    monkeypatch.setattr(lights.lento_dock, "connect", no_dock)
    with pytest.raises(RuntimeError, match="No serial ports"):
        lights.open_light()


def test_an_explicit_port_picks_the_driver_by_its_name(no_hardware, monkeypatch):
    monkeypatch.setattr(lights.lento_dock, "connect", lambda port=None: ("dock", port))
    assert lights.open_light("/dev/cu.usbserial-10") == ("dock", "/dev/cu.usbserial-10")
    assert lights.open_light("/dev/cu.usbmodem1101").port == "/dev/cu.usbmodem1101"


def test_the_simulator_can_stand_in_for_a_dock(monkeypatch):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")
    monkeypatch.setenv("NEGPY_SIMULATE_LIGHT", "lento")
    light = lights.open_light()
    try:
        assert isinstance(light, LentoDock)
        assert light.describe() == ("Lento Dock 35 (fw 1.0.224)", False, False)
    finally:
        light.close()
