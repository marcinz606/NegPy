"""Which light the Scan panel connects to: an RGB AIR or a Scanlight first, else a Lento Dock."""

import pytest

from negpy.infrastructure.capture import lights
from negpy.infrastructure.capture.lento_dock import LentoDock
from negpy.infrastructure.capture.rgbair import RgbAir


class _Light:
    def __init__(self, port=None):
        self.port = port


@pytest.fixture
def no_hardware(monkeypatch):
    monkeypatch.delenv("NEGPY_SIMULATE_HARDWARE", raising=False)
    monkeypatch.setattr(lights, "Scanlight", _Light)
    monkeypatch.setattr(lights, "RgbAir", _Light)
    monkeypatch.setattr(lights.rgbair, "find_port", lambda: None)
    monkeypatch.setattr(lights.rgbair, "is_rgbair_port", lambda port: port == "/dev/ttyACM7")


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
        assert light.describe() == ("Lento Dock 35 (fw 1.0.224)", True, False)
    finally:
        light.close()


def test_an_rgbair_is_used_before_the_scanlight_discovery(no_hardware, monkeypatch):
    monkeypatch.setattr(lights.rgbair, "find_port", lambda: "/dev/cu.usbmodem2101")
    monkeypatch.setattr(lights.scanlight, "discover_port", lambda: pytest.fail("the Scanlight discovery takes any usbmodem port"))
    assert lights.open_light().port == "/dev/cu.usbmodem2101"


def test_an_explicit_rgbair_port_opens_the_rgbair(no_hardware):
    assert lights.open_light("/dev/ttyACM7").port == "/dev/ttyACM7"


def test_the_simulator_can_stand_in_for_an_rgbair(monkeypatch):
    monkeypatch.setenv("NEGPY_SIMULATE_HARDWARE", "1")
    monkeypatch.setenv("NEGPY_SIMULATE_LIGHT", "rgbair")
    light = lights.open_light()
    try:
        assert isinstance(light, RgbAir)
        assert light.describe() == ("RGB AIR (fw 1.0.0)", True, False)
    finally:
        light.close()
