"""Lento Dock serial driver tests against the simulated dock: no hardware."""

import pytest

from negpy.infrastructure.capture import lento_dock
from negpy.infrastructure.capture.base import LightSource
from negpy.infrastructure.capture.lento_dock import LentoDock, connect, is_dock_port, to_duty
from negpy.infrastructure.simulated.lento_dock import SimDockSerial


@pytest.fixture
def sim():
    return SimDockSerial()


@pytest.fixture
def dock(sim):
    device = LentoDock(serial_obj=sim)
    yield device
    device.close()


def test_constructing_the_driver_needs_no_hardware(dock):
    assert isinstance(dock, LightSource)
    assert dock.is_connected()
    assert dock.info is not None and dock.info.model == "35"


def test_describe_offers_white_and_no_infrared(dock):
    detail, has_white, has_ir = dock.describe()
    assert detail == "Lento Dock 35 (fw 1.0.224)"
    assert (has_white, has_ir) == (True, False)


def test_levels_scale_to_the_docks_duty():
    assert [to_duty(v) for v in (0, 51, 255)] == [0, 200, 1000]
    with pytest.raises(ValueError):
        to_duty(256)


def test_first_color_enters_the_continuous_light(dock, sim):
    dock.set_color(255, 0, 0)
    assert sim.lit == (1000, 0, 0)
    assert sim.mode == "SEMI" and sim.expmode == "CONT"
    assert sim.lines.index("EXPMODE CONT") < sim.lines.index("CONTRGB 1000 0 0") < sim.lines.index("CONTHOLD 1")


def test_a_color_change_needs_no_new_hold(dock, sim):
    dock.set_color(255, 0, 0)
    dock.set_color(0, 128, 0)
    assert sim.lit == (0, 502, 0)
    assert sim.lines.count("CONTHOLD 1") == 1


def test_white_alone_lights_rgb_together(dock, sim):
    dock.set_color(w=102)
    assert sim.lit == (400, 400, 400)


def test_white_beside_narrowband_and_infrared_are_refused(dock):
    with pytest.raises(ValueError):
        dock.set_color(10, 0, 0, 255)
    with pytest.raises(ValueError):
        dock.set_color(ir=255)


def test_off_and_all_zero_darken_the_panel(dock, sim):
    dock.set_color(0, 0, 255)
    dock.off()
    assert sim.lit == (0, 0, 0)
    dock.set_color(0, 0, 255)
    dock.set_color(0, 0, 0)
    assert sim.lit == (0, 0, 0)


def test_off_before_any_color_sends_nothing(dock, sim):
    sent = len(sim.lines)
    dock.off()
    assert len(sim.lines) == sent


def test_close_restores_the_saved_mode(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock.set_color(255, 255, 255)
    assert (sim.mode, sim.expmode, sim.hold) == ("AUTO", "BRGB", 0)
    assert sim.lit == (0, 0, 0)
    assert not sim.is_open


def test_close_restores_a_custom_saved_mode():
    sim = SimDockSerial(saved_expmode="SEQ")
    with LentoDock(serial_obj=sim) as dock:
        dock.set_color(255, 0, 0)
    assert sim.expmode == "SEQ"


def test_close_never_leaves_the_continuous_mode_saved():
    # A crash leaves CONT and SEMI in the dock's EEPROM; the next session must not restore them.
    sim = SimDockSerial(saved_mode="SEMI", saved_expmode="CONT")
    with LentoDock(serial_obj=sim) as dock:
        dock.set_color(255, 0, 0)
    assert (sim.mode, sim.expmode) == ("AUTO", "BRGB")


def test_close_without_light_leaves_the_mode_alone(sim):
    with LentoDock(serial_obj=sim):
        pass
    assert not any(line.startswith(("MODE", "EXPMODE")) for line in sim.lines)


def test_a_refused_command_raises(dock, sim):
    sim._answer = lambda line: sim._say("ERR")  # the dock refuses everything from now on
    with pytest.raises(ValueError):
        dock.set_color(255, 0, 0)


def test_dock_ports_are_usb_serial_bridges():
    assert is_dock_port("/dev/cu.usbserial-10")
    assert is_dock_port("/dev/ttyUSB0")
    assert not is_dock_port("/dev/cu.usbmodem1101")  # a Scanlight (Pico CDC)
    assert not is_dock_port("/dev/cu.Bluetooth-Incoming-Port")


class _Port:
    def __init__(self, device):
        self.device = device


class _Opener:
    """Opens a simulated dock on `dock_on`, and a device that ignores IDENT elsewhere."""

    def __init__(self, dock_on):
        self.dock_on, self.opened, self.closed = dock_on, [], []

    def __call__(self, device):
        self.opened.append(device)
        opener = self

        class Silent:
            info = None

            def close(self):
                opener.closed.append(device)

        return LentoDock(serial_obj=SimDockSerial()) if device == self.dock_on else Silent()


@pytest.fixture
def bus(monkeypatch):
    lento_dock._not_a_dock.clear()
    ports = []
    monkeypatch.setattr("serial.tools.list_ports.comports", lambda: ports, raising=False)
    yield ports
    lento_dock._not_a_dock.clear()


def test_discovery_opens_only_bridge_ports_and_keeps_the_dock(bus):
    bus += [_Port("/dev/cu.usbmodem1101"), _Port("/dev/cu.usbserial-3"), _Port("/dev/cu.usbserial-10")]
    opener = _Opener(dock_on="/dev/cu.usbserial-10")
    dock = connect(opener=opener)
    try:
        assert opener.opened == ["/dev/cu.usbserial-3", "/dev/cu.usbserial-10"]
        assert opener.closed == ["/dev/cu.usbserial-3"]  # the dock's own connection stays open
        assert dock.info is not None and dock.is_connected()
    finally:
        dock.close()


def test_a_port_that_is_not_a_dock_is_not_opened_again(bus):
    bus += [_Port("/dev/cu.usbserial-3")]
    opener = _Opener(dock_on=None)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="No Lento Dock"):
            connect(opener=opener)
    assert opener.opened == ["/dev/cu.usbserial-3"]
    bus.clear()  # unplugged: a device plugged in on that port later is tried again
    with pytest.raises(RuntimeError):
        connect(opener=opener)
    bus += [_Port("/dev/cu.usbserial-3")]
    with pytest.raises(RuntimeError):
        connect(opener=opener)
    assert opener.opened == ["/dev/cu.usbserial-3", "/dev/cu.usbserial-3"]


def test_an_explicit_port_is_opened_without_discovery(bus):
    opener = _Opener(dock_on="/dev/cu.usbserial-10")
    dock = connect("/dev/cu.usbserial-10", opener=opener)
    dock.close()
    assert opener.opened == ["/dev/cu.usbserial-10"]


def test_a_dock_needs_a_port_or_a_serial_object():
    with pytest.raises(ValueError):
        LentoDock()


def test_reply_parsing():
    assert lento_dock.parse_kv("STATE link=1 mode=SEMI expmode=CONT") == {"link": "1", "mode": "SEMI", "expmode": "CONT"}
