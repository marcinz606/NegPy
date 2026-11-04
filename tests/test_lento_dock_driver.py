"""Lento Dock serial driver tests against the simulated dock: no hardware."""

import time

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


def test_take_control_enters_host_control_at_once(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock.take_control()
        assert (sim.mode, sim.expmode) == ("SEMI", "CONT")
        assert sim.lit == (0, 0, 0)
    assert (sim.mode, sim.expmode) == ("AUTO", "BRGB")


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


def test_advance_moves_one_frame_and_waits_until_it_is_in_place(sim):
    with LentoDock(serial_obj=sim) as dock:
        assert dock.advance_frame() == 0
        assert sim.frame == 1
        assert sim.mode == "SEMI"  # FRAME NEXT needs host control
        assert dock.advance_frame() == 1


def test_the_last_frame_is_scanned_before_the_strip_ends():
    # The sensor reads no film while the last frame is still in the window.
    sim = SimDockSerial(frames=3)
    with LentoDock(serial_obj=sim) as dock:
        assert [dock.advance_frame() for _ in range(2)] == [0, 1]
        assert sim.film == 0 and sim.frame == 2
        assert dock.advance_frame() is None
        assert sim.frame == 2  # nothing moved: the frame in the window was the last


def test_advance_without_film_reports_the_end():
    sim = SimDockSerial(frames=0)
    with LentoDock(serial_obj=sim) as dock:
        assert dock.advance_frame() is None
    assert "FRAME NEXT" not in sim.lines


def test_advance_restores_the_saved_mode_on_close(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock.advance_frame()
    assert (sim.mode, sim.expmode) == ("AUTO", "BRGB")


def test_a_refused_advance_raises(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock._enter_continuous()
        sim.mode = "AUTO"  # the dock dropped host control
        with pytest.raises(ValueError):
            dock.advance_frame()


def test_eject_moves_the_strip_out(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock.eject()
        assert sim.frames == 0
        assert dock.advance_frame() is None  # nothing left to scan


def test_the_light_holds_again_after_an_eject(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock.set_color(200, 110, 40)
        dock.eject()
        sim.hold = 0  # the dock may drop the hold when the strip leaves
        dock.set_color(200, 110, 40)
        assert sim.lit == tuple(to_duty(v) for v in (200, 110, 40))


def test_the_s_button_calls_the_scan_handler(sim):
    sim.answers_btn = False  # firmware that reports S only through its own exposure
    pressed = []
    with LentoDock(serial_obj=sim) as dock:
        dock.on_scan_button = lambda: pressed.append(1)
        dock.advance_frame()
        assert pressed == []  # NegPy's own commands are never a press
        sim.press_scan(hold_s=0.5)
        for _ in range(30):
            if pressed:
                break
            time.sleep(0.01)
        assert pressed == [1]  # at the press, not at the end of the dock's exposure
        assert dock.advance_frame() == 1  # the film waits for the dock's exposure to end
    assert pressed == [1]


def test_set_color_now_confirms_the_hold_once_then_writes_without_waiting(dock, sim, monkeypatch):
    dock.set_color_now()  # the first call, in the dark: continuous mode and the hold, both confirmed
    assert sim.expmode == "CONT" and sim.lines.count("CONTHOLD 1") == 1

    def _no_waiting(*_a, **_k):
        raise AssertionError("a timed switch must not wait for the dock's reply")

    monkeypatch.setattr(dock, "_await", _no_waiting)
    dock.set_color_now(r=255)
    dock.set_color_now(g=128)
    dock.set_color_now()
    deadline = time.monotonic() + 1.0
    while sim.lines[-1] != "CONTRGB 0 0 0" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sim.lines[-3:] == ["CONTRGB 1000 0 0", "CONTRGB 0 502 0", "CONTRGB 0 0 0"]
    assert sim.lines.count("CONTHOLD 1") == 1


def test_set_color_now_confirms_the_hold_again_after_off(dock, sim):
    dock.set_color_now(r=255)
    dock.off()
    dock.set_color_now()
    assert sim.lines.count("CONTHOLD 1") == 2


def test_the_keep_alive_reaches_the_dock_while_a_command_waits(sim, monkeypatch):
    monkeypatch.setattr(lento_dock, "KEEPALIVE_TICK_S", 0.02)
    monkeypatch.setattr(lento_dock, "PING_INTERVAL_S", 0.1)
    with LentoDock(serial_obj=sim) as dock:
        sent = len(sim.lines)
        with pytest.raises(TimeoutError):
            dock._command("SLOW", r"^NEVER", timeout=0.5)
        assert "PING" in sim.lines[sent:]


class _UnpluggedSerial(SimDockSerial):
    unplugged = False

    def write(self, data: bytes) -> None:
        if self.unplugged:
            raise OSError(6, "Device not configured")
        super().write(data)


def test_an_unplugged_dock_ends_the_keep_alive_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(lento_dock, "KEEPALIVE_TICK_S", 0.02)
    monkeypatch.setattr(lento_dock, "PING_INTERVAL_S", 0.0)
    sim = _UnpluggedSerial()
    dock = LentoDock(serial_obj=sim)
    sim.unplugged = True
    dock._keepalive.join(timeout=2.0)
    assert not dock._keepalive.is_alive()
    assert not dock.is_connected()
    assert sum("link lost" in r.getMessage() for r in caplog.records) == 1
    with pytest.raises(ConnectionError):
        dock.set_color(255, 0, 0)
    dock.close()


def test_a_strip_at_the_entrance_is_fed_to_its_first_frame(monkeypatch):
    monkeypatch.setattr(lento_dock, "AUTOALIGN_START_S", 0.0)
    sim = SimDockSerial(frames=4, fed=False)
    with LentoDock(serial_obj=sim) as dock:
        assert dock.strip_waiting()
        dock.insert_strip()
        assert sim.st == "READY" and not dock.strip_waiting()
        assert dock.advance_frame() == 0


def test_a_failed_feed_raises():
    sim = SimDockSerial(frames=0)
    sim.frames = 0
    with LentoDock(serial_obj=sim) as dock:
        with pytest.raises(RuntimeError, match="INSERT ERR"):
            dock.insert_strip()


def test_firmware_that_answers_ok_still_lights_the_panel():
    sim = SimDockSerial(ok_replies=True)
    with LentoDock(serial_obj=sim) as dock:
        dock.set_color(200, 110, 40)
        assert sim.lit == tuple(to_duty(v) for v in (200, 110, 40))


def test_a_busy_dock_gets_the_command_again(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock._enter_continuous()
        sim.busy = 2  # the dock's own exposure is running
        dock.set_color(200, 110, 40)
        assert sim.lines.count("CONTRGB 784 431 157") == 3
        assert sim.lit == tuple(to_duty(v) for v in (200, 110, 40))


def test_frameat_err_is_the_end_of_the_strip(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock._enter_continuous()
        sim._frame_next = lambda: (sim._say("OK"), sim._say("FRAMEAT ERR no_edge"), sim._say(sim._state()))
        assert dock.advance_frame() is None


def test_a_scan_request_is_the_s_button_and_counts_once(sim):
    sim.answers_btn = False  # firmware that reports S only through its own exposure
    pressed = []
    with LentoDock(serial_obj=sim) as dock:
        dock.on_scan_button = lambda: pressed.append(1)
        sim.request_scan()
        sim.press_scan()  # the same press may also start the dock's exposure
        for _ in range(50):
            if pressed:
                break
            time.sleep(0.01)
        time.sleep(0.1)
    assert pressed == [1]


def test_a_strip_the_dock_aligns_itself_is_not_fed_again(monkeypatch):
    import threading

    monkeypatch.setattr(lento_dock, "AUTOALIGN_START_S", 0.0)
    sim = SimDockSerial(frames=4)
    sim.st = "SEEK"  # the dock aligns a strip that just went in

    def aligned():
        sim.st = "READY"
        sim._say(sim._state())

    with LentoDock(serial_obj=sim) as dock:
        threading.Timer(0.3, aligned).start()
        assert not dock.strip_waiting()
    assert "INSERT" not in sim.lines


def _wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_a_tap_of_s_read_through_btn_works_as_scan(sim):
    pressed = []
    with LentoDock(serial_obj=sim) as dock:
        dock.on_scan_button = lambda: pressed.append(1)
        dock.set_color(200, 110, 40)  # NegPy holds the dock: the buttons are read
        _wait_for(lambda: "BTN?" in sim.lines)
        sim.shutter = 1
        time.sleep(0.3)
        assert pressed == []  # a press counts on release, so a hold can stay the dock's own
        sim.shutter = 0
        _wait_for(lambda: pressed)
    assert pressed == [1]


def test_a_held_s_is_left_to_the_dock(sim, monkeypatch):
    monkeypatch.setattr(lento_dock, "SCAN_BUTTON_HOLD_S", 0.3)
    pressed = []
    with LentoDock(serial_obj=sim) as dock:
        dock.on_scan_button = lambda: pressed.append(1)
        dock.set_color(200, 110, 40)
        _wait_for(lambda: "BTN?" in sim.lines)
        sim.shutter = 1
        time.sleep(0.5)
        sim.shutter = 0
        time.sleep(0.3)
    assert pressed == []


def test_the_arrows_jog_while_negpy_holds_the_dock(sim):
    sim.btnmode = "FRAME"
    with LentoDock(serial_obj=sim) as dock:
        dock.set_color(200, 110, 40)
        assert sim.btnmode == "JOG"
    assert sim.btnmode == "FRAME"  # the user's setting comes back


def test_the_dock_stays_awake_while_negpy_holds_it(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock.set_color(200, 110, 40)
        assert sim.sleep_s == 0
    assert sim.sleep_s == 90  # the user's timeout comes back


def test_a_silent_dock_reads_as_disconnected(sim):
    with LentoDock(serial_obj=sim) as dock:
        dock.set_color(200, 110, 40)
        sim.asleep = True
        time.sleep(0.3)  # the button polls go unanswered too
        with pytest.raises(TimeoutError):
            dock.set_color(100, 100, 100)
        assert not dock.is_connected()
        sim.asleep = False
