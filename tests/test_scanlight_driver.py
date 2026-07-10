"""Scanlight serial driver tests against a fake port — no hardware, no Scanlight.

These exist because the driver used to be exercised only through hardware: a constant
removed from `protocol.py` broke `Scanlight.__init__` for every user, and nothing in the
suite noticed. Constructing the driver is now a test.
"""

import queue
import threading
import time

import pytest

from negpy.infrastructure.capture import protocol as proto
from negpy.infrastructure.capture.base import LightSource
from negpy.infrastructure.capture.scanlight import Scanlight


class FakeSerial:
    """A serial port that records writes and replays queued device frames."""

    port = "/dev/fake"

    def __init__(self):
        self.written = bytearray()
        self._to_read: queue.Queue[bytes] = queue.Queue()
        self.closed = False

    def feed(self, header: int, payload: bytes) -> None:
        self._to_read.put(proto.encode_packet(header, payload))

    def read(self, _n: int) -> bytes:
        try:
            return self._to_read.get(timeout=0.02)
        except queue.Empty:
            return b""

    def write(self, data: bytes) -> None:
        self.written += data

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def light():
    fake = FakeSerial()
    device = Scanlight(serial_obj=fake)
    device.serial = fake  # the test's handle on the wire
    yield device
    device.close()


def test_constructing_the_driver_needs_no_hardware(light):
    # The constructor wires a response queue per solicited header; a header that no
    # longer exists in `protocol` used to raise AttributeError here, for every user.
    assert isinstance(light, LightSource)
    assert light.is_connected()


def test_set_color_writes_one_set_color_packet(light):
    light.set_color(10, 20, 30)
    assert bytes(light.serial.written) == proto.encode_set_color(10, 20, 30)


def test_white_and_narrowband_cannot_be_lit_together(light):
    # A narrowband exposure must see one band only; white plus a channel is neither.
    with pytest.raises(ValueError):
        light.set_color(10, 0, 0, 255)


def test_off_turns_every_channel_down(light):
    light.off()
    assert bytes(light.serial.written) == proto.encode_set_color(0, 0, 0, 0)


def test_get_fw_version_reads_the_device_reply(light):
    word = (1 << 16) | 2  # hw=1 (Scanlight v4), fw=2
    threading.Timer(0.02, lambda: light.serial.feed(proto.D2H_FW_VERSION, word.to_bytes(4, "big"))).start()
    assert light.get_fw_version(timeout=2.0) == (2, 1)


def test_led_temperature_arrives_unsolicited(light):
    light.serial.feed(proto.D2H_LED_TEMP, (42500).to_bytes(4, "big", signed=True))
    for _ in range(100):
        if light.last_temp_c is not None:
            break
        time.sleep(0.01)
    assert light.last_temp_c == 42.5


def test_a_header_the_codec_ignores_is_dropped_quietly(light):
    # The firmware also sends bus telemetry and NVM defaults; NegPy decodes neither.
    light.serial.feed(2, b"\x00\x00\x13\x88")  # D2H_VBUS in the firmware
    light.serial.feed(4, b"\x01\x02\x03")  # D2H_DEFAULT_RGB in the firmware
    time.sleep(0.05)
    assert light.is_connected()  # the reader thread survived both
