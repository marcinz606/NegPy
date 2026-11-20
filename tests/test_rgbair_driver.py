"""CuteNew RGB AIR driver: frame codec, CRC and the light commands, against a simulated port."""

import struct
import time

import pytest

from negpy.infrastructure.capture import rgbair
from negpy.infrastructure.simulated.rgbair import SimRgbAirSerial


def test_crc_is_ccitt_false():
    assert rgbair.crc16(b"123456789") == 0x29B1


def test_a_frame_survives_noise_split_reads_and_a_corrupt_frame():
    good = rgbair.encode_frame(rgbair.T_RESPONSE, 7, b"\x21\x00")
    bad = bytearray(rgbair.encode_frame(rgbair.T_RESPONSE, 8, b"\x03\x00"))
    bad[-1] ^= 0xFF
    stream = b"\x00\xaa" + bytes(bad) + good
    buf = bytearray()
    frames = []
    for i in range(len(stream)):
        buf.extend(stream[i : i + 1])
        frames += rgbair.decode_frames(buf)
    assert frames == [(rgbair.T_RESPONSE, 7, b"\x21\x00")]


def test_an_oversized_length_resynchronizes():
    buf = bytearray(b"\xaa\x55\x81\x01\xff\x00" + rgbair.encode_frame(rgbair.T_RESPONSE, 2, b"\x03\x00"))
    assert rgbair.decode_frames(buf) == [(rgbair.T_RESPONSE, 2, b"\x03\x00")]


def test_levels_map_to_brightness_1_to_100():
    assert [rgbair.to_brightness(v) for v in (0, 1, 128, 255)] == [1, 1, 50, 100]
    with pytest.raises(ValueError):
        rgbair.to_brightness(256)


def test_ident_needs_five_terminated_strings():
    with pytest.raises(ValueError):
        rgbair.parse_ident(b"a\x00b\x00c\x00d\x00e")


@pytest.fixture
def light():
    sim = SimRgbAirSerial()
    air = rgbair.RgbAir(serial_obj=sim)
    yield air, sim
    air.close()


def test_open_reads_identification_and_telemetry(light):
    air, _ = light
    assert air.info.protocol == "1.3"
    assert air.describe() == ("RGB AIR (fw 1.0.0)", True, False)
    deadline = time.monotonic() + 1
    while air.last_temp_c is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert air.last_temp_c == pytest.approx(30.3)


def test_a_channel_at_zero_is_off_by_its_enable_flag(light):
    air, sim = light
    air.set_color(r=255, b=128)
    assert sim.commands[-1] == bytes([0x03, 1, 100, 0, 100, 1, 50])
    assert sim.lit == (100, 0, 50)
    air.off()
    assert sim.lit == (0, 0, 0)


def test_white_lights_rgb_together_and_infrared_is_refused(light):
    air, sim = light
    air.set_color(w=255)
    assert sim.lit == (100, 100, 100)
    with pytest.raises(ValueError):
        air.set_color(ir=255)
    with pytest.raises(ValueError):
        air.set_color(r=10, w=10)


def test_a_busy_reply_raises_and_is_not_resent(light):
    air, sim = light
    sim.busy = True
    sent = len(sim.commands)
    with pytest.raises(RuntimeError, match="busy"):
        air.set_color(g=255)
    assert len(sim.commands) == sent + 1


def test_close_switches_the_light_off():
    sim = SimRgbAirSerial()
    air = rgbair.RgbAir(serial_obj=sim)
    air.set_color(g=200)
    air.close()
    assert sim.lit == (0, 0, 0) and not sim.is_open


def test_fault_events_are_logged_not_raised(light, caplog):
    air, _ = light
    air._dispatch(rgbair.T_EVENT, 0, bytes([3, 1, 1]) + struct.pack("<H", 712) + b"\x01")
    assert "overtemperature" in caplog.text
