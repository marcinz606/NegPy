"""Scanlight wire-protocol unit tests."""

import pytest

from negpy.infrastructure.capture import protocol as proto
from negpy.infrastructure.capture.base import CAPTURE_ORDER, Channel


# ---- Scanlight wire protocol ----------------------------------------------


def test_encode_packet_framing():
    pkt = proto.encode_packet(proto.H2D_GET_FW_VERSION)
    assert pkt == bytes([0xFE, proto.H2D_GET_FW_VERSION, 0])


def test_encode_set_color_payload():
    pkt = proto.encode_set_color(10, 20, 30)
    # start, header, length=6, then r,g,b,w,ir,save
    assert pkt == bytes([0xFE, proto.H2D_SET_COLOR, 6, 10, 20, 30, 0, 0, 0])


def test_encode_set_color_save_flag():
    assert proto.encode_set_color(1, 2, 3, save=True)[-1] == 1


@pytest.mark.parametrize("bad", [-1, 256, 999])
def test_encode_set_color_rejects_out_of_range(bad):
    with pytest.raises(proto.ProtocolError):
        proto.encode_set_color(bad, 0, 0)


def test_decoders_roundtrip():
    assert proto.decode_led_temp((37500).to_bytes(4, "big", signed=True)) == 37.5
    word = (0x0001 << 16) | 0x0002  # hw=1, fw=2
    assert proto.decode_fw_version(word.to_bytes(4, "big")) == (2, 1)


def test_channel_rgb_lights_only_one_channel():
    assert Channel.RED.rgb(200) == (200, 0, 0)
    assert Channel.GREEN.rgb(200) == (0, 200, 0)
    assert Channel.BLUE.rgb(200) == (0, 0, 200)
    assert [c.letter for c in CAPTURE_ORDER] == ["R", "G", "B"]


# ---- hardware identification (Scanlight family) ----------------------------


def test_describe_hardware_known_and_unknown():
    # HW_VERSION_ID values from jackw01's firmware config.h.
    assert proto.describe_hardware(0) == "Big Scanlight"
    assert proto.describe_hardware(1) == "Scanlight v4"
    # An id we don't have a name for falls back to the raw value (no crash, no mislabel).
    assert proto.describe_hardware(7) == "hw7"


def test_has_white_channel():
    # Only the Big Scanlight (0) and v4 (1) have a dedicated white LED.
    assert proto.has_white_channel(0) is True
    assert proto.has_white_channel(1) is True
    # v1-v3 (and any unrecognised id) are RGB-only.
    assert proto.has_white_channel(2) is False
    assert proto.has_white_channel(7) is False


def test_big_scanlight_set_color_is_wire_identical_to_v4():
    # Big Scanlight SET_COLOR firmware reads 5 bytes (R,G,B,W,IR) + byte[5]=save,
    # exactly what encode_set_color emits — so one encoder drives both devices.
    pkt = proto.encode_set_color(10, 20, 30, 40, save=False)
    assert pkt == bytes([0xFE, proto.H2D_SET_COLOR, 6, 10, 20, 30, 40, 0, 0])
