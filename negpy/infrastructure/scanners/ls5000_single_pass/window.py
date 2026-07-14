"""Minimal Nikon LS-5000 SET/GET WINDOW decoder used by the capture worker.

The offsets mirror ``coolscan3.c`` and the Nikon Scan wire capture. Keeping
this decoder inside the package makes the installed worker self-contained.
"""

from __future__ import annotations

import struct
from typing import TypedDict


CS3_COLOR_NAMES = {0: "none/auto", 1: "R", 2: "G", 3: "B", 9: "IR"}
SCAN_KIND_NAMES = {0x01: "NORMAL", 0x20: "AE", 0x40: "AE_WB"}


class WindowBlock(TypedDict):
    """Decoded fields from one validated 58-byte Nikon window block."""

    header8_hex: str
    color_id: int
    color_name: str
    resx: int
    resy: int
    upper_left_x: int
    upper_left_y: int
    width: int
    height: int
    brightness: int
    threshold: int
    contrast: int
    image_composition: int
    bit_depth: int
    reserved_35_47_hex: str
    multiread_byte: int
    samples_per_scan_minus1_nibble: int
    avg_negpos_byte: int
    is_negative: bool
    scanning_kind_byte: int
    scanning_kind_name: str
    scanning_mode_byte: int
    is_multi_sample: bool
    color_interleaving_byte: int
    ae_byte: int
    exposure_raw_10ns: int
    exposure_ms: float


def _be_word(payload: bytes | bytearray, offset: int) -> int:
    return (payload[offset] << 8) | payload[offset + 1]


def _be_long(payload: bytes | bytearray, offset: int) -> int:
    return struct.unpack(">I", payload[offset : offset + 4])[0]


def decode_window_block(payload: bytes | bytearray) -> WindowBlock | None:
    """Decode one exact 58-byte SET/GET WINDOW parameter block."""

    if len(payload) != 58:
        return None
    color = payload[8]
    scan_kind = payload[50]
    exposure = _be_long(payload, 54)
    return {
        "header8_hex": payload[0:8].hex(),
        "color_id": color,
        "color_name": CS3_COLOR_NAMES.get(color, f"0x{color:02x}"),
        "resx": _be_word(payload, 10),
        "resy": _be_word(payload, 12),
        "upper_left_x": _be_long(payload, 14),
        "upper_left_y": _be_long(payload, 18),
        "width": _be_long(payload, 22),
        "height": _be_long(payload, 26),
        "brightness": payload[30],
        "threshold": payload[31],
        "contrast": payload[32],
        "image_composition": payload[33],
        "bit_depth": payload[34],
        "reserved_35_47_hex": payload[35:48].hex(),
        "multiread_byte": payload[48],
        "samples_per_scan_minus1_nibble": (payload[48] >> 4) & 0x0F,
        "avg_negpos_byte": payload[49],
        "is_negative": (payload[49] & 0x01) == 0,
        "scanning_kind_byte": scan_kind,
        "scanning_kind_name": SCAN_KIND_NAMES.get(scan_kind, f"0x{scan_kind:02x}"),
        "scanning_mode_byte": payload[51],
        "is_multi_sample": payload[51] == 0x10,
        "color_interleaving_byte": payload[52],
        "ae_byte": payload[53],
        "exposure_raw_10ns": exposure,
        "exposure_ms": exposure * 10e-6,
    }


__all__ = ["WindowBlock", "decode_window_block"]
