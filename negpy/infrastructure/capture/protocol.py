"""Scanlight USB-CDC serial protocol — packet constants and codec helpers.

Vendored from jackw01's Scanlight automation firmware protocol via rohanpandula's
TriRGB `scanlightctl` (MIT). This is the subset NegPy speaks, byte-for-byte faithful
to the firmware; the device offers more (a 3.5mm shutter trigger, NVM defaults, bus
telemetry) that nothing here needs. The whole Scanlight family (v2, v4, Big Scanlight)
shares this exact wire protocol — same 0xFE framing, same packet headers, same
SET_COLOR layout — and all enumerate as the stock Pico CDC device, so this one codec
drives any of them. They differ only in the reported hardware ID (see
`describe_hardware`) and physical LED power.

Wire format (both directions):

    byte 0: 0xFE       start byte, always
    byte 1: header     packet type
    byte 2: length N   payload length in bytes
    bytes 3..3+N: data payload

Telemetry packets (D2H LED_TEMP) arrive every ~200ms unsolicited; responses
(D2H FW_VERSION) arrive only after the matching host request. Dispatch by header
byte, and drop headers this codec does not decode — never assume read order.
"""

from __future__ import annotations

START_BYTE = 0xFE

# Host-to-device headers (the firmware also defines 1 = GET_DEFAULT_RGB and
# 3 = SHUTTER_PULSE, which NegPy never sends)
H2D_SET_COLOR = 0
H2D_GET_FW_VERSION = 2

# Device-to-host headers (2 = VBUS and 4 = DEFAULT_RGB arrive but are dropped)
D2H_LED_TEMP = 1
D2H_FW_VERSION = 3


class ProtocolError(Exception):
    """Raised for malformed packets or out-of-range values."""


def encode_packet(header: int, data: bytes = b"") -> bytes:
    """Frame a single packet for transmission."""
    if not 0 <= header <= 255:
        raise ProtocolError(f"header out of range: {header}")
    if len(data) > 255:
        raise ProtocolError(f"data too long: {len(data)} bytes")
    return bytes([START_BYTE, header, len(data)]) + bytes(data)


def encode_set_color(r: int, g: int, b: int, w: int = 0, save: bool = False) -> bytes:
    """Build a PKT_H2D_SET_COLOR packet.

    IR byte is always 0 (ignored by v4 firmware). `save` writes the values to
    NVM as power-on defaults — finite write cycles, so it must be opt-in.
    """
    for name, value in (("r", r), ("g", g), ("b", b), ("w", w)):
        if not 0 <= value <= 255:
            raise ProtocolError(f"{name} channel out of range 0-255: {value}")
    payload = bytes([r, g, b, w, 0, 1 if save else 0])
    return encode_packet(H2D_SET_COLOR, payload)


def decode_led_temp(data: bytes) -> float:
    """LED_TEMP payload → degrees C (32-bit signed millidegrees, big-endian)."""
    if len(data) < 4:
        raise ProtocolError(f"LED_TEMP payload too short: {len(data)} bytes")
    return int.from_bytes(data[:4], "big", signed=True) / 1000.0


def decode_fw_version(data: bytes) -> tuple[int, int]:
    """FW_VERSION payload → (firmware_id, hardware_id).

    Firmware emits `FW + (HW << 16)` as a big-endian u32: low 16 bits FW,
    high 16 bits HW.
    """
    if len(data) < 4:
        raise ProtocolError(f"FW_VERSION payload too short: {len(data)} bytes")
    word = int.from_bytes(data[:4], "big")
    return word & 0xFFFF, (word >> 16) & 0xFFFF


# Known Scanlight hardware IDs, from HW_VERSION_ID in jackw01's firmware (config.h).
# The family shares this wire protocol, so NegPy drives any of them identically.
HARDWARE_NAMES = {0: "Big Scanlight", 1: "Scanlight v4"}


def describe_hardware(hw_id: int) -> str:
    """Friendly device name for a reported hardware ID (from `decode_fw_version`),
    or 'hw<n>' for an id we don't have a name for."""
    return HARDWARE_NAMES.get(hw_id, f"hw{hw_id}")
