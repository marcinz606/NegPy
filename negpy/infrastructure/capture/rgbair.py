"""CuteNew RGB AIR driver: the light's continuous RGB output, as a `LightSource`.

The RGB AIR speaks binary frames over USB CDC (customer protocol 1.3):

    AA 55 | TYPE u8 | SEQ u8 | LEN u16 LE | PAYLOAD | CRC16 u16 LE

TYPE 01 is a command, 81 its response (same SEQ), 82 telemetry and 83 an event (SEQ 0).
The CRC is CRC-16/CCITT-FALSE over TYPE, SEQ, LEN and PAYLOAD.

The light can also time its own R, G, B capture and fire the camera's shutter. NegPy does not
use that: it holds the light at the asked levels, and the camera's shutter sets the exposure,
as with a Scanlight. Brightness is 1-100 per channel; a channel at 0 is switched off by its
enable flag, as the protocol requires.

Test seam: pass `serial_obj` to inject a fake serial.
"""

from __future__ import annotations

import queue
import struct
import threading
from dataclasses import dataclass
from typing import Optional

from negpy.kernel.system.i18n import tr
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# The ST virtual COM port VID/PID, shared by many STM32 devices: identify by the product string.
USB_VID = 0x0483
USB_PID = 0x5740
USB_PRODUCT = "CUTENEW_RGB_AIR"

SOF = b"\xaa\x55"
MAX_PAYLOAD = 64
T_COMMAND, T_RESPONSE, T_TELEMETRY, T_EVENT = 0x01, 0x81, 0x82, 0x83
CMD_SET_ALL = 0x03
CMD_IDENT = 0x21
EVENT_FAULT = 0x03

SCANLIGHT_FULL_SCALE = 255
FULL_BRIGHTNESS = 100
REPLY_TIMEOUT_S = 1.0

_RESULTS = {1: "unknown command", 2: "invalid length", 3: "invalid parameter", 4: "busy", 5: "not implemented"}
_FAULTS = {1: "overtemperature", 2: "undervoltage", 3: "LED driver fault"}
_READER_DIED = object()


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_frame(kind: int, seq: int, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too long: {len(payload)} bytes")
    body = struct.pack("<BBH", kind, seq, len(payload)) + payload
    return SOF + body + struct.pack("<H", crc16(body))


def decode_frames(buf: bytearray) -> list[tuple[int, int, bytes]]:
    """Take every complete, valid frame out of `buf` as (type, seq, payload). A frame with a bad
    length or CRC is dropped one byte at a time, so the parser finds the next SOF."""
    frames = []
    while True:
        start = buf.find(SOF)
        if start < 0:
            del buf[: max(len(buf) - 1, 0)]  # keep a trailing AA, which may start the next SOF
            return frames
        del buf[:start]
        if len(buf) < 6:
            return frames
        kind, seq, length = struct.unpack_from("<BBH", buf, 2)
        if length > MAX_PAYLOAD:
            del buf[:1]
            continue
        if len(buf) < length + 8:
            return frames
        body = bytes(buf[2 : 6 + length])
        (crc,) = struct.unpack_from("<H", buf, 6 + length)
        if crc != crc16(body):
            del buf[:1]
            continue
        del buf[: length + 8]
        frames.append((kind, seq, body[4:]))


def to_brightness(level: int) -> int:
    """A 0-255 Scanlight level as the light's 1-100 brightness. 0 maps to 1: the channel is off by its enable flag."""
    if not 0 <= level <= SCANLIGHT_FULL_SCALE:
        raise ValueError(tr("light level out of range 0-{value}: {value1}", value=SCANLIGHT_FULL_SCALE, value1=level))
    return max(1, round(level * FULL_BRIGHTNESS / SCANLIGHT_FULL_SCALE))


@dataclass(frozen=True)
class RgbAirInfo:
    """What command 21 reports."""

    model: str
    serial: str
    hardware: str
    firmware: str
    protocol: str


def parse_ident(data: bytes) -> RgbAirInfo:
    """Five NUL-terminated ASCII strings. A missing terminator is a malformed reply."""
    fields = data.split(b"\x00")
    if len(fields) < 6:
        raise ValueError(tr("malformed RGB AIR identification"))
    return RgbAirInfo(*(f.decode("ascii", errors="replace") for f in fields[:5]))


def find_port() -> Optional[str]:
    """The serial port of an RGB AIR on the bus, or None. A port with the ST VID/PID and no
    product string counts too: the driver checks the identification when it opens it."""
    from serial.tools import list_ports

    ports = [p for p in list_ports.comports() if p.vid == USB_VID and p.pid == USB_PID]
    found = [p for p in ports if p.product == USB_PRODUCT] or [p for p in ports if not p.product]
    return found[0].device if found else None


def is_rgbair_port(device: str) -> bool:
    from serial.tools import list_ports

    return any(p.device == device and p.vid == USB_VID and p.pid == USB_PID for p in list_ports.comports())


class RgbAir:
    """Driver for the CuteNew RGB AIR, with the `Scanlight` driver's interface."""

    def __init__(self, port: Optional[str] = None, *, serial_obj=None, read_timeout_s: float = 0.1):
        if serial_obj is not None:
            self._serial = serial_obj
            self._port = getattr(serial_obj, "port", "<injected>")
        else:
            import serial

            found = port or find_port()
            if not found:
                raise RuntimeError(tr("No CuteNew RGB AIR found. Is it plugged in?"))
            self._port = found
            self._serial = serial.Serial(self._port, timeout=read_timeout_s)

        self._request_lock = threading.Lock()
        self._seq = 0
        self._responses: queue.Queue = queue.Queue()
        self._last_temp_c: Optional[float] = None
        self._stop = threading.Event()
        self._reader_error: Optional[BaseException] = None
        self._reader = threading.Thread(target=self._reader_loop, name="rgbair-reader", daemon=True)
        self._reader.start()
        try:
            self.info = parse_ident(self._request(CMD_IDENT))
        except Exception:
            self.close(turn_off=False)
            raise

    @property
    def port(self) -> str:
        return self._port

    @property
    def last_temp_c(self) -> Optional[float]:
        return self._last_temp_c

    def is_connected(self) -> bool:
        return self._reader.is_alive() and self._reader_error is None

    def describe(self) -> tuple[str, bool, bool]:
        """(status text, has a white channel, has an infrared channel). White is R, G and B lit
        together (see `set_color`), so the white-light presets work on this light."""
        return f"{self.info.model or 'RGB AIR'} (fw {self.info.firmware})", True, False

    def set_color(self, r: int = 0, g: int = 0, b: int = 0, w: int = 0, ir: int = 0, save: bool = False) -> None:
        """Light R, G and B. The light has no white or infrared LED: white alone lights R, G and B
        together at its level, and infrared is refused. The light saves every setting itself, so `save` does nothing."""
        if ir:
            raise ValueError(tr("The RGB AIR has no infrared light"))
        if w and (r or g or b):
            raise ValueError(tr("White and narrowband cannot be lit together"))
        if w:
            r = g = b = w
        fields = []
        for level in (r, g, b):
            fields += [1 if level else 0, to_brightness(level) if level else FULL_BRIGHTNESS]
        self._request(CMD_SET_ALL, bytes(fields))

    def off(self) -> None:
        self.set_color()

    def close(self, turn_off: bool = True) -> None:
        """Switch the light off first: the light keeps its output on when the USB link drops."""
        if turn_off and self.is_connected():
            try:
                self.off()
            except Exception:
                logger.exception("could not switch the RGB AIR off")
        self._stop.set()
        if self._reader.is_alive() and self._reader is not threading.current_thread():
            self._reader.join(timeout=1.0)
        try:
            if getattr(self._serial, "is_open", True):
                self._serial.close()
        except Exception:
            logger.exception("error closing RGB AIR serial port")

    def __enter__(self) -> "RgbAir":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _request(self, cmd: int, fields: bytes = b"") -> bytes:
        """Send one command and return its response data. A nonzero result raises. A command is
        never sent again on a timeout: SEQ does not deduplicate, so a resend can act twice."""
        if self._reader_error is not None:
            raise ConnectionError(tr("RGB AIR reader stopped: {value}", value=self._reader_error))
        with self._request_lock:
            self._seq = self._seq % 255 + 1  # SEQ 0 is for unsolicited reports
            seq = self._seq
            while not self._responses.empty():
                self._responses.get_nowait()
            self._serial.write(encode_frame(T_COMMAND, seq, bytes([cmd]) + fields))
            while True:
                try:
                    reply = self._responses.get(timeout=REPLY_TIMEOUT_S)
                except queue.Empty:
                    raise TimeoutError(tr("No reply from the RGB AIR to command {value:02X}", value=cmd)) from None
                if reply is _READER_DIED:
                    raise ConnectionError(tr("RGB AIR reader stopped: {value}", value=self._reader_error))
                rseq, payload = reply
                if rseq == seq and len(payload) >= 2 and payload[0] == cmd:
                    break
        if payload[1]:
            reason = _RESULTS.get(payload[1], f"result {payload[1]}")
            raise RuntimeError(tr("The RGB AIR refused command {value:02X}: {value1}", value=cmd, value1=reason))
        return payload[2:]

    def _reader_loop(self) -> None:
        buf = bytearray()
        try:
            while not self._stop.is_set():
                chunk = self._serial.read(256)
                if chunk:
                    buf.extend(chunk)
                    for kind, seq, payload in decode_frames(buf):
                        self._dispatch(kind, seq, payload)
        except BaseException as exc:  # noqa: BLE001 (surfaced to the caller)
            self._reader_error = exc
        finally:
            self._responses.put(_READER_DIED)

    def _dispatch(self, kind: int, seq: int, payload: bytes) -> None:
        if kind == T_RESPONSE:
            self._responses.put((seq, payload))
        elif kind == T_TELEMETRY and len(payload) >= 9:
            self._last_temp_c = struct.unpack_from("<h", payload)[0] / 10.0
        elif kind == T_EVENT and len(payload) >= 6 and payload[0] == EVENT_FAULT:
            # The spec leaves the polarity of the active byte open, so the fault is only logged.
            fault = _FAULTS.get(payload[1], f"fault {payload[1]}")
            value = struct.unpack_from("<H", payload, 3)[0]
            logger.warning("RGB AIR %s report: active=%d value=%d", fault, payload[2], value)
