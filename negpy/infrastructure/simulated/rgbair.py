"""A CuteNew RGB AIR serial port stand-in: answers identification and Set All RGB, and sends
one telemetry report on open."""

import queue
import struct

from negpy.infrastructure.capture import rgbair

_IDENT = b"RGB AIR\x00SIM0001\x00HW1.0\x001.0.0\x001.3\x00"


class SimRgbAirSerial:
    port = "sim-rgbair"

    def __init__(self) -> None:
        self.is_open = True
        self.commands: list[bytes] = []  # every command payload received, in order
        self.channels = ((0, 100), (0, 100), (0, 100))  # (on, brightness) for R, G, B
        self.busy = False
        self._out: queue.Queue[bytes] = queue.Queue()
        self._buf = bytearray()
        self._out.put(rgbair.encode_frame(rgbair.T_TELEMETRY, 0, struct.pack("<hHHHB", 303, 1886, 0, 0, 0)))

    @property
    def lit(self) -> tuple[int, ...]:
        return tuple(level if on else 0 for on, level in self.channels)

    def write(self, data: bytes) -> int:
        self._buf.extend(data)
        for kind, seq, payload in rgbair.decode_frames(self._buf):
            if kind == rgbair.T_COMMAND:
                self.commands.append(payload)
                self._out.put(rgbair.encode_frame(rgbair.T_RESPONSE, seq, self._answer(payload)))
        return len(data)

    def _answer(self, payload: bytes) -> bytes:
        cmd = payload[0]
        if cmd == rgbair.CMD_IDENT:
            return bytes([cmd, 0]) + _IDENT
        if cmd == rgbair.CMD_SET_ALL:
            if len(payload) != 7:
                return bytes([cmd, 2])
            if self.busy:
                return bytes([cmd, 4])
            f = payload[1:]
            if any(on > 1 for on in f[::2]) or any(not 1 <= v <= 100 for v in f[1::2]):
                return bytes([cmd, 3])
            self.channels = ((f[0], f[1]), (f[2], f[3]), (f[4], f[5]))
            return bytes([cmd, 0])
        return bytes([cmd, 1])

    def read(self, size: int = 1) -> bytes:
        try:
            return self._out.get(timeout=0.05)
        except queue.Empty:
            return b""

    def close(self) -> None:
        self.is_open = False
