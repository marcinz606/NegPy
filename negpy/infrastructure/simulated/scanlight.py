"""A Scanlight serial port stand-in: answers the firmware query and streams LED temperature."""

import math
import queue
import time

from negpy.infrastructure.capture import protocol as proto

_FIRMWARE_ID = 7
_HARDWARE_ID = 3  # Scanlight v4b: has the white channel
_TELEMETRY_INTERVAL_S = 0.2


class SimSerial:
    port = "sim"

    def __init__(self) -> None:
        self.is_open = True
        self._replies: queue.Queue[bytes] = queue.Queue()
        self._next_telemetry = 0.0

    def write(self, data: bytes) -> None:
        if len(data) >= 2 and data[0] == proto.START_BYTE and data[1] == proto.H2D_GET_FW_VERSION:
            word = (_HARDWARE_ID << 16) | _FIRMWARE_ID
            self._replies.put(proto.encode_packet(proto.D2H_FW_VERSION, word.to_bytes(4, "big")))

    def read(self, _n: int) -> bytes:
        now = time.monotonic()
        if now >= self._next_telemetry:
            self._next_telemetry = now + _TELEMETRY_INTERVAL_S
            millidegrees = int(35000 + 1500 * math.sin(now / 30))
            return proto.encode_packet(proto.D2H_LED_TEMP, millidegrees.to_bytes(4, "big", signed=True))
        try:
            return self._replies.get(timeout=0.02)
        except queue.Empty:
            return b""

    def close(self) -> None:
        self.is_open = False
