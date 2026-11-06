"""A Scanlight serial port stand-in: answers the firmware query, streams LED temperature, and keeps
the light's history so a simulated Bulb exposure can add up the light it saw."""

import math
import queue
import time
from typing import Optional

from negpy.infrastructure.capture import protocol as proto

_FIRMWARE_ID = 7
_HARDWARE_ID = 0  # Big Scanlight: white and infrared channels
_TELEMETRY_INTERVAL_S = 0.2
_HISTORY_S = 60.0  # longer than any Bulb exposure

_color = [0, 0, 0, 0, 0]
# (monotonic time, (r, g, b, w, ir)) at each change, oldest first.
_history: list[tuple[float, tuple[int, ...]]] = [(0.0, (0, 0, 0, 0, 0))]
# (start, end) of the last shutter pulse on the 3.5mm jack, None once a camera took it.
_pulse: list[Optional[tuple[float, float]]] = [None]


def current_color() -> tuple[int, int, int, int]:
    r, g, b, w = _color[:4]
    return r, g, b, w


def current_ir() -> int:
    return _color[4] if len(_color) > 4 else 0


def light_dose(t0: float, t1: float) -> tuple[float, ...]:
    """Level times seconds per channel (r, g, b, w, ir) between `t0` and `t1`."""
    dose = [0.0] * 5
    for i, (start, levels) in enumerate(_history):
        end = _history[i + 1][0] if i + 1 < len(_history) else t1
        span = min(end, t1) - max(start, t0)
        if span > 0:
            dose = [d + level * span for d, level in zip(dose, levels)]
    return tuple(dose)


def take_pulse() -> Optional[tuple[float, float]]:
    """The last shutter pulse once it has ended, consumed so one pulse makes one exposure."""
    pulse = _pulse[0]
    if pulse is None or pulse[1] > time.monotonic():
        return None
    _pulse[0] = None
    return pulse


def _record(levels: tuple[int, ...]) -> None:
    now = time.monotonic()
    _color[:] = levels
    _history.append((now, tuple(levels)))
    while len(_history) > 1 and _history[1][0] < now - _HISTORY_S:
        del _history[0]


class SimSerial:
    port = "sim"

    def __init__(self) -> None:
        self.is_open = True
        self._replies: queue.Queue[bytes] = queue.Queue()
        self._next_telemetry = 0.0

    def write(self, data: bytes) -> None:
        if len(data) >= 8 and data[0] == proto.START_BYTE and data[1] == proto.H2D_SET_COLOR:
            _record(tuple(data[3:8]))
        elif len(data) >= 4 and data[0] == proto.START_BYTE and data[1] == proto.H2D_SHUTTER_PULSE:
            now = time.monotonic()
            _pulse[0] = (now, now + data[3] * proto.SHUTTER_PULSE_UNIT_S)
        elif len(data) >= 2 and data[0] == proto.START_BYTE and data[1] == proto.H2D_GET_FW_VERSION:
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
