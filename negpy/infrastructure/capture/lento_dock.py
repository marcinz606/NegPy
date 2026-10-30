"""Lento Dock serial driver: the dock's continuous narrowband RGB light, as a `LightSource`.

The Dock (Lento, model 35 and 120) is a film carrier with a motorized transport and an RGB
panel. It speaks plain-text lines at 115200 baud: a command such as `CONTRGB 500 0 0`, and
replies such as `CONTRGB 500 0 0`, `OK`, `ERR` or `STATE key=value ...`.

The panel has two light modes. The pulsed modes time each exposure in the firmware; the
continuous mode (`EXPMODE CONT`) holds the panel at the `CONTRGB` levels, so the camera's
shutter sets the exposure, as with a Scanlight. This driver uses the continuous mode:

    MODE SEMI       the host controls the dock
    IDLE AUTO       the panel shows the continuous levels
    EXPMODE CONT    continuous light
    CONTRGB r g b   duty per channel, 0-1000
    CONTHOLD 1      keep the light on (refreshed, as the firmware lets it lapse)

The dock saves MODE and EXPMODE to its EEPROM, so `close` writes back the values it found.
Opening the port resets the dock, which then prints a BOOT line.

`connect()` finds the dock on the bus. Test seam: pass `serial_obj` to inject a fake serial.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BAUDRATE = 115200
DEFAULT_READ_TIMEOUT_S = 0.1
REPLY_TIMEOUT_S = 2.5
BOOT_TIMEOUT_S = 3.0

#: The Scanlight's 0-255 level is the dock's 0-1000 duty, scaled.
SCANLIGHT_FULL_SCALE = 255
DOCK_FULL_SCALE = 1000

#: The dock drops the host link after 15 s of silence, and lets CONTHOLD lapse.
PING_INTERVAL_S = 5.0
#: The motor moves one frame and checks the frame edge within this time.
ADVANCE_TIMEOUT_S = 15.0
EJECT_TIMEOUT_S = 30.0
HOLD_INTERVAL_S = 30.0

_KV = re.compile(r"(\w+)=(\S+)")

# Sentinel pushed to the reply queue when the reader thread exits, so a blocked request
# wakes immediately with the real cause.
_READER_DIED = "\x00reader-died"


def parse_kv(line: str) -> dict[str, str]:
    """`STATE link=1 mode=SEMI` -> {"link": "1", "mode": "SEMI"}."""
    return dict(_KV.findall(line))


def to_duty(level: int) -> int:
    """A 0-255 Scanlight level as the dock's 0-1000 duty."""
    if not 0 <= level <= SCANLIGHT_FULL_SCALE:
        raise ValueError(f"light level out of range 0-{SCANLIGHT_FULL_SCALE}: {level}")
    return round(level * DOCK_FULL_SCALE / SCANLIGHT_FULL_SCALE)


@dataclass(frozen=True)
class DockInfo:
    """What `IDENT` and `VER` report."""

    model: str
    firmware: str
    serial: str

    @property
    def name(self) -> str:
        return f"Lento Dock {self.model}" if self.model else "Lento Dock"


def is_dock_port(device: str) -> bool:
    """A serial port whose name the dock's USB-serial bridge can have on macOS and Linux."""
    name = device.lower()
    return any(tag in name for tag in ("usbserial", "wchusbserial", "slab_usbtouart", "ttyusb"))


#: Bridge ports that did not answer IDENT, skipped until they leave the bus: probing opens
#: the port, which resets whatever is on it, and the Scan panel looks for a light on a timer.
_not_a_dock: set[str] = set()


def connect(port: Optional[str] = None, *, opener=None) -> "LentoDock":
    """An open connection to the Lento Dock on `port`, or to the one found on the bus.

    Discovery opens each untried USB-serial bridge port and asks IDENT. The connection that
    answers is returned, not closed and reopened, because each open resets the dock.
    """
    opener = opener or LentoDock
    if port:
        return opener(port)
    from serial.tools import list_ports

    candidates = [p.device for p in list_ports.comports() if is_dock_port(p.device)]
    _not_a_dock.intersection_update(candidates)
    for device in candidates:
        if device in _not_a_dock:
            continue
        dock = None
        try:
            dock = opener(device)
            if dock.info is not None:
                return dock
        except Exception:
            logger.debug("no Lento Dock on %s", device, exc_info=True)
        if dock is not None:
            dock.close()
        _not_a_dock.add(device)
    raise RuntimeError("No Lento Dock found. Is it plugged in and is Dock Connect closed?")


class LentoDock:
    """Driver for the Lento Dock's continuous light, with the `Scanlight` driver's interface.

    Use as a context manager whenever possible: `close` restores the dock's saved mode and
    stops the reader thread, so the serial port releases.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        *,
        serial_obj=None,
        baudrate: int = DEFAULT_BAUDRATE,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
        boot_timeout_s: float = BOOT_TIMEOUT_S,
    ):
        if serial_obj is not None:
            self._serial = serial_obj
            self._port = getattr(serial_obj, "port", "<injected>")
        else:
            import serial

            if not port:
                raise ValueError("LentoDock needs a port; use connect() to find one")
            self._port = port
            self._serial = serial.Serial(port, baudrate=baudrate, timeout=read_timeout_s)

        self._write_lock = threading.Lock()
        self._state: dict[str, str] = {}
        self._replies: queue.Queue[str] = queue.Queue()
        self._booted = threading.Event()
        self._light_on = False
        self._continuous = False
        self._last_frameat = ""
        self._in_command = False
        self.on_scan_button: Optional[Callable[[], None]] = None  # called on the reader thread
        self._saved: dict[str, str] = {}
        self._last_tx = time.monotonic()
        self._last_hold = 0.0

        self._stop = threading.Event()
        self._reader_error: Optional[BaseException] = None
        self._reader = threading.Thread(target=self._reader_loop, name="lento-dock-reader", daemon=True)
        self._reader.start()

        # A real port resets the dock on open; a fake one may never print BOOT.
        self._booted.wait(boot_timeout_s)
        self.info = self._identify()
        self._keepalive = threading.Thread(target=self._keepalive_loop, name="lento-dock-keepalive", daemon=True)
        self._keepalive.start()

    # ----- public API (the Scanlight driver's) -----

    @property
    def port(self) -> str:
        return self._port

    @property
    def last_temp_c(self) -> Optional[float]:
        """The dock reports no LED temperature."""
        return None

    def is_connected(self) -> bool:
        return self._reader.is_alive() and self._reader_error is None

    def describe(self) -> tuple[str, bool, bool]:
        """(status text, has a white channel, has an infrared channel) for the Scan panel."""
        info = self.info or self._identify()
        if info is None:
            raise ConnectionError("Lento Dock does not answer IDENT")
        return f"{info.name} (fw {info.firmware})", False, False

    def set_color(self, r: int = 0, g: int = 0, b: int = 0, w: int = 0, ir: int = 0, save: bool = False) -> None:
        """Hold the panel at R, G, B. The continuous light has no white or infrared channel:
        white alone lights R, G and B together at its level, and infrared is refused."""
        if ir:
            raise ValueError("The Lento Dock has no infrared light")
        if w and (r or g or b):
            raise ValueError("White and narrowband cannot be lit together")
        if w:
            r = g = b = w
        duties = [to_duty(v) for v in (r, g, b)]
        if not any(duties):
            self.off()
            return
        self._enter_continuous()
        self._command(f"CONTRGB {duties[0]} {duties[1]} {duties[2]}", r"^(CONTRGB|ERR)")
        if not self._light_on:
            self._command("CONTHOLD 1", r"^(CONTHOLD|ERR)")
            self._last_hold = time.monotonic()
        self._light_on = True

    def off(self) -> None:
        if self._continuous:
            self._command("CONTRGB 0 0 0", r"^(CONTRGB|ERR)")
        self._light_on = False

    def advance_frame(self) -> Optional[int]:
        """Move the film one frame and wait until the frame is in the window. Returns the dock's
        frame index, or None at the end of the strip.

        The film sensor reads no film while the last frame is still in the window, so the frame
        reached with no film is a frame; the strip ends at the next advance.
        """
        self._enter_continuous()
        self._command("STATE?", r"^STATE\b")
        if self._state.get("film") == "0":
            return None
        self._last_frameat = ""
        self._command(
            "FRAME NEXT",
            r"^(OK|ERR)",
            then=((r"^(FRAMEAT|END_OF_ROLL|ROLLEND)\b", ADVANCE_TIMEOUT_S), (r"^STATE\b.*\bst=(READY|IDLE)\b", REPLY_TIMEOUT_S)),
        )
        if not self._last_frameat:
            return None
        return int(parse_kv(self._last_frameat).get("idx", "-1"))

    def eject(self) -> None:
        """Move the strip out of the dock. The last frame stops in the window, short of the exit."""
        self._enter_continuous()
        self._command("EJECT", r"^(OK|ERR)", then=((r"^(EJECTED\b|STATE\b.*\bentrance=0\b)", EJECT_TIMEOUT_S),))
        self._light_on = False  # the next set_color holds the light again, for a shot of the empty dock

    def close(self) -> None:
        try:
            if self.is_connected():
                self._restore()
        except Exception:
            logger.exception("could not restore the Lento Dock's mode")
        self._stop.set()
        for thread in (self._reader, getattr(self, "_keepalive", None)):
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        try:
            if getattr(self._serial, "is_open", True):
                self._serial.close()
        except Exception:
            logger.exception("error closing Lento Dock serial port")

    def __enter__(self) -> "LentoDock":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ----- internals -----

    def _identify(self) -> Optional[DockInfo]:
        try:
            ident = parse_kv(self._command("IDENT", r"^IDENT\b"))
            ver = parse_kv(self._command("VER", r"^VER\b"))
        except (TimeoutError, ConnectionError, ValueError):
            return None
        return DockInfo(model=ident.get("model", ""), firmware=ver.get("fw", "?"), serial=ident.get("serial", ""))

    def _enter_continuous(self) -> None:
        if self._continuous:
            return
        if not self._state:
            self._command("STATE?", r"^STATE\b")
        self._saved = {"MODE": self._state.get("mode", "AUTO"), "EXPMODE": self._state.get("expmode", "BRGB")}
        self._command("MODE SEMI", r"^(MODE=|ERR)")
        self._command("IDLE AUTO", r"^(IDLE=|ERR)")
        self._command("EXPMODE CONT", r"^(EXPMODE=|ERR)")
        self._continuous = True

    def _restore(self) -> None:
        if not self._continuous:
            return
        self._command("CONTRGB 0 0 0", r"^(CONTRGB|ERR)")
        self._command("CONTHOLD 0", r"^(CONTHOLD|ERR)")
        expmode = self._saved.get("EXPMODE", "BRGB")
        self._command(f"EXPMODE {'BRGB' if expmode == 'CONT' else expmode}", r"^(EXPMODE=|ERR)")
        mode = self._saved.get("MODE", "AUTO")
        self._command(f"MODE {'AUTO' if mode == 'SEMI' else mode}", r"^(MODE=|ERR)")
        self._continuous = self._light_on = False

    def _send(self, cmd: str) -> None:
        with self._write_lock:
            self._serial.write((cmd + "\r\n").encode("ascii"))
            self._last_tx = time.monotonic()

    def _command(self, cmd: str, expect: str, timeout: float = REPLY_TIMEOUT_S, then: tuple[tuple[str, float], ...] = ()) -> str:
        """Send `cmd` and return the first reply line that matches `expect`. Each `(pattern,
        timeout)` in `then` waits for a later line in turn, and the last line matched is returned."""
        if self._reader_error is not None:
            raise ConnectionError(f"Lento Dock reader stopped: {self._reader_error}")
        with self._write_lock:
            while not self._replies.empty():
                try:
                    self._replies.get_nowait()
                except queue.Empty:
                    break
            self._serial.write((cmd + "\r\n").encode("ascii"))
            self._last_tx = time.monotonic()
            self._in_command = True
            try:
                line = self._await(cmd, expect, timeout)
                for pattern, wait in then:
                    line = self._await(cmd, pattern, wait)
            finally:
                self._in_command = False
            return line

    def _await(self, cmd: str, expect: str, timeout: float) -> str:
        rx = re.compile(expect)
        deadline = time.monotonic() + timeout
        seen: list[str] = []
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                heard = "; ".join(seen[-8:]) or "nothing"
                raise TimeoutError(f"No reply to {cmd!r} within {timeout}s (the dock sent: {heard})")
            try:
                line = self._replies.get(timeout=left)
            except queue.Empty:
                continue
            if line == _READER_DIED:
                raise ConnectionError(f"Lento Dock reader stopped before replying to {cmd!r}")
            seen.append(line)
            if rx.search(line):
                if re.match(r"^ERR\b", line):
                    raise ValueError(f"Lento Dock refused {cmd!r}: {line}")
                return line

    def _keepalive_loop(self) -> None:
        while not self._stop.wait(1.0):
            try:
                now = time.monotonic()
                if self._light_on and now - self._last_hold >= HOLD_INTERVAL_S:
                    self._send("CONTHOLD 1")
                    self._last_hold = now
                elif now - self._last_tx >= PING_INTERVAL_S:
                    self._send("PING")
            except Exception:
                logger.debug("Lento Dock keep-alive failed", exc_info=True)

    def _reader_loop(self) -> None:
        buf = b""
        try:
            while not self._stop.is_set():
                chunk = self._serial.read(256)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.rstrip(b"\r").decode("ascii", errors="replace").strip()
                    if line:
                        self._dispatch(line)
        except BaseException as exc:  # noqa: BLE001 — surfaced to the caller
            self._reader_error = exc
        finally:
            self._replies.put(_READER_DIED)

    def _dispatch(self, line: str) -> None:
        logger.debug("Lento Dock: %s", line)
        if line.startswith("FRAMEAT"):
            self._last_frameat = line
        elif line.startswith(("END_OF_ROLL", "ROLLEND")):
            self._last_frameat = ""
        if line.startswith("STATE"):
            self._state = parse_kv(line)
        elif line.startswith("BOOT"):
            self._booted.set()
        elif line.startswith("SHOT done") and not self._in_command and self.on_scan_button is not None:
            # NegPy sends no SHOT: an exposure it did not ask for is the S button, which the dock
            # answers with its own exposure before it reports.
            try:
                self.on_scan_button()
            except Exception:
                logger.exception("Lento Dock scan button handler failed")
        self._replies.put(line)
