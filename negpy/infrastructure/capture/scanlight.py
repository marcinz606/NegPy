"""Scanlight USB-CDC device driver (v2 / v4 / Big Scanlight — one shared protocol).

Vendored from rohanpandula's TriRGB `scanlightctl` (MIT). A `Scanlight` instance
owns one serial port plus a background reader thread that decodes incoming
packets, updating cached telemetry (LED_TEMP, VBUS) or handing payloads to a
per-header response queue (FW_VERSION, DEFAULT_RGB). Host requests block on the
matching queue.

Test seam: pass `serial_obj` to inject a fake serial; then `port`/`baudrate`
are ignored.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

from negpy.infrastructure.capture import protocol as proto
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BAUDRATE = 115200
DEFAULT_READ_TIMEOUT_S = 0.1

# Every Scanlight (v2/v4/Big Scanlight) enumerates with the stock Pico CDC stdio
# descriptors — VID 0x2E8A, PID 0x000A on RP2040 or 0x0009 on RP2350/Pico 2.
PICO_VID = 0x2E8A
PICO_CDC_PIDS = {0x000A, 0x0009}

# Sentinel pushed to the response queues when the reader thread exits, so an
# in-flight _request() blocked on q.get() wakes immediately with the real cause.
_READER_DIED = object()


def discover_port() -> str:
    """Best-effort auto-discovery of the Scanlight CDC serial port on macOS."""
    from serial.tools import list_ports

    ports = list(list_ports.comports())

    pico = [p for p in ports if p.vid == PICO_VID and p.pid in PICO_CDC_PIDS]
    if len(pico) == 1:
        return pico[0].device
    if len(pico) > 1:
        raise RuntimeError(
            "Multiple Raspberry Pi Pico CDC ports found; set the port explicitly: "
            + ", ".join(f"{p.device} ({p.vid:04x}:{p.pid:04x})" for p in pico)
        )

    def fields(p) -> str:
        return " ".join(str(x or "") for x in (p.description, p.manufacturer, p.product, p.interface)).lower()

    named = [p for p in ports if "scanlight" in fields(p)]
    if len(named) == 1:
        return named[0].device
    if len(named) > 1:
        raise RuntimeError("Multiple Scanlight-like serial ports found; set the port explicitly: " + ", ".join(p.device for p in named))

    usbmodem = [p for p in ports if "usbmodem" in p.device.lower()]
    if len(usbmodem) == 1:
        return usbmodem[0].device

    if not ports:
        raise RuntimeError("No serial ports found. Is the Scanlight plugged in?")
    raise RuntimeError(
        "Could not auto-discover the Scanlight serial port. Set it explicitly. Available ports: " + ", ".join(p.device for p in ports)
    )


class Scanlight:
    """Driver for a Scanlight narrowband-RGB light source (v2 / v4 / Big Scanlight).

    All variants speak the same protocol, so one driver covers them; they differ
    only in reported hardware ID and physical LED power (the Big Scanlight has two
    LED arrays and negotiates higher USB-C power).

    Use as a context manager whenever possible — the background reader thread
    must be stopped cleanly for the serial port to release.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        *,
        serial_obj=None,
        baudrate: int = DEFAULT_BAUDRATE,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
    ):
        if serial_obj is not None:
            self._serial = serial_obj
            self._port = getattr(serial_obj, "port", "<injected>")
        else:
            import serial

            self._port = port or discover_port()
            self._serial = serial.Serial(self._port, baudrate=baudrate, timeout=read_timeout_s)

        self._lock = threading.Lock()
        self._last_temp_c: Optional[float] = None
        # One queue per solicited response header. Unsolicited telemetry (LED_TEMP) and
        # headers this codec ignores never land here — see `_dispatch`.
        self._response_queues: dict[int, queue.Queue] = {proto.D2H_FW_VERSION: queue.Queue()}

        self._reader_stop = threading.Event()
        self._reader_error: Optional[BaseException] = None
        self._reader_thread = threading.Thread(target=self._reader_loop, name="scanlight-reader", daemon=True)
        self._reader_thread.start()

    # ----- public API -----

    @property
    def port(self) -> str:
        return self._port

    @property
    def last_temp_c(self) -> Optional[float]:
        with self._lock:
            return self._last_temp_c

    def is_connected(self) -> bool:
        """True while the serial reader is alive and error-free (the device is still present).
        Goes False when the Pico is unplugged (the read raises and the reader thread exits), so
        callers can drop this handle and reconnect a freshly-enumerated device."""
        return self._reader_thread.is_alive() and self._reader_error is None

    def set_color(self, r: int = 0, g: int = 0, b: int = 0, w: int = 0, save: bool = False) -> None:
        """Set R, G, B, W channels. `save=True` writes to NVM (use sparingly)."""
        if w and (r or g or b):
            raise ValueError("White channel cannot be on simultaneously with any RGB channel")
        self._serial.write(proto.encode_set_color(r, g, b, w, save))

    def off(self) -> None:
        self.set_color(0, 0, 0, 0)

    def get_fw_version(self, timeout: float = 2.0) -> tuple[int, int]:
        """Request (firmware_id, hardware_id). Raises TimeoutError on no reply."""
        return self._request(proto.H2D_GET_FW_VERSION, proto.D2H_FW_VERSION, proto.decode_fw_version, timeout)

    def close(self) -> None:
        self._reader_stop.set()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        try:
            if getattr(self._serial, "is_open", True):
                self._serial.close()
        except Exception:
            logger.exception("error closing Scanlight serial port")

    def __enter__(self) -> "Scanlight":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ----- internals -----

    def _request(self, h2d_header: int, d2h_header: int, decoder, timeout: float):
        q = self._response_queues[d2h_header]
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        if self._reader_error is not None:
            raise self._reader_error
        if not self._reader_thread.is_alive():
            raise ConnectionError(f"Scanlight reader thread is not running; cannot service header {h2d_header}")
        self._serial.write(proto.encode_packet(h2d_header))
        try:
            data = q.get(timeout=timeout)
        except queue.Empty:
            if self._reader_error is not None:
                raise self._reader_error
            if not self._reader_thread.is_alive():
                raise ConnectionError(f"Scanlight reader thread stopped; no response to header {h2d_header}")
            raise TimeoutError(f"No response to header {h2d_header} within {timeout}s")
        if data is _READER_DIED:
            if self._reader_error is not None:
                raise self._reader_error
            raise ConnectionError(f"Scanlight reader stopped before responding to header {h2d_header}")
        if self._reader_error is not None:
            raise self._reader_error
        return decoder(data)

    def _reader_loop(self) -> None:
        buf = bytearray()
        try:
            while not self._reader_stop.is_set():
                chunk = self._serial.read(256)
                if chunk:
                    buf.extend(chunk)
                    self._consume(buf)
        except BaseException as exc:  # noqa: BLE001 — surfaced to the main thread
            self._reader_error = exc
        finally:
            for q in self._response_queues.values():
                try:
                    q.put_nowait(_READER_DIED)
                except queue.Full:
                    pass

    def _consume(self, buf: bytearray) -> None:
        """Parse as many complete packets from `buf` as available, in place."""
        while True:
            if not buf:
                return
            if buf[0] != proto.START_BYTE:
                idx = buf.find(bytes([proto.START_BYTE]))
                if idx < 0:
                    buf.clear()
                    return
                del buf[:idx]
                continue
            if len(buf) < 3:
                return
            length = buf[2]
            total = 3 + length
            if len(buf) < total:
                return
            header = buf[1]
            data = bytes(buf[3:total])
            del buf[:total]
            self._dispatch(header, data)

    def _dispatch(self, header: int, data: bytes) -> None:
        if header == proto.D2H_LED_TEMP:
            try:
                temp = proto.decode_led_temp(data)
            except proto.ProtocolError:
                return
            with self._lock:
                self._last_temp_c = temp
        elif header in self._response_queues:
            self._response_queues[header].put(data)
        # Unknown headers are silently dropped — forward-compat with newer firmware.
