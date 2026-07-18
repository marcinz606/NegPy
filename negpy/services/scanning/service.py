import os
import threading
from typing import Callable

from negpy.infrastructure.scanners.base import ScannerBackend, ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.scanners.sane_backend import SaneBackend, SaneSession
from negpy.kernel.system.logging import get_logger
from negpy.services.scanning.templating import render_scan_filename, require_sequence_varying_scan_filename

logger = get_logger(__name__)

_SCAN_IO_RETRY_ATTEMPTS = 3
_SCAN_IO_RETRY_DELAY_S = 0.5
# Stable SANE status strings for transport glitches worth one retry (a Coolscan's
# USB link occasionally hiccups mid-strip). A real error — bad option, missing
# frame — carries a different message and must fail fast.
_TRANSIENT_IO_MARKERS = ("error during device i/o", "device busy")


def _is_transient_scan_io(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_IO_MARKERS)


class ScannerService:
    """Orchestrates device enumeration, scan execution, and file writing."""

    def __init__(self) -> None:
        self._backend: ScannerBackend | None = None

    def _get_backend(self) -> ScannerBackend:
        if self._backend is None:
            self._backend = SaneBackend()
        return self._backend

    def list_devices(self) -> list[ScannerDevice]:
        return self._get_backend().list_devices()

    def refresh_devices(self) -> list[ScannerDevice]:
        backend = self._get_backend()
        if hasattr(backend, "refresh_devices"):
            return backend.refresh_devices()  # type: ignore[union-attr]
        return backend.list_devices()

    def open_session(self, device_id: str) -> SaneSession:
        """Open an exclusive device session for batch/roll workflows.

        The session owns the scanner until closed: one continuous SANE open,
        per-frame scan() calls, one release (close/eject) at the end. Raises
        when the backend has no session support.
        """
        open_session = getattr(self._get_backend(), "open_session", None)
        if not callable(open_session):
            raise RuntimeError("Scanner backend does not support exclusive device sessions")
        return open_session(device_id)

    def eject(self, device_id: str) -> bool:
        """Trigger the device's eject action where the backend supports it.

        Returns False cleanly when the backend has no eject method or the device
        exposes no usable eject option; raises when a present eject genuinely fails.
        """
        eject = getattr(self._get_backend(), "eject", None)
        if not callable(eject):
            return False
        return eject(device_id)

    def run_scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
        *,
        retry_delay: float = _SCAN_IO_RETRY_DELAY_S,
    ) -> ScanResult:
        """Scan, retrying once on a transient USB I/O glitch (fresh open each try)."""
        backend = self._get_backend()
        for attempt in range(1, _SCAN_IO_RETRY_ATTEMPTS + 1):
            try:
                return backend.scan(device_id, params, progress, cancel)
            except Exception as e:
                if attempt >= _SCAN_IO_RETRY_ATTEMPTS or cancel.is_set() or not _is_transient_scan_io(e):
                    raise
                logger.warning(
                    "Transient scanner I/O on %s (attempt %d/%d), retrying: %s",
                    device_id,
                    attempt,
                    _SCAN_IO_RETRY_ATTEMPTS,
                    e,
                )
                cancel.wait(retry_delay)  # interruptible settle before the retry
                if cancel.is_set():
                    raise
        raise RuntimeError("unreachable")  # pragma: no cover

    def write_result(
        self,
        result: ScanResult,
        output_folder: str,
        filename_pattern: str,
        output_format: str = "TIFF",
        seq: int | None = None,
    ) -> str:
        """Write ScanResult to disk. Returns path to the RGB file.

        Filename pattern is a Jinja2 template with variables: date, seq. `seq`
        seeds the collision search: single scans pass None (start at 1); a range
        batch passes the frame number so masters are frame-numbered.
        """
        from datetime import date as dt_date

        from negpy.services.scanning.writer import write_dng_linear, write_tiff_16bit

        os.makedirs(output_folder, exist_ok=True)

        date_str = dt_date.today().strftime("%Y%m%d")
        ext = ".dng" if output_format.upper() == "DNG" else ".tif"

        require_sequence_varying_scan_filename(filename_pattern, date_str)

        current = 1 if seq is None else seq
        while True:
            basename = render_scan_filename(filename_pattern, date_str, current)
            rgb_path = os.path.join(output_folder, basename)
            if not os.path.exists(rgb_path + ext):
                break
            current += 1

        if output_format.upper() == "DNG":
            rgb_path = write_dng_linear(result, rgb_path)
        else:
            rgb_path = write_tiff_16bit(result, rgb_path)

        return rgb_path
