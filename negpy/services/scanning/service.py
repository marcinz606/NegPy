import os
import threading
from typing import Callable

from negpy.infrastructure.scanners.base import (
    ScannerBackend,
    ScannerDevice,
    ScannerSession,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.per_frame_roll import PerFrameRollSession
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.scanners.roll import RollSession
from negpy.kernel.system.logging import get_logger
from negpy.services.scanning.templating import render_scan_filename, require_sequence_varying_scan_filename

logger = get_logger(__name__)

_SCAN_IO_RETRY_ATTEMPTS = 3
_SCAN_IO_RETRY_DELAY_S = 0.5


class ScannerService:
    """Orchestrates device enumeration, scan execution, and file writing.

    Knows nothing about any particular transport: the backend classifies its own
    failures (TransientScanError vs anything else) and reports its own
    capabilities. See ScannerBackend for what an implementation owes this class.
    """

    def __init__(self, backend: ScannerBackend | None = None) -> None:
        self._backend = backend

    def _get_backend(self) -> ScannerBackend:
        if self._backend is None:
            from negpy.infrastructure.scanners.sane_backend import SaneBackend

            self._backend = SaneBackend()
        return self._backend

    def list_devices(self) -> list[ScannerDevice]:
        return self._get_backend().list_devices()

    def refresh_devices(self) -> list[ScannerDevice]:
        return self._get_backend().refresh_devices()

    def open_session(self, device_id: str) -> ScannerSession:
        """Open an exclusive device session for batch/roll workflows.

        The session owns the scanner until closed: one continuous open, per-frame
        scan() calls, one release (close/eject) at the end.
        """
        return self._get_backend().open_session(device_id)

    def eject(self, device_id: str) -> bool:
        """Trigger the device's eject action.

        Returns False cleanly when the device exposes no usable eject action;
        raises when a present eject genuinely fails.
        """
        return self._get_backend().eject(device_id)

    def open_roll(self, device: ScannerDevice, *, dpi: int) -> RollSession:
        """Open a strip for whole-roll preview.

        Every backend reaches a strip one frame at a time today, so this always
        wraps. A backend with a native whole-roll traversal supplies its own
        RollSession instead, and this grows a branch then — not before.
        """
        return PerFrameRollSession(self._get_backend(), device, dpi=dpi)

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
            except TransientScanError as e:
                if attempt >= _SCAN_IO_RETRY_ATTEMPTS or cancel.is_set():
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
