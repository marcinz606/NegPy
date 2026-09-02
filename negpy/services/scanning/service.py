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

    def __init__(self, backend: ScannerBackend | None = None, backend_id: str | None = None) -> None:
        self._backend = backend
        self._backend_id = backend_id

    def _get_backend(self) -> ScannerBackend:
        if self._backend is None:
            from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID, create_backend

            self._backend = create_backend(self._backend_id or DEFAULT_BACKEND_ID)
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

    def detect_frames(self, device_id: str, *, film_format: str | None = None, film_type: str = "negative") -> int:
        """How many frames the loaded film carries, 0 where the transport cannot measure it.

        A feeder counts slots instead, and the caller has that from the device capabilities.
        """
        detect = getattr(self._get_backend(), "detect_frames", None)
        return 0 if detect is None else int(detect(device_id, film_format=film_format, film_type=film_type))

    def open_roll(
        self,
        device: ScannerDevice,
        *,
        dpi: int,
        film_format: str | None = None,
        film_type: str = "negative",
    ) -> RollSession:
        """Open a strip for whole-roll preview.

        A backend that reaches the whole strip natively supplies its own RollSession
        through `open_roll`; the rest are wrapped one frame at a time.
        """
        backend = self._get_backend()
        native = getattr(backend, "open_roll", None)
        if native is not None:
            return native(device, dpi=dpi, film_format=film_format, film_type=film_type)
        return PerFrameRollSession(backend, device, dpi=dpi)

    def run_scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float, str], None],
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

        from negpy.infrastructure.scanners.settings import MONO_TIFF
        from negpy.services.scanning.writer import write_tiff_16bit

        os.makedirs(output_folder, exist_ok=True)

        fmt = output_format.upper()
        date_str = dt_date.today().strftime("%Y%m%d")
        ext = ".tif"

        require_sequence_varying_scan_filename(filename_pattern, date_str)

        current = 1 if seq is None else seq
        while True:
            basename = render_scan_filename(filename_pattern, date_str, current)
            rgb_path = os.path.join(output_folder, basename)
            if not os.path.exists(rgb_path + ext):
                break
            current += 1

        rgb_path = write_tiff_16bit(result, rgb_path, mono=fmt == MONO_TIFF.upper())

        return rgb_path
