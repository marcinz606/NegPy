import os
import threading
from typing import Callable

from negpy.infrastructure.scanners.base import ScannerBackend, ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.scanners.sane_backend import SaneBackend
from negpy.kernel.system.logging import get_logger
from negpy.services.scanning.templating import render_scan_filename

logger = get_logger(__name__)


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

    def run_scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        backend = self._get_backend()
        return backend.scan(device_id, params, progress, cancel)

    def write_result(
        self,
        result: ScanResult,
        output_folder: str,
        filename_pattern: str,
        output_format: str = "TIFF",
    ) -> str:
        """Write ScanResult to disk. Returns path to the RGB file.

        Filename pattern is a Jinja2 template with variables: date, seq.
        """
        from datetime import date as dt_date

        from negpy.services.scanning.writer import write_dng_linear, write_tiff_16bit

        os.makedirs(output_folder, exist_ok=True)

        date_str = dt_date.today().strftime("%Y%m%d")
        ext = ".dng" if output_format.upper() == "DNG" else ".tif"

        seq = 1
        while True:
            basename = render_scan_filename(filename_pattern, date_str, seq)
            rgb_path = os.path.join(output_folder, basename)
            if not os.path.exists(rgb_path + ext):
                break
            seq += 1

        if output_format.upper() == "DNG":
            rgb_path = write_dng_linear(result, rgb_path)
        else:
            rgb_path = write_tiff_16bit(result, rgb_path)

        return rgb_path
