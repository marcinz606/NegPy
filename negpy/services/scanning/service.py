import os
import re
import threading
from typing import Callable

from negpy.infrastructure.scanners.base import ScannerBackend, ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.scanners.sane_backend import SaneBackend
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _make_sequence_number(folder: str, date: str) -> int:
    """Find the next sequence number for a (folder, date) pair.

    Scans existing files matching `scan_{date}_*.{tif,tiff,dng}` and returns max+1.
    """
    if not folder or not os.path.isdir(folder):
        return 1
    pattern = re.compile(rf"^scan_{re.escape(date)}_(\d+)\.(?:tiff?|dng)$", re.IGNORECASE)
    max_seq = 0
    try:
        for entry in os.scandir(folder):
            m = pattern.match(entry.name)
            if m:
                seq = int(m.group(1))
                if seq > max_seq:
                    max_seq = seq
    except OSError:
        pass
    return max_seq + 1


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

        Filename pattern supports {date} and {seq:03d} placeholders.
        """
        from datetime import date as dt_date

        from negpy.services.scanning.writer import write_dng_linear, write_tiff_16bit

        os.makedirs(output_folder, exist_ok=True)

        date_str = dt_date.today().strftime("%Y%m%d")
        seq = _make_sequence_number(output_folder, date_str)

        # Replace placeholders
        basename = filename_pattern
        basename = basename.replace("{date}", date_str)
        # Handle {seq:03d} and {seq}
        basename = re.sub(r"\{seq:0?(\d+)d\}", lambda m: str(seq).zfill(int(m.group(1))), basename)
        basename = basename.replace("{seq}", str(seq))

        rgb_path = os.path.join(output_folder, basename)

        if output_format.upper() == "DNG":
            rgb_path = write_dng_linear(result, rgb_path)
        else:
            rgb_path = write_tiff_16bit(result, rgb_path)

        return rgb_path
