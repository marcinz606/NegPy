import os
import threading
from typing import Callable

from negpy.infrastructure.scanners.base import ScannerBackend, ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.infrastructure.scanners.sane_backend import SaneBackend
from negpy.kernel.system.logging import get_logger
from negpy.services.scanning.templating import (
    render_scan_filename,
    require_sequence_varying_scan_filename,
)

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
        refresh = getattr(backend, "refresh_devices", None)
        if callable(refresh):
            return refresh()
        return backend.list_devices()

    def probe_device(self, device_id: str) -> ScannerDevice:
        """Return one device from a fresh backend enumeration."""

        backend = self._get_backend()
        try:
            strict_probe = getattr(backend, "probe_device", None)
            if callable(strict_probe):
                device = strict_probe(device_id)
                if device is not None:
                    return device
                devices: list[ScannerDevice] = []
            else:
                devices = self.refresh_devices()
        except Exception as exc:
            raise RuntimeError(f"Could not probe scanner device {device_id!r}: fresh enumeration failed: {exc}") from exc

        for device in devices:
            if device.id == device_id:
                return device
        raise RuntimeError(f"Scanner device {device_id!r} was not found during fresh enumeration")

    def run_scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult:
        backend = self._get_backend()
        return backend.scan(device_id, params, progress, cancel)

    def eject(self, device_id: str) -> bool:
        """Trigger a capability-gated film eject; False when unsupported.

        Mirrors the optional-method pattern in refresh_devices/probe_device
        above — only SaneBackend implements this today.
        """
        backend = self._get_backend()
        eject = getattr(backend, "eject", None)
        if not callable(eject):
            return False
        return bool(eject(device_id))

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
        require_sequence_varying_scan_filename(filename_pattern, date_str, seq)
        seen_basenames: set[str] = set()
        while True:
            basename = render_scan_filename(filename_pattern, date_str, seq)
            if basename in seen_basenames:
                raise ValueError(
                    "filename pattern repeated a basename while resolving an output collision"
                )
            seen_basenames.add(basename)
            rgb_path = os.path.join(output_folder, basename)
            if output_format.upper() == "DNG":
                reserved_paths = (rgb_path + ext,)
            else:
                reserved_paths = (
                    rgb_path + ext,
                    rgb_path + "_IR.tif",
                    rgb_path + "_IR_VALID.tif",
                    rgb_path + "_SCAN.json",
                )
            if not any(os.path.lexists(path) for path in reserved_paths):
                break
            seq += 1

        if output_format.upper() == "DNG":
            rgb_path = write_dng_linear(result, rgb_path)
        else:
            rgb_path = write_tiff_16bit(result, rgb_path)

        return rgb_path
