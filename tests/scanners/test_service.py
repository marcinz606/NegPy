"""Tests for ScannerService with a FakeBackend."""

import os
import threading
import time

import numpy as np
import pytest

from negpy.infrastructure.scanners.base import (
    ScanMode,
    ScannerCapabilities,
    ScannerDevice,
    TransientScanError,
)
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.service import _SCAN_IO_RETRY_ATTEMPTS, ScannerService


class FakeBackend:
    """In-memory ScannerBackend for testing."""

    def __init__(self, devices: list[ScannerDevice] | None = None) -> None:
        self._devices = devices or []
        self._should_raise: Exception | None = None
        self._scan_delay: float = 0.0
        self.scan_calls: int = 0
        self.transient_failures: int = 0  # raise a transient I/O error this many times, then succeed

    def list_devices(self) -> list[ScannerDevice]:
        if self._should_raise:
            raise self._should_raise
        return self._devices

    def refresh_devices(self) -> list[ScannerDevice]:
        return self.list_devices()

    def eject(self, device_id: str) -> bool:
        return False

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress,
        cancel: threading.Event,
    ) -> ScanResult:
        self.scan_calls += 1
        if self.transient_failures > 0:
            self.transient_failures -= 1
            raise TransientScanError("RGB scan failed: Error during device I/O")
        if self._should_raise:
            raise self._should_raise

        if progress:
            progress(0.0)

        # Simulate scan work
        if self._scan_delay > 0 and not cancel.is_set():
            time.sleep(min(self._scan_delay, 0.5))

        if cancel.is_set():
            raise RuntimeError("Scan cancelled")

        h, w = 100, 150
        rgb = np.ones((h, w, 3), dtype=np.uint16) * 30000

        ir = None
        if params.capture_ir:
            ir = np.ones((h, w), dtype=np.uint16) * 10000

        if progress:
            progress(1.0)

        return ScanResult(rgb=rgb, ir=ir, dpi=params.dpi, device_model="FakeScanner")


@pytest.fixture
def fake_caps() -> ScannerCapabilities:
    return ScannerCapabilities(
        ir_channel=True,
        supported_dpi=(300, 600, 1200, 2400, 3600),
        supported_depths=(8, 16),
        sources=(ScanMode.NEGATIVE, ScanMode.POSITIVE, ScanMode.TRANSPARENCY),
        max_area_mm=(36.0, 25.0),
    )


@pytest.fixture
def fake_device(fake_caps: ScannerCapabilities) -> ScannerDevice:
    return ScannerDevice(id="fake:001", vendor="FakeCorp", model="ScanMaster 9000", capabilities=fake_caps)


class TestScannerServiceWithFakeBackend:
    def test_list_devices(self, fake_device: ScannerDevice) -> None:
        service = ScannerService()
        service._backend = FakeBackend(devices=[fake_device])
        devices = service.list_devices()
        assert len(devices) == 1
        assert devices[0].id == "fake:001"
        assert devices[0].vendor == "FakeCorp"

    def test_run_scan(self, fake_device: ScannerDevice) -> None:
        service = ScannerService()
        service._backend = FakeBackend(devices=[fake_device])

        params = ScanParams(dpi=1200, depth=16, capture_ir=False)
        progress_values: list[float] = []
        cancel = threading.Event()

        result = service.run_scan(fake_device.id, params, lambda p: progress_values.append(p), cancel)

        assert result.rgb.shape == (100, 150, 3)
        assert result.rgb.dtype == np.uint16
        assert result.ir is None
        assert result.dpi == 1200
        assert progress_values == [0.0, 1.0]

    def test_scan_with_ir(self, fake_device: ScannerDevice) -> None:
        service = ScannerService()
        service._backend = FakeBackend(devices=[fake_device])

        params = ScanParams(dpi=2400, depth=16, capture_ir=True)
        cancel = threading.Event()

        result = service.run_scan(fake_device.id, params, lambda _: None, cancel)

        assert result.ir is not None
        assert result.ir.shape == (100, 150)

    def test_cancel_scan(self, fake_device: ScannerDevice) -> None:
        service = ScannerService()
        backend = FakeBackend(devices=[fake_device])
        backend._scan_delay = 5.0  # Long delay
        service._backend = backend

        params = ScanParams(dpi=1200, depth=16, capture_ir=False)
        cancel = threading.Event()

        # Set cancel immediately
        cancel.set()
        with pytest.raises(RuntimeError, match="Scan cancelled"):
            service.run_scan(fake_device.id, params, lambda _: None, cancel)

    def test_no_devices_returns_empty(self) -> None:
        service = ScannerService()
        service._backend = FakeBackend(devices=[])
        devices = service.list_devices()
        assert devices == []

    def test_run_scan_retries_once_on_transient_device_io(self, fake_device: ScannerDevice) -> None:
        service = ScannerService()
        backend = FakeBackend(devices=[fake_device])
        backend.transient_failures = 1  # one glitch, then a clean scan
        service._backend = backend

        params = ScanParams(dpi=1200, depth=16, capture_ir=False)
        result = service.run_scan(fake_device.id, params, lambda _: None, threading.Event(), retry_delay=0)

        assert result.rgb.shape == (100, 150, 3)
        assert backend.scan_calls == 2

    def test_run_scan_gives_up_after_bounded_transient_retries(self, fake_device: ScannerDevice) -> None:
        service = ScannerService()
        backend = FakeBackend(devices=[fake_device])
        backend.transient_failures = 99  # never recovers
        service._backend = backend

        params = ScanParams(dpi=1200, depth=16, capture_ir=False)
        with pytest.raises(RuntimeError, match="device I/O"):
            service.run_scan(fake_device.id, params, lambda _: None, threading.Event(), retry_delay=0)
        assert backend.scan_calls == _SCAN_IO_RETRY_ATTEMPTS  # bounded — not an infinite loop

    def test_run_scan_does_not_retry_a_non_transient_error(self, fake_device: ScannerDevice) -> None:
        service = ScannerService()
        backend = FakeBackend(devices=[fake_device])
        backend._should_raise = RuntimeError("Could not set frame=3")
        service._backend = backend

        params = ScanParams(dpi=1200, depth=16, capture_ir=False)
        with pytest.raises(RuntimeError, match="Could not set frame"):
            service.run_scan(fake_device.id, params, lambda _: None, threading.Event(), retry_delay=0)
        assert backend.scan_calls == 1  # a real error fails fast


class TestRenderScanFilename:
    def test_basic_template(self) -> None:
        from negpy.services.scanning.templating import render_scan_filename

        result = render_scan_filename('{{ date }}_{{ "%03d" % seq }}', "20260511", 1)
        assert result == "20260511_001"

    def test_seq_increments(self) -> None:
        from negpy.services.scanning.templating import render_scan_filename

        assert render_scan_filename('{{ date }}_{{ "%03d" % seq }}', "20260511", 5) == "20260511_005"

    def test_invalid_template_falls_back(self) -> None:
        from negpy.services.scanning.templating import render_scan_filename

        result = render_scan_filename("{{ unclosed", "20260511", 1)
        assert result == "20260511_001"

    def test_no_overwrite_increments(self) -> None:
        import tempfile

        import numpy as np

        from negpy.infrastructure.scanners.result import ScanResult

        h, w = 10, 10
        result = ScanResult(rgb=np.zeros((h, w, 3), dtype=np.uint16), ir=None, dpi=300, device_model="Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            service = ScannerService()
            service._backend = FakeBackend()
            pattern = '{{ date }}_{{ "%03d" % seq }}'

            path1 = service.write_result(result, tmpdir, pattern, "TIFF")
            path2 = service.write_result(result, tmpdir, pattern, "TIFF")

            assert os.path.exists(path1)
            assert os.path.exists(path2)
            assert path1 != path2

    def test_write_refuses_a_pattern_that_does_not_vary_with_sequence(self) -> None:
        import tempfile

        import numpy as np

        from negpy.infrastructure.scanners.result import ScanResult

        result = ScanResult(rgb=np.zeros((4, 4, 3), dtype=np.uint16), ir=None, dpi=300, device_model="Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            service = ScannerService()
            service._backend = FakeBackend()
            with pytest.raises(ValueError, match="different basename"):
                service.write_result(result, tmpdir, "fixed_name", "TIFF")
