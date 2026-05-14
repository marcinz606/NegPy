"""Tests for ScannerService with a FakeBackend."""

import os
import threading
import time

import numpy as np
import pytest

from negpy.infrastructure.scanners.base import ScanMode, ScannerCapabilities, ScannerDevice
from negpy.infrastructure.scanners.params import ScanParams
from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.service import ScannerService


class FakeBackend:
    """In-memory ScannerBackend for testing."""

    def __init__(self, devices: list[ScannerDevice] | None = None) -> None:
        self._devices = devices or []
        self._should_raise: Exception | None = None
        self._scan_delay: float = 0.0

    def list_devices(self) -> list[ScannerDevice]:
        if self._should_raise:
            raise self._should_raise
        return self._devices

    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress,
        cancel: threading.Event,
    ) -> ScanResult:
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


class TestSequenceNumber:
    def test_empty_folder_returns_1(self) -> None:
        from negpy.services.scanning.service import _make_sequence_number

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            assert _make_sequence_number(tmpdir, "20260511") == 1

    def test_existing_files_increment(self) -> None:
        from negpy.services.scanning.service import _make_sequence_number

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(1, 4):
                open(os.path.join(tmpdir, f"scan_20260511_{i:03d}.tif"), "w").close()
            assert _make_sequence_number(tmpdir, "20260511") == 4

    def test_ignores_other_dates(self) -> None:
        from negpy.services.scanning.service import _make_sequence_number

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "scan_20260510_001.tif"), "w").close()
            open(os.path.join(tmpdir, "scan_20260510_002.tif"), "w").close()
            assert _make_sequence_number(tmpdir, "20260511") == 1

    def test_ignores_different_patterns(self) -> None:
        from negpy.services.scanning.service import _make_sequence_number

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "other_file.tif"), "w").close()
            open(os.path.join(tmpdir, "scan_20260511_001.jpg"), "w").close()
            assert _make_sequence_number(tmpdir, "20260511") == 1
