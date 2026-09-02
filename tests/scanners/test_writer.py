"""Tests for the TIFF output writer."""

import os
import tempfile

import numpy as np
import tifffile

from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.writer import write_tiff_16bit


class TestTiffWriter:
    def test_writes_16bit_tiff(self) -> None:
        rgb = np.random.randint(0, 65535, (200, 300, 3), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=None, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "test_scan"))
            assert os.path.exists(path)
            assert path.endswith(".tif")

            # Round-trip readback
            readback = tifffile.imread(path)
            assert readback.shape == (200, 300, 3)
            assert readback.dtype == np.uint16

    def test_writes_ir_sidecar(self) -> None:
        rgb = np.random.randint(0, 65535, (100, 150, 3), dtype=np.uint16)
        ir = np.random.randint(0, 65535, (100, 150), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=ir, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "test_ir"))
            ir_path = path.replace(".tif", "_IR.tif")
            assert os.path.exists(path)
            assert os.path.exists(ir_path)

            ir_readback = tifffile.imread(ir_path)
            assert ir_readback.shape == (100, 150)

    def test_writes_ir_valid_sidecar(self) -> None:
        rgb = np.random.randint(0, 65535, (20, 30, 3), dtype=np.uint16)
        ir = np.random.randint(0, 65535, (20, 30), dtype=np.uint16)
        valid = np.ones((20, 30), dtype=bool)
        valid[5, 6] = False
        result = ScanResult(rgb=rgb, ir=ir, dpi=3600, device_model="T", ir_valid_mask=valid)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "test_valid"))
            valid_path = path.replace(".tif", "_IR_VALID.tif")
            assert os.path.exists(valid_path)

            readback = tifffile.imread(valid_path)
            assert readback.dtype == np.uint8
            assert readback[0, 0] == 255
            assert readback[5, 6] == 0

    def test_no_ir_valid_sidecar_when_mask_absent(self) -> None:
        rgb = np.random.randint(0, 65535, (10, 10, 3), dtype=np.uint16)
        ir = np.random.randint(0, 65535, (10, 10), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=ir, dpi=3600, device_model="T")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "test_novalid"))
            assert not os.path.exists(path.replace(".tif", "_IR_VALID.tif"))

    def test_adds_tif_extension(self) -> None:
        rgb = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = ScanResult(rgb=rgb, ir=None, dpi=300, device_model="T")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "noext"))
            assert path.endswith(".tif")

    def test_converts_non_uint16(self) -> None:
        rgb = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = ScanResult(rgb=rgb, ir=None, dpi=300, device_model="T")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "test8"))
            readback = tifffile.imread(path)
            assert readback.dtype == np.uint16


class TestHotFolderSeesOnlyFinishedScans:
    """The output folder may be watched, and it indexes on extension alone."""

    def test_the_tiff_being_written_is_not_offered_to_the_watcher(self, monkeypatch) -> None:
        from negpy.infrastructure.filesystem.watcher import FolderWatchService

        rgb = np.random.randint(0, 65535, (60, 80, 3), dtype=np.uint16)
        ir = np.random.randint(0, 65535, (60, 80), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=ir, dpi=3600, device_model="TestScanner")
        offered: list[str] = []
        real_imwrite = tifffile.imwrite

        with tempfile.TemporaryDirectory() as tmpdir:

            def watching_imwrite(file, data, **kwargs):
                # Probe with the half-written file still on disk, before its rename.
                real_imwrite(file, data, **kwargs)
                found = FolderWatchService.scan_for_new_files(tmpdir, set())
                assert os.path.abspath(str(file)) not in found
                offered.extend(found)

            monkeypatch.setattr(tifffile, "imwrite", watching_imwrite)
            path = write_tiff_16bit(result, os.path.join(tmpdir, "scan_001"))

            assert offered  # the probe ran, and only ever saw finished scans
            assert FolderWatchService.scan_for_new_files(tmpdir, set()) == [os.path.abspath(path)]


class TestMonoTiff:
    def test_writes_one_grey_plane(self) -> None:
        rgb = np.random.randint(0, 65535, (40, 60, 3), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=None, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "mono"), mono=True)

            readback = tifffile.imread(path)
            assert readback.shape == (40, 60)
            assert readback.dtype == np.uint16
            with tifffile.TiffFile(path) as tf:
                assert int(tf.pages[0].tags["PhotometricInterpretation"].value) == 1  # minisblack

    def test_the_grey_plane_is_the_mean_of_the_three(self) -> None:
        rgb = np.array([[[0, 0, 0], [10, 11, 12], [65535, 65535, 65535], [1, 2, 2]]], dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=None, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "mean"), mono=True)
            readback = tifffile.imread(path)

        # Rounded, not floored: 5/3 reads 2, and the top of the range does not wrap.
        assert readback.tolist() == [[0, 11, 65535, 2]]

    def test_the_ir_sidecar_still_comes_out_beside_it(self) -> None:
        rgb = np.random.randint(0, 65535, (20, 30, 3), dtype=np.uint16)
        ir = np.random.randint(0, 65535, (20, 30), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=ir, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tiff_16bit(result, os.path.join(tmpdir, "mono_ir"), mono=True)
            assert os.path.exists(path.replace(".tif", "_IR.tif"))
