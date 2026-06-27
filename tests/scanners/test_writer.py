"""Tests for TIFF and DNG output writers."""

import os
import tempfile

import numpy as np
import tifffile

from negpy.infrastructure.scanners.result import ScanResult
from negpy.services.scanning.writer import write_dng_linear, write_tiff_16bit


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


class TestDngWriter:
    def test_writes_linear_dng(self) -> None:
        rgb = np.random.randint(0, 65535, (200, 300, 3), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=None, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dng_linear(result, os.path.join(tmpdir, "test_scan"))
            assert os.path.exists(path)
            assert path.endswith(".dng")

            readback = tifffile.imread(path)
            assert readback.shape == (200, 300, 3)
            assert readback.dtype == np.uint16
            np.testing.assert_array_equal(readback, rgb)

            with tifffile.TiffFile(path) as tf:
                tags = tf.pages[0].tags
                assert int(tags["PhotometricInterpretation"].value) == 34892  # LinearRaw
                assert tuple(tags["DNGVersion"].value) == (1, 4, 0, 0)
                assert int(tags["SamplesPerPixel"].value) == 3
                # 3 plain colour samples, no ExtraSamples (matches pidng); marking colour
                # planes as extra makes some raw processors mis-demosaic the file.
                assert tags.get("ExtraSamples") is None

    def test_writes_dng_with_ir(self) -> None:
        rgb = np.random.randint(0, 65535, (100, 150, 3), dtype=np.uint16)
        ir = np.random.randint(0, 65535, (100, 150), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=ir, dpi=3600, device_model="TestScanner")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dng_linear(result, os.path.join(tmpdir, "test_ir"))
            assert os.path.exists(path)

            readback = tifffile.imread(path)
            assert readback.shape == (100, 150, 4)
            np.testing.assert_array_equal(readback[:, :, :3], rgb)
            np.testing.assert_array_equal(readback[:, :, 3], ir)
            with tifffile.TiffFile(path) as tf:
                assert int(tf.pages[0].tags["SamplesPerPixel"].value) == 4

    def test_adds_dng_extension(self) -> None:
        rgb = np.random.randint(0, 65535, (50, 50, 3), dtype=np.uint16)
        result = ScanResult(rgb=rgb, ir=None, dpi=300, device_model="T")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dng_linear(result, os.path.join(tmpdir, "noext"))
            assert path.endswith(".dng")
