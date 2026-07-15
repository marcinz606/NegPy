"""Scanner TIFFs must decode identically to their LinearRaw DNG twins."""

import os
import tempfile

import numpy as np
import tifffile
from PIL import ImageCms

from negpy.domain.models import ColorSpace
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.image.logic import srgb_to_linear
from negpy.services.scanning.writer import write_dng_linear, write_tiff_16bit


def _rgb16(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # dark linear-scan range, where the sRGB toe distorts most
    return rng.integers(0, 30000, (32, 48, 3), dtype=np.uint16)


def _load(path: str) -> tuple[np.ndarray, dict]:
    ctx, metadata = TiffLoader().load(path)
    with ctx as raw:
        return raw.data, metadata


class TestTiffEncodingAssumptions:
    def test_untagged_uint16_reads_linear(self) -> None:
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scan.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
            assert metadata["color_space"] == ColorSpace.ADOBE_RGB.value

    def test_untagged_uint8_gets_srgb_decode(self) -> None:
        data = np.linspace(0, 255, 32 * 48 * 3).reshape(32, 48, 3).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "photo.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path)
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 255.0), atol=1e-6)
            assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_srgb_icc_uint16_gets_srgb_decode(self) -> None:
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tagged.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])
            f32, metadata = _load(path)
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 65535.0), atol=1e-6)
            assert metadata["color_space"] == ColorSpace.SRGB.value


class TestScanRoundTripParity:
    def test_tiff_and_dng_decode_identically(self) -> None:
        from negpy.services.rendering.image_processor import ImageProcessor

        result = ScanResult(rgb=_rgb16(), ir=None, dpi=3600, device_model="TestScanner")
        proc = ImageProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = write_tiff_16bit(result, os.path.join(tmpdir, "pair"))
            dng_path = write_dng_linear(result, os.path.join(tmpdir, "pair"))
            tif_rgb, _ = proc._decode_sensor_rgb(tif_path, linear_raw=True)
            dng_rgb, _ = proc._decode_sensor_rgb(dng_path, linear_raw=True)
        np.testing.assert_array_equal(tif_rgb, dng_rgb)


class TestWrapperGamma:
    def test_gamma_1_1_is_linear_passthrough(self) -> None:
        data = np.linspace(0.0, 1.0, 300, dtype=np.float32).reshape(10, 10, 3)
        out = NonStandardFileWrapper(data).postprocess(gamma=(1, 1), output_bps=16)
        np.testing.assert_array_equal(out, (data * 65535.0).astype(np.uint16))

    def test_default_gamma_applies_bt709_encode(self) -> None:
        data = np.linspace(0.0, 1.0, 300, dtype=np.float32).reshape(10, 10, 3)
        out = NonStandardFileWrapper(data).postprocess(output_bps=16)
        expected = np.where(data < 0.018, data * 4.5, 1.099 * np.power(data, 1.0 / 2.222) - 0.099)
        np.testing.assert_allclose(out.astype(np.float32) / 65535.0, expected, atol=1.5 / 65535.0)
