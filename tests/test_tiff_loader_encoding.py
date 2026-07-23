"""Scanner TIFFs must decode identically to their LinearRaw DNG twins."""

import os
import tempfile

import numpy as np
import tifffile
from PIL import ImageCms

from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.infrastructure.scanners.result import ScanResult
from negpy.kernel.image.logic import srgb_to_linear
from negpy.services.scanning.writer import write_dng_linear, write_tiff_16bit


def _rgb16(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # dark linear-scan range, where the sRGB toe distorts most
    return rng.integers(0, 30000, (32, 48, 3), dtype=np.uint16)


def _load(path: str, linear_raw: bool = False) -> tuple[np.ndarray, dict]:
    ctx, metadata = TiffLoader().load(path, linear_raw=linear_raw)
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
            # Scanner-raw linear: no ColorSpace names it, so it must not claim one.
            assert metadata["color_space"] is None

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

    def test_linear_raw_ignores_srgb_icc_tag(self) -> None:
        """Linear RAW couples to the loader: a stitched/scanner TIFF that lies about
        being sRGB (see #588) must decode as literal linear data when it's on."""
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tagged.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])
            f32, metadata = _load(path, linear_raw=True)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
            assert metadata["color_space"] is None

    def test_linear_raw_ignores_untagged_uint8_srgb_assumption(self) -> None:
        data = np.linspace(0, 255, 32 * 48 * 3).reshape(32, 48, 3).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "photo.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path, linear_raw=True)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 255.0, atol=1e-7)
            assert metadata["color_space"] is None


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

    def test_tiff_and_dng_agree_on_same_as_source_target(self) -> None:
        """The twins must also export alike, not just decode alike."""
        from negpy.services.rendering.image_processor import ImageProcessor

        result = ScanResult(rgb=_rgb16(), ir=None, dpi=3600, device_model="TestScanner")
        proc = ImageProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = write_tiff_16bit(result, os.path.join(tmpdir, "pair"))
            dng_path = write_dng_linear(result, os.path.join(tmpdir, "pair"))
            _, tif_meta = proc._decode_sensor_rgb(tif_path, linear_raw=True)
            _, dng_meta = proc._decode_sensor_rgb(dng_path, linear_raw=True)
        assert tif_meta.get("color_space") is None
        assert dng_meta.get("color_space") is None


class TestUncharacterisedSourceResolvesToWorkingSpace:
    """A source with no embedded profile is already in the working space: "Same as
    Source" must export it without a needless conversion into a narrower gamut."""

    def test_none_resolves_to_working_space(self) -> None:
        source_cs = str({"color_space": None}.get("color_space") or WORKING_COLOR_SPACE)
        assert source_cs == WORKING_COLOR_SPACE

    def test_embedded_profile_still_wins(self) -> None:
        source_cs = str({"color_space": ColorSpace.SRGB.value}.get("color_space") or WORKING_COLOR_SPACE)
        assert source_cs == ColorSpace.SRGB.value

    def test_same_as_source_export_does_not_convert_uncharacterised_source(self) -> None:
        """The payoff: working == target, so the transform short-circuits."""
        from negpy.services.rendering.image_processor import ImageProcessor

        rng = np.random.default_rng(0)
        u16 = rng.integers(0, 65535, (16, 16, 3), dtype=np.uint16)
        proc = ImageProcessor()
        # source_cs for an untagged scan, as resolved above
        out, icc = proc._apply_color_management_u16_rgb(u16, WORKING_COLOR_SPACE, WORKING_COLOR_SPACE, None, None)
        np.testing.assert_array_equal(out, u16)
        assert icc is not None, "the file must still carry the working-space profile"


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
