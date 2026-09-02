"""How a scanner TIFF is encoded, and what the loader makes of it."""

import base64
import os
import tempfile

import numpy as np
import tifffile
from PIL import ImageCms

from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.kernel.image.logic import srgb_to_linear, working_oetf_decode
from negpy.features.process.logic import effective_linear_raw
from negpy.features.process.models import ProcessConfig, ProcessMode

# The standard Adobe RGB (1998) ICC profile — real-world scanner/export software tags
# TIFFs with exactly this, so the loader's Adobe RGB branch is tested against what
# actually reaches it, not a synthetic stand-in.
_ADOBE_RGB_ICC = base64.b64decode(
    "AAACMEFEQkUCEAAAbW50clJHQiBYWVogB88ABgADAAAAAAAAYWNzcEFQUEwAAAAAbm9uZQAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1BR"
    "EJFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKY3BydAAAAPwAAAAyZGVzYwAAATAAAABrd3RwdAAA"
    "AZwAAAAUYmtwdAAAAbAAAAAUclRSQwAAAcQAAAAOZ1RSQwAAAdQAAAAOYlRSQwAAAeQAAAAOclhZWgAAAfQAAAAUZ1hZWgAAAggAAAAUYlhZ"
    "WgAAAhwAAAAUdGV4dAAAAABDb3B5cmlnaHQgMTk5OSBBZG9iZSBTeXN0ZW1zIEluY29ycG9yYXRlZAAAAGRlc2MAAAAAAAAAEUFkb2JlIFJ"
    "HQiAoMTk5OCkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAFhZWiAAAAAAAADzUQABAAAAARbMWFlaIAAAAAAAAAAAAAAAAAAAAABjdXJ2AAAAAAAAAAECMwAAY3VydgAAAAAAAAAB"
    "AjMAAGN1cnYAAAAAAAAAAQIzAABYWVogAAAAAAAAnBgAAE+lAAAE/FhZWiAAAAAAAAA0jQAAoCwAAA+VWFlaIAAAAAAAACYxAAAQLwAAvpw="
)


def _rgb16(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # dark linear-scan range, where the sRGB toe distorts most
    return rng.integers(0, 30000, (32, 48, 3), dtype=np.uint16)


def _load(path: str, linear_raw: bool = False, positive_source: bool = False) -> tuple[np.ndarray, dict]:
    ctx, metadata = TiffLoader().load(path, linear_raw=linear_raw, positive_source=positive_source)
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

    def test_untagged_uint16_positive_source_gets_srgb_decode(self) -> None:
        """The ambiguity an untagged 16-bit file usually carries — scanner-linear or a
        finished positive — is already resolved once the caller says positive_source."""
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "positivized.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path, positive_source=True)
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 65535.0), atol=1e-6)
            assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_positive_source_does_not_override_an_actual_tag(self) -> None:
        """A real profile still wins — positive_source only fills the gap an untagged
        file leaves, it does not second-guess a tag that is already there."""
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tagged.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])
            f32, metadata = _load(path, positive_source=True)
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 65535.0), atol=1e-6)
            assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_positive_source_off_keeps_the_untagged_default(self) -> None:
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scan.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path, positive_source=False)
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
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

    def test_adobe_rgb_icc_uint16_gets_the_working_oetf_decode(self) -> None:
        """Adobe RGB's TRC is the working space's own gamma, so it decodes through
        working_oetf_decode rather than being merely identified and left un-decoded."""
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tagged.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(_ADOBE_RGB_ICC), _ADOBE_RGB_ICC, True)])
            f32, metadata = _load(path)
            expected = working_oetf_decode(data.astype(np.float32) / 65535.0)
            np.testing.assert_allclose(f32, expected, atol=1e-6)
            assert metadata["color_space"] == ColorSpace.ADOBE_RGB.value

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

    def test_linear_raw_still_wins_over_positive_source(self) -> None:
        """An explicit request for literal linear data overrides positive_source too,
        same precedence as effective_linear_raw's own stored-flag check."""
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scan.tif")
            tifffile.imwrite(path, data, photometric="rgb")
            f32, metadata = _load(path, linear_raw=True, positive_source=True)
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


class TestPositiveSourceOnTheTransferPath:
    """An already-positive TIFF loaded as Transparency (Normalize off) is not a raw
    scanner capture: Positive must reach the loader through effective_linear_raw
    so its sRGB tag decodes instead of being read as literal linear data."""

    def test_positive_source_reaches_the_srgb_decode(self) -> None:
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "positivized.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])

            process = ProcessConfig(process_mode=ProcessMode.E6, e6_normalize=False, positive_source=True)
            f32, metadata = _load(path, linear_raw=effective_linear_raw(process))
            np.testing.assert_allclose(f32, srgb_to_linear(data.astype(np.float32) / 65535.0), atol=1e-6)
            assert metadata["color_space"] == ColorSpace.SRGB.value

    def test_without_positive_source_the_tag_is_still_ignored(self) -> None:
        """The default: the same frame, minus the toggle, keeps today's forced-linear read."""
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        data = _rgb16()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "positivized.tif")
            tifffile.imwrite(path, data, photometric="rgb", extratags=[(34675, 7, len(icc), icc, True)])

            process = ProcessConfig(process_mode=ProcessMode.E6, e6_normalize=False, positive_source=False)
            f32, metadata = _load(path, linear_raw=effective_linear_raw(process))
            np.testing.assert_allclose(f32, data.astype(np.float32) / 65535.0, atol=1e-7)
            assert metadata["color_space"] is None


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
