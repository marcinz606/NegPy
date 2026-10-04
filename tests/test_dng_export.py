"""The DNG export: a DNG 1.4 LinearRaw file both Adobe-style readers and NegPy's
own LinearRaw loader accept, in two lossless variants (float16 + Deflate, and
16-bit integer uncompressed). Pixels go out scene-linear in working-space
primaries; the ICC leg never runs."""

import io
import os
import tempfile

import numpy as np
import pytest
import rawpy
import tifffile

from negpy.domain.models import ColorSpace, DngVariant, ExportConfig, ExportFormat, WorkspaceConfig, preset_from_export_config
from negpy.kernel.image.logic import working_oetf_encode
from negpy.services.export.dng import _COLOR_MATRIX_1, UNIQUE_CAMERA_MODEL, encode_dng
from negpy.services.rendering.image_processor import ImageProcessor


def _linear(seed: int = 7, shape: tuple = (48, 64, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.05 + 0.9 * rng.random(shape)).astype(np.float32)


def _subifd(data: bytes) -> tifffile.TiffPage:
    tf = tifffile.TiffFile(io.BytesIO(data))
    return tf.pages[0].pages[0]


class TestEncodeDng:
    @pytest.mark.parametrize("float_variant", [True, False])
    def test_layout_and_tags(self, float_variant):
        data = encode_dng(_linear(), float_variant=float_variant)
        with tifffile.TiffFile(io.BytesIO(data)) as tf:
            page0 = tf.pages[0]
            assert bytes(page0.tags[50706].value) == bytes([1, 4, 0, 0])  # DNGVersion
            assert page0.tags[50708].value == UNIQUE_CAMERA_MODEL
            assert page0.tags[50721].count == 9  # ColorMatrix1
            assert page0.tags[50778].value == 21  # CalibrationIlluminant1: D65
            assert page0.tags[50936].value == "NegPy Linear"  # ProfileName
            assert tuple(page0.tags[50940].value) == (0.0, 0.0, 1.0, 1.0)  # identity ProfileToneCurve
            neutral = page0.tags[50728].value
            assert tuple(neutral[0::2]) == (1, 1, 1) and tuple(neutral[1::2]) == (1, 1, 1)
            assert page0.subfiletype == 1  # the thumbnail
            sub = page0.pages[0]
            assert int(sub.photometric) == 34892  # LinearRaw
            assert sub.samplesperpixel == 3
            assert sub.shape == (48, 64, 3)
            if float_variant:
                assert sub.dtype == np.float16
                assert int(sub.compression) == 8  # Deflate
                assert int(sub.predictor) == 3  # floating-point predictor
            else:
                assert sub.dtype == np.uint16
                assert int(sub.compression) == 1
                assert sub.tags[50717].value == (65535, 65535, 65535)  # WhiteLevel

    def test_color_matrix_inverts_working_primaries(self):
        # ColorMatrix1 maps XYZ under CalibrationIlluminant1 (D65) to the working
        # space: the D65 white must come back as equal RGB, which is what
        # AsShotNeutral=(1,1,1) asserts.
        white_d65 = np.array([0.95047, 1.0, 1.08883])
        rgb = _COLOR_MATRIX_1 @ white_d65
        assert np.allclose(rgb, rgb[0], atol=1e-4)

    @pytest.mark.parametrize("float_variant", [True, False])
    def test_pixels_round_trip(self, float_variant):
        lin = _linear()
        sub = _subifd(encode_dng(lin, float_variant=float_variant))
        arr = sub.asarray().astype(np.float32)
        if not float_variant:
            arr /= 65535.0
        tol = 5e-4 if float_variant else 2e-5  # float16 mantissa vs uint16 quantization
        assert np.abs(arr - lin).max() < tol

    def test_libraw_decodes_the_integer_variant_pixels(self):
        # Values, not just shape: a LibRaw build without the DNG SDK opens a float
        # DNG black with no error, which a shape assertion cannot catch.
        lin = np.full((48, 64, 3), 0.5, dtype=np.float32)
        data = encode_dng(lin, float_variant=False)
        with tempfile.NamedTemporaryFile(suffix=".dng", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, user_wb=[1, 1, 1, 1], output_bps=16)
            assert rgb.shape == (48, 64, 3)
            assert abs(float(rgb.mean()) - 32768.0) < 300
        finally:
            os.unlink(path)

    def test_float_variant_reads_back_through_negpy_loader(self):
        # LibRaw cannot decode floating-point DNG, so NegPy's own tifffile path is
        # the float variant's read contract.
        from negpy.infrastructure.loaders.rawpy_loader import _peek_linear_dng_rgb

        lin = _linear()
        data = encode_dng(lin, float_variant=True)
        with tempfile.NamedTemporaryFile(suffix=".dng", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            peeked = _peek_linear_dng_rgb(path)
            assert peeked is not None
            rgb, _wb = peeked
            assert np.abs(rgb - lin).max() < 5e-4
        finally:
            os.unlink(path)

    def test_the_default_variant_is_integer(self):
        sub = _subifd(encode_dng(_linear()))
        assert sub.dtype == np.uint16

    def test_negpy_reader_finds_the_subifd_page(self):
        from negpy.infrastructure.loaders.rawpy_loader import _find_linearraw_page

        data = encode_dng(_linear(), float_variant=True)
        with tifffile.TiffFile(io.BytesIO(data)) as tf:
            page = _find_linearraw_page(tf, samples=3)
            assert page is not None
            assert page.shape == (48, 64, 3)


class TestEncodeExportDispatch:
    def _settings(self, variant: DngVariant):
        return preset_from_export_config(ExportConfig(export_fmt=ExportFormat.DNG, dng_variant=variant))

    @pytest.mark.parametrize("variant", [DngVariant.FLOAT, DngVariant.INTEGER])
    def test_dng_branch_bypasses_icc_and_decodes_the_oetf(self, variant):
        lin = _linear(shape=(32, 40, 3))
        encoded = np.asarray(working_oetf_encode(lin), dtype=np.float32)
        proc = ImageProcessor()
        bits, ext = proc._encode_export(encoded, self._settings(variant), ColorSpace.SRGB.value, "Adobe RGB")
        assert ext == "dng"
        sub = _subifd(bits)
        arr = sub.asarray().astype(np.float32)
        if variant == DngVariant.INTEGER:
            arr /= 65535.0
        # The export target color space above is ignored: pixels equal the pre-OETF linear.
        tol = 5e-4 if variant == DngVariant.FLOAT else 1e-4
        assert np.abs(arr - lin).max() < tol

    def test_greyscale_buffer_is_stacked(self):
        lin = _linear(shape=(24, 30))
        encoded = np.asarray(working_oetf_encode(lin), dtype=np.float32)
        proc = ImageProcessor()
        bits, _ = proc._encode_export(encoded, self._settings(DngVariant.INTEGER), ColorSpace.GREYSCALE.value, "Adobe RGB")
        assert _subifd(bits).shape == (24, 30, 3)


class TestConfigPlumbing:
    def test_dng_survives_a_config_round_trip(self):
        flat = WorkspaceConfig().to_dict()
        flat.update({"export_fmt": "DNG", "dng_variant": "integer"})
        c = WorkspaceConfig.from_flat_dict(flat)
        assert c.export.export_fmt == ExportFormat.DNG
        assert c.export.dng_variant == DngVariant.INTEGER

    def test_unknown_variant_coerces_to_integer(self):
        flat = WorkspaceConfig().to_dict()
        flat["dng_variant"] = "bogus"
        assert WorkspaceConfig.from_flat_dict(flat).export.dng_variant == DngVariant.INTEGER

    def test_preset_coerces_an_unknown_variant_too(self):
        from negpy.domain.models import ExportPreset

        preset = ExportPreset.from_dict({"dng_variant": "bogus"})
        assert preset.dng_variant == DngVariant.INTEGER

    def test_preset_carries_the_variant(self):
        preset = preset_from_export_config(ExportConfig(export_fmt=ExportFormat.DNG, dng_variant=DngVariant.INTEGER))
        assert preset.dng_variant == DngVariant.INTEGER
        assert preset.to_dict()["dng_variant"] == DngVariant.INTEGER
