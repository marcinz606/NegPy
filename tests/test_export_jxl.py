import numpy as np
import imagecodecs
import pytest

from negpy.domain.models import (
    JXL_TAGGABLE_SPACES,
    ColorSpace,
    ExportConfig,
    ExportFormat,
    ExportPreset,
    export_blocked,
    preset_from_export_config,
)
from negpy.kernel.image.logic import float_to_uint16, float_to_uint_luma
from negpy.services.rendering.image_processor import ImageProcessor, _JXL_COLOR


@pytest.fixture(scope="module")
def proc():
    return ImageProcessor()


def test_jxl_color_mapping_is_the_verified_table():
    """Pins the enumerated tags verified against the bundled ICC profiles. Rec 2020
    uses the Rec.709/BT.2020 OETF (BT709, not sRGB); GrayGamma2.2.icc carries the
    sRGB TRC despite its name (SRGB, not gamma 2.2)."""
    assert _JXL_COLOR == {
        "sRGB": ("RGB", "SRGB", "SRGB"),
        "P3 D65": ("RGB", "P3", "SRGB"),
        "Rec 2020": ("RGB", "BT2100", "BT709"),
        "Greyscale": ("GRAY", None, "SRGB"),
    }


def test_jxl_rgb_lossless_roundtrip_is_exact(proc):
    """Lossless JXL RGB export decodes back to the exact 16-bit samples. Working
    space == target sRGB makes colour management a no-op, isolating the codec."""
    buf = np.random.default_rng(0).random((16, 24, 3), dtype=np.float32)
    settings = ExportConfig(export_fmt=ExportFormat.JXL, jxl_lossless=True)

    bits, ext = proc._encode_export(buf, settings, ColorSpace.SRGB.value, working_color_space=ColorSpace.SRGB.value)
    assert ext == "jxl"
    decoded = imagecodecs.jpegxl_decode(bits)
    expected = float_to_uint16(buf)
    assert decoded.dtype == np.uint16
    assert decoded.shape == expected.shape
    np.testing.assert_array_equal(decoded, expected)


def test_jxl_greyscale_lossless_roundtrip_is_exact(proc):
    """Lossless JXL greyscale export decodes to the exact 16-bit luma (2D)."""
    buf = np.random.default_rng(1).random((16, 24, 3), dtype=np.float32)
    settings = ExportConfig(export_fmt=ExportFormat.JXL, jxl_lossless=True)

    bits, ext = proc._encode_export(buf, settings, ColorSpace.GREYSCALE.value, working_color_space=ColorSpace.GREYSCALE.value)
    assert ext == "jxl"
    decoded = imagecodecs.jpegxl_decode(bits)
    expected = float_to_uint_luma(np.ascontiguousarray(buf), bit_depth=16)
    assert decoded.dtype == np.uint16
    assert decoded.reshape(expected.shape).shape == expected.shape
    np.testing.assert_array_equal(decoded.reshape(expected.shape), expected)


def test_jxl_tag_is_actually_written(proc):
    """Different enumerated primaries must change the encoded stream — proof the tag
    is passed to libjxl, not silently dropped."""
    buf = np.random.default_rng(2).random((32, 32, 3), dtype=np.float32)
    s = ExportConfig(export_fmt=ExportFormat.JXL, jxl_lossless=True)
    srgb, _ = proc._encode_export(buf, s, ColorSpace.SRGB.value, working_color_space=ColorSpace.SRGB.value)
    rec2020, _ = proc._encode_export(buf, s, ColorSpace.REC2020.value, working_color_space=ColorSpace.REC2020.value)
    assert srgb != rec2020


@pytest.mark.parametrize(
    "space",
    [
        ColorSpace.ADOBE_RGB.value,
        ColorSpace.PROPHOTO.value,
        ColorSpace.ACES.value,
        ColorSpace.XYZ.value,
    ],
)
def test_jxl_rejects_unrepresentable_spaces(proc, space):
    """Unsupported spaces must hard-fail with a clear error, never fall back."""
    buf = np.random.default_rng(3).random((8, 8, 3), dtype=np.float32)
    settings = ExportConfig(export_fmt=ExportFormat.JXL)
    with pytest.raises(ValueError, match="JPEG XL"):
        proc._encode_export(buf, settings, space, working_color_space=space)


def test_other_formats_unaffected_by_jxl_branch(proc):
    """TIFF/PNG still encode through their own branches."""
    buf = np.random.default_rng(4).random((8, 8, 3), dtype=np.float32)
    for fmt, ext in ((ExportFormat.TIFF, "tiff"), (ExportFormat.PNG, "png")):
        settings = ExportConfig(export_fmt=fmt)
        bits, out_ext = proc._encode_export(buf, settings, ColorSpace.SRGB.value)
        assert out_ext == ext
        assert bits


def test_jxl_fields_roundtrip_through_preset():
    """jxl_* fields survive ExportConfig -> preset -> dict -> preset."""
    conf = ExportConfig(export_fmt=ExportFormat.JXL, jxl_lossless=False, jxl_distance=2.5, jxl_effort=9)
    preset = preset_from_export_config(conf)
    assert (preset.jxl_lossless, preset.jxl_distance, preset.jxl_effort) == (False, 2.5, 9)

    restored = ExportPreset.from_dict(preset.to_dict())
    assert (restored.jxl_lossless, restored.jxl_distance, restored.jxl_effort) == (False, 2.5, 9)


def test_export_blocked_pure_helper_mirrors_jxl_table():
    """The UI/controller gate must stay in sync with the encoder's tag table."""
    assert JXL_TAGGABLE_SPACES == set(_JXL_COLOR) | {ColorSpace.SAME_AS_SOURCE.value}
    assert export_blocked(ExportFormat.JXL, ColorSpace.PROPHOTO.value)
    assert not export_blocked(ExportFormat.JXL, ColorSpace.SRGB.value)
    assert not export_blocked(ExportFormat.JXL, ColorSpace.SAME_AS_SOURCE.value)
    assert not export_blocked(ExportFormat.TIFF, ColorSpace.PROPHOTO.value)
