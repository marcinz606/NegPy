import io

import numpy as np
import pytest
from PIL import Image

from negpy.domain.models import (
    ColorSpace,
    ExportConfig,
    ExportFormat,
    ExportPreset,
    preset_from_export_config,
)
from negpy.kernel.image.logic import float_to_uint8
from negpy.services.rendering.image_processor import ImageProcessor


@pytest.fixture(scope="module")
def proc():
    return ImageProcessor()


def _is_webp(bits: bytes) -> bool:
    return bits[:4] == b"RIFF" and bits[8:12] == b"WEBP"


def test_webp_lossless_roundtrip_is_exact(proc):
    """Lossless WebP decodes back to the exact 8-bit samples. Working space ==
    target sRGB makes colour management a no-op, isolating the codec."""
    buf = np.random.default_rng(0).random((16, 24, 3), dtype=np.float32)
    settings = ExportConfig(export_fmt=ExportFormat.WEBP, webp_lossless=True)

    bits, ext = proc._encode_export(buf, settings, ColorSpace.SRGB.value, working_color_space=ColorSpace.SRGB.value)
    assert ext == "webp"
    assert _is_webp(bits)

    decoded = np.asarray(Image.open(io.BytesIO(bits)).convert("RGB"))
    expected = float_to_uint8(buf)
    assert decoded.shape == expected.shape
    np.testing.assert_array_equal(decoded, expected)


def test_webp_lossy_differs_and_roundtrips(proc):
    """Lossy WebP keeps dimensions but is not pixel-exact (proves the flag flows)."""
    buf = np.random.default_rng(1).random((16, 24, 3), dtype=np.float32)
    lossy = ExportConfig(export_fmt=ExportFormat.WEBP, webp_lossless=False, webp_quality=80)
    lossless = ExportConfig(export_fmt=ExportFormat.WEBP, webp_lossless=True)

    lossy_bits, _ = proc._encode_export(buf, lossy, ColorSpace.SRGB.value, working_color_space=ColorSpace.SRGB.value)
    lossless_bits, _ = proc._encode_export(buf, lossless, ColorSpace.SRGB.value, working_color_space=ColorSpace.SRGB.value)
    assert lossy_bits != lossless_bits

    img = Image.open(io.BytesIO(lossy_bits))
    assert img.size == (24, 16)  # PIL size is (w, h)


def test_webp_greyscale_roundtrips(proc):
    """Greyscale WebP exports without error and keeps dimensions."""
    buf = np.random.default_rng(2).random((16, 24, 3), dtype=np.float32)
    settings = ExportConfig(export_fmt=ExportFormat.WEBP, webp_lossless=True)
    bits, ext = proc._encode_export(buf, settings, ColorSpace.GREYSCALE.value, working_color_space=ColorSpace.GREYSCALE.value)
    assert ext == "webp"
    assert Image.open(io.BytesIO(bits)).size == (24, 16)


def test_webp_rejects_oversized(proc):
    """Dimensions above the WebP 16383 px limit fail with a clear error."""
    buf = np.zeros((1, 16384, 3), dtype=np.float32)
    settings = ExportConfig(export_fmt=ExportFormat.WEBP)
    with pytest.raises(ValueError, match="16383"):
        proc._encode_export(buf, settings, ColorSpace.SRGB.value, working_color_space=ColorSpace.SRGB.value)


def test_webp_fields_roundtrip_through_preset():
    """webp_* fields survive ExportConfig -> preset -> dict -> preset."""
    conf = ExportConfig(export_fmt=ExportFormat.WEBP, webp_lossless=True, webp_quality=75, webp_method=6)
    preset = preset_from_export_config(conf)
    assert (preset.webp_lossless, preset.webp_quality, preset.webp_method) == (True, 75, 6)

    restored = ExportPreset.from_dict(preset.to_dict())
    assert (restored.webp_lossless, restored.webp_quality, restored.webp_method) == (True, 75, 6)
