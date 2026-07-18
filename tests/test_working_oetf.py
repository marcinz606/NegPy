"""Working-space output transfer function (OETF) — the encode applied as the final
engine step once the pipeline is scene-linear. Uses the Adobe RGB (1998) TRC
(a pure 563/256 power, no linear segment) so it round-trips with the working ICC profile."""

import numpy as np

from negpy.kernel.image.logic import working_oetf_decode, working_oetf_encode

_GAMMA = 563.0 / 256.0


def test_encode_known_gamma_values():
    x = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    enc = working_oetf_encode(x)
    # Adobe RGB gamma 563/256: 0->0, 1->1, 0.5 -> 0.5^(256/563) ≈ 0.7297.
    np.testing.assert_allclose(enc[0, 0], [0.0, 0.5 ** (1.0 / _GAMMA), 1.0], atol=1e-5)


def test_encode_is_pure_power_near_black():
    # Adobe RGB has no linear toe — the power law holds all the way down.
    x = np.array([[[0.0005, 0.001, 0.0015]]], dtype=np.float32)
    enc = working_oetf_encode(x)
    np.testing.assert_allclose(enc[0, 0], x[0, 0] ** (1.0 / _GAMMA), rtol=1e-5)


def test_roundtrip_identity():
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(1, -1, 1)
    x = ramp * np.ones((1, 1, 3), dtype=np.float32)
    back = working_oetf_decode(working_oetf_encode(x))
    np.testing.assert_allclose(back, x, atol=1e-5)


def test_encode_clamps_to_display_range():
    x = np.array([[[-0.5, 1.5, 0.2]]], dtype=np.float32)
    enc = working_oetf_encode(x)
    assert enc.min() >= 0.0 and enc.max() <= 1.0
    assert enc.dtype == np.float32


def test_encode_composes_with_working_icc():
    """The working OETF must match the working ICC profile's TRC, so encoding
    scene-linear then transforming working->sRGB (ICC) and decoding sRGB recovers
    the original linear value on the neutral axis."""
    from PIL import Image, ImageCms

    from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE, ColorSpaceRegistry

    path = ColorSpaceRegistry.get_icc_path(WORKING_COLOR_SPACE)
    if not path:
        import pytest

        pytest.skip("working ICC profile not available")

    lin = np.linspace(0.05, 0.95, 7, dtype=np.float32)
    gray = np.stack([lin, lin, lin], axis=-1).reshape(1, 7, 3)
    enc8 = np.clip(working_oetf_encode(gray) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    pro = ImageCms.getOpenProfile(path)
    srgb = ImageCms.createProfile("sRGB")
    xf = ImageCms.buildTransform(pro, srgb, "RGB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    out = np.asarray(ImageCms.applyTransform(Image.fromarray(enc8, "RGB"), xf)).astype(np.float32) / 255.0
    recovered = np.where(out <= 0.04045, out / 12.92, ((out + 0.055) / 1.055) ** 2.4)
    # 8-bit quantisation dominates the error; a TRC mismatch would be far larger.
    np.testing.assert_allclose(recovered[0, :, 1], lin, atol=0.01)
