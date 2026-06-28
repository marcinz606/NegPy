"""Working-space output transfer function (OETF) — the encode applied as the final
engine step once the pipeline is scene-linear. Uses the ProPhoto RGB (ROMM) TRC
(gamma 1.8 with a linear toe below 1/512) so it round-trips with the working ICC profile."""

import numpy as np

from negpy.kernel.image.logic import working_oetf_decode, working_oetf_encode


def test_encode_known_gamma_values():
    x = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    enc = working_oetf_encode(x)
    # ProPhoto gamma 1.8: 0->0, 1->1, 0.5 -> 0.5^(1/1.8) ≈ 0.6804.
    np.testing.assert_allclose(enc[0, 0], [0.0, 0.5 ** (1.0 / 1.8), 1.0], atol=1e-5)


def test_encode_linear_toe():
    # Below the 1/512 breakpoint the ROMM encode is linear with slope 16.
    x = np.array([[[0.0005, 0.001, 0.0015]]], dtype=np.float32)
    enc = working_oetf_encode(x)
    np.testing.assert_allclose(enc[0, 0], x[0, 0] * 16.0, atol=1e-6)


def test_toe_continuous_at_breakpoint():
    et = 1.0 / 512.0
    enc = float(working_oetf_encode(np.array([[[et]]], dtype=np.float32))[0, 0, 0])
    # The linear toe and the power segment meet at the breakpoint.
    np.testing.assert_allclose(enc, 16.0 * et, atol=1e-5)
    np.testing.assert_allclose(enc, et ** (1.0 / 1.8), atol=1e-4)


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


def test_encode_composes_with_prophoto_icc():
    """The working OETF must match the ProPhoto RGB ICC profile's TRC, so encoding
    scene-linear then transforming ProPhoto->sRGB (ICC) and decoding sRGB recovers
    the original linear value on the neutral axis."""
    from PIL import Image, ImageCms

    from negpy.domain.models import ColorSpace
    from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry

    path = ColorSpaceRegistry.get_icc_path(ColorSpace.PROPHOTO.value)
    if not path:
        import pytest

        pytest.skip("ProPhoto ICC profile not available")

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
