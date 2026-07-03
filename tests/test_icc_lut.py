import numpy as np
from PIL import Image, ImageCms
from negpy.infrastructure.display.icc_lut import (
    apply_icc_u16_rgb,
    apply_lut_f32,
    apply_lut_u16,
    build_3d_lut,
)


def _identity_profiles() -> tuple:
    srgb_a = ImageCms.createProfile("sRGB")
    srgb_b = ImageCms.createProfile("sRGB")
    return srgb_a, srgb_b


def test_build_3d_lut_shape_and_range() -> None:
    p_src, p_dst = _identity_profiles()
    lut = build_3d_lut(
        p_src,
        p_dst,
        ImageCms.Intent.RELATIVE_COLORIMETRIC,
        ImageCms.Flags.BLACKPOINTCOMPENSATION,
        size=17,
    )
    assert lut.shape == (17, 17, 17, 3)
    assert lut.dtype == np.float32
    assert lut.min() >= 0.0
    assert lut.max() <= 1.0


def test_apply_icc_u16_rgb_shape_and_dtype() -> None:
    p_src, p_dst = _identity_profiles()
    img = np.array(
        [[[0, 32768, 65535], [12345, 54321, 1000]]],
        dtype=np.uint16,
    )
    out = apply_icc_u16_rgb(
        img,
        p_src,
        p_dst,
        ImageCms.Intent.RELATIVE_COLORIMETRIC,
        ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    assert out.shape == img.shape
    assert out.dtype == np.uint16


def test_apply_icc_u16_rgb_identity_preserves_values() -> None:
    # sRGB -> sRGB should round-trip within LUT quantization error
    p_src, p_dst = _identity_profiles()
    img = np.array(
        [
            [[0, 0, 0], [65535, 65535, 65535]],
            [[32768, 16384, 49152], [8000, 40000, 25000]],
        ],
        dtype=np.uint16,
    )
    out = apply_icc_u16_rgb(
        img,
        p_src,
        p_dst,
        ImageCms.Intent.RELATIVE_COLORIMETRIC,
        ImageCms.Flags.BLACKPOINTCOMPENSATION,
        size=33,
    )
    # LUT-based identity transform has small (<~257 = 1/255 * 65535) error
    # from 8-bit PIL sampling. Use generous tolerance.
    diff = np.abs(out.astype(np.int32) - img.astype(np.int32))
    assert diff.max() < 600


def test_apply_icc_u16_rgb_endpoints_clamped() -> None:
    p_src, p_dst = _identity_profiles()
    img = np.array([[[0, 0, 0], [65535, 65535, 65535]]], dtype=np.uint16)
    out = apply_icc_u16_rgb(
        img,
        p_src,
        p_dst,
        ImageCms.Intent.RELATIVE_COLORIMETRIC,
        ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    # Identity should map 0→0 and 65535→65535 exactly (grid points).
    assert np.array_equal(out[0, 0], [0, 0, 0])
    assert np.array_equal(out[0, 1], [65535, 65535, 65535])


def test_apply_lut_u16_with_scale_lut() -> None:
    # Construct a LUT that halves intensity: out = in * 0.5
    n = 33
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    lut = np.stack((r * 0.5, g * 0.5, b * 0.5), axis=-1).astype(np.float32)

    img = np.full((4, 4, 3), 40000, dtype=np.uint16)
    out = apply_lut_u16(img, lut)
    # 40000 / 2 = 20000, interpolation keeps precision
    assert out.dtype == np.uint16
    assert abs(int(out[0, 0, 0]) - 20000) < 3


def _scale_lut(n: int = 9, factor: float = 0.5) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((r * factor, g * factor, b * factor), axis=-1).astype(np.float32)


def test_apply_lut_f32_dtype_and_layout_variants_match() -> None:
    # The no-copy fast path must not change results for f64 or non-contiguous inputs.
    lut = _scale_lut()
    rng = np.random.default_rng(42)
    img = rng.random((8, 8, 3)).astype(np.float32)
    ref = apply_lut_f32(img, lut)

    out_f64 = apply_lut_f32(img.astype(np.float64), lut)
    assert np.array_equal(ref, out_f64)

    wide = np.zeros((8, 16, 3), dtype=np.float32)
    wide[:, ::2, :] = img
    view = wide[:, ::2, :]
    assert not view.flags["C_CONTIGUOUS"]
    assert np.array_equal(ref, apply_lut_f32(view, lut))

    out_lut_f64 = apply_lut_f32(img, lut.astype(np.float64))
    assert np.array_equal(ref, out_lut_f64)


def test_apply_lut_f32_does_not_mutate_input() -> None:
    lut = _scale_lut()
    img = np.random.default_rng(7).random((6, 6, 3)).astype(np.float32)
    img_before = img.copy()
    apply_lut_f32(img, lut)
    assert np.array_equal(img, img_before)


def test_apply_lut_u16_lut_dtype_variants_match() -> None:
    lut = _scale_lut()
    img = np.full((4, 4, 3), 40000, dtype=np.uint16)
    ref = apply_lut_u16(img, lut)
    assert np.array_equal(ref, apply_lut_u16(img, lut.astype(np.float64)))


def test_apply_icc_u16_rgb_matches_pil_8bit_reference() -> None:
    # Verify our 16-bit LUT result matches PIL's native 8-bit transform
    # (within LUT interpolation error) on an image with mid-range values.
    src = ImageCms.createProfile("sRGB")
    dst = ImageCms.createProfile("sRGB")

    img_u8 = np.array(
        [
            [[10, 200, 100], [250, 0, 128]],
            [[128, 128, 128], [64, 64, 192]],
        ],
        dtype=np.uint8,
    )
    pil_in = Image.fromarray(img_u8, mode="RGB")
    pil_out = ImageCms.profileToProfile(
        pil_in,
        src,
        dst,
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        outputMode="RGB",
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    assert pil_out is not None
    ref_u8 = np.array(pil_out)

    img_u16 = (img_u8.astype(np.uint32) * 65535 // 255).astype(np.uint16)
    out_u16 = apply_icc_u16_rgb(
        img_u16,
        src,
        dst,
        ImageCms.Intent.RELATIVE_COLORIMETRIC,
        ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    out_u8 = (out_u16.astype(np.uint32) * 255 // 65535).astype(np.uint8)
    # Identity transform: should match within 1 code value after round-trip.
    assert np.max(np.abs(out_u8.astype(np.int32) - ref_u8.astype(np.int32))) <= 1
