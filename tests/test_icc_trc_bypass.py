"""Tests for the matrix/TRC ICC profile bypass.

Validates:
- Profile-type detection (matrix/TRC vs LUT-based)
- Primaries matrix extraction
- White-point check
- TRC independence (the actual bug being fixed)
- Non-D65 fallback
"""

import struct

import numpy as np

from negpy.infrastructure.display.icc_profile import (
    extract_primaries_matrix,
    extract_whitepoint,
    is_d65_whitepoint,
    is_matrix_trc_profile,
)
from negpy.kernel.image.logic import _WORKING_TO_XYZ, apply_primaries_transform


def _s15fixed16(val: float) -> bytes:
    return struct.pack(">i", int(round(val * 65536.0)))


def _xyz_tag(x: float, y: float, z: float) -> bytes:
    return b"XYZ " + b"\x00" * 4 + _s15fixed16(x) + _s15fixed16(y) + _s15fixed16(z)


def _trc_tag_gamma(gamma: float) -> bytes:
    """A curveType TRC with a single gamma value (count=1)."""
    return b"curv" + b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(gamma * 256.0))) + b"\x00\x00"


def _build_matrix_trc_icc(
    r_xyz: tuple[float, float, float],
    g_xyz: tuple[float, float, float],
    b_xyz: tuple[float, float, float],
    wtpt: tuple[float, float, float] = (0.9505, 1.0000, 1.0889),
    gamma: float = 2.2,
    add_a2b0: bool = False,
) -> bytes:
    """Build a minimal ICC v2 profile with matrix/TRC structure."""
    tag_data: list[tuple[bytes, bytes]] = []
    tag_data.append((b"rXYZ", _xyz_tag(*r_xyz)))
    tag_data.append((b"gXYZ", _xyz_tag(*g_xyz)))
    tag_data.append((b"bXYZ", _xyz_tag(*b_xyz)))
    tag_data.append((b"wtpt", _xyz_tag(*wtpt)))
    trc = _trc_tag_gamma(gamma)
    tag_data.append((b"rTRC", trc))
    tag_data.append((b"gTRC", trc))
    tag_data.append((b"bTRC", trc))
    if add_a2b0:
        tag_data.append((b"A2B0", b"mft2" + b"\x00" * 40))

    tag_count = len(tag_data)
    tag_table_size = tag_count * 12
    header_size = 128 + 4 + tag_table_size
    offset = header_size
    offsets: list[tuple[int, int]] = []
    for _, payload in tag_data:
        padded = len(payload)
        if padded % 4:
            padded += 4 - padded % 4
        offsets.append((offset, len(payload)))
        offset += padded
    total_size = offset

    header = bytearray(128)
    struct.pack_into(">I", header, 0, total_size)
    header[36:40] = b"acsp"
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[40:44] = b"APPL"

    tag_table = struct.pack(">I", tag_count)
    for i, (sig, _) in enumerate(tag_data):
        tag_table += sig + struct.pack(">II", offsets[i][0], offsets[i][1])

    body = b""
    for _, payload in tag_data:
        padded = len(payload)
        if padded % 4:
            payload += b"\x00" * (4 - padded % 4)
        body += payload

    return bytes(header) + tag_table + body


# sRGB primaries (D65)
_SRGB_R = (0.4361, 0.2225, 0.0139)
_SRGB_G = (0.3851, 0.7169, 0.0971)
_SRGB_B = (0.1431, 0.0606, 0.7141)
_D65_WP = (0.9505, 1.0000, 1.0889)
_D50_WP = (0.9642, 1.0000, 0.8249)


class TestProfileDetection:
    def test_matrix_trc_profile_detected(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B)
        assert is_matrix_trc_profile(icc)

    def test_lut_profile_not_detected_as_matrix(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, add_a2b0=True)
        assert not is_matrix_trc_profile(icc)

    def test_real_bundled_profiles(self):
        import os

        icc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icc")
        rgbscan = os.path.join(icc_dir, "RGBScan.icc")
        if os.path.exists(rgbscan):
            with open(rgbscan, "rb") as f:
                data = f.read()
            assert not is_matrix_trc_profile(data), "RGBScan.icc is LUT-based, must not be detected as matrix/TRC"


class TestPrimariesExtraction:
    def test_extract_srgb_primaries(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B)
        m = extract_primaries_matrix(icc)
        assert m is not None
        expected = np.array([_SRGB_R, _SRGB_G, _SRGB_B], dtype=np.float64).T
        np.testing.assert_allclose(m, expected, atol=1e-4)

    def test_returns_none_without_tags(self):
        header = bytearray(128)
        struct.pack_into(">I", header, 0, 132)
        header[36:40] = b"acsp"
        data = bytes(header) + struct.pack(">I", 0)
        assert extract_primaries_matrix(data) is None


class TestWhitepoint:
    def test_d65_detected(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, wtpt=_D65_WP)
        assert is_d65_whitepoint(icc)

    def test_d50_not_d65(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, wtpt=_D50_WP)
        assert not is_d65_whitepoint(icc)

    def test_extract_whitepoint_values(self):
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, wtpt=_D65_WP)
        wp = extract_whitepoint(icc)
        assert wp is not None
        np.testing.assert_allclose(wp, [0.9505, 1.0, 1.0889], atol=1e-3)


class TestTrcIndependence:
    """The core test: identical primaries with different TRCs must produce
    identical output through the matrix-only path."""

    def test_different_trcs_same_result(self):
        icc_g18 = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, gamma=1.8)
        icc_g24 = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, gamma=2.4)
        assert is_matrix_trc_profile(icc_g18)
        assert is_matrix_trc_profile(icc_g24)

        m1 = extract_primaries_matrix(icc_g18)
        m2 = extract_primaries_matrix(icc_g24)
        assert m1 is not None and m2 is not None
        np.testing.assert_array_equal(m1, m2)

        img = np.random.RandomState(42).rand(64, 64, 3).astype(np.float32)
        out1 = apply_primaries_transform(img, m1)
        out2 = apply_primaries_transform(img, m2)
        np.testing.assert_array_equal(out1, out2)


class TestApplyPrimariesTransform:
    def test_identity_for_working_space(self):
        """When the input primaries match the working space, the transform is identity."""
        img = np.random.RandomState(7).rand(32, 32, 3).astype(np.float32) * 0.8 + 0.1
        out = apply_primaries_transform(img, _WORKING_TO_XYZ.astype(np.float64))
        np.testing.assert_allclose(out, img, atol=1e-4)

    def test_output_clipped_to_01(self):
        img = np.ones((2, 2, 3), dtype=np.float32) * 2.0
        out = apply_primaries_transform(img, _WORKING_TO_XYZ.astype(np.float64))
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_shape_preserved(self):
        img = np.random.RandomState(0).rand(100, 50, 3).astype(np.float32)
        out = apply_primaries_transform(img, _WORKING_TO_XYZ.astype(np.float64))
        assert out.shape == img.shape
        assert out.dtype == np.float32


class TestImageProcessorBypass:
    def test_matrix_profile_bypasses(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        icc_data = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B)
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
            f.write(icc_data)
            path = f.name
        try:
            img = np.random.RandomState(1).rand(16, 16, 3).astype(np.float32)
            out, bypassed = ImageProcessor._try_matrix_bypass(img, path)
            assert bypassed
            assert out.shape == img.shape
        finally:
            os.unlink(path)

    def test_lut_profile_does_not_bypass(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        icc_data = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, add_a2b0=True)
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
            f.write(icc_data)
            path = f.name
        try:
            img = np.random.RandomState(1).rand(16, 16, 3).astype(np.float32)
            out, bypassed = ImageProcessor._try_matrix_bypass(img, path)
            assert not bypassed
            np.testing.assert_array_equal(out, img)
        finally:
            os.unlink(path)

    def test_non_d65_profile_does_not_bypass(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        icc_data = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, wtpt=_D50_WP)
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
            f.write(icc_data)
            path = f.name
        try:
            img = np.random.RandomState(1).rand(16, 16, 3).astype(np.float32)
            out, bypassed = ImageProcessor._try_matrix_bypass(img, path)
            assert not bypassed
        finally:
            os.unlink(path)

    def test_no_path_does_not_bypass(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        img = np.random.RandomState(1).rand(16, 16, 3).astype(np.float32)
        out, bypassed = ImageProcessor._try_matrix_bypass(img, None)
        assert not bypassed
