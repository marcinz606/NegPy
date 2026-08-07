"""Tests for the matrix/TRC ICC profile bypass.

Validates:
- Profile-type detection (matrix/TRC vs LUT-based)
- Primaries matrix extraction (PCS-D50 tag values -> D65-referenced result)
- TRC independence (the actual bug being fixed)
"""

import struct
from typing import Optional

import numpy as np
import pytest

from negpy.infrastructure.display.icc_profile import (
    extract_primaries_matrix,
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


def _chad_tag(m: np.ndarray) -> bytes:
    """An s15Fixed16ArrayType chromaticAdaptationTag (ICC spec §10.8)."""
    return b"sf32" + b"\x00" * 4 + b"".join(_s15fixed16(v) for v in m.flatten())


def _build_matrix_trc_icc(
    r_xyz: tuple[float, float, float],
    g_xyz: tuple[float, float, float],
    b_xyz: tuple[float, float, float],
    wtpt: tuple[float, float, float] = (0.9642, 1.0000, 0.8249),
    gamma: float = 2.2,
    add_a2b0: bool = False,
    chad: Optional[np.ndarray] = None,
) -> bytes:
    """Build a minimal ICC profile with matrix/TRC structure. ``wtpt`` defaults to the
    D50 PCS value real profiles carry; pass ``chad`` (a v4 profile always has one) to
    exercise the inv(chad) adaptation path instead of the generic Bradford fallback."""
    tag_data: list[tuple[bytes, bytes]] = []
    tag_data.append((b"rXYZ", _xyz_tag(*r_xyz)))
    tag_data.append((b"gXYZ", _xyz_tag(*g_xyz)))
    tag_data.append((b"bXYZ", _xyz_tag(*b_xyz)))
    tag_data.append((b"wtpt", _xyz_tag(*wtpt)))
    trc = _trc_tag_gamma(gamma)
    tag_data.append((b"rTRC", trc))
    tag_data.append((b"gTRC", trc))
    tag_data.append((b"bTRC", trc))
    if chad is not None:
        tag_data.append((b"chad", _chad_tag(chad)))
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


def _build_gray_icc() -> bytes:
    """Build a minimal GRAY-space ICC profile: desc/cprt/wtpt/kTRC/bkpt, no colorant tags
    at all — matches the real structure of a bundled/downloadable grayscale profile."""
    tag_data: list[tuple[bytes, bytes]] = [
        (b"wtpt", _xyz_tag(0.9642, 1.0, 0.8249)),
        (b"bkpt", _xyz_tag(0.0, 0.0, 0.0)),
        (b"kTRC", _trc_tag_gamma(2.2)),
    ]

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
    header[16:20] = b"GRAY"
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


# PCS-D50-relative sRGB primaries, as an ICC v4 file actually stores them
# (extracted from a real sRGB-primaries profile) — real profiles never store
# native-D65 values directly, since the PCS is always D50-relative.
_SRGB_R = (0.43603516, 0.22248840, 0.01391602)
_SRGB_G = (0.38511658, 0.71690369, 0.09706116)
_SRGB_B = (0.14305115, 0.06060791, 0.71392822)
_D50_WP = (0.9642, 1.0000, 0.8249)
_D65_WP = (0.9505, 1.0000, 1.0889)

# The chad tag from the same real profile that _SRGB_R/G/B were extracted from — the
# exact D65->D50 adaptation lcms2 applied when writing the file, as opposed to a
# generic Bradford assumption.
_SRGB_CHAD = np.array(
    [
        [1.04788208, 0.0229187, -0.05021667],
        [0.02958679, 0.99047852, -0.01707458],
        [-0.00924683, 0.01507568, 0.75167847],
    ]
)

# Known native-D65-referenced sRGB primaries (Lindbloom), what extract_primaries_matrix
# should recover from the PCS-D50 fixture above.
_SRGB_R_D65 = (0.4124564, 0.2126729, 0.0193339)
_SRGB_G_D65 = (0.3575761, 0.7151522, 0.1191920)
_SRGB_B_D65 = (0.1804375, 0.0721750, 0.9503041)

# PCS-D50 colorants + chad extracted from a real ACES (AP0) profile, D60-native — inverting
# chad alone lands on D60, not D65, unless the native white is itself re-adapted to D65.
_ACES_R = (0.9908905, 0.3618927, -0.00271606)
_ACES_G = (0.01223755, 0.72251892, 0.008255)
_ACES_B = (-0.03892517, -0.08441162, 0.81936646)
_ACES_CHAD = np.array(
    [
        [1.03416443, 0.01681519, -0.03747559],
        [0.0216217, 0.99223328, -0.01272583],
        [-0.00694275, 0.01132202, 0.8130188],
    ]
)

# PCS-D50 colorants + chad extracted from a real ProPhoto-primaries profile, D50-native —
# chad is near-identity here, so inv(chad) alone is nearly a no-op and stays at D50.
_PROPHOTO_R = (0.79771423, 0.28805542, 0.0)
_PROPHOTO_G = (0.13516235, 0.71186829, 1.526e-05)
_PROPHOTO_B = (0.03132629, 7.629e-05, 0.82489014)
_PROPHOTO_CHAD = np.array(
    [
        [9.99954220e-01, -3.05200000e-05, -3.05200000e-05],
        [-4.57800000e-05, 1.00004578e00, -1.52600000e-05],
        [-1.52600000e-05, 1.52600000e-05, 9.99740600e-01],
    ]
)

_D65_XYZ = np.array([0.95047, 1.0, 1.08883])


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

    def test_gray_profile_not_detected_as_matrix(self):
        """A GRAY-space profile (class=mntr, space=GRAY: desc/cprt/wtpt/kTRC/bkpt, no
        rXYZ/gXYZ/bXYZ) has no colorant tags at all — must not be mistaken for matrix/TRC,
        and extraction must return None rather than raise, so the bypass safely no-ops and
        the profile falls through to full CMS."""
        icc = _build_gray_icc()
        assert not is_matrix_trc_profile(icc)
        assert extract_primaries_matrix(icc) is None

        from negpy.services.rendering.image_processor import ImageProcessor

        import tempfile
        import os as _os

        with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
            f.write(icc)
            path = f.name
        try:
            img = np.random.RandomState(2).rand(8, 8, 3).astype(np.float32)
            out, bypassed = ImageProcessor._try_matrix_bypass(img, path)
            assert not bypassed
            np.testing.assert_array_equal(out, img)
        finally:
            _os.unlink(path)


class TestPrimariesExtraction:
    def test_extract_srgb_primaries_adapts_via_chad(self):
        """With a chad tag present (the v4 case), extraction inverts it exactly rather
        than assuming generic Bradford — recovers the D65-native sRGB matrix tightly."""
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, chad=_SRGB_CHAD)
        m = extract_primaries_matrix(icc)
        assert m is not None
        expected = np.array([_SRGB_R_D65, _SRGB_G_D65, _SRGB_B_D65], dtype=np.float64).T
        np.testing.assert_allclose(m, expected, atol=5e-4)

    def test_extract_srgb_primaries_falls_back_to_bradford_without_chad(self):
        """Without a chad tag (typical of v2 profiles), extraction still recovers
        D65-native primaries via generic Bradford D50->D65, just less precisely."""
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B)
        m = extract_primaries_matrix(icc)
        assert m is not None
        expected = np.array([_SRGB_R_D65, _SRGB_G_D65, _SRGB_B_D65], dtype=np.float64).T
        np.testing.assert_allclose(m, expected, atol=2e-3)

    def test_returns_none_without_tags(self):
        header = bytearray(128)
        struct.pack_into(">I", header, 0, 132)
        header[36:40] = b"acsp"
        data = bytes(header) + struct.pack(">I", 0)
        assert extract_primaries_matrix(data) is None

    @pytest.mark.parametrize(
        "r,g,b,chad,label",
        [
            (_SRGB_R, _SRGB_G, _SRGB_B, _SRGB_CHAD, "sRGB (D65-native)"),
            (_ACES_R, _ACES_G, _ACES_B, _ACES_CHAD, "ACES AP0 (D60-native)"),
            (_PROPHOTO_R, _PROPHOTO_G, _PROPHOTO_B, _PROPHOTO_CHAD, "ProPhoto (D50-native)"),
        ],
    )
    def test_extracted_primaries_are_d65_referenced(self, r, g, b, chad, label):
        """The whole point of extract_primaries_matrix is to hand back primaries expressed
        against D65 (to combine with this codebase's D65-native _XYZ_TO_WORKING). That must
        hold regardless of the profile's own native white point: R+G+B XYZ must sum to D65.
        inv(chad) alone only satisfies this for D65-native profiles (sRGB here) — for a
        D60-native (ACES) or D50-native (ProPhoto) profile it lands on the profile's own
        native white instead, which is the bug this test exists to catch."""
        icc = _build_matrix_trc_icc(r, g, b, chad=chad)
        m = extract_primaries_matrix(icc)
        assert m is not None
        white = m.sum(axis=1)
        np.testing.assert_allclose(white, _D65_XYZ, atol=1e-3, err_msg=f"{label}: white did not land on D65")

    def test_extracted_primaries_are_d65_referenced_without_chad(self):
        """Bradford-fallback path (no chad tag) must also land on D65 regardless of the
        profile's actual native white — it has no way to know it, so it always assumes D50."""
        icc = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B)
        m = extract_primaries_matrix(icc)
        assert m is not None
        np.testing.assert_allclose(m.sum(axis=1), _D65_XYZ, atol=1e-3)


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

    def test_bypasses_regardless_of_whitepoint_tag(self):
        """Real profiles carry a PCS-D50 wtpt; a D65-literal wtpt is only ever seen in
        hand-built fixtures. Either way, whitepoint is not part of the bypass gate."""
        from negpy.services.rendering.image_processor import ImageProcessor

        import tempfile
        import os

        for wtpt in (_D50_WP, _D65_WP):
            icc_data = _build_matrix_trc_icc(_SRGB_R, _SRGB_G, _SRGB_B, wtpt=wtpt)
            with tempfile.NamedTemporaryFile(suffix=".icc", delete=False) as f:
                f.write(icc_data)
                path = f.name
            try:
                img = np.random.RandomState(1).rand(16, 16, 3).astype(np.float32)
                out, bypassed = ImageProcessor._try_matrix_bypass(img, path)
                assert bypassed
            finally:
                os.unlink(path)

    def test_no_path_does_not_bypass(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        img = np.random.RandomState(1).rand(16, 16, 3).astype(np.float32)
        out, bypassed = ImageProcessor._try_matrix_bypass(img, None)
        assert not bypassed


class TestRealBundledProfiles:
    """Regression tests against the app's own shipped profiles, not synthetic fixtures."""

    @staticmethod
    def _icc_path(name: str) -> str:
        import os

        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "icc", name)

    def test_working_profile_is_identity(self):
        """Selecting the app's own working-space profile as Input ICC must be a near-no-op:
        its primaries are the working space's primaries, so the bypass matrix is I + eps
        (eps from chad-inversion + s15Fixed16 quantization, not exactly I). Tolerance is a
        few 16-bit LSB — measured max diff across seeds is ~2e-4 (~14 LSB)."""
        from negpy.services.rendering.image_processor import ImageProcessor

        path = self._icc_path("AdobeCompat-v4.icc")
        img = np.random.RandomState(3).rand(32, 32, 3).astype(np.float32) * 0.8 + 0.1
        out, bypassed = ImageProcessor._try_matrix_bypass(img, path)
        assert bypassed
        np.testing.assert_allclose(out, img, atol=5e-4)

    def test_rgbscan_not_bypassable(self):
        """RGBScan.icc (spac/Lab PCS, A2B0-only) is the shipped Narrowband Scan default.
        It must never take the matrix-only shortcut — its A2B0 curves are authored
        against the full-CMS decode and must keep running through it unchanged."""
        from negpy.services.rendering.image_processor import ImageProcessor

        path = self._icc_path("RGBScan.icc")
        assert not is_matrix_trc_profile(open(path, "rb").read())
        img = np.random.RandomState(4).rand(16, 16, 3).astype(np.float32)
        out, bypassed = ImageProcessor._try_matrix_bypass(img, path)
        assert not bypassed
        np.testing.assert_array_equal(out, img)

    def test_boundary_encoding_contract(self):
        """The buffer arriving at the boundary is pure-power ~2.2 encoded (working_oetf's
        563/256, not exactly 2.2, but within a fraction of an LSB). RGBScan.icc's CLUT
        input domain is sRGB piecewise. The A2B0 input curve is the transcode between the
        two — decode pure-power, re-encode sRGB — which is what this pins, so a change to
        the finish-step OETF fails this loudly instead of silently breaking the shipped
        narrowband-scan default."""
        import struct

        from negpy.kernel.image.logic import working_oetf_decode

        path = self._icc_path("RGBScan.icc")
        data = open(path, "rb").read()
        tag_count = struct.unpack_from(">I", data, 128)[0]
        tags = {}
        for i in range(tag_count):
            base = 132 + i * 12
            sig = data[base : base + 4]
            off, sz = struct.unpack_from(">II", data, base + 4)
            tags[sig] = (off, sz)
        off, _ = tags[b"A2B0"]
        assert data[off : off + 4] == b"mft2"
        n_in_entries = struct.unpack_from(">H", data, off + 48)[0]
        r_curve = np.array(struct.unpack_from(f">{n_in_entries}H", data, off + 52), dtype=np.float64) / 65535.0

        x = np.linspace(0.0, 1.0, n_in_entries)
        linear = working_oetf_decode(x.astype(np.float32)).astype(np.float64)
        srgb_encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)

        # The profile's curve was built against a plain gamma=2.2 assumption, not the
        # pipeline's exact 563/256 (2.19921875) — a known, tiny, sub-8-bit-LSB gap.
        assert np.max(np.abs(r_curve - srgb_encoded)) < 2e-4
