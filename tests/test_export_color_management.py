"""Export ICC fast path: the matrix/TRC kernel must match lcms within 1 LSB."""

import unittest

import numpy as np
from PIL import Image, ImageCms

from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
from negpy.infrastructure.display.icc_lut import apply_matrix_trc_u8
from negpy.services.rendering.image_processor import ImageProcessor


def _icc_bytes(space: str) -> bytes:
    path = ColorSpaceRegistry.get_icc_path(space)
    assert path is not None
    with open(path, "rb") as f:
        return f.read()


class TestMatrixTrcFastPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.arr = (rng.random((256, 384, 3)) * 255).astype(np.uint8)
        cls.working = "Adobe RGB"

    def _lcms_reference(self, target: str) -> np.ndarray:
        p_src = ImageProcessor._resolve_src_profile(self.working, None)
        p_dst = ImageProcessor._resolve_dst_profile(target, None)
        ref = ImageCms.profileToProfile(
            Image.fromarray(self.arr),
            p_src,
            p_dst,
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        return np.asarray(ref).astype(np.int16)

    def test_matches_lcms_within_one_lsb(self):
        src = _icc_bytes(self.working)
        for target in ("sRGB", "Rec 2020", "ProPhoto RGB"):
            with self.subTest(target=target):
                out = apply_matrix_trc_u8(self.arr, src, _icc_bytes(target))
                self.assertIsNotNone(out)
                delta = np.abs(out.astype(np.int16) - self._lcms_reference(target))
                self.assertLessEqual(delta.max(), 1)

    def test_same_space_is_identity(self):
        src = _icc_bytes(self.working)
        out = apply_matrix_trc_u8(self.arr, src, src)
        np.testing.assert_array_equal(out, self.arr)

    def test_non_matrix_profile_falls_through(self):
        """A profile without RGB colorant tags (greyscale) must return None so the
        caller takes the exact lcms path."""
        grey = _icc_bytes("Greyscale")
        self.assertIsNone(apply_matrix_trc_u8(self.arr, _icc_bytes(self.working), grey))


if __name__ == "__main__":
    unittest.main()
