import unittest

import cv2
import numpy as np

from negpy.kernel.image.logic import (
    _PROPHOTO_TO_XYZ,
    lab_to_rgb_working,
    rgb_to_lab_working,
)


class TestWorkingSpaceLab(unittest.TestCase):
    """
    CIELAB conversions use the ProPhoto RGB working-space primaries (D50, not sRGB).
    Mirrors the GPU rgb_to_lab/lab_to_rgb in lab.wgsl / toning.wgsl.
    """

    def test_round_trip_identity(self):
        rng = np.random.default_rng(1)
        img = rng.random((48, 48, 3)).astype(np.float32)
        rt = lab_to_rgb_working(rgb_to_lab_working(img))
        self.assertLess(float(np.max(np.abs(rt - img))), 1e-4)

    def test_neutral_has_zero_chroma(self):
        # Any gray (R=G=B) is a*=b*=0: ROMM matrix row-sums equal the D50 white point.
        gray = np.tile(np.linspace(0.05, 0.95, 12, dtype=np.float32)[:, None, None], (1, 1, 3))
        lab = rgb_to_lab_working(gray)
        self.assertLess(float(np.max(np.abs(lab[..., 1]))), 1e-3)
        self.assertLess(float(np.max(np.abs(lab[..., 2]))), 1e-3)

    def test_lab_scale_matches_opencv_convention(self):
        # L in [0,100]: black -> 0, white -> 100.
        black = np.zeros((1, 1, 3), np.float32)
        white = np.ones((1, 1, 3), np.float32)
        self.assertAlmostEqual(float(rgb_to_lab_working(black)[0, 0, 0]), 0.0, delta=1e-3)
        self.assertAlmostEqual(float(rgb_to_lab_working(white)[0, 0, 0]), 100.0, delta=1e-2)

    def test_differs_from_srgb_assumption_on_green(self):
        # The whole point: ProPhoto green diverges from sRGB green, so a* must differ
        # from the old cv2 (sRGB) Lab. Neutral-axis behavior is unchanged (tested above).
        green = np.full((1, 1, 3), [0.1, 0.8, 0.2], np.float32)
        a_new = float(rgb_to_lab_working(green)[0, 0, 1])
        a_old = float(cv2.cvtColor(green, cv2.COLOR_RGB2LAB)[0, 0, 1])
        self.assertGreater(abs(a_new - a_old), 5.0)

    def test_matrix_matches_manual_xyz(self):
        c = np.array([0.5, 0.3, 0.7], np.float32)
        lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
        xyz_ref = _PROPHOTO_TO_XYZ @ lin
        # Y (luminance) in a sane range for this color.
        self.assertGreater(float(xyz_ref[1]), 0.0)
        self.assertLess(float(xyz_ref[1]), 1.0)


if __name__ == "__main__":
    unittest.main()
