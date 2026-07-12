import unittest

import numpy as np

from negpy.features.exposure.densitometer import (
    DensitometerReading,
    compute_reading,
    format_reading,
    map_display_to_norm,
    zone_roman,
)
from negpy.features.exposure.normalization import LogNegativeBounds
from negpy.kernel.image.logic import working_oetf_encode


class TestMapDisplayToNorm(unittest.TestCase):
    def test_identity_no_border_no_roi(self):
        self.assertEqual(map_display_to_norm(0.5, 0.5, 100, 80, None, None, False, 100, 80), (50, 40))

    def test_border_inset(self):
        # 200x160 display, image content 100x80 at offset (50, 40).
        rect = (50, 40, 100, 80)
        self.assertEqual(map_display_to_norm(0.5, 0.5, 200, 160, rect, None, False, 100, 80), (50, 40))
        # Cursor on the border → None.
        self.assertIsNone(map_display_to_norm(0.1, 0.1, 200, 160, rect, None, False, 100, 80))

    def test_roi_offset(self):
        # Crop (y1=10, y2=50, x1=20, x2=60) of a 100x80 normalized frame.
        roi = (10, 50, 20, 60)
        self.assertEqual(map_display_to_norm(0.5, 0.5, 40, 40, None, roi, False, 100, 80), (40, 30))

    def test_border_and_roi_combined(self):
        rect = (10, 10, 40, 40)
        roi = (10, 50, 20, 60)
        # Center of content → center of roi.
        self.assertEqual(map_display_to_norm(0.5, 0.5, 60, 60, rect, roi, False, 100, 80), (40, 30))

    def test_crop_full_bypasses_roi(self):
        roi = (10, 50, 20, 60)
        self.assertEqual(map_display_to_norm(0.5, 0.5, 100, 80, None, roi, True, 100, 80), (50, 40))

    def test_clamps_to_frame(self):
        self.assertEqual(map_display_to_norm(1.0, 1.0, 100, 80, None, None, False, 100, 80), (99, 79))


class TestComputeReading(unittest.TestCase):
    _BOUNDS = LogNegativeBounds(floors=(0.0, 0.0, 0.0), ceils=(1.2, 1.0, 0.8))

    def test_delta_density_scales_with_bounds(self):
        r = compute_reading((0.5, 0.5, 0.5), self._BOUNDS, (0.5, 0.5, 0.5))
        self.assertAlmostEqual(r.dd_rgb[0], 0.6, places=6)
        self.assertAlmostEqual(r.dd_rgb[1], 0.5, places=6)
        self.assertAlmostEqual(r.dd_rgb[2], 0.4, places=6)
        self.assertAlmostEqual(r.val_luma, 0.5, places=6)

    def test_mid_gray_reads_zone_five(self):
        enc = float(working_oetf_encode(np.asarray([0.18], dtype=np.float32))[0])
        r = compute_reading((0.5, 0.5, 0.5), self._BOUNDS, (enc, enc, enc))
        self.assertAlmostEqual(r.zone, 5.0, places=2)
        self.assertAlmostEqual(r.print_density, -np.log10(0.18), places=2)

    def test_black_clamps(self):
        r = compute_reading((1.0, 1.0, 1.0), self._BOUNDS, (0.0, 0.0, 0.0))
        self.assertEqual(r.zone, 0.0)
        self.assertEqual(r.print_density, 4.0)


class TestFormatting(unittest.TestCase):
    def test_zone_roman_thirds(self):
        self.assertEqual(zone_roman(5.0), "V")
        self.assertEqual(zone_roman(4.33), "IV⅓")
        self.assertEqual(zone_roman(7.66), "VII⅔")
        self.assertEqual(zone_roman(10.0), "X")
        self.assertEqual(zone_roman(-1.0), "0")

    def test_format_reading(self):
        r = DensitometerReading((0.62, 0.85, 1.04), 0.7, 1.32, 4.33)
        self.assertEqual(format_reading(r), "ΔD 0.62·0.85·1.04 · D 1.32 · IV⅓")


if __name__ == "__main__":
    unittest.main()
