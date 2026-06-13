import unittest

import numpy as np

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.stats import negative_statistics


def _by_name(rows, name):
    return next(r for r in rows if r.name == name)


class TestNegativeStatistics(unittest.TestCase):
    def _rows(self, dr=1.3, anchor=0.46, slope=4.0, lo=0.0, hi=0.0):
        return negative_statistics(dr, anchor, slope, lo, hi)

    def test_density_bands(self):
        self.assertEqual(_by_name(self._rows(dr=0.6), "Density range").tag, "Low contrast")
        self.assertEqual(_by_name(self._rows(dr=1.3), "Density range").tag, "Normal")
        self.assertEqual(_by_name(self._rows(dr=2.2), "Density range").tag, "High contrast")
        self.assertEqual(_by_name(self._rows(dr=1.82), "Density range").value, "1.82")

    def test_exposure_key(self):
        a = EXPOSURE_CONSTANTS["assumed_anchor"]
        self.assertEqual(_by_name(self._rows(anchor=a), "Exposure").tag, "Balanced")
        self.assertEqual(_by_name(self._rows(anchor=a - 0.1), "Exposure").tag, "Low-key")
        self.assertEqual(_by_name(self._rows(anchor=a + 0.1), "Exposure").tag, "High-key")

    def test_exposure_ev_number(self):
        a = EXPOSURE_CONSTANTS["assumed_anchor"]
        # +0.1 normalized at dr 1.3 → +0.1*1.3/0.30103 ≈ +0.43 EV, brighter = +.
        row = _by_name(self._rows(anchor=a + 0.1, dr=1.3), "Exposure")
        self.assertIn("EV", row.value)
        self.assertTrue(row.value.startswith("+"))
        # No density range → label only, no EV number.
        self.assertEqual(_by_name(self._rows(anchor=a + 0.1, dr=None), "Exposure").value, "")

    def test_contrast_bands(self):
        self.assertEqual(_by_name(self._rows(slope=2.5), "Contrast").tag, "Soft")
        self.assertEqual(_by_name(self._rows(slope=4.5), "Contrast").tag, "Normal")
        self.assertEqual(_by_name(self._rows(slope=8.0), "Contrast").tag, "Hard")

    def test_contrast_iso_r_number(self):
        # ISO R: harder (higher slope) → lower R; softer → higher R.
        hard = _by_name(self._rows(slope=8.0), "Contrast").value
        soft = _by_name(self._rows(slope=2.5), "Contrast").value
        self.assertTrue(hard.startswith("R"))
        self.assertTrue(soft.startswith("R"))
        self.assertLess(int(hard[1:]), int(soft[1:]))

    def test_clipping_warn(self):
        clean = _by_name(self._rows(lo=0.001, hi=0.002), "Clipping")
        self.assertFalse(clean.warn)
        self.assertIn("%", clean.value)
        blown = _by_name(self._rows(lo=0.0, hi=0.05), "Clipping")
        self.assertTrue(blown.warn)

    def test_missing_inputs_blank(self):
        rows = negative_statistics(None, None, None, None, None)
        self.assertTrue(all(r.value == "—" for r in rows))


def test_clip_fractions_from_bin_array(qapp):
    from negpy.desktop.view.widgets.charts import HistogramWidget

    w = HistogramWidget()
    # (4, 256) bins (R, G, B, L): 10% of R in the black bin, 20% of G in white.
    buf = np.zeros((4, 256), dtype=np.float32)
    buf[0, 0] = 10.0
    buf[0, 128] = 90.0  # R: 10% shadows clipped
    buf[1, 255] = 20.0
    buf[1, 128] = 80.0  # G: 20% highlights clipped
    buf[2, 128] = 100.0
    buf[3, 128] = 100.0
    w.update_data(buf)
    lo, hi = w.clip_fractions()
    assert abs(lo - 0.10) < 1e-4
    assert abs(hi - 0.20) < 1e-4


if __name__ == "__main__":
    unittest.main()
