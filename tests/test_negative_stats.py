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

    def test_print_row_stops_and_cc(self):
        # density 1.5 @ ΔD 1.3 → (0.5·0.2·1.3)/0.30103 ≈ +0.43 stop (darker print).
        rows = negative_statistics(1.3, 0.46, 4.0, 0.0, 0.0, density=1.5, wb_cmy=(0.0, 0.6, -0.3))
        row = _by_name(rows, "Print")
        self.assertEqual(row.value, "+0.43 stop · 12M 6B")

    def test_print_row_neutral_shows_zero_stops(self):
        row = _by_name(negative_statistics(1.3, 0.46, 4.0, 0.0, 0.0, density=1.0, wb_cmy=(0.0, 0.0, 0.0)), "Print")
        self.assertEqual(row.value, "+0.00 stop")

    def test_print_row_absent_without_config(self):
        row = _by_name(negative_statistics(1.3, 0.46, 4.0, 0.0, 0.0), "Print")
        self.assertEqual(row.value, "—")

    def test_scan_clip_row_warns(self):
        clean = _by_name(negative_statistics(1.3, 0.46, 4.0, 0.0, 0.0, scan_clip=(0.001, 0.0, 0.0)), "Scan clip")
        self.assertFalse(clean.warn)
        blown = _by_name(negative_statistics(1.3, 0.46, 4.0, 0.0, 0.0, scan_clip=(0.031, 0.002, 0.0)), "Scan clip")
        self.assertTrue(blown.warn)
        self.assertEqual(blown.value, "R 3.1% · G 0.2% · B 0.0%")

    def test_scan_clip_fraction_measurement(self):
        from negpy.features.exposure.normalization import measure_clip_fractions

        img = np.full((64, 64, 3), 0.5, dtype=np.float32)
        img[:16, :, 0] = 1.0  # top quarter of R at sensor white
        r, g, b = measure_clip_fractions(img)
        self.assertAlmostEqual(r, 0.25, delta=0.01)
        self.assertEqual(g, 0.0)
        self.assertEqual(b, 0.0)


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


def test_histogram_log_scale_lifts_small_bins(qapp):
    from negpy.desktop.view.widgets.charts import HistogramWidget

    w = HistogramWidget()
    # A dominant peak plus a tiny tail bin: linear hides the tail, log reveals it.
    buf = np.zeros((4, 256), dtype=np.float32)
    buf[:, 128] = 1000.0
    buf[:, 200] = 1.0
    w.update_data(buf)

    assert w.log_scale() is False
    lin = w._display("l")
    assert abs(lin[128] - 1.0) < 1e-6
    assert abs(lin[200] - 0.001) < 1e-6  # 1 / 1000

    w.set_log_scale(True)
    assert w.log_scale() is True
    log = w._display("l")
    assert abs(log[128] - 1.0) < 1e-6  # peak still normalizes to 1
    # log1p(1) / log1p(1000) ≈ 0.0993 — two orders of magnitude more visible
    assert log[200] > lin[200] * 50
    assert 0.05 < log[200] < 0.15


def test_histogram_set_log_scale_idempotent_and_toggle(qapp):
    from negpy.desktop.view.widgets.charts import HistogramWidget

    w = HistogramWidget()
    received: list[bool] = []
    w.scale_changed.connect(received.append)

    # set_log_scale is a programmatic setter; it should not emit the user signal.
    w.set_log_scale(True)
    w.set_log_scale(True)  # no-op, already on
    assert w.log_scale() is True
    assert received == []

    w.set_log_scale(False)
    assert w.log_scale() is False


def test_histogram_empty_display_safe(qapp):
    from negpy.desktop.view.widgets.charts import HistogramWidget

    w = HistogramWidget()
    assert w._display("l") == []
    w.set_log_scale(True)
    assert w._display("r") == []


if __name__ == "__main__":
    unittest.main()
