import unittest

import numpy as np

from negpy.features.exposure.stats import negative_statistics
from negpy.features.retouch.logic import repair_coverage


def _names(rows):
    return [r.name for r in rows]


def _by_name(rows, name):
    return next(r for r in rows if r.name == name)


class TestRepairCoverage(unittest.TestCase):
    def test_nothing_repaired_is_all_zero(self):
        self.assertEqual(repair_coverage(None, None, None), (0.0, 0.0, 0.0))

    def test_fraction_is_share_of_the_mask(self):
        ir = np.zeros((10, 10), dtype=bool)
        ir[:2] = True
        self.assertAlmostEqual(repair_coverage(ir, None, None)[0], 0.20)

    def test_fraction_is_scale_free_across_grids(self):
        """IR and detection run at different resolutions; the same share must read the same."""
        small = np.zeros((10, 10), dtype=bool)
        small[0] = True
        big = np.zeros((100, 100), dtype=bool)
        big[:10] = True
        self.assertAlmostEqual(repair_coverage(small, None, None)[0], repair_coverage(big, None, None)[0])

    def test_same_grid_hair_masks_union_rather_than_sum(self):
        a = np.zeros((10, 10), dtype=bool)
        a[:3] = True
        b = np.zeros((10, 10), dtype=bool)
        b[2:4] = True
        # Overlapping rows 2; the union is rows 0-3 = 40%, the sum would be 50%.
        self.assertAlmostEqual(repair_coverage(None, None, [a, b])[2], 0.40)

    def test_cross_grid_hair_masks_take_the_largest(self):
        small = np.ones((4, 4), dtype=bool)
        big = np.zeros((100, 100), dtype=bool)
        big[:5] = True
        self.assertAlmostEqual(repair_coverage(None, None, [small, big])[2], 1.0)


class TestRepairRow(unittest.TestCase):
    def _rows(self, repair):
        return negative_statistics(1.3, 0.46, 0.0, 0.0, repair=repair)

    def test_absent_when_nothing_was_repaired(self):
        self.assertNotIn("Repair", _names(self._rows((0.0, 0.0, 0.0))))
        self.assertNotIn("Repair", _names(self._rows(None)))

    def test_lists_only_the_routes_that_fired(self):
        row = _by_name(self._rows((0.012, 0.0, 0.0)), "Repair")
        self.assertEqual(row.value, "IR 1.20%")
        self.assertFalse(row.warn)

    def test_warns_once_a_route_rewrites_more_than_five_percent(self):
        self.assertTrue(_by_name(self._rows((0.06, 0.0, 0.0)), "Repair").warn)
        self.assertFalse(_by_name(self._rows((0.04, 0.0, 0.0)), "Repair").warn)

    def test_sits_after_scan_clip(self):
        rows = _names(negative_statistics(1.3, 0.46, 0.0, 0.0, scan_clip=(0.5, 0.0, 0.0), repair=(0.02, 0.0, 0.0)))
        self.assertEqual(rows[-2:], ["Scan clip", "Repair"])


class TestStatsWidgetCapacity(unittest.TestCase):
    def test_shows_every_optional_row(self):
        """Repair pushed the read-out past the four rows the widget used to hold."""
        from negpy.desktop.view.widgets.stats import NegativeStatsWidget

        rows = negative_statistics(1.3, 0.46, 0.0, 0.0, scan_clip=(0.5, 0.0, 0.0), repair=(0.02, 0.0, 0.0))
        widget = NegativeStatsWidget()
        widget.update_stats(rows)
        self.assertGreaterEqual(widget._ROWS, len(rows))
        self.assertEqual([lbl.text() for lbl in widget._names[: len(rows)]], _names(rows))

    def test_unused_rows_are_hidden_not_blank(self):
        from negpy.desktop.view.widgets.stats import NegativeStatsWidget

        widget = NegativeStatsWidget()
        widget.update_stats(negative_statistics(1.3, 0.46, 0.0, 0.0))
        self.assertTrue(widget._names[0].isVisible() or widget._names[0].isVisibleTo(widget))
        self.assertFalse(widget._names[5].isVisibleTo(widget))


if __name__ == "__main__":
    unittest.main()
