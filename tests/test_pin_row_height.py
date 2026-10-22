import unittest

from negpy.desktop.view.widgets.stats import ZonePlacementRows


class TestPinRowHeight(unittest.TestCase):
    """The Analysis chart is the only stretch-1 child of a fixed-height pane, so a pin row
    that grows with its glyphs resizes the chart on every refresh (#961)."""

    def test_pin_row_heights_are_fixed(self):
        rows = ZonePlacementRows()
        rows.refresh([(0, "IV⅓", 5.0, None, True)])
        for label in (rows._names[0], rows._target_labels[0], rows._lands[0]):
            self.assertEqual(label.minimumHeight(), label.maximumHeight())


if __name__ == "__main__":
    unittest.main()
