import unittest

from negpy.desktop.view.widgets.stats import DensitometerRow, ZonePlacementRows
from negpy.features.exposure.densitometer import DensitometerReading, zone_roman


def _reading(zone: float) -> DensitometerReading:
    return DensitometerReading((0.12, 0.34, 0.56), 0.4, 1.23, zone)


class TestProbeRowHeight(unittest.TestCase):
    """The Analysis chart is the only stretch-1 child of a fixed-height pane, so a probe
    row that grows with its glyphs resizes the chart on every hover (#961)."""

    def test_probe_value_height_is_fixed(self):
        row = DensitometerRow()
        value = row._value
        self.assertEqual(value.minimumHeight(), value.maximumHeight())

    def test_probe_row_hint_constant_across_readings(self):
        row = DensitometerRow()
        row.set_reading(None)
        empty = row.sizeHint().height()
        row.set_reading(_reading(4.0))
        self.assertEqual(zone_roman(4.0), "IV")
        whole = row.sizeHint().height()
        row.set_reading(_reading(4.33))
        self.assertEqual(zone_roman(4.33), "IV⅓")
        third = row.sizeHint().height()
        self.assertEqual((empty, whole), (third, third))

    def test_pin_row_heights_are_fixed(self):
        rows = ZonePlacementRows()
        rows.refresh([(0, "IV⅓", 5.0, None, True)])
        for label in (rows._names[0], rows._target_labels[0], rows._lands[0]):
            self.assertEqual(label.minimumHeight(), label.maximumHeight())


if __name__ == "__main__":
    unittest.main()
