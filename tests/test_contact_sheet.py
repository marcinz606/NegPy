import math
import unittest

import numpy as np
from PIL import Image

from negpy.services.export.contact_sheet import ContactSheetService


def _tile(h: int, w: int, value: int = 200) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestContactSheetService(unittest.TestCase):
    def test_paginates_at_38(self):
        tiles = [_tile(100, 150) for _ in range(40)]
        sheets = ContactSheetService.build_sheets(tiles)
        self.assertEqual(len(sheets), 2)

    def test_exactly_38_is_one_sheet(self):
        sheets = ContactSheetService.build_sheets([_tile(100, 150) for _ in range(38)])
        self.assertEqual(len(sheets), 1)

    def test_39_splits_to_two(self):
        sheets = ContactSheetService.build_sheets([_tile(100, 150) for _ in range(39)])
        self.assertEqual(len(sheets), 2)

    def test_empty_yields_no_sheets(self):
        self.assertEqual(ContactSheetService.build_sheets([]), [])

    def test_auto_grid_is_square_ish(self):
        cols, rows = ContactSheetService.grid_dims(38)
        self.assertEqual(cols, 7)
        self.assertEqual(rows, 6)
        self.assertGreaterEqual(cols * rows, 38)

    def test_grid_dims_general(self):
        for n in (1, 4, 9, 12, 25, 37):
            cols, rows = ContactSheetService.grid_dims(n)
            self.assertEqual(cols, math.ceil(math.sqrt(n)))
            self.assertEqual(rows, math.ceil(n / cols))

    def test_sheet_is_rgb_pil_image(self):
        sheets = ContactSheetService.build_sheets([_tile(100, 150) for _ in range(6)])
        self.assertIsInstance(sheets[0], Image.Image)
        self.assertEqual(sheets[0].mode, "RGB")

    def test_background_is_black_and_tiles_present(self):
        sheets = ContactSheetService.build_sheets([_tile(100, 150) for _ in range(4)])
        arr = np.asarray(sheets[0])
        # Top-left corner is margin -> pure black.
        self.assertTrue(np.all(arr[0, 0] == 0))
        # Some non-black content exists (the tiles).
        self.assertTrue(np.any(arr > 0))

    def test_mixed_aspect_tiles_fit_within_cells(self):
        # Portrait + landscape mixed; must not raise and stays a single sheet.
        tiles = [_tile(300, 100), _tile(100, 300), _tile(200, 200)]
        sheets = ContactSheetService.build_sheets(tiles)
        self.assertEqual(len(sheets), 1)
        self.assertIsInstance(sheets[0], Image.Image)

    def test_custom_max_tiles_paginates(self):
        sheets = ContactSheetService.build_sheets([_tile(100, 150) for _ in range(5)], max_tiles=2)
        self.assertEqual(len(sheets), 3)

    def test_custom_layout_dimensions(self):
        # 4 tiles -> 2x2 grid; sheet size derives from cell_px/gap/margin.
        sheets = ContactSheetService.build_sheets(
            [_tile(200, 200) for _ in range(4)],
            show_labels=False,
            cell_px=200,
            gap=4,
            margin=8,
        )
        arr = np.asarray(sheets[0])
        cols = rows = 2
        expected_w = 8 * 2 + cols * 200 + (cols - 1) * 4
        expected_h = 8 * 2 + rows * 200 + (rows - 1) * 4
        self.assertEqual(arr.shape[1], expected_w)
        self.assertEqual(arr.shape[0], expected_h)

    def test_custom_background_color(self):
        sheets = ContactSheetService.build_sheets([_tile(100, 150)], background_color="#ff0000")
        arr = np.asarray(sheets[0])
        self.assertTrue(np.all(arr[0, 0] == [255, 0, 0]))

    def test_labels_increase_sheet_height(self):
        tiles = [_tile(100, 150) for _ in range(4)]
        labels = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        without = ContactSheetService.build_sheets(tiles, show_labels=False, cell_px=100, gap=4, margin=8)
        with_labels = ContactSheetService.build_sheets(
            tiles, labels=labels, show_labels=True, cell_px=100, gap=4, margin=8
        )
        self.assertGreater(np.asarray(with_labels[0]).shape[0], np.asarray(without[0]).shape[0])

    def test_caption_band_present(self):
        # Band tint = round(0.85*bg + 0.15*fg); with black bg + green label -> (0,38,0).
        sheets = ContactSheetService.build_sheets(
            [_tile(200, 200)],
            labels=["frame_one.jpg"],
            show_labels=True,
            background_color="#000000",
            label_color="#00ff00",
            cell_px=200,
        )
        arr = np.asarray(sheets[0])
        band = np.array([0, 38, 0])
        self.assertTrue(np.any(np.all(arr == band, axis=-1)))


if __name__ == "__main__":
    unittest.main()
