import unittest

from negpy.desktop.view.main_window import _DEFAULT_H, _DEFAULT_W, _clamp_geometry


def _inside(geo, avail):
    x, y, w, h = geo
    ax, ay, aw, ah = avail
    return ax <= x and ay <= y and x + w <= ax + aw and y + h <= ay + ah


class TestClampGeometry(unittest.TestCase):
    SMALL = (0, 0, 1366, 728)  # 1368x768 minus a taskbar

    def test_oversized_saved_shrinks_to_fit(self):
        geo = _clamp_geometry((10, 10, _DEFAULT_W, _DEFAULT_H), self.SMALL)
        self.assertEqual(geo, (0, 0, 1366, 728))
        self.assertTrue(_inside(geo, self.SMALL))

    def test_offscreen_position_pulled_inside(self):
        geo = _clamp_geometry((-50, -30, 800, 600), self.SMALL)
        self.assertTrue(_inside(geo, self.SMALL))
        # far-positive position is pulled back so the window stays fully visible
        geo2 = _clamp_geometry((5000, 5000, 800, 600), self.SMALL)
        self.assertTrue(_inside(geo2, self.SMALL))

    def test_default_centered_and_clamped(self):
        geo = _clamp_geometry(None, self.SMALL)
        self.assertEqual(geo, (0, 0, 1366, 728))  # default exceeds work area -> filled
        # on a big screen the default size is centered, not stretched
        big = (0, 0, 2560, 1440)
        x, y, w, h = _clamp_geometry(None, big)
        self.assertEqual((w, h), (_DEFAULT_W, _DEFAULT_H))
        self.assertEqual((x, y), ((2560 - _DEFAULT_W) // 2, (1440 - _DEFAULT_H) // 2))

    def test_screen_offset_respected(self):
        # second monitor whose work area starts at x=1920
        avail = (1920, 0, 1366, 728)
        geo = _clamp_geometry((0, 0, 1000, 700), avail)
        self.assertTrue(_inside(geo, avail))
        self.assertGreaterEqual(geo[0], 1920)


if __name__ == "__main__":
    unittest.main()
