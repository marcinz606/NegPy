import unittest
from types import SimpleNamespace

import pytest
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import QApplication, QMainWindow

from conftest import FakeRepo

from negpy.desktop.view.main_window import MainWindow, _DEFAULT_H, _DEFAULT_W, _clamp_geometry


class _GeometryRepo(FakeRepo):
    def save_global_settings(self, settings: dict) -> None:
        self.data.update(settings)


class _GeometryWindow(MainWindow):
    def __init__(self, repo: FakeRepo) -> None:
        QMainWindow.__init__(self)
        self.controller = SimpleNamespace(session=SimpleNamespace(repo=repo))
        self._restore_window_geometry()

    def showEvent(self, event: QShowEvent) -> None:
        QMainWindow.showEvent(self, event)


@pytest.mark.parametrize("mode", ["normal", "maximized", "fullscreen", "minimized"])
def test_window_state_survives_restart(qapp: QApplication, mode: str) -> None:
    repo = _GeometryRepo()
    first = _GeometryWindow(repo)
    first.resize(480, 320)
    first.move(60, 70)
    first.show()
    qapp.processEvents()
    normal = first.geometry()
    if mode == "fullscreen":
        first.showFullScreen()
    elif mode == "maximized":
        first.showMaximized()
    elif mode == "minimized":
        first.showMinimized()
    qapp.processEvents()
    first.close()

    second = _GeometryWindow(repo)
    try:
        second.show()
        qapp.processEvents()
        assert second.isFullScreen() == (mode == "fullscreen")
        assert second.isMaximized() == (mode == "maximized")
        assert not second.isMinimized()
        second.showNormal()
        qapp.processEvents()
        assert second.geometry() == normal
    finally:
        second.close()

    third = _GeometryWindow(repo)
    try:
        third.show()
        qapp.processEvents()
        assert not third.isFullScreen()
        assert not third.isMaximized()
        assert third.geometry() == normal
    finally:
        third.close()


@pytest.mark.parametrize("encoded", [None, "not valid geometry", "", 123])
@pytest.mark.parametrize("maximized", [False, True])
def test_legacy_geometry_is_used_when_qt_geometry_is_unavailable(qapp: QApplication, encoded: str | int | None, maximized: bool) -> None:
    repo = _GeometryRepo(window_geometry=[60, 70, 480, 320], window_geometry_qt=encoded, window_maximized=maximized)
    window = _GeometryWindow(repo)
    try:
        assert window.geometry().getRect() == (60, 70, 480, 320)
        assert window.isMaximized() == maximized
    finally:
        window.close()


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
