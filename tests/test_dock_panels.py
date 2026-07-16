import sys
import unittest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow

from negpy.desktop.view.widgets.pinnable_dock import PinnableDockWidget

if not QApplication.instance():
    _app = QApplication(sys.argv)


class TestPinnableDockWidget(unittest.TestCase):
    def test_floating_title_bar_shows_pin_and_redocks(self):
        host = QMainWindow()
        host.resize(800, 600)
        pinned = {"count": 0}

        def on_pin() -> None:
            pinned["count"] += 1
            dock.setFloating(False)
            host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        dock = PinnableDockWidget("Controls", host, pin_tooltip="Dock controls panel", on_pin=on_pin)
        host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        host.show()
        QApplication.processEvents()

        self.assertIsNone(dock.titleBarWidget())

        dock.setFloating(True)
        QApplication.processEvents()
        self.assertIsNotNone(dock.titleBarWidget())
        self.assertTrue(dock.isFloating())

        bar = dock.titleBarWidget()
        self.assertIsNotNone(bar)

        from PyQt6.QtWidgets import QToolButton

        pin = bar.findChild(QToolButton)
        self.assertIsNotNone(pin)
        pin.click()
        QApplication.processEvents()

        self.assertEqual(pinned["count"], 1)
        self.assertFalse(dock.isFloating())
        self.assertIsNone(dock.titleBarWidget())

    def test_docked_title_bar_is_native(self):
        host = QMainWindow()
        dock = PinnableDockWidget("Session", host, pin_tooltip="Dock session panel", on_pin=lambda: None)
        host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        host.show()
        QApplication.processEvents()

        dock.setFloating(True)
        QApplication.processEvents()
        dock.setFloating(False)
        QApplication.processEvents()

        self.assertIsNone(dock.titleBarWidget())


class TestDockRestoreState(unittest.TestCase):
    """Restoring the startup snapshot returns a dock to its original edge and width,
    the behaviour MainWindow's pin and Reset Panel Layout rely on."""

    def _make_host(self):
        host = QMainWindow()
        host.resize(1200, 800)
        from PyQt6.QtWidgets import QTextEdit

        host.setCentralWidget(QTextEdit())

        drawer = PinnableDockWidget("Controls", host, pin_tooltip="Dock", on_pin=lambda: None)
        drawer.setObjectName("controls_dock")
        drawer.setWidget(QTextEdit())
        host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, drawer)

        session = PinnableDockWidget("Session", host, pin_tooltip="Dock", on_pin=lambda: None)
        session.setObjectName("session_dock")
        session.setWidget(QTextEdit())
        host.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, session)

        default_state = host.saveState()
        host.show()
        QApplication.processEvents()
        return host, drawer, session, default_state

    def test_restore_returns_floating_moved_dock_to_home_edge(self):
        host, drawer, _session, default_state = self._make_host()
        original_width = drawer.width()

        drawer.setFloating(True)
        QApplication.processEvents()
        drawer.move(40, 40)
        QApplication.processEvents()
        self.assertTrue(drawer.isFloating())

        host.restoreState(default_state)
        QApplication.processEvents()

        self.assertFalse(drawer.isFloating())
        self.assertEqual(host.dockWidgetArea(drawer), Qt.DockWidgetArea.RightDockWidgetArea)
        self.assertEqual(drawer.width(), original_width)

    def test_restore_resets_resized_dock_width(self):
        host, drawer, session, default_state = self._make_host()
        default_width = drawer.width()

        host.resizeDocks([drawer], [default_width + 300], Qt.Orientation.Horizontal)
        QApplication.processEvents()
        self.assertNotEqual(drawer.width(), default_width)

        host.restoreState(default_state)
        QApplication.processEvents()

        self.assertEqual(drawer.width(), default_width)
        self.assertEqual(host.dockWidgetArea(session), Qt.DockWidgetArea.LeftDockWidgetArea)


if __name__ == "__main__":
    unittest.main()
