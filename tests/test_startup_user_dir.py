"""Startup must survive a user directory it cannot create.

Everything NegPy keeps — databases, caches, presets, logs — lives under one directory,
and it is created before QApplication exists. A path that cannot be made (a Linux-style
/home/... on macOS, an unlinked OneDrive Documents on Windows, issue #651) used to abort
the process: the exception hook tried to raise a QMessageBox with no application object,
Qt called qFatal(), and the real cause was buried under "Must construct a QApplication
before a QWidget".
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from negpy.desktop.main import UserDirectoryError, _bootstrap_environment, _install_exception_hook


class TestBootstrapReportsAnUnusableDirectory(unittest.TestCase):
    def _raise_makedirs(self, errno, strerror, failing):
        def fake(path, exist_ok=False):
            if path == failing:
                raise OSError(errno, strerror, path)

        return fake

    def test_it_names_the_directory_and_the_reason(self):
        with patch("negpy.desktop.main.BASE_USER_DIR", "/home/nobody/NegPy-dev"):
            with patch("os.makedirs", self._raise_makedirs(45, "Operation not supported", "/home/nobody/NegPy-dev")):
                with self.assertRaises(UserDirectoryError) as caught:
                    _bootstrap_environment()

        message = str(caught.exception)
        self.assertEqual(caught.exception.failed_dir, "/home/nobody/NegPy-dev")
        self.assertIn("/home/nobody/NegPy-dev", message)
        self.assertIn("Operation not supported", message)
        self.assertIn("45", message)

    def test_it_points_at_the_override_when_one_is_set(self):
        with patch.dict(os.environ, {"NEGPY_USER_DIR": "/home/nobody/NegPy-dev"}):
            with patch("negpy.desktop.main.BASE_USER_DIR", "/home/nobody/NegPy-dev"):
                with patch("os.makedirs", self._raise_makedirs(45, "Operation not supported", "/home/nobody/NegPy-dev")):
                    with self.assertRaises(UserDirectoryError) as caught:
                        _bootstrap_environment()

        message = str(caught.exception)
        self.assertIn("NEGPY_USER_DIR", message)
        self.assertIn("$(HOME)", message, "the .env.local correction is the actionable part")

    def test_a_directory_that_can_be_made_raises_nothing(self):
        with patch("os.makedirs"), patch("negpy.desktop.main.CrosstalkProfiles"), patch("negpy.desktop.main.GearProfiles"):
            _bootstrap_environment()


class TestExceptionHookBeforeQApplication(unittest.TestCase):
    """A widget without a QApplication aborts the process, so the hook must not build one."""

    def setUp(self):
        self._saved_hook = sys.excepthook
        self.addCleanup(lambda: setattr(sys, "excepthook", self._saved_hook))

    def _fire(self):
        _install_exception_hook()
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())

    def test_no_dialog_is_built_with_no_application(self):
        box = MagicMock()
        with patch("PyQt6.QtWidgets.QMessageBox", box), patch("PyQt6.QtWidgets.QApplication") as app:
            app.instance.return_value = None
            with patch("sys.__excepthook__") as fallback:
                self._fire()

        box.critical.assert_not_called()
        fallback.assert_called_once()

    def test_the_dialog_is_still_shown_once_the_app_is_up(self):
        box = MagicMock()
        with patch("PyQt6.QtWidgets.QMessageBox", box), patch("PyQt6.QtWidgets.QApplication") as app:
            app.instance.return_value = object()
            self._fire()

        box.critical.assert_called_once()


if __name__ == "__main__":
    unittest.main()
