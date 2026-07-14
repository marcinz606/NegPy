"""The global exception hook logs unhandled (slot) exceptions instead of letting PyQt abort."""

import sys
from unittest.mock import patch


def test_exception_hook_logs_and_survives():
    import negpy.desktop.main as m

    old_hook = sys.excepthook
    try:
        with patch("PyQt6.QtWidgets.QMessageBox.critical") as box:
            m._install_exception_hook()
            hook = sys.excepthook
            assert hook is not old_hook  # a custom hook is installed
            with patch.object(m.logger, "critical") as crit:
                try:
                    raise ValueError("boom in a slot")
                except ValueError:
                    hook(*sys.exc_info())  # must NOT re-raise / abort the process
                crit.assert_called_once()  # the full traceback was logged (to negpy.log)
            box.assert_called_once()  # and the user got a non-fatal notice
    finally:
        sys.excepthook = old_hook


def test_exception_hook_passes_keyboard_interrupt_through():
    import negpy.desktop.main as m

    old_hook = sys.excepthook
    try:
        m._install_exception_hook()
        with patch("sys.__excepthook__") as default:
            m_hook = sys.excepthook
            m_hook(KeyboardInterrupt, KeyboardInterrupt(), None)  # Ctrl-C keeps the normal behaviour
            default.assert_called_once()
    finally:
        sys.excepthook = old_hook
