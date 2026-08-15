"""The application style's deliberate deviations from Fusion."""

import sys

import pytest
from PyQt6.QtWidgets import QStyle

from negpy.desktop.main import _AppStyle


@pytest.fixture
def style(qapp):
    return _AppStyle("Fusion")


def test_tooltips_wait_longer_than_the_fusion_default(style):
    assert style.styleHint(QStyle.StyleHint.SH_ToolTip_WakeUpDelay) == 1400


@pytest.mark.skipif(sys.platform != "darwin", reason="the hint is only overridden on macOS")
def test_no_mnemonic_underline_on_macos(style):
    """Qt puts the mnemonic in its own standard-button text ("&Yes"), but macOS binds no
    mnemonics — QKeySequence.mnemonic() is empty there — so Fusion's underline marks a key
    that cannot be pressed. The native style draws none; match it."""
    from PyQt6.QtGui import QKeySequence

    assert QKeySequence.mnemonic("&Yes").isEmpty(), "macOS bound a mnemonic; the underline may be honest now"
    assert style.styleHint(QStyle.StyleHint.SH_UnderlineShortcut) == 0


@pytest.mark.skipif(sys.platform == "darwin", reason="only macOS suppresses the underline")
def test_mnemonic_underline_kept_where_it_works(style):
    """Alt+letter is a real binding on Windows and Linux — do not take the cue away."""
    assert style.styleHint(QStyle.StyleHint.SH_UnderlineShortcut) != 0
