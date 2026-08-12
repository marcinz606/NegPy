"""A toast that has to explain something must stay on the canvas.

Most toasts are a few words ("merging exposures"), but a failure has to say what went wrong
and what to do about it. Unwrapped, such a message ran off both edges of the canvas.
"""

from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.canvas.hud import _TOAST_WIDTH_RATIO, CanvasHud

LONG = "Nikon High Efficiency (HE) raw — NegPy cannot decode this format. Re-shoot as Lossless Compressed, or convert to DNG."
SHORT = "merging exposures"


def _hud(qapp, width=1400, height=900):
    """The parent is returned too: dropping it lets Qt delete the HUD as its child."""
    parent = QWidget()
    parent.resize(width, height)
    hud = CanvasHud(parent)
    hud.resize(width, height)
    hud._keepalive = parent
    return hud


def test_a_long_message_stays_within_the_canvas(qapp):
    hud = _hud(qapp)
    hud.showMessage(LONG, 5000)
    hud.toast.adjustSize()
    assert hud.toast.width() <= hud.width(), "the toast ran off the canvas"
    assert hud.toast.width() <= int(hud.width() * _TOAST_WIDTH_RATIO) + 1, "it should leave room around itself"


def test_a_long_message_wraps_rather_than_truncating(qapp):
    """Wrapping, not eliding — the remedy is in the second half of the sentence."""
    hud = _hud(qapp)
    hud.showMessage(LONG, 5000)
    hud.toast.adjustSize()
    assert hud.toast.height() > hud.toast.fontMetrics().lineSpacing(), "expected more than one line"
    assert hud.toast.text().replace("\n", " ").strip() == LONG.lower(), "no text may be dropped"


def test_a_short_message_keeps_its_natural_width(qapp):
    """The floor must not inflate an ordinary toast into a banner."""
    hud = _hud(qapp)
    hud.showMessage(SHORT, 3000)
    hud.toast.adjustSize()
    assert hud.toast.width() < int(hud.width() * _TOAST_WIDTH_RATIO), "a short toast should not fill the cap"
    assert hud.toast.height() <= hud.toast.fontMetrics().lineSpacing() * 2


def test_it_fits_before_the_hud_has_ever_been_resized(qapp):
    """A load failure can post a toast during startup, before any resize event."""
    parent = QWidget()
    parent.resize(1200, 800)
    hud = CanvasHud(parent)
    hud._keepalive = parent
    hud.showMessage(LONG, 5000)
    hud.toast.adjustSize()
    assert hud.toast.width() <= max(1200, hud.width())


def test_a_narrow_canvas_still_gets_a_usable_width(qapp):
    hud = _hud(qapp, width=420)
    hud.showMessage(LONG, 5000)
    hud.toast.adjustSize()
    assert hud.toast.width() >= 320, "the floor keeps it from collapsing to a sliver"
