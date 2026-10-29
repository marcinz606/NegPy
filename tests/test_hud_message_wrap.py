"""A message that has to explain something must stay inside the canvas strip.

Most messages are a few words ("merging exposures"), but a failure has to say what went wrong
and what to do about it, so the strip grows for it instead of running off the canvas.
"""

from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.canvas.hud import CanvasHud

LONG = "Nikon High Efficiency (HE) raw — NegPy cannot decode this format. Re-shoot as Lossless Compressed, or convert to DNG."
SHORT = "merging exposures"


def _shown_strip(qapp, text, width):
    parent = QWidget()
    parent.resize(width, 400)
    hud = CanvasHud(parent)
    hud.resize(width, hud.minimumHeight())
    hud.update_info("roll.nef · 4032 x 3024 px", "Color Negative")
    hud.showMessage(text, 5000)
    hud.layout().activate()
    hud._keepalive = parent
    return hud


def test_a_long_message_wraps_inside_the_strip(qapp):
    hud = _shown_strip(qapp, LONG, 700)
    msg = hud.message
    assert msg.geometry().right() <= hud.width()
    assert msg.heightForWidth(msg.width()) > msg.fontMetrics().lineSpacing(), "expected more than one line"
    assert msg.text() == LONG, "no text may be dropped"


def test_a_short_message_stays_on_one_line(qapp):
    hud = _shown_strip(qapp, SHORT, 1400)
    msg = hud.message
    assert msg.heightForWidth(msg.width()) <= msg.fontMetrics().lineSpacing() + 2
