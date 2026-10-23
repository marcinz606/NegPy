from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.styles.templates import hint_label, load_stylesheet, set_hint_kind
from negpy.desktop.view.styles.theme import THEME


def _hint(host: QWidget, text: str, kind: str):
    lbl = hint_label(text)
    lbl.setParent(host)
    set_hint_kind(lbl, kind)
    lbl.resize(320, 40)
    return lbl


def _icon_pixels(lbl, color: str) -> int:
    """Pixels near *color* in the label's left margin."""
    image = lbl.grab().toImage()
    want = QColor(color)
    near = 0
    for x in range(lbl.contentsMargins().left()):
        for y in range(image.height()):
            c = image.pixelColor(x, y)
            near += abs(c.red() - want.red()) + abs(c.green() - want.green()) + abs(c.blue() - want.blue()) < 60
    return near


def test_a_warning_or_an_error_leads_with_an_icon_in_its_color(qapp):
    host = QWidget()
    host.setStyleSheet(load_stylesheet())
    for kind, color in (("warning", THEME.warn_amber), ("error", THEME.error_text)):
        lbl = _hint(host, "Something needs a look.", kind)
        assert lbl.contentsMargins().left() > THEME.space_sm
        assert _icon_pixels(lbl, color) > 0, kind


def test_a_plain_hint_keeps_no_icon_room_and_loses_it_when_it_stops_warning(qapp):
    host = QWidget()
    host.setStyleSheet(load_stylesheet())
    lbl = _hint(host, "Detected: sRGB", "muted")
    assert lbl.contentsMargins().left() == 0
    set_hint_kind(lbl, "warning")
    assert lbl.contentsMargins().left() > 0
    set_hint_kind(lbl, "muted")
    assert lbl.contentsMargins().left() == 0
