from PyQt6.QtWidgets import QLabel, QPushButton

from negpy.desktop.main import AppEventFilter
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.styles.templates import wrap_tooltip


def _installed(qapp):
    guard = AppEventFilter(qapp)
    qapp.installEventFilter(guard)
    return guard


def test_a_plain_tooltip_wraps_and_keeps_its_line_breaks(qapp):
    guard = _installed(qapp)
    try:
        label = QLabel()
        label.setToolTip("Filter the sheet.\nFields: film & camera")
        assert label.toolTip() == "<qt>Filter the sheet.<br>Fields: film &amp; camera</qt>"
    finally:
        qapp.removeEventFilter(guard)


def test_an_icon_only_button_is_named_after_its_tooltip(qapp):
    guard = _installed(qapp)
    try:
        icon_btn = QPushButton()
        icon_btn.setToolTip("Hot Folder — automatically load new images")
        assert icon_btn.accessibleName() == "Hot Folder"

        chip_btn = QPushButton()
        chip_btn.setToolTip(wrap_tooltip(tooltip_with_shortcut("Undo", "undo")))
        assert chip_btn.accessibleName() == "Undo"

        text_btn = QPushButton("Scan")
        text_btn.setToolTip("Scan the selected frames")
        assert text_btn.accessibleName() == ""
    finally:
        qapp.removeEventFilter(guard)
