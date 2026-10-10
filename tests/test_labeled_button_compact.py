from PyQt6.QtWidgets import QHBoxLayout, QWidget

from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, labeled_action, labeled_toggle, tool_toggle


def _row(width: int, *buttons) -> QWidget:
    root = QWidget()
    row = QHBoxLayout(root)
    row.setContentsMargins(0, 0, 0, 0)
    for btn in buttons:
        row.addWidget(btn, 1)
    root.resize(width, 40)
    root.show()
    return root


def test_labeled_buttons_can_shrink_to_their_icon(qapp):
    for btn in (labeled_action("fa5s.layer-group", " Detect All", "tip"), labeled_toggle("fa5s.camera", " CA", False, "tip")):
        assert btn.minimumSizeHint().width() == ICON_BUTTON_WIDTH
        assert btn.sizeHint().width() > ICON_BUTTON_WIDTH


def test_a_narrow_row_squeezes_the_labels_and_keeps_the_text(qapp):
    buttons = [labeled_action("fa5s.layer-group", f" Action {i}", "tip") for i in range(3)]
    root = _row(3 * ICON_BUTTON_WIDTH + 20, *buttons)

    assert all(b.width() < b.sizeHint().width() for b in buttons)
    assert all(b._compact() for b in buttons)
    assert buttons[0].text() == " Action 0"

    root.resize(3 * buttons[0].sizeHint().width() + 40, 40)
    qapp.processEvents()
    assert not any(b._compact() for b in buttons)


def test_text_only_and_icon_only_buttons_keep_their_own_minimum(qapp):
    text_only = tool_toggle("", "Linear", "tip")
    icon_only = tool_toggle("fa5s.magic", "", "tip")
    assert text_only.minimumSizeHint().width() > ICON_BUTTON_WIDTH
    assert not icon_only._compact()
