"""Simple mode: lending a card to another place and back, the trimmed tab bar, and where the
right panel sends a tab or card it hides."""

from unittest.mock import MagicMock

from PyQt6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from negpy.desktop.view.sidebar.right_panel import SIMPLE_EDIT_SECTIONS, SIMPLE_HIDDEN_GROUPS, RightPanel
from negpy.desktop.view.simple_mode import ROLL_CARDS
from negpy.desktop.view.widgets.borrow import borrow
from negpy.desktop.view.widgets.overflow_bar import OverflowBar


def _column(*names: str) -> tuple[QWidget, QVBoxLayout, list[QLabel]]:
    page = QWidget()
    layout = QVBoxLayout(page)
    labels = [QLabel(name) for name in names]
    for label in labels:
        layout.addWidget(label)
    return page, layout, labels


def test_a_borrowed_card_returns_to_its_own_slot():
    page, layout, (a, b, c) = _column("a", "b", "c")
    restore = borrow(b)

    elsewhere = QWidget()
    QVBoxLayout(elsewhere).addWidget(b)
    assert b.parentWidget() is elsewhere

    restore()
    assert b.parentWidget() is page
    assert [layout.itemAt(i).widget() for i in range(layout.count())] == [a, b, c]


def test_a_borrowed_stack_page_keeps_every_other_page_at_its_index():
    """The right panel finds its tabs' pages by index, so lending one must not shift the rest."""
    stack = QStackedWidget()
    pages = [QWidget() for _ in range(3)]
    for p in pages:
        stack.addWidget(p)
    stack.setCurrentWidget(pages[2])

    restore = borrow(pages[1])
    elsewhere = QWidget()
    QVBoxLayout(elsewhere).addWidget(pages[1])
    assert stack.count() == 3
    assert stack.widget(2) is pages[2]
    assert stack.currentWidget() is pages[2]

    restore()
    assert [stack.widget(i) for i in range(3)] == pages
    assert stack.currentWidget() is pages[2]


def test_lending_the_front_page_leaves_its_slot_in_front():
    stack = QStackedWidget()
    pages = [QWidget() for _ in range(2)]
    for p in pages:
        stack.addWidget(p)
    stack.setCurrentWidget(pages[0])

    restore = borrow(pages[0])
    elsewhere = QWidget()
    QVBoxLayout(elsewhere).addWidget(pages[0])
    restore()
    assert stack.currentWidget() is pages[0]


def test_an_excluded_tab_leaves_the_bar_and_its_overflow_menu():
    bar = OverflowBar(min_item=10)
    buttons = [QLabel(str(i)) for i in range(4)]
    for i, b in enumerate(buttons):
        bar.add_button(b, str(i))
    bar.resize(400, bar.height())

    bar.set_excluded(1, True)
    assert 1 not in bar._visible_items(400)
    assert buttons[1].isHidden()

    bar.resize(12, bar.height())
    bar._relayout()
    assert 1 not in bar._hidden

    bar.set_excluded(1, False)
    assert 1 in bar._visible_items(400)


def test_simple_mode_keeps_library_edit_and_scan():
    assert set(SIMPLE_HIDDEN_GROUPS) == {"roll", "metadata", "export"}


def test_the_simple_edit_page_holds_the_asked_for_cards_in_order():
    assert SIMPLE_EDIT_SECTIONS == (
        "color_section",
        "tone_section",
        "paper_section",
        "autocrop_section",
        "geometry_section",
        "lab_section",
        "retouch_section",
        "finish_section",
    )


def test_every_roll_card_is_a_roll_tab_card():
    from negpy.desktop.view.sidebar.right_panel import _ROLL_SECTION_ATTRS

    attrs = {attr for attr, _label in ROLL_CARDS}
    assert attrs - {"film_section"} == set(_ROLL_SECTION_ATTRS)


def _simple_stub() -> MagicMock:
    panel = MagicMock()
    panel._simple_mode = True
    return panel


def test_in_simple_mode_the_roll_tab_opens_a_popup():
    panel = _simple_stub()
    RightPanel.show_tab_by_key(panel, "roll")
    panel.popup_requested.emit.assert_called_once_with("roll:film_section")
    panel._switch_group.assert_not_called()


def test_in_simple_mode_the_export_tab_opens_export_preferences():
    panel = _simple_stub()
    RightPanel.show_tab_by_key(panel, "export")
    panel.popup_requested.emit.assert_called_once_with("export")


def test_in_simple_mode_a_frame_tab_lands_on_its_card_in_edit():
    panel = _simple_stub()
    RightPanel.show_tab_by_key(panel, "color")
    panel.show_simple_edit.assert_called_once_with("lab_section")


def test_in_simple_mode_a_roll_card_opens_in_a_popup():
    panel = _simple_stub()
    RightPanel.reveal_section(panel, "sensor_section")
    panel.popup_requested.emit.assert_called_once_with("roll:sensor_section")


def test_in_simple_mode_an_edit_card_is_shown_on_the_edit_page():
    panel = _simple_stub()
    RightPanel.reveal_section(panel, "retouch_section")
    panel.show_simple_edit.assert_called_once_with("retouch_section")


def test_the_edit_tab_click_lands_on_the_edit_page_in_simple_mode():
    panel = _simple_stub()
    panel._group_keys = ["library", "roll", "frame"]
    RightPanel._on_group_clicked(panel, 2)
    panel.show_simple_edit.assert_called_once_with()
    panel._switch_group.assert_not_called()


def test_advanced_mode_group_clicks_are_unchanged():
    panel = MagicMock()
    panel._simple_mode = False
    panel._group_keys = ["library", "roll", "frame"]
    RightPanel._on_group_clicked(panel, 2)
    panel._switch_group.assert_called_once_with(2)


def test_a_popup_action_runs_then_closes_and_gives_the_card_back():
    """Export Preferences: Enter (the default button) exports, then the pop-up closes."""
    from PyQt6.QtWidgets import QPushButton

    from negpy.desktop.view.simple_mode import CardPopup

    host = QWidget()
    page, layout, (a, b, c) = _column("a", "b", "c")
    ran = []
    popup = CardPopup("Test", b, host, action=("Export", lambda: ran.append(b.parentWidget() is not page)))
    popup.show()
    default = next(btn for btn in popup.findChildren(QPushButton) if btn.isDefault())
    assert default.text() == "Export"

    default.click()
    assert ran == [True]
    assert b.parentWidget() is page
    assert [layout.itemAt(i).widget() for i in range(layout.count())] == [a, b, c]


def test_enter_closes_a_popup_without_an_action():
    """A Roll card pop-up: Enter (the default Close button) closes it and gives the card back."""
    from PyQt6.QtWidgets import QPushButton

    from negpy.desktop.view.simple_mode import CardPopup

    host = QWidget()
    page, layout, (a, b, c) = _column("a", "b", "c")
    popup = CardPopup("Test", b, host)
    popup.show()
    default = next(btn for btn in popup.findChildren(QPushButton) if btn.isDefault())
    assert default.text() == "Close"

    default.click()
    assert not popup.isVisible()
    assert b.parentWidget() is page
    assert [layout.itemAt(i).widget() for i in range(layout.count())] == [a, b, c]
