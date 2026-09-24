"""The tab header bar: how many of the tab's cards are edited, and the reset, apply and
expand that reach all of them at once. Cards keep their own headers."""

from unittest.mock import MagicMock, patch

from negpy.desktop.settings_catalog import COLOR_FIELDS, GEOMETRY_FIELDS, rows_for_fields, rows_for_section
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.tab_header import TabHeader


def _card(count: int, hidden: bool = False) -> CollapsibleSection:
    section = CollapsibleSection("Card")
    section.set_modified(count)
    section.setVisible(not hidden)
    return section


def test_the_state_line_reads_how_many_cards_are_edited():
    header = TabHeader("Exposure")
    header.bind([_card(2), _card(0), _card(3)])

    assert header.title_label.text() == "2 of 3 cards edited"


def test_the_state_line_says_so_when_nothing_is_edited():
    header = TabHeader("Exposure")
    header.bind([_card(0), _card(0)])

    assert header.title_label.text() == "No cards edited"


def test_the_cards_button_collapses_then_expands_every_card():
    header = TabHeader("Exposure")
    cards = [_card(0), _card(0)]
    header.bind(cards)

    header.cards_btn.click()
    assert [c.toggle_button.isChecked() for c in cards] == [False, False]

    header.cards_btn.click()
    assert [c.toggle_button.isChecked() for c in cards] == [True, True]


def test_the_cards_button_leaves_a_retired_card_alone():
    header = TabHeader("Exposure")
    live, hidden = _card(0), _card(0, hidden=True)
    header.bind([live, hidden])

    header.cards_btn.click()

    assert live.toggle_button.isChecked() is False
    assert hidden.toggle_button.isChecked() is True


def test_count_is_the_sum_of_the_cards_below_it():
    header = TabHeader("Exposure")
    header.bind([_card(2), _card(3)])

    header.refresh()

    assert header.modified_count == 5


def test_a_card_the_mode_retired_counts_for_nothing():
    header = TabHeader("Exposure")
    header.bind([_card(2), _card(3, hidden=True)])

    header.refresh()

    assert header.modified_count == 2


def test_reset_fires_every_touched_card_once_confirmed():
    header = TabHeader("Exposure")
    touched, untouched, hidden = _card(2), _card(0), _card(4, hidden=True)
    fired = []
    for name, section in (("touched", touched), ("untouched", untouched), ("hidden", hidden)):
        section.reset_requested.connect(lambda n=name: fired.append(n))
    header.bind([touched, untouched, hidden])

    with patch("negpy.desktop.view.widgets.tab_header.confirm_reset_tab", return_value=True):
        header.reset_requested.emit()

    assert fired == ["touched"]


def test_reset_does_nothing_when_the_confirm_is_cancelled():
    header = TabHeader("Exposure")
    section = _card(2)
    fired = []
    section.reset_requested.connect(lambda: fired.append(1))
    header.bind([section])

    with patch("negpy.desktop.view.widgets.tab_header.confirm_reset_tab", return_value=False):
        header.reset_requested.emit()

    assert fired == []


def _panel_stub(hidden=()) -> MagicMock:
    panel = MagicMock()
    for key in ("geometry", "color", "tone", "local", "lab"):
        section = MagicMock()
        section.isHidden.return_value = key in hidden
        setattr(panel, f"{key}_section", section)
    return panel


def test_apply_offers_every_row_the_tabs_cards_own():
    panel = _panel_stub()

    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=None) as dialog:
        ControlsPanel._apply_tab(panel, ("geometry", "lab"))

    rows = dialog.call_args.kwargs["rows"]
    assert rows == list(dict.fromkeys(rows_for_fields(GEOMETRY_FIELDS) + rows_for_section("lab")))


def test_apply_leaves_out_a_card_the_mode_retired():
    panel = _panel_stub(hidden=("color",))

    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=None) as dialog:
        ControlsPanel._apply_tab(panel, ("color", "lab"))

    assert dialog.call_args.kwargs["rows"] == rows_for_section("lab")


def test_a_card_with_no_catalog_rows_travels_with_nothing():
    """Dodge & Burn counts on the header but has no row: a mask drawn on one frame means
    nothing on the next."""
    panel = _panel_stub()

    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=None) as dialog:
        ControlsPanel._apply_tab(panel, ("local",))

    dialog.assert_not_called()


def test_a_whole_roll_apply_is_recorded():
    """The controller splits the rows per card (record_roll_apply)."""
    panel = _panel_stub()
    applied = (rows_for_fields(GEOMETRY_FIELDS) + rows_for_fields(COLOR_FIELDS), "roll")

    with patch("negpy.desktop.view.sidebar.controls_panel.open_apply_dialog", return_value=applied):
        ControlsPanel._apply_tab(panel, ("geometry", "color"))

    panel.controller.record_roll_apply.assert_called_once_with(applied[0])


def test_a_selection_apply_records_nothing():
    """section_push is what a whole-roll apply put on the roll; a selection is not that."""
    panel = _panel_stub()

    with patch(
        "negpy.desktop.view.sidebar.controls_panel.open_apply_dialog",
        return_value=(rows_for_fields(GEOMETRY_FIELDS), "selection"),
    ):
        ControlsPanel._apply_tab(panel, ("geometry",))

    panel.controller.record_roll_apply.assert_not_called()


def _revertible(available: bool, hidden: bool = False) -> CollapsibleSection:
    section = _card(0, hidden=hidden)
    section.set_roll_revert(available)
    return section


def test_reset_to_roll_shows_when_any_card_can_reset():
    header = TabHeader("Exposure")

    header.bind([_revertible(False), _revertible(True)])
    assert header.roll_revert_btn.isHidden() is False

    header.bind([_revertible(False), _revertible(False)])
    assert header.roll_revert_btn.isHidden() is True


def test_a_retired_card_does_not_count_for_reset_to_roll():
    header = TabHeader("Exposure")
    header.bind([_revertible(True, hidden=True), _revertible(False)])

    assert header.roll_revert_btn.isHidden() is True


def test_reset_to_roll_stays_beside_the_reset_arrow():
    header = TabHeader("Exposure")
    row = header._header_row

    assert row.indexOf(header.roll_revert_btn) == row.indexOf(header.reset_btn) + 1
    assert row.indexOf(header.apply_btn) == row.indexOf(header.roll_revert_btn) + 1


def test_reset_to_roll_on_a_tab_skips_a_retired_card():
    panel = _panel_stub(hidden=("color",))

    ControlsPanel.revert_cards_to_roll(panel, ("color", "tone", "lab"))

    panel.controller.revert_to_roll.assert_called_once_with(["tone", "lab"])


def test_reset_to_roll_on_the_roll_tab_reaches_both_optics_cards():
    panel = MagicMock()
    for key in ("process", "optics"):
        getattr(panel, f"{key}_section").isHidden.return_value = False

    ControlsPanel.revert_cards_to_roll(panel, ("process", "optics"))

    panel.controller.revert_to_roll.assert_called_once_with(["process", "lens", "flatfield"])
