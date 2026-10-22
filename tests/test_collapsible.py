"""Tests for the shared collapsible section container (widgets/collapsible.py)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QSplitter, QWidget

from conftest import FakeRepo
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection, fit_pane, make_section

if not QApplication.instance():
    _app = QApplication(sys.argv)


class TestCollapsibleDefault:
    def test_has_a_chevron_and_a_checkable_header(self) -> None:
        section = CollapsibleSection("Title")
        assert section.collapsible is True
        assert section.chevron_label is not None
        assert section.toggle_button.isCheckable() is True

    def test_unchecking_the_header_hides_the_content(self) -> None:
        section = CollapsibleSection("Title", expanded=True)
        section.toggle_button.setChecked(False)
        assert section.content_area.isHidden() is True


class TestNonCollapsible:
    def test_has_no_chevron(self) -> None:
        section = CollapsibleSection("Title", collapsible=False)
        assert section.chevron_label is None

    def test_stays_expanded_regardless_of_the_expanded_argument(self) -> None:
        section = CollapsibleSection("Title", expanded=False, collapsible=False)
        assert section.content_area.isHidden() is False

    def test_header_is_not_checkable_and_a_click_does_not_collapse_it(self) -> None:
        section = CollapsibleSection("Title", collapsible=False)
        assert section.toggle_button.isCheckable() is False
        section.toggle_button.click()
        assert section.content_area.isHidden() is False

    def test_expand_is_a_no_op(self) -> None:
        section = CollapsibleSection("Title", collapsible=False)
        section.expand()
        assert section.content_area.isHidden() is False

    def test_info_button_is_unaffected(self) -> None:
        section = CollapsibleSection("Title", collapsible=False, info=True)
        assert section.info_btn is not None

    def test_title_and_icon_are_unaffected(self) -> None:
        import qtawesome as qta

        icon = qta.icon("fa5s.cog")
        section = CollapsibleSection("Title", collapsible=False, icon=icon)
        assert section.title_label.text() == "Title"


class TestScopeButtons:
    def test_hidden_until_set_scope_buttons_is_called(self) -> None:
        section = CollapsibleSection("Calibration")
        assert section.frame_btn is None
        assert section.roll_btn is None

    def test_visible_true_shows_them_visible_false_hides_them(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_scope_buttons(visible=True, scope="roll")
        assert section.frame_btn.isHidden() is False
        assert section.roll_btn.isHidden() is False
        section.set_scope_buttons(visible=False, scope="roll")
        assert section.frame_btn.isHidden() is True
        assert section.roll_btn.isHidden() is True

    def test_the_active_half_reads_the_scope(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_scope_buttons(visible=True, scope="roll")
        assert (section.roll_btn.isChecked(), section.frame_btn.isChecked()) == (True, False)
        section.set_scope_buttons(visible=True, scope="frame")
        assert (section.roll_btn.isChecked(), section.frame_btn.isChecked()) == (False, True)

    @pytest.mark.parametrize(("scope", "color"), [("frame", THEME.border_scope_frame), ("roll", THEME.border_checked_hover)])
    def test_the_active_half_takes_its_scope_stripes_color(self, scope, color) -> None:
        section = CollapsibleSection("Calibration")
        section.set_scope_buttons(visible=True, scope=scope)

        def glyph(btn) -> set:
            image = btn.icon().pixmap(32, 32).toImage()
            return {image.pixelColor(x, y).name() for x in range(32) for y in range(32) if image.pixelColor(x, y).alpha() == 255}

        active, other = (section.frame_btn, section.roll_btn) if scope == "frame" else (section.roll_btn, section.frame_btn)
        assert glyph(active) == {QColor(color).name()}
        assert glyph(other) == {QColor(THEME.text_hint).name()}

    def test_roll_can_be_shown_disabled_rather_than_hidden(self) -> None:
        """Frames that are not one roll have no roll to move values to. The pair still
        reads Frame so the scope is stated, rather than vanishing and leaving it unsaid."""
        section = CollapsibleSection("Calibration")

        section.set_scope_buttons(visible=True, scope="frame", roll_enabled=False)

        assert section.roll_btn.isHidden() is False
        assert section.roll_btn.isEnabled() is False
        assert section.frame_btn.isEnabled() is True
        assert (section.frame_btn.isChecked(), section.roll_btn.isChecked()) == (True, False)

    def test_a_disabled_roll_half_is_enabled_again_once_there_is_a_roll(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_scope_buttons(visible=True, scope="frame", roll_enabled=False)

        section.set_scope_buttons(visible=True, scope="roll")

        assert section.roll_btn.isEnabled() is True

    def test_clicking_the_inactive_half_emits_its_scope(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_scope_buttons(visible=True, scope="roll")
        received = []
        section.scope_selected.connect(received.append)

        section.frame_btn.click()

        assert received == ["frame"]

    def test_clicking_the_active_half_emits_nothing_and_stays_checked(self) -> None:
        """The pair is a readout as much as a control: a click on the half already active
        must not leave it unchecked, which would read as a third, meaningless state."""
        section = CollapsibleSection("Calibration")
        section.set_scope_buttons(visible=True, scope="roll")
        received = []
        section.scope_selected.connect(received.append)

        section.roll_btn.click()

        assert received == []
        assert section.roll_btn.isChecked() is True

    def test_reuses_the_same_buttons_across_calls(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_scope_buttons(visible=True, scope="roll")
        first = (section.frame_btn, section.roll_btn)
        section.set_scope_buttons(visible=True, scope="frame")
        assert (section.frame_btn, section.roll_btn) == first

    def test_the_header_stripe_follows_the_lit_button(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_modified(1)

        section.set_scope_buttons(visible=True, scope="frame")
        assert section.toggle_button.property("scope") == "frame"

        section.set_scope_buttons(visible=True, scope="roll")
        assert section.toggle_button.property("scope") == "roll"

    def test_an_untouched_section_is_never_striped(self) -> None:
        """A stripe down every card says nothing; it marks the ones holding something
        other than their defaults."""
        section = CollapsibleSection("Calibration")

        section.set_scope_buttons(visible=True, scope="roll")
        assert section.toggle_button.property("scope") == ""

        section.set_modified(2)
        assert section.toggle_button.property("scope") == "roll"

        section.set_modified(0)
        assert section.toggle_button.property("scope") == ""

    def test_the_stripe_clears_with_the_pair(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_modified(1)
        section.set_scope_buttons(visible=False, scope="roll")
        assert section.toggle_button.property("scope") == ""

    def test_the_card_body_is_never_striped(self) -> None:
        """The stripe is the header's alone: a bar down the whole card read as a different
        kind of section from every other sidebar's."""
        section = CollapsibleSection("Calibration")
        section.set_modified(1)
        section.set_scope_buttons(visible=True, scope="frame")
        assert section.content_area.property("scope") is None

    def test_the_scope_does_not_touch_the_title_or_its_modified_count(self) -> None:
        """title_label's "· count" is set_modified's own, unrelated fact (how far
        from NegPy's defaults) -- the scope pair must never chain onto it."""
        section = CollapsibleSection("Calibration")
        section.set_modified(2)
        section.set_scope_buttons(visible=True, scope="frame")
        assert section.title_label.text() == "Calibration · 2"


class TestMakeSection:
    def test_collapsible_reads_and_persists_the_setting(self) -> None:
        repo = FakeRepo(section_expanded_demo=False)
        section = make_section(repo, "Demo", "demo", QWidget(), "fa5s.cog")
        assert section.content_area.isHidden() is True
        section.toggle_button.setChecked(True)
        assert repo.data["section_expanded_demo"] is True

    def test_non_collapsible_ignores_the_persisted_setting(self) -> None:
        repo = FakeRepo(section_expanded_demo=False)
        section = make_section(repo, "Demo", "demo", QWidget(), "fa5s.cog", collapsible=False)
        assert section.content_area.isHidden() is False

    def test_non_collapsible_never_writes_the_setting(self) -> None:
        repo = FakeRepo()
        make_section(repo, "Demo", "demo", QWidget(), "fa5s.cog", collapsible=False)
        assert "section_expanded_demo" not in repo.data


class TestResetToRoll:
    def test_hidden_until_the_frame_differs_from_the_roll(self) -> None:
        section = CollapsibleSection("Tone")
        assert section.roll_revert_btn.isHidden() is True

        section.set_roll_revert(True)
        assert section.roll_revert_btn.isHidden() is False
        assert section.roll_revert_available is True

        section.set_roll_revert(False)
        assert section.roll_revert_btn.isHidden() is True

    def test_a_click_asks_for_the_reset(self) -> None:
        section = CollapsibleSection("Tone")
        fired: list[bool] = []
        section.roll_revert_requested.connect(lambda: fired.append(True))
        section.set_roll_revert(True)

        section.roll_revert_btn.click()

        assert fired == [True]

    def test_it_sits_right_after_the_reset_arrow(self) -> None:
        section = CollapsibleSection("Tone")
        row = section._header_row
        assert row.indexOf(section.roll_revert_btn) == row.indexOf(section.reset_btn) + 1


def test_a_long_combo_does_not_hold_the_card_wide() -> None:
    from PyQt6.QtWidgets import QComboBox, QVBoxLayout

    from negpy.desktop.view.styles.templates import FIELD_LABEL_WIDTH

    body = QWidget()
    combo = QComboBox()
    combo.addItem("Kodak Ektar 100 (approx), a very long crosstalk matrix name")
    QVBoxLayout(body).addWidget(combo)
    section = CollapsibleSection("Calibration")

    section.set_content(body)

    assert combo.minimumWidth() == FIELD_LABEL_WIDTH
    assert combo.sizeHint().width() > FIELD_LABEL_WIDTH


def test_a_field_row_value_reserve_gives_way_in_a_narrow_card() -> None:
    from PyQt6.QtWidgets import QComboBox, QVBoxLayout

    from negpy.desktop.view.styles.templates import field_row
    from negpy.desktop.view.widgets.sliders import CompactSlider, align_slider_columns

    body = QWidget()
    col = QVBoxLayout(body)
    col.addWidget(CompactSlider("Strength", 0.0, 1.0, 0.5))
    combo = QComboBox()
    combo.addItem("Generic C41")
    row = field_row("Matrix", combo)
    col.addLayout(row)

    align_slider_columns(body)
    align_slider_columns(body)

    spacers = [row.itemAt(i).spacerItem() for i in range(row.count()) if row.itemAt(i).spacerItem() is not None]
    assert len(spacers) == 1
    assert spacers[0].sizeHint().width() > 0
    assert spacers[0].minimumSize().width() == 0


@pytest.mark.parametrize("index", [0, 1])
def test_a_folded_pane_holds_its_header_height_and_opens_to_its_saved_height(qapp, index):
    splitter = QSplitter(Qt.Orientation.Vertical)
    splitter.addWidget(QWidget())
    splitter.addWidget(QWidget())
    splitter.resize(200, 600)
    splitter.show()
    qapp.processEvents()
    pane = splitter.widget(index)

    fit_pane(splitter, index, False, 250, 30, 120)
    assert splitter.sizes()[index] == 30 and pane.minimumHeight() == pane.maximumHeight() == 30
    fit_pane(splitter, index, True, 250, 30, 120)
    assert splitter.sizes()[index] == 250 and pane.maximumHeight() > 600
    fit_pane(splitter, index, True, 900, 30, 120)
    assert splitter.sizes()[1 - index] == 120, "the other pane keeps its least height"
    splitter.hide()


def test_a_header_over_a_chart_draws_its_title_above_the_chart() -> None:
    section = CollapsibleSection("Tone", background_widget=QWidget())
    stack = section.toggle_button.parentWidget().layout()
    assert stack.currentWidget() is section.toggle_button
