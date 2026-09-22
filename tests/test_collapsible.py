"""Tests for the shared collapsible section container (widgets/collapsible.py)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

from PyQt6.QtWidgets import QApplication, QWidget

from conftest import FakeRepo
from negpy.desktop.view.widgets.collapsible import CollapsibleSection, make_section

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
