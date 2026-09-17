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


class TestLockButton:
    def test_hidden_until_set_lock_button_is_called(self) -> None:
        section = CollapsibleSection("Calibration")
        assert section.lock_btn is None

    def test_visible_true_shows_it_visible_false_hides_it(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_lock_button(visible=True, locked=False)
        assert section.lock_btn.isHidden() is False
        section.set_lock_button(visible=False, locked=False)
        assert section.lock_btn.isHidden() is True

    def test_clicking_it_emits_the_opposite_of_the_current_state(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_lock_button(visible=True, locked=False)
        received = []
        section.lock_toggled.connect(received.append)

        section.lock_btn.click()

        assert received == [True]

    def test_clicking_a_locked_button_emits_false(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_lock_button(visible=True, locked=True)
        received = []
        section.lock_toggled.connect(received.append)

        section.lock_btn.click()

        assert received == [False]

    def test_reuses_the_same_button_across_calls(self) -> None:
        section = CollapsibleSection("Calibration")
        section.set_lock_button(visible=True, locked=False)
        first = section.lock_btn
        section.set_lock_button(visible=True, locked=True)
        assert section.lock_btn is first


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
