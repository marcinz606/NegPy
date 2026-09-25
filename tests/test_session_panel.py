from unittest.mock import MagicMock, patch

import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.sidebar.session_panel import SessionPanel
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.updater import UpdateInfo
from negpy.services.assets import rolls as rolls_service


def _controller(tmp_path, roots: list[str]) -> MagicMock:
    """A mock controller around a real session — the film strip needs a real model.
    Each path in *roots* becomes a recognized folder roll."""
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    for path in roots:
        rolls_service.recognize_folder(repo, path)

    controller = MagicMock()
    controller.session = DesktopSessionManager(repo)
    controller.library_roots.return_value = roots
    controller.has_rolls.return_value = bool(roots)
    return controller


@pytest.fixture
def panel(qapp, tmp_path, monkeypatch):
    # The update check would hit the network on construction.
    monkeypatch.setattr("negpy.desktop.view.widgets.update_dialog.find_update", lambda *a, **k: None)
    root = tmp_path / "library"
    root.mkdir()
    return SessionPanel(_controller(tmp_path, [str(root)]))


def test_library_sits_above_the_film_strip(panel):
    browser = panel.file_browser
    splitter = browser.sections_splitter

    assert splitter.indexOf(browser.library_section) < splitter.indexOf(browser.frames_section)
    assert browser.library_section.content_area.isAncestorOf(panel.library_tree)
    assert browser.frames_section.content_area.isAncestorOf(browser.list_view)


def test_thumbnail_size_slider_lives_in_the_film_strip_tally_row(panel):
    """Only the thumbnail grid reads it, so it belongs with the frames it resizes, not
    the search row shared with Library."""
    browser = panel.file_browser

    assert browser.frames_section.content_area.isAncestorOf(browser.thumb_size_slider)
    assert not browser.library_section.content_area.isAncestorOf(browser.thumb_size_slider)


def test_the_search_row_sits_above_both_sections(panel):
    browser = panel.file_browser
    layout = browser.layout()
    rows = [layout.itemAt(i) for i in range(layout.count())]
    search_at = next(i for i, item in enumerate(rows) if item.layout() and item.layout().indexOf(browser.search_input) >= 0)
    splitter_at = next(i for i, item in enumerate(rows) if item.widget() is browser.sections_splitter)

    assert search_at < splitter_at


def test_tree_is_shown_when_the_library_has_rolls(panel):
    assert panel.library_tree.isVisibleTo(panel)


def test_the_library_shows_even_with_no_rolls(qapp, tmp_path, monkeypatch):
    """The section is where rolls arrive, so hiding it when empty hides the only route
    to a first one. Its own empty label says what to do."""
    monkeypatch.setattr("negpy.desktop.view.widgets.update_dialog.find_update", lambda *a, **k: None)

    panel = SessionPanel(_controller(tmp_path, []))

    assert panel.library_tree.isVisibleTo(panel)
    assert panel.file_browser.library_section.isVisibleTo(panel)
    assert panel.library_tree.empty_label.isVisibleTo(panel.library_tree)


def test_a_collapsed_section_keeps_only_its_header(panel):
    browser = panel.file_browser
    header = browser.library_section.toggle_button.height()

    browser.library_section.toggle_button.setChecked(False)

    assert browser.library_section.maximumHeight() == header
    library, frames = browser.sections_splitter.sizes()
    assert library == header
    # The film strip is still expanded, so it keeps the rest.
    assert frames > header


def test_collapsing_both_sections_keeps_the_panel_top_aligned(panel, qapp):
    """With nothing expanded the leftover height must not spread into the gaps (#754).

    Shows the real top-level ancestor (panel), not just the embedded browser: a
    QSplitter's live layout needs a genuinely shown/sized top-level window to position
    its children correctly, unlike a plain QLayout, which computes geometry analytically
    regardless of show state.
    """
    browser = panel.file_browser
    browser.library_section.toggle_button.setChecked(False)
    browser.frames_section.toggle_button.setChecked(False)

    panel.resize(300, 900)
    panel.show()
    qapp.processEvents()

    assert browser.frames_section.y() < 200  # ~40 stacked, ~707 spread


def test_expanding_a_section_gives_it_back_a_share(panel):
    browser = panel.file_browser
    header = browser.frames_section.toggle_button.height()
    browser.frames_section.toggle_button.setChecked(False)
    assert browser.sections_splitter.sizes()[1] == header

    browser.frames_section.toggle_button.setChecked(True)

    assert browser.sections_splitter.sizes()[1] > header
    assert browser.frames_section.maximumHeight() > 1000


def test_both_open_favors_the_film_strip(panel):
    """The tree finds a roll, glanced at occasionally; the sheet is where the work happens
    and starts with most of the room. Loose bound: pixel sizes, not a pure integer ratio,
    so the splitter's own rounding shifts this around, and the point is "clearly the
    minority share", not an exact number the user can drag away from anyway."""
    library, frames = panel.file_browser.sections_splitter.sizes()

    assert library / (library + frames) < 0.4


def test_dragging_the_splitter_persists_the_split(panel):
    """Drives the persist path the way a real drag does: splitterMoved fires after Qt has
    already applied the new sizes, so this saves whatever sizes() reports, not the
    argument setSizes was given (Qt rescales it if the splitter has not been shown)."""
    browser = panel.file_browser
    browser.sections_splitter.setSizes([300, 300])
    expected = browser.sections_splitter.sizes()

    browser._on_sections_splitter_moved()

    assert panel.controller.session.repo.get_global_setting("session_sections_splitter_sizes") == expected


def test_a_new_panel_restores_the_saved_split(qapp, tmp_path, monkeypatch):
    """setSizes on an unshown splitter rescales its argument to fit a not-yet-laid-out
    guess at the total, so this checks the restore path is wired to the saved value,
    not the pixel sizes an unshown splitter ends up reporting."""
    monkeypatch.setattr("negpy.desktop.view.widgets.update_dialog.find_update", lambda *a, **k: None)
    root = tmp_path / "library"
    root.mkdir()
    controller = _controller(tmp_path, [str(root)])
    controller.session.repo.save_global_setting("session_sections_splitter_sizes", [111, 222])

    with patch("negpy.desktop.view.sidebar.files.QSplitter.setSizes") as set_sizes:
        SessionPanel(controller)

    assert [111, 222] in [list(c.args[0]) for c in set_sizes.call_args_list]


def test_the_library_button_expands_a_collapsed_section(panel):
    panel.file_browser.library_section.toggle_button.setChecked(False)

    panel.file_browser.library_requested.emit(True)

    assert panel.file_browser.library_section.toggle_button.isChecked()
    assert panel.file_browser.library_section.isVisibleTo(panel)


def test_the_library_button_prompts_an_import_when_the_library_is_empty(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.desktop.view.widgets.update_dialog.find_update", lambda *a, **k: None)
    panel = SessionPanel(_controller(tmp_path, []))
    prompted = []
    monkeypatch.setattr(panel.library_tree, "prompt_import_folder", lambda: prompted.append(1) or False)

    panel.file_browser.library_requested.emit(True)

    assert prompted


def test_the_sheet_and_the_roll_list_sort_independently(panel):
    tree_before = (panel.library_tree._sort_order, panel.library_tree._sort_descending)

    panel.file_browser._apply_sort_direction(True)
    panel.file_browser._apply_sort_order("date")

    assert (panel.library_tree._sort_order, panel.library_tree._sort_descending) == tree_before
    panel.library_tree.sort_btn.ascending_action.trigger()
    panel.library_tree.sort_btn.descending_action.trigger()
    assert panel.library_tree._sort_descending is True
    assert panel.file_browser.act_sort_date.isChecked() and panel.file_browser.act_sort_desc.isChecked()


def test_a_roll_change_drops_the_cached_walk(panel):
    panel.library_tree.rolls_changed.emit()

    panel.controller.invalidate_library_walk.assert_called_once_with()


def test_toggle_collapses_and_restores_the_library(panel):
    section = panel.file_browser.library_section

    panel.toggle_library_tree()
    assert not section.toggle_button.isChecked()
    assert not section.content_area.isVisibleTo(section)
    assert section.toggle_button.isVisibleTo(section)  # the header is the way back

    panel.toggle_library_tree()
    assert section.toggle_button.isChecked()
    assert section.content_area.isVisibleTo(section)


def test_section_states_are_remembered(panel):
    panel.toggle_library_tree()
    panel.file_browser.frames_section.toggle_button.setChecked(False)

    repo = panel.controller.session.repo
    assert repo.get_global_setting("section_expanded_library") is False
    assert repo.get_global_setting("section_expanded_frames") is False


# --- the update notice ----------------------------------------------------


def _update(**overrides) -> UpdateInfo:
    fields = dict(
        version="9.9.9",
        notes="## What's new",
        page_url="https://example.test/release",
        asset_name="NegPy-9.9.9-x86_64.AppImage",
        download_url="https://example.test/asset",
        size=1,
    )
    return UpdateInfo(**{**fields, **overrides})


def test_no_notice_until_a_newer_release_turns_up(panel):
    panel._on_update_checked(None)

    assert not panel.update_label.isVisibleTo(panel)
    assert panel.update_info is None


def test_the_notice_names_the_new_version(panel, monkeypatch):
    monkeypatch.setattr("negpy.kernel.system.updater.install_kind", lambda: "appimage")

    panel._on_update_checked(_update())

    assert panel.update_label.isVisibleTo(panel)
    assert "9.9.9" in panel.update_label.text()
    assert "#update" in panel.update_label.text()  # the click stays in the app


def test_clicking_the_notice_opens_the_update_window(panel, monkeypatch):
    opened = []
    monkeypatch.setattr(
        "negpy.desktop.view.sidebar.session_panel.UpdateDialog", lambda info, parent: MagicMock(exec=lambda: opened.append(info))
    )
    panel._on_update_checked(_update())

    panel.update_label.linkActivated.emit("#update")

    assert opened and opened[0].version == "9.9.9"


def test_the_window_stays_shut_while_the_running_version_is_current(panel, monkeypatch):
    monkeypatch.setattr(
        "negpy.desktop.view.sidebar.session_panel.UpdateDialog",
        lambda info, parent: pytest.fail("no update to show"),
    )

    panel.show_update_dialog()


def test_a_manual_check_runs_one_thread_at_a_time(panel, monkeypatch):
    started = []
    monkeypatch.setattr("negpy.desktop.view.sidebar.session_panel.start_update_check", lambda cb: started.append(cb))

    panel.check_for_updates()
    panel.check_for_updates()

    assert len(started) == 1


def test_a_manual_check_can_run_again_once_nothing_is_new(panel, monkeypatch):
    started = []
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.information", lambda *a, **k: None)
    monkeypatch.setattr("negpy.desktop.view.sidebar.session_panel.start_update_check", lambda cb: started.append(cb))
    panel.check_for_updates()

    panel._on_manual_check(None)
    panel.check_for_updates()

    assert len(started) == 2


def test_a_found_update_is_announced_with_its_version(panel):
    seen = []
    panel.update_found.connect(seen.append)

    panel._on_update_checked(_update())

    assert seen == ["9.9.9"]


def test_clearing_the_library_leaves_the_section_in_place(panel):
    panel.controller.library_cleared.emit()

    assert panel.file_browser.library_section.isVisibleTo(panel)
