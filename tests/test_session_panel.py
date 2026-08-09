from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.sidebar.session_panel import SessionPanel
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.updater import UpdateInfo


def _controller(tmp_path, roots: list[str]) -> MagicMock:
    """A mock controller around a real session — the film strip needs a real model."""
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    repo.save_global_setting("library_roots", roots)

    controller = MagicMock()
    controller.session = DesktopSessionManager(repo)
    controller.library_roots.return_value = roots
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
    order = [browser.layout().itemAt(i).widget() for i in range(browser.layout().count())]

    assert order.index(browser.library_section) < order.index(browser.frames_section)
    assert browser.library_section.content_area.isAncestorOf(panel.library_tree)
    assert browser.frames_section.content_area.isAncestorOf(browser.list_view)


def test_the_search_row_sits_above_both_sections(panel):
    browser = panel.file_browser
    layout = browser.layout()
    rows = [layout.itemAt(i) for i in range(layout.count())]
    search_at = next(i for i, item in enumerate(rows) if item.layout() and item.layout().indexOf(browser.search_input) >= 0)
    library_at = next(i for i, item in enumerate(rows) if item.widget() is browser.library_section)

    assert search_at < library_at


def test_tree_is_shown_when_the_library_has_roots(panel):
    assert panel.library_tree.isVisibleTo(panel)


def test_tree_hidden_when_no_roots(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.desktop.view.widgets.update_dialog.find_update", lambda *a, **k: None)

    panel = SessionPanel(_controller(tmp_path, []))

    assert not panel.library_tree.isVisibleTo(panel)


def test_a_collapsed_section_keeps_only_its_header(panel):
    browser = panel.file_browser
    header = browser.library_section.toggle_button.height()

    browser.library_section.toggle_button.setChecked(False)

    assert browser.library_section.maximumHeight() == header
    assert browser.layout().stretch(browser.layout().indexOf(browser.library_section)) == 0
    # The film strip is still expanded, so it keeps its share.
    assert browser.layout().stretch(browser.layout().indexOf(browser.frames_section)) > 0


def test_collapsing_both_sections_keeps_the_panel_top_aligned(panel, qapp):
    """With nothing expanded the leftover height must not spread into the gaps (#754)."""
    browser = panel.file_browser
    browser.library_section.toggle_button.setChecked(False)
    browser.frames_section.toggle_button.setChecked(False)

    browser.resize(300, 900)
    browser.show()
    qapp.processEvents()

    assert browser.frames_section.y() < 200  # ~115 stacked, ~707 spread


def test_expanding_a_section_gives_it_back_a_share(panel):
    browser = panel.file_browser
    browser.frames_section.toggle_button.setChecked(False)
    assert browser.layout().stretch(browser.layout().indexOf(browser.frames_section)) == 0

    browser.frames_section.toggle_button.setChecked(True)

    assert browser.layout().stretch(browser.layout().indexOf(browser.frames_section)) > 0
    assert browser.frames_section.maximumHeight() > 1000


def test_both_open_splits_the_panel_40_60(panel):
    """The tree finds a roll, the sheet is where the work happens."""
    browser = panel.file_browser
    layout = browser.layout()

    library = layout.stretch(layout.indexOf(browser.library_section))
    frames = layout.stretch(layout.indexOf(browser.frames_section))

    assert library / (library + frames) == pytest.approx(0.4)


def test_opening_an_image_less_folder_from_the_tree_loads_nothing(panel, tmp_path):
    """The tree navigates and the strip loads: a folder with no images of its own has
    nothing to load, so nothing is hashed."""
    (tmp_path / "library" / "roll_a").mkdir()

    panel.library_tree.folders_activated.emit([str(tmp_path / "library")])

    panel.controller.open_library_folder.assert_not_called()
    panel.controller.open_library_folders.assert_not_called()


def test_opening_a_roll_from_the_tree_reaches_the_film_strip(panel, tmp_path, monkeypatch):
    roll = tmp_path / "library" / "roll_a"
    roll.mkdir()
    (roll / "a1.NEF").write_bytes(b"1")
    monkeypatch.setattr(panel.file_browser, "_confirm_load", lambda count, label: True)

    panel.library_tree.folders_activated.emit([str(roll)])

    panel.controller.open_library_folder.assert_called_once_with(str(roll), add_to_session=False)


def test_the_library_button_reveals_the_primary_folder(panel, tmp_path):
    panel.file_browser.library_requested.emit(True)

    assert panel.library_tree.tree.currentItem().text(0) == "library"


def test_sorting_the_sheet_sorts_the_tree(panel):
    panel.file_browser._apply_sort_direction(True)

    assert panel.library_tree._sort_descending is True


def test_changing_roots_drops_the_cached_walk(panel):
    panel.library_tree.roots_changed.emit()

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
    assert repo.get_global_setting("library_section_expanded") is False
    assert repo.get_global_setting("frames_section_expanded") is False


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
