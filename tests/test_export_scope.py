"""The Export split button dispatches each scope to the matching controller call."""

from unittest.mock import MagicMock

from negpy.desktop.view.sidebar.export import ExportSidebar


def _sidebar(scope: str) -> MagicMock:
    sidebar = MagicMock()
    sidebar._export_scope = scope
    sidebar.state.linear_output = False
    sidebar.controller = MagicMock()
    return sidebar


def test_scope_all_exports_every_visible_frame():
    sidebar = _sidebar("all")
    ExportSidebar.run_export(sidebar, "all")
    sidebar.controller.request_batch_export.assert_called_once_with()
    sidebar.controller.request_export.assert_not_called()


def test_scope_selected_exports_selection():
    sidebar = _sidebar("selected")
    ExportSidebar.run_export(sidebar, "selected")
    sidebar.controller.request_export_selected.assert_called_once_with()


def test_scope_current_exports_current_frame():
    sidebar = _sidebar("current")
    ExportSidebar.run_export(sidebar, "current")
    sidebar.controller.request_export.assert_called_once_with()


def test_the_button_exports_its_own_sticky_scope():
    sidebar = _sidebar("selected")
    ExportSidebar._on_export_clicked(sidebar)
    sidebar.export_chosen_scope.assert_called_once_with()


def test_the_export_key_follows_the_buttons_choice():
    """Ctrl+E (⌘E) exports the frame, the selection or the roll, whichever the button is set to."""
    for scope in ("current", "selected", "all"):
        sidebar = _sidebar(scope)
        ExportSidebar.export_chosen_scope(sidebar)
        sidebar.run_export.assert_called_once_with(scope)


def test_every_scope_names_the_export_key_in_its_tooltip():
    assert set(ExportSidebar._EXPORT_SCOPE_SHORTCUTS) == set(ExportSidebar._EXPORT_SCOPES)
    assert set(ExportSidebar._EXPORT_SCOPE_SHORTCUTS.values()) == {"export"}


def test_run_export_takes_its_scope_from_the_caller_not_the_button():
    """The menu bar's Export > Roll exports the roll while the button stays on Frame."""
    sidebar = _sidebar("current")
    ExportSidebar.run_export(sidebar, "all")
    sidebar.controller.request_batch_export.assert_called_once_with()
    sidebar.controller.request_export.assert_not_called()


def test_run_export_rejects_an_unknown_scope():
    import pytest

    with pytest.raises(ValueError):
        ExportSidebar.run_export(_sidebar("current"), "everything")


def test_retired_scope_keys_map_to_a_live_scope():
    """A persisted "all_current"/"all_saved" must not silently reset to the
    single-frame scope now that the two collapsed into one."""
    assert ExportSidebar._RETIRED_EXPORT_SCOPES
    for legacy, current in ExportSidebar._RETIRED_EXPORT_SCOPES.items():
        assert legacy not in ExportSidebar._EXPORT_SCOPES
        assert current in ExportSidebar._EXPORT_SCOPES
