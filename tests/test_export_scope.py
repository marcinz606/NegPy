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
    ExportSidebar._on_export_clicked(sidebar)
    sidebar.controller.request_batch_export.assert_called_once_with()
    sidebar.controller.request_export.assert_not_called()


def test_scope_selected_exports_selection():
    sidebar = _sidebar("selected")
    ExportSidebar._on_export_clicked(sidebar)
    sidebar.controller.request_export_selected.assert_called_once_with()


def test_scope_current_exports_current_frame():
    sidebar = _sidebar("current")
    ExportSidebar._on_export_clicked(sidebar)
    sidebar.controller.request_export.assert_called_once_with()


def test_retired_scope_keys_map_to_a_live_scope():
    """A persisted "all_current"/"all_saved" must not silently reset to the
    single-frame scope now that the two collapsed into one."""
    assert ExportSidebar._RETIRED_EXPORT_SCOPES
    for legacy, current in ExportSidebar._RETIRED_EXPORT_SCOPES.items():
        assert legacy not in ExportSidebar._EXPORT_SCOPES
        assert current in ExportSidebar._EXPORT_SCOPES
