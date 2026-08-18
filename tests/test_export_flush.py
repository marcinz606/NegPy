"""Export UI flush is invoked before export entry points read Destination/filename."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.domain.models import ExportPresetOutputMode, WorkspaceConfig, ExportConfig


def test_flush_export_ui_invokes_callback():
    flush = MagicMock()
    controller = MagicMock()
    controller.flush_export_settings = flush
    AppController._flush_export_ui(controller)
    flush.assert_called_once()


def test_flush_export_ui_noop_without_callback():
    controller = MagicMock()
    controller.flush_export_settings = None
    AppController._flush_export_ui(controller)  # must not raise


def test_request_export_calls_flush_export_ui():
    flush_ui = MagicMock()
    controller = MagicMock()
    controller._flush_export_ui = flush_ui
    controller._batch_busy = MagicMock(return_value=False)
    controller.state = AppState()
    controller.state.current_file_path = "/tmp/shot.tif"
    controller.state.current_file_hash = "abc"
    controller.state.source_exif = {}
    controller.state.flat_output = False
    controller.state.gpu_enabled = False
    controller.state.workspace_color_space = "Adobe RGB"
    controller.state.icc_output_path = None
    controller.state.config = replace(
        WorkspaceConfig(),
        export=ExportConfig(
            output_mode=ExportPresetOutputMode.ABSOLUTE,
            export_path="C:/Exports",
        ),
    )
    controller._ensure_valid_export_path = MagicMock(return_value="C:/Exports")
    controller.effective_input_icc = MagicMock(return_value=None)
    controller._diptych_task = lambda info: (info, None)
    controller._run_export_tasks = MagicMock()

    AppController.request_export(controller)

    flush_ui.assert_called_once()
    controller._run_export_tasks.assert_called_once()


def test_flush_export_settings_stops_timer_and_persists():
    from negpy.desktop.view.sidebar.export import ExportSidebar

    sidebar = MagicMock()
    sidebar.update_timer = MagicMock()
    sidebar._persist_all_export_settings = MagicMock()
    ExportSidebar._flush_export_settings(sidebar)
    sidebar.update_timer.stop.assert_called_once()
    sidebar._persist_all_export_settings.assert_called_once()
