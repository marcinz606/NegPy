"""Sync Bounds: the active frame's metering bounds and nothing else. The full Apply
picker reaches the same two axes, but only after every edited row has been unticked."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QDialog, QLabel

from negpy.desktop.session import AppState
from negpy.desktop.view.widgets.granular_settings_dialog import SyncBoundsDialog, open_sync_bounds_dialog
from negpy.domain.models import WorkspaceConfig

_MODULE = "negpy.desktop.view.widgets.granular_settings_dialog"


def _session(metered: bool = True, files: int = 3, selected=(0, 1)) -> MagicMock:
    session = MagicMock()
    session.state = AppState()
    state = session.state
    state.uploaded_files = [{"path": f"/tmp/IMG_000{i}.cr2", "hash": f"h{i}"} for i in range(files)]
    state.selected_file_idx = 0
    state.selected_indices = list(selected)
    cfg = WorkspaceConfig()
    if metered:
        cfg = replace(cfg, process=replace(cfg.process, local_floors=(0.1, 0.2, 0.3), local_ceils=(0.9, 0.8, 0.7)))
    state.config = cfg
    session.asset_model.visible_actual_indices.return_value = list(range(files))
    return session


def test_both_axes_are_ticked_to_begin_with(qapp):
    dlg = SyncBoundsDialog(None, (0.1, 0.2, 0.3), (0.9, 0.8, 0.7), "IMG_0000.cr2", 2, 2)
    assert dlg.bounds_flags() == (True, True)
    assert dlg.apply_btn.isEnabled()


def test_the_source_bounds_are_shown_so_the_frame_is_identifiable(qapp):
    dlg = SyncBoundsDialog(None, (0.1, 0.2, 0.3), (0.9, 0.8, 0.7), "IMG_0000.cr2", 2, 2)
    labels = [w.text() for w in dlg.findChildren(QLabel)]
    assert any("IMG_0000.cr2" in t for t in labels)
    assert any("0.1 / 0.2 / 0.3 → 0.9 / 0.8 / 0.7" == t for t in labels)


def test_unticking_both_axes_leaves_nothing_to_apply(qapp):
    dlg = SyncBoundsDialog(None, (0.1, 0.2, 0.3), (0.9, 0.8, 0.7), "IMG_0000.cr2", 2, 2)
    dlg.luma_box.setChecked(False)
    assert dlg.apply_btn.isEnabled()
    dlg.color_box.setChecked(False)
    assert not dlg.apply_btn.isEnabled()


def test_syncing_carries_the_axes_and_scope_and_no_setting_rows(qapp):
    session = _session()

    with patch.object(SyncBoundsDialog, "exec", lambda dlg: QDialog.DialogCode.Accepted):
        open_sync_bounds_dialog(None, session)

    rows, flags, scope = session.sync_selected_settings.call_args.args
    assert rows == []
    assert flags == (True, True)
    assert scope == "selection"


def test_one_axis_travels_alone(qapp):
    session = _session()

    def run(dlg):
        dlg.color_box.setChecked(False)
        return QDialog.DialogCode.Accepted

    with patch.object(SyncBoundsDialog, "exec", run):
        open_sync_bounds_dialog(None, session)

    assert session.sync_selected_settings.call_args.args[1] == (True, False)


def test_cancelling_syncs_nothing(qapp):
    session = _session()

    with patch.object(SyncBoundsDialog, "exec", lambda dlg: QDialog.DialogCode.Rejected):
        open_sync_bounds_dialog(None, session)

    session.sync_selected_settings.assert_not_called()


def test_an_unrendered_frame_has_no_bounds_to_give(qapp):
    session = _session(metered=False)

    with patch(f"{_MODULE}.SyncBoundsDialog") as dlg_cls:
        open_sync_bounds_dialog(None, session)

    dlg_cls.assert_not_called()
    session.sync_selected_settings.assert_not_called()
    assert "Render" in session.settings_synced.emit.call_args.args[0]


def test_a_lone_frame_has_nothing_to_sync_to(qapp):
    session = _session(files=1, selected=(0,))

    with patch(f"{_MODULE}.SyncBoundsDialog") as dlg_cls:
        open_sync_bounds_dialog(None, session)

    dlg_cls.assert_not_called()
    session.sync_selected_settings.assert_not_called()


def test_a_frame_on_the_roll_baseline_gives_that_baseline_on(qapp):
    """_source_effective_bounds' rule: what the frame renders with, not what it metered."""
    session = _session()
    cfg = session.state.config
    session.state.config = replace(
        cfg,
        process=replace(
            cfg.process,
            use_luma_average=True,
            locked_floors=(0.4, 0.5, 0.6),
            locked_ceils=(0.7, 0.6, 0.5),
        ),
    )
    seen = {}

    def run(dlg):
        seen["text"] = [w.text() for w in dlg.findChildren(QLabel)]
        return QDialog.DialogCode.Rejected

    with patch.object(SyncBoundsDialog, "exec", run):
        open_sync_bounds_dialog(None, session)

    assert any("0.4 / 0.5 / 0.6 → 0.7 / 0.6 / 0.5" == t for t in seen["text"])
