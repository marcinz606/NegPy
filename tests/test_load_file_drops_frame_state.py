"""State tied to the frame being left does not carry over to the next one."""

from unittest.mock import MagicMock, patch

from negpy.desktop.controller import AppController
from negpy.domain.models import WorkspaceConfig


def test_load_file_drops_marked_keystone_lines_and_the_flat_peek():
    ctrl = MagicMock()
    ctrl._prefetch_gen = 0
    ctrl._keystone_lines = {"left": ((0.1, 0.1), (0.1, 0.9)), "right": ((0.9, 0.1), (0.9, 0.9))}
    ctrl.state.flat_peek = True
    ctrl.state.config = WorkspaceConfig()
    ctrl._render_memo.get.return_value = None

    AppController.load_file(ctrl, "/p/b.tif")

    assert ctrl._keystone_lines == {}
    ctrl.keystone_lines_cleared.emit.assert_called_once()
    assert ctrl.state.flat_peek is False
    ctrl.flat_peek_changed.emit.assert_called_once_with(False)


def test_the_diptych_view_refuses_canvas_tools_and_edits():
    """Both halves are on screen, each with its own edit, so a canvas point names neither."""
    from negpy.desktop.session import AppState, ToolMode

    ctrl = MagicMock()
    ctrl.state = AppState()
    ctrl.active_diptych.return_value = ({"hash": "h"}, (WorkspaceConfig(), WorkspaceConfig()))
    ctrl._diptych_blocks_canvas = lambda: AppController._diptych_blocks_canvas(ctrl)

    AppController.set_active_tool(ctrl, ToolMode.CLONE)
    AppController.handle_canvas_clicked(ctrl, 0.5, 0.5)
    AppController.handle_local_mask_created(ctrl, "polygon", [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])

    assert ctrl.state.active_tool == ToolMode.NONE
    ctrl.session.update_config.assert_not_called()
    ctrl._handle_wb_pick.assert_not_called()


def test_load_file_drops_the_left_frames_geometry_grid():
    import numpy as np

    from negpy.desktop.session import AppState

    ctrl = MagicMock()
    ctrl._prefetch_gen = 0
    ctrl.state = AppState()
    ctrl.state.last_metrics["uv_grid"] = np.zeros((2, 2, 2), np.float32)
    ctrl.state.last_metrics["active_roi"] = (0, 2, 0, 2)
    ctrl._render_memo.get.return_value = None
    ctrl._retain_displayed_texture.return_value = None

    AppController.load_file(ctrl, "/p/b.tif")

    assert "uv_grid" not in ctrl.state.last_metrics
    assert "active_roi" not in ctrl.state.last_metrics


def test_undoing_a_decode_level_field_re_decodes_the_source():
    """Undo restores Linear RAW without apply_config's check; the render must not run on the old decode."""
    from dataclasses import replace

    from negpy.desktop.session import AppState
    from negpy.services.rendering.lens import lens_decode_token, metadata_lens_corrections
    from negpy.services.rendering.source_identity import source_token

    ctrl = MagicMock()
    ctrl.state = AppState()
    ctrl.state.current_file_path = "/p/a.dng"
    decoded = WorkspaceConfig()
    ctrl._decoded_source_token = source_token(decoded)
    ctrl._foreground_preview_generation = None
    ctrl.state.config = replace(decoded, process=replace(decoded.process, linear_raw=not decoded.process.linear_raw))
    ctrl.state.preview_lens_token = lens_decode_token(metadata_lens_corrections(ctrl.state.config), ctrl.state.config.flatfield)

    AppController.request_render(ctrl)

    ctrl.load_file.assert_called_once_with("/p/a.dng", preserve_zoom=True)


def test_a_crop_offset_drag_drops_the_cached_bounds_on_release():
    """The drag ticks already wrote the value, so the release must not compare against them."""
    from dataclasses import replace

    from negpy.desktop.session import AppState

    ctrl = MagicMock()
    ctrl.state = AppState()
    cfg = WorkspaceConfig()
    ctrl.state.config = replace(cfg, process=replace(cfg.process, local_floors=(0.1, 0.1, 0.1), local_ceils=(0.9, 0.9, 0.9)))
    ctrl._previewed_meter_cards = set()
    ctrl.refuse_crop_edit.return_value = False
    ctrl._with_card_values = AppController._with_card_values
    ctrl.apply_config.side_effect = lambda c, **k: setattr(ctrl.state, "config", c)

    AppController.set_roll_default(ctrl, "autocrop", persist=False, readback_metrics=False, autocrop_offset=12)
    AppController.set_roll_default(ctrl, "autocrop", autocrop_offset=12)

    assert ctrl.state.config.geometry.autocrop_offset == 12
    assert ctrl.state.config.process.local_floors == (0.0, 0.0, 0.0)


def test_exit_cancels_long_batches_before_joining_their_threads():
    """quit() takes effect between slots; a batch runs inside one, so it must be told to stop."""
    ctrl = MagicMock()
    ctrl._cleaned_up = False
    order: list[str] = []
    for name in ("export_worker", "stitch_worker", "hdr_worker", "frame_merge_worker", "embedding_worker", "norm_worker"):
        getattr(ctrl, name).cancel.side_effect = lambda n=name: order.append(f"cancel {n}")
    for name in ("export_thread", "thumb_thread", "norm_thread"):
        getattr(ctrl, name).wait.side_effect = lambda *a, n=name: order.append(f"join {n}")

    with patch("negpy.desktop.controller.GPUDevice"):
        AppController.cleanup(ctrl)

    assert order.index("cancel export_worker") < order.index("join export_thread")
    assert order.index("cancel frame_merge_worker") < order.index("join export_thread")
    assert order.index("cancel embedding_worker") < order.index("join thumb_thread")
    assert order.index("cancel norm_worker") < order.index("join norm_thread")


def test_deleting_a_mask_keeps_the_last_mask_hidden():
    from dataclasses import replace

    from negpy.desktop.session import AppState
    from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask

    ctrl = MagicMock()
    ctrl.state = AppState()
    ctrl.state.current_file_hash = "h"
    tri = ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))
    cfg = WorkspaceConfig()
    ctrl.state.config = replace(cfg, local=LocalAdjustmentsConfig(masks=(LocalMask(vertices=tri),) * 3))
    ctrl.state.local_hidden_masks = {2}
    ctrl.session.update_config.side_effect = lambda c, **k: setattr(ctrl.state, "config", c)

    with patch("negpy.desktop.view.confirm.confirm_delete_mask", return_value=True):
        AppController.delete_local_mask(ctrl, 0)

    assert ctrl.state.local_hidden_masks == {1}
