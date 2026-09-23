"""Raw Decode's Highlight Recovery combo: Slide only, greyed without a camera matrix."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.demosaic import DemosaicSidebar
from negpy.features.process.models import ProcessMode


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    return controller, DemosaicSidebar(controller)


def test_highlight_reconstruction_hidden_off_e6(qapp):
    controller, sidebar = _sidebar()
    sidebar.sync_ui()
    assert sidebar.highlight_combo.isHidden()
    assert sidebar.highlight_label.isHidden()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    sidebar.sync_ui()
    assert not sidebar.highlight_combo.isHidden()
    assert not sidebar.highlight_label.isHidden()


def test_highlight_reconstruction_buckets_an_unnamed_level_under_reconstruct(qapp):
    """Any stored level besides 0 or 2 is some Reconstruct level (3-9); the combo
    must not end up on the wrong entry."""
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, highlight_reconstruction=7))
    sidebar.sync_ui()
    assert sidebar.highlight_combo.currentIndex() == 2


def test_highlight_reconstruction_buckets_an_invalid_level_under_off(qapp):
    """A level effective_highlight_reconstruction itself would reject (a hand-edited
    sidecar) must not show Reconstruct selected while the decode actually clips."""
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, highlight_reconstruction=1))
    sidebar.sync_ui()
    assert sidebar.highlight_combo.currentIndex() == 0


def test_highlight_reconstruction_click_reaches_the_controller(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    controller.state.preview_cam_xyz = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    sidebar.sync_ui()

    sidebar.highlight_combo.setCurrentIndex(1)
    controller.set_roll_default.assert_called_once_with("demosaic", highlight_reconstruction=2)


def test_highlight_reconstruction_greyed_without_a_camera_matrix(qapp):
    """A scanner TIFF or JPEG carries no camera matrix (see preview_cam_xyz), and no
    sensor CFA data for reconstruction to recover from — the combo stays visible (still
    a Transparency source) but disabled, not silently inert with no explanation."""
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))

    controller.state.preview_cam_xyz = None
    sidebar.sync_ui()
    assert not sidebar.highlight_combo.isHidden()
    assert not sidebar.highlight_combo.isEnabled()

    controller.state.preview_cam_xyz = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    sidebar.sync_ui()
    assert sidebar.highlight_combo.isEnabled()


def test_highlight_reconstruction_is_a_raw_decode_roll_default():
    from negpy.services.assets import rolls

    assert "highlight_reconstruction" in rolls.card_fields("demosaic")
    assert "highlight_reconstruction" not in rolls.card_fields("process")
