from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.process import ProcessSidebar


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    return controller, ProcessSidebar(controller)


def test_channel_selector_retargets_and_syncs(qapp):
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        process=replace(
            cfg.process,
            white_point_offset=0.1,
            black_point_offset=-0.05,
            white_point_trim_red=0.08,
            black_point_trim_red=-0.02,
        ),
    )
    sidebar.sync_ui()

    assert sidebar._wp_field() == "white_point_offset"
    assert sidebar._bp_field() == "black_point_offset"
    assert abs(sidebar.white_point_slider.value() - 0.1) < 1e-9
    assert abs(sidebar.black_point_slider.value() - (-0.05)) < 1e-9

    sidebar.ch_r_btn.setChecked(True)

    assert sidebar._wp_field() == "white_point_trim_red"
    assert sidebar._bp_field() == "black_point_trim_red"
    assert abs(sidebar.white_point_slider.value() - 0.08) < 1e-9
    assert abs(sidebar.black_point_slider.value() - (-0.02)) < 1e-9
    assert sidebar.white_point_slider.label.text() == "White Point R"
    assert sidebar.black_point_slider.label.text() == "Black Point R"

    sidebar.ch_global_btn.setChecked(True)
    assert abs(sidebar.white_point_slider.value() - 0.1) < 1e-9
    assert sidebar.white_point_slider.label.text() == "White Point"


def test_channel_selector_hidden_in_bw(qapp):
    controller, sidebar = _sidebar()

    sidebar.sync_ui()
    assert not sidebar.ch_r_btn.isHidden()
    sidebar.ch_r_btn.setChecked(True)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode="B&W"))
    sidebar.sync_ui()
    for w in (sidebar.ch_global_btn, sidebar.ch_r_btn, sidebar.ch_g_btn, sidebar.ch_b_btn):
        assert w.isHidden()
    assert sidebar._channel_index() == 0
    assert sidebar._wp_field() == "white_point_offset"


def test_lock_bounds_disables_wp_bp_and_selector(qapp):
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, lock_bounds=True))
    sidebar.sync_ui()
    for w in (sidebar.white_point_slider, sidebar.black_point_slider, sidebar.ch_global_btn, sidebar.ch_r_btn):
        assert not w.isEnabled()


def test_analysis_region_dot_reflects_committed_region_not_just_tool_state(qapp):
    """Confirming a freehand region closes the draw tool (button unchecks), so the
    dot is the only remaining cue that a region is active and overriding the
    Analysis Buffer slider — it must track analysis_rect, not active_tool."""
    controller, sidebar = _sidebar()

    sidebar.sync_ui()
    assert not sidebar.analysis_region_btn.edited_dot.isVisibleTo(sidebar.analysis_region_btn)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, analysis_rect=(0.1, 0.1, 0.9, 0.9)))
    sidebar.sync_ui()

    assert sidebar.analysis_region_btn.edited_dot.isVisibleTo(sidebar.analysis_region_btn)
    assert not sidebar.analysis_region_btn.isChecked()  # tool itself is closed
    assert not sidebar.analysis_buffer_slider.isEnabled()

    controller.state.config = replace(cfg, process=replace(cfg.process, analysis_rect=None))
    sidebar.sync_ui()
    assert not sidebar.analysis_region_btn.edited_dot.isVisibleTo(sidebar.analysis_region_btn)
