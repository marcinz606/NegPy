from dataclasses import replace
from unittest.mock import MagicMock, patch

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.process import ProcessSidebar
from negpy.features.process.models import ProcessMode


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    return controller, ProcessSidebar(controller)


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


def test_mode_buttons_track_config_and_switch_mode(qapp):
    controller, sidebar = _sidebar()
    sidebar.sync_ui()

    color_btn, bw_btn, slide_btn = sidebar.mode_btns
    assert color_btn.isChecked()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    sidebar.sync_ui()
    assert slide_btn.isChecked()
    assert not color_btn.isChecked() and not bw_btn.isChecked()

    bw_btn.click()
    controller.set_process_mode.assert_called_once_with(ProcessMode.BW)


def test_lock_bounds_sits_in_the_analysis_row_and_hides_on_the_transparency_transfer(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    sidebar.sync_ui()
    assert sidebar.lock_bounds_btn.isHidden()


def test_positive_is_slide_only(qapp):
    """Positive is a Slide-only fact, so it is hidden in the negative modes and always
    live on Slide."""
    controller, sidebar = _sidebar()
    sidebar.sync_ui()
    assert sidebar.positive_source_btn.isHidden()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    sidebar.sync_ui()
    assert not sidebar.positive_source_btn.isHidden()
    assert sidebar.positive_source_btn.isEnabled()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
    sidebar.sync_ui()
    assert sidebar.positive_source_btn.isHidden()


def test_positive_lives_in_mode_bar_not_the_normalization_body(qapp):
    """Whether the source is already a finished positive is a fact about the file, not
    a Normalization setting -- it lives beside Film Mode in mode_bar, the "Film Mode"
    card's own content, not inside Normalization's."""
    _, sidebar = _sidebar()
    assert sidebar.positive_source_btn.parentWidget() is sidebar.mode_bar


def test_positive_toggle_reaches_the_controller(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    sidebar.sync_ui()

    sidebar.positive_source_btn.setChecked(True)
    controller.set_positive_source.assert_called_once_with(True)


def _row_index_containing(layout, widget) -> int:
    """Index within *layout* of the (possibly nested) row that directly holds *widget*."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is widget:
            return i
        row = item.layout()
        if row is not None and any(row.itemAt(j).widget() is widget for j in range(row.count())):
            return i
    raise AssertionError(f"{widget} not found in layout")


def test_the_analysis_bar_holds_everything_that_meters_this_frame(qapp):
    """One ANALYSIS block above the Roll Baseline picker: what the meters read, then
    how the measurement is clipped, then the nudge on what came out."""
    _, sidebar = _sidebar()
    col = sidebar.analysis_bar.layout()

    order = [
        _row_index_containing(col, w)
        for w in (
            sidebar.analysis_buffer_slider,
            sidebar.analysis_region_btn,
            sidebar.luma_range_clip_slider,
            sidebar.ch_global_btn,
            sidebar.white_point_slider,
        )
    ]
    assert order == sorted(order)
    assert col.itemAt(0).widget().text() == "ANALYSIS"
    assert _row_index_containing(col, sidebar.clear_analysis_region_btn) == _row_index_containing(col, sidebar.analysis_region_btn)


def test_the_tonal_range_header_opens_the_clip_and_point_controls(qapp):
    """Marks off what shapes the measurement from what the measurement reads."""
    _, sidebar = _sidebar()
    col = sidebar.analysis_bar.layout()

    header_i = col.indexOf(sidebar.tonal_range_header)
    assert header_i == _row_index_containing(col, sidebar.luma_range_clip_slider) - 1
    assert header_i > _row_index_containing(col, sidebar.analysis_region_btn)


def test_the_region_buttons_carry_their_names(qapp):
    _, sidebar = _sidebar()
    assert sidebar.analysis_region_btn.text().strip() == "Draw Region"
    assert sidebar.clear_analysis_region_btn.text().strip() == "Clear Region"


def test_average_toggles_ride_the_baseline_bar_not_the_analysis(qapp):
    """Use Luma/Color Average is about the roll baseline, so it travels with the picker
    in baseline_bar, apart from the metering controls."""
    _, sidebar = _sidebar()
    bar = sidebar.baseline_bar.layout()
    assert _row_index_containing(bar, sidebar.use_luma_avg_btn) == _row_index_containing(bar, sidebar.use_color_avg_btn)
    for widget in (sidebar.luma_range_clip_slider, sidebar.white_point_slider):
        assert not sidebar.baseline_bar.isAncestorOf(widget)


def test_lock_bounds_shares_the_region_row(qapp):
    _, sidebar = _sidebar()
    col = sidebar.analysis_bar.layout()
    assert _row_index_containing(col, sidebar.lock_bounds_btn) == _row_index_containing(col, sidebar.analysis_region_btn)


def test_the_point_header_opens_the_channel_row_and_its_sliders(qapp):
    """The R/G/B selector scopes White/Black Point only, so it sits under their own
    header, after the clip sliders it does not reach."""
    _, sidebar = _sidebar()
    col = sidebar.analysis_bar.layout()
    header_i = col.indexOf(sidebar.point_header)
    assert header_i > _row_index_containing(col, sidebar.color_range_clip_slider)
    assert _row_index_containing(col, sidebar.ch_r_btn) == header_i + 1
    assert _row_index_containing(col, sidebar.white_point_slider) == header_i + 2


def test_average_toggles_sync_from_config(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, use_luma_average=True, use_color_average=False))
    sidebar.sync_ui()
    assert sidebar.use_luma_avg_btn.isChecked()
    assert not sidebar.use_color_avg_btn.isChecked()


def test_average_toggles_hide_on_the_transparency_transfer(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    sidebar.sync_ui()
    assert sidebar.use_luma_avg_btn.isHidden()
    assert sidebar.use_color_avg_btn.isHidden()


def test_use_luma_average_toggle_reaches_the_controller(qapp):
    """Goes through set_roll_default on the Roll Analysis card -- flipping it locks that
    card to this frame, and drops roll_name since a single picked baseline no longer
    applies."""
    controller, sidebar = _sidebar()
    sidebar.use_luma_avg_btn.setChecked(True)
    args, kwargs = controller.set_roll_default.call_args
    assert args[0] == "baseline"
    assert kwargs["use_luma_average"] is True
    assert kwargs["roll_name"] is None


def test_use_color_average_toggle_reaches_the_controller(qapp):
    controller, sidebar = _sidebar()
    sidebar.use_color_avg_btn.setChecked(True)
    args, kwargs = controller.set_roll_default.call_args
    assert args[0] == "baseline"
    assert kwargs["use_color_average"] is True
    assert kwargs["roll_name"] is None


def test_white_black_point_retarget_and_sync(qapp):
    """The Tonal Range block retargets through its own Global/R/G/B selector."""
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
    assert sidebar.ch_r_btn.edited_dot.isVisibleTo(sidebar.ch_r_btn)

    sidebar.ch_global_btn.setChecked(True)
    assert abs(sidebar.white_point_slider.value() - 0.1) < 1e-9
    assert sidebar.white_point_slider.label.text() == "White Point"


def test_white_black_point_stay_visible_on_the_transparency_transfer(qapp):
    """They deviate the transfer path's fixed window the same way they deviate a
    measured one (TransparencyBaseProcessor), unlike the metering
    controls above, which have nothing to act on there."""
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    sidebar.sync_ui()

    assert not sidebar.white_point_slider.isHidden()
    assert not sidebar.black_point_slider.isHidden()
    assert sidebar.analysis_buffer_slider.isHidden()


def test_white_black_point_disabled_when_bounds_are_locked(qapp):
    """Trims shift the frozen bounds themselves, so nudging stops once locked."""
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, lock_bounds=True))
    sidebar.sync_ui()

    assert not sidebar.white_point_slider.isEnabled()
    assert not sidebar.black_point_slider.isEnabled()


def test_white_black_point_ignore_the_lock_on_the_transparency_transfer(qapp):
    """The transfer path's window is fixed, never measured, so a Lock Bounds left on
    from another frame or mode has nothing there to freeze."""
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, lock_bounds=True))
    sidebar.sync_ui()

    assert sidebar.white_point_slider.isEnabled()
    assert sidebar.black_point_slider.isEnabled()


def test_white_black_point_write_the_normalization_roll_card(qapp):
    controller, sidebar = _sidebar()

    sidebar._on_white_point_changed(0.15, persist=True)
    sidebar.ch_g_btn.setChecked(True)
    sidebar._on_black_point_changed(-0.05, persist=True)

    calls = [(c.args[0], c.kwargs) for c in controller.set_roll_default.call_args_list]
    assert ("process", {"persist": True, "readback_metrics": True, "white_point_offset": 0.15}) in calls
    assert ("process", {"persist": True, "readback_metrics": True, "black_point_trim_green": -0.05}) in calls


def test_white_black_point_are_normalization_roll_defaults():
    from negpy.services.assets import rolls

    fields = rolls.card_fields("process")
    for layer in ("red", "green", "blue"):
        assert f"white_point_trim_{layer}" in fields and f"black_point_trim_{layer}" in fields
    assert "white_point_offset" in fields and "black_point_offset" in fields
    assert "use_luma_average" not in fields
    assert rolls.card_fields("baseline") == ("use_luma_average", "use_color_average", "use_cast_average")


def test_reanalyze_frame_clears_local_bounds_and_persists(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, local_floors=(0.1, 0.1, 0.1), local_ceils=(0.9, 0.9, 0.9)))
    sidebar.sync_ui()

    sidebar.reanalyze_frame_btn.click()

    new_cfg = controller.apply_config.call_args.args[0]
    assert not new_cfg.process.is_local_initialized
    assert controller.apply_config.call_args.kwargs == {"persist": True}


def test_reanalyze_frame_is_disabled_when_nothing_would_be_measured(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    sidebar.sync_ui()
    assert sidebar.reanalyze_frame_btn.isEnabled()

    controller.state.config = replace(cfg, process=replace(cfg.process, lock_bounds=True))
    sidebar.sync_ui()
    assert not sidebar.reanalyze_frame_btn.isEnabled()

    controller.state.config = replace(cfg, process=replace(cfg.process, use_luma_average=True, use_color_average=True))
    sidebar.sync_ui()
    assert not sidebar.reanalyze_frame_btn.isEnabled()


def test_baseline_hint_names_where_the_bounds_came_from(qapp):
    from negpy.services.assets import rolls

    controller, sidebar = _sidebar()
    cfg = controller.state.config
    riding = dict(use_luma_average=True, locked_floors=(0.1, 0.1, 0.1), locked_ceils=(0.9, 0.9, 0.9))

    sidebar.sync_ui()
    assert not sidebar.baseline_source_hint.isVisibleTo(sidebar.baseline_bar)

    controller.state.config = replace(cfg, process=replace(cfg.process, baseline_source="frame:f003.tif", **riding))
    sidebar.sync_ui()
    assert sidebar.baseline_source_hint.isVisibleTo(sidebar.baseline_bar)
    assert sidebar.baseline_source_hint.text() == "Baseline: Frame “f003.tif”"

    with patch.object(rolls, "roll_for_id", return_value={"name": "Tri-X"}):
        controller.state.config = replace(cfg, process=replace(cfg.process, baseline_source="roll:r1", **riding))
        sidebar.sync_ui()
    assert sidebar.baseline_source_hint.text() == "Baseline: Roll “Tri-X”"

    controller.state.config = replace(cfg, process=replace(cfg.process, use_color_average=True))
    sidebar.sync_ui()
    assert sidebar.baseline_source_hint.text().startswith("No baseline yet")
