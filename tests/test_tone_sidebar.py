from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.tone import ToneSidebar
from negpy.features.process.models import ProcessMode


def _combo_items(combo):
    return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]


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


def test_tone_reset_covers_dye_separation():
    """The section header's reset button resets the fields listed in _TONE_FIELDS, so
    every control the panel shows has to be in it — a renamed field that falls out of
    the list leaves a visible slider its own reset can't clear."""
    from negpy.desktop.view.sidebar.controls_panel import _TONE_FIELDS

    for field in (
        "dye_separation",
        "dye_separation_trim_red",
        "dye_separation_trim_green",
        "dye_separation_trim_blue",
        "separation_damping",
    ):
        assert field in _TONE_FIELDS


def test_separation_damping_locked_without_a_separation_push(qapp):
    """It redistributes Dye Separation's push and has no effect of its own, so at
    separation 1.0 it renders nothing — a live slider there reads as broken."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    sidebar.sync_ui()
    assert not sidebar.separation_damping_slider.isEnabled()

    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, dye_separation=1.3))
    sidebar.sync_ui()
    assert sidebar.separation_damping_slider.isEnabled()


def test_paper_combo_rebuilt_only_when_entries_change(qapp):
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    sidebar.sync_ui()
    items = _combo_items(sidebar.paper_combo)
    assert items

    clears = []
    orig_clear = sidebar.paper_combo.clear
    sidebar.paper_combo.clear = lambda: (clears.append(1), orig_clear())[1]

    sidebar.sync_ui()  # unchanged process mode -> no rebuild
    assert clears == []
    assert _combo_items(sidebar.paper_combo) == items


def test_channel_selector_retargets_and_syncs(qapp):
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        exposure=replace(
            cfg.exposure,
            grade_trim_red=15.0,
            toe_trim_red=0.4,
            shoulder_trim_red=-0.2,
            midtone_gamma_trim_red=0.15,
            toe_width_trim_red=1.2,
            shoulder_width_trim_red=-0.6,
            paper_black=True,
            midtone_gamma=0.25,
            shadow_density=-0.45,
            highlight_density=0.2,
            shadow_grade=-12.0,
            highlight_grade=8.0,
            shadow_grade_trim_red=5.0,
            highlight_grade_trim_red=-3.0,
            dye_separation=1.3,
            dye_separation_trim_red=0.25,
        ),
    )
    sidebar.sync_ui()

    # Global page: shared curve values, ISO-R grade slider shown.
    assert sidebar._curve_field("toe") == "toe"
    assert not sidebar.grade_slider.isHidden()
    assert sidebar.grade_trim_slider.isHidden()
    assert not sidebar.toe_w_slider.isHidden()
    assert sidebar.toe_w_trim_slider.isHidden()
    assert not sidebar.dye_separation_slider.isHidden()
    assert sidebar.dye_separation_trim_slider.isHidden()
    assert abs(sidebar.dye_separation_slider.value() - 1.3) < 1e-9
    assert sidebar.paper_black_btn.isChecked()
    assert abs(sidebar.midtone_gamma_slider.value() - 0.25) < 1e-9
    assert abs(sidebar.shadow_density_slider.value() - (-0.45)) < 1e-9
    assert abs(sidebar.highlight_density_slider.value() - 0.2) < 1e-9
    assert abs(sidebar.shadow_grade_slider.value() - (-12.0)) < 1e-9
    assert abs(sidebar.highlight_grade_slider.value() - 8.0) < 1e-9
    assert sidebar.shadow_density_slider in sidebar._global_only
    assert sidebar.highlight_density_slider in sidebar._global_only
    # Split grade follows the channel selector (per-layer trims), not global-only.
    assert sidebar.shadow_grade_slider not in sidebar._global_only
    assert sidebar.highlight_grade_slider not in sidebar._global_only
    # Long tooltips must be rich text so Qt word-wraps them; tooltips that carry
    # their own markup (shortcut chips) must not get double-escaped. Shortcut-bearing
    # sliders get their tooltips from ControlsPanel.apply_shortcut_tooltips (single
    # source), so only locally-tooltipped widgets are asserted here.
    assert sidebar.grade_trim_slider.toolTip().startswith("<qt>")
    assert sidebar.paper_black_btn.toolTip().startswith("<qt>")
    assert "&lt;" not in sidebar.grade_trim_slider.toolTip()

    # Red page: sliders retarget to the red trims; global-only controls grey out.
    sidebar.ch_r_btn.setChecked(True)

    assert sidebar._curve_field("toe") == "toe_trim_red"
    assert sidebar._curve_field("shoulder") == "shoulder_trim_red"
    assert sidebar._curve_field("midtone_gamma") == "midtone_gamma_trim_red"
    assert sidebar._curve_field("shadow_grade") == "shadow_grade_trim_red"
    assert sidebar._curve_field("highlight_grade") == "highlight_grade_trim_red"
    assert sidebar._curve_field("toe_width") == "toe_width_trim_red"
    assert sidebar._curve_field("shoulder_width") == "shoulder_width_trim_red"
    assert sidebar._curve_field("dye_separation") == "dye_separation_trim_red"
    assert sidebar.grade_slider.isHidden()
    assert not sidebar.grade_trim_slider.isHidden()
    assert sidebar.grade_trim_slider.value() == 15.0
    assert sidebar.toe_w_slider.isHidden()
    assert not sidebar.toe_w_trim_slider.isHidden()
    assert sidebar.dye_separation_slider.isHidden()
    assert not sidebar.dye_separation_trim_slider.isHidden()
    assert abs(sidebar.dye_separation_trim_slider.value() - 0.25) < 1e-9
    assert sidebar.dye_separation_trim_slider.label.text() == "Dye Separation R"
    assert abs(sidebar.toe_slider.value() - 0.4) < 1e-9
    assert abs(sidebar.sh_slider.value() - (-0.2)) < 1e-9
    assert abs(sidebar.midtone_gamma_slider.value() - 0.15) < 1e-9
    assert abs(sidebar.toe_w_trim_slider.value() - 1.2) < 1e-9
    assert abs(sidebar.sh_w_trim_slider.value() - (-0.6)) < 1e-9
    assert abs(sidebar.shadow_grade_slider.value() - 5.0) < 1e-9
    assert abs(sidebar.highlight_grade_slider.value() - (-3.0)) < 1e-9
    assert sidebar.shadow_grade_slider.label.text() == "Shadows Grade R"
    assert sidebar.toe_slider.label.text() == "Toe R"
    assert sidebar.midtone_gamma_slider.label.text() == "Snap R"
    assert sidebar.toe_w_trim_slider.label.text() == "Toe Width R"
    assert sidebar.sh_w_trim_slider.label.text() == "Shoulder Width R"
    assert sidebar.midtone_gamma_slider.isEnabled()
    assert sidebar.midtone_gamma_slider not in sidebar._global_only
    for w in sidebar._global_only:
        assert not w.isEnabled()

    # Back to Global: values and enablement restore.
    sidebar.ch_global_btn.setChecked(True)
    assert sidebar.toe_slider.value() == 0.0
    assert abs(sidebar.midtone_gamma_slider.value() - 0.25) < 1e-9
    assert not sidebar.toe_w_slider.isHidden()
    for w in sidebar._global_only:
        assert w.isEnabled()


def test_channel_selector_hidden_in_bw(qapp):
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    sidebar.sync_ui()
    assert not sidebar.ch_r_btn.isHidden()
    sidebar.ch_r_btn.setChecked(True)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
    sidebar.sync_ui()
    for w in (sidebar.ch_global_btn, sidebar.ch_r_btn, sidebar.ch_g_btn, sidebar.ch_b_btn):
        assert w.isHidden()
    # Forced back to the Global page.
    assert sidebar._channel_index() == 0
    assert not sidebar.grade_slider.isHidden()
    # Dye Separation is a color control: gone on a single-emulsion B&W paper.
    assert sidebar.dye_separation_slider.isHidden()
    assert sidebar.dye_separation_trim_slider.isHidden()


def test_white_black_point_retarget_and_sync(qapp):
    """White Point/Black Point live on ProcessConfig, not ExposureConfig like the rest
    of this panel, but retarget through the same Global/R/G/B selector."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

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
    assert sidebar.ch_r_btn.edited_dot.isVisibleTo(sidebar.ch_r_btn)

    sidebar.ch_global_btn.setChecked(True)
    assert abs(sidebar.white_point_slider.value() - 0.1) < 1e-9
    assert sidebar.white_point_slider.label.text() == "White Point"


def test_white_black_point_write_to_process_not_exposure(qapp):
    """Unlike everything else this panel writes, White/Black Point are ProcessConfig
    fields -- update_config_section must target "process", not "exposure"."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    sidebar._on_white_point_changed(0.15, persist=True)

    args, kwargs = controller.apply_config.call_args
    assert args[0].process.white_point_offset == 0.15


def test_white_black_point_stay_visible_on_the_transparency_transfer(qapp):
    """White/Black Point deviate the transfer path's fixed window the same way they
    deviate a measured one (NormalizationProcessor._process_transparency), so they
    still have something to act on and stay visible, unlike the print-curve controls
    that don't apply there."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=False))
    sidebar.sync_ui()

    assert not sidebar.white_point_slider.isHidden()
    assert not sidebar.black_point_slider.isHidden()
    assert not sidebar.tonal_range_header.isHidden()


def test_auto_density_grade_hide_on_a_raw_slide_but_stay_on_a_positive(qapp):
    """They meter the frame to pick a look, which the transfer path exists to avoid for
    a deliberate camera exposure -- but a Positive frame carries no such bracket, so
    they run there exactly as on a negative (transfer_auto_terms)."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(
        cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=False, positive_source=False)
    )
    sidebar.sync_ui()
    assert sidebar.auto_density_btn.isHidden()
    assert sidebar.auto_grade_btn.isHidden()
    # The rest of the paper-model controls stay hidden either way.
    assert sidebar.paper_dmin_btn.isHidden()

    controller.state.config = replace(controller.state.config, process=replace(controller.state.config.process, positive_source=True))
    sidebar.sync_ui()
    assert not sidebar.auto_density_btn.isHidden()
    assert not sidebar.auto_grade_btn.isHidden()
    assert sidebar.paper_dmin_btn.isHidden()


def test_tonal_range_header_sits_directly_above_white_point(qapp):
    """Marks White/Black Point off from the print-curve controls below -- they come
    from a different pipeline stage (Normalization) and only share this card's
    Global/R/G/B selector, not its print-curve subject."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    header_i = sidebar.layout.indexOf(sidebar.tonal_range_header)
    assert header_i >= 0
    row_i = _row_index_containing(sidebar.layout, sidebar.white_point_slider)
    assert header_i == row_i - 1


def test_white_black_point_disabled_when_bounds_are_locked(qapp):
    """Trims shift the same frozen bounds, so further nudging is disabled once locked
    -- unlike Grade/Toe/Shoulder, which have nothing to do with Lock Bounds."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, lock_bounds=True))
    sidebar.sync_ui()

    assert not sidebar.white_point_slider.isEnabled()
    assert not sidebar.black_point_slider.isEnabled()
    assert sidebar.grade_slider.isEnabled()


def test_white_black_point_ignore_the_lock_on_the_transparency_transfer(qapp):
    """The transfer path's window is fixed, never measured, so a Lock Bounds left on
    from another frame or mode has nothing there to freeze."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=False, lock_bounds=True))
    sidebar.sync_ui()

    assert sidebar.white_point_slider.isEnabled()
    assert sidebar.black_point_slider.isEnabled()
