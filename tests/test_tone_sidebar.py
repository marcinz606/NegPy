from dataclasses import replace
from unittest.mock import MagicMock

import pytest

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
    """The section header's reset button resets the fields listed in TONE_FIELDS, so
    every control the panel shows has to be in it — a renamed field that falls out of
    the list leaves a visible slider its own reset can't clear."""
    from negpy.desktop.settings_catalog import TONE_FIELDS

    for field in (
        "dye_separation",
        "dye_separation_trim_red",
        "dye_separation_trim_green",
        "dye_separation_trim_blue",
        "separation_damping",
    ):
        assert field in TONE_FIELDS


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


def test_separation_damping_armed_by_a_trim_alone(qapp):
    """A per-channel trim also gives Dye Separation a real per-pixel push even with the
    global value left at its neutral 1.0 — the enabled check must ask the same question
    the pipeline does (per_channel_dye_separation), not just the global scalar."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, dye_separation_trim_red=0.3))
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
    sidebar.ch_btn.setCurrentIndex(1)

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
    sidebar.ch_btn.setCurrentIndex(0)
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
    assert not sidebar.ch_btn.isHidden()
    sidebar.ch_btn.setCurrentIndex(1)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
    sidebar.sync_ui()
    assert sidebar.ch_btn.isHidden()
    # Forced back to the Global page.
    assert sidebar._channel_index() == 0
    assert not sidebar.grade_slider.isHidden()
    # Dye Separation is a color control: gone on a single-emulsion B&W paper.
    assert sidebar.dye_separation_slider.isHidden()
    assert sidebar.dye_separation_trim_slider.isHidden()


def test_auto_density_grade_stay_on_a_raw_slide_and_a_positive(qapp):
    """Both meter a slide on the transfer curve (transfer_auto_terms), raw or Positive."""
    controller = MagicMock()
    controller.state = AppState()
    sidebar = ToneSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, positive_source=False))
    sidebar.sync_ui()
    assert not sidebar.auto_btn.isHidden()
    # The rest of the paper-model controls stay hidden either way.
    assert sidebar.paper_dmin_btn.isHidden()

    controller.state.config = replace(controller.state.config, process=replace(controller.state.config.process, positive_source=True))
    sidebar.sync_ui()
    assert not sidebar.auto_btn.isHidden()
    assert sidebar.paper_dmin_btn.isHidden()


def test_dye_separation_trim_swaps_per_channel_on_transfer_too(qapp):
    """The transfer curve now wires the per-channel trims the same way the print path
    does, so the global/trim swap on the channel tabs must match — not the old
    print-only exemption that kept the trim hidden and the global slider always shown."""
    controller = MagicMock()
    controller.state = AppState()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6),
        exposure=replace(cfg.exposure, dye_separation=1.3, dye_separation_trim_red=0.25),
    )
    sidebar = ToneSidebar(controller)
    sidebar.sync_ui()

    assert not sidebar.dye_separation_slider.isHidden()
    assert sidebar.dye_separation_trim_slider.isHidden()
    assert not sidebar.separation_damping_slider.isHidden()

    sidebar.ch_btn.setCurrentIndex(1)
    assert sidebar.dye_separation_slider.isHidden()
    assert not sidebar.dye_separation_trim_slider.isHidden()
    assert abs(sidebar.dye_separation_trim_slider.value() - 0.25) < 1e-9
    assert sidebar.separation_damping_slider.isHidden()


def _metered_sidebar():
    from negpy.features.exposure.normalization import LogNegativeBounds

    controller = MagicMock()
    controller.state = AppState()
    controller.state.current_file_hash = "frame"
    controller.state.auto_meters["frame"] = {
        "metered_anchor": 0.55,
        "textural_range": 1.1,
        "norm_density_range": 1.4,
        "shadow_point": 0.9,
        "highlight_point": 0.03,
        "final_bounds": LogNegativeBounds((0.2, 0.5, 0.8), (1.6, 1.9, 2.1)),
    }
    return controller, ToneSidebar(controller)


def test_auto_driven_sliders_show_what_prints(qapp):
    from negpy.desktop.auto_sliders import shown_values

    controller, sidebar = _metered_sidebar()
    sidebar.sync_ui()
    shown = shown_values(controller.state.config, controller.state.auto_meters["frame"])

    assert sidebar.density_slider.value() == pytest.approx(shown["density"], abs=0.005)
    assert sidebar.grade_slider.value() == pytest.approx(shown["grade"], abs=0.5)
    assert sidebar.highlight_density_slider.value() == pytest.approx(shown["highlight_density"], abs=0.005)
    # The meter's own choice is the default, so an untouched frame reads as unedited.
    for slider in (sidebar.density_slider, sidebar.grade_slider, sidebar.highlight_density_slider):
        assert not slider._edited_dot.isVisibleTo(slider)


def test_auto_driven_slider_edit_stores_the_offset(qapp):
    from negpy.desktop.auto_sliders import stored_value

    controller, sidebar = _metered_sidebar()
    sidebar.sync_ui()
    sidebar.density_slider.valueCommitted.emit(1.5)

    committed = controller.apply_config.call_args.args[0].exposure
    expected = stored_value(controller.state.config, controller.state.auto_meters["frame"], "density", 1.5)
    assert committed.density == pytest.approx(expected)
    assert committed.auto_exposure


def test_reset_on_an_auto_driven_slider_restores_the_neutral_offset(qapp):
    controller, sidebar = _metered_sidebar()
    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, density=1.3))
    sidebar.sync_ui()
    sidebar.density_slider.valueCommitted.emit(sidebar.density_slider.default_value())

    assert controller.apply_config.call_args.args[0].exposure.density == 1.0


def test_auto_toggle_flips_only_the_flag_so_the_slider_drops_the_meter(qapp):
    controller, sidebar = _metered_sidebar()
    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, density=1.2))
    sidebar.sync_ui()
    metered = sidebar.density_slider.value()

    sidebar.auto_density_action.setChecked(False)
    off = controller.apply_config.call_args.args[0]
    assert not off.exposure.auto_exposure
    assert off.exposure.density == 1.2

    controller.state.config = off
    sidebar.sync_ui()
    assert sidebar.density_slider.value() == pytest.approx(1.2)
    assert metered != pytest.approx(1.2)
