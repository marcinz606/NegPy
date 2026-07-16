from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.tone import ToneSidebar


def _combo_items(combo):
    return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]


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
            true_black=True,
            midtone_gamma=0.25,
            shadow_density=-0.45,
            highlight_density=0.2,
            shadow_grade=-12.0,
            highlight_grade=8.0,
            shadow_grade_trim_red=5.0,
            highlight_grade_trim_red=-3.0,
        ),
    )
    sidebar.sync_ui()

    # Global page: shared curve values, ISO-R grade slider shown.
    assert sidebar._curve_field("toe") == "toe"
    assert not sidebar.grade_slider.isHidden()
    assert sidebar.grade_trim_slider.isHidden()
    assert not sidebar.toe_w_slider.isHidden()
    assert sidebar.toe_w_trim_slider.isHidden()
    assert sidebar.true_black_btn.isChecked()
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
    assert sidebar.true_black_btn.toolTip().startswith("<qt>")
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
    assert sidebar.grade_slider.isHidden()
    assert not sidebar.grade_trim_slider.isHidden()
    assert sidebar.grade_trim_slider.value() == 15.0
    assert sidebar.toe_w_slider.isHidden()
    assert not sidebar.toe_w_trim_slider.isHidden()
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
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode="B&W"))
    sidebar.sync_ui()
    for w in (sidebar.ch_global_btn, sidebar.ch_r_btn, sidebar.ch_g_btn, sidebar.ch_b_btn):
        assert w.isHidden()
    # Forced back to the Global page.
    assert sidebar._channel_index() == 0
    assert not sidebar.grade_slider.isHidden()
