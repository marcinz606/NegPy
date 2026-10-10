from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.paper import PaperSidebar
from negpy.features.process.models import ProcessMode


def _sidebar(**exposure):
    controller = MagicMock()
    controller.state = AppState()
    cfg = controller.state.config
    controller.state.config = replace(cfg, exposure=replace(cfg.exposure, **exposure))
    sidebar = PaperSidebar(controller)
    sidebar.sync_ui()
    return controller, sidebar


def _combo_items(combo):
    return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]


def test_paper_reset_covers_every_paper_control():
    """The card's reset clears PAPER_FIELDS, so a control that falls out of the list
    leaves a visible slider its own reset cannot clear."""
    from negpy.desktop.settings_catalog import PAPER_FIELDS

    for field in ("paper_profile", "paper_black", "paper_dmin", "midtone_gamma", "toe", "shoulder", "dye_separation", "separation_damping"):
        assert field in PAPER_FIELDS


def test_separation_damping_locked_without_a_separation_push(qapp):
    """It redistributes Dye Separation's push and has no effect of its own, so at
    separation 1.0 it renders nothing — a live slider there reads as broken."""
    controller, sidebar = _sidebar()
    assert not sidebar.separation_damping_slider.isEnabled()

    conf = controller.state.config
    controller.state.config = replace(conf, exposure=replace(conf.exposure, dye_separation=1.3))
    sidebar.sync_ui()
    assert sidebar.separation_damping_slider.isEnabled()


def test_separation_damping_armed_by_a_trim_alone(qapp):
    """A per-channel trim gives Dye Separation a real push with the global value at 1.0."""
    _controller, sidebar = _sidebar(dye_separation_trim_red=0.3)
    assert sidebar.separation_damping_slider.isEnabled()


def test_paper_combo_rebuilt_only_when_entries_change(qapp):
    _controller, sidebar = _sidebar()
    items = _combo_items(sidebar.paper_combo)
    assert items

    clears = []
    orig_clear = sidebar.paper_combo.clear
    sidebar.paper_combo.clear = lambda: (clears.append(1), orig_clear())[1]

    sidebar.sync_ui()  # unchanged process mode -> no rebuild
    assert clears == []
    assert _combo_items(sidebar.paper_combo) == items


def test_channel_selector_retargets_and_syncs(qapp):
    _controller, sidebar = _sidebar(
        toe_trim_red=0.4,
        shoulder_trim_red=-0.2,
        midtone_gamma_trim_red=0.15,
        toe_width_trim_red=1.2,
        shoulder_width_trim_red=-0.6,
        paper_black=True,
        midtone_gamma=0.25,
        dye_separation=1.3,
        dye_separation_trim_red=0.25,
    )

    assert sidebar._curve_field("toe") == "toe"
    assert not sidebar.toe_w_slider.isHidden()
    assert sidebar.toe_w_trim_slider.isHidden()
    assert not sidebar.dye_separation_slider.isHidden()
    assert sidebar.dye_separation_trim_slider.isHidden()
    assert abs(sidebar.dye_separation_slider.value() - 1.3) < 1e-9
    assert sidebar.paper_black_btn.isChecked()
    assert sidebar.paper_black_btn.toolTip().startswith("<qt>")
    assert abs(sidebar.midtone_gamma_slider.value() - 0.25) < 1e-9

    sidebar.ch_btn.setCurrentIndex(1)

    for base in ("toe", "shoulder", "midtone_gamma", "toe_width", "shoulder_width", "dye_separation"):
        assert sidebar._curve_field(base) == f"{base}_trim_red"
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
    assert sidebar.toe_slider.label.text() == "Toe R"
    assert sidebar.midtone_gamma_slider.label.text() == "Snap R"
    assert sidebar.toe_w_trim_slider.label.text() == "Toe Width R"
    assert sidebar.sh_w_trim_slider.label.text() == "Shoulder Width R"
    assert sidebar.midtone_gamma_slider.isEnabled()
    assert sidebar.ch_btn._edited[1]
    for w in sidebar._global_only:
        assert not w.isEnabled()

    sidebar.ch_btn.setCurrentIndex(0)
    assert sidebar.toe_slider.value() == 0.0
    assert abs(sidebar.midtone_gamma_slider.value() - 0.25) < 1e-9
    assert not sidebar.toe_w_slider.isHidden()
    for w in sidebar._global_only:
        assert w.isEnabled()


def test_a_tone_trim_marks_no_paper_layer(qapp):
    """Each card's selector dots only its own trims."""
    _controller, sidebar = _sidebar(grade_trim_red=15.0, shadow_grade_trim_red=5.0)
    assert not any(sidebar.ch_btn._edited)


def test_channel_selector_and_dye_separation_hidden_in_bw(qapp):
    controller, sidebar = _sidebar()
    sidebar.ch_btn.setCurrentIndex(1)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
    sidebar.sync_ui()
    assert sidebar.ch_btn.isHidden()
    assert sidebar._channel_index() == 0
    assert sidebar.dye_separation_slider.isHidden()
    assert sidebar.dye_separation_trim_slider.isHidden()


def test_paper_model_controls_hidden_on_a_slide_and_a_positive(qapp):
    controller = MagicMock()
    controller.state = AppState()
    sidebar = PaperSidebar(controller)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, positive_source=False))
    sidebar.sync_ui()
    assert sidebar.paper_dmin_btn.isHidden()
    assert sidebar.paper_combo.isHidden()
    assert sidebar.paper_label.isHidden()
    assert not sidebar.toe_slider.isHidden()

    controller.state.config = replace(controller.state.config, process=replace(controller.state.config.process, positive_source=True))
    sidebar.sync_ui()
    assert sidebar.paper_dmin_btn.isHidden()


def test_dye_separation_trim_swaps_per_channel_on_transfer_too(qapp):
    """The transfer curve wires the per-channel trims like the print path, so the global/trim
    swap on the channel tabs matches."""
    controller = MagicMock()
    controller.state = AppState()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6),
        exposure=replace(cfg.exposure, dye_separation=1.3, dye_separation_trim_red=0.25),
    )
    sidebar = PaperSidebar(controller)
    sidebar.sync_ui()

    assert not sidebar.dye_separation_slider.isHidden()
    assert sidebar.dye_separation_trim_slider.isHidden()
    assert not sidebar.separation_damping_slider.isHidden()

    sidebar.ch_btn.setCurrentIndex(1)
    assert sidebar.dye_separation_slider.isHidden()
    assert not sidebar.dye_separation_trim_slider.isHidden()
    assert abs(sidebar.dye_separation_trim_slider.value() - 0.25) < 1e-9
    assert sidebar.separation_damping_slider.isHidden()


def test_mini_curve_paints_base_and_channel_traces(qapp):
    from negpy.desktop.view.widgets.charts import MiniCurveWidget

    w = MiniCurveWidget()
    w.resize(200, 24)
    base = [(x / 10, x / 10) for x in range(11)]
    w.set_curves(base, [base, base, base])
    assert not w.grab().isNull()
    w.set_curves([], [])
    assert not w.grab().isNull()
