"""Flat Scan grays out the cards and controls that change only the print."""

from dataclasses import replace
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QWidget

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.color import ColorSidebar
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.sidebar.paper import PaperSidebar
from negpy.domain import models
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _panel():
    controller = MagicMock()
    controller.refuse_crop_edit.return_value = False
    controller.crop_edits_blocked.return_value = False
    controller.state = AppState()
    controller.state.config = DEFAULT_WORKSPACE_CONFIG
    controller.active_diptych.return_value = None
    panel = ControlsPanel(controller)
    panel._sync_all_sidebars()
    return controller, panel


def _flat(controller, panel, on: bool) -> None:
    controller.state.flat_peek = on
    panel._sync_flat_scan()


def _widgets(panel) -> list[QWidget]:
    """Every widget of the panel's pages and cards (the Roll cards sit outside the pages)."""
    roots = [p["widget"] for p in panel.pages] + [v for k, v in vars(panel).items() if k.endswith("_section")]
    return list({id(w): w for r in roots for w in (r, *r.findChildren(QWidget))}.values())


def _states(widgets: list[QWidget], probe) -> list[bool]:
    return [probe(w) for w in widgets]


def test_print_only_cards_gray_out_in_flat_scan(qapp):
    controller, panel = _panel()
    _flat(controller, panel, True)

    for name in ("tone", "local", "lab", "altproc", "toning", "retouch", "finish"):
        section = getattr(panel, f"{name}_section")
        assert not section.content_area.isEnabled(), name
        assert section.toggle_button.isEnabled(), name
    for name in ("geometry", "autocrop", "assembly"):
        assert getattr(panel, f"{name}_section").isEnabled(), name


def test_the_hint_shows_only_on_tabs_with_a_grayed_control(qapp):
    controller, panel = _panel()
    _flat(controller, panel, False)
    assert all(h.isHidden() for h in panel.flat_scan_hints.values())

    _flat(controller, panel, True)
    assert set(panel.flat_scan_hints) == {"tone", "color", "finish"}
    assert not any(h.isHidden() for h in panel.flat_scan_hints.values())


def test_leaving_flat_scan_restores_every_control(qapp):
    controller, panel = _panel()
    widgets = _widgets(panel)
    before = _states(widgets, QWidget.isEnabled)
    _flat(controller, panel, True)
    assert _states(widgets, QWidget.isEnabled) != before
    _flat(controller, panel, False)
    assert _states(widgets, QWidget.isEnabled) == before


def test_no_control_is_hidden_by_flat_scan(qapp):
    controller, panel = _panel()
    widgets = [w for w in _widgets(panel) if w not in panel.flat_scan_hints.values()]
    before = _states(widgets, QWidget.isHidden)
    _flat(controller, panel, True)
    after = _states(widgets, QWidget.isHidden)
    assert [w for w, was, now in zip(widgets, before, after) if now and not was] == []


def _sidebar(cls, *, flat=True, **process):
    controller = MagicMock()
    controller.state = AppState()
    controller.state.flat_peek = flat
    controller.session.repo.get_global_setting.return_value = None
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, **process))
    sidebar = cls(controller)
    sidebar.sync_ui()
    return controller, sidebar


def test_global_color_sliders_stay_live_and_region_sliders_gray_out(qapp):
    _controller, sidebar = _sidebar(ColorSidebar)
    live = (sidebar.temp_slider, sidebar.cyan_slider, sidebar.magenta_slider, sidebar.yellow_slider, sidebar.pick_wb_btn)
    assert all(w.isEnabled() for w in live)
    assert not sidebar.cast_removal_slider.isEnabled()
    assert not sidebar.ring_btn.isEnabled()

    sidebar.region_btn.setCurrentIndex(1)
    assert not any(w.isEnabled() for w in live)
    assert sidebar.region_btn.isEnabled()

    sidebar.state.flat_peek = False
    sidebar.sync_ui()
    assert all(w.isEnabled() for w in (*live, sidebar.cast_removal_slider, sidebar.ring_btn))


def test_print_stock_stays_live_on_a_color_negative(qapp):
    _controller, sidebar = _sidebar(PaperSidebar)
    assert sidebar.paper_combo.isEnabled()
    for w in (sidebar.paper_black_btn, sidebar.paper_dmin_btn, sidebar.ch_btn, sidebar.dye_separation_slider, sidebar.paper_strip_btn):
        assert not w.isEnabled()
    assert not sidebar.curve_fold.content.isEnabled()
    assert sidebar.curve_fold.isEnabled()

    _controller, bw = _sidebar(PaperSidebar, process_mode=ProcessMode.BW)
    assert not bw.paper_combo.isEnabled()


def test_a_field_added_to_the_flat_set_enables_its_control(qapp, monkeypatch):
    monkeypatch.setitem(models._FLAT_FIELDS, "exposure", models._FLAT_FIELDS["exposure"] | {"dye_separation"})
    _controller, sidebar = _sidebar(PaperSidebar)
    assert sidebar.dye_separation_slider.isEnabled()
    assert not sidebar.paper_black_btn.isEnabled()


def test_pick_better_and_the_auto_mirror_gray_out(qapp):
    controller, panel = _panel()
    _flat(controller, panel, True)
    assert not panel.pick_better_btn.isEnabled()
    assert not panel.auto_mirror_btn.isEnabled()
    _flat(controller, panel, False)
    assert panel.pick_better_btn.isEnabled()
    assert panel.auto_mirror_btn.isEnabled()


def test_cast_average_grays_out_in_flat_scan(qapp):
    controller, panel = _panel()
    process = replace(controller.state.config.process, process_mode=ProcessMode.C41, use_cast_average=True)
    controller.state.config = replace(controller.state.config, process=process)
    controller.state.active_roll_id = "roll"
    panel._sync_all_sidebars()
    cast = panel.process_sidebar.use_cast_avg_btn
    assert cast.isEnabled()
    controller.state.flat_peek = True
    panel._sync_all_sidebars()
    assert not cast.isEnabled()
    assert panel.process_sidebar.use_luma_avg_btn.isEnabled()
    assert not panel.roll_flat_scan_hint.isHidden()
    assert panel.baseline_section.content_area.isEnabled()

    controller.state.flat_peek = False
    panel._sync_all_sidebars()
    assert panel.roll_flat_scan_hint.isHidden()


def test_export_warns_when_the_canvas_is_flat_and_the_intent_is_not(qapp):
    from conftest import FakeController, FakeRepo

    from negpy.desktop.view.sidebar.export import ExportSidebar

    sidebar = ExportSidebar(FakeController(repo=FakeRepo()))
    hint = sidebar.flat_scan_mismatch_hint
    assert hint.isHidden()

    sidebar.state.flat_peek = True
    sidebar._on_flat_peek_changed(True)
    assert not hint.isHidden()

    sidebar.state.flat_output = True
    sidebar._on_flat_output_changed(True)
    assert hint.isHidden()

    sidebar.state.flat_peek = False
    sidebar.state.flat_output = False
    sidebar._on_flat_peek_changed(False)
    assert hint.isHidden()


def test_overlay_toggles_gray_out_in_flat_scan(qapp):
    from unittest.mock import MagicMock

    from negpy.desktop.view.canvas.toolbar import ActionToolbar
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    flags = ("zones_overlay", "clipping_overlay")
    stub = MagicMock()
    stub.session.state = AppState()
    stub._overlay_actions = {f: MagicMock() for f in flags}
    stub._ov_view_actions = {f: MagicMock() for f in flags}
    stub._info_actions = [MagicMock() for _ in range(3)]
    stub._source_actions = [MagicMock() for _ in range(32)]
    stub.session.state.flat_peek = True
    ActionToolbar._sync_view_state(stub)
    for f in flags:
        stub._overlay_actions[f].setEnabled.assert_called_with(False)
        stub._ov_view_actions[f].setEnabled.assert_called_with(False)
    stub.session.state.flat_peek = False
    ActionToolbar._sync_view_state(stub)
    stub._overlay_actions["zones_overlay"].setEnabled.assert_called_with(True)

    panel = MagicMock()
    RightPanel._sync_flat_overlay_btns(panel, True)
    panel.clipping_btn.setEnabled.assert_called_once_with(False)
    panel.zone_highlight_btn.setEnabled.assert_called_once_with(False)


def test_flat_scan_ends_a_shadow_or_highlight_pick(qapp):
    from negpy.desktop.session import ToolMode

    controller, sidebar = _sidebar(ColorSidebar)
    sidebar.state.active_tool = ToolMode.WB_PICK
    sidebar.sync_ui()
    controller.set_active_tool.assert_not_called()

    sidebar.region_btn.setCurrentIndex(1)
    controller.set_active_tool.assert_called_once_with(ToolMode.NONE)

    controller.set_active_tool.reset_mock()
    sidebar.state.flat_peek = False
    sidebar.sync_ui()
    controller.set_active_tool.assert_not_called()
