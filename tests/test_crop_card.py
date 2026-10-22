"""Frame › Geometry: the Crop card holds every crop control; Straighten holds none."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _panel():
    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = DEFAULT_WORKSPACE_CONFIG
    controller.session.repo.get_global_setting.return_value = None
    return controller, ControlsPanel(controller)


def _row_of(sidebar, widget):
    return next(
        sidebar.layout.itemAt(i).layout()
        for i in range(sidebar.layout.count())
        if sidebar.layout.itemAt(i).layout() is not None and sidebar.layout.itemAt(i).layout().indexOf(widget) != -1
    )


def test_draw_and_auto_crop_share_the_first_row(qapp):
    _controller, panel = _panel()
    crop = panel.autocrop_sidebar

    row = _row_of(crop, crop.manual_crop_btn)

    assert row.indexOf(crop.auto_frame_btn) != -1
    assert (crop.manual_crop_btn.text().strip(), crop.auto_frame_btn.text().strip()) == ("Draw", "Auto Crop")


def test_draw_arms_the_crop_tool_and_its_x_clears_a_drawn_crop(qapp):
    controller, panel = _panel()
    crop = panel.autocrop_sidebar

    crop.manual_crop_btn.setChecked(True)
    controller.set_active_tool.assert_called_with(ToolMode.CROP_MANUAL)
    crop.manual_crop_btn.clear_btn.click()
    assert controller.reset_crop.call_count == 1


def test_the_x_shows_only_for_a_drawn_crop(qapp):
    controller, panel = _panel()
    crop = panel.autocrop_sidebar
    cfg = controller.state.config
    for geometry, shown in (
        ({"crop_rect": (0.1, 0.1, 0.9, 0.9)}, True),
        ({"crop_rect": (0.1, 0.1, 0.9, 0.9), "crop_from_auto": True}, False),
        ({}, False),
    ):
        controller.state.config = replace(cfg, geometry=replace(cfg.geometry, **geometry))
        crop.sync_ui()
        assert crop.manual_crop_btn.clear_btn.isHidden() is not shown, geometry


def test_the_ratio_follows_the_stored_value_and_writes_through_set_crop_ratio(qapp):
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, geometry=replace(cfg.geometry, autocrop_ratio="6:7"))

    panel._sync_all_sidebars()
    assert panel.autocrop_sidebar.ratio_combo.currentText() == "6:7"
    controller.set_crop_ratio.assert_not_called()

    panel.autocrop_sidebar.ratio_combo.setCurrentText("5:4")
    controller.set_crop_ratio.assert_called_with("5:4")


def test_the_settings_fold_holds_the_detection_settings_and_starts_closed(qapp):
    _controller, panel = _panel()
    crop = panel.autocrop_sidebar

    assert not crop.settings_fold.is_open()
    for widget in (crop.mode_combo, crop.offset_slider, crop.rebate_trim_slider, crop.square_btn, crop.auto_crop_all_btn):
        assert crop.settings_fold.isAncestorOf(widget)


def test_the_straighten_card_holds_no_crop_control(qapp):
    _controller, panel = _panel()
    geo = panel.geometry_sidebar

    for name in ("manual_crop_btn", "auto_crop_btn", "clear_crop_btn", "ratio_combo", "guide_combo"):
        assert not hasattr(geo, name), name
    for btn in (geo.rot_l_btn, geo.rot_r_btn, geo.flip_h_btn, geo.flip_v_btn):
        assert btn.toolTip()


def test_the_auto_crop_shortcut_toggles_the_crop_cards_wand():
    from negpy.desktop.view.keyboard_shortcuts import ShortcutManager

    manager = ShortcutManager.__new__(ShortcutManager)
    manager.window = MagicMock()

    manager._build_actions()["auto_crop"]()

    manager.window.controls_panel.autocrop_sidebar.auto_frame_btn.toggle.assert_called_once_with()
