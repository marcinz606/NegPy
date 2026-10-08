"""Frame › Geometry's Ratio mirrors the Roll › Crop card's: one roll field, two combos."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _panel():
    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = DEFAULT_WORKSPACE_CONFIG
    return controller, ControlsPanel(controller)


def test_either_combo_writes_the_same_roll_field(qapp):
    controller, panel = _panel()

    panel.geometry_sidebar.ratio_combo.setCurrentText("6:7")
    controller.set_crop_ratio.assert_called_with("6:7")

    panel.autocrop_sidebar.ratio_combo.setCurrentText("5:4")
    controller.set_crop_ratio.assert_called_with("5:4")


def test_both_combos_follow_the_stored_ratio(qapp):
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, geometry=replace(cfg.geometry, autocrop_ratio="6:7"))

    panel._sync_all_sidebars()

    assert panel.geometry_sidebar.ratio_combo.currentText() == "6:7"
    assert panel.autocrop_sidebar.ratio_combo.currentText() == "6:7"
    controller.set_crop_ratio.assert_not_called()


def test_geometry_crop_header_holds_only_the_hand_drawn_crop_tools(qapp):
    """Auto Crop is a roll setting on the Crop card; Geometry keeps the Crop tool and Reset."""
    from PyQt6.QtWidgets import QPushButton

    _controller, panel = _panel()
    geo = panel.geometry_sidebar
    row = next(
        geo.layout.itemAt(i).layout()
        for i in range(geo.layout.count())
        if geo.layout.itemAt(i).layout() is not None and geo.layout.itemAt(i).layout().indexOf(geo.manual_crop_btn) != -1
    )
    buttons = [row.itemAt(i).widget() for i in range(row.count()) if isinstance(row.itemAt(i).widget(), QPushButton)]
    assert buttons == [geo.manual_crop_btn, geo.clear_crop_btn]


def test_the_auto_crop_shortcut_toggles_the_crop_cards_wand():
    from negpy.desktop.view.keyboard_shortcuts import ShortcutManager

    manager = ShortcutManager.__new__(ShortcutManager)
    manager.window = MagicMock()

    manager._build_actions()["auto_crop"]()

    manager.window.controls_panel.autocrop_sidebar.auto_frame_btn.toggle.assert_called_once_with()
