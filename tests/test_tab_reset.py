"""Resetting a whole tab from its icon's context menu, and the table both it and the
per-panel header resets read.

The table is the single source of truth for what a panel owns: a tab reset folds several
entries into one config so the whole tab costs one undo step, and a header reset folds one.
"""

from dataclasses import replace
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import QPoint, Qt

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import _SECTION_RESETS, ControlsPanel, _apply_section_resets
from negpy.desktop.view.sidebar.right_panel import RightPanel

_SETUP_KEYS = ["sensor", "demosaic", "process", "roll"]


def _panel():
    controller = MagicMock()
    controller.state = AppState()
    return controller, ControlsPanel(controller)


def _edited_setup(config):
    return replace(
        config,
        process=replace(
            config.process,
            crosstalk_strength=0.4,
            demosaic_preview="vng",
            white_point_offset=0.2,
            use_luma_average=True,
            roll_name="PORTRA-04",
        ),
    )


# --- the table -------------------------------------------------------------------------


def test_every_tab_panel_has_a_reset(qapp):
    """The menu maps section attributes onto table keys by name, so a panel renamed on one
    side and not the other would silently stop resetting."""
    _controller, panel = _panel()
    for page in panel.pages:
        for attr in page["sections"]:
            assert attr.removesuffix("_section") in _SECTION_RESETS, attr


def test_a_tab_reset_defaults_every_panel_on_it(qapp):
    config = _edited_setup(AppState().config)

    reset = _apply_section_resets(config, _SETUP_KEYS)

    assert reset.process == AppState().config.process
    assert reset.exposure == config.exposure
    assert reset.lab == config.lab


def test_a_panel_reset_leaves_the_other_tabs_alone(qapp):
    config = AppState().config
    config = replace(config, lab=replace(config.lab, saturation=0.5), toning=replace(config.toning, sepia_strength=0.3))

    reset = _apply_section_resets(config, ["lab"])

    assert reset.lab == AppState().config.lab
    assert reset.toning == config.toning


def test_reset_is_identity_on_a_clean_tab(qapp):
    """The menu item greys off this, and reset_sections skips a render on it."""
    config = AppState().config

    assert _apply_section_resets(config, _SETUP_KEYS) is config


def test_a_roll_reset_drops_the_cached_bounds(qapp):
    config = AppState().config
    config = replace(
        config,
        process=replace(config.process, use_luma_average=True, roll_name="PORTRA-04", local_floors=(0.1, 0.1, 0.1)),
    )

    reset = _apply_section_resets(config, ["roll"])

    assert reset.process.use_luma_average is False
    assert reset.process.roll_name is None
    assert reset.process.local_floors == (0.0, 0.0, 0.0)


def test_a_locked_frame_keeps_its_bounds(qapp):
    config = AppState().config
    config = replace(config, process=replace(config.process, use_luma_average=True, lock_bounds=True, local_floors=(0.1, 0.1, 0.1)))

    reset = _apply_section_resets(config, ["roll"])

    assert reset.process.local_floors == (0.1, 0.1, 0.1)


# --- the commit ------------------------------------------------------------------------


def test_a_tab_reset_commits_once(qapp):
    """One config, one apply_config, so one Ctrl+Z walks the whole tab back."""
    controller, panel = _panel()
    controller.state.config = _edited_setup(controller.state.config)

    panel.reset_sections(_SETUP_KEYS)

    assert controller.apply_config.call_count == 1
    applied = controller.apply_config.call_args[0][0]
    assert applied.process == AppState().config.process


def test_a_clean_tab_commits_nothing(qapp):
    controller, panel = _panel()

    panel.reset_sections(_SETUP_KEYS)

    controller.apply_config.assert_not_called()


def test_resetting_dodge_and_burn_drops_the_selection(qapp):
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, local=replace(cfg.local, masks=[{"kind": "brush"}]))
    controller.state.local_selected_mask = 0

    panel.reset_sections(["local"])

    assert controller.state.local_selected_mask == -1


def test_a_header_reset_asks_for_its_own_panel(qapp):
    controller, panel = _panel()
    controller.state.config = _edited_setup(controller.state.config)

    panel.sensor_section.reset_requested.emit()

    applied = controller.apply_config.call_args[0][0]
    default = AppState().config.process
    assert applied.process.crosstalk_strength == default.crosstalk_strength
    assert applied.process.white_point_offset == 0.2, "Normalization is a different panel"


# --- the menu --------------------------------------------------------------------------


def _right_panel_stub(*, keys, can_reset=True):
    panel = MagicMock()
    panel._tab_sections = {0: ["sensor_section"]}
    panel._tab_names = ["Setup"]
    panel.controls_panel.reset_keys_for.return_value = keys
    panel.controls_panel.can_reset_sections.return_value = can_reset
    return panel


def _exec_menu(panel, menu):
    with patch("negpy.desktop.view.sidebar.right_panel.QMenu", return_value=menu):
        RightPanel._show_tab_menu(panel, 0, QPoint(0, 0))


def test_right_click_offers_a_reset_for_the_tab(qapp):
    menu = MagicMock()
    _exec_menu(_right_panel_stub(keys=_SETUP_KEYS), menu)

    menu.addAction.assert_called_once()
    assert menu.addAction.call_args[0][0] == "Reset Setup to Defaults"


def test_a_tab_with_no_settings_has_no_menu(qapp):
    menu = MagicMock()
    _exec_menu(_right_panel_stub(keys=[]), menu)

    menu.addAction.assert_not_called()
    menu.exec.assert_not_called()


def test_the_item_is_greyed_on_a_clean_tab(qapp):
    menu = MagicMock()
    _exec_menu(_right_panel_stub(keys=_SETUP_KEYS, can_reset=False), menu)

    menu.addAction.return_value.setEnabled.assert_called_once_with(False)


def test_the_item_resets_every_panel_on_the_tab(qapp):
    menu = MagicMock()
    panel = _right_panel_stub(keys=_SETUP_KEYS)
    _exec_menu(panel, menu)

    menu.addAction.return_value.triggered.connect.call_args[0][0]()

    panel.controls_panel.reset_sections.assert_called_once_with(_SETUP_KEYS)


def test_only_the_workflow_tabs_offer_a_reset(qapp):
    """Against the real tab strip, so a tab added without a menu entry is caught here."""
    from conftest import FakeController

    panel = RightPanel(FakeController())
    offered = {}
    menu = MagicMock()
    with patch("negpy.desktop.view.sidebar.right_panel.QMenu", return_value=menu):
        for i, name in enumerate(panel._tab_names):
            menu.addAction.reset_mock()
            assert panel._tab_buttons[i].contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
            panel._tab_buttons[i].customContextMenuRequested.emit(QPoint(4, 4))
            if menu.addAction.called:
                offered[name] = menu.addAction.call_args[0][0]

    assert offered == {
        "Setup": "Reset Setup to Defaults",
        "Geometry": "Reset Geometry to Defaults",
        "Exposure": "Reset Exposure to Defaults",
        "Lab & Toning": "Reset Lab & Toning to Defaults",
        "Finish": "Reset Finish to Defaults",
    }


def test_the_keyboard_route_takes_the_tab_on_screen(qapp):
    panel = _right_panel_stub(keys=_SETUP_KEYS)
    panel._active_index = 0

    RightPanel.reset_active_tab(panel)

    panel.controls_panel.reset_sections.assert_called_once_with(_SETUP_KEYS)
