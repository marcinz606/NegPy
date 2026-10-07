"""Dodge & Burn panel: the per-mask Burn and Grade controls, and how a mask is
labelled once it carries a grade as well as (or instead of) print exposure."""

from dataclasses import replace
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QPushButton

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.keyboard_shortcuts import _toggle_tool_button
from negpy.desktop.view.sidebar.local import LocalSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskKey, MaskShape
from negpy.features.process.models import ProcessMode

SQUARE = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))


def _sidebar(*masks: LocalMask, selected: int = 0):
    controller = MagicMock()
    controller.state = AppState()
    cfg = controller.state.config
    controller.state.config = replace(cfg, local=LocalAdjustmentsConfig(masks=masks))
    controller.state.local_selected_mask = selected
    return controller, LocalSidebar(controller)


def _row_text(sidebar: LocalSidebar, index: int = 0) -> str:
    """The mask row label. It is the second widget, after the shape icon."""
    row = sidebar.mask_list.itemWidget(sidebar.mask_list.item(index))
    return row.layout().itemAt(1).widget().text()


def test_grade_slider_is_disabled_without_a_selection(qapp):
    _, sidebar = _sidebar(selected=-1)
    sidebar.sync_ui()

    assert not sidebar.grade_slider.isEnabled()


def test_grade_slider_syncs_from_the_selected_mask(qapp):
    _, sidebar = _sidebar(
        LocalMask(vertices=SQUARE, stops=1.0, grade=-20.0),
        LocalMask(vertices=SQUARE, stops=-0.5, grade=15.0),
        selected=1,
    )
    sidebar.sync_ui()

    assert sidebar.grade_slider.isEnabled()
    assert sidebar.grade_slider.value() == 15.0


def test_moving_the_grade_slider_edits_only_that_mask(qapp):
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0), selected=0)
    sidebar.sync_ui()

    # setValue is the external-sync path and blocks signals; adjust_by is a gesture.
    sidebar.grade_slider.adjust_by(-25.0)

    controller.update_selected_local_mask.assert_called_with(grade=-25.0)


def test_a_grade_only_mask_is_labelled_grade(qapp):
    """Strength 0 with a grade is neither dodge nor burn, and an EV of +0.00 would
    read as a dodge that does nothing."""
    _, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=0.0, grade=-30.0))
    sidebar.sync_ui()

    assert "Grade" in _row_text(sidebar) and "-30 R" in _row_text(sidebar)
    assert "st" not in _row_text(sidebar)


def test_a_burn_with_a_grade_shows_both(qapp):
    _, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0, grade=-20.0))
    sidebar.sync_ui()

    text = _row_text(sidebar)
    assert "Burn" in text and "+1.00 st" in text and "-20 R" in text


def test_burn_slider_syncs_and_is_exposure_signed(qapp):
    """Positive is a burn, matching Print Density and the Finishing edge burn."""
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=0.75), selected=0)
    sidebar.sync_ui()

    assert sidebar.burn_slider.value() == 0.75
    assert "Burn" in _row_text(sidebar)

    sidebar.burn_slider.adjust_by(-1.5)
    controller.update_selected_local_mask.assert_called_with(stops=-0.75)


def test_a_dodge_is_a_negative_burn(qapp):
    _, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=-0.5))
    sidebar.sync_ui()

    text = _row_text(sidebar)
    assert "Dodge" in text and "-0.50 st" in text


def test_a_fresh_mask_starts_at_the_frames_own_exposure(qapp):
    """Default 0 stops: drawing a mask must not change the print until it is given
    a value, so the range can run the full +/-2 stops either way."""
    _, sidebar = _sidebar(LocalMask(vertices=SQUARE))
    sidebar.sync_ui()

    assert sidebar.burn_slider.value() == 0.0
    assert (sidebar.burn_slider._min, sidebar.burn_slider._max) == (-2.0, 2.0)
    assert "Grade" in _row_text(sidebar)


def test_each_draw_tool_arms_its_own_mode(qapp):
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE))
    for btn, mode in (
        (sidebar.draw_btn, ToolMode.LOCAL_DRAW),
        (sidebar.oval_btn, ToolMode.LOCAL_OVAL),
        (sidebar.gradient_btn, ToolMode.LOCAL_GRADIENT),
    ):
        btn.setChecked(True)
        controller.set_active_tool.assert_called_with(mode)
        btn.setChecked(False)
        controller.set_active_tool.assert_called_with(ToolMode.NONE)


def test_feather_is_inert_on_a_card_edge(qapp):
    """The handle distance sets the softness, so the slider does not apply."""
    _, sidebar = _sidebar(LocalMask(vertices=((0.2, 0.5), (0.8, 0.5)), shape=MaskShape.GRADIENT))
    sidebar.sync_ui()

    assert not sidebar.feather_slider.isEnabled()
    assert sidebar.burn_slider.isEnabled()


def test_the_row_invert_toggle_syncs_and_flips_that_mask(qapp):
    """Invert sits on the row like the eye and trash, so it acts on its own mask, selected or not."""
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0), LocalMask(vertices=SQUARE, stops=1.0, invert=True), selected=0)
    sidebar.sync_ui()

    row = sidebar.mask_list.itemWidget(sidebar.mask_list.item(1))
    invert = row.layout().itemAt(3).widget()
    assert invert.isChecked()
    invert.click()

    controller.set_local_mask_inverted.assert_called_with(1, False)


def test_the_shape_icon_click_toggles_enabled(qapp):
    """Clicking the shape icon flips the mask's current enabled state."""
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0, enabled=True))
    sidebar.sync_ui()

    row = sidebar.mask_list.itemWidget(sidebar.mask_list.item(0))
    shape_btn = row.layout().itemAt(0).widget()
    shape_btn.click()

    controller.set_local_mask_enabled.assert_called_with(0, False)


def test_a_disabled_mask_grays_out_its_row_text(qapp):
    _, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0, enabled=False))
    sidebar.sync_ui()

    row = sidebar.mask_list.itemWidget(sidebar.mask_list.item(0))
    label = row.layout().itemAt(1).widget()
    assert THEME.text_muted in label.styleSheet()


def test_grabbing_a_slider_tells_the_canvas_to_drop_the_tint(qapp):
    """The mask tint sits over the area being judged, so it steps aside for the gesture."""
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0))
    sidebar.sync_ui()

    for slider in (sidebar.burn_slider, sidebar.grade_slider, sidebar.feather_slider):
        controller.local_drag_changed.emit.reset_mock()
        slider.slider.setSliderDown(True)
        controller.local_drag_changed.emit.assert_called_once_with(True)
        slider.slider.setSliderDown(False)
        controller.local_drag_changed.emit.assert_called_with(False)


def test_all_tones_leaves_the_zone_controls_off(qapp):
    _, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0))
    sidebar.sync_ui()

    assert sidebar.tone_btn.currentIndex() == 0
    assert not sidebar.key_zone_slider.isEnabled()
    assert not sidebar.key_softness_slider.isEnabled()


def test_choosing_highlights_limits_the_selected_mask(qapp):
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0))
    sidebar.sync_ui()

    sidebar.tone_btn.choice_menu.actions()[1].trigger()

    controller.update_selected_local_mask.assert_called_with(key=MaskKey.HIGHLIGHTS)


def test_a_limited_mask_syncs_its_zone_and_names_it_in_the_row(qapp):
    _, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0, key=MaskKey.SHADOWS, key_zone=4.0, key_softness=2.0))
    sidebar.sync_ui()

    assert sidebar.tone_btn.currentIndex() == 2
    assert sidebar.key_zone_slider.isEnabled() and sidebar.key_zone_slider.value() == 4.0
    assert sidebar.key_softness_slider.value() == 2.0
    assert "≤IV" in _row_text(sidebar)


def test_moving_the_tone_zone_edits_the_mask(qapp):
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0, key=MaskKey.HIGHLIGHTS))
    sidebar.sync_ui()

    sidebar.key_zone_slider.adjust_by(1.0)

    controller.update_selected_local_mask.assert_called_with(key_zone=7.0)


def test_a_fifth_mask_cannot_be_limited(qapp):
    """Four limited masks fill the GPU's shape planes; a fifth would print unlimited."""
    limited = [LocalMask(vertices=SQUARE, stops=1.0, key=MaskKey.HIGHLIGHTS) for _ in range(4)]
    _, sidebar = _sidebar(*limited, LocalMask(vertices=SQUARE, stops=1.0), selected=4)
    sidebar.sync_ui()

    assert not sidebar.tone_btn.choice_menu.actions()[1].isEnabled()
    assert not sidebar.tone_btn.choice_menu.actions()[2].isEnabled()
    assert sidebar.tone_btn.currentIndex() == 0


def test_a_limited_mask_among_four_stays_editable(qapp):
    limited = [LocalMask(vertices=SQUARE, stops=1.0, key=MaskKey.HIGHLIGHTS) for _ in range(4)]
    _, sidebar = _sidebar(*limited, selected=2)
    sidebar.sync_ui()

    assert sidebar.tone_btn.choice_menu.actions()[2].isEnabled()


def test_the_masks_header_counts_the_frames_masks(qapp):
    _, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0), LocalMask(vertices=SQUARE, stops=-0.5))
    sidebar.sync_ui()
    assert sidebar.masks_header.text() == "MASKS · 2"


def _set_mode(controller, mode: ProcessMode) -> None:
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=mode))


def test_a_slide_grays_the_panel_and_says_why(qapp):
    """The slide's transfer curve takes no dodge/burn map, so a mask drawn there would do nothing."""
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0, key=MaskKey.HIGHLIGHTS))
    _set_mode(controller, ProcessMode.E6)
    sidebar.sync_ui()

    controls = (
        sidebar.draw_btn,
        sidebar.oval_btn,
        sidebar.gradient_btn,
        sidebar.mask_list,
        sidebar.burn_slider,
        sidebar.grade_slider,
        sidebar.feather_slider,
        sidebar.tone_btn,
        sidebar.key_zone_slider,
        sidebar.key_softness_slider,
    )
    assert not any(w.isEnabled() for w in controls)
    assert not sidebar.slide_hint.isHidden()
    assert sidebar.burn_slider.value() == 1.0

    _set_mode(controller, ProcessMode.C41)
    sidebar.sync_ui()

    assert all(w.isEnabled() for w in controls)
    assert sidebar.slide_hint.isHidden()


def test_a_slide_puts_an_armed_mask_tool_down(qapp):
    controller, sidebar = _sidebar()
    controller.state.active_tool = ToolMode.LOCAL_OVAL
    sidebar.sync_ui()
    controller.cancel_active_tool.assert_not_called()

    _set_mode(controller, ProcessMode.E6)
    sidebar.sync_ui()

    controller.cancel_active_tool.assert_called_once()


def test_a_grayed_tool_shortcut_shows_its_tab_without_arming(qapp):
    window = MagicMock()
    button = QPushButton("Draw")
    button.setCheckable(True)
    button.setEnabled(False)

    _toggle_tool_button(window, "tone", button)

    window.right_panel.show_tab_by_key.assert_called_once_with("tone")
    assert not button.isChecked()
    window.controller.set_status.assert_called_once_with("Draw not available", 1500, kind="warning")

    button.setEnabled(True)
    _toggle_tool_button(window, "tone", button)
    assert button.isChecked()


def test_a_shortcut_puts_down_an_armed_tool_on_a_grayed_page(qapp):
    """A read-only page grays an armed tool's button; its shortcut must still put the tool down."""
    window = MagicMock()
    button = QPushButton("Heal")
    button.setCheckable(True)
    button.setChecked(True)
    button.setEnabled(False)

    _toggle_tool_button(window, "finish", button)

    assert not button.isChecked()
    window.controller.set_status.assert_not_called()
