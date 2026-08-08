"""Dodge & Burn panel: the per-mask Burn and Grade controls, and how a mask is
labelled once it carries a grade as well as (or instead of) print exposure."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.sidebar.local import LocalSidebar
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskShape

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


def test_invert_toggle_syncs_and_writes_back(qapp):
    controller, sidebar = _sidebar(LocalMask(vertices=SQUARE, stops=1.0, invert=True))
    sidebar.sync_ui()

    assert sidebar.invert_btn.isChecked()
    assert "inv" in _row_text(sidebar)
    sidebar.invert_btn.setChecked(False)
    controller.update_selected_local_mask.assert_called_with(invert=False)
