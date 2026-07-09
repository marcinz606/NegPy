from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.colour import ColourSidebar
from negpy.features.exposure.logic import wb_to_kelvin


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    controller.session.repo.get_global_setting.return_value = None
    return controller, ColourSidebar(controller)


def test_region_selector_retargets_sliders_and_temperature(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        exposure=replace(cfg.exposure, wb_magenta=0.2, shadow_cyan=0.1, shadow_magenta=-0.3, shadow_yellow=0.4),
    )
    sidebar.sync_ui()

    # Global page: global values.
    assert abs(sidebar.magenta_slider.value() - 0.2) < 1e-9
    assert abs(sidebar.temp_slider.value() - wb_to_kelvin(0.2, 0.0)) < 1.0

    # Shadows page: sliders and temperature lever retarget to the shadow pair,
    # and the WB picker is scoped to the region via AppState.
    sidebar.region_shadow_btn.setChecked(True)
    assert abs(sidebar.cyan_slider.value() - 0.1) < 1e-9
    assert abs(sidebar.magenta_slider.value() - (-0.3)) < 1e-9
    assert abs(sidebar.yellow_slider.value() - 0.4) < 1e-9
    assert abs(sidebar.temp_slider.value() - wb_to_kelvin(-0.3, 0.4)) < 1.0
    assert controller.state.wb_pick_region == 1
    assert sidebar._region_my(controller.state.config.exposure) == (-0.3, 0.4)
    assert sidebar._REGION_MY[sidebar._region_index()] == ("shadow_magenta", "shadow_yellow")


def test_temperature_writes_selected_region_fields(qapp):
    controller, sidebar = _sidebar()
    sidebar.region_highlight_btn.setChecked(True)

    sidebar._on_temp_changed(4500.0)

    call = controller.session.update_config.call_args
    assert call is not None
    new_exposure = call.args[0].exposure
    assert new_exposure.highlight_magenta != 0.0 or new_exposure.highlight_yellow != 0.0
    assert new_exposure.wb_magenta == 0.0 and new_exposure.wb_yellow == 0.0


def test_region_reset_zeroes_selected_region_only(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        exposure=replace(cfg.exposure, wb_magenta=0.2, shadow_cyan=0.1, shadow_magenta=-0.3, shadow_yellow=0.4),
    )
    sidebar.region_shadow_btn.setChecked(True)

    sidebar._on_region_reset()

    new_exposure = controller.session.update_config.call_args.args[0].exposure
    assert new_exposure.shadow_cyan == 0.0
    assert new_exposure.shadow_magenta == 0.0
    assert new_exposure.shadow_yellow == 0.0
    assert new_exposure.wb_magenta == 0.2  # other regions untouched


def test_temperature_lock_is_per_region(qapp):
    controller, sidebar = _sidebar()
    sidebar.region_shadow_btn.setChecked(True)

    sidebar.temp_lock_btn.setChecked(True)  # user locks the shadow temperature

    key, value = controller.session.repo.save_global_setting.call_args.args
    assert key == "wb_temp_lock_shadow"
    assert value is not None

    sidebar.region_highlight_btn.setChecked(True)
    sidebar.temp_lock_btn.setChecked(True)
    key, _ = controller.session.repo.save_global_setting.call_args.args
    assert key == "wb_temp_lock_highlight"
