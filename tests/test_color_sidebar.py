from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.color import ColorSidebar
from negpy.features.exposure.logic import wb_to_kelvin


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    controller.session.repo.get_global_setting.return_value = None
    return controller, ColorSidebar(controller)


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
    sidebar.region_btn.setCurrentIndex(1)
    assert abs(sidebar.cyan_slider.value() - 0.1) < 1e-9
    assert abs(sidebar.magenta_slider.value() - (-0.3)) < 1e-9
    assert abs(sidebar.yellow_slider.value() - 0.4) < 1e-9
    assert abs(sidebar.temp_slider.value() - wb_to_kelvin(-0.3, 0.4)) < 1.0
    assert controller.state.wb_pick_region == 1
    assert sidebar._region_my(controller.state.config.exposure) == (-0.3, 0.4)
    assert sidebar._REGION_MY[sidebar._region_index()] == ("shadow_magenta", "shadow_yellow")


def test_temperature_writes_selected_region_fields(qapp):
    controller, sidebar = _sidebar()
    sidebar.region_btn.setCurrentIndex(2)

    sidebar._on_temp_changed(4500.0)

    call = controller.apply_config.call_args
    assert call is not None
    new_exposure = call.args[0].exposure
    assert new_exposure.highlight_magenta != 0.0 or new_exposure.highlight_yellow != 0.0
    assert new_exposure.wb_magenta == 0.0 and new_exposure.wb_yellow == 0.0


def test_cast_removal_is_a_filtration_field(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, exposure=replace(cfg.exposure, cast_removal_strength=0.3))
    sidebar.sync_ui()
    assert abs(sidebar.cast_removal_slider.value() - 0.3) < 1e-9

    sidebar.cast_removal_slider.valueCommitted.emit(0.6)
    applied = controller.apply_config.call_args.args[0]
    assert applied.exposure.cast_removal_strength == 0.6


def test_cast_removal_shows_wherever_the_render_honours_it(qapp):
    """A shown slider the render ignores is a dead control, and a hidden one it honours strands a
    live setting. The frame carries grays across the meter's three luma bands."""
    import numpy as np

    from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import ProcessMode
    from negpy.services.rendering.engine import DarkroomEngine

    controller = MagicMock()
    controller.state = AppState()
    controller.active_diptych.return_value = None
    panel = ControlsPanel(controller)
    rng = np.random.default_rng(4)
    v = np.geomspace(5e-4, 0.9, 64 * 64).astype(np.float32).reshape(64, 64)
    img = np.ascontiguousarray(np.stack([v, v * 0.9, v * 0.78], -1) + rng.uniform(0, 1e-4, (64, 64, 3)).astype(np.float32))

    for mode in (ProcessMode.C41, ProcessMode.E6, ProcessMode.BW):
        base = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode=mode))
        renders = [
            DarkroomEngine().process(img, replace(base, exposure=replace(base.exposure, cast_removal_strength=s)), f"cast_{mode}_{s}")
            for s in (0.0, 1.0)
        ]
        controller.state.config = base
        panel._sync_all_sidebars()
        shown = not panel.color_section.isHidden() and not panel.color_sidebar.cast_removal_slider.isHidden()
        assert shown == (not np.allclose(renders[0], renders[1])), mode
