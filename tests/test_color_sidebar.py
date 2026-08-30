from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.color import ColorSidebar
from negpy.features.exposure.logic import wb_to_kelvin
from negpy.features.process.models import ProcessMode


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

    call = controller.apply_config.call_args
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

    new_exposure = controller.apply_config.call_args.args[0].exposure
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


def test_cast_removal_is_colour_only(qapp):
    """Both colour processes solve against a neutral axis, so both show the slider. B&W
    collapses to one density before the curve and has nothing to balance."""
    controller, sidebar = _sidebar()

    for mode in (ProcessMode.C41, ProcessMode.E6):
        cfg = controller.state.config
        controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=mode))
        sidebar.sync_ui()
        assert not sidebar.cast_removal_slider.isHidden(), mode

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
    sidebar.sync_ui()
    assert sidebar.cast_removal_slider.isHidden()


def test_cast_removal_reaches_the_render_wherever_it_is_visible(qapp):
    """Pins the two halves together: a visible slider that the render ignores is a dead
    control, and a hidden one the render honours strands a live setting. Asserted against
    the render, not a repeated mode list. The frame carries greys across the meter's three
    luma bands, so every mode that *can* solve does."""
    import numpy as np

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.engine import DarkroomEngine

    controller, sidebar = _sidebar()
    rng = np.random.default_rng(4)
    v = np.geomspace(5e-4, 0.9, 64 * 64).astype(np.float32).reshape(64, 64)
    img = np.stack([v, v * 0.9, v * 0.78], -1)
    img = np.ascontiguousarray(img + rng.uniform(0, 1e-4, img.shape).astype(np.float32))

    for mode, normalize in ((ProcessMode.C41, True), (ProcessMode.E6, True), (ProcessMode.E6, False), (ProcessMode.BW, True)):
        s = WorkspaceConfig()
        base = replace(s, process=replace(s.process, process_mode=mode, e6_normalize=normalize))
        renders = [
            DarkroomEngine().process(
                img, replace(base, exposure=replace(base.exposure, cast_removal_strength=v)), f"cast_{mode}_{normalize}_{v}"
            )
            for v in (0.0, 1.0)
        ]
        render_honours_it = not np.allclose(renders[0], renders[1])

        cfg = controller.state.config
        controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=mode, e6_normalize=normalize))
        sidebar.sync_ui()
        visible = not sidebar.cast_removal_slider.isHidden()
        assert visible == render_honours_it, f"{mode} normalize={normalize}"
