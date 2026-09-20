from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.features.process.models import ProcessMode


def _panel():
    controller = MagicMock()
    controller.state = AppState()
    return controller, ControlsPanel(controller)


def test_calibration_edits_light_their_section(qapp):
    """Calibration owns fields on ProcessConfig, so it was never counted — its header
    showed no count and its reset button stayed hidden forever."""
    controller, panel = _panel()
    panel._sync_modified_dots()
    assert panel.sensor_section.modified_count == 0
    assert not panel.sensor_section.reset_btn.isVisible()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, crosstalk_strength=0.4, hue_trim=3.0))
    panel._sync_modified_dots()

    assert panel.sensor_section.modified_count == 2


def test_flat_field_edits_light_their_section(qapp):
    controller, panel = _panel()
    panel._sync_modified_dots()
    assert panel.flatfield_section.modified_count == 0

    cfg = controller.state.config
    controller.state.config = replace(cfg, flatfield=replace(cfg.flatfield, apply=True, profile_id="rig-1"))
    panel._sync_modified_dots()

    assert panel.flatfield_section.modified_count == 2


def test_calibration_fields_are_not_double_counted_in_process(qapp):
    """The two sections share ProcessConfig; a Calibration edit must not also inflate
    the Process count."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, crosstalk_strength=0.4))
    panel._sync_modified_dots()

    assert panel.sensor_section.modified_count == 1
    assert panel.process_section.modified_count == 0


def test_calibration_reset_button_restores_its_fields(qapp):
    """The header's reset button was shown by the count above but never connected, so it
    was a control that did nothing."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, crosstalk_strength=0.4, hue_trim=3.0))

    panel.sensor_section.reset_requested.emit()

    applied = controller.apply_config.call_args[0][0]
    assert applied.process.crosstalk_strength == cfg.process.crosstalk_strength
    assert applied.process.hue_trim == cfg.process.hue_trim


def test_flat_field_reset_button_restores_its_section(qapp):
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, flatfield=replace(cfg.flatfield, apply=True, profile_id="rig-1"))

    panel.flatfield_section.reset_requested.emit()

    applied = controller.apply_config.call_args[0][0]
    assert applied.flatfield == cfg.flatfield


def test_transparency_at_true_default_shows_color_unmodified(qapp):
    """cast_removal_strength's real default on a slide is 0 (cast_removal_for_mode), not
    the bare ExposureConfig 0.5 — an untouched transparency must not show as modified."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6),
        exposure=replace(cfg.exposure, cast_removal_strength=0.0),
    )
    panel._sync_modified_dots()

    assert panel.color_section.modified_count == 0


def test_color_reset_zeroes_cast_removal_on_transparency(qapp):
    """Resetting Color on a slide must land on cast_removal_for_mode's default (0), not
    the flat ExposureConfig default (0.5), or the reset reintroduces a gray-balance the
    live render never wants on a transparency."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6),
        exposure=replace(cfg.exposure, cast_removal_strength=0.0, wb_cyan=0.3),
    )

    panel.color_section.reset_requested.emit()

    new_config = controller.session.update_config.call_args[0][0]
    assert new_config.exposure.cast_removal_strength == 0.0
    assert new_config.exposure.wb_cyan == 0.0
