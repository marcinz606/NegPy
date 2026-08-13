from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel


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
    controller.state.config = replace(cfg, flatfield=replace(cfg.flatfield, apply=True, k1=-0.02))
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
