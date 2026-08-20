"""Drag ticks from the Process/Sensor sliders must not request metrics readback.

A tick with readback_metrics=True is treated as a settled frame: metrics passes,
a blocking map_sync, a thumbnail readback and a full right-panel refresh.
"""

from unittest.mock import MagicMock

from negpy.desktop.view.sidebar.process import ProcessSidebar
from negpy.desktop.view.sidebar.sensor import SensorSidebar


def _readback_kwarg(panel) -> bool:
    return panel.update_config_section.call_args.kwargs["readback_metrics"]


def _process_stub() -> MagicMock:
    panel = MagicMock()
    panel._wp_field.return_value = "white_point_offset"
    panel._bp_field.return_value = "black_point_offset"
    panel.state.config.process = MagicMock(lock_bounds=False)
    return panel


def test_process_sliders_follow_persist() -> None:
    for handler in (
        ProcessSidebar._on_white_point_changed,
        ProcessSidebar._on_black_point_changed,
        ProcessSidebar._on_buffer_changed,
        ProcessSidebar._on_luma_range_clip_changed,
        ProcessSidebar._on_color_range_clip_changed,
    ):
        for persist in (False, True):
            panel = _process_stub()
            handler(panel, 0.1, persist=persist)
            assert _readback_kwarg(panel) is persist, handler.__name__


def test_sensor_sliders_follow_persist() -> None:
    for handler in (
        SensorSidebar._on_crosstalk_strength_changed,
        SensorSidebar._on_hue_trim_changed,
    ):
        for persist in (False, True):
            panel = MagicMock()
            panel.state.config.process = MagicMock(lock_bounds=False)
            handler(panel, 0.1, persist=persist)
            assert _readback_kwarg(panel) is persist, handler.__name__
