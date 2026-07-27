"""The sensor sidebar builds, and surfaces the Linear RAW gate.

A baked matrix is inert while Linear RAW is off (effective_sensor_matrix returns
None there), so the panel must say so rather than silently doing nothing.
"""

import sys
from dataclasses import replace
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QApplication

from negpy.domain.models import WorkspaceConfig
from negpy.desktop.view.sidebar.sensor import SensorSidebar

if not QApplication.instance():
    _app = QApplication(sys.argv)

_MATRIX = (1.0, -0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _sidebar(linear_raw=True, sensor_matrix=None):
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, linear_raw=linear_raw, sensor_matrix=sensor_matrix))
    ctrl = MagicMock()
    ctrl.state.config = cfg
    return SensorSidebar(ctrl)


def test_sidebar_builds_with_all_controls():
    w = _sidebar()
    for attr in ("sensor_combo", "calibrate_sensor_btn", "linear_raw_hint"):
        assert hasattr(w, attr), attr


def test_hint_only_when_a_matrix_is_inert():
    # isHidden, not isVisible: the sidebar is never shown here, which would report
    # every child as invisible regardless of its own setVisible state.
    assert _sidebar(linear_raw=True, sensor_matrix=_MATRIX).linear_raw_hint.isHidden()
    assert _sidebar(linear_raw=False, sensor_matrix=None).linear_raw_hint.isHidden()
    assert not _sidebar(linear_raw=False, sensor_matrix=_MATRIX).linear_raw_hint.isHidden()


def test_sync_ui_tracks_linear_raw_toggle():
    w = _sidebar(linear_raw=True, sensor_matrix=_MATRIX)
    assert w.linear_raw_hint.isHidden()

    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, linear_raw=False))
    w.sync_ui()
    assert not w.linear_raw_hint.isHidden()

    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, linear_raw=True))
    w.sync_ui()
    assert w.linear_raw_hint.isHidden()
