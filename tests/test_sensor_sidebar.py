"""The sensor sidebar builds, and greys itself out when a control can't apply.

A baked matrix is inert while Linear RAW is off, and on any transparency (see
unmix_block_reason), so the panel shows "None" and disables its controls rather than
leaving a profile that looks selected but does nothing. The capture choice greys out the
same way instead of hiding: a hidden sticky setting is one whose state the user cannot see.
"""

import os
import sys
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.sensor import SensorSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.services.assets.sensor import SensorProfiles
from negpy.features.process.models import ProcessMode

if not QApplication.instance():
    _app = QApplication(sys.argv)

_MATRIX = (1.0, -0.1, 0.0, 0.0, 1.1, -0.3, 0.0, -0.3, 1.1)
_NAME = "My Sensor"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "sensor_dir", str(tmp_path))
    with open(os.path.join(tmp_path, "my_sensor.toml"), "w", encoding="utf-8") as f:
        f.write(f'name = "{_NAME}"\nmatrix = [[1.0, -0.1, 0.0], [0.0, 1.1, -0.3], [0.0, -0.3, 1.1]]\n')


def _sidebar(linear_raw=True, profile=_NAME, sensor_matrix=_MATRIX):
    cfg = WorkspaceConfig()
    cfg = replace(
        cfg,
        process=replace(cfg.process, linear_raw=linear_raw, sensor_profile=profile, sensor_matrix=sensor_matrix),
    )
    ctrl = MagicMock()
    ctrl.state.config = cfg
    return SensorSidebar(ctrl)


def test_sidebar_builds_with_all_controls():
    w = _sidebar()
    for attr in ("sensor_combo", "calibrate_sensor_btn", "sensor_hint"):
        assert hasattr(w, attr), attr


def test_profile_is_live_with_linear_raw_on():
    w = _sidebar(linear_raw=True)
    assert w.sensor_combo.currentText() == _NAME
    assert w.sensor_combo.isEnabled()
    assert w.calibrate_sensor_btn.isEnabled()
    # isHidden, not isVisible: the sidebar is never shown here, which would report
    # every child as invisible regardless of its own setVisible state.
    assert w.sensor_hint.isHidden()


def test_panel_greys_out_with_linear_raw_off():
    w = _sidebar(linear_raw=False)
    assert w.sensor_combo.currentText() == SensorProfiles.NONE_NAME
    assert not w.sensor_combo.isEnabled()
    assert w.calibrate_sensor_btn.isEnabled()  # a profile is made from its own shots, not this frame
    assert not w.sensor_hint.isHidden()


def test_hint_shows_even_without_a_baked_profile():
    # Otherwise the controls would be greyed with nothing explaining why.
    w = _sidebar(linear_raw=False, profile=SensorProfiles.NONE_NAME, sensor_matrix=None)
    assert not w.sensor_hint.isHidden()


def _empty_crosstalk_gallery(tmp_path, monkeypatch):
    """No bundled matrices at all, so 'this process has none' is the fixture, not the repo."""
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path / "ct"))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(tmp_path / "_none"))


def test_crosstalk_stays_reachable_with_no_matrices_for_the_process(tmp_path, monkeypatch):
    """The editor is the only way to build one, so hiding the section would leave no route in.
    The empty dropdown and its Strength slider are disabled instead."""
    _empty_crosstalk_gallery(tmp_path, monkeypatch)
    w = _sidebar()
    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    w.sync_ui()

    assert not w.crosstalk_header.isHidden()
    assert not w.manage_crosstalk_btn.isHidden()
    assert w.manage_crosstalk_btn.isEnabled()
    assert not w.crosstalk_combo.isEnabled()
    assert not w.crosstalk_strength_slider.isEnabled()
    assert not w.crosstalk_hint.isHidden()


def test_crosstalk_controls_go_live_once_a_matrix_exists(tmp_path, monkeypatch):
    _empty_crosstalk_gallery(tmp_path, monkeypatch)
    CrosstalkProfiles.save("My Slide Rig", [1.0, -0.05, 0.0, 0.0, 1.0, 0.0, 0.0, -0.05, 1.0], process=ProcessMode.E6)

    w = _sidebar()
    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    w.sync_ui()

    assert w.crosstalk_combo.isEnabled()
    assert w.crosstalk_strength_slider.isEnabled()
    assert w.crosstalk_hint.isHidden()
    assert "My Slide Rig" in w._crosstalk_names()


def test_crosstalk_hidden_on_bw(tmp_path, monkeypatch):
    """One emulsion, nothing to unmix — no editor route needed either."""
    _empty_crosstalk_gallery(tmp_path, monkeypatch)
    w = _sidebar()
    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
    w.sync_ui()

    for widget in (w.crosstalk_header, w.crosstalk_combo, w.manage_crosstalk_btn, w.crosstalk_strength_rail):
        assert widget.isHidden()
    assert w.crosstalk_hint.isHidden()


def test_gate_is_display_only_and_survives_a_round_trip(monkeypatch):
    w = _sidebar(linear_raw=True)
    # Forcing the combo to "None" fires currentTextChanged; if that reached the
    # handler it would persist sensor_profile="None" and lose the user's choice.
    writes = []
    monkeypatch.setattr(w, "update_config_section", lambda *a, **k: writes.append(k))

    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, linear_raw=False))
    w.sync_ui()
    assert w.sensor_combo.currentText() == SensorProfiles.NONE_NAME
    assert writes == []
    assert w.state.config.process.sensor_profile == _NAME
    assert w.state.config.process.sensor_matrix == _MATRIX

    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, linear_raw=True))
    w.sync_ui()
    assert w.sensor_combo.currentText() == _NAME
    assert w.sensor_combo.isEnabled()
    assert w.sensor_hint.isHidden()
    assert writes == []


def _to_mode(w, **process):
    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, **process))
    w.sync_ui()


def _to_rgbscan(w, **rgbscan):
    cfg = w.state.config
    w.state.config = replace(cfg, rgbscan=replace(cfg.rgbscan, **rgbscan))
    w.sync_ui()


def test_unmix_is_refused_on_a_transparency_whatever_linear_raw_says():
    """A sticky profile from a negative rig must not stay live when the frame is a slide."""
    w = _sidebar(linear_raw=True)
    _to_mode(w, process_mode=ProcessMode.E6, positive_source=True)

    assert w.sensor_combo.currentText() == SensorProfiles.NONE_NAME
    assert not w.sensor_combo.isEnabled()
    assert w.calibrate_sensor_btn.isEnabled()  # a profile is made from its own shots, not this frame
    assert not w.sensor_hint.isHidden()
    assert "transparency" in w.sensor_hint.text().lower()


def test_the_profile_comes_back_on_a_negative():
    """Greying is display-only, so the selection survives a film-process round trip."""
    w = _sidebar(linear_raw=True)
    _to_mode(w, process_mode=ProcessMode.E6)
    assert w.state.config.process.sensor_profile == _NAME

    _to_mode(w, process_mode=ProcessMode.C41)
    assert w.sensor_combo.currentText() == _NAME
    assert w.sensor_combo.isEnabled()
    assert w.sensor_hint.isHidden()


def test_a_triplet_names_its_own_reason_even_under_camera_wb():
    # The decode hint names a fix that cannot lift the gate on a triplet; the frame's reason wins.
    w = _sidebar(linear_raw=False)
    _to_rgbscan(w, enabled=True, green_path="/x/g.arw", blue_path="/x/b.arw")

    assert not w.sensor_combo.isEnabled()
    assert "triplet" in w.sensor_hint.text().lower()


def test_profile_is_greyed_out_on_a_trichrome_triplet():
    """The render skips the unmix for a triplet, so a carried-over profile must not read as live."""
    w = _sidebar(linear_raw=True)
    _to_rgbscan(w, enabled=True, green_path="/x/g.arw", blue_path="/x/b.arw")

    assert w.sensor_combo.currentText() == SensorProfiles.NONE_NAME
    assert not w.sensor_combo.isEnabled()
    assert "triplet" in w.sensor_hint.text().lower() and not w.sensor_hint.isHidden()
    assert w.state.config.process.sensor_profile == _NAME  # display-only

    _to_rgbscan(w, enabled=False)
    assert w.sensor_combo.currentText() == _NAME and w.sensor_combo.isEnabled()


@pytest.mark.parametrize(
    ("file_hash", "process", "applied"),
    [
        ("abc", {}, True),
        ("abc", {"linear_raw": False}, True),  # kept like any choice under Linear RAW off
        ("abc", {"process_mode": ProcessMode.E6, "positive_source": True}, False),
        ("", {}, False),
    ],
)
def test_a_saved_profile_reaches_only_a_frame_that_can_keep_it(tmp_path, file_hash, process, applied):
    """Calibration runs with no frame open or on a blocked frame; a slide or no frame only lists the profile."""
    w = _sidebar(linear_raw=True, profile=SensorProfiles.NONE_NAME, sensor_matrix=None)
    _to_mode(w, **process)
    w.state.current_file_hash = file_hash
    with open(os.path.join(tmp_path, "new_sensor.toml"), "w", encoding="utf-8") as f:
        f.write('name = "New Sensor"\nmatrix = [[1.0, 0.0, 0.0], [-0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]\n')
    assert w.sensor_combo.findText("New Sensor") < 0
    w.controller.reset_mock()

    w._on_sensor_profile_saved("New Sensor")

    assert w.sensor_combo.findText("New Sensor") >= 0  # the save rebuilds the list
    if applied:
        w.controller.set_roll_default.assert_called_once()
        kwargs = w.controller.set_roll_default.call_args.kwargs
        assert kwargs["sensor_profile"] == "New Sensor"
        assert kwargs["sensor_matrix"] == (1.0, 0.0, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert not w.controller.set_status.called
    else:
        assert not w.controller.set_roll_default.called
        assert "New Sensor" in w.controller.set_status.call_args.args[0]  # says it was saved
