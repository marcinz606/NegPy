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

import numpy as np
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


def _fade_sidebar(**process):
    cfg = WorkspaceConfig()
    ctrl = MagicMock()
    ctrl.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, **process))
    ctrl.session.repo.get_global_setting.return_value = "midtone"
    ctrl.state.current_file_hash = "frame-a"
    w = SensorSidebar(ctrl)
    w.state = ctrl.state
    return w, ctrl


def test_a_failed_estimate_keeps_the_survivals_and_shows_why():
    w, ctrl = _fade_sidebar(fade_ratio_g=0.6, fade_ratio_b=0.7)
    from negpy.desktop.workers.chart_fit import ChartSource

    flat = np.full((32, 32, 3), 0.5, dtype=np.float32)  # no grays to read
    ctrl.chart_fit_source.return_value = ChartSource(flat, ctrl.state.config, "flat", None, None, (), 1)

    w._on_estimate_fade()

    ctrl.set_roll_default.assert_not_called()
    assert w.fade_ratio_g_slider.value() == pytest.approx(0.6)
    assert w.fade_ratio_b_slider.value() == pytest.approx(0.7)
    assert w.fade_estimate_hint.text()
    assert not w.fade_estimate_hint.isHidden()


def _stained_slide():
    from tests.test_fade_fixtures import make_fixture

    fx = make_fixture("stained", 0.8, 0.6, (0.0, -0.05, -0.4), seed=3)
    return np.power(10.0, fx.faded_log).astype(np.float32)


def _open(ctrl, image):
    """The frame Estimate renders: the controller's chart source over this image."""
    from negpy.desktop.workers.chart_fit import ChartSource

    ctrl.chart_fit_source.return_value = ChartSource(image, ctrl.state.config, "frame", None, None, (), 1)


def test_an_estimate_on_a_stained_slide_suggests_remove_stain_and_stores_no_white():
    # No analysis border: the fixture's clear end is its left edge.
    w, ctrl = _fade_sidebar(analysis_buffer=0.0)
    _open(ctrl, _stained_slide())

    w._on_estimate_fade()

    assert "Remove Stain" in w.fade_stain_hint.text()
    assert "Remove Stain" not in w.fade_estimate_hint.text()
    assert w.fade_white_anchor_btn.property("suggested") is True
    stored = ctrl.set_roll_default.call_args.kwargs
    assert "fade_white_b" not in stored
    assert not w.fade_white_anchor_btn.isChecked()


def test_turning_on_a_suggested_remove_stain_estimates_again_with_it():
    w, ctrl = _fade_sidebar(analysis_buffer=0.0)
    _open(ctrl, _stained_slide())
    w._on_estimate_fade()

    w.fade_white_anchor_btn.setChecked(True)

    calls = [c.kwargs for c in ctrl.set_roll_default.call_args_list[-2:]]
    assert calls[0] == {"fade_white_anchor": True}
    assert calls[1]["fade_white_b"] == pytest.approx(-0.4, abs=0.02)
    assert calls[1]["fade_ratio_b"] == pytest.approx(0.6, abs=0.03)
    assert w.fade_stain_hint.text() == ""
    assert w.fade_white_anchor_btn.property("suggested") is False


def test_turning_on_remove_stain_unsuggested_estimates_too():
    w, ctrl = _fade_sidebar(analysis_buffer=0.0)
    _open(ctrl, _stained_slide())

    w.fade_white_anchor_btn.setChecked(True)

    assert ctrl.set_roll_default.call_args_list[0].kwargs == {"fade_white_anchor": True}
    assert ctrl.set_roll_default.call_args.kwargs["fade_white_b"] == pytest.approx(-0.4, abs=0.02)


def test_turning_off_remove_stain_estimates_again_without_it():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar(fade_white_anchor=True, fade_ratio_b=0.8)
    w.fade_white_anchor_btn.blockSignals(True)
    w.fade_white_anchor_btn.setChecked(True)
    w.fade_white_anchor_btn.blockSignals(False)
    with patch.object(w, "_run_estimate") as run:
        w.fade_white_anchor_btn.setChecked(False)
    run.assert_called_once_with(white_anchor=False)


def test_turning_off_remove_stain_before_an_estimate_runs_none():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar(fade_white_anchor=True)
    w.fade_white_anchor_btn.blockSignals(True)
    w.fade_white_anchor_btn.setChecked(True)
    w.fade_white_anchor_btn.blockSignals(False)
    with patch.object(w, "_run_estimate") as run:
        w.fade_white_anchor_btn.setChecked(False)
    run.assert_not_called()


def test_turning_on_remove_stain_with_no_stain_says_so():
    w, ctrl = _fade_sidebar()
    _open(ctrl, np.full((32, 32, 3), 0.5, dtype=np.float32))

    w.fade_white_anchor_btn.setChecked(True)

    assert ctrl.set_roll_default.call_args.kwargs == {"fade_white_anchor": True}
    assert "No stain found to remove." in w.fade_estimate_hint.text()


def test_a_failed_estimate_on_a_camera_raw_stores_the_white_without_a_roll_lock():
    from negpy.desktop.workers.chart_fit import ChartSource

    w, ctrl = _fade_sidebar()
    camera = [[0.6722, -0.0635, -0.0963], [-0.4287, 1.246, 0.2046], [-0.0908, 0.2268, 0.5762]]
    flat = np.full((32, 32, 3), 0.5, dtype=np.float32)
    ctrl.chart_fit_source.return_value = ChartSource(flat, ctrl.state.config, "frame", camera, [2.0, 1.0, 1.5], (), 1)

    w._on_estimate_fade()

    ctrl.set_roll_default.assert_not_called()
    assert ctrl.set_fade_white.call_args.kwargs["fade_white_space"]


def test_the_estimate_row_holds_auto_red_and_remove_stain_and_folds_the_survivals():
    w, _ctrl = _fade_sidebar()
    row = w.fade_estimate_row
    widgets = [row.itemAt(i).widget() for i in range(row.count())]
    assert widgets == [w.fade_survival_fold.button, w.estimate_fade_btn, w.fade_auto_red_btn, w.fade_white_anchor_btn]
    assert w.fade_auto_red_btn.isCheckable()
    assert not w.fade_survival_fold.is_open()
    rail = w.fade_survival_fold.content
    assert all(s.parent() is rail for s in (w.fade_ratio_r_slider, w.fade_ratio_g_slider, w.fade_ratio_b_slider))


def test_an_estimate_with_remove_stain_on_stores_the_white_with_the_survivals():
    w, ctrl = _fade_sidebar(fade_white_anchor=True, analysis_buffer=0.0)
    _open(ctrl, _stained_slide())

    w._on_estimate_fade()

    stored = ctrl.set_roll_default.call_args.kwargs
    assert stored["fade_white_b"] == pytest.approx(-0.4, abs=0.02)
    assert stored["fade_ratio_b"] == pytest.approx(0.6, abs=0.03)
    assert w.fade_stain_hint.text() == ""


def test_a_failed_estimate_with_remove_stain_on_stores_no_white_off_clear_film():
    w, ctrl = _fade_sidebar(fade_white_anchor=True)
    _open(ctrl, np.full((32, 32, 3), 0.5, dtype=np.float32))  # no grays, and no white near full scale

    w._on_estimate_fade()

    ctrl.set_roll_default.assert_not_called()


def test_turning_remove_stain_off_clears_its_hint():
    w, ctrl = _fade_sidebar(analysis_buffer=0.0)
    _open(ctrl, _stained_slide())
    w.fade_white_anchor_btn.setChecked(True)
    assert w.fade_estimate_hint.text()

    w.fade_white_anchor_btn.setChecked(False)

    assert w.fade_estimate_hint.text() == ""


def test_the_estimate_hint_stays_with_its_frame():
    w, ctrl = _fade_sidebar(analysis_buffer=0.0)
    _open(ctrl, _stained_slide())
    w._on_estimate_fade()
    w.sync_ui()
    assert "Remove Stain" in w.fade_stain_hint.text()

    ctrl.state.current_file_hash = "frame-b"
    w.sync_ui()

    assert w.fade_estimate_hint.text() == ""
    assert w.fade_stain_hint.text() == ""
    assert w.fade_white_anchor_btn.property("suggested") is False


def test_remove_stain_is_a_fade_roll_default():
    w, ctrl = _fade_sidebar()
    ctrl.chart_fit_source.return_value = None
    w.fade_white_anchor_btn.setChecked(True)
    args = ctrl.set_roll_default.call_args
    assert args.args[0] == "fade" and args.kwargs["fade_white_anchor"] is True
    assert w.fade_estimate_hint.text() == "No image loaded to estimate from."


@pytest.mark.parametrize(
    ("mode", "positive", "shown"), [(ProcessMode.E6, False, True), (ProcessMode.E6, True, True), (ProcessMode.C41, False, False)]
)
def test_fade_restoration_shows_only_on_a_slide(mode, positive, shown):
    w, _ctrl = _fade_sidebar()
    w.state.config = replace(w.state.config, process=replace(w.state.config.process, process_mode=mode, positive_source=positive))
    w.sync_ui()
    assert w.fade_fold.content.isHidden() is not shown
    assert w.fade_header.isHidden() is not shown


def _slide_sidebar(**process):
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, **process))
    ctrl = MagicMock()
    ctrl.state.config = cfg
    ctrl.state.preview_cam_xyz = None
    ctrl.state.preview_camera_wb = None
    ctrl.state.last_metrics = {}
    ctrl.roll_crossover_table.return_value = None
    w = SensorSidebar(ctrl)
    w.sync_ui()
    return w, ctrl


_TABLE = (-1.5, -2.5, 0.2, 0.1, 0.2, -0.3, -1.5, 0.3, 0.0, 0.0, 0.0, -0.5, 0.3, -0.1, -0.1, 0.2)


def test_crossover_waits_for_a_table():
    w, _ = _slide_sidebar()
    assert not w.fade_crossover_slider.isEnabled()
    assert "Estimate" in w.fade_crossover_hint.text()
    w, _ = _slide_sidebar(fade_crossover_table=_TABLE)
    assert w.fade_crossover_slider.isEnabled()
    assert "Crossover" in w.fade_crossover_hint.text()


def test_the_hint_names_the_shadow_and_highlight_casts():
    w, _ = _slide_sidebar(fade_crossover_table=_TABLE)
    text = w.fade_crossover_hint.text()
    assert "Shadows lean" in text and "Highlights lean" in text


def test_the_crossover_slider_stores_a_fraction_on_the_fade_card():
    w, ctrl = _slide_sidebar(fade_crossover_table=_TABLE)
    w.fade_crossover_slider.valueCommitted.emit(40.0)
    ctrl.set_roll_default.assert_called_with("fade", persist=True, readback_metrics=True, fade_crossover=0.4)


def test_neutrals_waits_on_an_estimate():
    w, _ = _slide_sidebar()
    assert not w.fade_neutrals_mix_slider.isHidden() and not w.fade_neutrals_mix_slider.isEnabled()
    w, _ = _slide_sidebar(fade_crossover_table=_TABLE)
    assert w.fade_neutrals_mix_slider.isEnabled()
    w, _ = _slide_sidebar(fade_ratio_b=0.8)
    assert w.fade_neutrals_mix_slider.isEnabled()


def test_the_mix_slider_stores_a_fraction_on_the_fade_card():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar(fade_crossover_table=_TABLE)
    with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=None):
        w.fade_neutrals_mix_slider.valueCommitted.emit(30.0)
    ctrl.set_roll_default.assert_called_with("fade", persist=True, readback_metrics=False, fade_neutrals_mix=0.3)


def test_the_mix_slider_does_nothing_before_an_estimate():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar()
    with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()):
        w.fade_neutrals_mix_slider.valueCommitted.emit(30.0)
    ctrl.set_roll_default.assert_not_called()


def test_estimate_reads_the_mix():
    from unittest.mock import patch

    from negpy.features.process import fade

    w, _ = _slide_sidebar(fade_neutrals_mix=0.4)
    with (
        patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=np.zeros((8, 8, 3), np.float32)),
        patch.object(fade, "blend_gray_readings", wraps=fade.blend_gray_readings) as blend,
    ):
        w._run_estimate(False)
    assert blend.call_args.args[1] == 0.4


def _mix_grid():
    from fade_crossover_scene import CROSSOVER, build_scene, estimate_grid, scan_image

    return estimate_grid(scan_image(CROSSOVER, build_scene(96, 72)))


def test_a_mix_drag_estimates_live_and_reads_the_grays_once():
    from unittest.mock import patch

    from negpy.features.process import fade

    w, ctrl = _slide_sidebar(fade_crossover_table=_TABLE)
    with (
        patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()),
        patch.object(fade, "gray_readings", wraps=fade.gray_readings) as readings,
    ):
        for v in (20.0, 40.0, 60.0):
            w.fade_neutrals_mix_slider.valueChanged.emit(v)
    assert readings.call_count == 1
    calls = [c for c in ctrl.set_roll_default.call_args_list if "fade_ratio_b" in c.kwargs]
    assert [c.kwargs["fade_neutrals_mix"] for c in calls] == [0.2, 0.4, 0.6]
    assert all(c.kwargs["persist"] is False and "fade_crossover_table" not in c.kwargs for c in calls)
    assert len({c.kwargs["fade_ratio_b"] for c in calls}) == 3


def test_estimate_on_a_frame_without_negpy_grays_jumps_to_nc_s():
    from unittest.mock import patch

    from negpy.features.process import fade

    w, ctrl = _slide_sidebar()
    with (
        patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()),
        patch.object(fade, "neutral_axis_from_grid", return_value=(None, "No grays.")),
    ):
        w._on_estimate_fade()
        kwargs = ctrl.set_roll_default.call_args.kwargs
        assert kwargs["fade_neutrals_mix"] == 1.0 and kwargs["fade_ratio_b"] != 1.0
        assert w.fade_neutrals_mix_slider.value() == 100.0
        assert w.fade_estimate_hint.text().startswith("This frame has few clear grays.")
        # Back at 0 the survivals stay at 1, and the slider stays open on this frame.
        w.fade_neutrals_mix_slider.valueChanged.emit(0.0)
        kwargs = ctrl.set_roll_default.call_args.kwargs
        assert (kwargs["fade_ratio_r"], kwargs["fade_ratio_g"], kwargs["fade_ratio_b"]) == (1.0, 1.0, 1.0)
        assert w._neutrals_open(ctrl.state.config.process)


def test_a_mix_release_keeps_the_frame_table():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar(fade_crossover_table=_TABLE)
    with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()):
        w.fade_neutrals_mix_slider.valueCommitted.emit(50.0)
    kwargs = ctrl.set_roll_default.call_args.kwargs
    assert kwargs["persist"] is True and kwargs["fade_neutrals_mix"] == 0.5
    assert "fade_crossover_table" not in kwargs


def test_a_mix_release_measures_no_table_on_a_frame_without_one():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar(fade_ratio_b=0.8)
    with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()):
        w.fade_neutrals_mix_slider.valueCommitted.emit(50.0)
    assert "fade_crossover_table" not in ctrl.set_roll_default.call_args.kwargs


def test_a_survival_edit_keeps_the_frame_table():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar(fade_crossover_table=_TABLE)
    with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()):
        w._on_fade_ratio_r_changed(0.8)
        w._on_fade_ratio_changed(0.9, 0.8)
    for call in ctrl.set_roll_default.call_args_list:
        assert "fade_crossover_table" not in call.kwargs


def test_an_estimate_press_measures_a_new_table():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar(fade_crossover_table=_TABLE)
    with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()):
        w._on_estimate_fade()
    assert ctrl.set_roll_default.call_args.kwargs["fade_crossover_table"] != _TABLE


def test_the_hint_says_whose_table_the_frame_uses():
    w, ctrl = _slide_sidebar(fade_crossover_table=_TABLE)
    assert not w.fade_crossover_hint.text().startswith("Table from")
    ctrl.roll_crossover_table.return_value = _TABLE
    w.sync_ui()
    assert w.fade_crossover_hint.text().startswith("Table from the roll.")
    ctrl.roll_crossover_table.return_value = tuple(v + 0.1 for v in _TABLE)
    w.sync_ui()
    assert w.fade_crossover_hint.text().startswith("Table from this frame.")


def test_estimate_picks_the_neutrals_that_leave_the_grays_most_neutral():
    from unittest.mock import patch

    from negpy.features.process import fade

    grid = _mix_grid()
    readings = fade.gray_readings(grid)
    spreads = [
        fade.neutral_spread(fade.estimate_fade_with(grid, lambda _, m=m: fade.blend_gray_readings(readings, m)).crossover)
        for m in fade.NEUTRALS_CANDIDATES
    ]
    w, ctrl = _slide_sidebar()
    with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=grid):
        w._on_estimate_fade()
    kwargs = ctrl.set_roll_default.call_args.kwargs
    assert kwargs["fade_neutrals_mix"] == fade.NEUTRALS_CANDIDATES[int(np.argmin(spreads))]
    assert w.fade_neutrals_mix_slider.value() == kwargs["fade_neutrals_mix"] * 100.0
    assert w.fade_estimate_hint.text().startswith("Gray Mix set to")
    assert "fade_crossover_table" in kwargs and "fade_crossover" not in kwargs


def test_estimate_suggests_crossover_only_while_it_is_off():
    from unittest.mock import patch

    from negpy.features.process import fade

    for crossover, suggested in ((0.0, True), (0.5, False)):
        w, ctrl = _slide_sidebar(fade_crossover=crossover)
        with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()):
            w._on_estimate_fade()
        kwargs = ctrl.set_roll_default.call_args.kwargs
        assert fade.crossover_amount(kwargs["fade_crossover_table"]) >= fade.CROSSOVER_FLOOR
        assert "fade_crossover" not in kwargs
        assert ("raise Crossover" in w.fade_estimate_hint.text()) is suggested


def test_estimate_suggests_auto_red_in_amber_on_a_short_range_while_it_is_off():
    from unittest.mock import patch

    for auto, short, suggested in ((False, True, True), (False, False, False), (True, True, False)):
        w, ctrl = _slide_sidebar(fade_auto_red=auto)
        with (
            patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()),
            patch.object(type(w), "_short_range", return_value=short),
        ):
            w._on_estimate_fade()
        assert ("Restore Range" in w.fade_red_hint.text()) is suggested
        assert bool(w.fade_auto_red_btn.property("suggested")) is suggested


def _apply(ctrl):
    kwargs = ctrl.set_roll_default.call_args.kwargs
    fields = {k: v for k, v in kwargs.items() if k.startswith("fade_")}
    ctrl.state.config = replace(ctrl.state.config, process=replace(ctrl.state.config.process, **fields))
    return kwargs


def test_auto_red_on_fits_red_and_estimate_keeps_it_and_off_puts_estimate_s_red_back():
    from unittest.mock import patch

    from negpy.features.process.fade import estimate_red

    w, ctrl = _slide_sidebar()
    fit = ({"fade_ratio_r": 0.6, "fade_density_shift": -0.2}, "")
    with (
        patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=_mix_grid()),
        patch.object(type(w), "_auto_red_changes", return_value=fit),
    ):
        w._on_estimate_fade()
        estimated = _apply(ctrl)
        w.fade_auto_red_btn.setChecked(True)
        on = _apply(ctrl)
        assert on["fade_auto_red"] is True and on["fade_ratio_r"] != estimated["fade_ratio_r"]
        w._on_estimate_fade()
        again = _apply(ctrl)
        assert (again["fade_ratio_r"], again["fade_density_shift"]) == (0.6, -0.2)
        assert again["fade_ratio_r"] != estimate_red(again["fade_ratio_g"], again["fade_ratio_b"])
        assert "fade_crossover_table" in again
        w.fade_auto_red_btn.setChecked(False)
    off = _apply(ctrl)
    assert off["fade_auto_red"] is False and off["fade_density_shift"] == 0.0
    assert off["fade_ratio_r"] == estimate_red(again["fade_ratio_g"], again["fade_ratio_b"])


def test_auto_red_with_nothing_to_read_stays_off():
    from unittest.mock import patch

    w, ctrl = _slide_sidebar()
    with patch("negpy.desktop.workers.chart_fit.fade_estimate_grid", return_value=None):
        w.fade_auto_red_btn.setChecked(True)
    assert not w.fade_auto_red_btn.isChecked()
    ctrl.set_roll_default.assert_not_called()
    assert w.fade_red_hint.text()


def test_sync_shows_the_frame_s_auto_red():
    w, _ = _slide_sidebar(fade_auto_red=True)
    assert w.fade_auto_red_btn.isChecked()


def test_a_hand_red_turns_auto_red_off():
    w, ctrl = _slide_sidebar(fade_auto_red=True)
    w.fade_ratio_r_slider.valueCommitted.emit(0.7)
    kwargs = ctrl.set_roll_default.call_args.kwargs
    assert kwargs["fade_ratio_r"] == 0.7 and kwargs["fade_auto_red"] is False
    assert kwargs["fade_density_shift"] == 0.0
