import os

import numpy as np
import pytest

from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.sensor import SensorProfiles


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "sensor_dir", str(tmp_path))


def test_list_get_and_none(tmp_path):
    _write(
        os.path.join(tmp_path, "my_sensor.toml"),
        'name = "My Sensor"\nmatrix = [[1.0, -0.1, 0.0], [0.0, 1.1, -0.3], [0.0, -0.3, 1.1]]\n',
    )
    assert SensorProfiles.list_profiles() == ["None", "My Sensor"]
    assert SensorProfiles.get_matrix("My Sensor") == [1.0, -0.1, 0.0, 0.0, 1.1, -0.3, 0.0, -0.3, 1.1]
    assert SensorProfiles.get_matrix("None") is None
    assert SensorProfiles.get_matrix("missing") is None
    assert SensorProfiles.is_bundled("None")
    assert not SensorProfiles.is_bundled("My Sensor")


def test_name_falls_back_to_stem(tmp_path):
    _write(os.path.join(tmp_path, "rig_a.toml"), "matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\n")
    assert "rig_a" in SensorProfiles.list_profiles()


def test_save_round_trips_and_delete(tmp_path):
    matrix = [1.004, -0.02, 0.001, -0.118, 1.01, -0.04, -0.042, -0.149, 1.006]
    path = SensorProfiles.save("Setup A", matrix)
    assert os.path.isfile(path)
    assert path.startswith(str(tmp_path))
    assert SensorProfiles.get_matrix("Setup A") == matrix
    SensorProfiles.delete("Setup A")
    assert SensorProfiles.get_matrix("Setup A") is None
    SensorProfiles.delete("Nonexistent")  # no-op, no raise


def test_malformed_skipped(tmp_path):
    _write(os.path.join(tmp_path, "bad_shape.toml"), "matrix = [[1.0, 0.0], [0.0, 1.0]]\n")
    _write(os.path.join(tmp_path, "bad_toml.toml"), "matrix = [[[not valid\n")
    _write(os.path.join(tmp_path, "no_matrix.toml"), 'name = "x"\n')
    assert SensorProfiles.list_profiles() == ["None"]


def _fake_decode(monkeypatch, captures):
    def fake(self, path):
        return np.tile(np.array(captures[path], dtype=np.float32), (16, 16, 1))

    monkeypatch.setattr("negpy.desktop.view.widgets.sensor_calibration_dialog.SensorCalibrationDialog._decode", fake)


def test_dialog_compute_and_save(tmp_path, monkeypatch, qapp):
    from negpy.desktop.view.widgets.sensor_calibration_dialog import SensorCalibrationDialog

    _fake_decode(monkeypatch, {"R": (0.9, 0.1, 0.03), "G": (0.05, 0.5, 0.15), "B": (0.04, 0.3, 0.95)})
    saved = []
    dlg = SensorCalibrationDialog()
    dlg.profile_saved.connect(saved.append)
    dlg._paths = {"R": "R", "G": "G", "B": "B"}
    dlg.name_edit.setText("Test Sensor")
    dlg._compute_and_save()

    assert saved == ["Test Sensor"]
    m = np.array(SensorProfiles.get_matrix("Test Sensor")).reshape(3, 3)
    # Inverse of the normalized leakage: green/blue off-diagonals go negative.
    assert m[1, 2] < 0 and m[2, 1] < 0
    assert not dlg.result_label.isHidden()
    assert "clipped" not in dlg.result_label.text()


def test_dialog_warns_on_clipped_capture(monkeypatch, qapp):
    from negpy.desktop.view.widgets.sensor_calibration_dialog import SensorCalibrationDialog

    _fake_decode(monkeypatch, {"R": (0.99, 0.1, 0.03), "G": (0.05, 0.5, 0.15), "B": (0.04, 0.3, 1.0)})
    dlg = SensorCalibrationDialog()
    dlg._paths = {"R": "R", "G": "G", "B": "B"}
    dlg.name_edit.setText("Hot")
    dlg._compute_and_save()
    assert "clipped" in dlg.result_label.text()
    assert SensorProfiles.get_matrix("Hot") is not None  # warned, still saved


def test_dialog_reports_error_and_saves_nothing(monkeypatch, qapp):
    from negpy.desktop.view.widgets.sensor_calibration_dialog import SensorCalibrationDialog

    _fake_decode(monkeypatch, {"R": (0.0, 0.1, 0.03), "G": (0.05, 0.5, 0.15), "B": (0.04, 0.3, 1.0)})
    saved = []
    dlg = SensorCalibrationDialog()
    dlg.profile_saved.connect(saved.append)
    dlg._paths = {"R": "R", "G": "G", "B": "B"}
    dlg.name_edit.setText("Bad")
    dlg._compute_and_save()
    assert saved == []
    assert "Could not build" in dlg.result_label.text()
    assert SensorProfiles.list_profiles() == ["None"]


def test_dialog_blocks_reserved_name(qapp):
    from negpy.desktop.view.widgets.sensor_calibration_dialog import SensorCalibrationDialog

    dlg = SensorCalibrationDialog()
    dlg._paths = {"R": "R", "G": "G", "B": "B"}
    dlg.name_edit.setText("None")
    assert not dlg.compute_btn.isEnabled()


def _sensor_sidebar():
    from unittest.mock import MagicMock

    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar.sensor import SensorSidebar

    controller = MagicMock()
    controller.state = AppState()
    return controller, SensorSidebar(controller)


def test_sidebar_lists_profiles(qapp, tmp_path):
    SensorProfiles.save("Rig", [1, 0, 0, 0, 1, 0, 0, 0, 1])
    controller, sidebar = _sensor_sidebar()
    assert [sidebar.sensor_combo.itemText(i) for i in range(sidebar.sensor_combo.count())] == ["None", "Rig"]


def test_sidebar_profile_change_bakes_matrix_and_clears_bounds(qapp, tmp_path):
    from dataclasses import replace

    matrix = [1.0, -0.1, 0.0, 0.0, 1.1, -0.3, 0.0, -0.3, 1.1]
    SensorProfiles.save("Rig", matrix)

    controller, sidebar = _sensor_sidebar()
    controller.state.config = replace(
        controller.state.config,
        process=replace(controller.state.config.process, local_floors=(0.1, 0.1, 0.1)),
    )
    sidebar._on_sensor_profile_changed("Rig")

    cfg = controller.session.update_config.call_args.args[0]
    assert cfg.process.sensor_profile == "Rig"
    assert cfg.process.sensor_matrix == tuple(matrix)
    assert cfg.process.local_floors == (0.0, 0.0, 0.0)

    sidebar._on_sensor_profile_changed("None")
    cfg = controller.session.update_config.call_args.args[0]
    assert cfg.process.sensor_matrix is None


def test_sidebar_sync_rebuilds_combo_after_save(qapp, tmp_path):
    controller, sidebar = _sensor_sidebar()
    assert sidebar.sensor_combo.count() == 1
    SensorProfiles.save("New Rig", [1, 0, 0, 0, 1, 0, 0, 0, 1])
    sidebar.sync_ui()
    assert [sidebar.sensor_combo.itemText(i) for i in range(sidebar.sensor_combo.count())] == ["None", "New Rig"]
