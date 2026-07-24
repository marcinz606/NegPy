from PyQt6.QtWidgets import QComboBox, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label
from negpy.features.process.models import invalidate_local_bounds
from negpy.services.assets.sensor import SensorProfiles


class SensorSidebar(BaseSidebar):
    """
    Panel for sensor-crosstalk calibration (single-shot narrowband scans).
    """

    def _init_ui(self) -> None:
        conf = self.state.config.process

        row = QHBoxLayout()
        self.sensor_label = field_label("Profile")
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(SensorProfiles.list_profiles())
        self.sensor_combo.setCurrentText(conf.sensor_profile)
        self.sensor_combo.setToolTip(
            "<table width='280'><tr><td>"
            "Sensor crosstalk correction for single-shot narrowband scans: un-mixes the camera's "
            "cross-channel response in the LINEAR capture, before inversion — a fixed property of "
            "your sensor + light, independent of film. Calibrate it from three bare-light R/G/B "
            "exposures; custom .toml matrices live in the NegPy/sensor folder. Skipped automatically "
            "for RGB-triplet assets. Re-run Batch Analysis after changing this."
            "</td></tr></table>"
        )
        self.calibrate_sensor_btn = self._icon_action("fa5s.vials", "Calibrate the sensor from three bare-light R/G/B exposures", width=32)
        row.addWidget(self.sensor_label)
        row.addWidget(self.sensor_combo, 1)
        row.addWidget(self.calibrate_sensor_btn)
        self.layout.addLayout(row)

    def _connect_signals(self) -> None:
        self.sensor_combo.currentTextChanged.connect(self._on_sensor_profile_changed)
        self.calibrate_sensor_btn.clicked.connect(self._open_sensor_calibration)

    def _on_sensor_profile_changed(self, name: str) -> None:
        # Bake the matrix like crosstalk does; the per-frame bounds were analyzed
        # under the previous mix, so clear them.
        matrix = SensorProfiles.get_matrix(name)
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            sensor_profile=name,
            sensor_matrix=tuple(matrix) if matrix is not None else None,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _open_sensor_calibration(self) -> None:
        from negpy.desktop.view.widgets.sensor_calibration_dialog import SensorCalibrationDialog

        dlg = SensorCalibrationDialog(parent=self)
        dlg.profile_saved.connect(self._on_sensor_profile_saved)
        dlg.exec()

    def _on_sensor_profile_saved(self, name: str) -> None:
        self._on_sensor_profile_changed(name)
        self.sync_ui()  # rebuild the combo (now includes the new profile) and select it

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            profiles = SensorProfiles.list_profiles()
            if profiles != [self.sensor_combo.itemText(i) for i in range(self.sensor_combo.count())]:
                self.sensor_combo.clear()
                self.sensor_combo.addItems(profiles)
            self.sensor_combo.setCurrentText(conf.sensor_profile)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        self.sensor_combo.blockSignals(blocked)
