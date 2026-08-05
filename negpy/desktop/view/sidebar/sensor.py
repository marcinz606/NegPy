from PyQt6.QtWidgets import QComboBox, QDialog, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label, section_subheader, wrap_tooltip
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.process.models import ProcessMode, invalidate_local_bounds
from negpy.features.process.sensor import sensor_unmix_available
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.services.assets.sensor import SensorProfiles


class SensorSidebar(BaseSidebar):
    """
    Capture-side colour corrections, one per cause and not interchangeable: the
    camera's cross-channel response (linear capture), the film's dye absorptions
    (negative density), and an odd light spectrum's hue rotation (the print).
    """

    def _init_ui(self) -> None:
        conf = self.state.config.process

        self.layout.addWidget(section_subheader("TRICHROME CALIBRATION"))

        row = QHBoxLayout()
        self.sensor_label = field_label("Profile")
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(SensorProfiles.list_profiles())
        self.sensor_combo.setToolTip(
            "<table width='280'><tr><td>"
            "Sensor crosstalk correction for single-shot narrowband scans: un-mixes the camera's "
            "cross-channel response in the LINEAR capture, before inversion — a fixed property of "
            "your sensor + light, independent of film. Calibrate it from three bare-light R/G/B "
            "exposures; custom .toml matrices live in the NegPy/sensor folder. Skipped automatically "
            "for RGB-triplet assets and when Linear RAW is off. Re-run Batch Analysis after changing this."
            "</td></tr></table>"
        )
        self.calibrate_sensor_btn = self._icon_action("fa5s.vials", "Calibrate the sensor from three bare-light R/G/B exposures", width=32)
        row.addWidget(self.sensor_label)
        row.addWidget(self.sensor_combo, 1)
        row.addWidget(self.calibrate_sensor_btn)
        self.layout.addLayout(row)

        # Muted, not warning: this is the normal state for anyone not using Linear
        # RAW, so it explains the greyed controls rather than flagging a problem.
        self.linear_raw_hint = hint_label("Requires Linear RAW.")
        # Disabled widgets don't receive the hover that raises a tooltip, so the
        # detail hangs off this label rather than the combo it describes.
        self.linear_raw_hint.setToolTip(
            wrap_tooltip(
                "Sensor profiles are calibrated against neutral white balance. With Linear RAW "
                "off, RAW decodes carry the camera's as-shot gains instead, which would misapply "
                "the matrix — so it is skipped. Your selection is remembered."
            )
        )
        self.layout.addWidget(self.linear_raw_hint)

        self.crosstalk_header = section_subheader("CROSSTALK")
        self.layout.addWidget(self.crosstalk_header)

        matrix_row = QHBoxLayout()
        self.crosstalk_label = field_label("Matrix")
        self.crosstalk_combo = QComboBox()
        self._fill_crosstalk_combo()
        self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
        # Wrap the long tooltip in a fixed-width table so Qt word-wraps it to the
        # panel width instead of rendering one line that runs off the screen (plain
        # text tooltips are not auto-wrapped — only rich text is).
        self.crosstalk_combo.setToolTip(
            "<table width='280'><tr><td>"
            "Channel unmix on the raw NEGATIVE densities, before analysis and inversion — the domain "
            "where every cause of channel mixing is linear. The film's dyes absorb outside their own "
            "band, but so do your light's spectrum and your sensor's colour filters, and here they all "
            "arrive as the same kind of error. So read a profile as <b>your whole scanning setup</b>, "
            "not just the stock.<br><br>"
            "<b>The bundled film matrices are read off published spec sheets, not measured</b> — they "
            "are marked (approx) for that reason. They describe the film's dyes alone, so they are only "
            "the whole story where the capture reads each dye cleanly: a true RGB scan (a Coolscan-style "
            "mono sensor lit one band at a time) or a calibrated trichrome rig. Under a broadband light "
            "and a Bayer sensor your capture adds its own mixing on top, and a dyes-only matrix will not "
            "describe it.<br><br>"
            "So treat them as starting points and expect to tune: raise Strength until colours separate "
            "without going garish, and if a stock or a light gives you trouble, open the editor, nudge "
            "the six off-diagonal terms and save your own profile — name it after the combination "
            "('Gold 200 + Spectracolor'). A profile measured on your own rig beats any datasheet. "
            "Custom .toml matrices live in the NegPy/crosstalk folder (see docs/CROSSTALK.md).<br><br>"
            "Re-run Batch Analysis after changing this."
            "</td></tr></table>"
        )
        self.manage_crosstalk_btn = self._icon_action(
            "fa5s.sliders-h", "Open the crosstalk matrix editor — view, copy and edit density-unmix profiles", width=32
        )
        matrix_row.addWidget(self.crosstalk_label)
        matrix_row.addWidget(self.crosstalk_combo, 1)
        matrix_row.addWidget(self.manage_crosstalk_btn)
        self.layout.addLayout(matrix_row)

        self.crosstalk_strength_slider = CompactSlider("Strength", 0.0, 1.0, conf.crosstalk_strength, has_neutral=True)
        self.layout.addWidget(self.crosstalk_strength_slider)

        self.layout.addWidget(section_subheader("LIGHT SOURCE"))

        self.hue_trim_slider = CompactSlider("Hue Trim", -30.0, 30.0, conf.hue_trim, step=0.5, precision=10, has_neutral=True, unit="°")
        self.hue_trim_slider.setToolTip(
            "<table width='280'><tr><td>"
            "Hue Trim — rotates every hue by a fixed angle (degrees) to undo the rotation an unusual "
            "scanning light imposes. Narrowband LED and odd-phosphor sources shift hues by a near-constant "
            "angle (yellows reading orange, greens olive) that white balance cannot fix, because it is a "
            "rotation rather than a cast. Neutrals are unaffected, so it does not disturb cast removal. "
            "Leave at 0 for a standard broadband light."
            "</td></tr></table>"
        )
        self.layout.addWidget(self.hue_trim_slider)

        self._apply_gate(conf)

    @staticmethod
    def _heading_row(heading: str) -> str:
        return f"— {heading} —"

    def _expected_crosstalk_rows(self) -> list:
        """The rows _fill_crosstalk_combo would produce, for change detection."""
        rows: list = []
        for heading, names in CrosstalkProfiles.grouped_profiles():
            rows.append(self._heading_row(heading))
            rows.extend(names)
        return rows

    def _fill_crosstalk_combo(self) -> None:
        """Rebuild the matrix dropdown with a non-selectable heading per provenance group.

        Qt has no group concept, so headings are combo rows disabled through the model.
        Bracketing them keeps a heading from colliding with a profile name, which would let
        setCurrentText land on one.
        """
        self.crosstalk_combo.clear()
        for heading, names in CrosstalkProfiles.grouped_profiles():
            self.crosstalk_combo.addItem(self._heading_row(heading))
            item = self.crosstalk_combo.model().item(self.crosstalk_combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
            for name in names:
                self.crosstalk_combo.addItem(name)

    def _crosstalk_names(self) -> list:
        """Selectable profile names currently in the combo, headings excluded."""
        model = self.crosstalk_combo.model()
        return [
            self.crosstalk_combo.itemText(i)
            for i in range(self.crosstalk_combo.count())
            if model.item(i) is None or model.item(i).isEnabled()
        ]

    def _apply_gate(self, conf) -> None:
        """Grey the sensor unmix and show "None" while it cannot be applied.

        Display-only: conf.sensor_profile is left alone, so the selection comes back intact
        after a Linear RAW round-trip. Crosstalk and Hue Trim do not depend on the decode
        basis, so they stay enabled.
        """
        available = sensor_unmix_available(conf)
        self.sensor_combo.setCurrentText(conf.sensor_profile if available else SensorProfiles.NONE_NAME)
        self.sensor_combo.setEnabled(available)
        self.calibrate_sensor_btn.setEnabled(available)
        self.linear_raw_hint.setVisible(not available)

    def _connect_signals(self) -> None:
        self.sensor_combo.currentTextChanged.connect(self._on_sensor_profile_changed)
        self.calibrate_sensor_btn.clicked.connect(self._open_sensor_calibration)

        self.crosstalk_combo.currentTextChanged.connect(self._on_crosstalk_profile_changed)
        self.manage_crosstalk_btn.clicked.connect(self._open_crosstalk_editor)
        self.crosstalk_strength_slider.valueChanged.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=False))
        self.crosstalk_strength_slider.valueCommitted.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=True))

        self.hue_trim_slider.valueChanged.connect(lambda v: self._on_hue_trim_changed(v, persist=False))
        self.hue_trim_slider.valueCommitted.connect(lambda v: self._on_hue_trim_changed(v, persist=True))

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

    def _on_crosstalk_profile_changed(self, name: str) -> None:
        # Bake the matrix into the config so saved edits stay reproducible if the
        # profile file is later moved/deleted. The persisted per-frame bounds were
        # analyzed under the previous matrix — clear them so the stretch re-derives
        # from the unmixed data (otherwise the mask redistribution leaks through).
        matrix = CrosstalkProfiles.get_matrix(name)
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            crosstalk_profile=name,
            crosstalk_matrix=matrix,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_crosstalk_strength_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            crosstalk_strength=val,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _open_crosstalk_editor(self) -> None:
        from negpy.desktop.view.widgets.crosstalk_editor_dialog import CrosstalkEditorDialog

        conf = self.state.config.process
        self._crosstalk_snapshot = (conf.crosstalk_profile, conf.crosstalk_matrix, conf.crosstalk_strength)
        dlg = CrosstalkEditorDialog(conf.crosstalk_profile, conf.crosstalk_strength, parent=self)
        dlg.matrix_previewed.connect(self._on_crosstalk_preview)
        dlg.profiles_changed.connect(self.sync_ui)
        dlg.finished.connect(lambda result: self._on_crosstalk_editor_finished(dlg, result))
        self._crosstalk_dialog = dlg  # keep a reference so the modeless dialog isn't GC'd
        dlg.show()

    def _on_crosstalk_preview(self, matrix: object, strength: float) -> None:
        self.update_config_section(
            "process",
            persist=False,
            render=True,
            crosstalk_matrix=tuple(matrix) if matrix is not None else None,
            crosstalk_strength=strength,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_crosstalk_editor_finished(self, dlg, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            name = dlg.selected_name() or CrosstalkProfiles.DEFAULT_NAME
            snap_strength = self._crosstalk_snapshot[2]
            self.update_config_section(
                "process",
                persist=True,
                render=True,
                crosstalk_profile=name,
                # Default stores no matrix (falls back to the built-in) by convention.
                crosstalk_matrix=None if name == CrosstalkProfiles.DEFAULT_NAME else tuple(dlg.working_matrix()),
                # Preview strength is view-only; only adopt it if the edit had crosstalk off.
                crosstalk_strength=dlg.preview_strength() if snap_strength == 0 else snap_strength,
                **invalidate_local_bounds(self.state.config.process),
            )
        else:
            profile, matrix, strength = self._crosstalk_snapshot
            self.update_config_section(
                "process",
                persist=True,
                render=True,
                crosstalk_profile=profile,
                crosstalk_matrix=matrix,
                crosstalk_strength=strength,
                **invalidate_local_bounds(self.state.config.process),
            )
        self.sync_ui()

    def _on_hue_trim_changed(self, val: float, persist: bool = True) -> None:
        # Sticky on commit only, so a drag doesn't write every intermediate value.
        self.update_config_section("process", hue_trim=val, persist=persist)
        if persist:
            self.controller.session.repo.save_global_setting("last_hue_trim", float(val))

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            profiles = SensorProfiles.list_profiles()
            if profiles != [self.sensor_combo.itemText(i) for i in range(self.sensor_combo.count())]:
                self.sensor_combo.clear()
                self.sensor_combo.addItems(profiles)
            self._apply_gate(conf)

            # Headings included, so a changed `type` rebuilds too (the name set alone would not).
            if self._expected_crosstalk_rows() != [self.crosstalk_combo.itemText(i) for i in range(self.crosstalk_combo.count())]:
                self._fill_crosstalk_combo()
            self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
            self.crosstalk_strength_slider.setValue(conf.crosstalk_strength)
            # Nothing to unmix on one B&W emulsion.
            is_bw = conf.process_mode == ProcessMode.BW
            for w in (self.crosstalk_header, self.crosstalk_label, self.crosstalk_combo, self.manage_crosstalk_btn):
                w.setVisible(not is_bw)
            self.crosstalk_strength_slider.setVisible(not is_bw)

            self.hue_trim_slider.setValue(conf.hue_trim)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.sensor_combo, self.crosstalk_combo, self.crosstalk_strength_slider, self.hue_trim_slider):
            w.blockSignals(blocked)
