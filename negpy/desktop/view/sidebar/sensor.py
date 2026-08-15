from PyQt6.QtWidgets import QComboBox, QDialog, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label, section_subheader, wrap_tooltip
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.process.models import ProcessMode, invalidate_local_bounds
from negpy.features.process.sensor import unmix_block_reason
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.services.assets.sensor import SensorProfiles


class SensorSidebar(BaseSidebar):
    """
    Capture-side color corrections, one per cause and not interchangeable: the
    camera's cross-channel response (linear capture), the film's dye absorptions
    (negative density), and an odd light spectrum's hue rotation (the print).
    """

    def _init_ui(self) -> None:
        conf = self.state.config.process

        self.capture_header = section_subheader("CAPTURE")
        self.layout.addWidget(self.capture_header)

        self.linear_raw_btn = self._small_toggle(
            "fa5s.sliders-h",
            "Linear RAW",
            conf.linear_raw,
            "Decode RAW with neutral multipliers (1,1,1,1) — bypasses as-shot camera white balance for a clean starting point",
        )
        self.narrowband_scan_btn = self._small_toggle(
            "mdi6.led-strip-variant",
            "Narrowband",
            conf.narrowband_scan,
            "Correct trichrome narrowband RGB scans oversaturation with the bundled input profile. "
            "An explicit Input ICC in Export settings overrides it. Not applied to transparencies: "
            "the profile describes narrowband capture of negative dyes",
        )
        self.scan_setup_btn = self._icon_action(
            "mdi6.lightbulb-on-outline",
            "Scanning setup — set Linear RAW and Narrowband from your camera/scanner and its light source",
            width=28,
        )
        capture_row = QHBoxLayout()
        capture_row.addWidget(self.linear_raw_btn, 1)
        capture_row.addWidget(self.narrowband_scan_btn, 1)
        capture_row.addWidget(self.scan_setup_btn)
        self.layout.addLayout(capture_row)

        # Greyed rather than hidden: these are sticky settings, so a hidden one is a setting the
        # user cannot see the state of. Hiding them is what let a rig's narrowband pair follow a
        # frame into Transparency unnoticed.
        self.capture_hint = hint_label("")
        self.capture_hint.setVisible(False)  # text and tooltip are set per film process in sync_ui
        self.layout.addWidget(self.capture_hint)

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
            "for RGB-triplet assets, when Linear RAW is off, and on transparencies — which are not "
            "scanned with narrowband light. Re-run Batch Analysis after changing this."
            "</td></tr></table>"
        )
        self.calibrate_sensor_btn = self._icon_action("fa5s.vials", "Calibrate the sensor from three bare-light R/G/B exposures", width=32)
        row.addWidget(self.sensor_label)
        row.addWidget(self.sensor_combo, 1)
        row.addWidget(self.calibrate_sensor_btn)
        self.layout.addLayout(row)

        # Muted, not warning: this is the normal state for anyone not using Linear RAW, so it
        # explains the greyed controls rather than flagging a problem. Text and tooltip are set
        # per reason in _apply_gate.
        self.sensor_hint = hint_label("Requires Linear RAW.")
        self.layout.addWidget(self.sensor_hint)

        self.crosstalk_header = section_subheader("CROSSTALK")
        self.layout.addWidget(self.crosstalk_header)

        matrix_row = QHBoxLayout()
        self.crosstalk_label = field_label("Matrix")
        self.crosstalk_combo = QComboBox()
        self._fill_crosstalk_combo()
        self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
        # Wrap the long tooltip in a fixed-width table, so Qt word-wraps it to the panel width
        # instead of rendering one line that runs off the screen. Qt auto-wraps rich text only.
        self.crosstalk_combo.setToolTip(
            "<table width='280'><tr><td>"
            "Channel unmix on the raw NEGATIVE densities, before analysis and inversion — the domain "
            "where every cause of channel mixing is linear. The film's dyes absorb outside their own "
            "band, but so do your light's spectrum and your sensor's color filters, and here they all "
            "arrive as the same kind of error. So read a profile as <b>your whole scanning setup</b>, "
            "not just the stock.<br><br>"
            "<b>The bundled film matrices are read off published spec sheets, not measured</b> — they "
            "are marked (approx) for that reason. They describe the film's dyes alone, so they are only "
            "the whole story where the capture reads each dye cleanly: a true RGB scan (a Coolscan-style "
            "mono sensor lit one band at a time) or a calibrated trichrome rig. Under a broadband light "
            "and a Bayer sensor your capture adds its own mixing on top, and a dyes-only matrix will not "
            "describe it.<br><br>"
            "So treat them as starting points and expect to tune: raise Strength until colors separate "
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

        # Shown when the film process has no matrices yet. Muted, not a warning: it is the normal
        # state for any process NegPy ships nothing for.
        self.crosstalk_hint = hint_label("No matrices for this film process — build one in the editor.")
        self.crosstalk_hint.setToolTip(
            wrap_tooltip(
                "A matrix describes one film's dye set, so it only appears here in the process it was "
                "saved for. Open the editor to start one from identity, set its Process, and save it."
            )
        )
        self.layout.addWidget(self.crosstalk_hint)

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

    def _expected_crosstalk_rows(self, process_mode=None) -> list:
        """The rows _fill_crosstalk_combo would produce, for change detection."""
        rows: list = []
        for heading, names in CrosstalkProfiles.grouped_profiles(process_mode):
            rows.append(self._heading_row(heading))
            rows.extend(names)
        return rows

    def _fill_crosstalk_combo(self, process_mode=None) -> None:
        """Rebuild the matrix dropdown with a non-selectable heading per provenance group.

        Qt has no group concept, so headings are combo rows disabled through the model.
        Bracketing them keeps a heading from colliding with a profile name, which would let
        setCurrentText land on one.
        """
        self.crosstalk_combo.clear()
        for heading, names in CrosstalkProfiles.grouped_profiles(process_mode):
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

    # Disabled widgets do not receive the hover that raises a tooltip, so the detail hangs off
    # the hint label rather than the combo it describes.
    _SENSOR_BLOCKED = {
        "linear_raw": (
            "Requires Linear RAW.",
            "Sensor profiles are calibrated against neutral white balance. With Linear RAW "
            "off, RAW decodes carry the camera's as-shot gains instead, which would misapply "
            "the matrix — so it is skipped. Your selection is remembered.",
        ),
        "transparency": (
            "Not applied to a transparency.",
            "The unmix corrects a narrowband light and your sensor's filters against each other, "
            "so it only means anything for a capture made under narrowband light — and narrowband "
            "is not used for slides. A profile carried over from your negative rig would correct "
            "for a light this frame was not shot under. Your selection is remembered, and applies "
            "again on a negative.",
        ),
    }

    def _apply_gate(self, conf) -> None:
        """Grey the sensor unmix and show "None" while it cannot be applied, saying why.

        Display-only: conf.sensor_profile is left alone, so the selection comes back intact
        after a Linear RAW or film-process round-trip. Crosstalk and Hue Trim depend on
        neither the decode basis nor the light, so they stay enabled.
        """
        reason = unmix_block_reason(conf)
        available = not reason
        self.sensor_combo.setCurrentText(conf.sensor_profile if available else SensorProfiles.NONE_NAME)
        self.sensor_combo.setEnabled(available)
        self.calibrate_sensor_btn.setEnabled(available)
        self.sensor_hint.setVisible(bool(reason))
        if reason:
            text, tip = self._SENSOR_BLOCKED[reason]
            self.sensor_hint.setText(text)
            self.sensor_hint.setToolTip(wrap_tooltip(tip))

    def _connect_signals(self) -> None:
        self.linear_raw_btn.toggled.connect(self._on_linear_raw_toggled)
        self.narrowband_scan_btn.toggled.connect(self._on_narrowband_scan_toggled)
        self.scan_setup_btn.clicked.connect(self._open_scan_setup)

        self.sensor_combo.currentTextChanged.connect(self._on_sensor_profile_changed)
        self.calibrate_sensor_btn.clicked.connect(self._open_sensor_calibration)

        self.crosstalk_combo.currentTextChanged.connect(self._on_crosstalk_profile_changed)
        self.manage_crosstalk_btn.clicked.connect(self._open_crosstalk_editor)
        self.crosstalk_strength_slider.valueChanged.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=False))
        self.crosstalk_strength_slider.valueCommitted.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=True))

        self.hue_trim_slider.valueChanged.connect(lambda v: self._on_hue_trim_changed(v, persist=False))
        self.hue_trim_slider.valueCommitted.connect(lambda v: self._on_hue_trim_changed(v, persist=True))

    def _on_linear_raw_toggled(self, checked: bool) -> None:
        from dataclasses import replace

        new_config = replace(
            self.state.config,
            process=replace(
                self.state.config.process,
                linear_raw=checked,
                **invalidate_local_bounds(self.state.config.process),
            ),
        )
        # linear_raw switches use_camera_wb, so it is a source change: apply_config re-decodes and
        # suppresses the bounds analysis over the stale buffer.
        self.controller.apply_config(new_config, persist=True)

    def _on_narrowband_scan_toggled(self, checked: bool) -> None:
        self.update_config_section("process", narrowband_scan=checked, persist=True, render=True)

    def _open_scan_setup(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.show_scan_setup()

    def _on_sensor_profile_changed(self, name: str) -> None:
        # Bake the matrix like crosstalk does. The per-frame bounds were analyzed under the
        # previous mix, so clear them.
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
        # Bake the matrix into the config, so saved edits stay reproducible if the profile file is
        # later moved or deleted. The persisted per-frame bounds were analyzed under the previous
        # matrix, so clear them and the stretch re-derives from the unmixed data. Otherwise the
        # mask redistribution leaks through.
        matrix = CrosstalkProfiles.get_matrix(name)
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            crosstalk_profile=name,
            crosstalk_matrix=matrix,
            # Baked with the matrix so the render can gate on it without disk I/O.
            crosstalk_process=CrosstalkProfiles.get_process(name),
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
        self._crosstalk_snapshot = (conf.crosstalk_profile, conf.crosstalk_matrix, conf.crosstalk_strength, conf.crosstalk_process)
        dlg = CrosstalkEditorDialog(conf.crosstalk_profile, conf.crosstalk_strength, conf.process_mode, parent=self)
        dlg.matrix_previewed.connect(self._on_crosstalk_preview)
        dlg.profiles_changed.connect(self.sync_ui)
        dlg.finished.connect(lambda result: self._on_crosstalk_editor_finished(dlg, result))
        self._crosstalk_dialog = dlg  # keep a reference so the modeless dialog isn't GC'd
        dlg.show()

    def _on_crosstalk_preview(self, matrix: object, strength: float, process: str) -> None:
        # The process rides along: the render gates the unmix on it, so a preview without it shows
        # nothing whenever the edited profile is for another film.
        self.update_config_section(
            "process",
            persist=False,
            render=True,
            crosstalk_matrix=tuple(matrix) if matrix is not None else None,
            crosstalk_strength=strength,
            crosstalk_process=process,
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
                crosstalk_process=dlg.selected_process(),
                **invalidate_local_bounds(self.state.config.process),
            )
        else:
            profile, matrix, strength, process = self._crosstalk_snapshot
            self.update_config_section(
                "process",
                persist=True,
                render=True,
                crosstalk_profile=profile,
                crosstalk_matrix=matrix,
                crosstalk_strength=strength,
                crosstalk_process=process,
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
            self.linear_raw_btn.setChecked(conf.linear_raw)
            self.narrowband_scan_btn.setChecked(conf.narrowband_scan)
            # Two different reasons, so two different gates. Narrowband is refused for any
            # transparency, because the bundled profile describes narrowband capture of negative
            # dyes. Linear RAW is inert only on the *transfer*, where the camera matrix folds the
            # as-shot multipliers back in. With Normalize on it still decides the decode, so it
            # stays live there.
            from negpy.features.exposure.transfer import is_transparency_transfer

            e6 = conf.process_mode == ProcessMode.E6
            transfer = is_transparency_transfer(conf.process_mode, conf.e6_normalize)
            self.narrowband_scan_btn.setEnabled(not e6)
            self.linear_raw_btn.setEnabled(not transfer)
            self.scan_setup_btn.setEnabled(not e6)
            self.capture_hint.setVisible(e6)
            if e6:
                self.capture_hint.setText(
                    "Narrowband is not used for slides." if not transfer else "Not applied to an as-captured transparency."
                )
                self.capture_hint.setToolTip(
                    wrap_tooltip(
                        "Narrowband's bundled input profile describes narrowband capture of *negative* "
                        "dyes, and a slide has a different dye set, so on a transparency it would correct "
                        "for film that is not there. Its real payoffs — defeating the orange mask, clean "
                        "separation before a high-gain inversion — belong to negatives."
                        + (
                            " Linear RAW is inert here too: the camera matrix folds the as-shot multipliers "
                            "back in, so the render is the same either way."
                            if transfer
                            else " Linear RAW still applies, and stays live."
                        )
                        + " Both settings are remembered, and apply again on a negative."
                    )
                )

            profiles = SensorProfiles.list_profiles()
            if profiles != [self.sensor_combo.itemText(i) for i in range(self.sensor_combo.count())]:
                self.sensor_combo.clear()
                self.sensor_combo.addItems(profiles)
            self._apply_gate(conf)

            # Headings included, so a changed `type` rebuilds too (the name set alone would not).
            if self._expected_crosstalk_rows(conf.process_mode) != [
                self.crosstalk_combo.itemText(i) for i in range(self.crosstalk_combo.count())
            ]:
                self._fill_crosstalk_combo(conf.process_mode)
            self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
            self.crosstalk_strength_slider.setValue(conf.crosstalk_strength)
            # Nothing to unmix on one B&W emulsion. Every other process keeps the section even with
            # no matrices of its own, because the editor is the only way to make one and hiding it
            # would leave no route in. The empty dropdown and the Strength slider it feeds are
            # disabled instead, and a hint says why.
            is_bw = conf.process_mode == ProcessMode.BW
            has_profiles = bool(CrosstalkProfiles.grouped_profiles(conf.process_mode))
            for w in (self.crosstalk_header, self.crosstalk_label, self.crosstalk_combo, self.manage_crosstalk_btn):
                w.setVisible(not is_bw)
            self.crosstalk_strength_slider.setVisible(not is_bw)
            self.crosstalk_hint.setVisible(not is_bw and not has_profiles)
            self.crosstalk_combo.setEnabled(has_profiles)
            self.crosstalk_strength_slider.setEnabled(has_profiles)

            self.hue_trim_slider.setValue(conf.hue_trim)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.linear_raw_btn,
            self.narrowband_scan_btn,
            self.sensor_combo,
            self.crosstalk_combo,
            self.crosstalk_strength_slider,
            self.hue_trim_slider,
        ):
            w.blockSignals(blocked)
