from typing import Optional

from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QDialog, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label, wrap_tooltip
from negpy.desktop.view.widgets.file_dialogs import last_open_folder
from negpy.desktop.view.sidebar.scan_setup import resolved_rig
from negpy.desktop.view.widgets.fold import group_fold
from negpy.desktop.view.widgets.sliders import CompactSlider, SliderGroup
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.features.process.models import ProcessMode, SensorUnmix, invalidate_local_bounds
from negpy.features.process.sensor import unmix_block_reason
from negpy.features.rgbscan.models import is_rgb_triplet
from negpy.features.stitch.models import stitch_has_triplets
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.services.assets.sensor import SensorProfiles


_UNMIX_CHOICES = (
    (
        SensorUnmix.LINEAR,
        tr("Removes the leak exactly, but colors that the film barely passes can print speckled."),
    ),
    (
        SensorUnmix.TWO_SCALE,
        tr("Like Linear, but very strong colors come from a slightly blurred copy, so they do not speckle."),
    ),
    (
        SensorUnmix.DENSITY,
        tr("Works on film density, with no speckle, but strong colors are slightly less vivid."),
    ),
)

_CROSSTALK_FIELDS = ("crosstalk_profile", "crosstalk_matrix", "crosstalk_strength", "crosstalk_process")


class SensorSidebar(BaseSidebar):
    """
    Capture-side color corrections, one per cause and not interchangeable: the
    camera's cross-channel response (linear capture), the film's dye absorptions
    (negative density), and an odd light spectrum's hue rotation (the print).
    """

    def _init_ui(self) -> None:
        conf = self.state.config.process
        self._crosstalk_dialog: Optional[QDialog] = None
        self._crosstalk_snapshot: dict = {}  # the frame's crosstalk fields when the editor opened
        self._crosstalk_previewed: dict = {}  # the fields the editor's last preview set

        row = QHBoxLayout()
        self.sensor_label = field_label(tr("Profile"))
        self.sensor_combo = StepCombo()
        self.sensor_combo.addItems(SensorProfiles.list_profiles())
        self.sensor_combo.setToolTip(
            tr(
                "Sensor crosstalk profile: removes the color leak between channels in a single exposure under narrowband light; "
                "run Roll Analysis again after a change."
            )
        )
        self.calibrate_sensor_btn = self._icon_action(
            "fa5s.vials", tr("Make a profile from three shots of the bare light: red, green and blue.")
        )
        row.addWidget(self.sensor_label)
        row.addWidget(self.sensor_combo, 1)

        unmix_row = QHBoxLayout()
        self.unmix_label = field_label(tr("Method"))
        # Plain strings: Qt's findData cannot match a StrEnum member.
        self.unmix_btn = StepCombo.of(
            [mode.label for mode, _tip in _UNMIX_CHOICES],
            tr("How the calibration is applied; run Roll Analysis again after a change."),
            [mode.value for mode, _tip in _UNMIX_CHOICES],
        )
        for i, (_mode, tip) in enumerate(_UNMIX_CHOICES):
            self.unmix_btn.set_item_tooltip(i, wrap_tooltip(tip))
        self.unmix_btn.setCurrentIndex(max(0, self.unmix_btn.findData(str(conf.sensor_unmix))))
        unmix_row.addWidget(self.unmix_label)
        unmix_row.addWidget(self.unmix_btn, 1)

        # Muted, not warning: this is the normal state for anyone not using Linear RAW, so it
        # explains the greyed controls rather than flagging a problem. Text and tooltip are set
        # per reason in _apply_gate.
        self.sensor_hint = hint_label("")
        narrowband_header, self.narrowband_fold = group_fold(
            "sensor_narrowband",
            tr("Sensor Crosstalk Profile"),
            row,
            unmix_row,
            self.sensor_hint,
            buttons=(self.calibrate_sensor_btn,),
        )
        self.layout.addLayout(narrowband_header)
        self.layout.addWidget(self.narrowband_fold)

        matrix_row = QHBoxLayout()
        self.crosstalk_label = field_label(tr("Matrix"))
        self.crosstalk_combo = StepCombo()
        self._fill_crosstalk_combo()
        self._select_crosstalk(conf.crosstalk_profile)
        self.crosstalk_combo.setToolTip(
            tr("Crosstalk matrix: separates colors that the film's dyes, your light and your sensor mix together.")
        )
        self.manage_crosstalk_btn = self._icon_action("fa5s.sliders-h", tr("Open the crosstalk matrix editor."))
        self.measure_chart_btn = self._icon_action(
            "mdi6.palette-swatch-outline", tr("Measure a matrix from a color chart shot on this film and scanned on this rig…")
        )
        matrix_row.addWidget(self.crosstalk_label)
        matrix_row.addWidget(self.crosstalk_combo, 1)

        # Shown when the film process has no matrices yet. Muted, not a warning: it is the normal
        # state for any process NegPy ships nothing for.
        self.crosstalk_hint = hint_label(tr("No matrices for this film type yet. Make one in the editor."))
        self.crosstalk_hint.setToolTip(
            wrap_tooltip(tr("A matrix shows only for the film type that it was saved for, so open the editor to make one."))
        )
        self.crosstalk_strength_slider = CompactSlider(tr("Strength"), 0.0, 1.0, conf.crosstalk_strength, has_neutral=True)
        self.crosstalk_strength_rail = SliderGroup(self.crosstalk_strength_slider)
        crosstalk_header, crosstalk_fold = group_fold(
            "sensor_crosstalk",
            tr("Crosstalk"),
            matrix_row,
            self.crosstalk_hint,
            self.crosstalk_strength_rail,
            buttons=(self.measure_chart_btn, self.manage_crosstalk_btn),
        )
        self.crosstalk_header = crosstalk_fold.button
        self.layout.addLayout(crosstalk_header)
        self.layout.addWidget(crosstalk_fold)

        self.hue_trim_slider = CompactSlider(tr("Hue Trim"), -30.0, 30.0, conf.hue_trim, step=0.5, precision=10, has_neutral=True, unit="°")
        self.hue_trim_slider.setToolTip(
            tr("Rotates all hues, for a light that turns yellows orange or greens olive; keep 0 for white light.")
        )
        light_header, self.light_fold = group_fold("sensor_light", tr("Light Source"), self.hue_trim_slider)
        self.layout.addLayout(light_header)
        self.layout.addWidget(self.light_fold)

        self._apply_gate(conf)

    def _expected_crosstalk_rows(self, process_mode=None) -> list:
        """The rows _fill_crosstalk_combo would produce, for change detection."""
        rows: list = []
        for heading, names in CrosstalkProfiles.grouped_profiles(process_mode):
            rows.append(tr(heading))
            rows.extend(names)
        return rows

    def _fill_crosstalk_combo(self, process_mode=None) -> None:
        """Rebuild the matrix dropdown with a caption row per provenance group. A profile carries
        its name as item data and a caption none, so selection by data never lands on a caption
        that shares a profile's name."""
        self.crosstalk_combo.clear()
        for heading, names in CrosstalkProfiles.grouped_profiles(process_mode):
            self.crosstalk_combo.add_header(tr(heading))
            for name in names:
                self.crosstalk_combo.addItem(name, name)

    def _select_crosstalk(self, name: str) -> None:
        idx = self.crosstalk_combo.findData(name)
        if idx >= 0:
            self.crosstalk_combo.setCurrentIndex(idx)

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
            tr("Requires Linear RAW, for a Single Exposure under narrowband light."),
            tr("A sensor crosstalk profile needs Linear RAW, so the profile is skipped while it is off and your choice is kept."),
        ),
        "triplet": (
            tr("Not used for a Trichrome triplet."),
            tr("A Trichrome triplet has no sensor leak to remove, so the profile is skipped and your choice is kept."),
        ),
        "transparency": (
            tr("Not applied to a transparency."),
            tr("Slides are not scanned with narrowband light, so the profile is skipped and your choice is kept for negatives."),
        ),
    }

    def _apply_gate(self, conf) -> None:
        """Gray the sensor unmix and show "None" while it cannot be applied, saying why.

        Display-only: conf.sensor_profile is left alone, so the selection comes back intact
        after a Linear RAW or film-process round-trip. Crosstalk and Hue Trim depend on
        neither the decode basis nor the light, so they stay enabled. Calibrate stays enabled too: it
        reads its own three shots, not this frame.
        """
        reason = self._unmix_reason()
        available = not reason
        self.sensor_combo.setCurrentText(conf.sensor_profile if available else SensorProfiles.NONE_NAME)
        self.sensor_combo.setEnabled(available)
        self.unmix_btn.setEnabled(available and conf.sensor_matrix is not None)
        self.sensor_hint.setVisible(bool(reason))
        if reason:
            text, tip = self._SENSOR_BLOCKED[reason]
            self.sensor_hint.setText(text)
            self.sensor_hint.setToolTip(wrap_tooltip(tip))

    def _unmix_reason(self) -> str:
        config = self.state.config
        triplet = is_rgb_triplet(config.rgbscan) or stitch_has_triplets(config.stitch)
        return ("triplet" if triplet else "") or unmix_block_reason(
            config.process
        )  # the frame's own reason first: a decode change cannot lift it

    def _connect_signals(self) -> None:
        self.sensor_combo.currentTextChanged.connect(self._on_sensor_profile_changed)
        self.unmix_btn.currentIndexChanged.connect(self._on_unmix_changed)
        self.calibrate_sensor_btn.clicked.connect(self._open_sensor_calibration)

        self.crosstalk_combo.currentTextChanged.connect(self._on_crosstalk_profile_changed)
        self.manage_crosstalk_btn.clicked.connect(self._open_crosstalk_editor)
        self.controller.session.frame_previews_ending.connect(self._end_crosstalk_editor)
        self.controller.session.session_emptied.connect(self._end_crosstalk_editor)
        self.measure_chart_btn.clicked.connect(self._open_chart_calibration)
        self.crosstalk_strength_slider.valueChanged.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=False))
        self.crosstalk_strength_slider.valueCommitted.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=True))

        self.hue_trim_slider.valueChanged.connect(lambda v: self._on_hue_trim_changed(v, persist=False))
        self.hue_trim_slider.valueCommitted.connect(lambda v: self._on_hue_trim_changed(v, persist=True))

    def _on_sensor_profile_changed(self, name: str) -> None:
        # Bake the matrix like crosstalk does. The per-frame bounds were analyzed under the
        # previous mix, so clear them.
        matrix = SensorProfiles.get_matrix(name)
        self.controller.set_roll_default(
            "sensor",
            sensor_profile=name,
            sensor_matrix=tuple(matrix) if matrix is not None else None,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_unmix_changed(self, _index: int) -> None:
        self.controller.set_roll_default(
            "sensor",
            sensor_unmix=SensorUnmix(self.unmix_btn.currentData()),
            **invalidate_local_bounds(self.state.config.process),
        )

    def _open_sensor_calibration(self) -> None:
        from negpy.desktop.view.widgets.sensor_calibration_dialog import SensorCalibrationDialog

        repo = self.controller.session.repo
        dlg = SensorCalibrationDialog(parent=self, start_dir=last_open_folder(repo), repo=repo, controller=self.controller)
        dlg.profile_saved.connect(self._on_sensor_profile_saved)
        dlg.exec()

    def _on_sensor_profile_saved(self, name: str) -> None:
        # Linear RAW off keeps the choice, as the gate does; a slide or a triplet never uses one.
        if self.state.current_file_hash and self._unmix_reason() in ("", "linear_raw"):
            self._on_sensor_profile_changed(name)
        else:
            self.controller.set_status(tr("Saved sensor profile “{name}”. Select it on a single-shot negative.", name=name), 6000)
        self.sync_ui()  # rebuild the combo, which now lists the new profile

    def _on_crosstalk_profile_changed(self, name: str) -> None:
        # Bake the matrix into the config, so saved edits stay reproducible if the profile file is
        # later moved or deleted. The persisted per-frame bounds were analyzed under the previous
        # matrix, so clear them and the stretch re-derives from the unmixed data. Otherwise the
        # mask redistribution leaks through.
        matrix = CrosstalkProfiles.get_matrix(name)
        self.controller.set_roll_default(
            "sensor",
            crosstalk_profile=name,
            crosstalk_matrix=matrix,
            # Baked with the matrix so the render can gate on it without disk I/O.
            crosstalk_process=CrosstalkProfiles.get_process(name),
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_crosstalk_strength_changed(self, val: float, persist: bool = True) -> None:
        self.controller.set_roll_default(
            "sensor",
            persist=persist,
            readback_metrics=persist,
            crosstalk_strength=val,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _open_crosstalk_editor(self) -> None:
        from negpy.desktop.view.widgets.crosstalk_editor_dialog import CrosstalkEditorDialog

        if self._crosstalk_dialog is not None:
            self._crosstalk_dialog.raise_()
            self._crosstalk_dialog.activateWindow()
            return
        conf = self.state.config.process
        self._crosstalk_snapshot = {f: getattr(conf, f) for f in _CROSSTALK_FIELDS}
        self._crosstalk_previewed = {}
        dlg = CrosstalkEditorDialog(
            conf.crosstalk_profile, conf.crosstalk_strength, conf.process_mode, parent=self, repo=self.controller.session.repo
        )
        dlg.matrix_previewed.connect(self._on_crosstalk_preview)
        dlg.profiles_changed.connect(self.sync_ui)
        dlg.finished.connect(lambda result: self._on_crosstalk_editor_finished(dlg, result))
        self._crosstalk_dialog = dlg  # keep a reference so the modeless dialog isn't GC'd
        dlg.show()

    def _open_chart_calibration(self) -> None:
        from negpy.desktop.view.widgets.chart_calibration_dialog import ChartCalibrationDialog

        source = self.controller.chart_fit_source()
        if source is None:
            self.controller.set_status(tr("Open a whole frame of the chart to measure it."), 4000, "warning")
            return
        dlg = ChartCalibrationDialog(source, self.state.workspace_color_space, parent=self, repo=self.controller.session.repo)
        dlg.profile_saved.connect(self._on_chart_profile_saved)
        dlg.exec()

    def _on_chart_profile_saved(self, name: str) -> None:
        self.controller.set_roll_default(
            "sensor",
            crosstalk_profile=name,
            crosstalk_matrix=CrosstalkProfiles.get_matrix(name),
            crosstalk_process=CrosstalkProfiles.get_process(name),
            crosstalk_strength=1.0,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.sync_ui()

    def _on_crosstalk_preview(self, matrix: object, strength: float, process: str) -> None:
        # The process rides along: the render gates the unmix on it, so a preview without it shows
        # nothing whenever the edited profile is for another film.
        self._crosstalk_previewed = {
            "crosstalk_matrix": tuple(matrix) if matrix is not None else None,
            "crosstalk_strength": strength,
            "crosstalk_process": process,
        }
        self.controller.set_roll_default(
            "sensor", persist=False, **self._crosstalk_previewed, **invalidate_local_bounds(self.state.config.process)
        )

    def _end_crosstalk_editor(self) -> None:
        """Close the editor and take its preview back, so the frame's edit is stored without it."""
        dlg = self._crosstalk_dialog
        if dlg is None:
            return
        self._crosstalk_dialog = None
        if self.state.current_file_hash and self._crosstalk_previewed:
            self.controller.session.discard_preview()
            # A persisted edit made during the preview stored it too; only a field the user has not
            # changed since goes back, and the bounds re-meter instead of taking stale values.
            conf = self.state.config.process
            snapshot = self._crosstalk_snapshot
            changes = {f: snapshot[f] for f, v in self._crosstalk_previewed.items() if getattr(conf, f) == v != snapshot[f]}
            if changes:
                self.update_config_section("process", render=False, **changes, **invalidate_local_bounds(conf))
        dlg.reject()

    def _on_crosstalk_editor_finished(self, dlg, result: int) -> None:
        if dlg is not self._crosstalk_dialog:
            return  # ended by _end_crosstalk_editor, which restored the frame
        self._crosstalk_dialog = None
        if result == QDialog.DialogCode.Accepted:
            name = dlg.selected_name() or CrosstalkProfiles.DEFAULT_NAME
            snap_strength = self._crosstalk_snapshot["crosstalk_strength"]
            self.controller.set_roll_default(
                "sensor",
                crosstalk_profile=name,
                # Default stores no matrix (falls back to the built-in) by convention.
                crosstalk_matrix=None if name == CrosstalkProfiles.DEFAULT_NAME else tuple(dlg.working_matrix()),
                # Preview strength is view-only; only adopt it if the edit had crosstalk off.
                crosstalk_strength=dlg.preview_strength() if snap_strength == 0 else snap_strength,
                crosstalk_process=dlg.selected_process(),
                **invalidate_local_bounds(self.state.config.process),
            )
        else:
            self.controller.set_roll_default("sensor", **self._crosstalk_snapshot, **invalidate_local_bounds(self.state.config.process))
        self.sync_ui()

    def _on_hue_trim_changed(self, val: float, persist: bool = True) -> None:
        # Sticky on commit only, so a drag doesn't write every intermediate value.
        self.controller.set_roll_default("sensor", hue_trim=val, persist=persist, readback_metrics=persist)

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            profiles = SensorProfiles.list_profiles()
            if profiles != [self.sensor_combo.itemText(i) for i in range(self.sensor_combo.count())]:
                self.sensor_combo.clear()
                self.sensor_combo.addItems(profiles)
            self.unmix_btn.setCurrentIndex(max(0, self.unmix_btn.findData(str(conf.sensor_unmix))))
            self._apply_gate(conf)

            # Headings included, so a changed `type` rebuilds too (the name set alone would not).
            if self._expected_crosstalk_rows(conf.process_mode) != [
                self.crosstalk_combo.itemText(i) for i in range(self.crosstalk_combo.count())
            ]:
                self._fill_crosstalk_combo(conf.process_mode)
            self._select_crosstalk(conf.crosstalk_profile)
            self.crosstalk_strength_slider.setValue(conf.crosstalk_strength)
            # Nothing to unmix on one B&W emulsion. Every other process keeps the section even with
            # no matrices of its own, because the editor is the only way to make one and hiding it
            # would leave no route in. The empty dropdown and the Strength slider it feeds are
            # disabled instead, and a hint says why.
            is_bw = conf.process_mode == ProcessMode.BW
            has_profiles = bool(CrosstalkProfiles.grouped_profiles(conf.process_mode))
            for w in (self.crosstalk_header, self.crosstalk_label, self.crosstalk_combo, self.manage_crosstalk_btn):
                w.setVisible(not is_bw)
            self.crosstalk_strength_rail.setVisible(not is_bw)
            # The charts' published colors are for a print of the scene, which only a negative makes.
            self.measure_chart_btn.setVisible(conf.process_mode == ProcessMode.C41)
            self.crosstalk_hint.setVisible(not is_bw and not has_profiles)
            self.crosstalk_combo.setEnabled(has_profiles)
            self.crosstalk_strength_slider.setEnabled(has_profiles)

            self.hue_trim_slider.setValue(conf.hue_trim)

            # Only a single-exposure narrowband camera rig leaks between bands or turns hues; a group
            # that holds a setting stays in view whatever the rig.
            camera_narrowband = resolved_rig(self.controller.session.repo, self.state) == ("camera", "narrowband")
            narrowband = camera_narrowband or conf.sensor_matrix is not None
            self.narrowband_fold.content.setVisible(narrowband)
            self.calibrate_sensor_btn.setVisible(narrowband)
            self.light_fold.content.setVisible(camera_narrowband or conf.hue_trim != 0)
        finally:
            self.block_signals(False)

    def has_controls(self) -> bool:
        """Whether any group shows, so the card can hide when the rig and film leave it empty."""
        return any(not w.isHidden() for w in (self.narrowband_fold.content, self.light_fold.content, self.crosstalk_header))

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.sensor_combo,
            self.unmix_btn,
            self.crosstalk_combo,
            self.crosstalk_strength_slider,
            self.hue_trim_slider,
        ):
            w.blockSignals(blocked)
