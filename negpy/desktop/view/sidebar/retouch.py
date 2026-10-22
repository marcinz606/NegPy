from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QHBoxLayout
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.desktop.view.widgets.sliders import CompactSlider, SliderGroup
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, field_row, header_row, hint_label, section_subheader, wrap_tooltip
from negpy.features.retouch.models import HEAL_SIZE_MAX, HEAL_SIZE_MIN, IR_METHOD_NEGPY, IR_METHOD_OPENICE

_IR_REMOVAL_TIP = tr("IR Removal: remove dust and scratches with the scanner's infrared channel.")
_IR_THRESH_TIP = tr("Lower finds more dust; higher removes less.")
# Order matches _IR_METHOD_KEYS.
_IR_METHOD_LABELS = ("NegPy", tr("OpenICE"))
_IR_METHOD_KEYS = (IR_METHOD_NEGPY, IR_METHOD_OPENICE)
# Order matches _OVERLAY_MODES.
_OVERLAY_LABELS = (tr("Off"), tr("Marked"), tr("IR"))
_OVERLAY_MODES = ("off", "marked", "ir")
_IR_METHOD_TIP = wrap_tooltip(tr("How the film under a defect is rebuilt: OpenICE keeps more fine detail but is less tested."))
_OPTICAL_TIP = tr("Optical Removal: find and remove dust without an infrared channel, and right-drag on the picture to protect an area.")
_RIGHT_CLICK_TIP = tr("When on, a right-click on the picture brings back the mark under it, without a menu.")


def _clone_hint(picking: bool, has_source: bool) -> str:
    if picking or not has_source:
        return tr("Click the photo to pick the area to copy from.")
    return tr("Paint over the defect; the dashed circle marks the source.")


class RetouchSidebar(BaseSidebar):
    """
    Panel for dust removal and healing.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.retouch

        # Applies to every detection source.
        self.overlay_btn = SegmentedChoice(
            tuple(("", label) for label in _OVERLAY_LABELS),
            tr("Show what dust removal found (Marked) or the scan's infrared channel (IR)."),
        )
        self.layout.addLayout(field_row(tr("Overlay"), self.overlay_btn))

        self.auto_dust_btn = self._small_toggle("fa5s.magic", tr("Optical Removal"), conf.dust_remove, _OPTICAL_TIP, align_left=True)
        self.right_click_btn = self._tool_toggle("mdi.cursor-default-click", "", _RIGHT_CLICK_TIP)
        self.right_click_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.right_click_btn.setChecked(self.state.right_click_excludes)
        optical_row = QHBoxLayout()
        optical_row.addWidget(self.auto_dust_btn, 1)
        optical_row.addWidget(self.right_click_btn)
        self.layout.addLayout(optical_row)
        self.threshold_slider = CompactSlider(tr("Spot Threshold"), 0.01, 1.0, conf.dust_threshold)
        self.hair_threshold_slider = CompactSlider(tr("Hair Threshold"), 0.01, 1.0, conf.dust_hair_threshold)
        self.auto_size_slider = CompactSlider(tr("Size"), 2.0, 8.0, float(conf.dust_size), step=1.0, precision=1, unit=" px")
        self.optical_rail = SliderGroup(self.threshold_slider, self.hair_threshold_slider, self.auto_size_slider)
        self.layout.addWidget(self.optical_rail)

        self.ir_dust_btn = self._small_toggle("fa5s.broom", tr("IR Removal"), conf.ir_dust_remove, _IR_REMOVAL_TIP, align_left=True)
        self.ir_method_btn = StepCombo()
        for label, key in zip(_IR_METHOD_LABELS, _IR_METHOD_KEYS):
            self.ir_method_btn.addItem(label, key)
        ir_row = QHBoxLayout()
        ir_row.addWidget(self.ir_dust_btn, 1)
        ir_row.addWidget(self.ir_method_btn)
        self.layout.addLayout(ir_row)
        self.ir_threshold_slider = CompactSlider(tr("IR Threshold"), 0.05, 0.95, float(conf.ir_threshold))
        self.ir_rail = SliderGroup(self.ir_threshold_slider)
        self.layout.addWidget(self.ir_rail)

        # Restored whenever the scan has IR (never let a stale "No IR channel" tip linger).
        self._ir_tooltips = {
            self.ir_dust_btn: _IR_REMOVAL_TIP,
            self.ir_threshold_slider: _IR_THRESH_TIP,
            self.ir_method_btn: _IR_METHOD_TIP,
        }

        self.heals_subheader = section_subheader(tr("MANUAL HEAL · 0"))
        self.pick_dust_btn = self._tool_toggle(
            "fa5s.eye-dropper",
            tr("Heal"),
            tr("Heal: click or paint over dust to remove it, and clean film under the brush stays as it is."),
        )
        self.pick_scratch_btn = self._tool_toggle(
            "fa5s.pen-nib",
            tr("Scratch"),
            tr("Scratch: click points along a scratch or hair, then press Enter, or Esc to cancel."),
        )
        self.pick_line_btn = self._tool_toggle(
            "fa5s.grip-lines",
            tr("Line"),
            tr("Line: click once on a long straight scratch, and the whole line is found and repaired."),
        )
        self.undo_btn = self._icon_action("fa5s.undo", tr("Undo Last: remove the most recent manual heal"))
        self.clear_btn = self._icon_action("fa5s.trash-alt", tr("Clear All: remove all manual heals and keep the automatic dust removal."))
        self.layout.addLayout(header_row(self.heals_subheader, self.undo_btn, self.clear_btn))
        tools_row = QHBoxLayout()
        for btn in (self.pick_dust_btn, self.pick_scratch_btn, self.pick_line_btn):
            tools_row.addWidget(btn, 1)
        self.layout.addLayout(tools_row)

        self.manual_size_slider = CompactSlider(
            tr("Brush Size"), HEAL_SIZE_MIN, HEAL_SIZE_MAX, float(conf.manual_dust_size), step=1.0, precision=1, unit=" px"
        )
        self.manual_size_slider.setToolTip(tr("Brush size, which Alt+scroll on the picture also changes."))
        self.line_threshold_slider = CompactSlider(tr("Line Sensitivity"), 0.05, 0.95, float(conf.scratch_threshold))
        self.line_threshold_slider.setToolTip(tr("Lower finds fainter lines; raise it if the repair spreads onto good film."))
        self.layout.addWidget(SliderGroup(self.manual_size_slider, self.line_threshold_slider))

        self.clone_subheader = section_subheader(tr("CLONE · 0"))
        self.clone_btn = self._tool_toggle(
            "fa5s.clone",
            tr("Clone"),
            tr("Clone: Alt-click to pick an area, then copy its film over a defect."),
        )
        self.clone_match_btn = self._small_toggle(
            "fa5s.adjust",
            tr("Match Tone"),
            conf.clone_match_tone,
            tr("Keep the copied grain and detail, but match the brightness and color around the destination"),
        )
        self.clone_undo_btn = self._icon_action("fa5s.undo", tr("Undo Last: remove the most recent clone stroke"))
        self.clone_clear_btn = self._icon_action("fa5s.trash-alt", tr("Clear All: remove all clone strokes"))
        self.layout.addLayout(header_row(self.clone_subheader, self.clone_undo_btn, self.clone_clear_btn))
        self.clone_source_btn = self._tool_toggle(
            "fa5s.crosshairs",
            tr("Set Source"),
            tr("Pick a new source with the next click on the photo"),
        )
        clone_row = QHBoxLayout()
        for btn in (self.clone_btn, self.clone_source_btn, self.clone_match_btn):
            clone_row.addWidget(btn, 1)
        self.layout.addLayout(clone_row)
        self.clone_hint = hint_label()
        self.layout.addWidget(self.clone_hint)
        self.clone_strength_slider = CompactSlider(tr("Strength"), 0.0, 100.0, conf.clone_strength * 100.0, step=1.0, precision=0, unit="%")
        self.clone_strength_slider.setToolTip(
            wrap_tooltip(tr("How much of the source covers the destination, where lower lets the original show through"))
        )
        self.clone_feather_slider = CompactSlider(tr("Feather"), 0.0, 100.0, conf.clone_feather * 100.0, step=1.0, precision=0, unit="%")
        self.clone_feather_slider.setToolTip(wrap_tooltip(tr("Softness of the brush edge. 0 is a hard edge.")))
        self.clone_rail = SliderGroup(self.clone_strength_slider, self.clone_feather_slider)
        self.layout.addWidget(self.clone_rail)

        self.layout.addStretch()

        self._set_ir_controls_enabled(self.state.has_ir)
        self._sync_rails()

    def _connect_signals(self) -> None:
        # The toggle is the way back: excluded patches are dropped with it, so the removal
        # returns everywhere when it is switched on again.
        self.auto_dust_btn.toggled.connect(
            lambda c: self.update_config_section("retouch", persist=True, render=True, dust_remove=c, dust_exclusion_strokes=[])
        )
        self.auto_dust_btn.toggled.connect(lambda _c: self._sync_rails())
        self.right_click_btn.toggled.connect(self.controller.session.set_right_click_excludes)
        for slider, field, cast in (
            (self.threshold_slider, "dust_threshold", float),
            (self.hair_threshold_slider, "dust_hair_threshold", float),
            (self.auto_size_slider, "dust_size", int),
            (self.line_threshold_slider, "scratch_threshold", float),
            (self.ir_threshold_slider, "ir_threshold", float),
        ):
            slider.valueChanged.connect(
                lambda v, f=field, c=cast: self.update_config_section("retouch", readback_metrics=False, **{f: c(v)})
            )
            slider.valueCommitted.connect(lambda v, f=field, c=cast: self.update_config_section("retouch", persist=True, **{f: c(v)}))
        self.pick_dust_btn.toggled.connect(self._on_pick_toggled)
        self.pick_scratch_btn.toggled.connect(self._on_scratch_toggled)
        self.pick_line_btn.toggled.connect(self._on_line_toggled)
        self.manual_size_slider.valueChanged.connect(lambda v: self.update_config_section("retouch", render=False, manual_dust_size=int(v)))
        self.manual_size_slider.valueCommitted.connect(
            lambda v: self.update_config_section("retouch", render=False, persist=True, manual_dust_size=int(v))
        )
        self.clone_btn.toggled.connect(self._on_clone_toggled)
        self.clone_source_btn.toggled.connect(self.controller.arm_clone_source)
        self.clone_match_btn.toggled.connect(
            lambda c: self.update_config_section("retouch", render=False, persist=True, clone_match_tone=c)
        )
        self.clone_strength_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", render=False, clone_strength=float(v) / 100.0)
        )
        self.clone_strength_slider.valueCommitted.connect(
            lambda v: self.update_config_section("retouch", render=False, persist=True, clone_strength=float(v) / 100.0)
        )
        self.clone_feather_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", render=False, clone_feather=float(v) / 100.0)
        )
        self.clone_feather_slider.valueCommitted.connect(
            lambda v: self.update_config_section("retouch", render=False, persist=True, clone_feather=float(v) / 100.0)
        )
        self.clone_undo_btn.clicked.connect(self.controller.undo_last_clone)
        self.clone_clear_btn.clicked.connect(self.controller.clear_clones)
        self.undo_btn.clicked.connect(self.controller.undo_last_retouch)
        self.clear_btn.clicked.connect(self.controller.clear_retouch)
        self.overlay_btn.currentChanged.connect(lambda i: self.controller.set_dust_overlay(_OVERLAY_MODES[i]))

        self.ir_dust_btn.toggled.connect(
            lambda c: self.update_config_section("retouch", persist=True, render=True, ir_dust_remove=c, ir_attenuation=c)
        )
        self.ir_dust_btn.toggled.connect(lambda _c: self._sync_rails())
        self.ir_method_btn.currentIndexChanged.connect(
            lambda _i: self.update_config_section("retouch", persist=True, render=True, ir_method=self.ir_method_btn.currentData())
        )

    def _sync_overlay(self) -> None:
        mode = self.state.dust_overlay_mode
        # An IR overlay on a frame with no IR plane draws nothing, so it reads as Off.
        if mode not in _OVERLAY_MODES or (mode == "ir" and not self.state.has_ir):
            mode = "off"
        self.overlay_btn.setCurrentIndex(_OVERLAY_MODES.index(mode))
        self.overlay_btn.set_choice_enabled(2, self.state.has_ir)

    def _sync_rails(self) -> None:
        """Each rail shows while what it tunes acts: a removal on, or a tool armed."""
        self.optical_rail.setVisible(self.auto_dust_btn.isChecked())
        has_ir = self.state.has_ir
        self.ir_dust_btn.setVisible(has_ir)
        self.ir_method_btn.setVisible(has_ir)
        self.ir_rail.setVisible(has_ir and self.ir_dust_btn.isChecked())
        # The brush is sized from the canvas while an exclusion is painted, which happens with
        # no tool active, so Optical Removal shows the value too. Sensitivity stands in for brush
        # size on a line: the band grows from the scratch, so what is tuned is how readily it is followed.
        # The Light Leak card's Paint brush is sized here too.
        brush = any(b.isChecked() for b in (self.pick_dust_btn, self.pick_scratch_btn, self.clone_btn, self.auto_dust_btn))
        brush = brush or self.state.active_tool == ToolMode.LEAK_PAINT
        self.manual_size_slider.setVisible(brush)
        self.line_threshold_slider.setVisible(self.pick_line_btn.isChecked())
        self.clone_rail.setVisible(self.clone_btn.isChecked())

    def _on_pick_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.DUST_PICK if checked else ToolMode.NONE)
        self._sync_rails()

    def _on_scratch_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.SCRATCH_PICK if checked else ToolMode.NONE)
        self._sync_rails()

    def _on_clone_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.CLONE if checked else ToolMode.NONE)
        self._sync_rails()

    def _on_line_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.SCRATCH_LINE if checked else ToolMode.NONE)
        self._sync_rails()

    def _set_ir_controls_enabled(self, enabled: bool) -> None:
        for w, tip in self._ir_tooltips.items():
            w.setEnabled(enabled)
            w.setToolTip(tip if enabled else tr("No IR channel in this scan"))

    def sync_ui(self) -> None:
        conf = self.state.config.retouch
        self.block_signals(True)
        try:
            self.auto_dust_btn.setChecked(conf.dust_remove)
            self.right_click_btn.setChecked(self.state.right_click_excludes)
            self.threshold_slider.setValue(conf.dust_threshold)
            self.hair_threshold_slider.setValue(conf.dust_hair_threshold)
            self.auto_size_slider.setValue(float(conf.dust_size))
            self.manual_size_slider.setValue(float(conf.manual_dust_size))
            self.pick_dust_btn.setChecked(self.state.active_tool == ToolMode.DUST_PICK)
            self.pick_scratch_btn.setChecked(self.state.active_tool == ToolMode.SCRATCH_PICK)
            self.pick_line_btn.setChecked(self.state.active_tool == ToolMode.SCRATCH_LINE)
            cloning = self.state.active_tool == ToolMode.CLONE
            self.clone_btn.setChecked(cloning)
            self.clone_source_btn.setChecked(cloning and self.state.clone_picking)
            self.clone_hint.setText(_clone_hint(self.state.clone_picking, self.state.clone_source is not None))
            self.clone_hint.setVisible(cloning)
            self.clone_match_btn.setChecked(conf.clone_match_tone)
            self.clone_strength_slider.setValue(conf.clone_strength * 100.0)
            self.clone_feather_slider.setValue(conf.clone_feather * 100.0)
            num_clones = len(conf.clone_strokes)
            self.clone_subheader.setText(tr("CLONE · {num_clones}", num_clones=num_clones))
            self.clone_undo_btn.setEnabled(num_clones > 0)
            self.clone_clear_btn.setEnabled(num_clones > 0)
            self.line_threshold_slider.setValue(float(conf.scratch_threshold))

            num_heals = len(conf.manual_dust_spots) + len(conf.manual_heal_strokes) + len(conf.scratch_lines)
            self.heals_subheader.setText(tr("MANUAL HEAL · {num_heals}", num_heals=num_heals))

            has_heals = num_heals > 0
            self.undo_btn.setEnabled(has_heals)
            self.clear_btn.setEnabled(has_heals)

            # Show unchecked on non-IR files: the config value is inert without an IR plane, and a
            # checked but greyed button reads as stuck on.
            self.ir_dust_btn.setChecked(conf.ir_dust_remove and self.state.has_ir)
            self.ir_threshold_slider.setValue(float(conf.ir_threshold))
            method = conf.ir_method if conf.ir_method in _IR_METHOD_KEYS else IR_METHOD_NEGPY
            self.ir_method_btn.setCurrentIndex(self.ir_method_btn.findData(method))
            self._set_ir_controls_enabled(self.state.has_ir)
            if self.state.has_ir and self.state.ir_degenerate:
                self.ir_dust_btn.setToolTip(tr("This film shows the picture in infrared too (B&W, Kodachrome), so IR Removal is off."))

            self._sync_overlay()
            self._sync_rails()
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.auto_dust_btn,
            self.right_click_btn,
            self.threshold_slider,
            self.auto_size_slider,
            self.manual_size_slider,
            self.pick_dust_btn,
            self.pick_scratch_btn,
            self.pick_line_btn,
            self.line_threshold_slider,
            self.clone_btn,
            self.clone_source_btn,
            self.clone_match_btn,
            self.clone_strength_slider,
            self.clone_feather_slider,
            self.ir_dust_btn,
            self.ir_threshold_slider,
            self.ir_method_btn,
            self.overlay_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
