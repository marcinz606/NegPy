from PyQt6.QtWidgets import QComboBox, QPushButton, QHBoxLayout
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.templates import field_label, section_subheader, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.features.retouch.models import IR_METHOD_NEGPY, IR_METHOD_OPENICE

_IR_REMOVAL_TIP = (
    "Use the scanner's infrared channel to remove dust and scratches (invisible to the colour dyes): faint "
    "semi-transparent specks are divided back out to recover the image underneath, opaque cores and hairs are "
    "rebuilt from clean neighbouring film, and only defects too wide to see across are inpainted."
)
_IR_THRESH_TIP = "Lower catches more dust, higher is conservative. Smooth response, no cliff."
# Order matches _IR_METHOD_KEYS.
_IR_METHOD_LABELS = ("NegPy", "OpenICE")
_IR_METHOD_KEYS = (IR_METHOD_NEGPY, IR_METHOD_OPENICE)
_IR_METHOD_TIP = wrap_tooltip(
    "How the film under a defect is rebuilt. NegPy divides semi-transparent dust out, fills "
    "opaque cores with a weighted average of the clean film around them, and transplants grain "
    "from the nearest clean pixel. OpenICE works in log density and restores detail instead of "
    "averaging it away, keeping texture under a speck; it measures clear-film level and "
    "dye-to-infrared crosstalk per frame. Better on fine detail, less proven across scanners."
)
_OPTICAL_TIP = (
    "Find and remove dust specks on the visible scan by local contrast — no infrared channel needed. "
    "Set sensitivity with Threshold and Size below."
)


class RetouchSidebar(BaseSidebar):
    """
    Panel for dust removal and healing.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.retouch

        # --- Overlay inspector (applies to every detection source) ----------
        self.overlay_btn = QPushButton(" Overlay: Off")
        self.overlay_btn.setIcon(qta.icon("fa5s.eye", color=THEME.text_primary))
        self.overlay_btn.setToolTip(
            "Cycle the dust-detection overlay: Off → Marked → IR. Enable Optical / IR Removal so the overlay has detected spots to show."
        )
        self.layout.addWidget(self.overlay_btn)

        # --- OPTICAL REMOVAL (visible-scan speck detection) -----------------
        self.layout.addWidget(section_subheader("OPTICAL REMOVAL"))
        self.auto_dust_btn = self._small_toggle("fa5s.magic", "Optical Removal", conf.dust_remove, _OPTICAL_TIP)
        self.layout.addWidget(self.auto_dust_btn)
        auto_row = QHBoxLayout()
        self.threshold_slider = CompactSlider("Threshold", 0.01, 1.0, conf.dust_threshold)
        self.auto_size_slider = CompactSlider("Size", 3.0, 8.0, float(conf.dust_size), step=1.0, precision=1, unit=" px")
        auto_row.addWidget(self.threshold_slider)
        auto_row.addWidget(self.auto_size_slider)
        self.layout.addLayout(auto_row)

        # --- IR REMOVAL ------------------------------------------------------
        self.ir_subheader = section_subheader("IR REMOVAL")
        self.layout.addWidget(self.ir_subheader)
        method_row = QHBoxLayout()
        self.ir_method_label = field_label("Method")
        self.ir_method_combo = QComboBox()
        self.ir_method_combo.addItems(_IR_METHOD_LABELS)
        self.ir_method_combo.setToolTip(_IR_METHOD_TIP)
        method_row.addWidget(self.ir_method_label)
        method_row.addWidget(self.ir_method_combo, 1)
        self.layout.addLayout(method_row)

        self.ir_dust_btn = self._small_toggle("fa5s.broom", "IR Removal", conf.ir_dust_remove, _IR_REMOVAL_TIP)
        self.ir_threshold_slider = CompactSlider("IR Threshold", 0.05, 0.95, float(conf.ir_threshold))
        self.ir_threshold_slider.setToolTip(_IR_THRESH_TIP)
        ir_row = QHBoxLayout()
        ir_row.addWidget(self.ir_dust_btn, stretch=1)
        ir_row.addWidget(self.ir_threshold_slider, stretch=1)
        self.layout.addLayout(ir_row)

        # Restored whenever the scan has IR (never let a stale "No IR channel" tip linger).
        self._ir_tooltips = {
            self.ir_subheader: "Detect and remove dust/scratches using the scanner's infrared channel",
            self.ir_dust_btn: _IR_REMOVAL_TIP,
            self.ir_threshold_slider: _IR_THRESH_TIP,
            self.ir_method_label: _IR_METHOD_TIP,
            self.ir_method_combo: _IR_METHOD_TIP,
        }

        # --- MANUAL HEAL (bottom) -------------------------------------------
        self.heals_subheader = section_subheader("MANUAL HEAL · 0")
        self.layout.addWidget(self.heals_subheader)
        tools_row = QHBoxLayout()
        self.pick_dust_btn = self._tool_toggle(
            "fa5s.eye-dropper",
            "Heal Tool",
            "Paint over dust to heal it. The brush is a search area — only the marks inside it are repaired, clean grain is left alone",
        )
        self.pick_scratch_btn = self._tool_toggle(
            "fa5s.pen-nib",
            "Scratch Tool",
            "Heal a scratch or hair: click points along it, double-click or Enter to finish, Esc cancels. "
            "Backspace deletes the last entered point; right-click an existing scratch overlay to delete it",
        )
        tools_row.addWidget(self.pick_dust_btn)
        tools_row.addWidget(self.pick_scratch_btn)
        self.layout.addLayout(tools_row)

        self.manual_size_slider = CompactSlider("Brush Size", 2.0, 16.0, float(conf.manual_dust_size), step=1.0, precision=1, unit=" px")
        self.manual_size_slider.setToolTip("Diameter of the manual brush, matching the on-screen cursor")
        self.layout.addWidget(self.manual_size_slider)

        actions_row = QHBoxLayout()
        self.undo_btn = QPushButton(" Undo Last")
        self.undo_btn.setIcon(qta.icon("fa5s.undo", color=THEME.text_primary))
        self.undo_btn.setToolTip("Remove the most recent manual heal")

        self.clear_btn = QPushButton(" Clear All")
        self.clear_btn.setIcon(qta.icon("fa5s.trash-alt", color=THEME.text_primary))
        self.clear_btn.setToolTip("Remove all manual heals (auto-detected dust is unaffected)")

        actions_row.addWidget(self.undo_btn, 1)
        actions_row.addWidget(self.clear_btn, 1)
        self.layout.addLayout(actions_row)

        self.layout.addStretch()

        self._set_ir_controls_enabled(self.state.has_ir)

    def _connect_signals(self) -> None:
        self.auto_dust_btn.toggled.connect(lambda c: self.update_config_section("retouch", persist=True, render=True, dust_remove=c))
        self.threshold_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", readback_metrics=False, dust_threshold=v)
        )
        self.auto_size_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", readback_metrics=False, dust_size=int(v))  # TODO: precision loss from int cast
        )
        self.pick_dust_btn.toggled.connect(self._on_pick_toggled)
        self.pick_scratch_btn.toggled.connect(self._on_scratch_toggled)
        self.manual_size_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", render=False, persist=True, manual_dust_size=int(v))
        )
        self.undo_btn.clicked.connect(self.controller.undo_last_retouch)
        self.clear_btn.clicked.connect(self.controller.clear_retouch)
        self.overlay_btn.clicked.connect(self._on_overlay_clicked)

        self.ir_dust_btn.toggled.connect(
            lambda c: self.update_config_section("retouch", persist=True, render=True, ir_dust_remove=c, ir_attenuation=c)
        )
        self.ir_threshold_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", readback_metrics=False, ir_threshold=float(v))
        )
        self.ir_method_combo.currentIndexChanged.connect(
            lambda i: self.update_config_section("retouch", persist=True, render=True, ir_method=_IR_METHOD_KEYS[i])
        )

    def _on_overlay_clicked(self) -> None:
        # cycle_dust_overlay emits dust_overlay_changed (repaints the canvas) but
        # not config_updated, so the sidebar never re-syncs — update the label here.
        self.controller.cycle_dust_overlay()
        self._sync_overlay_label()

    def _sync_overlay_label(self) -> None:
        label = {"off": "Off", "marked": "Marked", "ir": "IR"}.get(self.state.dust_overlay_mode, "Off")
        self.overlay_btn.setText(f" Overlay: {label}")

    def _on_pick_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.DUST_PICK if checked else ToolMode.NONE)
        self.manual_size_slider.setVisible(checked or self.pick_scratch_btn.isChecked())

    def _on_scratch_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.SCRATCH_PICK if checked else ToolMode.NONE)
        self.manual_size_slider.setVisible(checked or self.pick_dust_btn.isChecked())

    def _set_ir_controls_enabled(self, enabled: bool) -> None:
        for w, tip in self._ir_tooltips.items():
            w.setEnabled(enabled)
            w.setToolTip(tip if enabled else "No IR channel in this scan")

    def sync_ui(self) -> None:
        conf = self.state.config.retouch
        self.block_signals(True)
        try:
            self.auto_dust_btn.setChecked(conf.dust_remove)
            self.threshold_slider.setValue(conf.dust_threshold)
            self.auto_size_slider.setValue(float(conf.dust_size))
            self.manual_size_slider.setValue(float(conf.manual_dust_size))
            self.pick_dust_btn.setChecked(self.state.active_tool == ToolMode.DUST_PICK)
            self.pick_scratch_btn.setChecked(self.state.active_tool == ToolMode.SCRATCH_PICK)
            self.manual_size_slider.setVisible(self.state.active_tool in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK))

            num_heals = len(conf.manual_dust_spots) + len(conf.manual_heal_strokes)
            self.heals_subheader.setText(f"MANUAL HEAL · {num_heals}")

            has_heals = num_heals > 0
            self.undo_btn.setEnabled(has_heals)
            self.clear_btn.setEnabled(has_heals)

            # Show unchecked on non-IR files: the config value is inert without an IR
            # plane, and a checked-but-greyed button reads as stuck-on.
            self.ir_dust_btn.setChecked(conf.ir_dust_remove and self.state.has_ir)
            self.ir_threshold_slider.setValue(float(conf.ir_threshold))
            method = conf.ir_method if conf.ir_method in _IR_METHOD_KEYS else IR_METHOD_NEGPY
            self.ir_method_combo.setCurrentIndex(_IR_METHOD_KEYS.index(method))
            self._set_ir_controls_enabled(self.state.has_ir)
            if self.state.has_ir and self.state.ir_degenerate:
                self.ir_dust_btn.setToolTip("IR channel carries image content (B&W / Kodachrome) — IR correction disabled for this frame")

            self._sync_overlay_label()
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.auto_dust_btn,
            self.threshold_slider,
            self.auto_size_slider,
            self.manual_size_slider,
            self.pick_dust_btn,
            self.pick_scratch_btn,
            self.ir_dust_btn,
            self.ir_threshold_slider,
            self.ir_method_combo,
        ]
        for w in widgets:
            w.blockSignals(blocked)
