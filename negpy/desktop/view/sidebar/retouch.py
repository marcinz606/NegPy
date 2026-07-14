from collections.abc import Mapping

from PyQt6.QtWidgets import QPushButton, QHBoxLayout
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.templates import (
    hint_label,
    section_subheader,
    set_hint_kind,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut


class RetouchSidebar(BaseSidebar):
    """
    Panel for dust removal and healing.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.retouch

        auto_row = QHBoxLayout()
        self.threshold_slider = CompactSlider("Threshold", 0.01, 1.0, conf.dust_threshold)
        self.threshold_slider.setToolTip("Minimum brightness delta to classify a pixel as dust")
        self.auto_size_slider = CompactSlider("Auto Size", 3.0, 8.0, float(conf.dust_size), step=1.0, precision=1, unit=" px")
        self.auto_size_slider.setToolTip("Maximum radius (px) of dust spots detected automatically")
        auto_row.addWidget(self.threshold_slider)
        auto_row.addWidget(self.auto_size_slider)
        self.layout.addLayout(auto_row)

        buttons_row = QHBoxLayout()
        self.auto_dust_btn = self._small_toggle(
            "fa5s.magic",
            "Auto Dust",
            conf.dust_remove,
            "Enable automatic dust removal using the Threshold and Auto Size settings above",
        )

        self.pick_dust_btn = self._tool_toggle(
            "fa5s.eye-dropper",
            "Heal Tool",
            tooltip_with_shortcut(
                "Toggle heal tool — click a dust spot to heal it. Right-click an existing heal overlay to delete it",
                "pick_dust",
            ),
        )
        self.pick_scratch_btn = self._tool_toggle(
            "fa5s.pen-nib",
            "Scratch Tool",
            "Heal a scratch or hair: click points along it, double-click or Enter to finish, Esc cancels. "
            "Backspace deletes the last entered point; right-click an existing scratch overlay to delete it",
        )

        buttons_row.addWidget(self.auto_dust_btn)
        buttons_row.addWidget(self.pick_dust_btn)
        buttons_row.addWidget(self.pick_scratch_btn)
        self.layout.addLayout(buttons_row)

        self.manual_size_slider = CompactSlider("Brush Size", 2.0, 16.0, float(conf.manual_dust_size), step=1.0, precision=1, unit=" px")
        self.manual_size_slider.setToolTip("Radius of the manual heal brush")
        self.layout.addWidget(self.manual_size_slider)

        self.heals_subheader = section_subheader("HEALS · 0")
        self.layout.addWidget(self.heals_subheader)

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

        self.ir_subheader = section_subheader("IR DUST")
        self.layout.addWidget(self.ir_subheader)

        ir_row = QHBoxLayout()
        self.ir_dust_btn = self._small_toggle(
            "fa5s.broom",
            "IR Dust",
            conf.ir_dust_remove,
            "Use scanner IR channel to detect and inpaint dust/scratches",
        )
        self.ir_threshold_slider = CompactSlider("IR Thresh", 0.05, 0.95, float(conf.ir_threshold))
        self.ir_threshold_slider.setToolTip("IR dust sensitivity — lower catches more (risk false positives); higher is conservative")
        ir_row.addWidget(self.ir_dust_btn, stretch=1)
        ir_row.addWidget(self.ir_threshold_slider, stretch=1)
        self.layout.addLayout(ir_row)

        self.ir_status_label = hint_label("")
        self.ir_status_label.setAccessibleName("Infrared dust repair status")
        self.layout.addWidget(self.ir_status_label)

        self.layout.addStretch()

        self._ir_enabled_tooltips = (
            (self.ir_subheader, self.ir_subheader.toolTip()),
            (self.ir_dust_btn, self.ir_dust_btn.toolTip()),
            (self.ir_threshold_slider, self.ir_threshold_slider.toolTip()),
        )
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

        self.ir_dust_btn.toggled.connect(lambda c: self.update_config_section("retouch", persist=True, render=True, ir_dust_remove=c))
        self.ir_threshold_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", readback_metrics=False, ir_threshold=float(v))
        )
        self.controller.metrics_available.connect(self._on_metrics_available)

    def _on_pick_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.DUST_PICK if checked else ToolMode.NONE)
        self.manual_size_slider.setVisible(checked or self.pick_scratch_btn.isChecked())

    def _on_scratch_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.SCRATCH_PICK if checked else ToolMode.NONE)
        self.manual_size_slider.setVisible(checked or self.pick_dust_btn.isChecked())

    def _set_ir_controls_enabled(self, enabled: bool) -> None:
        unavailable_tip = "No IR channel in this scan"
        for widget, enabled_tip in self._ir_enabled_tooltips:
            widget.setEnabled(enabled)
            widget.setToolTip(enabled_tip if enabled else unavailable_tip)

    def _on_metrics_available(self, metrics: Mapping[str, object]) -> None:
        """Expose whether the requested IR repair actually ran."""

        if not isinstance(metrics, Mapping):
            metrics = {}
        self._refresh_ir_status(self._metrics_for_active_file(metrics))

    def _metrics_for_active_file(
        self,
        metrics: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Reject late render metrics that belong to another displayed frame."""

        active_hash = self.state.current_file_hash
        if not active_hash or metrics.get("source_hash") != active_hash:
            return {}
        return metrics

    def _refresh_ir_status(self, metrics: Mapping[str, object]) -> None:
        if not self.state.has_ir:
            message = "IR repair unavailable · no IR channel"
            kind = "muted"
        elif not self.state.config.retouch.ir_dust_remove:
            message = "IR repair is off"
            kind = "muted"
        elif metrics.get("ir_dust_status") == "applied":
            pixels = metrics.get("ir_dust_mask_pixels")
            if type(pixels) is int and pixels >= 0:
                message = f"IR repair applied · {pixels:,} pixels repaired"
            else:
                message = "IR repair applied"
            kind = "muted"
        elif metrics.get("ir_dust_status") == "skipped":
            reason = metrics.get("ir_dust_skipped")
            detail = reason if isinstance(reason, str) and reason else "safety check failed"
            message = f"IR repair skipped · {detail}"
            kind = "warning"
        else:
            message = "IR repair is on · waiting for render"
            kind = "muted"
        self.ir_status_label.setText(message)
        self.ir_status_label.setToolTip(message)
        set_hint_kind(self.ir_status_label, kind)

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
            self.heals_subheader.setText(f"HEALS · {num_heals}")

            has_heals = num_heals > 0
            self.undo_btn.setEnabled(has_heals)
            self.clear_btn.setEnabled(has_heals)

            self.ir_dust_btn.setChecked(conf.ir_dust_remove)
            self.ir_threshold_slider.setValue(float(conf.ir_threshold))
            self._set_ir_controls_enabled(self.state.has_ir)
            self._refresh_ir_status(
                self._metrics_for_active_file(self.state.last_metrics)
            )
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
        ]
        for w in widgets:
            w.blockSignals(blocked)
