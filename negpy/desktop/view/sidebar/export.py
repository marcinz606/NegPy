import os

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.color_mgmt import ColorService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry


class ExportSidebar(BaseSidebar):
    """
    Panel for export settings and batch processing.
    """

    @staticmethod
    def _icc_row_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setFixedWidth(52)
        return label

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)

        self.layout.addWidget(section_subheader("ICC"))

        self.soft_proof_checkbox = QCheckBox("Soft proof")
        self.soft_proof_checkbox.setChecked(self.state.soft_proof_enabled)
        self.soft_proof_checkbox.setToolTip("Simulate the Output profile (incl. paper/printer) in the preview")
        self.layout.addWidget(self.soft_proof_checkbox)

        enum_mapped = {ColorSpaceRegistry.get_icc_path(cs.value) for cs in ColorSpace}
        enum_mapped.discard(None)
        custom_profiles = [p for p in ColorService.get_available_profiles() if p not in enum_mapped]

        self.input_profiles = ["None"] + custom_profiles
        self.input_combo = QComboBox()
        self.input_combo.addItems([os.path.basename(p) for p in self.input_profiles])
        self.input_combo.setToolTip("Source/input ICC profile — soft-proofed in the preview")
        in_path = self.state.icc_input_path
        self.input_combo.setCurrentText(os.path.basename(in_path) if in_path else "None")
        in_row = QHBoxLayout()
        in_row.addWidget(self._icc_row_label("Input"))
        in_row.addWidget(self.input_combo)
        self.layout.addLayout(in_row)

        self.output_spaces = [cs.value for cs in ColorSpace]
        self.output_profiles = custom_profiles
        self.output_map = list(self.output_spaces) + list(self.output_profiles)
        self.output_combo = QComboBox()
        self.output_combo.addItems(self.output_spaces + [os.path.basename(p) for p in self.output_profiles])
        self.output_combo.setToolTip("Output color space or custom ICC profile — soft-proofed in the preview")
        out_path = self.state.icc_output_path
        self.output_combo.setCurrentText(
            os.path.basename(out_path) if out_path else self.state.config.export.export_color_space
        )
        out_row = QHBoxLayout()
        out_row.addWidget(self._icc_row_label("Output"))
        out_row.addWidget(self.output_combo)
        self.layout.addLayout(out_row)

        self.display_spaces = [
            ColorSpace.SRGB.value,
            ColorSpace.P3_D65.value,
            ColorSpace.ADOBE_RGB.value,
            ColorSpace.REC2020.value,
            ColorSpace.PROPHOTO.value,
        ]
        self.display_map = [None] + self.display_spaces
        self.display_combo = QComboBox()
        self.display_combo.addItems(["As detected"] + self.display_spaces)
        self.display_combo.setToolTip(
            "Monitor profile the preview is displayed on (affects preview only, not export)"
        )
        override = self.state.monitor_profile_override
        self.display_combo.setCurrentText(
            override if override in self.display_spaces else "As detected"
        )
        disp_row = QHBoxLayout()
        disp_row.addWidget(self._icc_row_label("Display"))
        disp_row.addWidget(self.display_combo)
        self.layout.addLayout(disp_row)

        self.display_detected_label = QLabel()
        self.display_detected_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 10px;")
        self.layout.addWidget(self.display_detected_label)
        self._refresh_display_info()

        # Presets section
        self.layout.addWidget(section_subheader("PRESETS"))

        self._presets_scroll = QScrollArea()
        self._presets_scroll.setWidgetResizable(True)
        self._presets_scroll.setMaximumHeight(160)
        self._presets_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._presets_scroll.setStyleSheet(
            f"QScrollArea {{ border: 1px solid {THEME.border_primary}; background: {THEME.bg_dark}; }}"
        )
        self._presets_container = QWidget()
        self._presets_container.setStyleSheet(f"background: {THEME.bg_dark};")
        self._presets_inner = QVBoxLayout(self._presets_container)
        self._presets_inner.setContentsMargins(4, 4, 4, 4)
        self._presets_inner.setSpacing(2)
        self._presets_scroll.setWidget(self._presets_container)
        self.layout.addWidget(self._presets_scroll)

        self._no_presets_label = QLabel("No presets — click Manage to add some.")
        self._no_presets_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: 10px;")
        self._no_presets_label.setWordWrap(True)
        self._presets_inner.addWidget(self._no_presets_label)
        self._preset_checkboxes: list[QCheckBox] = []

        manage_btn = QPushButton(" Manage Presets")
        manage_btn.setObjectName("manage_presets_btn")
        manage_btn.setIcon(qta.icon("fa5s.sliders-h", color=THEME.text_primary))
        manage_btn.clicked.connect(self._open_presets_dialog)
        self.layout.addWidget(manage_btn)

        self.layout.addWidget(section_subheader("BATCH"))

        batch_row = QHBoxLayout()
        self.batch_export_btn = QPushButton(" Export All")
        self.batch_export_btn.setObjectName("batch_export_btn")
        self.batch_export_btn.setFixedHeight(40)
        self.batch_export_btn.setIcon(qta.icon("fa5s.images", color=THEME.text_primary))

        self.apply_all_btn = QPushButton(" Sync export settings")
        self.apply_all_btn.setFixedHeight(40)
        self.apply_all_btn.setCheckable(True)
        self.apply_all_btn.setChecked(True)
        self.apply_all_btn.setToolTip("Apply current export settings to all files")
        self._update_apply_all_style(True)

        batch_row.addWidget(self.batch_export_btn)
        batch_row.addWidget(self.apply_all_btn)
        self.layout.addLayout(batch_row)

        self.layout.addStretch()

        self._rebuild_preset_rows()

    def _connect_signals(self) -> None:
        self.soft_proof_checkbox.toggled.connect(self.controller.set_soft_proof)
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        self.controller.monitor_profile_changed.connect(self._refresh_display_info)
        self.apply_all_btn.toggled.connect(self._update_apply_all_style)
        self.batch_export_btn.clicked.connect(
            lambda: self.controller.request_batch_export(
                override_settings=self.apply_all_btn.isChecked()
            )
        )

    def _rebuild_preset_rows(self) -> None:
        """Rebuild the preset checkbox list from state."""
        # Remove old checkboxes
        for cb in self._preset_checkboxes:
            self._presets_inner.removeWidget(cb)
            cb.deleteLater()
        self._preset_checkboxes.clear()

        presets = self.state.export_presets
        has_presets = bool(presets)
        self._no_presets_label.setVisible(not has_presets)

        for i, preset in enumerate(presets):
            cb = QCheckBox(preset.name)
            cb.setChecked(preset.enabled)
            cb.setStyleSheet(f"color: {THEME.text_primary};")
            cb.stateChanged.connect(lambda state, idx=i: self._on_preset_toggled(idx, state))
            self._presets_inner.addWidget(cb)
            self._preset_checkboxes.append(cb)

        self._presets_inner.addStretch()

    def _on_preset_toggled(self, idx: int, state: int) -> None:
        presets = self.state.export_presets
        if 0 <= idx < len(presets):
            presets[idx].enabled = state == Qt.CheckState.Checked.value
            self.controller.session.save_export_presets()

    def _open_presets_dialog(self) -> None:
        from negpy.desktop.view.widgets.export_presets_dialog import ExportPresetsDialog

        dlg = ExportPresetsDialog(self.state.export_presets, parent=self)
        dlg.presets_changed.connect(self._on_presets_changed)
        dlg.exec()

    def _on_presets_changed(self, presets: list) -> None:
        self.state.export_presets = presets
        self.controller.session.save_export_presets()
        self._rebuild_preset_rows()

    def _update_apply_all_style(self, checked: bool) -> None:
        if checked:
            self.apply_all_btn.setStyleSheet("""
                QPushButton {
                    background-color: #222222;
                    color: white;
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 4px;
                }
            """)
            self.apply_all_btn.setIcon(qta.icon("fa5s.clone", color="white"))
        else:
            self.apply_all_btn.setStyleSheet("font-weight: bold;")
            self.apply_all_btn.setIcon(qta.icon("fa5s.clone", color=THEME.text_primary))

    def _on_input_changed(self, index: int) -> None:
        path = self.input_profiles[index]
        self.state.icc_input_path = path if path != "None" else None
        self.controller.session.save_icc_prefs()
        self.controller.request_render()

    def _on_output_changed(self, index: int) -> None:
        if index < len(self.output_spaces):
            self.state.icc_output_path = None
            self.controller.session.save_icc_prefs()
            from dataclasses import replace
            new_export = replace(
                self.state.config.export,
                export_color_space=self.output_map[index],
            )
            self.controller.session.update_config(
                replace(self.state.config, export=new_export), persist=True, render=True
            )
            self.controller.request_render()
        else:
            self.state.icc_output_path = self.output_map[index]
            self.controller.session.save_icc_prefs()
            self.controller.request_render()

    def _on_display_changed(self, index: int) -> None:
        self.controller.set_monitor_override(self.display_map[index])

    def _refresh_display_info(self) -> None:
        from negpy.infrastructure.display.color_mgmt import profile_description

        desc = profile_description(self.state.monitor_icc_detected_bytes)
        self.display_detected_label.setText(f"Detected: {desc}")
        self.display_combo.setItemText(0, f"As detected ({desc})")

    def _current_export_color_space(self) -> str:
        idx = self.output_combo.currentIndex()
        if 0 <= idx < len(self.output_spaces):
            return self.output_map[idx]
        return self.state.config.export.export_color_space

    def sync_ui(self) -> None:
        self.block_signals(True)
        try:
            self.soft_proof_checkbox.setChecked(self.state.soft_proof_enabled)
            in_path = self.state.icc_input_path
            self.input_combo.setCurrentText(os.path.basename(in_path) if in_path else "None")
            out_path = self.state.icc_output_path
            self.output_combo.setCurrentText(
                os.path.basename(out_path) if out_path else self.state.config.export.export_color_space
            )
            override = self.state.monitor_profile_override
            self.display_combo.setCurrentText(
                override if override in self.display_spaces else "As detected"
            )
            self._refresh_display_info()
        finally:
            self.block_signals(False)

        self._rebuild_preset_rows()

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.soft_proof_checkbox,
            self.input_combo,
            self.output_combo,
            self.display_combo,
        ]
        for w in widgets:
            w.blockSignals(blocked)
