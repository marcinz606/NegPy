from PyQt6.QtWidgets import (
    QPushButton,
    QComboBox,
    QLabel,
    QHBoxLayout,
    QInputDialog,
)
import qtawesome as qta
from src.desktop.view.widgets.sliders import SignalSlider, CompactSlider
from src.desktop.view.styles.theme import THEME
from src.desktop.view.sidebar.base import BaseSidebar
from src.desktop.session import ToolMode
from src.domain.models import ProcessMode


class ExposureSidebar(BaseSidebar):
    """
    Adjustment panel for Density, Grade, and Process Mode.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.exposure
        mode = self.state.config.process_mode

        mode_label = QLabel("Process Mode:")
        mode_label.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold;"
        )
        self.layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([m.value for m in ProcessMode])
        self.mode_combo.setCurrentText(mode)
        self.mode_combo.setStyleSheet(
            f"font-size: {THEME.font_size_base}px; padding: 4px;"
        )
        self.layout.addWidget(self.mode_combo)

        buffer_row = QHBoxLayout()
        self.analysis_buffer_slider = SignalSlider(
            "Analysis Buffer", 0.0, 0.25, conf.analysis_buffer
        )
        self.analyze_roll_btn = QPushButton()
        self.analyze_roll_btn.setFixedSize(32, 32)
        self.analyze_roll_btn.setIcon(qta.icon("fa5s.search", color=THEME.text_primary))
        self.analyze_roll_btn.setToolTip("Analyze entire roll to find average baseline")

        self.use_roll_avg_btn = QPushButton()
        self.use_roll_avg_btn.setCheckable(True)
        self.use_roll_avg_btn.setFixedSize(32, 32)
        self.use_roll_avg_btn.setToolTip("Switch between Roll average and Local auto")
        self._update_roll_avg_btn_style(conf.use_roll_average)

        buffer_row.addWidget(self.analysis_buffer_slider)
        buffer_row.addWidget(self.analyze_roll_btn)
        buffer_row.addWidget(self.use_roll_avg_btn)
        self.layout.addLayout(buffer_row)

        roll_row = QHBoxLayout()
        self.roll_combo = QComboBox()
        self.roll_combo.setPlaceholderText("Select Saved Roll...")
        self._refresh_rolls()

        self.load_roll_btn = QPushButton()
        self.load_roll_btn.setFixedSize(32, 32)
        self.load_roll_btn.setIcon(qta.icon("fa5s.upload", color=THEME.text_primary))
        self.load_roll_btn.setToolTip("Apply selected Roll to session")

        self.save_roll_btn = QPushButton()
        self.save_roll_btn.setFixedSize(32, 32)
        self.save_roll_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))
        self.save_roll_btn.setToolTip("Save current normalization as new Roll")

        self.delete_roll_btn = QPushButton()
        self.delete_roll_btn.setFixedSize(32, 32)
        self.delete_roll_btn.setIcon(qta.icon("fa5s.trash", color=THEME.text_primary))
        self.delete_roll_btn.setToolTip("Delete selected Roll")

        roll_row.addWidget(self.roll_combo, stretch=1)
        roll_row.addWidget(self.load_roll_btn)
        roll_row.addWidget(self.save_roll_btn)
        roll_row.addWidget(self.delete_roll_btn)
        self.layout.addLayout(roll_row)

        wb_header = QLabel("White Balance")
        wb_header.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        self.layout.addWidget(wb_header)

        self.cyan_slider = SignalSlider(
            "Cyan", -1.0, 1.0, conf.wb_cyan, color="#00b1b1"
        )
        self.magenta_slider = SignalSlider(
            "Magenta", -1.0, 1.0, conf.wb_magenta, color="#b100b1"
        )
        self.yellow_slider = SignalSlider(
            "Yellow", -1.0, 1.0, conf.wb_yellow, color="#b1b100"
        )
        self.layout.addWidget(self.cyan_slider)
        self.layout.addWidget(self.magenta_slider)
        self.layout.addWidget(self.yellow_slider)

        wb_btn_row = QHBoxLayout()
        self.pick_wb_btn = QPushButton(" Pick WB")
        self.pick_wb_btn.setCheckable(True)
        self.pick_wb_btn.setIcon(qta.icon("fa5s.eye-dropper", color=THEME.text_primary))
        self.pick_wb_btn.setStyleSheet(
            f"font-size: {THEME.font_size_base}px; padding: 8px;"
        )

        self.camera_wb_btn = QPushButton(" Camera WB")
        self.camera_wb_btn.setCheckable(True)
        self.camera_wb_btn.setChecked(conf.use_camera_wb)
        self.camera_wb_btn.setIcon(qta.icon("fa5s.camera", color=THEME.text_primary))
        self.camera_wb_btn.setStyleSheet(
            f"font-size: {THEME.font_size_base}px; padding: 8px;"
        )

        wb_btn_row.addWidget(self.pick_wb_btn)
        wb_btn_row.addWidget(self.camera_wb_btn)
        self.layout.addLayout(wb_btn_row)

        basics_header = QLabel("Print Exposure")
        basics_header.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        self.layout.addWidget(basics_header)

        self.density_slider = SignalSlider("Density", -0.0, 2.0, conf.density)
        self.grade_slider = SignalSlider("Grade", 0.0, 5.0, conf.grade)

        self.layout.addWidget(self.density_slider)
        self.layout.addWidget(self.grade_slider)

        toe_label = QLabel("Toe (Shadows)")
        toe_label.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        self.layout.addWidget(toe_label)

        self.toe_slider = CompactSlider("Toe", -1.0, 1.0, conf.toe)
        self.layout.addWidget(self.toe_slider)

        toe_row = QHBoxLayout()
        self.toe_w_slider = CompactSlider("Width", 0.1, 5.0, conf.toe_width)
        self.toe_h_slider = CompactSlider("Hardness", 0.1, 5.0, conf.toe_hardness)
        toe_row.addWidget(self.toe_w_slider)
        toe_row.addWidget(self.toe_h_slider)
        self.layout.addLayout(toe_row)

        shld_label = QLabel("Shoulder (Highlights)")
        shld_label.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        self.layout.addWidget(shld_label)

        self.sh_slider = CompactSlider("Shoulder", -1.0, 1.0, conf.shoulder)
        self.layout.addWidget(self.sh_slider)

        sh_row = QHBoxLayout()
        self.sh_w_slider = CompactSlider("Width", 0.1, 5.0, conf.shoulder_width)
        self.sh_h_slider = CompactSlider("Hardness", 0.1, 5.0, conf.shoulder_hardness)
        sh_row.addWidget(self.sh_w_slider)
        sh_row.addWidget(self.sh_h_slider)
        self.layout.addLayout(sh_row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        self.cyan_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, wb_cyan=v
            )
        )
        self.magenta_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, wb_magenta=v
            )
        )
        self.yellow_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, wb_yellow=v
            )
        )

        self.density_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, density=v
            )
        )
        self.grade_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, grade=v
            )
        )
        self.analysis_buffer_slider.valueChanged.connect(self._on_buffer_changed)
        self.analyze_roll_btn.clicked.connect(
            self.controller.request_batch_normalization
        )
        self.use_roll_avg_btn.toggled.connect(self._on_use_roll_average_toggled)

        self.load_roll_btn.clicked.connect(self._on_load_roll)
        self.save_roll_btn.clicked.connect(self._on_save_roll)
        self.delete_roll_btn.clicked.connect(self._on_delete_roll)

        self.pick_wb_btn.toggled.connect(self._on_pick_wb_toggled)
        self.camera_wb_btn.toggled.connect(self._on_camera_wb_toggled)

        self.toe_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, toe=v
            )
        )
        self.toe_w_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, toe_width=v
            )
        )
        self.toe_h_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, toe_hardness=v
            )
        )

        self.sh_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, shoulder=v
            )
        )
        self.sh_w_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, shoulder_width=v
            )
        )
        self.sh_h_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, shoulder_hardness=v
            )
        )

    def _on_pick_wb_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.WB_PICK if checked else ToolMode.NONE)

    def _on_camera_wb_toggled(self, checked: bool) -> None:
        self.update_config_section(
            "exposure", render=False, persist=True, use_camera_wb=checked
        )
        if self.state.current_file_path:
            self.controller.load_file(self.state.current_file_path)

    def _on_mode_changed(self, mode: str) -> None:
        self.update_config_root(process_mode=mode, persist=True)

    def _on_buffer_changed(self, val: float) -> None:
        """
        Updates analysis buffer and forces local re-analysis.
        """
        self.update_config_section(
            "exposure",
            persist=True,
            render=True,
            analysis_buffer=val,
            local_floors=(0.0, 0.0, 0.0),
            local_ceils=(0.0, 0.0, 0.0),
        )

    def _on_use_roll_average_toggled(self, checked: bool) -> None:
        """
        Toggles between Roll-wide baseline and Local auto-exposure.
        Forcing re-analysis when switching to Local.
        """
        if not checked:
            self.update_config_section(
                "exposure",
                persist=True,
                render=True,
                use_roll_average=False,
                local_floors=(0.0, 0.0, 0.0),
                local_ceils=(0.0, 0.0, 0.0),
                roll_name=None,
            )
        else:
            self.update_config_section(
                "exposure", persist=True, render=True, use_roll_average=True
            )

    def _refresh_rolls(self) -> None:
        """
        Populates roll dropdown from database.
        """
        current = self.roll_combo.currentText()
        self.roll_combo.blockSignals(True)
        self.roll_combo.clear()
        rolls = self.controller.session.repo.list_normalization_rolls()
        self.roll_combo.addItems(rolls)
        if current in rolls:
            self.roll_combo.setCurrentText(current)
        else:
            self.roll_combo.setCurrentIndex(-1)
        self.roll_combo.blockSignals(False)

    def _on_load_roll(self) -> None:
        """
        Applies selected roll to session.
        """
        name = self.roll_combo.currentText()
        if name:
            self.controller.apply_normalization_roll(name)

    def _on_save_roll(self) -> None:
        """
        Prompts user for name and saves current normalization.
        """
        name, ok = QInputDialog.getText(self, "Save Roll", "Enter name for this roll:")
        if ok and name:
            self.controller.save_current_normalization_as_roll(name)
            self._refresh_rolls()
            self.roll_combo.setCurrentText(name)

    def _on_delete_roll(self) -> None:
        """
        Removes selected roll from DB.
        """
        name = self.roll_combo.currentText()
        if name:
            self.controller.session.repo.delete_normalization_roll(name)
            self._refresh_rolls()

    def _update_roll_avg_btn_style(self, checked: bool) -> None:
        """
        Updates button icon and color based on active state.
        """
        if checked:
            self.use_roll_avg_btn.setIcon(qta.icon("fa5s.lock", color="white"))
            self.use_roll_avg_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME.accent_primary};
                    border-radius: 4px;
                }}
            """)
        else:
            self.use_roll_avg_btn.setIcon(
                qta.icon("fa5s.lock-open", color=THEME.text_primary)
            )
            self.use_roll_avg_btn.setStyleSheet("")

    def sync_ui(self) -> None:
        conf = self.state.config.exposure
        mode = self.state.config.process_mode

        self.block_signals(True)
        try:
            self.mode_combo.setCurrentText(mode)
            self.cyan_slider.setValue(conf.wb_cyan)
            self.magenta_slider.setValue(conf.wb_magenta)
            self.yellow_slider.setValue(conf.wb_yellow)

            self.pick_wb_btn.setChecked(self.state.active_tool == ToolMode.WB_PICK)
            self.camera_wb_btn.setChecked(conf.use_camera_wb)

            self.density_slider.setValue(conf.density)
            self.grade_slider.setValue(conf.grade)
            self.analysis_buffer_slider.setValue(conf.analysis_buffer)
            self.use_roll_avg_btn.setChecked(conf.use_roll_average)
            self._update_roll_avg_btn_style(conf.use_roll_average)
            self._refresh_rolls()
            if conf.roll_name:
                self.roll_combo.setCurrentText(conf.roll_name)

            self.toe_slider.setValue(conf.toe)
            self.toe_w_slider.setValue(conf.toe_width)
            self.toe_h_slider.setValue(conf.toe_hardness)

            self.sh_slider.setValue(conf.shoulder)
            self.sh_w_slider.setValue(conf.shoulder_width)
            self.sh_h_slider.setValue(conf.shoulder_hardness)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all sliders and buttons.
        """
        widgets = [
            self.mode_combo,
            self.cyan_slider,
            self.magenta_slider,
            self.yellow_slider,
            self.pick_wb_btn,
            self.camera_wb_btn,
            self.density_slider,
            self.grade_slider,
            self.analysis_buffer_slider,
            self.analyze_roll_btn,
            self.use_roll_avg_btn,
            self.roll_combo,
            self.load_roll_btn,
            self.toe_slider,
            self.toe_w_slider,
            self.toe_h_slider,
            self.sh_slider,
            self.sh_w_slider,
            self.sh_h_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
