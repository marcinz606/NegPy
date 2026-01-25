from PyQt6.QtWidgets import (
    QPushButton,
    QComboBox,
    QLabel,
    QHBoxLayout,
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

        self.analysis_buffer_slider = SignalSlider(
            "Analysis Buffer", 0.0, 0.25, conf.analysis_buffer
        )
        self.layout.addWidget(self.analysis_buffer_slider)

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
        self.analysis_buffer_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "exposure", readback_metrics=False, analysis_buffer=v
            )
        )

        self.pick_wb_btn.toggled.connect(self._on_pick_wb_toggled)
        self.camera_wb_btn.toggled.connect(self._on_camera_wb_toggled)

        # Curve signals
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

            self.toe_slider.setValue(conf.toe)
            self.toe_w_slider.setValue(conf.toe_width)
            self.toe_h_slider.setValue(conf.toe_hardness)

            self.sh_slider.setValue(conf.shoulder)
            self.sh_w_slider.setValue(conf.shoulder_width)
            self.sh_h_slider.setValue(conf.shoulder_hardness)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """Helper to block/unblock all sliders and buttons."""
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
            self.toe_slider,
            self.toe_w_slider,
            self.toe_h_slider,
            self.sh_slider,
            self.sh_w_slider,
            self.sh_h_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
