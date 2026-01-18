from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QHBoxLayout,
)
from src.desktop.view.widgets.sliders import SignalSlider, CompactSlider
from src.desktop.view.styles.theme import THEME
from src.desktop.controller import AppController
from src.desktop.session import ToolMode


class ExposureSidebar(QWidget):
    """
    Adjustment panel for Density, Grade, and Process Mode.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)
        layout.setSpacing(12)

        conf = self.state.config.exposure
        mode = self.state.config.process_mode

        # 1. Process Mode
        mode_label = QLabel("Process Mode:")
        mode_label.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold;"
        )
        layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["C41", "B&W"])
        self.mode_combo.setCurrentText(mode)
        self.mode_combo.setStyleSheet(
            f"font-size: {THEME.font_size_base}px; padding: 4px;"
        )
        layout.addWidget(self.mode_combo)

        # 2. WB Sliders
        wb_header = QLabel("White Balance (CMY Shifts)")
        wb_header.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        layout.addWidget(wb_header)

        cmy_row = QHBoxLayout()
        self.cyan_slider = CompactSlider(
            "Cyan", -1.0, 1.0, conf.wb_cyan, color="#00b1b1"
        )
        self.magenta_slider = CompactSlider(
            "Magenta", -1.0, 1.0, conf.wb_magenta, color="#b100b1"
        )
        self.yellow_slider = CompactSlider(
            "Yellow", -1.0, 1.0, conf.wb_yellow, color="#b1b100"
        )
        cmy_row.addWidget(self.cyan_slider)
        cmy_row.addWidget(self.magenta_slider)
        cmy_row.addWidget(self.yellow_slider)
        layout.addLayout(cmy_row)

        self.pick_wb_btn = QPushButton("Pick WB")
        self.pick_wb_btn.setCheckable(True)
        self.pick_wb_btn.setStyleSheet(
            f"font-size: {THEME.font_size_base}px; padding: 8px;"
        )
        layout.addWidget(self.pick_wb_btn)

        # 3. Print Basics
        basics_header = QLabel("Print Exposure")
        basics_header.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        layout.addWidget(basics_header)

        self.density_slider = SignalSlider("Density", -1.0, 3.0, conf.density)
        self.grade_slider = SignalSlider("Grade", 0.0, 5.0, conf.grade)

        layout.addWidget(self.density_slider)
        layout.addWidget(self.grade_slider)

        # 4. H&D Curve (Toe/Shoulder)
        toe_label = QLabel("Toe Control")
        toe_label.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        layout.addWidget(toe_label)

        toe_row = QHBoxLayout()
        self.toe_slider = CompactSlider("Toe", -1.0, 1.0, conf.toe)
        self.toe_w_slider = CompactSlider("Width", 0.1, 5.0, conf.toe_width)
        self.toe_h_slider = CompactSlider("Hardness", 0.1, 5.0, conf.toe_hardness)
        toe_row.addWidget(self.toe_slider)
        toe_row.addWidget(self.toe_w_slider)
        toe_row.addWidget(self.toe_h_slider)
        layout.addLayout(toe_row)

        shld_label = QLabel("Shoulder Control")
        shld_label.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        layout.addWidget(shld_label)

        sh_row = QHBoxLayout()
        self.sh_slider = CompactSlider("Shoulder", -1.0, 1.0, conf.shoulder)
        self.sh_w_slider = CompactSlider("Width", 0.1, 5.0, conf.shoulder_width)
        self.sh_h_slider = CompactSlider("Hardness", 0.1, 5.0, conf.shoulder_hardness)
        sh_row.addWidget(self.sh_slider)
        sh_row.addWidget(self.sh_w_slider)
        sh_row.addWidget(self.sh_h_slider)
        layout.addLayout(sh_row)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.cyan_slider.valueChanged.connect(lambda v: self._update_wb("wb_cyan", v))
        self.magenta_slider.valueChanged.connect(
            lambda v: self._update_wb("wb_magenta", v)
        )
        self.yellow_slider.valueChanged.connect(
            lambda v: self._update_wb("wb_yellow", v)
        )
        self.density_slider.valueChanged.connect(self._on_density_changed)
        self.grade_slider.valueChanged.connect(self._on_grade_changed)
        self.pick_wb_btn.toggled.connect(self._on_pick_wb_toggled)

        # Curve signals
        self.toe_slider.valueChanged.connect(lambda v: self._update_exp("toe", v))
        self.toe_w_slider.valueChanged.connect(
            lambda v: self._update_exp("toe_width", v)
        )
        self.toe_h_slider.valueChanged.connect(
            lambda v: self._update_exp("toe_hardness", v)
        )
        self.sh_slider.valueChanged.connect(lambda v: self._update_exp("shoulder", v))
        self.sh_w_slider.valueChanged.connect(
            lambda v: self._update_exp("shoulder_width", v)
        )
        self.sh_h_slider.valueChanged.connect(
            lambda v: self._update_exp("shoulder_hardness", v)
        )

    def _update_wb(self, field: str, val: float) -> None:
        from dataclasses import replace

        new_exposure = replace(self.state.config.exposure, **{field: val})
        new_config = replace(self.state.config, exposure=new_exposure)
        self.controller.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def _update_exp(self, field: str, val: float) -> None:
        from dataclasses import replace

        new_exposure = replace(self.state.config.exposure, **{field: val})
        self.controller.session.update_config(
            replace(self.state.config, exposure=new_exposure), persist=True
        )
        self.controller.request_render()

    def _on_pick_wb_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.WB_PICK if checked else ToolMode.NONE)

    def _on_density_changed(self, val: float) -> None:
        from dataclasses import replace

        new_exposure = replace(self.state.config.exposure, density=val)
        new_config = replace(self.state.config, exposure=new_exposure)
        self.controller.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def _on_grade_changed(self, val: float) -> None:
        from dataclasses import replace

        new_exposure = replace(self.state.config.exposure, grade=val)
        self.controller.session.update_config(
            replace(self.state.config, exposure=new_exposure), persist=True
        )
        self.controller.request_render()

    def _on_mode_changed(self, mode: str) -> None:
        from dataclasses import replace

        new_config = replace(self.state.config, process_mode=mode)
        self.controller.session.update_config(new_config)
        self.controller.request_render()

    def sync_ui(self) -> None:
        """
        Updates slider values from current state config.
        """
        conf = self.state.config.exposure
        mode = self.state.config.process_mode

        self.mode_combo.blockSignals(True)
        self.cyan_slider.blockSignals(True)
        self.magenta_slider.blockSignals(True)
        self.yellow_slider.blockSignals(True)
        self.density_slider.blockSignals(True)
        self.grade_slider.blockSignals(True)
        self.toe_slider.blockSignals(True)
        self.toe_w_slider.blockSignals(True)
        self.toe_h_slider.blockSignals(True)
        self.sh_slider.blockSignals(True)
        self.sh_w_slider.blockSignals(True)
        self.sh_h_slider.blockSignals(True)

        try:
            self.mode_combo.setCurrentText(mode)
            self.cyan_slider.setValue(conf.wb_cyan)
            self.magenta_slider.setValue(conf.wb_magenta)
            self.yellow_slider.setValue(conf.wb_yellow)
            self.density_slider.setValue(conf.density)
            self.grade_slider.setValue(conf.grade)
            self.toe_slider.setValue(conf.toe)
            self.toe_w_slider.setValue(conf.toe_width)
            self.toe_h_slider.setValue(conf.toe_hardness)
            self.sh_slider.setValue(conf.shoulder)
            self.sh_w_slider.setValue(conf.shoulder_width)
            self.sh_h_slider.setValue(conf.shoulder_hardness)
        finally:
            self.mode_combo.blockSignals(False)
            self.cyan_slider.blockSignals(False)
            self.magenta_slider.blockSignals(False)
            self.yellow_slider.blockSignals(False)
            self.density_slider.blockSignals(False)
            self.grade_slider.blockSignals(False)
            self.toe_slider.blockSignals(False)
            self.toe_w_slider.blockSignals(False)
            self.toe_h_slider.blockSignals(False)
            self.sh_slider.blockSignals(False)
            self.sh_w_slider.blockSignals(False)
            self.sh_h_slider.blockSignals(False)
