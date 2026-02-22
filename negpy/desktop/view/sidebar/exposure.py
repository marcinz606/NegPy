from PyQt6.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QComboBox,
)
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode


class ExposureSidebar(BaseSidebar):
    """
    Adjustment panel for White Balance and Characterstic Curve (Sigmoid).
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.exposure

        self.region_combo = QComboBox()
        self.region_combo.addItems(["Global", "Shadows", "Highlights"])
        self.region_combo.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 4px;")
        self.layout.addWidget(self.region_combo)

        self.cyan_slider = CompactSlider("Cyan", -1.0, 1.0, conf.wb_cyan, color="#00b1b1", has_neutral=True)
        self.cyan_slider.slider.setObjectName("cyan_slider")
        self.magenta_slider = CompactSlider("Magenta", -1.0, 1.0, conf.wb_magenta, color="#b100b1", has_neutral=True)
        self.magenta_slider.slider.setObjectName("magenta_slider")
        self.yellow_slider = CompactSlider("Yellow", -1.0, 1.0, conf.wb_yellow, color="#b1b100", has_neutral=True)
        self.yellow_slider.slider.setObjectName("yellow_slider")
        self.layout.addWidget(self.cyan_slider)
        self.layout.addWidget(self.magenta_slider)
        self.layout.addWidget(self.yellow_slider)

        wb_btn_row = QHBoxLayout()
        self.pick_wb_btn = QPushButton(" Pick WB")
        self.pick_wb_btn.setCheckable(True)
        self.pick_wb_btn.setIcon(qta.icon("fa5s.eye-dropper", color=THEME.text_primary))
        self.pick_wb_btn.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 8px;")

        self.camera_wb_btn = QPushButton(" Camera WB")
        self.camera_wb_btn.setCheckable(True)
        self.camera_wb_btn.setChecked(conf.use_camera_wb)
        self.camera_wb_btn.setIcon(qta.icon("fa5s.camera", color=THEME.text_primary))
        self.camera_wb_btn.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 8px;")

        wb_btn_row.addWidget(self.pick_wb_btn)
        wb_btn_row.addWidget(self.camera_wb_btn)
        self.layout.addLayout(wb_btn_row)

        self.density_slider = CompactSlider("Density", -0.0, 2.0, conf.density)
        self.grade_slider = CompactSlider("Grade", 0.0, 5.0, conf.grade)

        self.layout.addWidget(self.density_slider)
        self.layout.addWidget(self.grade_slider)

        self.shadows_slider = CompactSlider("Shadows", -1.0, 1.0, conf.shadows, has_neutral=True)
        self.layout.addWidget(self.shadows_slider)

        self.toe_slider = CompactSlider("Toe", -1.0, 1.0, conf.toe)
        self.layout.addWidget(self.toe_slider)

        toe_row = QHBoxLayout()
        self.toe_w_slider = CompactSlider("Width", 0.1, 5.0, conf.toe_width)
        self.toe_h_slider = CompactSlider("Hardness", 0.1, 5.0, conf.toe_hardness)
        toe_row.addWidget(self.toe_w_slider)
        toe_row.addWidget(self.toe_h_slider)
        self.layout.addLayout(toe_row)

        self.highlights_slider = CompactSlider("Highlights", -1.0, 1.0, conf.highlights, has_neutral=True)
        self.layout.addWidget(self.highlights_slider)

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
        self.region_combo.currentIndexChanged.connect(self.sync_ui)

        self.cyan_slider.valueChanged.connect(self._on_cyan_changed)
        self.magenta_slider.valueChanged.connect(self._on_magenta_changed)
        self.yellow_slider.valueChanged.connect(self._on_yellow_changed)

        self.density_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, density=v))
        self.grade_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, grade=v))

        self.shadows_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, shadows=v))
        self.highlights_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, highlights=v))

        self.pick_wb_btn.toggled.connect(self._on_pick_wb_toggled)
        self.camera_wb_btn.toggled.connect(self._on_camera_wb_toggled)

        self.toe_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, toe=v))
        self.toe_w_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, toe_width=v))
        self.toe_h_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, toe_hardness=v))

        self.sh_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, shoulder=v))
        self.sh_w_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, shoulder_width=v))
        self.sh_h_slider.valueChanged.connect(lambda v: self.update_config_section("exposure", readback_metrics=False, shoulder_hardness=v))

    def _on_cyan_changed(self, v: float) -> None:
        idx = self.region_combo.currentIndex()
        if idx == 0:
            self.update_config_section("exposure", readback_metrics=False, wb_cyan=v)
        elif idx == 1:
            self.update_config_section("exposure", readback_metrics=False, shadow_cyan=v)
        elif idx == 2:
            self.update_config_section("exposure", readback_metrics=False, highlight_cyan=v)

    def _on_magenta_changed(self, v: float) -> None:
        idx = self.region_combo.currentIndex()
        if idx == 0:
            self.update_config_section("exposure", readback_metrics=False, wb_magenta=v)
        elif idx == 1:
            self.update_config_section("exposure", readback_metrics=False, shadow_magenta=v)
        elif idx == 2:
            self.update_config_section("exposure", readback_metrics=False, highlight_magenta=v)

    def _on_yellow_changed(self, v: float) -> None:
        idx = self.region_combo.currentIndex()
        if idx == 0:
            self.update_config_section("exposure", readback_metrics=False, wb_yellow=v)
        elif idx == 1:
            self.update_config_section("exposure", readback_metrics=False, shadow_yellow=v)
        elif idx == 2:
            self.update_config_section("exposure", readback_metrics=False, highlight_yellow=v)

    def _on_pick_wb_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.WB_PICK if checked else ToolMode.NONE)

    def _on_camera_wb_toggled(self, checked: bool) -> None:
        self.update_config_section("exposure", render=False, persist=True, use_camera_wb=checked)
        if self.state.current_file_path:
            # Clear local analysis to force fresh normalization bounds for new WB
            self.update_config_section(
                "process",
                render=False,
                persist=True,
                local_floors=(0.0, 0.0, 0.0),
                local_ceils=(0.0, 0.0, 0.0),
                local_shadow_cast=(0.0, 0.0, 0.0),
            )
            self.controller.load_file(self.state.current_file_path)

    def sync_ui(self) -> None:
        conf = self.state.config.exposure

        self.block_signals(True)
        try:
            idx = self.region_combo.currentIndex()
            if idx == 0:
                self.cyan_slider.setValue(conf.wb_cyan)
                self.magenta_slider.setValue(conf.wb_magenta)
                self.yellow_slider.setValue(conf.wb_yellow)
            elif idx == 1:
                self.cyan_slider.setValue(conf.shadow_cyan)
                self.magenta_slider.setValue(conf.shadow_magenta)
                self.yellow_slider.setValue(conf.shadow_yellow)
            elif idx == 2:
                self.cyan_slider.setValue(conf.highlight_cyan)
                self.magenta_slider.setValue(conf.highlight_magenta)
                self.yellow_slider.setValue(conf.highlight_yellow)

            self.pick_wb_btn.setChecked(self.state.active_tool == ToolMode.WB_PICK)
            self.camera_wb_btn.setChecked(conf.use_camera_wb)

            self.density_slider.setValue(conf.density)
            self.grade_slider.setValue(conf.grade)

            self.shadows_slider.setValue(conf.shadows)
            self.highlights_slider.setValue(conf.highlights)

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
            self.region_combo,
            self.cyan_slider,
            self.magenta_slider,
            self.yellow_slider,
            self.pick_wb_btn,
            self.camera_wb_btn,
            self.density_slider,
            self.grade_slider,
            self.shadows_slider,
            self.highlights_slider,
            self.toe_slider,
            self.toe_w_slider,
            self.toe_h_slider,
            self.sh_slider,
            self.sh_w_slider,
            self.sh_h_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
