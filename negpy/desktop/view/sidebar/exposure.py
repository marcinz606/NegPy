import qtawesome as qta
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
)

from negpy.desktop.session import ToolMode
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider


class ExposureSidebar(BaseSidebar):
    """
    Adjustment panel for White Balance and Characterstic Curve (Sigmoid).
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.exposure

        region_row = QHBoxLayout()
        region_row.setSpacing(4)
        self.region_global_btn = QPushButton("Global")
        self.region_shadow_btn = QPushButton("Shadows")
        self.region_highlight_btn = QPushButton("Highlights")
        btn_style = f"font-size: {THEME.font_size_base}px; padding: 8px;"
        for btn in (self.region_global_btn, self.region_shadow_btn, self.region_highlight_btn):
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            region_row.addWidget(btn)
        self.region_global_btn.setChecked(True)
        self.region_btn_group = QButtonGroup(self)
        self.region_btn_group.setExclusive(True)
        self.region_btn_group.addButton(self.region_global_btn, 0)
        self.region_btn_group.addButton(self.region_shadow_btn, 1)
        self.region_btn_group.addButton(self.region_highlight_btn, 2)
        self.layout.addLayout(region_row)

        self.cyan_slider = CompactSlider("Cyan", -1.0, 1.0, conf.wb_cyan, color="#00b1b1", has_neutral=True)
        self.cyan_slider.slider.setObjectName("cyan_slider")
        self.cyan_slider.setToolTip("Cyan–Red white balance shift; applies to the selected region (Global/Shadows/Highlights)")
        self.magenta_slider = CompactSlider("Magenta", -1.0, 1.0, conf.wb_magenta, color="#b100b1", has_neutral=True)
        self.magenta_slider.slider.setObjectName("magenta_slider")
        self.magenta_slider.setToolTip("Magenta–Green white balance shift; applies to the selected region")
        self.yellow_slider = CompactSlider("Yellow", -1.0, 1.0, conf.wb_yellow, color="#b1b100", has_neutral=True)
        self.yellow_slider.slider.setObjectName("yellow_slider")
        self.yellow_slider.setToolTip("Yellow–Blue white balance shift; applies to the selected region")
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

        self.density_slider = CompactSlider("Density", 0.0, 2.0, conf.density)
        self.density_slider.setToolTip("Overall exposure — higher values darken the print (more density in the negative)")
        self.grade_slider = CompactSlider("Grade", 0.0, 5.0, conf.grade)
        self.grade_slider.setToolTip("Contrast grade: 0 = soft, 5 = very hard (controls sigmoid steepness)")

        self.layout.addWidget(self.density_slider)
        self.layout.addWidget(self.grade_slider)

        toe_row = QHBoxLayout()
        self.toe_w_slider = CompactSlider("Width", 0.1, 5.0, conf.toe_width)
        self.toe_w_slider.setToolTip("Width of the shadow toe transition zone")
        self.toe_slider = CompactSlider("Toe", -1.0, 1.0, conf.toe)
        self.toe_slider.setToolTip("Shadow toe lift: positive raises shadows, negative deepens blacks")
        toe_row.addWidget(self.toe_slider)
        toe_row.addWidget(self.toe_w_slider)
        self.layout.addLayout(toe_row)

        sh_row = QHBoxLayout()
        self.sh_slider = CompactSlider("Shoulder", -1.0, 1.0, conf.shoulder)
        self.sh_slider.setToolTip("Highlight shoulder roll: positive compresses highlights, negative extends them")
        self.sh_w_slider = CompactSlider("Width", 0.1, 5.0, conf.shoulder_width)
        self.sh_w_slider.setToolTip("Width of the highlight shoulder transition zone")
        sh_row.addWidget(self.sh_slider)
        sh_row.addWidget(self.sh_w_slider)
        self.layout.addLayout(sh_row)

        self.layout.addStretch()

    def _region_index(self) -> int:
        return self.region_btn_group.checkedId()

    def _connect_signals(self) -> None:
        self.region_btn_group.idToggled.connect(lambda _id, checked: self.sync_ui() if checked else None)

        self.cyan_slider.valueChanged.connect(self._on_cyan_changed)
        self.magenta_slider.valueChanged.connect(self._on_magenta_changed)
        self.yellow_slider.valueChanged.connect(self._on_yellow_changed)

        # Persistence signals for Undo/Redo
        self.cyan_slider.valueCommitted.connect(lambda v: self._on_cyan_changed(v, persist=True))
        self.magenta_slider.valueCommitted.connect(lambda v: self._on_magenta_changed(v, persist=True))
        self.yellow_slider.valueCommitted.connect(lambda v: self._on_yellow_changed(v, persist=True))

        self.density_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, density=v)
        )
        self.density_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, density=v)
        )

        self.grade_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, grade=v)
        )
        self.grade_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, grade=v)
        )

        self.pick_wb_btn.toggled.connect(self._on_pick_wb_toggled)
        self.camera_wb_btn.toggled.connect(self._on_camera_wb_toggled)

        self.toe_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, toe=v)
        )
        self.toe_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, toe=v)
        )

        self.toe_w_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, toe_width=v)
        )
        self.toe_w_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, toe_width=v)
        )

        self.sh_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, shoulder=v)
        )
        self.sh_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, shoulder=v)
        )

        self.sh_w_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, shoulder_width=v)
        )
        self.sh_w_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, shoulder_width=v)
        )

    def _on_cyan_changed(self, v: float, persist: bool = False) -> None:
        idx = self._region_index()
        if idx == 0:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, wb_cyan=v)
        elif idx == 1:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, shadow_cyan=v)
        elif idx == 2:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, highlight_cyan=v)

    def _on_magenta_changed(self, v: float, persist: bool = False) -> None:
        idx = self._region_index()
        if idx == 0:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, wb_magenta=v)
        elif idx == 1:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, shadow_magenta=v)
        elif idx == 2:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, highlight_magenta=v)

    def _on_yellow_changed(self, v: float, persist: bool = False) -> None:
        idx = self._region_index()
        if idx == 0:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, wb_yellow=v)
        elif idx == 1:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, shadow_yellow=v)
        elif idx == 2:
            self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, highlight_yellow=v)

    def _on_pick_wb_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.WB_PICK if checked else ToolMode.NONE)

    def _on_camera_wb_toggled(self, checked: bool) -> None:
        from dataclasses import replace

        new_config = replace(
            self.state.config,
            exposure=replace(self.state.config.exposure, use_camera_wb=checked),
            process=replace(self.state.config.process, local_floors=(0.0, 0.0, 0.0), local_ceils=(0.0, 0.0, 0.0)),
        )
        # render=False: don't analyse bounds on stale (pre-reload) raw data
        self.controller.session.update_config(new_config, persist=True, render=False)
        if self.state.current_file_path:
            self.controller.load_file(self.state.current_file_path)

    # Base hues per slider; varied by region for visual context
    _CMY_COLORS = {
        # region_index: (cyan_color, magenta_color, yellow_color)
        0: ("#00b1b1", "#b100b1", "#b1b100"),  # Global — full saturation
        1: ("#007a9c", "#7a009c", "#7a7a00"),  # Shadows — cooler/darker
        2: ("#00d4c8", "#d400a0", "#d4c800"),  # Highlights — brighter/warmer
    }

    def _update_cmy_label_colors(self, idx: int) -> None:
        c, m, y = self._CMY_COLORS.get(idx, self._CMY_COLORS[0])
        fs = THEME.font_size_base
        self.cyan_slider.label.setStyleSheet(f"font-size: {fs}px; color: {c};")
        self.magenta_slider.label.setStyleSheet(f"font-size: {fs}px; color: {m};")
        self.yellow_slider.label.setStyleSheet(f"font-size: {fs}px; color: {y};")

    def sync_ui(self) -> None:
        conf = self.state.config.exposure

        self.block_signals(True)
        try:
            idx = self._region_index()
            self._update_cmy_label_colors(idx)
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

            self.toe_slider.setValue(conf.toe)
            self.toe_w_slider.setValue(conf.toe_width)

            self.sh_slider.setValue(conf.shoulder)
            self.sh_w_slider.setValue(conf.shoulder_width)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all sliders and buttons.
        """
        widgets = [
            self.region_global_btn,
            self.region_shadow_btn,
            self.region_highlight_btn,
            self.cyan_slider,
            self.magenta_slider,
            self.yellow_slider,
            self.pick_wb_btn,
            self.camera_wb_btn,
            self.density_slider,
            self.grade_slider,
            self.toe_slider,
            self.toe_w_slider,
            self.sh_slider,
            self.sh_w_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
