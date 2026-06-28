import qtawesome as qta
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from negpy.desktop.session import ToolMode
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.process.models import invalidate_local_bounds


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
        self.region_global_btn.setToolTip("Apply CMY white balance to the entire tonal range")
        self.region_shadow_btn.setToolTip("Apply CMY white balance biased toward shadow (low-density) areas")
        self.region_highlight_btn.setToolTip("Apply CMY white balance biased toward highlight (high-density) areas")
        self.region_btn_group = QButtonGroup(self)
        self.region_btn_group.setExclusive(True)
        self.region_btn_group.addButton(self.region_global_btn, 0)
        self.region_btn_group.addButton(self.region_shadow_btn, 1)
        self.region_btn_group.addButton(self.region_highlight_btn, 2)
        self.layout.addLayout(region_row)

        self.cyan_slider = CompactSlider("Cyan", -1.0, 1.0, conf.wb_cyan, has_neutral=True)
        self.cyan_slider.slider.setObjectName("cyan_slider")
        self.cyan_slider.setToolTip("Cyan–Red white balance shift; applies to the selected region (Global/Shadows/Highlights)")
        self.magenta_slider = CompactSlider("Magenta", -1.0, 1.0, conf.wb_magenta, has_neutral=True)
        self.magenta_slider.slider.setObjectName("magenta_slider")
        self.magenta_slider.setToolTip(
            tooltip_with_shortcut("Magenta–Green white balance shift; applies to the selected region  E/D", None)
        )
        self.yellow_slider = CompactSlider("Yellow", -1.0, 1.0, conf.wb_yellow, has_neutral=True)
        self.yellow_slider.slider.setObjectName("yellow_slider")
        self.yellow_slider.setToolTip(tooltip_with_shortcut("Yellow–Blue white balance shift; applies to the selected region  R/F", None))
        self.layout.addWidget(self.cyan_slider)
        self.layout.addWidget(self.magenta_slider)
        self.layout.addWidget(self.yellow_slider)

        # Tone-control toggles, paired two per row (equal width).
        self.pick_wb_btn = QPushButton(" Pick WB")
        self.pick_wb_btn.setCheckable(True)
        self.pick_wb_btn.setIcon(qta.icon("fa5s.eye-dropper", color=THEME.text_primary))
        self.pick_wb_btn.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 8px;")
        self.pick_wb_btn.setToolTip(tooltip_with_shortcut("Pick white balance from canvas", "pick_wb"))
        self.linear_raw_btn = QPushButton(" Linear RAW")
        self.linear_raw_btn.setCheckable(True)
        self.linear_raw_btn.setChecked(conf.linear_raw)
        self.linear_raw_btn.setIcon(qta.icon("fa5s.sliders-h", color=THEME.text_primary))
        self.linear_raw_btn.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 8px;")
        self.linear_raw_btn.setToolTip(
            "Decode RAW with neutral multipliers (1,1,1,1) — bypasses as-shot camera white balance for a clean starting point"
        )
        self.surround_btn = self._labeled_toggle(
            "fa5s.eye",
            " Contrast Lift",
            conf.surround,
            "Contrast Lift: a gentle fixed contrast expansion about paper white. Prints viewed in a "
            "normal (dim) surround read flatter than a 1:1 reproduction, so preferred tone "
            "reproduction (Bartleson-Breneman) calls for a slightly higher system gamma (~1.1) — "
            "this darkens midtones a touch and adds snap, uniformly on every frame.",
        )
        self.cast_removal_btn = self._labeled_toggle(
            "fa5s.palette",
            " Cast Removal",
            conf.cast_removal,
            "Cast Removal: automatically neutralizes the color cast a negative leaves in the print — "
            "balances each color layer so grays stay neutral from the deep shadows through the "
            "highlights (C-41).",
        )

        tone_row1 = QHBoxLayout()
        tone_row1.addWidget(self.pick_wb_btn, 1)
        tone_row1.addWidget(self.linear_raw_btn, 1)
        self.layout.addLayout(tone_row1)
        tone_row2 = QHBoxLayout()
        tone_row2.addWidget(self.cast_removal_btn, 1)
        tone_row2.addWidget(self.surround_btn, 1)
        self.layout.addLayout(tone_row2)

        self.density_slider = CompactSlider("Density", 0.0, 2.0, conf.density)
        self.density_slider.setToolTip(tooltip_with_shortcut("Overall exposure — higher values darken the print", "density_up"))
        self.grade_slider = CompactSlider("ISO-R Grade", 50.0, 180.0, conf.grade, step=1.0, inverted=True)
        self.grade_slider.setToolTip(
            tooltip_with_shortcut(
                "Contrast (ISO R paper exposure range): R180 = very soft, R50 = very hard; R110 ≈ grade 2 paper",
                "grade_up",
            )
        )

        self.auto_density_btn = self._icon_toggle(
            "fa5s.magic",
            conf.auto_exposure,
            "Auto Density: meter each frame's midtone and anchor the print exposure there, so dense "
            "and flat negatives land at a consistent brightness instead of needing per-frame trimming",
        )
        density_row = QHBoxLayout()
        density_row.addWidget(self.auto_density_btn)
        density_row.addWidget(self.density_slider)
        self.layout.addLayout(density_row)

        self.auto_grade_btn = self._icon_toggle(
            "fa5s.balance-scale",
            conf.auto_normalize_contrast,
            "Auto Grade: normalize contrast across the roll — render every negative through the same "
            "curve so dense negatives stop printing over-contrasty and flat ones stop printing muddy",
        )
        grade_row = QHBoxLayout()
        grade_row.addWidget(self.auto_grade_btn)
        grade_row.addWidget(self.grade_slider)
        self.layout.addLayout(grade_row)

        self.flare_btn = self._icon_toggle(
            "fa5s.sun",
            conf.flare,
            "Flare: veiling-glare floor that lifts the deepest print blacks and softens the toe "
            "(film look) while leaving paper white fixed",
        )
        toe_row = QHBoxLayout()
        self.toe_w_slider = CompactSlider("Width", 0.1, 5.0, conf.toe_width)
        self.toe_w_slider.setToolTip("Width of the shadow toe transition zone")
        self.toe_slider = CompactSlider("Toe", -1.0, 1.0, conf.toe)
        self.toe_slider.setToolTip("Shadow toe lift: positive raises shadows, negative deepens blacks")
        toe_row.addWidget(self.flare_btn)
        toe_row.addWidget(self.toe_slider)
        toe_row.addWidget(self.toe_w_slider)
        self.layout.addLayout(toe_row)

        self.paper_dmin_btn = self._icon_toggle(
            "fa5s.file",
            conf.paper_dmin,
            "Paper White: simulate paper base density (Dmin 0.06) — whites print at ~0.93 instead of pure white, like a real print",
        )
        sh_row = QHBoxLayout()
        self.sh_slider = CompactSlider("Shoulder", -1.0, 1.0, conf.shoulder)
        self.sh_slider.setToolTip("Highlight shoulder roll: positive compresses highlights, negative extends them")
        self.sh_w_slider = CompactSlider("Width", 0.1, 5.0, conf.shoulder_width)
        self.sh_w_slider.setToolTip("Width of the highlight shoulder transition zone")
        sh_row.addWidget(self.paper_dmin_btn)
        sh_row.addWidget(self.sh_slider)
        sh_row.addWidget(self.sh_w_slider)
        self.layout.addLayout(sh_row)

        paper_row = QHBoxLayout()
        self.paper_label = QLabel("Paper Profile")
        self.paper_label.setStyleSheet(f"font-size: {THEME.font_size_base}px;")
        self.paper_combo = QComboBox()
        self.paper_combo.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 4px;")
        self.paper_combo.setToolTip(
            "Darkroom paper profile — re-shapes the H&D curve (and colour, on RA4) to a classic "
            "stock as a baseline; Grade / Density / toe / shoulder still trim on top."
        )
        self._populate_paper_combo(self.state.config.process.process_mode)
        idx = self.paper_combo.findData(conf.paper_profile)
        if idx >= 0:
            self.paper_combo.setCurrentIndex(idx)
        paper_row.addWidget(self.paper_label)
        paper_row.addWidget(self.paper_combo, 1)
        self.layout.addLayout(paper_row)

        self.layout.addStretch()

    def _populate_paper_combo(self, process_mode: str) -> None:
        """Fill the paper dropdown with the papers valid for the current process
        mode (neutral default + the mode's kind)."""
        from negpy.features.exposure.papers import profiles_for_mode

        self.paper_combo.clear()
        for key, prof in profiles_for_mode(process_mode):
            self.paper_combo.addItem(prof.label, key)

    def _on_paper_changed(self, _idx: int) -> None:
        key = self.paper_combo.currentData()
        if key is None:  # separator row
            return
        self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, paper_profile=key)

    def _icon_toggle(self, icon_name: str, checked: bool, tooltip: str) -> QPushButton:
        """Compact icon-only checkable button placed beside a slider."""
        btn = QPushButton()
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary))
        btn.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 6px;")
        btn.setFixedWidth(36)
        btn.setToolTip(tooltip)
        return btn

    def _labeled_toggle(self, icon_name: str, label: str, checked: bool, tooltip: str) -> QPushButton:
        """Labeled checkable button (icon + text), styled like Pick WB / Linear RAW."""
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary))
        btn.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 8px;")
        btn.setToolTip(tooltip)
        return btn

    def _region_index(self) -> int:
        return self.region_btn_group.checkedId()

    def _connect_signals(self) -> None:
        self.paper_combo.currentIndexChanged.connect(self._on_paper_changed)
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
        self.linear_raw_btn.toggled.connect(self._on_linear_raw_toggled)
        self.paper_dmin_btn.toggled.connect(
            lambda checked: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, paper_dmin=checked)
        )
        self.cast_removal_btn.toggled.connect(
            lambda checked: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, cast_removal=checked)
        )
        self.flare_btn.toggled.connect(
            lambda checked: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, flare=checked)
        )
        self.surround_btn.toggled.connect(
            lambda checked: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, surround=checked)
        )
        self.auto_density_btn.toggled.connect(
            lambda checked: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, auto_exposure=checked)
        )
        self.auto_grade_btn.toggled.connect(
            lambda checked: self.update_config_section(
                "exposure", render=True, persist=True, readback_metrics=True, auto_normalize_contrast=checked
            )
        )

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

    def _on_linear_raw_toggled(self, checked: bool) -> None:
        from dataclasses import replace

        new_config = replace(
            self.state.config,
            exposure=replace(self.state.config.exposure, linear_raw=checked),
            process=replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process)),
        )
        # render=False: don't analyse bounds on stale (pre-reload) raw data
        self.controller.session.update_config(new_config, persist=True, render=False)
        if self.state.current_file_path:
            self.controller.load_file(self.state.current_file_path)

    def sync_ui(self) -> None:
        conf = self.state.config.exposure

        self.block_signals(True)
        try:
            from negpy.features.process.models import ProcessMode

            mode = self.state.config.process.process_mode
            self._populate_paper_combo(mode)
            paper_idx = self.paper_combo.findData(conf.paper_profile)
            self.paper_combo.setCurrentIndex(paper_idx if paper_idx >= 0 else 0)
            hide_paper = mode == ProcessMode.E6
            self.paper_combo.setVisible(not hide_paper)
            self.paper_label.setVisible(not hide_paper)

            idx = self._region_index()
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
            self.linear_raw_btn.setChecked(conf.linear_raw)

            self.density_slider.setValue(conf.density)
            self.grade_slider.setValue(conf.grade)

            self.toe_slider.setValue(conf.toe)
            self.toe_w_slider.setValue(conf.toe_width)

            self.sh_slider.setValue(conf.shoulder)
            self.sh_w_slider.setValue(conf.shoulder_width)

            self.paper_dmin_btn.setChecked(conf.paper_dmin)
            self.flare_btn.setChecked(conf.flare)
            self.cast_removal_btn.setChecked(conf.cast_removal)
            self.surround_btn.setChecked(conf.surround)
            self.auto_density_btn.setChecked(conf.auto_exposure)
            self.auto_grade_btn.setChecked(conf.auto_normalize_contrast)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all sliders and buttons.
        """
        widgets = [
            self.paper_combo,
            self.region_global_btn,
            self.region_shadow_btn,
            self.region_highlight_btn,
            self.cyan_slider,
            self.magenta_slider,
            self.yellow_slider,
            self.pick_wb_btn,
            self.linear_raw_btn,
            self.density_slider,
            self.grade_slider,
            self.toe_slider,
            self.toe_w_slider,
            self.sh_slider,
            self.sh_w_slider,
            self.paper_dmin_btn,
            self.flare_btn,
            self.cast_removal_btn,
            self.surround_btn,
            self.auto_density_btn,
            self.auto_grade_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
