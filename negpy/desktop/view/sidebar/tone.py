from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider


class ToneSidebar(BaseSidebar):
    """Density, Grade, H&D curve (toe/shoulder), flare, paper white, contrast lift, paper profile."""

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.exposure

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
            "fa5s.smog",
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

        self.surround_btn = self._labeled_toggle(
            "fa5s.eye",
            " Contrast Lift",
            conf.surround,
            "Contrast Lift: a gentle fixed contrast expansion about paper white. Prints viewed in a "
            "normal (dim) surround read flatter than a 1:1 reproduction, so preferred tone "
            "reproduction (Bartleson-Breneman) calls for a slightly higher system gamma (~1.1) — "
            "this darkens midtones a touch and adds snap, uniformly on every frame.",
        )
        self.layout.addWidget(self.surround_btn)

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

    def _connect_signals(self) -> None:
        self.paper_combo.currentIndexChanged.connect(self._on_paper_changed)

        for slider, field in (
            (self.density_slider, "density"),
            (self.grade_slider, "grade"),
            (self.toe_slider, "toe"),
            (self.toe_w_slider, "toe_width"),
            (self.sh_slider, "shoulder"),
            (self.sh_w_slider, "shoulder_width"),
        ):
            slider.valueChanged.connect(
                lambda v, f=field: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, **{f: v})
            )
            slider.valueCommitted.connect(
                lambda v, f=field: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, **{f: v})
            )

        for btn, field in (
            (self.flare_btn, "flare"),
            (self.paper_dmin_btn, "paper_dmin"),
            (self.surround_btn, "surround"),
            (self.auto_density_btn, "auto_exposure"),
            (self.auto_grade_btn, "auto_normalize_contrast"),
        ):
            btn.toggled.connect(
                lambda checked, f=field: self.update_config_section(
                    "exposure", render=True, persist=True, readback_metrics=True, **{f: checked}
                )
            )

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

            self.density_slider.setValue(conf.density)
            self.grade_slider.setValue(conf.grade)
            self.toe_slider.setValue(conf.toe)
            self.toe_w_slider.setValue(conf.toe_width)
            self.sh_slider.setValue(conf.shoulder)
            self.sh_w_slider.setValue(conf.shoulder_width)

            self.paper_dmin_btn.setChecked(conf.paper_dmin)
            self.flare_btn.setChecked(conf.flare)
            self.surround_btn.setChecked(conf.surround)
            self.auto_density_btn.setChecked(conf.auto_exposure)
            self.auto_grade_btn.setChecked(conf.auto_normalize_contrast)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.paper_combo,
            self.density_slider,
            self.grade_slider,
            self.toe_slider,
            self.toe_w_slider,
            self.sh_slider,
            self.sh_w_slider,
            self.paper_dmin_btn,
            self.flare_btn,
            self.surround_btn,
            self.auto_density_btn,
            self.auto_grade_btn,
        ):
            w.blockSignals(blocked)
