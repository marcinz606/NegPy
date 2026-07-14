import qtawesome as qta
from PyQt6.QtWidgets import QButtonGroup, QComboBox, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.widgets.sliders import CompactSlider

_CH_SUFFIX = ("red", "green", "blue")
_CH_LABEL = ("", " R", " G", " B")
_CH_COLORS = ("#ff5a5a", "#5adc78", "#5f96ff")


class ToneSidebar(BaseSidebar):
    """Print/zone density, Grade, paper white, and a labeled Paper Response
    group (paper profile + Snap/Toe/Shoulder) — with a [Global/R/G/B] channel
    selector scoping Grade/Toe/Shoulder to per-layer trims (crossover correction)."""

    def _init_ui(self) -> None:
        conf = self.state.config.exposure

        self.density_slider = CompactSlider("Print Density", 0.0, 2.0, conf.density)
        self.grade_slider = CompactSlider("ISO-R Grade", 50.0, 180.0, conf.grade, step=1.0, inverted=True)
        self.grade_trim_slider = CompactSlider("Grade", -30.0, 30.0, 0.0, step=1.0, inverted=True)
        self.grade_trim_slider.setToolTip(
            "Crossover correction — this layer's contrast trim in ISO-R points on top of the Grade: "
            "filtration can only shift a dye layer's curve, this rotates its slope, fixing casts that "
            "differ between shadows and highlights. Midtone neutrality is preserved."
        )
        self.grade_trim_slider.setVisible(False)

        # Channel selector: Global = the shared curve; R/G/B = per-layer trims.
        self.ch_global_btn = self._labeled_toggle("fa5s.globe", " Global", True, "Global — edit the shared H&D curve (all layers)")
        self.ch_r_btn = self._labeled_toggle(
            "fa5s.circle", " Red", False, "Red layer — per-layer Grade/Toe/Shoulder/Width/Snap trims for the cyan-dye emulsion"
        )
        self.ch_g_btn = self._labeled_toggle(
            "fa5s.circle", " Green", False, "Green layer — per-layer Grade/Toe/Shoulder/Width/Snap trims for the magenta-dye emulsion"
        )
        self.ch_b_btn = self._labeled_toggle(
            "fa5s.circle", " Blue", False, "Blue layer — per-layer Grade/Toe/Shoulder/Width/Snap trims for the yellow-dye emulsion"
        )
        for btn, color in zip((self.ch_r_btn, self.ch_g_btn, self.ch_b_btn), _CH_COLORS):
            btn.setIcon(qta.icon("fa5s.circle", color=color))
        self.ch_btn_group = QButtonGroup(self)
        self.ch_btn_group.setExclusive(True)
        for i, btn in enumerate((self.ch_global_btn, self.ch_r_btn, self.ch_g_btn, self.ch_b_btn)):
            self.ch_btn_group.addButton(btn, i)
        # (button, that channel's trim fields) for the edited dot.
        self._channel_buttons = tuple(
            (
                btn,
                (
                    f"grade_trim_{ch}",
                    f"shadow_grade_trim_{ch}",
                    f"highlight_grade_trim_{ch}",
                    f"toe_trim_{ch}",
                    f"shoulder_trim_{ch}",
                    f"midtone_gamma_trim_{ch}",
                    f"toe_width_trim_{ch}",
                    f"shoulder_width_trim_{ch}",
                ),
            )
            for btn, ch in zip((self.ch_r_btn, self.ch_g_btn, self.ch_b_btn), _CH_SUFFIX)
        )
        ch_row = QHBoxLayout()
        for btn in (self.ch_global_btn, self.ch_r_btn, self.ch_g_btn, self.ch_b_btn):
            ch_row.addWidget(btn, 1)
        self.layout.addLayout(ch_row)

        self.auto_density_btn = self._small_toggle(
            "fa5s.magic",
            "Auto Density",
            conf.auto_exposure,
            "Auto Density: meter each frame's midtone and anchor the print exposure there, so dense "
            "and flat negatives land at a consistent brightness instead of needing per-frame trimming",
        )
        self.auto_grade_btn = self._small_toggle(
            "fa5s.balance-scale",
            "Auto Grade",
            conf.auto_normalize_contrast,
            "Auto Grade: normalize contrast across the roll — render every negative through the same "
            "curve so dense negatives stop printing over-contrasty and flat ones stop printing muddy",
        )
        auto_row = QHBoxLayout()
        auto_row.addWidget(self.auto_density_btn, 1)
        auto_row.addWidget(self.auto_grade_btn, 1)
        self.layout.addLayout(auto_row)
        self.layout.addWidget(self.density_slider)

        self.true_black_btn = self._small_toggle(
            "fa5s.circle",
            "True Black",
            conf.true_black,
            "True Black — black point compensation: maps the paper's Dmax to display black, like an "
            "ICC relative-colorimetric soft-proof; the adapted eye reads paper black as black. "
            "A lifted toe and shadow colour survive; pull Toe negative to clip deep shadows to exact black.",
        )
        self.paper_dmin_btn = self._small_toggle(
            "fa5s.file",
            "Paper White",
            conf.paper_dmin,
            "Paper White: simulate paper base density (Dmin 0.06) — whites print at ~0.93 instead of pure white, like a real print",
        )
        self.shadow_density_slider = CompactSlider("Shadows Density", -0.9, 0.9, conf.shadow_density)
        self.highlight_density_slider = CompactSlider("Highlights Density", -0.5, 0.5, conf.highlight_density)
        zone_density_row = QHBoxLayout()
        zone_density_row.addWidget(self.shadow_density_slider)
        zone_density_row.addWidget(self.highlight_density_slider)
        self.layout.addLayout(zone_density_row)

        grade_row = QHBoxLayout()
        grade_row.addWidget(self.grade_slider)
        grade_row.addWidget(self.grade_trim_slider)
        self.layout.addLayout(grade_row)

        self.shadow_grade_slider = CompactSlider("Shadows Grade", -50.0, 50.0, conf.shadow_grade, step=1.0, inverted=True)
        self.highlight_grade_slider = CompactSlider("Highlights Grade", -50.0, 50.0, conf.highlight_grade, step=1.0, inverted=True)
        split_grade_row = QHBoxLayout()
        split_grade_row.addWidget(self.shadow_grade_slider)
        split_grade_row.addWidget(self.highlight_grade_slider)
        self.layout.addLayout(split_grade_row)

        paper_header = section_subheader("PAPER RESPONSE")
        paper_header.setToolTip(
            "The paper's characteristic (Hurter–Driffield) curve: how print density responds to "
            "exposure. Snap bends the midtone gamma, Toe shapes the shadow roll-off into paper "
            "black, Shoulder the highlight roll-off into paper white — each knee with its own "
            "Width, per dye layer via the Global/R/G/B selector."
        )
        self.layout.addWidget(paper_header)

        self.paper_combo = QComboBox()
        self.paper_combo.setToolTip(
            "Darkroom paper profile — re-shapes the H&D curve (and colour, on RA4) to a classic "
            "stock as a baseline; Grade / Density / toe / shoulder still trim on top."
        )
        self._populate_paper_combo(self.state.config.process.process_mode)
        idx = self.paper_combo.findData(conf.paper_profile)
        if idx >= 0:
            self.paper_combo.setCurrentIndex(idx)
        self.layout.addWidget(self.paper_combo)

        paper_toggle_row = QHBoxLayout()
        paper_toggle_row.addWidget(self.true_black_btn, 1)
        paper_toggle_row.addWidget(self.paper_dmin_btn, 1)
        self.layout.addLayout(paper_toggle_row)

        self.midtone_gamma_slider = CompactSlider("Snap", -0.5, 0.5, conf.midtone_gamma)
        snap_row = QHBoxLayout()
        snap_row.addWidget(self.midtone_gamma_slider)
        self.layout.addLayout(snap_row)

        toe_row = QHBoxLayout()
        self.toe_w_slider = CompactSlider("Toe Width", 0.1, 5.0, conf.toe_width)
        self.toe_w_trim_slider = CompactSlider("Toe Width", -2.0, 2.0, 0.0)
        self.toe_w_trim_slider.setToolTip(
            "This layer's toe width trim on top of the global Toe Width — per-layer roll-off extent "
            "(sharpness crossover): how far this layer's shadow knee reaches up the tonal scale."
        )
        self.toe_w_trim_slider.setVisible(False)
        self.toe_slider = CompactSlider("Toe", -1.0, 1.0, conf.toe)
        toe_row.addWidget(self.toe_slider)
        toe_row.addWidget(self.toe_w_slider)
        toe_row.addWidget(self.toe_w_trim_slider)
        self.layout.addLayout(toe_row)

        sh_row = QHBoxLayout()
        self.sh_slider = CompactSlider("Shoulder", -1.0, 1.0, conf.shoulder)
        self.sh_w_slider = CompactSlider("Shoulder Width", 0.1, 5.0, conf.shoulder_width)
        self.sh_w_trim_slider = CompactSlider("Shoulder Width", -2.0, 2.0, 0.0)
        self.sh_w_trim_slider.setToolTip(
            "This layer's shoulder width trim on top of the global Width — per-layer roll-off extent "
            "(sharpness crossover): how far this layer's highlight knee reaches down the tonal scale."
        )
        self.sh_w_trim_slider.setVisible(False)
        sh_row.addWidget(self.sh_slider)
        sh_row.addWidget(self.sh_w_slider)
        sh_row.addWidget(self.sh_w_trim_slider)
        self.layout.addLayout(sh_row)

        self.layout.addStretch()

        # Global-only controls, greyed while a channel page is active.
        self._global_only = (
            self.density_slider,
            self.auto_density_btn,
            self.auto_grade_btn,
            self.paper_dmin_btn,
            self.true_black_btn,
            self.paper_combo,
            self.shadow_density_slider,
            self.highlight_density_slider,
        )

    def _channel_index(self) -> int:
        return max(self.ch_btn_group.checkedId(), 0)

    def _curve_field(self, base: str) -> str:
        idx = self._channel_index()
        return base if idx == 0 else f"{base}_trim_{_CH_SUFFIX[idx - 1]}"

    def _populate_paper_combo(self, process_mode: str) -> None:
        """Fill the paper dropdown with the papers valid for the current process
        mode (neutral default + the mode's kind)."""
        from negpy.features.exposure.papers import profiles_for_mode

        entries = [(prof.label, key) for key, prof in profiles_for_mode(process_mode)]
        current = [(self.paper_combo.itemText(i), self.paper_combo.itemData(i)) for i in range(self.paper_combo.count())]
        if entries == current:
            return
        self.paper_combo.clear()
        for label, key in entries:
            self.paper_combo.addItem(label, key)

    def _on_paper_changed(self, _idx: int) -> None:
        key = self.paper_combo.currentData()
        if key is None:  # separator row
            return
        self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, paper_profile=key)

    def _connect_signals(self) -> None:
        self.paper_combo.currentIndexChanged.connect(self._on_paper_changed)
        self.ch_btn_group.idToggled.connect(lambda _id, checked: self.sync_ui() if checked else None)

        for slider, field in (
            (self.density_slider, "density"),
            (self.grade_slider, "grade"),
            (self.toe_w_slider, "toe_width"),
            (self.sh_w_slider, "shoulder_width"),
            (self.shadow_density_slider, "shadow_density"),
            (self.highlight_density_slider, "highlight_density"),
        ):
            slider.valueChanged.connect(
                lambda v, f=field: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, **{f: v})
            )
            slider.valueCommitted.connect(
                lambda v, f=field: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, **{f: v})
            )
            slider.dragStarted.connect(lambda f=field: self.controller.tone_drag_changed.emit(f))
            slider.dragEnded.connect(lambda: self.controller.tone_drag_changed.emit(""))

        # Toe/shoulder/snap/split-grade retarget to the selected channel's trim field at emit time.
        for slider, base in (
            (self.toe_slider, "toe"),
            (self.sh_slider, "shoulder"),
            (self.midtone_gamma_slider, "midtone_gamma"),
            (self.shadow_grade_slider, "shadow_grade"),
            (self.highlight_grade_slider, "highlight_grade"),
        ):
            slider.valueChanged.connect(
                lambda v, b=base: self.update_config_section(
                    "exposure", render=True, persist=False, readback_metrics=False, **{self._curve_field(b): v}
                )
            )
            slider.valueCommitted.connect(
                lambda v, b=base: self.update_config_section(
                    "exposure", render=True, persist=True, readback_metrics=True, **{self._curve_field(b): v}
                )
            )
            slider.dragStarted.connect(lambda b=base: self.controller.tone_drag_changed.emit(b))
            slider.dragEnded.connect(lambda: self.controller.tone_drag_changed.emit(""))

        grade_trim_field = lambda: f"grade_trim_{_CH_SUFFIX[self._channel_index() - 1]}"  # noqa: E731
        self.grade_trim_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, **{grade_trim_field(): v})
        )
        self.grade_trim_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, **{grade_trim_field(): v})
        )

        # Width trims live on separate sliders (trim domain ±2 vs global 0.1–5).
        for slider, base in ((self.toe_w_trim_slider, "toe_width"), (self.sh_w_trim_slider, "shoulder_width")):
            slider.valueChanged.connect(
                lambda v, b=base: self.update_config_section(
                    "exposure", render=True, persist=False, readback_metrics=False, **{self._curve_field(b): v}
                )
            )
            slider.valueCommitted.connect(
                lambda v, b=base: self.update_config_section(
                    "exposure", render=True, persist=True, readback_metrics=True, **{self._curve_field(b): v}
                )
            )

        for btn, field in (
            (self.paper_dmin_btn, "paper_dmin"),
            (self.true_black_btn, "true_black"),
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
            self.paper_combo.setVisible(mode != ProcessMode.E6)

            # Per-layer trims are meaningless on a single-emulsion B&W paper.
            is_bw = mode == ProcessMode.BW
            if is_bw and self._channel_index() != 0:
                self.ch_global_btn.setChecked(True)
            for w in (self.ch_global_btn, self.ch_r_btn, self.ch_g_btn, self.ch_b_btn):
                w.setVisible(not is_bw)

            idx = self._channel_index()
            global_mode = idx == 0
            suffix = _CH_LABEL[idx]
            self.grade_slider.setVisible(global_mode)
            self.grade_trim_slider.setVisible(not global_mode)
            self.toe_w_slider.setVisible(global_mode)
            self.toe_w_trim_slider.setVisible(not global_mode)
            self.sh_w_slider.setVisible(global_mode)
            self.sh_w_trim_slider.setVisible(not global_mode)
            self.toe_slider.label.setText("Toe" + suffix)
            self.sh_slider.label.setText("Shoulder" + suffix)
            self.midtone_gamma_slider.label.setText("Snap" + suffix)
            self.shadow_grade_slider.label.setText("Shadows Grade" + suffix)
            self.highlight_grade_slider.label.setText("Highlights Grade" + suffix)
            if global_mode:
                self.toe_slider.setValue(conf.toe)
                self.sh_slider.setValue(conf.shoulder)
                self.midtone_gamma_slider.setValue(conf.midtone_gamma)
                self.shadow_grade_slider.setValue(conf.shadow_grade)
                self.highlight_grade_slider.setValue(conf.highlight_grade)
            else:
                ch = _CH_SUFFIX[idx - 1]
                self.grade_trim_slider.label.setText("Grade" + suffix)
                self.grade_trim_slider.setValue(getattr(conf, f"grade_trim_{ch}"))
                self.toe_slider.setValue(getattr(conf, f"toe_trim_{ch}"))
                self.sh_slider.setValue(getattr(conf, f"shoulder_trim_{ch}"))
                self.midtone_gamma_slider.setValue(getattr(conf, f"midtone_gamma_trim_{ch}"))
                self.shadow_grade_slider.setValue(getattr(conf, f"shadow_grade_trim_{ch}"))
                self.highlight_grade_slider.setValue(getattr(conf, f"highlight_grade_trim_{ch}"))
                self.toe_w_trim_slider.label.setText("Toe Width" + suffix)
                self.toe_w_trim_slider.setValue(getattr(conf, f"toe_width_trim_{ch}"))
                self.sh_w_trim_slider.label.setText("Shoulder Width" + suffix)
                self.sh_w_trim_slider.setValue(getattr(conf, f"shoulder_width_trim_{ch}"))
            for w in self._global_only:
                w.setEnabled(global_mode)

            for btn, fields in self._channel_buttons:
                btn.edited_dot.set_active(any(getattr(conf, f) != 0.0 for f in fields))

            self.density_slider.setValue(conf.density)
            self.grade_slider.setValue(conf.grade)
            self.toe_w_slider.setValue(conf.toe_width)
            self.sh_w_slider.setValue(conf.shoulder_width)
            self.shadow_density_slider.setValue(conf.shadow_density)
            self.highlight_density_slider.setValue(conf.highlight_density)

            self.paper_dmin_btn.setChecked(conf.paper_dmin)
            self.true_black_btn.setChecked(conf.true_black)
            self.auto_density_btn.setChecked(conf.auto_exposure)
            self.auto_grade_btn.setChecked(conf.auto_normalize_contrast)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.paper_combo,
            self.ch_global_btn,
            self.ch_r_btn,
            self.ch_g_btn,
            self.ch_b_btn,
            self.density_slider,
            self.grade_slider,
            self.grade_trim_slider,
            self.toe_slider,
            self.toe_w_slider,
            self.toe_w_trim_slider,
            self.sh_slider,
            self.sh_w_slider,
            self.sh_w_trim_slider,
            self.midtone_gamma_slider,
            self.shadow_density_slider,
            self.highlight_density_slider,
            self.shadow_grade_slider,
            self.highlight_grade_slider,
            self.paper_dmin_btn,
            self.true_black_btn,
            self.auto_density_btn,
            self.auto_grade_btn,
        ):
            w.blockSignals(blocked)
