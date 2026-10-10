from PyQt6.QtWidgets import QComboBox, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.sidebar.tone import _CH_LABEL, _CH_SUFFIX, channel_selector
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, field_row
from negpy.desktop.view.widgets.sliders import CompactSlider, SliderGroup
from negpy.features.exposure.logic import per_channel_dye_separation


class PaperSidebar(BaseSidebar):
    """The paper's characteristic curve: profile, Paper White/Black, Dye Separation, Snap,
    Toe and Shoulder, with its own [Global/R/G/B] selector scoping the curve controls to
    per-layer trims."""

    def _init_ui(self) -> None:
        conf = self.state.config.exposure

        self.ch_btn = channel_selector(
            "Global edits the paper's shared H&D curve (all layers). Red, Green and Blue edit per-layer "
            "Snap/Toe/Shoulder/Width/Dye Separation trims for the cyan-, magenta- and yellow-dye emulsions"
        )
        self._channel_fields = tuple(
            (
                f"toe_trim_{ch}",
                f"shoulder_trim_{ch}",
                f"midtone_gamma_trim_{ch}",
                f"toe_width_trim_{ch}",
                f"shoulder_width_trim_{ch}",
                f"dye_separation_trim_{ch}",
            )
            for ch in _CH_SUFFIX
        )
        self.paper_black_btn = self._small_toggle(
            "fa5s.circle",
            "",
            conf.paper_black,
            "Paper Black — show the paper's real Dmax as a slightly lifted, milky black instead of "
            "compensating it to pure display black. Off (default) applies black point compensation, "
            "like an ICC relative-colorimetric soft-proof, so the adapted eye reads paper black as "
            "black; on preserves the paper's true maximum density.",
        )
        self.paper_dmin_btn = self._small_toggle(
            "fa5s.file",
            "",
            conf.paper_dmin,
            "Paper White: simulate paper base density (Dmin 0.06) — whites print at ~0.93 instead of pure white, like a real print",
        )
        self.paper_black_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.paper_dmin_btn.setFixedWidth(ICON_BUTTON_WIDTH)

        ch_row = QHBoxLayout()
        ch_row.addWidget(self.ch_btn, 1)
        # Holds the icons at the right edge when B&W hides the selector.
        ch_row.addStretch()
        ch_row.addWidget(self.paper_black_btn)
        ch_row.addWidget(self.paper_dmin_btn)
        self.layout.addLayout(ch_row)

        self.paper_combo = QComboBox()
        self.paper_combo.setToolTip(
            "Darkroom paper profile — re-shapes the H&D curve (and color, on RA4) to a classic "
            "stock as a baseline; Grade / Density / toe / shoulder still trim on top."
        )
        self._populate_paper_combo(self.state.config.process.process_mode)
        idx = self.paper_combo.findData(conf.paper_profile)
        if idx >= 0:
            self.paper_combo.setCurrentIndex(idx)
        paper_row = field_row("Paper", self.paper_combo)
        self.paper_label = paper_row.itemAt(0).widget()
        self.layout.addLayout(paper_row)

        # Density-domain saturation, composed into the same dye_mix slot as the paper's real dye
        # crosstalk, rather than a post-hoc Lab-space a*/b*
        self.dye_separation_slider = CompactSlider("Dye Separation", 0.5, 1.5, conf.dye_separation, has_neutral=True)
        self.dye_separation_trim_slider = CompactSlider("Dye Separation", -0.4, 0.4, 0.0, has_neutral=True)
        self.dye_separation_trim_slider.setToolTip(
            "This layer's Dye Separation trim on top of the global value — pushes/pulls this "
            "channel's density separation independently. Neutrals stay flat at any trim value."
        )
        self.dye_separation_trim_slider.setVisible(False)
        # Redistributes the slider above by each pixel's own chroma. Inert at 1.0 separation, so
        # it is disabled there rather than reading as broken.
        self.separation_damping_slider = CompactSlider("Separation Damping", 0.0, 1.0, conf.separation_damping)
        self.layout.addWidget(self.dye_separation_slider)
        self.layout.addWidget(self.dye_separation_trim_slider)
        self.separation_damping_rail = SliderGroup(self.separation_damping_slider)
        self.layout.addWidget(self.separation_damping_rail)

        self.midtone_gamma_slider = CompactSlider("Snap", -0.5, 0.5, conf.midtone_gamma)
        self.layout.addWidget(self.midtone_gamma_slider)

        self.toe_w_slider = CompactSlider("Toe Width", 0.1, 5.0, conf.toe_width)
        self.toe_w_trim_slider = CompactSlider("Toe Width", -2.0, 2.0, 0.0)
        self.toe_w_trim_slider.setToolTip(
            "This layer's toe width trim on top of the global Toe Width — per-layer roll-off extent "
            "(sharpness crossover): how far this layer's shadow knee reaches up the tonal scale."
        )
        self.toe_w_trim_slider.setVisible(False)
        self.toe_slider = CompactSlider("Toe", -1.0, 1.0, conf.toe)
        self.layout.addWidget(self.toe_slider)
        self.layout.addWidget(SliderGroup(self.toe_w_slider, self.toe_w_trim_slider))

        self.sh_slider = CompactSlider("Shoulder", -1.0, 1.0, conf.shoulder)
        self.sh_w_slider = CompactSlider("Shoulder Width", 0.1, 5.0, conf.shoulder_width)
        self.sh_w_trim_slider = CompactSlider("Shoulder Width", -2.0, 2.0, 0.0)
        self.sh_w_trim_slider.setToolTip(
            "This layer's shoulder width trim on top of the global Width — per-layer roll-off extent "
            "(sharpness crossover): how far this layer's highlight knee reaches down the tonal scale."
        )
        self.sh_w_trim_slider.setVisible(False)
        self.layout.addWidget(self.sh_slider)
        self.layout.addWidget(SliderGroup(self.sh_w_slider, self.sh_w_trim_slider))

        self.layout.addStretch()

        # Global-only controls, greyed while a channel page is active.
        self._global_only = (self.paper_dmin_btn, self.paper_black_btn, self.paper_combo)

    def _channel_index(self) -> int:
        return self.ch_btn.currentIndex()

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
        self.ch_btn.currentChanged.connect(lambda _i: self.sync_ui())

        for slider, field in (
            (self.toe_w_slider, "toe_width"),
            (self.sh_w_slider, "shoulder_width"),
            (self.dye_separation_slider, "dye_separation"),
            (self.separation_damping_slider, "separation_damping"),
        ):
            slider.valueChanged.connect(
                lambda v, f=field: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, **{f: v})
            )
            slider.valueCommitted.connect(
                lambda v, f=field: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, **{f: v})
            )
            slider.dragStarted.connect(lambda f=field: self.controller.tone_drag_changed.emit(f))
            slider.dragEnded.connect(lambda: self.controller.tone_drag_changed.emit(""))

        # Toe/shoulder/snap retarget to the selected channel's trim field at emit time.
        for slider, base in (
            (self.toe_slider, "toe"),
            (self.sh_slider, "shoulder"),
            (self.midtone_gamma_slider, "midtone_gamma"),
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

        # Width/dye-separation trims live on separate sliders (differing trim vs global domain).
        for slider, base in (
            (self.toe_w_trim_slider, "toe_width"),
            (self.sh_w_trim_slider, "shoulder_width"),
            (self.dye_separation_trim_slider, "dye_separation"),
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

        for btn, field in ((self.paper_dmin_btn, "paper_dmin"), (self.paper_black_btn, "paper_black")):
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
            from negpy.features.process.path import RenderPath, render_path

            mode = self.state.config.process.process_mode
            self._populate_paper_combo(mode)
            paper_idx = self.paper_combo.findData(conf.paper_profile)
            self.paper_combo.setCurrentIndex(paper_idx if paper_idx >= 0 else 0)
            self.paper_combo.setVisible(mode != ProcessMode.E6)
            self.paper_label.setVisible(mode != ProcessMode.E6)

            # On the transfer path the render starts from the capture, so the paper model has
            # nothing to act on; Toe and Shoulder drive the transfer curve instead.
            transfer = render_path(self.state.config.process) is not RenderPath.PRINT
            for w in (self.paper_dmin_btn, self.paper_black_btn, self.midtone_gamma_slider):
                w.setVisible(not transfer)
            # Per-layer trims are meaningless on a single-emulsion B&W paper.
            is_bw = mode == ProcessMode.BW
            if is_bw:
                self.ch_btn.setCurrentIndex(0)
            self.ch_btn.setVisible(not is_bw)

            idx = self._channel_index()
            global_mode = idx == 0
            suffix = _CH_LABEL[idx]
            self.toe_w_slider.setVisible(global_mode)
            self.toe_w_trim_slider.setVisible(not global_mode)
            self.sh_w_slider.setVisible(global_mode)
            self.sh_w_trim_slider.setVisible(not global_mode)
            # Dye Separation swaps the same way on both paths (features/transparency/logic.py).
            # Separation Damping has no per-channel trim, so it stays global-view-only.
            self.dye_separation_slider.setVisible(global_mode and not is_bw)
            self.dye_separation_trim_slider.setVisible(not global_mode and not is_bw)
            self.separation_damping_slider.setVisible(global_mode and not is_bw)
            self.separation_damping_rail.setVisible(global_mode and not is_bw)
            self.toe_slider.label.setText("Toe" + suffix)
            self.sh_slider.label.setText("Shoulder" + suffix)
            self.midtone_gamma_slider.label.setText("Snap" + suffix)
            if global_mode:
                self.toe_slider.setValue(conf.toe)
                self.sh_slider.setValue(conf.shoulder)
                self.midtone_gamma_slider.setValue(conf.midtone_gamma)
            else:
                ch = _CH_SUFFIX[idx - 1]
                self.toe_slider.setValue(getattr(conf, f"toe_trim_{ch}"))
                self.sh_slider.setValue(getattr(conf, f"shoulder_trim_{ch}"))
                self.midtone_gamma_slider.setValue(getattr(conf, f"midtone_gamma_trim_{ch}"))
                self.toe_w_trim_slider.label.setText("Toe Width" + suffix)
                self.toe_w_trim_slider.setValue(getattr(conf, f"toe_width_trim_{ch}"))
                self.sh_w_trim_slider.label.setText("Shoulder Width" + suffix)
                self.sh_w_trim_slider.setValue(getattr(conf, f"shoulder_width_trim_{ch}"))
                self.dye_separation_trim_slider.label.setText("Dye Separation" + suffix)
                self.dye_separation_trim_slider.setValue(getattr(conf, f"dye_separation_trim_{ch}"))
            for w in self._global_only:
                w.setEnabled(global_mode)

            for i, fields in enumerate(self._channel_fields, start=1):
                self.ch_btn.set_edited(i, any(getattr(conf, f) != 0.0 for f in fields))
            self.toe_w_slider.setValue(conf.toe_width)
            self.sh_w_slider.setValue(conf.shoulder_width)
            self.dye_separation_slider.setValue(conf.dye_separation)
            self.separation_damping_slider.setValue(conf.separation_damping)
            # It redistributes Dye Separation's push and does nothing on its own, so at 1.0
            # separation on every channel it is dead — a per-channel trim also arms it.
            sep_k3 = per_channel_dye_separation(
                conf.dye_separation,
                (conf.dye_separation_trim_red, conf.dye_separation_trim_green, conf.dye_separation_trim_blue),
            )
            self.separation_damping_slider.setEnabled(sep_k3 != (1.0, 1.0, 1.0))

            self.paper_dmin_btn.setChecked(conf.paper_dmin)
            self.paper_black_btn.setChecked(conf.paper_black)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.paper_combo,
            self.ch_btn,
            self.toe_slider,
            self.toe_w_slider,
            self.toe_w_trim_slider,
            self.sh_slider,
            self.sh_w_slider,
            self.sh_w_trim_slider,
            self.midtone_gamma_slider,
            self.dye_separation_slider,
            self.dye_separation_trim_slider,
            self.separation_damping_slider,
            self.paper_dmin_btn,
            self.paper_black_btn,
        ):
            w.blockSignals(blocked)
