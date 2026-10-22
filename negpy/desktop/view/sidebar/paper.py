from negpy.kernel.system.i18n import tr
import qtawesome as qta
from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.sidebar.tone import _CH_LABEL, _CH_SUFFIX, channel_selector
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.styles.templates import (
    ICON_BUTTON_WIDTH,
    PAPER_STRIP_SWEEP,
    field_row,
    gradient_icon,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.fold import group_fold, rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.features.exposure.logic import per_channel_dye_separation
from negpy.features.exposure.papers import DEFAULT_PROFILE_KEY, profiles_for_mode

# Each PaperProfile.medium's (icon, header) in the Print stock list.
_MEDIA = {
    "digital": ("mdi6.monitor", tr("Digital")),
    "paper": ("mdi6.file-image-outline", tr("Paper")),
    "film": ("mdi6.filmstrip", tr("Print film")),
}


class PaperSidebar(BaseSidebar):
    """The paper's characteristic curve: profile, Paper White/Black, Dye Separation, Snap,
    Toe and Shoulder, with its own [Global/R/G/B] selector scoping the curve controls to
    per-layer trims."""

    def _init_ui(self) -> None:
        conf = self.state.config.exposure

        self.ch_btn = channel_selector(
            tr("Global changes all colors, and R, G or B changes one color, for a cast that differs in shadows and highlights.")
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
            tr("Paper Black: show the black of real paper, a little lighter than pure black."),
        )
        self.paper_dmin_btn = self._small_toggle(
            "fa5s.file",
            "",
            conf.paper_dmin,
            tr("Paper White: show the white of real paper, a little darker than pure white."),
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

        self.paper_combo = StepCombo()
        self.paper_combo.setToolTip(tr("The paper the picture prints on, which sets the contrast curve and, on color paper, the colors."))
        self._populate_paper_combo(self.state.config.process.process_mode)
        idx = self.paper_combo.findData(conf.paper_profile)
        if idx >= 0:
            self.paper_combo.setCurrentIndex(idx)
        self.paper_strip_btn = self._tool_toggle("mdi.view-column-outline", "", self._paper_strip_tooltip())
        self.paper_strip_btn.setIcon(gradient_icon("mdi.view-column-outline", PAPER_STRIP_SWEEP))
        self.paper_strip_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.paper_strip_btn.setProperty("value_column", True)
        self.paper_strip_btn.clicked.connect(lambda checked: self.controller.toggle_paper_strip(force=checked))
        paper_row = field_row(tr("Print stock"), self.paper_combo, self.paper_strip_btn)
        self.paper_label = paper_row.itemAt(0).widget()
        self.layout.addLayout(paper_row)

        # Density-domain saturation on the dye amounts, before the stock's dyes, not a Lab a*/b* step.
        self.dye_separation_slider = CompactSlider(tr("Dye Separation"), 0.5, 1.5, conf.dye_separation, has_neutral=True)
        self.dye_separation_trim_slider = CompactSlider(tr("Dye Separation"), -0.4, 0.4, 0.0, has_neutral=True)
        self.dye_separation_trim_slider.setToolTip(tr("Dye Separation of this color layer only."))
        self.dye_separation_trim_slider.setVisible(False)
        # Redistributes the slider above by each pixel's own chroma, so it is inert at 1.0 separation.
        self.separation_damping_slider = CompactSlider(tr("Separation Damping"), 0.0, 1.0, conf.separation_damping)
        self.layout.addWidget(self.dye_separation_slider)
        self.layout.addWidget(self.dye_separation_trim_slider)
        self.separation_damping_fold = rail_fold("paper_separation", self.dye_separation_slider, self.separation_damping_slider)
        self.layout.addWidget(self.separation_damping_fold)

        self.midtone_gamma_slider = CompactSlider(tr("Snap"), -0.5, 0.5, conf.midtone_gamma)

        self.toe_w_slider = CompactSlider(tr("Toe Width"), 0.1, 5.0, conf.toe_width)
        self.toe_w_trim_slider = CompactSlider(tr("Toe Width"), -2.0, 2.0, 0.0)
        self.toe_w_trim_slider.setToolTip(tr("Toe Width of this color layer only."))
        self.toe_w_trim_slider.setVisible(False)
        self.toe_slider = CompactSlider(tr("Toe"), -1.0, 1.0, conf.toe)

        self.sh_slider = CompactSlider(tr("Shoulder"), -1.0, 1.0, conf.shoulder)
        self.sh_w_slider = CompactSlider(tr("Shoulder Width"), 0.1, 5.0, conf.shoulder_width)
        self.sh_w_trim_slider = CompactSlider(tr("Shoulder Width"), -2.0, 2.0, 0.0)
        self.sh_w_trim_slider.setToolTip(tr("Shoulder Width of this color layer only."))
        self.sh_w_trim_slider.setVisible(False)
        curve_header, curve_fold = group_fold(
            "paper_curve_shape",
            tr("Curve Shape"),
            self.midtone_gamma_slider,
            self.toe_slider,
            rail_fold("paper_toe", self.toe_slider, self.toe_w_slider, self.toe_w_trim_slider),
            self.sh_slider,
            rail_fold("paper_shoulder", self.sh_slider, self.sh_w_slider, self.sh_w_trim_slider),
            default_open=True,
        )
        self.layout.addLayout(curve_header)
        self.layout.addWidget(curve_fold)

        self.layout.addStretch()

        # Global-only controls, grayed while a channel page is active. The Paper Test Strip button
        # is one too, synced with the proof in _sync_paper_strip_btn.
        self._global_only = (self.paper_dmin_btn, self.paper_black_btn, self.paper_combo)

    def _channel_index(self) -> int:
        return self.ch_btn.currentIndex()

    @staticmethod
    def _paper_strip_tooltip(printing: bool = False) -> str:
        if printing:
            return tr("Printing the paper test strip…")
        return tooltip_with_shortcut(
            tr("Paper Test Strip: print the frame on every paper side by side, then keep the one you like."),
            "toggle_paper_strip",
        )

    def _sync_paper_strip_btn(self, _up: bool = False) -> None:
        # One shared slot, so the kind gates this or the other proofs light this button too.
        mine = self.state.test_strip_kind == "paper"
        pending = self.state.test_strip_pending and mine
        self.paper_strip_btn.setChecked((self.state.test_strip or self.state.test_strip_pending) and mine)
        self.paper_strip_btn.setEnabled(not pending and self._channel_index() == 0)
        self.paper_strip_btn.setToolTip(wrap_tooltip(self._paper_strip_tooltip(printing=pending)))

    def _curve_field(self, base: str) -> str:
        idx = self._channel_index()
        return base if idx == 0 else f"{base}_trim_{_CH_SUFFIX[idx - 1]}"

    def _populate_paper_combo(self, process_mode: str) -> None:
        """Fill the paper dropdown with the papers valid for the current process
        mode (the Digital default + the mode's kind), under a header per medium."""
        entries: list[tuple[str, str | None, str]] = []
        for key, prof in profiles_for_mode(process_mode):
            if not entries or entries[-1][2] != prof.medium:
                entries.append((_MEDIA[prof.medium][1], None, prof.medium))
            entries.append((prof.label, key, prof.medium))
        current = [(self.paper_combo.itemText(i), self.paper_combo.itemData(i)) for i in range(self.paper_combo.count())]
        if [(label, key) for label, key, _ in entries] == current:
            return
        self.paper_combo.clear()
        for label, key, medium in entries:
            if key is None:
                self.paper_combo.add_header(label)
            else:
                icon = qta.icon(_MEDIA[medium][0], color=THEME.text_primary, color_disabled=THEME.text_muted)
                self.paper_combo.addItem(icon, label, key)

    def _on_paper_changed(self, _idx: int) -> None:
        key = self.paper_combo.currentData()
        if key is None:  # separator row
            return
        self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, paper_profile=key)

    def _connect_signals(self) -> None:
        # Session state, not config, so the button follows the controller rather than sync_ui.
        self.controller.test_strip_changed.connect(self._sync_paper_strip_btn)
        self.paper_combo.currentIndexChanged.connect(self._on_paper_changed)
        self.ch_btn.currentChanged.connect(lambda _i: self.sync_ui())
        self.dye_separation_slider.valueChanged.connect(lambda _v: self._sync_damping_rail())

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
            self.paper_combo.setCurrentIndex(paper_idx if paper_idx >= 0 else self.paper_combo.findData(DEFAULT_PROFILE_KEY))
            self.paper_combo.setVisible(mode != ProcessMode.E6)
            self.paper_label.setVisible(mode != ProcessMode.E6)
            self.paper_strip_btn.setVisible(mode != ProcessMode.E6)

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
            self.toe_slider.label.setText(tr("Toe") + suffix)
            self.sh_slider.label.setText(tr("Shoulder") + suffix)
            self.midtone_gamma_slider.label.setText(tr("Snap") + suffix)
            if global_mode:
                self.toe_slider.setValue(conf.toe)
                self.sh_slider.setValue(conf.shoulder)
                self.midtone_gamma_slider.setValue(conf.midtone_gamma)
            else:
                ch = _CH_SUFFIX[idx - 1]
                self.toe_slider.setValue(getattr(conf, f"toe_trim_{ch}"))
                self.sh_slider.setValue(getattr(conf, f"shoulder_trim_{ch}"))
                self.midtone_gamma_slider.setValue(getattr(conf, f"midtone_gamma_trim_{ch}"))
                self.toe_w_trim_slider.label.setText(tr("Toe Width") + suffix)
                self.toe_w_trim_slider.setValue(getattr(conf, f"toe_width_trim_{ch}"))
                self.sh_w_trim_slider.label.setText(tr("Shoulder Width") + suffix)
                self.sh_w_trim_slider.setValue(getattr(conf, f"shoulder_width_trim_{ch}"))
                self.dye_separation_trim_slider.label.setText(tr("Dye Separation") + suffix)
                self.dye_separation_trim_slider.setValue(getattr(conf, f"dye_separation_trim_{ch}"))
            for w in self._global_only:
                w.setEnabled(global_mode)
            self._sync_paper_strip_btn()

            for i, fields in enumerate(self._channel_fields, start=1):
                self.ch_btn.set_edited(i, any(getattr(conf, f) != 0.0 for f in fields))
            self.toe_w_slider.setValue(conf.toe_width)
            self.sh_w_slider.setValue(conf.shoulder_width)
            self.dye_separation_slider.setValue(conf.dye_separation)
            self.separation_damping_slider.setValue(conf.separation_damping)
            self._sync_damping_rail()

            self.paper_dmin_btn.setChecked(conf.paper_dmin)
            self.paper_black_btn.setChecked(conf.paper_black)
        finally:
            self.block_signals(False)

    def _sync_damping_rail(self) -> None:
        # It redistributes Dye Separation's push and does nothing on its own, so it is dead at 1.0
        # separation on every channel. A per-channel trim also arms it. No trim form, so global view only.
        from negpy.features.process.models import ProcessMode

        conf = self.state.config.exposure
        sep_k3 = per_channel_dye_separation(
            self.dye_separation_slider.value(),
            (conf.dye_separation_trim_red, conf.dye_separation_trim_green, conf.dye_separation_trim_blue),
        )
        is_bw = self.state.config.process.process_mode == ProcessMode.BW
        self.separation_damping_fold.content.setVisible(self._channel_index() == 0 and not is_bw)
        self.separation_damping_slider.setEnabled(sep_k3 != (1.0, 1.0, 1.0))

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
