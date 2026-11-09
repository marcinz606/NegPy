from dataclasses import replace
from typing import Optional

from negpy.kernel.system.i18n import tr
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, hint_label, labeled_action, section_subheader, set_toggle_suggested
from negpy.desktop.view.widgets.fold import rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.process.fade import RATIO_BOUNDS, RED_SURVIVAL_BOUNDS, estimate_red
from negpy.features.process.fade_estimate import GrayMemo, auto_red_changes, run_estimate
from negpy.features.process.path import RenderPath, render_path

_REMOVE_STAIN_ICON = "fa5s.tint-slash"
_AUTO_RED_ICON = "mdi6.arrow-expand-vertical"


def _estimated(conf) -> bool:
    """Whether Estimate (or a hand setting) has set this frame's survivals."""
    return conf.fade_crossover_table is not None or (conf.fade_ratio_r, conf.fade_ratio_g, conf.fade_ratio_b) != (1.0, 1.0, 1.0)


class FadeSidebar(BaseSidebar):
    """Fade Restoration: gives a faded slide its dye layers back."""

    def _init_ui(self) -> None:
        conf = self.state.config.process
        # Green and blue are ratios to red, all a neutral can constrain; red's absolute survival has its own control.
        self.fade_ratio_r_slider = CompactSlider(
            tr("Red Survival"), *RED_SURVIVAL_BOUNDS, conf.fade_ratio_r, step=0.01, precision=1000, has_neutral=True
        )
        self.fade_ratio_r_slider.setToolTip(tr("How much red dye is left in the slide. Lower it if the faded slide still looks flat."))
        self.fade_ratio_g_slider = CompactSlider(
            tr("Green Survival"), *RATIO_BOUNDS, conf.fade_ratio_g, step=0.01, precision=1000, has_neutral=True
        )
        self.fade_ratio_g_slider.setToolTip(tr("How much green dye is left, compared with red. Below 1.0 means green faded more than red."))
        self.fade_ratio_b_slider = CompactSlider(
            tr("Blue Survival"), *RATIO_BOUNDS, conf.fade_ratio_b, step=0.01, precision=1000, has_neutral=True
        )
        self.fade_ratio_b_slider.setToolTip(tr("How much blue dye is left, compared with red. Below 1.0 means blue faded more than red."))

        # Tooltip set with its shortcut in controls_panel.apply_shortcut_tooltips.
        self.fade_white_anchor_btn = self._labeled_toggle(_REMOVE_STAIN_ICON, tr("Remove Stain"), conf.fade_white_anchor, "")

        self.fade_estimate_row = QHBoxLayout()
        self.estimate_fade_btn = labeled_action(
            "fa5s.magic",
            tr("Estimate"),
            tr("Sets the Survival sliders from the grays in this frame. If it finds no result, set them by hand."),
        )
        # Tooltip set with its shortcut in controls_panel.apply_shortcut_tooltips.
        self.fade_auto_red_btn = self._small_toggle(_AUTO_RED_ICON, "", conf.fade_auto_red, "")
        self.fade_auto_red_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.fade_estimate_row.addWidget(self.estimate_fade_btn, 1)
        self.fade_estimate_row.addWidget(self.fade_auto_red_btn)
        self.fade_estimate_row.addWidget(self.fade_white_anchor_btn)

        self.fade_roll_header = section_subheader(tr("ROLL"))
        self.fade_roll_row = QHBoxLayout()
        self.estimate_roll_btn = labeled_action(
            "fa5s.layer-group",
            tr("Estimate Roll"),
            tr("Reads every slide on the roll, or the selected ones, and gives every frame one set of fade values."),
        )
        self.share_crossover_btn = labeled_action(
            "fa5s.retweet",
            tr("Share Crossover"),
            tr("Gives the crossover of this frame to every frame on the roll: use a frame with clear grays in light and dark areas."),
        )
        self.fade_roll_row.addWidget(self.estimate_roll_btn, 1)
        self.fade_roll_row.addWidget(self.share_crossover_btn, 1)
        self.fade_roll_result_hint = hint_label("")
        self.fade_roll_result_hint.setVisible(False)
        # The last roll result, shown on every frame of that roll: (roll id, text, crossover suggestion, tooltip).
        self._roll_hint: tuple[Optional[str], str, str, str] = (None, "", "", "")

        # The estimate, or which fail-closed condition fired.
        self.fade_estimate_hint = hint_label("")
        self.fade_estimate_hint.setVisible(False)
        # Estimate's suggestion to turn on Remove Stain, which also paints the toggle's icon amber.
        self.fade_stain_hint = hint_label("", kind="warning")
        self.fade_stain_hint.setVisible(False)
        # The suggestion to turn on Restore Range on a frame with a short range, or why Restore Range found nothing to set.
        self.fade_red_hint = hint_label("", kind="warning")
        self.fade_red_hint.setVisible(False)
        self._fade_hint_hash: Optional[str] = None
        # (grid, key, gray readings) of the last Estimate, which a Gray Mix drag blends again.
        self._gray_memo = GrayMemo()
        # The frame where only the tone-band reader found grays: Gray Mix stays open there at every value.
        self._tone_bands_hash: Optional[str] = None

        self.fade_crossover_slider = CompactSlider(tr("Crossover"), 0.0, 100.0, conf.fade_crossover * 100.0, step=1.0, unit="%")
        self.fade_crossover_slider.setToolTip(
            tr("Removes the color shift that changes from the shadows to the highlights, which the Survival sliders cannot fix.")
        )
        # What the frame's crossover table found, or why Crossover is grayed out.
        self.fade_crossover_hint = hint_label("")
        self.fade_crossover_hint.setVisible(False)

        self.fade_neutrals_mix_slider = CompactSlider(tr("Gray Mix"), 0.0, 100.0, conf.fade_neutrals_mix * 100.0, step=1.0, unit="%")
        self.fade_neutrals_mix_slider.setToolTip(
            tr("Changes which grays Estimate trusts: drag it if the colors on this frame still look off.")
        )
        # Estimate sets the Survival sliders, so they fold under its row.
        self.fade_survival_fold = rail_fold(
            "fade_survival", self.fade_estimate_row, self.fade_ratio_r_slider, self.fade_ratio_g_slider, self.fade_ratio_b_slider
        )
        for item in (
            self.fade_estimate_row,
            self.fade_survival_fold,
            self.fade_estimate_hint,
            self.fade_stain_hint,
            self.fade_red_hint,
            self.fade_neutrals_mix_slider,
            self.fade_crossover_slider,
            self.fade_crossover_hint,
            self.fade_roll_header,
            self.fade_roll_row,
            self.fade_roll_result_hint,
        ):
            if isinstance(item, QHBoxLayout):
                self.layout.addLayout(item)
            else:
                self.layout.addWidget(item)

    def _connect_signals(self) -> None:
        self.fade_crossover_slider.valueChanged.connect(lambda v: self._on_fade_crossover_changed(v, persist=False))
        self.fade_crossover_slider.valueCommitted.connect(lambda v: self._on_fade_crossover_changed(v, persist=True))
        self.fade_neutrals_mix_slider.valueChanged.connect(lambda v: self._on_fade_neutrals_mix_changed(v, persist=False))
        self.fade_neutrals_mix_slider.valueCommitted.connect(lambda v: self._on_fade_neutrals_mix_changed(v, persist=True))
        self.fade_ratio_r_slider.valueChanged.connect(lambda v: self._on_fade_ratio_r_changed(v, persist=False))
        self.fade_ratio_r_slider.valueCommitted.connect(lambda v: self._on_fade_ratio_r_changed(v, persist=True))
        self.fade_ratio_g_slider.valueChanged.connect(
            lambda v: self._on_fade_ratio_changed(v, self.fade_ratio_b_slider.value(), persist=False)
        )
        self.fade_ratio_g_slider.valueCommitted.connect(
            lambda v: self._on_fade_ratio_changed(v, self.fade_ratio_b_slider.value(), persist=True)
        )
        self.fade_ratio_b_slider.valueChanged.connect(
            lambda v: self._on_fade_ratio_changed(self.fade_ratio_g_slider.value(), v, persist=False)
        )
        self.fade_ratio_b_slider.valueCommitted.connect(
            lambda v: self._on_fade_ratio_changed(self.fade_ratio_g_slider.value(), v, persist=True)
        )
        self.fade_white_anchor_btn.toggled.connect(self._on_fade_white_anchor_toggled)
        self.estimate_fade_btn.clicked.connect(self._on_estimate_fade)
        self.fade_auto_red_btn.toggled.connect(self._on_auto_red_toggled)
        self.estimate_roll_btn.clicked.connect(self.controller.estimate_fade_roll)
        self.share_crossover_btn.clicked.connect(self._on_share_crossover)
        self.controller.fade_roll_hint.connect(self._show_roll_hint)

    def _on_fade_ratio_r_changed(self, val: float, persist: bool = True) -> None:
        # A hand Red ends Restore Range, so the toggle never shows on over a value it did not set.
        # The shift belongs to Restore Range's fit, so it goes with it, as when the toggle turns off.
        changes = {"fade_ratio_r": val, "fade_auto_red": False, "fade_density_shift": 0.0}
        self.controller.set_roll_default("fade", persist=persist, readback_metrics=persist, **changes)

    def _on_fade_ratio_changed(self, ratio_g: float, ratio_b: float, persist: bool = True) -> None:
        self.controller.set_roll_default("fade", persist=persist, readback_metrics=persist, fade_ratio_g=ratio_g, fade_ratio_b=ratio_b)

    def _on_fade_neutrals_mix_changed(self, val: float, persist: bool = True) -> None:
        """Gray Mix runs Estimate again on the gray readings it has, and leaves the crossover table as it is."""
        if not self._neutrals_open(self.state.config.process):
            return
        self._run_estimate(self.state.config.process.fade_white_anchor, mix=val / 100.0, live=not persist, keep_table=True)

    def _on_auto_red_toggled(self, checked: bool) -> None:
        """On: Red Survival and the density shift from the frame's range. Off: Estimate's Red for the
        frame's green and blue, and no shift."""
        from negpy.desktop.workers.chart_fit import fade_estimate_grid

        conf = self.state.config.process
        if not checked:
            changes = {
                "fade_auto_red": False,
                "fade_ratio_r": estimate_red(conf.fade_ratio_g, conf.fade_ratio_b),
                "fade_density_shift": 0.0,
            }
            self._show_red_hint("")
            self.controller.set_roll_default("fade", **changes)
            return
        source = self.controller.chart_fit_source()
        grid = None if source is None else fade_estimate_grid(source)
        changes, reason = None, tr("No image loaded to estimate from.")
        if grid is not None:
            base = replace(conf, fade_ratio_r=estimate_red(conf.fade_ratio_g, conf.fade_ratio_b), fade_density_shift=0.0)
            changes, reason = auto_red_changes(base, source.cam_xyz, source.camera_wb, grid)
        self._show_red_hint(reason)
        if changes is None:
            with QSignalBlocker(self.fade_auto_red_btn):
                self.fade_auto_red_btn.setChecked(False)
            return
        set_toggle_suggested(self.fade_auto_red_btn, _AUTO_RED_ICON, False)
        changes = {**changes, "fade_auto_red": True}
        self.controller.set_roll_default("fade", **changes)

    def _show_red_hint(self, text: str) -> None:
        self._fade_hint_hash = self.state.current_file_hash
        self.fade_red_hint.setText(text)
        self.fade_red_hint.setVisible(bool(text))

    def _on_fade_crossover_changed(self, val: float, persist: bool = True) -> None:
        self.controller.set_roll_default("fade", persist=persist, readback_metrics=persist, fade_crossover=val / 100.0)

    def _sync_crossover(self, conf) -> None:
        """Crossover acts once the survivals do: a frame without a table measures one in the render as
        Crossover rises. The hint names what the table found, or that Crossover waits on an Estimate."""
        from negpy.features.process.fade import crossover_description, fade_matrix

        self.fade_crossover_slider.setValue(conf.fade_crossover * 100.0)
        table = conf.fade_crossover_table
        acts = fade_matrix(conf) is not None
        self.fade_crossover_slider.setEnabled(table is not None or acts)
        self.share_crossover_btn.setEnabled(acts and self.state.active_roll_id is not None)
        if table is not None:
            text = self._table_source(table) + crossover_description(table)
        else:
            text = "" if acts else tr("Click Estimate to measure the crossover.")
        self.fade_crossover_hint.setText(text)
        self.fade_neutrals_mix_slider.setValue(conf.fade_neutrals_mix * 100.0)
        self.fade_neutrals_mix_slider.setEnabled(self._neutrals_open(conf))
        self.fade_auto_red_btn.setChecked(conf.fade_auto_red)

    def _on_share_crossover(self) -> None:
        """Measures this frame's own table, whatever table it holds now, at the roll's survivals (its own
        where the roll has none): every frame that takes the table renders it after those."""
        from negpy.desktop.workers.chart_fit import fade_estimate_grid
        from negpy.features.process.fade_estimate import measured_crossover

        conf = self.state.config.process
        source = self.controller.chart_fit_source()
        grid = None if source is None else fade_estimate_grid(source)
        measured = (
            {}
            if grid is None
            else measured_crossover(
                conf,
                replace(conf, **self.controller.roll_survivals()),
                source.cam_xyz,
                source.camera_wb,
                grid,
                self._gray_memo,
                conf.fade_neutrals_mix,
            )
        )
        if not measured:
            self._show_fade_hint(tr("No clear grays on this frame to measure the crossover."))
            return
        self.controller.share_crossover(measured["fade_crossover_table"], measured["fade_crossover_space"])

    def _table_source(self, table) -> str:
        """Which table the frame uses, once its roll holds one, as a lead-in to the hint."""
        from negpy.services.assets.rolls import same_value

        roll_table = self.controller.roll_crossover_table()
        if roll_table is None:
            return ""
        if not same_value(table, roll_table):
            return tr("Table from this frame.") + " "
        source = self.controller.roll_crossover_source()
        return (tr("Table from the roll, measured on {name}.", name=source) if source else tr("Table from the roll.")) + " "

    def _show_roll_hint(self, text: str, suggestion: str, details: str) -> None:
        self._roll_hint = (self.state.active_roll_id, text, suggestion, details)
        self._sync_roll_hint()

    def _sync_roll_hint(self) -> None:
        """The crossover suggestion goes once the roll holds a shared crossover."""
        roll_id, text, suggestion, details = self._roll_hint
        here = roll_id is not None and roll_id == self.state.active_roll_id
        if suggestion and not self.controller.roll_crossover_source():
            text = f"{text} {suggestion}"
        shown = text if here else ""
        self.fade_roll_result_hint.setText(shown)
        self.fade_roll_result_hint.setToolTip(details if here else "")
        self.fade_roll_result_hint.setVisible(bool(shown))

    def _show_fade_hint(self, text: str, stain: str = "") -> None:
        """The hints belong to the frame they were written on; sync_ui clears them on another frame.
        `stain` is the suggestion to turn on Remove Stain, "" for none."""
        self._fade_hint_hash = self.state.current_file_hash
        self.fade_estimate_hint.setText(text)
        self.fade_estimate_hint.setVisible(bool(text))
        self._show_stain_suggestion(stain)

    def _show_stain_suggestion(self, text: str) -> None:
        self.fade_stain_hint.setText(text)
        self.fade_stain_hint.setVisible(bool(text))
        set_toggle_suggested(self.fade_white_anchor_btn, _REMOVE_STAIN_ICON, bool(text))

    def _on_fade_white_anchor_toggled(self, checked: bool) -> None:
        estimated = _estimated(self.state.config.process)
        self.controller.set_roll_default("fade", fade_white_anchor=checked)
        # The survivals were read with the other setting, so Estimate reads them again: on, above the
        # frame's own white; off, on an estimated frame, without it.
        if checked or estimated:
            self._run_estimate(white_anchor=checked)
        else:
            self._show_fade_hint("")

    def _neutrals_open(self, conf) -> bool:
        """Gray Mix runs Estimate again, so it waits for one on this frame."""
        return _estimated(conf) or (self._tone_bands_hash is not None and self._tone_bands_hash == self.state.current_file_hash)

    def _on_estimate_fade(self) -> None:
        self._run_estimate(self.state.config.process.fade_white_anchor)

    def _run_estimate(self, white_anchor: bool, mix: Optional[float] = None, live: bool = False, keep_table: bool = False) -> None:
        """Estimate with *mix* (the stored Gray Mix when None); see `fade_estimate.run_estimate`."""
        from negpy.desktop.workers.chart_fit import fade_estimate_grid

        conf = self.state.config.process
        persist = not live
        if render_path(conf) is RenderPath.PRINT:
            return
        source = self.controller.chart_fit_source()
        grid = None if source is None else fade_estimate_grid(source)
        if grid is None:
            self._show_fade_hint(tr("No image loaded to estimate from."))
            if mix is not None:
                self.controller.set_roll_default("fade", persist=persist, readback_metrics=False, fade_neutrals_mix=mix)
            return
        out = run_estimate(
            conf,
            grid,
            source.cam_xyz,
            source.camera_wb,
            self._gray_memo,
            white_anchor=white_anchor,
            mix=mix,
            live=live,
            keep_table=keep_table,
        )
        if out.tone_bands:
            self._tone_bands_hash = self.state.current_file_hash
        if out.mix is not None:
            with QSignalBlocker(self.fade_neutrals_mix_slider):
                self.fade_neutrals_mix_slider.setValue(out.mix * 100.0)
        if out.survivals is not None:
            for slider, value in zip((self.fade_ratio_r_slider, self.fade_ratio_g_slider, self.fade_ratio_b_slider), out.survivals):
                with QSignalBlocker(slider):
                    slider.setValue(value)
        self._show_fade_hint(out.hint, out.stain)
        if out.red_hint is not None:
            self._show_red_hint(out.red_hint)
            set_toggle_suggested(self.fade_auto_red_btn, _AUTO_RED_ICON, out.suggest_red)
        if out.fade_white:
            self.controller.set_fade_white(**out.fade_white)
        if out.changes:
            self.controller.set_roll_default("fade", persist=persist, readback_metrics=persist, **out.changes)

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            self.fade_ratio_r_slider.setValue(conf.fade_ratio_r)
            self.fade_ratio_g_slider.setValue(conf.fade_ratio_g)
            self.fade_ratio_b_slider.setValue(conf.fade_ratio_b)
            self.fade_white_anchor_btn.setChecked(conf.fade_white_anchor)
            self._sync_crossover(conf)
            self._sync_roll_hint()
            if self._fade_hint_hash != self.state.current_file_hash:
                self.fade_estimate_hint.setText("")
                self.fade_red_hint.setText("")
                set_toggle_suggested(self.fade_auto_red_btn, _AUTO_RED_ICON, False)
                self._show_stain_suggestion("")
            self.fade_estimate_hint.setVisible(bool(self.fade_estimate_hint.text()))
            self.fade_red_hint.setVisible(bool(self.fade_red_hint.text()))
            self.fade_crossover_hint.setVisible(bool(self.fade_crossover_hint.text()))
            self.fade_stain_hint.setVisible(bool(self.fade_stain_hint.text()))
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.fade_ratio_r_slider,
            self.fade_ratio_g_slider,
            self.fade_ratio_b_slider,
            self.fade_white_anchor_btn,
            self.fade_auto_red_btn,
            self.fade_crossover_slider,
            self.fade_neutrals_mix_slider,
        ):
            w.blockSignals(blocked)
