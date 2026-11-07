from dataclasses import replace
from typing import Optional

from negpy.kernel.system.i18n import tr
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, hint_label, labeled_action, set_toggle_suggested
from negpy.desktop.view.widgets.fold import rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.process.fade import RATIO_BOUNDS, RED_SURVIVAL_BOUNDS, FadeEstimate, estimate_red, stain_suggested
from negpy.features.process.path import RenderPath, render_path

_REMOVE_STAIN_ICON = "fa5s.tint-slash"
_AUTO_RED_ICON = "mdi6.arrow-expand-vertical"
#: Estimate suggests Restore Range only where its Red would be this share of Estimate's or less: the frame's
#: span is well short of AUTO_RED_SPAN, so a frame near the target stays quiet.
AUTO_RED_HINT_RATIO = 0.85


def _estimate_hint(chosen: "FadeEstimate") -> str:
    """The result, or why there is none."""
    if chosen.reason and chosen.green == 1.0 and chosen.blue == 1.0:
        return chosen.reason
    red = chosen.red if chosen.red is not None else 1.0
    text = tr("Estimated: red {r}, green {g}, blue {b}.", r=f"{red:.2f}", g=f"{chosen.green:.2f}", b=f"{chosen.blue:.2f}")
    if chosen.reason:
        text = chosen.reason + " " + text
    return text


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
        self._gray_memo: Optional[tuple] = None
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
            changes, reason = self._auto_red_changes(base, source, grid)
        self._show_red_hint(reason)
        if changes is None:
            with QSignalBlocker(self.fade_auto_red_btn):
                self.fade_auto_red_btn.setChecked(False)
            return
        set_toggle_suggested(self.fade_auto_red_btn, _AUTO_RED_ICON, False)
        changes = {**changes, "fade_auto_red": True}
        self.controller.set_roll_default("fade", **changes)

    @staticmethod
    def _auto_red_changes(conf, source, grid) -> tuple[Optional[dict], str]:
        """`fade_auto_red.auto_red` for the survivals and shift in *conf*."""
        from negpy.features.process.fade import render_white_space
        from negpy.features.process.fade_auto_red import auto_red, survival_points

        space = render_white_space(conf, conf.process_mode, source.cam_xyz, source.camera_wb)
        g, b, shift = conf.fade_ratio_g, conf.fade_ratio_b, conf.fade_density_shift

        def points_at(red: float):
            return survival_points(grid, conf, (red, g, b), conf.fade_white_anchor, bool(space), space, shift)

        return auto_red(points_at, conf.fade_ratio_r, shift)

    def _show_red_hint(self, text: str) -> None:
        self._fade_hint_hash = self.state.current_file_hash
        self.fade_red_hint.setText(text)
        self.fade_red_hint.setVisible(bool(text))

    def _short_range(self, conf, source, grid) -> bool:
        """Whether Restore Range would set Red Survival at AUTO_RED_HINT_RATIO of *conf*'s or less."""
        changes, _ = self._auto_red_changes(conf, source, grid)
        return changes is not None and changes["fade_ratio_r"] <= conf.fade_ratio_r * AUTO_RED_HINT_RATIO

    def _on_fade_crossover_changed(self, val: float, persist: bool = True) -> None:
        self.controller.set_roll_default("fade", persist=persist, readback_metrics=persist, fade_crossover=val / 100.0)

    def _sync_crossover(self, conf) -> None:
        """Crossover acts on a frame with a table; the hint names what the table found, or why the
        slider is grayed out."""
        from negpy.features.process.fade import crossover_description

        self.fade_crossover_slider.setValue(conf.fade_crossover * 100.0)
        table = conf.fade_crossover_table
        self.fade_crossover_slider.setEnabled(table is not None)
        self.fade_crossover_hint.setText(
            tr("Click Estimate to measure the crossover.") if table is None else self._table_source(table) + crossover_description(table)
        )
        self.fade_neutrals_mix_slider.setValue(conf.fade_neutrals_mix * 100.0)
        self.fade_neutrals_mix_slider.setEnabled(self._neutrals_open(conf))
        self.fade_auto_red_btn.setChecked(conf.fade_auto_red)

    def _table_source(self, table) -> str:
        """Which table the frame uses, once its roll holds one, as a lead-in to the hint."""
        from negpy.services.assets.rolls import same_value

        roll_table = self.controller.roll_crossover_table()
        if roll_table is None:
            return ""
        return (tr("Table from the roll.") if same_value(table, roll_table) else tr("Table from this frame.")) + " "

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

    def _tone_bands_only(self, grid) -> bool:
        """Whether the last gray reading on *grid* found grays only in the tone bands."""
        from negpy.features.process.fade import tone_bands_only

        memo = self._gray_memo
        return memo is not None and memo[0] is grid and tone_bands_only(memo[2])

    def _gray_reader(self, grid, anchor: bool, space: str, mix: float):
        """Estimate's gray reader on *grid* at Gray Mix *mix*. The readings depend on the grid and the
        anchor, never on the mix, so they are kept for the next call."""
        from negpy.features.process.fade import blend_gray_readings, gray_readings

        memo_key = (anchor, space)

        def reader(gray_grid):
            memo = self._gray_memo
            if memo is None or memo[0] is not grid or memo[1] != memo_key:
                memo = self._gray_memo = (grid, memo_key, gray_readings(gray_grid))
            return blend_gray_readings(memo[2], mix)

        return reader

    def _measured_crossover(self, new_conf, mix: Optional[float] = None) -> dict:
        """The crossover table and its tag measured after *new_conf*'s survivals, {} with none."""
        from negpy.desktop.workers.chart_fit import fade_estimate_grid
        from negpy.features.process.fade import crossover_space, fade_density, render_white_space, survival_crossover

        conf = self.state.config.process
        mix = conf.fade_neutrals_mix if mix is None else mix
        if render_path(conf) is RenderPath.PRINT:
            return {}
        source = self.controller.chart_fit_source()
        grid = None if source is None else fade_estimate_grid(source)
        if grid is None:
            return {}
        space = render_white_space(new_conf, conf.process_mode, source.cam_xyz, source.camera_wb)
        anchor = conf.fade_white_anchor
        reader = self._gray_reader(grid, anchor, space, mix)
        rgb = (new_conf.fade_ratio_r, new_conf.fade_ratio_g, new_conf.fade_ratio_b)
        table = survival_crossover(
            grid, reader, rgb, white_anchor=anchor, camera_raw=bool(space), white_space=space, shift=new_conf.fade_density_shift
        )
        if table is None:
            return {}
        tag = crossover_space(fade_density(new_conf, conf.process_mode, space), space, new_conf)
        return {"fade_crossover_table": table, "fade_crossover_space": tag}

    def _on_estimate_fade(self) -> None:
        self._run_estimate(self.state.config.process.fade_white_anchor)

    def _run_estimate(self, white_anchor: bool, mix: Optional[float] = None, live: bool = False, keep_table: bool = False) -> None:
        """Estimate with *mix* (the stored Gray Mix when None). *live* is a Gray Mix drag: no crossover table,
        no undo step. *keep_table* (Gray Mix) writes no crossover table: only a press of Estimate measures one."""
        from negpy.desktop.workers.chart_fit import fade_estimate_grid
        from negpy.features.process.fade import (
            CROSSOVER_FLOOR,
            crossover_amount,
            crossover_space,
            estimate_fade_with,
            fade_density,
            pick_neutrals,
            render_white_space,
            toward_no_change,
        )

        conf = self.state.config.process
        mix_changes = {} if mix is None else {"fade_neutrals_mix": mix}
        mix = conf.fade_neutrals_mix if mix is None else mix
        persist = not live
        if render_path(conf) is RenderPath.PRINT:
            return
        source = self.controller.chart_fit_source()
        grid = None if source is None else fade_estimate_grid(source)
        if grid is None:
            self._show_fade_hint(tr("No image loaded to estimate from."))
            if mix_changes:
                self.controller.set_roll_default("fade", persist=persist, readback_metrics=False, **mix_changes)
            return
        anchor = white_anchor
        space = render_white_space(conf, conf.process_mode, source.cam_xyz, source.camera_wb)
        reader = self._gray_reader(grid, anchor, space, mix)
        # A press of Estimate picks Gray Mix and measures the tables then; a drag measures none.
        pick = not live and not mix_changes
        chosen = estimate_fade_with(
            grid, reader, white_anchor=anchor, camera_raw=bool(space), white_space=space, crossover=not live and not pick and not keep_table
        )
        # Where NegPy's gray finder finds no grays, Estimate reads the tone bands' at full Gray Mix, and
        # Gray Mix moves from no change to that estimate.
        tone_bands = self._tone_bands_only(grid)
        picked = None
        if pick and chosen.red is not None:
            memo = self._gray_memo
            found = None if tone_bands or memo is None else pick_neutrals(grid, memo[2], anchor, bool(space), space)
            if found is None:
                chosen = estimate_fade_with(grid, reader, white_anchor=anchor, camera_raw=bool(space), white_space=space)
            else:
                picked, chosen = found
                mix, mix_changes = picked, {"fade_neutrals_mix": picked}
                with QSignalBlocker(self.fade_neutrals_mix_slider):
                    self.fade_neutrals_mix_slider.setValue(picked * 100.0)
        if tone_bands:
            self._tone_bands_hash = self.state.current_file_hash
            if not mix_changes:
                mix, mix_changes = 1.0, {"fade_neutrals_mix": 1.0}
                with QSignalBlocker(self.fade_neutrals_mix_slider):
                    self.fade_neutrals_mix_slider.setValue(100.0)
            chosen = toward_no_change(chosen, mix)
        failed = bool(chosen.reason) and chosen.green == 1.0 and chosen.blue == 1.0
        # The white is stored with the survivals; a render measures one only for a stale camera RAW
        # white. A failed reading stores it only where it is plainly a stain on clear film. A camera
        # RAW's render scales density above its white, so it stores the white with Remove Stain off too.
        white = {}
        if (anchor or space) and chosen.white is not None and (not failed or space or stain_suggested(chosen.white)):
            white = dict(zip(("fade_white_r", "fade_white_g", "fade_white_b"), chosen.white), fade_white_space=space)
        stain = ""
        if not anchor and chosen.white is not None and stain_suggested(chosen.white):
            stain = tr("The clear areas have a yellow stain: turn on Remove Stain.")
        hint = _estimate_hint(chosen)
        if picked is not None:
            hint = tr("Gray Mix set to {n} %.", n=f"{picked * 100:.0f}") + " " + hint
        if tone_bands:
            hint = tr("This frame has few clear grays. Drag Gray Mix to set how much Estimate changes.") + " " + hint
        if anchor and not white and (conf.fade_white_r, conf.fade_white_g, conf.fade_white_b) == (0.0, 0.0, 0.0):
            hint += " " + tr("No stain found to remove.")
        self._show_fade_hint(hint, stain)
        if failed:
            if space and white and persist:
                self.controller.set_fade_white(**white)
            rest = {**({} if space else white), **mix_changes}
            if rest:
                self.controller.set_roll_default("fade", persist=persist, readback_metrics=persist, **rest)
            return
        red = chosen.red if chosen.red is not None else conf.fade_ratio_r
        for slider, value in (
            (self.fade_ratio_r_slider, red),
            (self.fade_ratio_g_slider, chosen.green),
            (self.fade_ratio_b_slider, chosen.blue),
        ):
            with QSignalBlocker(slider):
                slider.setValue(value)
        # Estimate's table is read with no shift, and a new red makes Restore Range's shift stale.
        survivals = dict(
            fade_ratio_r=red, fade_ratio_g=chosen.green, fade_ratio_b=chosen.blue, fade_density_shift=0.0, **white, **mix_changes
        )
        # Restore Range keeps its Red: it runs again on the new grays, and the table is measured after it.
        short = not live and not conf.fade_auto_red and self._short_range(replace(conf, **survivals), source, grid)
        auto = None
        if conf.fade_auto_red and live:
            # A Gray Mix drag keeps the fit, and the release fits Red again.
            survivals.update(fade_ratio_r=conf.fade_ratio_r, fade_density_shift=conf.fade_density_shift)
            with QSignalBlocker(self.fade_ratio_r_slider):
                self.fade_ratio_r_slider.setValue(conf.fade_ratio_r)
        elif conf.fade_auto_red:
            auto, _ = self._auto_red_changes(replace(conf, **survivals), source, grid)
        if auto is not None:
            survivals.update(auto)
            with QSignalBlocker(self.fade_ratio_r_slider):
                self.fade_ratio_r_slider.setValue(auto["fade_ratio_r"])
        # A press measures the table after these survivals; a drag keeps the last one until the release.
        crossover, measured = {}, {}
        if not keep_table and (auto is not None or (tone_bands and chosen.crossover is None)) and not live:
            measured = self._measured_crossover(replace(conf, **survivals), mix)
            survivals = {**survivals, **measured}
        if not keep_table and not measured and chosen.crossover is not None:
            new_conf = replace(conf, **survivals)
            tag = crossover_space(fade_density(new_conf, conf.process_mode, space), space, new_conf)
            crossover = dict(fade_crossover_table=chosen.crossover, fade_crossover_space=tag)
        if not live:
            table = crossover.get("fade_crossover_table", survivals.get("fade_crossover_table"))
            if conf.fade_crossover == 0.0 and isinstance(table, tuple) and crossover_amount(table) >= CROSSOVER_FLOOR:
                self._show_fade_hint(hint + " " + tr("The colors shift from shadows to highlights: raise Crossover."), stain)
            suggest_red = short and not conf.fade_auto_red
            self._show_red_hint(tr("This frame has a short range: turn on Restore Range.") if suggest_red else "")
            set_toggle_suggested(self.fade_auto_red_btn, _AUTO_RED_ICON, suggest_red)
        self.controller.set_roll_default("fade", persist=persist, readback_metrics=persist, **survivals, **crossover)

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            self.fade_ratio_r_slider.setValue(conf.fade_ratio_r)
            self.fade_ratio_g_slider.setValue(conf.fade_ratio_g)
            self.fade_ratio_b_slider.setValue(conf.fade_ratio_b)
            self.fade_white_anchor_btn.setChecked(conf.fade_white_anchor)
            self._sync_crossover(conf)
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
