from negpy.kernel.system.i18n import tr
from dataclasses import replace

from PyQt6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout

from negpy.desktop.auto_sliders import shown_values, stored_value
from negpy.desktop.view.confirm import warn_targets_affect_unbaked
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, TEST_STRIP_SWEEP, gradient_icon, hint_label, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice, ToggleMenuButton
from negpy.desktop.view.widgets.fold import group_fold, rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.hdr.models import hdr_active
from negpy.features.exposure.auto_sliders import NEUTRAL
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, TUNABLE_TARGETS, apply_targets

_ISO_R_MIN = float(EXPOSURE_CONSTANTS["iso_r_min"])
_ISO_R_MAX = float(EXPOSURE_CONSTANTS["iso_r_max"])

_CH_SUFFIX = ("red", "green", "blue")
_CH_LABEL = ("", " R", " G", " B")
_CH_COLORS = (THEME.channel_red_text, THEME.channel_green_text, THEME.channel_blue_text)


def channel_selector(tooltip: str) -> SegmentedChoice:
    """[Global/R/G/B]: Global edits the shared curve, a channel its per-layer trims."""
    return SegmentedChoice(
        (("fa5s.globe", tr("Global")), *(("fa5s.circle", n, c) for n, c in zip(("Red", "Green", "Blue"), _CH_COLORS))),
        tooltip,
        icon_only=(1, 2, 3),
    )


class ToneSidebar(BaseSidebar):
    """Print/zone density, Grade, split Grade, Preflash, Diffusion and Contrast Mask, with a
    [Global/R/G/B] channel selector scoping Grade and split Grade to per-layer trims
    (crossover correction)."""

    def _init_ui(self) -> None:
        conf = self.state.config.exposure

        self.density_slider = CompactSlider(tr("Print Density"), 0.0, 2.0, conf.density)
        # Travel matches the kernel's grade clamp, so the slider never offers a grade it would clip.
        self.grade_slider = CompactSlider(tr("ISO-R Grade"), _ISO_R_MIN, _ISO_R_MAX, conf.grade, step=1.0, inverted=True, unit=" R")
        self.grade_trim_slider = CompactSlider(tr("Grade"), -30.0, 30.0, 0.0, step=1.0, inverted=True, unit=" R")
        self.grade_trim_slider.setToolTip(
            tr("Contrast of this color layer only, for a color cast that differs between shadows and highlights.")
        )
        self.grade_trim_slider.setVisible(False)

        # Channel selector: Global = the shared curve; R/G/B = per-layer trims.
        self.ch_btn = channel_selector(
            tr("Global changes all colors, and R, G or B changes one color, for a cast that differs in shadows and highlights.")
        )
        # Each channel's trim fields, for the edited dot.
        self._channel_fields = tuple(
            (
                f"grade_trim_{ch}",
                f"shadow_grade_trim_{ch}",
                f"highlight_grade_trim_{ch}",
            )
            for ch in _CH_SUFFIX
        )
        self.auto_btn = ToggleMenuButton("fa5s.magic", "", tr("Auto: let NegPy set the brightness and contrast of each frame."))
        self.auto_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.auto_density_action = self.auto_btn.add_toggle(
            tr("Auto Density"),
            tr("Sets the brightness of each frame for you, so the frames of a roll print alike."),
        )
        self.auto_grade_action = self.auto_btn.add_toggle(
            tr("Auto Grade"),
            tr("Sets the contrast of each frame for you, so contrasty and flat negatives both print well."),
        )
        # Not an add_toggle option: the button's checked look follows the two real ones.
        self.auto_both_action = self.auto_btn.choice_menu.addAction(tr("Auto Density and Grade"))
        self.auto_both_action.setCheckable(True)
        self.auto_both_action.plain_tooltip = tr("Turn Auto Density and Auto Grade on or off together")
        self.auto_btn.choice_menu.addSeparator()
        # Not an add_toggle option either: it meters nothing, so the button's look ignores it.
        self.hold_black_action = self.auto_btn.choice_menu.addAction(tr("Hold Black"))
        self.hold_black_action.setCheckable(True)
        self.hold_black_action.setChecked(conf.hold_black)
        self.hold_black_action.plain_tooltip = tr("Hold Black: when you lighten the shadows, the deepest blacks stay black.")
        self.auto_btn.choice_menu.addSeparator()
        targets_action = self.auto_btn.choice_menu.addAction(tr("Set Targets…"))
        targets_action.setToolTip(tr("Set the brightness and contrast that Auto aims for, for every frame."))
        targets_action.triggered.connect(self._open_targets_dialog)
        self.auto_bake_action = self.auto_btn.choice_menu.addAction(tr("Bake Auto Density/Grade"))
        self.auto_bake_action.setToolTip(tr("Keep this frame's automatic values as fixed settings and turn Auto off for it."))
        self.auto_bake_action.triggered.connect(self._bake_auto_density_grade)
        self.auto_density_action.setChecked(conf.auto_exposure)
        self.auto_grade_action.setChecked(conf.auto_normalize_contrast)
        self.test_strip_btn = self._tool_toggle("mdi.view-grid-outline", "", self._test_strip_tooltip())
        self.test_strip_btn.setIcon(gradient_icon("mdi.view-grid-outline", TEST_STRIP_SWEEP))
        self.test_strip_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.test_strip_btn.clicked.connect(lambda checked: self.controller.toggle_test_strip(force=checked))

        ch_row = QHBoxLayout()
        ch_row.addWidget(self.ch_btn, 1)
        # Holds the icons at the right edge when B&W hides the selector.
        ch_row.addStretch()
        ch_row.addWidget(self.auto_btn)
        ch_row.addWidget(self.test_strip_btn)
        self.layout.addLayout(ch_row)
        # Disabled widgets get no hover, so the reason hangs off the hint under them.
        self.auto_merged_hint = hint_label(tr("Not applied to a merged bracket."))
        self.auto_merged_hint.setToolTip(
            wrap_tooltip(tr("A merged bracket sets its brightness with Render Exposure, so unmerge the frame to use Auto."))
        )
        self.auto_merged_hint.setVisible(False)
        self.layout.addWidget(self.auto_merged_hint)
        self.layout.addWidget(self.density_slider)

        self.shadow_density_slider = CompactSlider(tr("Shadows Density"), -1.0, 1.0, conf.shadow_density)
        self.highlight_density_slider = CompactSlider(tr("Highlights Density"), -1.0, 1.0, conf.highlight_density)
        self.layout.addWidget(rail_fold("tone_density", self.density_slider, self.shadow_density_slider, self.highlight_density_slider))

        grade_row = QVBoxLayout()
        grade_row.addWidget(self.grade_slider)
        grade_row.addWidget(self.grade_trim_slider)
        self.layout.addLayout(grade_row)

        self.shadow_grade_slider = CompactSlider(tr("Shadows Grade"), -50.0, 50.0, conf.shadow_grade, step=1.0, inverted=True, unit=" R")
        self.highlight_grade_slider = CompactSlider(
            tr("Highlights Grade"), -50.0, 50.0, conf.highlight_grade, step=1.0, inverted=True, unit=" R"
        )
        self.split_grade_fold = rail_fold("tone_split_grade", self.grade_slider, self.shadow_grade_slider, self.highlight_grade_slider)
        self.split_grade_fold.attach(self.grade_trim_slider)
        self.layout.addWidget(self.split_grade_fold)

        # Inverted like ISO-R Grade, so dragging right hardens on both controls.
        self.contrast_mask_slider = CompactSlider(tr("Contrast Mask"), -0.5, 0.5, conf.contrast_mask, has_neutral=True, inverted=True)
        self.contrast_mask_slider.setToolTip(
            tr("Contrast Mask: lowers overall contrast and keeps fine detail, and negative values add contrast.")
        )
        self.mask_spacer_slider = CompactSlider(tr("Mask Spacer"), 2.0, 6.0, conf.mask_spacer, unit="%")
        self.mask_spacer_slider.setToolTip(tr("Mask Spacer: how soft the contrast mask is, where thicker acts on large areas only."))
        # Light given to the paper, not the shape of its curve, so it sits with the exposure controls.
        self.preflash_slider = CompactSlider(tr("Preflash"), 0.0, 1.0, conf.preflash)
        self.diffusion_slider = CompactSlider(tr("Diffusion"), 0.0, 1.0, conf.diffusion)
        self.diffusion_slider.setToolTip(tr("Diffusion: a soft glow that spreads dark areas into bright ones and softens edges, off at 0."))
        self.mask_spacer_fold = rail_fold("tone_mask_spacer", self.contrast_mask_slider, self.mask_spacer_slider)
        self.contrast_mask_slider.valueChanged.connect(lambda _v: self._sync_spacer())
        aids_header, self.aids_fold = group_fold(
            "tone_aids", tr("Printing Aids"), self.preflash_slider, self.diffusion_slider, self.contrast_mask_slider, self.mask_spacer_fold
        )
        self.layout.addLayout(aids_header)
        self.layout.addWidget(self.aids_fold)

        self.layout.addStretch()

        # Global-only controls, greyed while a channel page is active.
        self._global_only = (
            self.density_slider,
            self.test_strip_btn,
            self.auto_btn,
            self.shadow_density_slider,
            self.highlight_density_slider,
            # A pan masking film is neutral: the mask is one plane subtracted as equal
            # density from every layer, so it has no per-channel form to trim.
            self.contrast_mask_slider,
            # A diffuser spreads every layer's light alike.
            self.diffusion_slider,
        )

    def _open_targets_dialog(self) -> None:
        from negpy.desktop.view.widgets.exposure_targets_dialog import ExposureTargetsDialog

        self._targets_snapshot = {k: float(EXPOSURE_CONSTANTS[k]) for k in TUNABLE_TARGETS}
        dlg = ExposureTargetsDialog(self._targets_snapshot, parent=self, repo=self.controller.session.repo)
        dlg.targets_previewed.connect(self._on_targets_preview)
        dlg.finished.connect(lambda result: self._on_targets_finished(dlg, result))
        self._targets_dialog = dlg  # keep a reference so the modeless dialog isn't GC'd
        dlg.show()

    def _on_targets_preview(self, values: dict) -> None:
        apply_targets(values)
        self.controller.request_render(readback_metrics=True)

    def _on_targets_finished(self, dlg, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            values = dlg.values()
            apply_targets(values)
            self.controller.session.repo.save_global_setting("exposure_targets", values)
            if any(abs(values[k] - self._targets_snapshot[k]) > 1e-6 for k in values):
                warn_targets_affect_unbaked(self, self.controller.session.repo)
        else:
            apply_targets(self._targets_snapshot)
        self.controller.request_render(readback_metrics=True)

    def _bake_auto_density_grade(self) -> None:
        """Freeze this frame's metered Auto Density and Auto Grade into its settings, then turn both off.
        The baked values are the sliders' shown values, so the print does not move."""
        shown = shown_values(self.state.config, self._meters())
        changes: dict = dict(shown)
        changes["auto_exposure"] = False
        changes["auto_normalize_contrast"] = False
        self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, **changes)

    def _channel_index(self) -> int:
        return self.ch_btn.currentIndex()

    def _curve_field(self, base: str) -> str:
        idx = self._channel_index()
        return base if idx == 0 else f"{base}_trim_{_CH_SUFFIX[idx - 1]}"

    @staticmethod
    def _test_strip_tooltip(printing: bool = False) -> str:
        if printing:
            return tr("Printing the test strip…")
        return tooltip_with_shortcut(
            tr("Test Strip: show a grid of versions at different brightness and contrast, then click the one you like."),
            "toggle_test_strip",
        )

    def _sync_test_strip_btn(self, _up: bool) -> None:
        """ponytail: the whole strip is one job with no progress reporting — the button is
        icon-only, so it goes dead with a 'printing' tooltip rather than changing its
        label. Per-patch progress if 36 renders ever feels long."""
        # One shared slot, so the kind gates this or the ring lights this button too.
        mine = self.state.test_strip_kind == "tone"
        pending = self.state.test_strip_pending and mine
        self.test_strip_btn.setChecked((self.state.test_strip or self.state.test_strip_pending) and mine)
        self.test_strip_btn.setEnabled(not pending)
        self.test_strip_btn.setToolTip(wrap_tooltip(self._test_strip_tooltip(printing=pending)))

    def _connect_signals(self) -> None:
        # The strip is session state, not config, and any render drops it, so the button has to
        # follow the controller rather than sync_ui.
        self.controller.test_strip_changed.connect(self._sync_test_strip_btn)
        self.ch_btn.currentChanged.connect(lambda _i: self.sync_ui())

        # White Point/Black Point live on ProcessConfig, not ExposureConfig like the rest of
        # this panel, so they write to a different config section than the loop below.

        for field, slider in self._driven_sliders().items():
            slider.valueChanged.connect(lambda v, f=field: self._set_driven(f, v, persist=False))
            slider.valueCommitted.connect(lambda v, f=field: self._set_driven(f, v, persist=True))
            slider.dragStarted.connect(lambda f=field: self.controller.tone_drag_changed.emit(f))
            slider.dragEnded.connect(lambda: self.controller.tone_drag_changed.emit(""))
        self.controller.image_updated.connect(self._sync_driven)
        self.hold_black_action.toggled.connect(
            lambda checked: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, hold_black=checked)
        )

        for slider, field in (
            (self.contrast_mask_slider, "contrast_mask"),
            (self.mask_spacer_slider, "mask_spacer"),
            (self.diffusion_slider, "diffusion"),
            (self.preflash_slider, "preflash"),
        ):
            slider.valueChanged.connect(
                lambda v, f=field: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, **{f: v})
            )
            slider.valueCommitted.connect(
                lambda v, f=field: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, **{f: v})
            )
            slider.dragStarted.connect(lambda f=field: self.controller.tone_drag_changed.emit(f))
            slider.dragEnded.connect(lambda: self.controller.tone_drag_changed.emit(""))

        # Split grade retargets to the selected channel's trim field at emit time.
        for slider, base in (
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

        for btn, field in (
            (self.auto_density_action, "auto_exposure"),
            (self.auto_grade_action, "auto_normalize_contrast"),
        ):
            btn.toggled.connect(
                lambda checked, f=field: self.update_config_section(
                    "exposure", render=True, persist=True, readback_metrics=True, **{f: checked}
                )
            )
        self.auto_both_action.toggled.connect(
            lambda checked: self.update_config_section(
                "exposure", render=True, persist=True, readback_metrics=True, auto_exposure=checked, auto_normalize_contrast=checked
            )
        )

    def _driven_sliders(self) -> dict:
        return {
            "density": self.density_slider,
            "grade": self.grade_slider,
            "shadow_density": self.shadow_density_slider,
            "highlight_density": self.highlight_density_slider,
        }

    def _meters(self) -> dict:
        return self.state.auto_meters.get(self.state.current_file_hash or "", {})

    def _sync_driven(self) -> None:
        """Sliders an auto drives show meter + offset, and reset to the meter's own choice."""
        config, meters = self.state.config, self._meters()
        shown = shown_values(config, meters)
        for field, slider in self._driven_sliders().items():
            default = NEUTRAL[field]
            if field in shown:
                neutral = replace(config, exposure=replace(config.exposure, **{field: NEUTRAL[field]}))
                default = shown_values(neutral, meters)[field]
            slider.blockSignals(True)
            slider.set_default(default)
            slider.setValue(shown.get(field, getattr(config.exposure, field)))
            slider.blockSignals(False)

    def _set_driven(self, field: str, value: float, persist: bool) -> None:
        # A value at the slider's default is a reset: back to the neutral offset exactly.
        if self._driven_sliders()[field].is_default(value):
            stored = NEUTRAL[field]
        else:
            stored = stored_value(self.state.config, self._meters(), field, value)
        self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, **{field: stored})

    def sync_ui(self) -> None:
        conf = self.state.config.exposure
        self.block_signals(True)
        try:
            from negpy.features.process.models import ProcessMode

            mode = self.state.config.process.process_mode

            # On the transfer path (an as-captured Slide, or any Positive frame) the render
            # starts from the capture, so the paper model has nothing to act on. Density,
            # Grade, Toe and Shoulder drive the transfer curve instead (features/transparency/logic.py).
            from negpy.features.process.path import RenderPath, render_path

            proc = self.state.config.process
            transfer = render_path(proc) is not RenderPath.PRINT
            # Shadows and Highlights Density stay live on the transfer path: the curve implements
            # Zone Density with the print's own weights, and they are the only controls there that
            # open shadows without moving the whole scale. Split Grade does not, because it rotates
            # contrast about the same centres and the transfer curve has no per-zone slope to rotate.
            # The transfer curve takes no dodge/burn map, and the mask rides it; it takes no
            # diffusion plane or flash either, and always fades a lift at the bottom of its window.
            for w in (self.split_grade_fold.content, self.aids_fold.content, self.hold_black_action):
                w.setVisible(not transfer)
            # Per-layer trims are meaningless on a single-emulsion B&W paper.
            is_bw = mode == ProcessMode.BW
            if is_bw:
                self.ch_btn.setCurrentIndex(0)
            self.ch_btn.setVisible(not is_bw)

            idx = self._channel_index()
            global_mode = idx == 0
            suffix = _CH_LABEL[idx]
            self.grade_slider.setVisible(global_mode)
            self.grade_trim_slider.setVisible(not global_mode)
            self.shadow_grade_slider.label.setText(tr("Shadows Grade") + suffix)
            self.highlight_grade_slider.label.setText(tr("Highlights Grade") + suffix)
            if global_mode:
                self.shadow_grade_slider.setValue(conf.shadow_grade)
                self.highlight_grade_slider.setValue(conf.highlight_grade)
            else:
                ch = _CH_SUFFIX[idx - 1]
                self.grade_trim_slider.label.setText(tr("Grade") + suffix)
                self.grade_trim_slider.setValue(getattr(conf, f"grade_trim_{ch}"))
                self.shadow_grade_slider.setValue(getattr(conf, f"shadow_grade_trim_{ch}"))
                self.highlight_grade_slider.setValue(getattr(conf, f"highlight_grade_trim_{ch}"))
            for w in self._global_only:
                w.setEnabled(global_mode)
            # WorkspaceConfig holds both off on a merge; greyed so the reason can show.
            merged = hdr_active(self.state.config.hdr)
            for action in (self.auto_density_action, self.auto_grade_action, self.auto_both_action):
                action.setEnabled(not merged)
            # Bake is a no-op when both autos are already off, and is unreachable on a merge (the auto button is off).
            self.auto_bake_action.setEnabled(not merged and (conf.auto_exposure or conf.auto_normalize_contrast))
            self.auto_merged_hint.setVisible(merged)

            for i, fields in enumerate(self._channel_fields, start=1):
                self.ch_btn.set_edited(i, any(getattr(conf, f) != 0.0 for f in fields))
            self._sync_driven()
            self.contrast_mask_slider.setValue(conf.contrast_mask)
            self.mask_spacer_slider.setValue(conf.mask_spacer)
            self.diffusion_slider.setValue(conf.diffusion)
            self.preflash_slider.setValue(conf.preflash)
            self.hold_black_action.setChecked(conf.hold_black)
            self._sync_spacer()
            self.auto_density_action.setChecked(conf.auto_exposure)
            self.auto_grade_action.setChecked(conf.auto_normalize_contrast)
            self.auto_both_action.setChecked(conf.auto_exposure and conf.auto_normalize_contrast)
            self.auto_btn.refresh()
        finally:
            self.block_signals(False)

    def _sync_spacer(self) -> None:
        # Out of _global_only: that tuple means enabled exactly when global.
        self.mask_spacer_slider.setEnabled(self._channel_index() == 0 and self.contrast_mask_slider.value() != 0.0)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.ch_btn,
            self.density_slider,
            self.grade_slider,
            self.grade_trim_slider,
            self.shadow_density_slider,
            self.highlight_density_slider,
            self.shadow_grade_slider,
            self.highlight_grade_slider,
            self.contrast_mask_slider,
            self.mask_spacer_slider,
            self.diffusion_slider,
            self.preflash_slider,
            self.hold_black_action,
            self.auto_density_action,
            self.auto_grade_action,
            self.auto_both_action,
        ):
            w.blockSignals(blocked)
