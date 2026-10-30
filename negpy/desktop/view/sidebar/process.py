from negpy.kernel.system.i18n import tr
from negpy.kernel.system.text import count_of
import math

import numpy as np
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.session import ToolMode
from negpy.desktop.view.sidebar.base import BaseSidebar, SliderPage
from negpy.desktop.view.sidebar.tone import _CH_LABEL, _CH_SUFFIX, channel_pages, channel_selector
from negpy.desktop.view.styles.templates import (
    ICON_BUTTON_WIDTH,
    clearable_tool_toggle,
    hint_label,
    header_row,
    section_subheader,
    set_hint_kind,
    wrap_tooltip,
)
from negpy.desktop.view.widgets.stats import CLIPPING_TOOLTIP
from negpy.features.exposure.stats import clipping_row
from negpy.services.assets import rolls
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.segmented_choice import RightClickToggleAll, SegmentedChoice
from negpy.desktop.view.widgets.fold import group_fold, rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.hdr.logic import output_scale
from negpy.features.hdr.models import ANCHOR_EV_UNSET, hdr_active
from negpy.features.process.models import ProcessMode, invalidate_local_bounds

# Luma Range Clip slider mapping: positions 0 to 100 clip the histogram tails, and
# negative positions map to an outward log-density margin, a gentler-than-zero stretch.
_LUMA_MARGIN_MIN = 1e-6
_LUMA_MARGIN_MAX = 1.0

# Color Clip slider: the per-channel-balance sampling depth, log-interpolated around the
# neutral (pos 0 = base_color_clip). It sets the luma-band depth of the same-pixel
# dense-end refs and the clip percentile of the thin-end fallback pass. The ends reach
# _COLOR_CLIP_MIN (gentlest, near-extreme bounds) and _COLOR_CLIP_MAX (tightest balance).
_COLOR_CLIP_NEUTRAL = float(EXPOSURE_CONSTANTS["base_color_clip"])
_COLOR_CLIP_MIN = 1e-6
_COLOR_CLIP_MAX = 5.0

# Mode bar: one film icon per mode, with the color carrying which one. Orange mask,
# silver grey, slide blue.
_MODES = (
    (ProcessMode.C41, tr("Color"), THEME.mode_c41, tr("Color Negative (C-41): orange-masked negative")),
    (ProcessMode.BW, tr("B&W"), THEME.mode_bw, tr("B&W: a black-and-white print, from a B&W negative or a color negative.")),
    (ProcessMode.E6, tr("Slide"), THEME.mode_e6, tr("Transparency: slide or reversal film")),
)


def _luma_range_slider_to_value(pos: float) -> float:
    if pos >= 0:
        return math.pow(10, 0.05 * pos - 5)
    lo, hi = math.log10(_LUMA_MARGIN_MIN), math.log10(_LUMA_MARGIN_MAX)
    margin = math.pow(10, lo + (-pos / 100.0) * (hi - lo))
    return -margin


def _luma_range_value_to_slider(v: float) -> float:
    if v >= 0:
        return 20 * (math.log10(max(v, 1e-5)) + 5)
    lo, hi = math.log10(_LUMA_MARGIN_MIN), math.log10(_LUMA_MARGIN_MAX)
    return -100.0 * (math.log10(-v) - lo) / (hi - lo)


def _color_slider_to_value(pos: float) -> float:
    ln = math.log10(_COLOR_CLIP_NEUTRAL)
    end = math.log10(_COLOR_CLIP_MAX if pos >= 0 else _COLOR_CLIP_MIN)
    return math.pow(10, ln + (abs(pos) / 100.0) * (end - ln))


def _color_value_to_slider(v: float) -> float:
    ln = math.log10(_COLOR_CLIP_NEUTRAL)
    lv = math.log10(min(max(v, _COLOR_CLIP_MIN), _COLOR_CLIP_MAX))
    if v >= _COLOR_CLIP_NEUTRAL:
        return 100.0 * (lv - ln) / (math.log10(_COLOR_CLIP_MAX) - ln)
    return -100.0 * (lv - ln) / (math.log10(_COLOR_CLIP_MIN) - ln)


class ProcessSidebar(BaseSidebar):
    """
    Panel for core film processing, normalization, and roll management.
    """

    def set_clipping(self, clip_low: float | None, clip_high: float | None) -> None:
        """Show the print's shadow and highlight clipping shares: the Analysis stats' Clipping row."""
        row = clipping_row(clip_low, clip_high)
        self.clipping_hint.setText(tr("{value}: {value1}", value=row.name, value1=row.value))
        set_hint_kind(self.clipping_hint, "warning" if row.warn else "muted")

    def _init_ui(self) -> None:
        conf = self.state.config.process

        # The top of the "Setup" Roll-tab card -- ControlsPanel wraps it in that section,
        # so it carries no header of its own; deliberately not in
        # self.layout, the same reason analysis_bar below is not.
        self.mode_bar = QWidget()
        mode_col = QVBoxLayout(self.mode_bar)
        mode_col.setContentsMargins(0, 0, 0, 0)
        mode_col.setSpacing(THEME.space_sm)

        mode_row = QHBoxLayout()
        mode_col.addLayout(mode_row)
        self.mode_btn = SegmentedChoice(tuple(("mdi6.film", label, color) for _mode, label, color, _tip in _MODES), tr("Film process"))
        for i, (*_rest, tip) in enumerate(_MODES):
            self.mode_btn.set_choice_tooltip(i, tip)
        mode_row.addWidget(self.mode_btn, 1)

        # Lives beside Film Mode, not inside Normalization: whether the source is
        # already a finished positive is a fact about the file, not a Normalization
        # setting to dig for.
        self.positive_source_btn = self._labeled_toggle(
            "fa5s.image",
            tr("Positive"),
            conf.positive_source,
            (tr("Positive, slides only: the file is already a positive, such as a scanned print or another app's export.")),
        )
        mode_col.addWidget(self.positive_source_btn)

        self.lock_bounds_btn = self._small_toggle(
            "fa5s.lock",
            "",
            False,
            tr("Lock Bounds: keep this frame's measured black and white points, even when the crop or Roll Analysis changes."),
        )

        # Everything that measures this frame, or nudges what the measurement produced.
        # ControlsPanel places it on the Metering card, so it sits outside self.layout --
        # the same reason mode_bar sits above every Roll-tab card.
        self.analysis_bar = QWidget()
        # Nested in a card body, so it takes the body's transparent rule, not the global QWidget fill.
        self.analysis_bar.setObjectName("collapsible_content_body")
        analysis_col = QVBoxLayout(self.analysis_bar)
        analysis_col.setContentsMargins(0, 0, 0, 0)
        analysis_col.setSpacing(THEME.space_sm)
        self.analysis_buffer_slider = CompactSlider(tr("Analysis Buffer"), 0.0, 0.25, conf.analysis_buffer)
        self.reanalyze_frame_btn = self._icon_action("fa5s.redo", tr("Reanalyze Frame: measure this frame again from its current crop."))
        self.lock_bounds_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.analysis_header = section_subheader(tr("ANALYSIS"))
        analysis_col.addLayout(header_row(self.analysis_header, self.reanalyze_frame_btn, self.lock_bounds_btn))
        self.clipping_hint = hint_label("")
        self.clipping_hint.setToolTip(wrap_tooltip(CLIPPING_TOOLTIP))
        analysis_col.addWidget(self.clipping_hint)
        self.set_clipping(None, None)

        self.analysis_region_btn = clearable_tool_toggle(
            "fa5s.vector-square",
            tr("Draw Region"),
            tr("Draw a region on the picture for NegPy to measure, then double-click inside it to confirm."),
            tr("Clear Region: measure with the Analysis Buffer again."),
        )
        self.clear_analysis_region_btn = self.analysis_region_btn.clear_btn
        self.film_base_btn = clearable_tool_toggle(
            "fa5s.crosshairs",
            tr("Film Base"),
            tr("Film Base: click the clear film between frames to set the film base of the open roll, or of this frame without one."),
            tr("Clear Film Base: forget the film base this frame uses."),
        )
        self.clear_film_base_btn = self.film_base_btn.clear_btn
        tools_row = QHBoxLayout()
        tools_row.addWidget(self.analysis_region_btn, 1)
        tools_row.addWidget(self.film_base_btn, 1)
        analysis_col.addLayout(tools_row)

        initial_luma_slider_val = _luma_range_value_to_slider(conf.luma_range_clip)
        self.luma_range_clip_slider = CompactSlider(
            tr("Luma Range Clip"), -100, 100, initial_luma_slider_val, precision=1, step=1, has_neutral=True
        )
        initial_color_slider_val = _color_value_to_slider(conf.color_range_clip)
        self.color_range_clip_slider = CompactSlider(
            tr("Color Clip"), -100, 100, initial_color_slider_val, precision=1, step=1, has_neutral=True
        )
        self.clip_fold = rail_fold("metering_clip", self.analysis_buffer_slider, self.luma_range_clip_slider, self.color_range_clip_slider)
        analysis_col.addWidget(self.analysis_buffer_slider)
        analysis_col.addWidget(self.clip_fold)

        self.ch_btn = channel_selector(
            tr("Global changes all colors, and R, G or B changes one color layer."),
        )
        self.white_point_slider = CompactSlider(tr("White Point"), -0.25, 0.25, conf.white_point_offset, has_neutral=True)
        self.black_point_slider = CompactSlider(tr("Black Point"), -0.25, 0.25, conf.black_point_offset, has_neutral=True)
        point_row, point_fold = group_fold(
            "metering_points", tr("White / Black Point"), self.ch_btn, self.white_point_slider, self.black_point_slider
        )
        self.point_header = point_fold.button
        analysis_col.addLayout(point_row)
        analysis_col.addWidget(point_fold)

        # Which baseline each axis' bounds come from: the roll's shared meter or the frame's
        # own analysis. ControlsPanel places it at the top of the Roll Analysis card.
        self.baseline_bar = QWidget()
        # Nested in a card body, so it takes the body's transparent rule, not the global QWidget fill.
        self.baseline_bar.setObjectName("collapsible_content_body")
        baseline_col = QVBoxLayout(self.baseline_bar)
        baseline_col.setContentsMargins(0, 0, 0, 0)
        baseline_col.setSpacing(THEME.space_sm)
        self.avg_header = section_subheader(tr("USE AVERAGE"))
        baseline_col.addWidget(self.avg_header)
        self.baseline_source_hint = hint_label("")
        baseline_col.addWidget(self.baseline_source_hint)
        avg_row = QHBoxLayout()
        self.use_luma_avg_btn = self._small_toggle(
            "mdi6.brightness-6",
            tr("Luma"),
            conf.use_luma_average,
            tr("Use the roll's average black and white points, with color still per frame; right-click turns every average on or off."),
        )
        self.use_color_avg_btn = self._small_toggle(
            "mdi6.palette-outline",
            tr("Color"),
            conf.use_color_average,
            tr("Use the roll's average color balance, with black and white points still per frame."),
        )
        self.use_cast_avg_btn = self._small_toggle(
            "mdi6.scale-balance",
            tr("Cast"),
            conf.use_cast_average,
            tr("Color negatives only: use the roll's gray balance for Cast Removal, so frames under one light match."),
        )
        for btn in (self.use_luma_avg_btn, self.use_color_avg_btn, self.use_cast_avg_btn):
            avg_row.addWidget(btn)
        baseline_col.addLayout(avg_row)
        self._use_all_averages = RightClickToggleAll([self.use_luma_avg_btn, self.use_color_avg_btn, self.use_cast_avg_btn])

        # Render exposure for a merged bracket, continuous rather than snapped to the frames that
        # happen to have been shot. The menu still offers those and writes a frame name; this
        # writes a value and wins. 0 = the reference, the brightest unclipped frame, which is the
        # most a merge can open at. output_scale clamps above it.
        self.render_ev_slider = CompactSlider(tr("Render Exposure"), -4.0, 0.0, 0.0, step=0.05, unit=" st")
        self.render_ev_slider.setToolTip(
            wrap_tooltip(tr("Which exposure of a merged bracket to show, in stops below the brightest usable shot."))
        )
        # Live again: the merge is cached unscaled, so a change of exposure is one multiply on the
        # cached buffer rather than another decode of the bracket. valueChanged is already
        # trailing-debounced, so a drag costs a few of those, not a few decodes.
        self.render_ev_slider.valueChanged.connect(lambda v: self.controller.set_hdr_anchor_ev(float(v), persist=False))
        self.render_ev_slider.valueCommitted.connect(lambda v: self.controller.set_hdr_anchor_ev(float(v)))
        self.render_ev_slider.setVisible(False)

        self.layout.addWidget(self.render_ev_slider)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.mode_btn.currentChanged.connect(lambda i: self._on_mode_changed(_MODES[i][0]))
        self.lock_bounds_btn.toggled.connect(self._on_lock_bounds_toggled)

        self.reanalyze_frame_btn.clicked.connect(self._on_reanalyze_frame)
        self.analysis_buffer_slider.valueChanged.connect(lambda v: self._on_buffer_changed(v, persist=False))
        self.analysis_buffer_slider.valueCommitted.connect(lambda v: self._on_buffer_changed(v, persist=True))
        self.analysis_buffer_slider.dragStarted.connect(lambda: self.controller.analysis_buffer_drag_changed.emit(True))
        self.analysis_buffer_slider.dragEnded.connect(lambda: self.controller.analysis_buffer_drag_changed.emit(False))
        self.analysis_region_btn.toggled.connect(self._on_analysis_region_toggled)
        self.clear_analysis_region_btn.clicked.connect(self.controller.clear_analysis_region)
        self.film_base_btn.toggled.connect(lambda on: self.controller.set_active_tool(ToolMode.FILM_BASE if on else ToolMode.NONE))
        self.clear_film_base_btn.clicked.connect(self.controller.clear_film_base)

        self.luma_range_clip_slider.valueChanged.connect(lambda v: self._on_luma_range_clip_changed(v, persist=False))
        self.luma_range_clip_slider.valueCommitted.connect(lambda v: self._on_luma_range_clip_changed(v, persist=True))

        self.color_range_clip_slider.valueChanged.connect(lambda v: self._on_color_range_clip_changed(v, persist=False))
        self.color_range_clip_slider.valueCommitted.connect(lambda v: self._on_color_range_clip_changed(v, persist=True))

        self.positive_source_btn.toggled.connect(self._on_positive_source_toggled)
        self.use_luma_avg_btn.toggled.connect(self._on_use_luma_average_toggled)
        self.use_color_avg_btn.toggled.connect(self._on_use_color_average_toggled)
        self.use_cast_avg_btn.toggled.connect(self._on_use_cast_average_toggled)
        self._use_all_averages.toggledAll.connect(self._on_use_all_averages)

        self.white_point_slider.valueChanged.connect(lambda v: self._on_white_point_changed(v, persist=False))
        self.white_point_slider.valueCommitted.connect(lambda v: self._on_white_point_changed(v, persist=True))
        self.black_point_slider.valueChanged.connect(lambda v: self._on_black_point_changed(v, persist=False))
        self.black_point_slider.valueCommitted.connect(lambda v: self._on_black_point_changed(v, persist=True))
        self.ch_btn.currentChanged.connect(lambda _i: self.sync_ui())
        self.sync_ui()

    def _channel_index(self) -> int:
        return self.ch_btn.currentIndex()

    def slider_pages(self) -> tuple[SliderPage, ...]:
        return channel_pages(
            self.ch_btn,
            self.state.config.process,
            (
                (tr("White Point"), self.white_point_slider, self.white_point_slider, "white_point_offset", "white_point_trim"),
                (tr("Black Point"), self.black_point_slider, self.black_point_slider, "black_point_offset", "black_point_trim"),
            ),
        )

    def _wp_field(self) -> str:
        idx = self._channel_index()
        return "white_point_offset" if idx == 0 else f"white_point_trim_{_CH_SUFFIX[idx - 1]}"

    def _bp_field(self) -> str:
        idx = self._channel_index()
        return "black_point_offset" if idx == 0 else f"black_point_trim_{_CH_SUFFIX[idx - 1]}"

    def _on_white_point_changed(self, val: float, persist: bool = True) -> None:
        self.controller.set_roll_default("process", persist=persist, readback_metrics=persist, **{self._wp_field(): val})

    def _on_black_point_changed(self, val: float, persist: bool = True) -> None:
        self.controller.set_roll_default("process", persist=persist, readback_metrics=persist, **{self._bp_field(): val})

    def _on_lock_bounds_toggled(self, checked: bool) -> None:
        self.update_config_section("process", lock_bounds=checked, persist=True, render=False)
        self.sync_ui()

    def _on_mode_changed(self, mode: str) -> None:
        self.controller.set_process_mode(mode)
        self.sync_ui()

    def _on_positive_source_toggled(self, checked: bool) -> None:
        self.controller.set_positive_source(checked)

    def _on_use_luma_average_toggled(self, checked: bool) -> None:
        self._toggle_roll_axis(use_luma_average=checked)

    def _on_use_color_average_toggled(self, checked: bool) -> None:
        self._toggle_roll_axis(use_color_average=checked)

    def _on_use_cast_average_toggled(self, checked: bool) -> None:
        self.controller.set_roll_average(use_cast_average=checked)

    def _on_use_all_averages(self, checked: bool) -> None:
        fields = {
            self.use_luma_avg_btn: "use_luma_average",
            self.use_color_avg_btn: "use_color_average",
            self.use_cast_avg_btn: "use_cast_average",
        }
        self.controller.set_roll_average(**{f: checked for a, f in fields.items() if not a.isHidden() and a.isEnabled()})

    def _toggle_roll_axis(self, **axis: bool) -> None:
        self.controller.set_roll_average(**axis)

    def _on_analysis_region_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.ANALYSIS_DRAW if checked else ToolMode.NONE)

    def _on_buffer_changed(self, val: float, persist: bool = True) -> None:
        self.controller.set_roll_default(
            "process",
            persist=persist,
            readback_metrics=persist,
            analysis_buffer=val,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.controller.analysis_buffer_preview_requested.emit(val)

    def _cast_baseline_ready(self, conf) -> bool:
        """A neutral axis on this frame, or on the baseline its Cast toggle would take."""
        if conf.locked_neutral_axis is not None:
            return True
        state = self.controller.state
        found = rolls.frame_saved_baseline(
            self.controller.session.repo, state.active_roll_id, state.current_file_hash or "", rolls.average_scope(conf)
        )
        return found is not None and found[0]["axis"] is not None

    def _update_baseline_source_hint(self, conf, transfer: bool) -> None:
        """Names the baseline the average axes read; hidden while neither axis rides one."""
        riding = (conf.use_luma_average or conf.use_color_average) and not transfer
        self.baseline_source_hint.setVisible(riding)
        if not riding:
            return
        if conf.is_locked_initialized:
            repo = self.controller.session.repo
            stale = rolls.frame_baseline_stale(repo, conf)
            outlier = rolls.frame_is_outlier(repo, conf, self.controller.state.current_file_hash or "")
            set_hint_kind(self.baseline_source_hint, "warning" if stale or outlier else "muted")
            baseline = rolls.baseline_label(repo, conf)
            if stale:
                text = tr("Baseline: {value}, from an older analysis", value=baseline)
            elif outlier:
                text = tr(
                    "Baseline: {value}. This frame's color is far from the average, so the average can shift its color", value=baseline
                )
            else:
                text = tr("Baseline: {value}", value=baseline)
            self.baseline_source_hint.setText(text)
        elif self.controller.measuring_baseline() is True:
            set_hint_kind(self.baseline_source_hint, "muted")
            self.baseline_source_hint.setText(tr("Measuring the average…"))
        else:
            set_hint_kind(self.baseline_source_hint, "warning")
            self.baseline_source_hint.setText(tr("No baseline yet: this frame uses its own analysis until Roll Analysis runs"))

    def _sync_reanalyze_btn(self, locked: bool) -> None:
        on_selection = not self.controller.session.selection_is_open_frame()
        self.reanalyze_frame_btn.setEnabled(on_selection or not locked)
        if on_selection:
            n = count_of(len(self.state.selected_indices), "frame").title()
            tip = tr("Reanalyze {value}: measure each selected frame again on its own bounds.", value=n)
        else:
            tip = tr("Reanalyze Frame: measure this frame again from its current crop.")
        self.reanalyze_frame_btn.setToolTip(wrap_tooltip(tip))

    def _on_reanalyze_frame(self) -> None:
        self.controller.reanalyze_selection()

    def _on_luma_range_clip_changed(self, val: float, persist: bool = True) -> None:
        self.controller.set_roll_default(
            "process",
            persist=persist,
            readback_metrics=persist,
            luma_range_clip=_luma_range_slider_to_value(val),
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_color_range_clip_changed(self, val: float, persist: bool = True) -> None:
        self.controller.set_roll_default(
            "process",
            persist=persist,
            readback_metrics=persist,
            color_range_clip=_color_slider_to_value(val),
            **invalidate_local_bounds(self.state.config.process),
        )

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            self.mode_btn.setCurrentIndex(next(i for i, (mode, *_rest) in enumerate(_MODES) if mode == conf.process_mode))
            self.analysis_buffer_slider.setValue(conf.analysis_buffer)
            self.luma_range_clip_slider.setValue(_luma_range_value_to_slider(conf.luma_range_clip))
            self.color_range_clip_slider.setValue(_color_value_to_slider(conf.color_range_clip))
            self.use_luma_avg_btn.setChecked(conf.use_luma_average)
            self.use_color_avg_btn.setChecked(conf.use_color_average)
            self.use_cast_avg_btn.setChecked(conf.use_cast_average)

            # Transparency transfer: the stretch is a fixed window anchored to the decoder's white
            # level, so nothing that tunes a measured stretch has anything to act on.
            from negpy.features.process.path import RenderPath, render_path

            is_e6 = conf.process_mode == ProcessMode.E6
            transfer = render_path(conf) is not RenderPath.PRINT

            # Greyed on a merge, not hidden: the render already ignores it, since WorkspaceConfig
            # holds that invariant, and a control that vanishes teaches nothing about why.
            merged = hdr_active(self.state.config.hdr)

            self.positive_source_btn.setVisible(is_e6)
            self.positive_source_btn.setChecked(conf.positive_source)

            # Only a merge has a render exposure to choose, and only the transfer path uses a fixed
            # window for it to mean anything against.
            self.render_ev_slider.setVisible(merged and transfer)
            if merged:
                hdr = self.state.config.hdr
                ev = float(hdr.hdr_anchor_ev)
                if ev >= ANCHOR_EV_UNSET:
                    # Unset means the bracket's middle exposure, which is only 0 EV when it clamps there. A
                    # bare 0.00 would misreport where the picture sits on any bracket spread either side of
                    # the reference.
                    scale = output_scale([float(r) for r in hdr.hdr_ratios])
                    ev = float(np.log2(scale)) if scale > 0 else 0.0
                self.render_ev_slider.blockSignals(True)
                self.render_ev_slider.setValue(ev)
                self.render_ev_slider.blockSignals(False)

            self.lock_bounds_btn.setChecked(conf.lock_bounds)

            has_region = conf.analysis_rect is not None
            self.analysis_region_btn.setChecked(self.state.active_tool == ToolMode.ANALYSIS_DRAW)
            self.clear_analysis_region_btn.setVisible(has_region)
            has_base = self.controller.clearable_film_base()
            self.film_base_btn.setChecked(self.state.active_tool == ToolMode.FILM_BASE)
            self.clear_film_base_btn.setVisible(has_base)

            for w in (
                self.analysis_buffer_slider,
                self.analysis_region_btn,
                self.film_base_btn,
                self.avg_header,
                self.use_luma_avg_btn,
                self.use_color_avg_btn,
                self.clip_fold.content,
                self.lock_bounds_btn,
                self.reanalyze_frame_btn,
            ):
                w.setVisible(not transfer)
            self.use_cast_avg_btn.setVisible(conf.process_mode == ProcessMode.C41)
            # A roll baseline is what the averages read, so without a roll they only turn off;
            # the Roll Analysis hint above says why.
            has_roll = bool(self.controller.state.active_roll_id)
            self.use_luma_avg_btn.setEnabled(has_roll or conf.use_luma_average)
            self.use_color_avg_btn.setEnabled(has_roll or conf.use_color_average)
            self.use_cast_avg_btn.setEnabled((has_roll and self._cast_baseline_ready(conf)) or conf.use_cast_average)
            self._update_baseline_source_hint(conf, transfer)

            idx = self._channel_index()
            suffix = _CH_LABEL[idx]
            self.white_point_slider.label.setText(tr("White Point") + suffix)
            self.black_point_slider.label.setText(tr("Black Point") + suffix)
            if idx == 0:
                self.white_point_slider.setValue(conf.white_point_offset)
                self.black_point_slider.setValue(conf.black_point_offset)
            else:
                ch = _CH_SUFFIX[idx - 1]
                self.white_point_slider.setValue(getattr(conf, f"white_point_trim_{ch}"))
                self.black_point_slider.setValue(getattr(conf, f"black_point_trim_{ch}"))
            for i, ch in enumerate(_CH_SUFFIX, start=1):
                self.ch_btn.set_edited(i, getattr(conf, f"white_point_trim_{ch}") != 0.0 or getattr(conf, f"black_point_trim_{ch}") != 0.0)

            # Locked bounds are frozen, so there is nothing left to nudge. The transfer path's
            # window is never measured, so Lock Bounds does not reach it.
            for w in (self.white_point_slider, self.black_point_slider):
                w.setEnabled(transfer or not conf.lock_bounds)

            locked = conf.lock_bounds
            # Each clip slider is disabled when its axis rides the roll baseline. The analysis buffer
            # matters only when at least one axis still analyzes locally, and a freehand analysis
            # region overrides it entirely.
            self.analysis_buffer_slider.setEnabled(not locked and not has_region and not (conf.use_luma_average and conf.use_color_average))
            self._sync_reanalyze_btn(locked)
            self.luma_range_clip_slider.setEnabled(not locked and not conf.use_luma_average)
            self.color_range_clip_slider.setEnabled(not locked and not conf.use_color_average)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all sliders and buttons.
        """
        widgets = [
            self.mode_btn,
            self.lock_bounds_btn,
            self.analysis_buffer_slider,
            self.analysis_region_btn,
            self.use_luma_avg_btn,
            self.use_color_avg_btn,
            self.use_cast_avg_btn,
            self.luma_range_clip_slider,
            self.color_range_clip_slider,
            self.positive_source_btn,
            self.ch_btn,
            self.white_point_slider,
            self.black_point_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
