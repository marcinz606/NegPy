import math

import numpy as np
import qtawesome as qta
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.session import ToolMode
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.sidebar.tone import _CH_COLORS, _CH_LABEL, _CH_SUFFIX
from negpy.desktop.view.styles.templates import EditedDot, hint_label, section_subheader, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.hdr.logic import output_scale
from negpy.features.hdr.models import ANCHOR_EV_UNSET, hdr_active
from negpy.features.process.models import ProcessMode, invalidate_local_bounds

# Luma Range Clip slider mapping: positions 0..100 clip the histogram tails; negative
# positions -100..0 map to an outward log-density margin (gentler-than-zero stretch).
_LUMA_MARGIN_MIN = 1e-6
_LUMA_MARGIN_MAX = 1.0

# Color Clip slider: the per-channel-balance sampling depth, log-interpolated
# around the neutral (pos 0 = base_color_clip) — the luma-band depth of the
# same-pixel dense-end refs and the clip percentile of the thin-end/fallback
# pass. The ends reach _COLOR_CLIP_MIN (gentlest, near-extreme bounds) and
# _COLOR_CLIP_MAX (tightest channel balance).
_COLOR_CLIP_NEUTRAL = float(EXPOSURE_CONSTANTS["base_color_clip"])
_COLOR_CLIP_MIN = 1e-6
_COLOR_CLIP_MAX = 5.0

# Mode bar: one film icon per mode, the color carries which one — orange mask,
# silver grey, slide blue.
_MODES = (
    (ProcessMode.C41, " Color", "#E08A3C", "Color Negative (C-41) — orange-masked negative"),
    (ProcessMode.BW, " B&&W", "#8C8C8C", "B&W Negative — panchromatic silver negative"),
    (ProcessMode.E6, " Slide", "#4FB0D8", "Transparency — slide / reversal film"),
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

    def _init_ui(self) -> None:
        conf = self.state.config.process

        # Lives above every Setup collapsible: ControlsPanel adds it to the page, so it
        # is deliberately not in self.layout.
        self.mode_bar = QWidget()
        mode_col = QVBoxLayout(self.mode_bar)
        mode_col.setContentsMargins(0, 0, 0, 0)
        mode_col.setSpacing(THEME.space_sm)

        self.autodetect_btn = self._small_toggle("mdi6.auto-fix", "", False, "Auto-detect the film process on load")
        self.autodetect_btn.setFixedWidth(28)
        header_row = QHBoxLayout()
        header_row.addWidget(section_subheader("Process"))
        header_row.addStretch(1)
        header_row.addWidget(self.autodetect_btn)
        mode_col.addLayout(header_row)

        mode_row = QHBoxLayout()
        mode_col.addLayout(mode_row)
        self.mode_btns = []
        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.setExclusive(True)
        for i, (mode, label, color, tip) in enumerate(_MODES):
            btn = self._labeled_toggle("mdi6.film", label, mode == conf.process_mode, tip)
            btn.setIcon(qta.icon("mdi6.film", color=color))
            self.mode_btn_group.addButton(btn, i)
            mode_row.addWidget(btn, 1)
            self.mode_btns.append(btn)

        self.lock_bounds_btn = self._small_toggle(
            "fa5s.lock",
            "",
            False,
            "Lock Bounds — freeze normalization bounds so crop and analysis sliders no longer re-analyze",
        )

        buf_row = QHBoxLayout()
        self.analysis_buffer_slider = CompactSlider("Analysis Buffer", 0.0, 0.25, conf.analysis_buffer)
        self.analysis_region_btn = self._tool_toggle(
            "fa5s.vector-square",
            "",
            "Draw a freehand analysis region on the image — the meters read exactly that area "
            "(overrides the Analysis Buffer). Double-click inside it to confirm.",
        )
        # Confirming a region closes the tool (unchecking the toggle), so the dot is
        # the only cue left that it's still overriding the Analysis Buffer slider.
        self.analysis_region_btn.edited_dot = EditedDot(self.analysis_region_btn)
        self.clear_analysis_region_btn = self._icon_action(
            "fa5s.times", "Clear the freehand analysis region (fall back to the Analysis Buffer)", width=None
        )
        # Slider takes half the row; the three buttons split the other half, equal
        # stretch (not fixed widths) being what keeps them the same size.
        buf_row.addWidget(self.analysis_buffer_slider, 3)
        for btn in (self.analysis_region_btn, self.clear_analysis_region_btn, self.lock_bounds_btn):
            buf_row.addWidget(btn, 1)
        self.layout.addLayout(buf_row)

        clip_row = QHBoxLayout()
        initial_luma_slider_val = _luma_range_value_to_slider(conf.luma_range_clip)
        self.luma_range_clip_slider = CompactSlider(
            "Luma Range Clip", -100, 100, initial_luma_slider_val, precision=1, step=1, has_neutral=True
        )
        initial_color_slider_val = _color_value_to_slider(conf.color_range_clip)
        self.color_range_clip_slider = CompactSlider(
            "Color Clip", -100, 100, initial_color_slider_val, precision=1, step=1, has_neutral=True
        )
        clip_row.addWidget(self.luma_range_clip_slider)
        clip_row.addWidget(self.color_range_clip_slider)
        self.layout.addLayout(clip_row)

        # Channel selector scoped to the White/Black Point row below it:
        # Global = the shared offsets; R/G/B = per-layer trims (film-base / Dmax).
        self.ch_global_btn = self._labeled_toggle("fa5s.globe", " Global", True, "Global — shared white/black point offsets (all layers)")
        self.ch_r_btn = self._labeled_toggle(
            "fa5s.circle", " Red", False, "Red layer — per-layer white/black point trims (cyan-dye film base / Dmax)"
        )
        self.ch_g_btn = self._labeled_toggle(
            "fa5s.circle", " Green", False, "Green layer — per-layer white/black point trims (magenta-dye film base / Dmax)"
        )
        self.ch_b_btn = self._labeled_toggle(
            "fa5s.circle", " Blue", False, "Blue layer — per-layer white/black point trims (yellow-dye film base / Dmax)"
        )
        for btn, color in zip((self.ch_r_btn, self.ch_g_btn, self.ch_b_btn), _CH_COLORS):
            btn.setIcon(qta.icon("fa5s.circle", color=color))
        self.ch_btn_group = QButtonGroup(self)
        self.ch_btn_group.setExclusive(True)
        for i, btn in enumerate((self.ch_global_btn, self.ch_r_btn, self.ch_g_btn, self.ch_b_btn)):
            self.ch_btn_group.addButton(btn, i)
        self._channel_buttons = tuple(
            (btn, (f"white_point_trim_{ch}", f"black_point_trim_{ch}"))
            for btn, ch in zip((self.ch_r_btn, self.ch_g_btn, self.ch_b_btn), _CH_SUFFIX)
        )
        ch_row = QHBoxLayout()
        for btn in (self.ch_global_btn, self.ch_r_btn, self.ch_g_btn, self.ch_b_btn):
            ch_row.addWidget(btn, 1)
        self.layout.addLayout(ch_row)

        wp_bp_row = QHBoxLayout()
        self.white_point_slider = CompactSlider("White Point", -0.25, 0.25, conf.white_point_offset, has_neutral=True)
        self.black_point_slider = CompactSlider("Black Point", -0.25, 0.25, conf.black_point_offset, has_neutral=True)
        wp_bp_row.addWidget(self.white_point_slider)
        wp_bp_row.addWidget(self.black_point_slider)
        self.layout.addLayout(wp_bp_row)

        # Render exposure for a merged bracket, continuous rather than snapped to the frames
        # that happen to have been shot. The menu still offers those, and writes a frame name;
        # this writes a value and wins. 0 = the reference (the brightest unclipped frame),
        # which is the most a merge can open at — output_scale clamps above it.
        self.render_ev_slider = CompactSlider("Render Exposure", -4.0, 0.0, 0.0, step=0.05, unit=" EV")
        self.render_ev_slider.setToolTip(
            wrap_tooltip(
                "Which exposure a merged bracket renders at, in stops below the reference frame. "
                "The reference is the longest capture that does not clip, so it is the brightest "
                "the merge can open at — a slide's own highlights are denser than clear film, so "
                "that is usually brighter than the shot you metered for.<br><br>"
                "Right-click the frame for <b>Render exposure</b> to snap to an exposure you "
                "actually shot; this slider goes anywhere between them."
            )
        )
        # Live again: the merge is cached unscaled, so a change of exposure is one multiply
        # on the cached buffer rather than another decode of the bracket. valueChanged is
        # already trailing-debounced, so a drag costs a few of those, not a few decodes.
        self.render_ev_slider.valueChanged.connect(lambda v: self.controller.set_hdr_anchor_ev(float(v), persist=False))
        self.render_ev_slider.valueCommitted.connect(lambda v: self.controller.set_hdr_anchor_ev(float(v)))
        self.render_ev_slider.setVisible(False)

        self.normalize_e6_btn = QPushButton(" Normalize")
        self.normalize_e6_btn.setCheckable(True)
        self.normalize_e6_btn.setIcon(qta.icon("fa5s.magic", color=THEME.text_primary))
        self.normalize_e6_btn.setChecked(conf.e6_normalize)
        self.normalize_e6_btn.setToolTip(
            wrap_tooltip(
                "Normalize: stretch the histogram to the full dynamic range, metered per frame, "
                "and print it through the paper model. This is a rescue tool for <b>faded or "
                "expired slides</b>, where the dyes have lost their range and a per-frame stretch "
                "puts it back. On a slide that was exposed as intended it stretches a range that is "
                "mostly not picture, which reads washed out.<br><br>"
                "Off (the default) renders the slide as the capture — the camera's own color matrix "
                "and a fixed exposure window — so it opens looking like it does in any raw converter, "
                "and a bracketed set stays a bracketed set. Print controls give way to a plain "
                "transfer curve (Density, Grade, Toe, Shoulder)."
            )
        )
        self.layout.addWidget(self.normalize_e6_btn)
        self.layout.addWidget(self.render_ev_slider)

        # Disabled widgets get no hover, so the detail hangs off the hint, not the button.
        self.normalize_merged_hint = hint_label("Not applied to a merged bracket.")
        self.normalize_merged_hint.setToolTip(
            wrap_tooltip(
                "A merge already places the tones: Render exposure picks which exposure it "
                "prints at. Normalize would meter the merged frame and stretch it to full, "
                "which divides that choice straight back out — the anchor would stop doing "
                "anything. The two are not wanted together in any case: Normalize rescues "
                "faded film, and fading compresses the density range a bracket exists to "
                "capture. Unmerge the frame if you need the stretch."
            )
        )
        self.normalize_merged_hint.setVisible(False)
        self.layout.addWidget(self.normalize_merged_hint)

        self.layout.addStretch()

    def _channel_index(self) -> int:
        return max(self.ch_btn_group.checkedId(), 0)

    def _wp_field(self) -> str:
        idx = self._channel_index()
        return "white_point_offset" if idx == 0 else f"white_point_trim_{_CH_SUFFIX[idx - 1]}"

    def _bp_field(self) -> str:
        idx = self._channel_index()
        return "black_point_offset" if idx == 0 else f"black_point_trim_{_CH_SUFFIX[idx - 1]}"

    def _connect_signals(self) -> None:
        self.mode_btn_group.idToggled.connect(lambda i, checked: self._on_mode_changed(_MODES[i][0]) if checked else None)
        self.ch_btn_group.idToggled.connect(lambda _id, checked: self.sync_ui() if checked else None)
        self.autodetect_btn.toggled.connect(lambda c: self.controller.toggle_autodetect(c))
        self.lock_bounds_btn.toggled.connect(self._on_lock_bounds_toggled)

        self.analysis_buffer_slider.valueChanged.connect(lambda v: self._on_buffer_changed(v, persist=False))
        self.analysis_buffer_slider.valueCommitted.connect(lambda v: self._on_buffer_changed(v, persist=True))
        self.analysis_region_btn.toggled.connect(self._on_analysis_region_toggled)
        self.clear_analysis_region_btn.clicked.connect(self.controller.clear_analysis_region)

        self.luma_range_clip_slider.valueChanged.connect(lambda v: self._on_luma_range_clip_changed(v, persist=False))
        self.luma_range_clip_slider.valueCommitted.connect(lambda v: self._on_luma_range_clip_changed(v, persist=True))

        self.color_range_clip_slider.valueChanged.connect(lambda v: self._on_color_range_clip_changed(v, persist=False))
        self.color_range_clip_slider.valueCommitted.connect(lambda v: self._on_color_range_clip_changed(v, persist=True))

        self.white_point_slider.valueChanged.connect(lambda v: self._on_white_point_changed(v, persist=False))
        self.white_point_slider.valueCommitted.connect(lambda v: self._on_white_point_changed(v, persist=True))

        self.black_point_slider.valueChanged.connect(lambda v: self._on_black_point_changed(v, persist=False))
        self.black_point_slider.valueCommitted.connect(lambda v: self._on_black_point_changed(v, persist=True))

        self.normalize_e6_btn.toggled.connect(self._on_normalize_e6_toggled)
        self.sync_ui()

    def _on_white_point_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section("process", persist=persist, **{self._wp_field(): val})

    def _on_black_point_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section("process", persist=persist, **{self._bp_field(): val})

    def _on_lock_bounds_toggled(self, checked: bool) -> None:
        self.update_config_section("process", lock_bounds=checked, persist=True, render=False)
        self.sync_ui()

    def _on_mode_changed(self, mode: str) -> None:
        self.update_config_section(
            "process",
            process_mode=mode,
            render=True,
            persist=True,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.sync_ui()

    def _on_normalize_e6_toggled(self, checked: bool) -> None:
        self.update_config_section(
            "process",
            e6_normalize=checked,
            render=True,
            persist=True,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_analysis_region_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.ANALYSIS_DRAW if checked else ToolMode.NONE)

    def _on_buffer_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            analysis_buffer=val,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.controller.analysis_buffer_preview_requested.emit(val)

    def _on_luma_range_clip_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            luma_range_clip=_luma_range_slider_to_value(val),
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_color_range_clip_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            color_range_clip=_color_slider_to_value(val),
            **invalidate_local_bounds(self.state.config.process),
        )

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            # Exclusive group: checking the current mode unchecks the rest.
            for btn, (mode, *_rest) in zip(self.mode_btns, _MODES):
                if mode == conf.process_mode:
                    btn.setChecked(True)
            self.analysis_buffer_slider.setValue(conf.analysis_buffer)
            self.luma_range_clip_slider.setValue(_luma_range_value_to_slider(conf.luma_range_clip))
            self.color_range_clip_slider.setValue(_color_value_to_slider(conf.color_range_clip))

            # Transparency transfer: the stretch is a fixed window anchored to the decoder's
            # white level, so nothing that tunes a measured stretch has anything to act on.
            from negpy.features.exposure.transfer import is_transparency_transfer

            transfer = is_transparency_transfer(conf.process_mode, conf.e6_normalize)

            # Per-layer WP/BP trims are meaningless on single-emulsion B&W, and the
            # selector goes with the sliders it scopes when those are hidden below.
            is_bw_sel = conf.process_mode == ProcessMode.BW
            hide_channels = is_bw_sel or transfer
            if hide_channels and self._channel_index() != 0:
                self.ch_global_btn.setChecked(True)
            for w in (self.ch_global_btn, self.ch_r_btn, self.ch_g_btn, self.ch_b_btn):
                w.setVisible(not hide_channels)

            idx = self._channel_index()
            suffix = _CH_LABEL[idx]
            self.white_point_slider.label.setText("White Point" + suffix)
            self.black_point_slider.label.setText("Black Point" + suffix)
            if idx == 0:
                self.white_point_slider.setValue(conf.white_point_offset)
                self.black_point_slider.setValue(conf.black_point_offset)
            else:
                ch = _CH_SUFFIX[idx - 1]
                self.white_point_slider.setValue(getattr(conf, f"white_point_trim_{ch}"))
                self.black_point_slider.setValue(getattr(conf, f"black_point_trim_{ch}"))
            for btn, fields in self._channel_buttons:
                btn.edited_dot.set_active(any(getattr(conf, f) != 0.0 for f in fields))

            is_e6 = conf.process_mode == ProcessMode.E6
            # Greyed on a merge, not hidden: the render already ignores it (WorkspaceConfig
            # holds that invariant), and a control that vanishes teaches nothing about why.
            merged = hdr_active(self.state.config.hdr)
            self.normalize_e6_btn.setVisible(is_e6)
            self.normalize_e6_btn.setChecked(conf.e6_normalize)
            self.normalize_e6_btn.setEnabled(not merged)

            # Only a merge has a render exposure to choose, and only the transfer path uses
            # a fixed window for it to mean anything against.
            self.render_ev_slider.setVisible(merged and transfer)
            if merged:
                hdr = self.state.config.hdr
                ev = float(hdr.hdr_anchor_ev)
                if ev >= ANCHOR_EV_UNSET:
                    # Unset means the bracket's middle exposure, which is only 0 EV when it
                    # clamps there. Showing a bare 0.00 would misreport where the picture
                    # actually sits on any bracket spread either side of the reference.
                    scale = output_scale([float(r) for r in hdr.hdr_ratios])
                    ev = float(np.log2(scale)) if scale > 0 else 0.0
                self.render_ev_slider.blockSignals(True)
                self.render_ev_slider.setValue(ev)
                self.render_ev_slider.blockSignals(False)
            self.normalize_merged_hint.setVisible(is_e6 and merged)

            self.lock_bounds_btn.setChecked(conf.lock_bounds)
            self.autodetect_btn.setChecked(self.state.autodetect_enabled)

            has_region = conf.analysis_rect is not None
            self.analysis_region_btn.setChecked(self.state.active_tool == ToolMode.ANALYSIS_DRAW)
            self.analysis_region_btn.edited_dot.set_active(has_region)
            self.clear_analysis_region_btn.setEnabled(has_region)

            for w in (
                self.analysis_buffer_slider,
                self.analysis_region_btn,
                self.clear_analysis_region_btn,
                self.luma_range_clip_slider,
                self.color_range_clip_slider,
                self.lock_bounds_btn,
                self.white_point_slider,
                self.black_point_slider,
            ):
                w.setVisible(not transfer)

            locked = conf.lock_bounds
            # Each clip slider is disabled when its axis rides the roll baseline; the
            # analysis buffer only matters when at least one axis still analyzes locally,
            # and is overridden entirely by a freehand analysis region.
            self.analysis_buffer_slider.setEnabled(not locked and not has_region and not (conf.use_luma_average and conf.use_color_average))
            self.luma_range_clip_slider.setEnabled(not locked and not conf.use_luma_average)
            self.color_range_clip_slider.setEnabled(not locked and not conf.use_color_average)
            # Trims shift the same frozen bounds, so the selector locks with them.
            for w in (self.white_point_slider, self.black_point_slider, self.ch_global_btn, self.ch_r_btn, self.ch_g_btn, self.ch_b_btn):
                w.setEnabled(not locked)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all sliders and buttons.
        """
        widgets = [
            # The group, not just its buttons: QButtonGroup is notified internally, so a
            # blocked button still makes it emit idToggled and re-enter the mode handler.
            self.mode_btn_group,
            *self.mode_btns,
            self.autodetect_btn,
            self.lock_bounds_btn,
            self.ch_global_btn,
            self.ch_r_btn,
            self.ch_g_btn,
            self.ch_b_btn,
            self.analysis_buffer_slider,
            self.analysis_region_btn,
            self.luma_range_clip_slider,
            self.color_range_clip_slider,
            self.white_point_slider,
            self.black_point_slider,
            self.normalize_e6_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
