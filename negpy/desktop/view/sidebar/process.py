import math

import qtawesome as qta
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.process.models import ProcessMode, invalidate_local_bounds
from negpy.services.assets.crosstalk import CrosstalkProfiles

# Luma Range Clip slider mapping: positions 0..100 clip the histogram tails; negative
# positions -100..0 map to an outward log-density margin (gentler-than-zero stretch).
_LUMA_MARGIN_MIN = 1e-6
_LUMA_MARGIN_MAX = 1.0

# Colour Clip slider: the absolute per-channel-balance clip percentile, log-interpolated
# around the neutral (pos 0 = base_color_clip). The ends reach _COLOR_CLIP_MIN (gentlest,
# near-extreme bounds) and _COLOR_CLIP_MAX (tightest channel balance).
_COLOR_CLIP_NEUTRAL = float(EXPOSURE_CONSTANTS["base_color_clip"])
_COLOR_CLIP_MIN = 1e-6
_COLOR_CLIP_MAX = 5.0


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
        self.layout.setSpacing(12)
        conf = self.state.config.process

        mode_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([m.value for m in ProcessMode])
        self.mode_combo.setCurrentText(conf.process_mode)
        self.mode_combo.setToolTip("Film process mode: C41 (colour negative), B&W (panchromatic), E-6 (slide/reversal)")
        self.lock_bounds_btn = QPushButton()
        self.lock_bounds_btn.setCheckable(True)
        self.lock_bounds_btn.setIcon(qta.icon("fa5s.lock", color=THEME.text_primary))
        self.lock_bounds_btn.setToolTip("Freeze normalization bounds — crop and analysis sliders no longer re-analyze")
        self.lock_bounds_btn.setFixedWidth(28)
        self.autodetect_btn = QPushButton()
        self.autodetect_btn.setCheckable(True)
        self.autodetect_btn.setIcon(qta.icon("mdi6.auto-fix", color=THEME.text_primary))
        self.autodetect_btn.setToolTip("Auto-detect film process (C41/B&W/E-6) on load")
        self.autodetect_btn.setFixedWidth(28)
        mode_row.addWidget(self.mode_combo, stretch=1)
        mode_row.addWidget(self.autodetect_btn)
        mode_row.addWidget(self.lock_bounds_btn)
        self.layout.addLayout(mode_row)

        self.linear_raw_btn = QPushButton(" Linear RAW")
        self.linear_raw_btn.setCheckable(True)
        self.linear_raw_btn.setChecked(conf.linear_raw)
        self.linear_raw_btn.setIcon(qta.icon("fa5s.sliders-h", color=THEME.text_primary))
        self.linear_raw_btn.setToolTip(
            "Decode RAW with neutral multipliers (1,1,1,1) — bypasses as-shot camera white balance for a clean starting point"
        )
        self.layout.addWidget(self.linear_raw_btn)

        buf_row = QHBoxLayout()
        self.analysis_buffer_slider = CompactSlider("Analysis Buffer", 0.0, 0.25, conf.analysis_buffer)
        self.analysis_buffer_slider.setToolTip(
            "Crops the analysis region inward to exclude film borders and rebate from exposure calculations"
        )
        buf_row.addWidget(self.analysis_buffer_slider)
        self.layout.addLayout(buf_row)

        clip_row = QHBoxLayout()
        initial_luma_slider_val = _luma_range_value_to_slider(conf.luma_range_clip)
        self.luma_range_clip_slider = CompactSlider(
            "Luma Range Clip", -100, 100, initial_luma_slider_val, precision=1, step=1, has_neutral=True
        )
        self.luma_range_clip_slider.setToolTip(
            "Tonal-range normalization (black/white-point span). Neutral already applies a small robust clip. "
            "Positive: clips the top/bottom for more aggressive highlight/shadow recovery. "
            "Negative: outward headroom — leaves lifted blacks / unclipped highlights for a gentler stretch (double-click to reset)"
        )
        initial_color_slider_val = _color_value_to_slider(conf.color_range_clip)
        self.color_range_clip_slider = CompactSlider(
            "Colour Clip", -100, 100, initial_color_slider_val, precision=1, step=1, has_neutral=True
        )
        self.color_range_clip_slider.setToolTip(
            "Per-channel colour-balance clip percentile (orange-mask cast removal), independent of tonal range. "
            "Neutral: P1 clip (1st percentile). "
            "Negative: gentler, samples nearer the extremes. Positive: tighter channel balance (double-click to reset)"
        )
        clip_row.addWidget(self.luma_range_clip_slider)
        clip_row.addWidget(self.color_range_clip_slider)
        self.layout.addLayout(clip_row)

        wp_bp_row = QHBoxLayout()
        self.white_point_slider = CompactSlider("White Point", -0.25, 0.25, conf.white_point_offset, has_neutral=True)
        self.black_point_slider = CompactSlider("Black Point", -0.25, 0.25, conf.black_point_offset, has_neutral=True)
        wp_bp_row.addWidget(self.white_point_slider)
        wp_bp_row.addWidget(self.black_point_slider)
        self.layout.addLayout(wp_bp_row)

        self.layout.addWidget(section_subheader("CROSSTALK"))

        matrix_row = QHBoxLayout()
        self.crosstalk_label = field_label("Matrix")
        self.crosstalk_combo = QComboBox()
        self.crosstalk_combo.addItems(CrosstalkProfiles.list_profiles())
        self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
        self.crosstalk_combo.setToolTip(
            "Spectral crosstalk (dye unmix): applies the film's crosstalk matrix to the raw NEGATIVE "
            "densities before analysis and inversion — the physically correct domain (the matrices are "
            "derived from negative dye-density curves). 'Default' is built-in; drop custom .toml matrices "
            "in the NegPy/crosstalk folder (see docs/CROSSTALK.md). Re-run Batch Analysis after changing this"
        )
        matrix_row.addWidget(self.crosstalk_label)
        matrix_row.addWidget(self.crosstalk_combo, 1)
        self.layout.addLayout(matrix_row)

        self.crosstalk_strength_slider = CompactSlider("Separation", 0.0, 1.0, conf.crosstalk_strength, has_neutral=True)
        self.crosstalk_strength_slider.setToolTip(
            "Strength of the spectral-crosstalk unmix (0 = off): richer colour separation, physically applied to the negative"
        )
        self.layout.addWidget(self.crosstalk_strength_slider)

        self.normalize_e6_btn = QPushButton(" Normalize")
        self.normalize_e6_btn.setCheckable(True)
        self.normalize_e6_btn.setIcon(qta.icon("fa5s.magic", color=THEME.text_primary))
        self.normalize_e6_btn.setChecked(conf.e6_normalize)
        self.normalize_e6_btn.setToolTip("Automatically stretch the histogram to full dynamic range")
        self.layout.addWidget(self.normalize_e6_btn)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.autodetect_btn.toggled.connect(lambda c: self.controller.toggle_autodetect(c))
        self.lock_bounds_btn.toggled.connect(self._on_lock_bounds_toggled)
        self.linear_raw_btn.toggled.connect(self._on_linear_raw_toggled)

        self.analysis_buffer_slider.valueChanged.connect(lambda v: self._on_buffer_changed(v, persist=False))
        self.analysis_buffer_slider.valueCommitted.connect(lambda v: self._on_buffer_changed(v, persist=True))

        self.luma_range_clip_slider.valueChanged.connect(lambda v: self._on_luma_range_clip_changed(v, persist=False))
        self.luma_range_clip_slider.valueCommitted.connect(lambda v: self._on_luma_range_clip_changed(v, persist=True))

        self.color_range_clip_slider.valueChanged.connect(lambda v: self._on_color_range_clip_changed(v, persist=False))
        self.color_range_clip_slider.valueCommitted.connect(lambda v: self._on_color_range_clip_changed(v, persist=True))

        self.white_point_slider.valueChanged.connect(lambda v: self._on_white_point_changed(v, persist=False))
        self.white_point_slider.valueCommitted.connect(lambda v: self._on_white_point_changed(v, persist=True))

        self.black_point_slider.valueChanged.connect(lambda v: self._on_black_point_changed(v, persist=False))
        self.black_point_slider.valueCommitted.connect(lambda v: self._on_black_point_changed(v, persist=True))

        self.crosstalk_combo.currentTextChanged.connect(self._on_crosstalk_profile_changed)
        self.crosstalk_strength_slider.valueChanged.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=False))
        self.crosstalk_strength_slider.valueCommitted.connect(lambda v: self._on_crosstalk_strength_changed(v, persist=True))
        self.normalize_e6_btn.toggled.connect(self._on_normalize_e6_toggled)
        self.sync_ui()

    def _on_white_point_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section("process", white_point_offset=val, persist=persist)

    def _on_black_point_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section("process", black_point_offset=val, persist=persist)

    def _on_lock_bounds_toggled(self, checked: bool) -> None:
        self.update_config_section("process", lock_bounds=checked, persist=True, render=False)
        self.sync_ui()

    def _on_linear_raw_toggled(self, checked: bool) -> None:
        from dataclasses import replace

        new_config = replace(
            self.state.config,
            process=replace(
                self.state.config.process,
                linear_raw=checked,
                **invalidate_local_bounds(self.state.config.process),
            ),
        )
        # render=False: don't analyse bounds on stale (pre-reload) raw data
        self.controller.session.update_config(new_config, persist=True, render=False)
        if self.state.current_file_path:
            self.controller.load_file(self.state.current_file_path)

    def _on_mode_changed(self, mode: str) -> None:
        self.update_config_section(
            "process",
            process_mode=mode,
            render=True,
            persist=True,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.sync_ui()

    def _on_crosstalk_profile_changed(self, name: str) -> None:
        # Bake the matrix into the config so saved edits stay reproducible if the
        # profile file is later moved/deleted. The persisted per-frame bounds were
        # analyzed under the previous matrix — clear them so the stretch re-derives
        # from the unmixed data (otherwise the mask redistribution leaks through).
        matrix = CrosstalkProfiles.get_matrix(name)
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            crosstalk_profile=name,
            crosstalk_matrix=matrix,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_crosstalk_strength_changed(self, val: float, persist: bool = True) -> None:
        self.update_config_section(
            "process",
            persist=persist,
            render=True,
            crosstalk_strength=val,
            **invalidate_local_bounds(self.state.config.process),
        )

    def _on_normalize_e6_toggled(self, checked: bool) -> None:
        self.update_config_section(
            "process",
            e6_normalize=checked,
            render=True,
            persist=True,
            **invalidate_local_bounds(self.state.config.process),
        )

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
            self.mode_combo.setCurrentText(conf.process_mode)
            self.analysis_buffer_slider.setValue(conf.analysis_buffer)
            self.luma_range_clip_slider.setValue(_luma_range_value_to_slider(conf.luma_range_clip))
            self.color_range_clip_slider.setValue(_color_value_to_slider(conf.color_range_clip))
            self.white_point_slider.setValue(conf.white_point_offset)
            self.black_point_slider.setValue(conf.black_point_offset)

            is_e6 = conf.process_mode == ProcessMode.E6
            self.normalize_e6_btn.setVisible(is_e6)
            self.normalize_e6_btn.setChecked(conf.e6_normalize)

            self.lock_bounds_btn.setChecked(conf.lock_bounds)
            self.linear_raw_btn.setChecked(conf.linear_raw)

            profiles = CrosstalkProfiles.list_profiles()
            if profiles != [self.crosstalk_combo.itemText(i) for i in range(self.crosstalk_combo.count())]:
                self.crosstalk_combo.clear()
                self.crosstalk_combo.addItems(profiles)
            self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
            self.crosstalk_strength_slider.setValue(conf.crosstalk_strength)
            is_bw = conf.process_mode == ProcessMode.BW
            self.crosstalk_label.setVisible(not is_bw)
            self.crosstalk_combo.setVisible(not is_bw)
            self.crosstalk_strength_slider.setVisible(not is_bw)
            self.autodetect_btn.setChecked(self.state.autodetect_enabled)

            locked = conf.lock_bounds
            # Each clip slider is disabled when its axis rides the roll baseline; the
            # analysis buffer only matters when at least one axis still analyzes locally.
            self.analysis_buffer_slider.setEnabled(not locked and not (conf.use_luma_average and conf.use_colour_average))
            self.luma_range_clip_slider.setEnabled(not locked and not conf.use_luma_average)
            self.color_range_clip_slider.setEnabled(not locked and not conf.use_colour_average)
            for w in (self.white_point_slider, self.black_point_slider):
                w.setEnabled(not locked)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all sliders and buttons.
        """
        widgets = [
            self.mode_combo,
            self.autodetect_btn,
            self.lock_bounds_btn,
            self.linear_raw_btn,
            self.analysis_buffer_slider,
            self.luma_range_clip_slider,
            self.color_range_clip_slider,
            self.white_point_slider,
            self.black_point_slider,
            self.crosstalk_combo,
            self.crosstalk_strength_slider,
            self.normalize_e6_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
