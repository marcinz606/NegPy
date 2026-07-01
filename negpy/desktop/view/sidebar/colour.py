import qtawesome as qta
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel

from negpy.desktop.session import ToolMode
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider


class ColourSidebar(BaseSidebar):
    """White balance (region CMY + Pick WB) and Cast Removal."""

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.exposure

        # White Balance header row: the Pick WB eyedropper acts on the CMY group below it.
        self.pick_wb_btn = self._icon_toggle(
            "fa5s.eye-dropper",
            False,
            tooltip_with_shortcut("Pick white balance from canvas", "pick_wb"),
        )
        wb_header_row = QHBoxLayout()
        self.wb_label = QLabel("Global White Balance")
        self.wb_label.setStyleSheet(
            f"font-size: {THEME.font_size_base}px; color: {THEME.text_secondary}; font-weight: {THEME.weight_semibold};"
        )
        wb_header_row.addWidget(self.wb_label)
        wb_header_row.addStretch()
        wb_header_row.addWidget(self.pick_wb_btn)
        self.layout.addLayout(wb_header_row)

        # Region selector as an icon column: one exclusive toggle to the left of each CMY
        # slider. The region applies to all three sliders — the row alignment is visual.
        self.region_global_btn = self._icon_toggle("fa5s.globe", True, "Global — apply CMY white balance to the entire tonal range")
        self.region_shadow_btn = self._icon_toggle("fa5s.moon", False, "Shadows — bias CMY white balance toward shadow (low-density) areas")
        self.region_highlight_btn = self._icon_toggle(
            "fa5s.sun", False, "Highlights — bias CMY white balance toward highlight (high-density) areas"
        )
        self.region_btn_group = QButtonGroup(self)
        self.region_btn_group.setExclusive(True)
        self.region_btn_group.addButton(self.region_global_btn, 0)
        self.region_btn_group.addButton(self.region_shadow_btn, 1)
        self.region_btn_group.addButton(self.region_highlight_btn, 2)
        # (button, icon, region CMY fields) — icon tints edited-yellow when any field is set.
        self._region_icons = (
            (self.region_global_btn, "fa5s.globe", ("wb_cyan", "wb_magenta", "wb_yellow")),
            (self.region_shadow_btn, "fa5s.moon", ("shadow_cyan", "shadow_magenta", "shadow_yellow")),
            (self.region_highlight_btn, "fa5s.sun", ("highlight_cyan", "highlight_magenta", "highlight_yellow")),
        )

        self.cyan_slider = CompactSlider("Cyan", -1.0, 1.0, conf.wb_cyan, has_neutral=True)
        self.cyan_slider.slider.setObjectName("cyan_slider")
        self.cyan_slider.setToolTip("Cyan–Red white balance shift; applies to the selected region (Global/Shadows/Highlights)")
        self.magenta_slider = CompactSlider("Magenta", -1.0, 1.0, conf.wb_magenta, has_neutral=True)
        self.magenta_slider.slider.setObjectName("magenta_slider")
        self.magenta_slider.setToolTip(
            tooltip_with_shortcut("Magenta–Green white balance shift; applies to the selected region  E/D", None)
        )
        self.yellow_slider = CompactSlider("Yellow", -1.0, 1.0, conf.wb_yellow, has_neutral=True)
        self.yellow_slider.slider.setObjectName("yellow_slider")
        self.yellow_slider.setToolTip(tooltip_with_shortcut("Yellow–Blue white balance shift; applies to the selected region  R/F", None))
        for region_btn, slider in (
            (self.region_global_btn, self.cyan_slider),
            (self.region_shadow_btn, self.magenta_slider),
            (self.region_highlight_btn, self.yellow_slider),
        ):
            row = QHBoxLayout()
            row.addWidget(region_btn)
            row.addWidget(slider)
            self.layout.addLayout(row)

        self.cast_removal_slider = CompactSlider("Cast Removal", 0.0, 1.0, conf.cast_removal_strength)
        self.cast_removal_slider.setToolTip(
            "Cast Removal: neutralizes the colour cast a negative leaves in the print — balances each "
            "colour layer so greys stay neutral from deep shadows through highlights (C-41). 0 = off, "
            "1 = full."
        )
        self.auto_cast_btn = self._icon_toggle(
            "fa5s.palette",
            conf.auto_cast_removal,
            "Auto Cast Removal: bias the strength by the frame's own neutral references — clean greys "
            "get full correction, scenes with few true neutrals get a gentler touch to avoid over-correcting. "
            "The slider still trims on top.",
        )
        cast_row = QHBoxLayout()
        cast_row.addWidget(self.auto_cast_btn)
        cast_row.addWidget(self.cast_removal_slider)
        self.layout.addLayout(cast_row)

        self.layout.addStretch()

    def _region_index(self) -> int:
        return self.region_btn_group.checkedId()

    def _connect_signals(self) -> None:
        self.region_btn_group.idToggled.connect(lambda _id, checked: self.sync_ui() if checked else None)

        self.cyan_slider.valueChanged.connect(self._on_cyan_changed)
        self.magenta_slider.valueChanged.connect(self._on_magenta_changed)
        self.yellow_slider.valueChanged.connect(self._on_yellow_changed)
        self.cyan_slider.valueCommitted.connect(lambda v: self._on_cyan_changed(v, persist=True))
        self.magenta_slider.valueCommitted.connect(lambda v: self._on_magenta_changed(v, persist=True))
        self.yellow_slider.valueCommitted.connect(lambda v: self._on_yellow_changed(v, persist=True))

        self.pick_wb_btn.toggled.connect(self._on_pick_wb_toggled)
        self.cast_removal_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, cast_removal_strength=v)
        )
        self.cast_removal_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, cast_removal_strength=v)
        )
        self.auto_cast_btn.toggled.connect(
            lambda checked: self.update_config_section(
                "exposure", render=True, persist=True, readback_metrics=True, auto_cast_removal=checked
            )
        )

    def _on_cyan_changed(self, v: float, persist: bool = False) -> None:
        field = ("wb_cyan", "shadow_cyan", "highlight_cyan")[self._region_index()]
        self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, **{field: v})

    def _on_magenta_changed(self, v: float, persist: bool = False) -> None:
        field = ("wb_magenta", "shadow_magenta", "highlight_magenta")[self._region_index()]
        self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, **{field: v})

    def _on_yellow_changed(self, v: float, persist: bool = False) -> None:
        field = ("wb_yellow", "shadow_yellow", "highlight_yellow")[self._region_index()]
        self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, **{field: v})

    def _on_pick_wb_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.WB_PICK if checked else ToolMode.NONE)

    def sync_ui(self) -> None:
        conf = self.state.config.exposure
        self.block_signals(True)
        try:
            idx = self._region_index()
            self.wb_label.setText(("Global", "Shadows", "Highlights")[idx] + " White Balance")
            channels = (
                ("wb_cyan", "wb_magenta", "wb_yellow"),
                ("shadow_cyan", "shadow_magenta", "shadow_yellow"),
                ("highlight_cyan", "highlight_magenta", "highlight_yellow"),
            )[idx]
            self.cyan_slider.setValue(getattr(conf, channels[0]))
            self.magenta_slider.setValue(getattr(conf, channels[1]))
            self.yellow_slider.setValue(getattr(conf, channels[2]))

            for btn, icon_name, fields in self._region_icons:
                edited = any(getattr(conf, f) != 0.0 for f in fields)
                color = THEME.accent_edited if edited else THEME.text_primary
                btn.setIcon(qta.icon(icon_name, color=color))

            self.pick_wb_btn.setChecked(self.state.active_tool == ToolMode.WB_PICK)
            self.cast_removal_slider.setValue(conf.cast_removal_strength)
            self.auto_cast_btn.setChecked(conf.auto_cast_removal)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.region_global_btn,
            self.region_shadow_btn,
            self.region_highlight_btn,
            self.cyan_slider,
            self.magenta_slider,
            self.yellow_slider,
            self.pick_wb_btn,
            self.cast_removal_slider,
            self.auto_cast_btn,
        ):
            w.blockSignals(blocked)
