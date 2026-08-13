from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout

from negpy.desktop.session import ToolMode
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import wrap_tooltip
from negpy.desktop.view.widgets.sliders import CompactSlider, KelvinSlider
from negpy.features.exposure.logic import kelvin_to_wb, wb_to_kelvin


class ColorSidebar(BaseSidebar):
    """White balance (region CMY + Pick WB) and Cast Removal."""

    def _init_ui(self) -> None:
        conf = self.state.config.exposure

        # Region selector, same idiom as the tone page's channel selector.
        self.region_global_btn = self._labeled_toggle(
            "fa5s.globe", " Global", True, "Global — apply temperature and CMY white balance to the entire tonal range"
        )
        self.region_shadow_btn = self._labeled_toggle(
            "fa5s.moon", " Shadows", False, "Shadows — bias temperature and CMY white balance toward shadow (low-density) areas"
        )
        self.region_highlight_btn = self._labeled_toggle(
            "fa5s.sun", " Highlights", False, "Highlights — bias temperature and CMY white balance toward highlight (high-density) areas"
        )
        self.region_btn_group = QButtonGroup(self)
        self.region_btn_group.setExclusive(True)
        self.region_btn_group.addButton(self.region_global_btn, 0)
        self.region_btn_group.addButton(self.region_shadow_btn, 1)
        self.region_btn_group.addButton(self.region_highlight_btn, 2)
        # (button, region CMY fields) — the edited dot shows when any field is set.
        self._region_buttons = (
            (self.region_global_btn, ("wb_cyan", "wb_magenta", "wb_yellow")),
            (self.region_shadow_btn, ("shadow_cyan", "shadow_magenta", "shadow_yellow")),
            (self.region_highlight_btn, ("highlight_cyan", "highlight_magenta", "highlight_yellow")),
        )
        region_row = QHBoxLayout()
        for btn in (self.region_global_btn, self.region_shadow_btn, self.region_highlight_btn):
            region_row.addWidget(btn, 1)
        self.layout.addLayout(region_row)

        self.pick_wb_btn = self._tool_toggle(
            "fa5s.eye-dropper",
            "Pick WB",
            tooltip_with_shortcut(
                "Pick a neutral grey from the canvas — solves the selected region's CMY so the patch prints neutral",
                "pick_wb",
            ),
        )
        self.temp_lock_btn = self._small_toggle(
            "fa5s.thermometer-half",
            "Roll Lock",
            self.controller.session.repo.get_global_setting("wb_temp_lock") is not None,
            "Roll lock — every newly opened frame re-aims this region's temperature to the target "
            "(its own tint preserved); committing the slider while locked updates the target. "
            "Each region (Global/Shadows/Highlights) holds its own lock.",
        )
        self.region_reset_btn = self._icon_action(
            "fa5s.undo",
            "Reset the selected region's white balance — Temperature and Cyan/Magenta/Yellow back to neutral",
        )
        self.ring_btn = self._tool_toggle("mdi.target", "", self._ring_tooltip())
        self.ring_btn.setFixedWidth(36)

        tools_row = QHBoxLayout()
        tools_row.addWidget(self.pick_wb_btn, 1)
        tools_row.addWidget(self.temp_lock_btn, 1)
        tools_row.addWidget(self.region_reset_btn)
        tools_row.addWidget(self.ring_btn)
        self.layout.addLayout(tools_row)

        # Temperature lever over the selected region's M/Y pair (real darkroom: cyan stays 0).
        self.temp_slider = KelvinSlider("Temperature")
        self.temp_slider.setValue(wb_to_kelvin(conf.wb_magenta, conf.wb_yellow))
        self._temp_anchor = None
        self.layout.addWidget(self.temp_slider)

        self.cyan_slider = CompactSlider("Cyan", -1.0, 1.0, conf.wb_cyan, has_neutral=True)
        self.cyan_slider.slider.setObjectName("cyan_slider")
        self.magenta_slider = CompactSlider("Magenta", -1.0, 1.0, conf.wb_magenta, has_neutral=True)
        self.magenta_slider.slider.setObjectName("magenta_slider")
        self.yellow_slider = CompactSlider("Yellow", -1.0, 1.0, conf.wb_yellow, has_neutral=True)
        self.yellow_slider.slider.setObjectName("yellow_slider")
        for slider in (self.cyan_slider, self.magenta_slider, self.yellow_slider):
            self.layout.addWidget(slider)

        self.cast_removal_slider = CompactSlider("Cast Removal", 0.0, 1.0, conf.cast_removal_strength)
        self.cast_removal_slider.setToolTip(
            "Cast Removal: neutralizes the color cast a negative leaves in the print — balances each "
            "color layer so greys stay neutral from deep shadows through highlights. 0 = off, 1 = full."
            "<br><br>Color Negative only, and hidden elsewhere: it defeats the orange mask, a manufactured cast "
            "that is not part of the picture. A slide has no mask — its cast IS the photograph — so "
            "solving for a neutral axis there would remove the light you shot in. For a slide use "
            "Temperature and the CMY sliders, or Hue Trim for an odd light source."
        )
        self.layout.addWidget(self.cast_removal_slider)

        self.layout.addStretch()

    _REGION_MY = (
        ("wb_magenta", "wb_yellow"),
        ("shadow_magenta", "shadow_yellow"),
        ("highlight_magenta", "highlight_yellow"),
    )
    _LOCK_KEYS = ("wb_temp_lock", "wb_temp_lock_shadow", "wb_temp_lock_highlight")

    def _region_index(self) -> int:
        return self.region_btn_group.checkedId()

    def _region_my(self, conf) -> tuple:
        m_field, y_field = self._REGION_MY[self._region_index()]
        return getattr(conf, m_field), getattr(conf, y_field)

    def _on_region_reset(self) -> None:
        fields = self._region_buttons[self._region_index()][1]
        self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, **{f: 0.0 for f in fields})

    def _connect_signals(self) -> None:
        self.region_btn_group.idToggled.connect(lambda _id, checked: self.sync_ui() if checked else None)
        self.region_reset_btn.clicked.connect(self._on_region_reset)

        self.temp_slider.dragStarted.connect(self._on_temp_drag_started)
        self.temp_slider.dragEnded.connect(self._on_temp_drag_ended)
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        self.temp_slider.valueCommitted.connect(lambda v: self._on_temp_changed(v, persist=True))
        self.temp_lock_btn.toggled.connect(self._on_temp_lock_toggled)

        self.cyan_slider.valueChanged.connect(self._on_cyan_changed)
        self.magenta_slider.valueChanged.connect(self._on_magenta_changed)
        self.yellow_slider.valueChanged.connect(self._on_yellow_changed)
        self.cyan_slider.valueCommitted.connect(lambda v: self._on_cyan_changed(v, persist=True))
        self.magenta_slider.valueCommitted.connect(lambda v: self._on_magenta_changed(v, persist=True))
        self.yellow_slider.valueCommitted.connect(lambda v: self._on_yellow_changed(v, persist=True))

        self.pick_wb_btn.toggled.connect(self._on_pick_wb_toggled)
        self.ring_btn.clicked.connect(lambda checked: self.controller.toggle_ring_around(force=checked))
        # Session state, not config, so the button follows the controller rather than sync_ui.
        self.controller.test_strip_changed.connect(self._sync_ring_btn)
        self.cast_removal_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=False, readback_metrics=False, cast_removal_strength=v)
        )
        self.cast_removal_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", render=True, persist=True, readback_metrics=True, cast_removal_strength=v)
        )

    @staticmethod
    def _ring_tooltip(printing: bool = False) -> str:
        if printing:
            return "Printing the color ring-around…"
        return tooltip_with_shortcut(
            "Color Ring-Around: print the frame as a 5×5 mosaic — the centre patch neutral, the "
            "ring stepping 2cc at a time out to ±4cc on the magenta and yellow axes. Click the patch "
            "that looks neutral to keep its filtration. The 90° rotate controls turn the ladder while "
            "it is up. With Cast Removal on the patches separate less, since it corrects toward "
            "neutral underneath.",
            "toggle_ring_around",
        )

    def _sync_ring_btn(self, _up: bool) -> None:
        # One shared slot, so the kind gates this or the tone strip lights this button too.
        mine = self.state.test_strip_kind == "color"
        pending = self.state.test_strip_pending and mine
        self.ring_btn.setChecked((self.state.test_strip or self.state.test_strip_pending) and mine)
        self.ring_btn.setEnabled(not pending)
        self.ring_btn.setToolTip(wrap_tooltip(self._ring_tooltip(printing=pending)))

    def _on_temp_drag_started(self) -> None:
        # Anchor (M, Y) for the whole drag: re-projecting an already-clipped
        # pair on every tick would corrupt the tint component.
        self._temp_anchor = self._region_my(self.state.config.exposure)

    def _on_temp_drag_ended(self) -> None:
        self._temp_anchor = None

    def _on_temp_changed(self, kelvin: float, persist: bool = False) -> None:
        m0, y0 = self._temp_anchor or self._region_my(self.state.config.exposure)
        m2, y2 = kelvin_to_wb(kelvin, m0, y0)
        m_field, y_field = self._REGION_MY[self._region_index()]
        self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, **{m_field: m2, y_field: y2})
        if persist and self.temp_lock_btn.isChecked():
            # Store the achieved temperature (post-clip), not the requested one.
            self.controller.session.repo.save_global_setting(self._LOCK_KEYS[self._region_index()], wb_to_kelvin(m2, y2))

    def _on_temp_lock_toggled(self, checked: bool) -> None:
        key = self._LOCK_KEYS[self._region_index()]
        self.controller.session.repo.save_global_setting(key, float(self.temp_slider.value()) if checked else None)

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
            self.state.wb_pick_region = idx
            channels = (
                ("wb_cyan", "wb_magenta", "wb_yellow"),
                ("shadow_cyan", "shadow_magenta", "shadow_yellow"),
                ("highlight_cyan", "highlight_magenta", "highlight_yellow"),
            )[idx]
            self.cyan_slider.setValue(getattr(conf, channels[0]))
            self.magenta_slider.setValue(getattr(conf, channels[1]))
            self.yellow_slider.setValue(getattr(conf, channels[2]))
            self.temp_slider.setValue(wb_to_kelvin(*self._region_my(conf)))
            locked = self.controller.session.repo.get_global_setting(self._LOCK_KEYS[idx]) is not None
            self.temp_lock_btn.setChecked(locked)

            for btn, fields in self._region_buttons:
                btn.edited_dot.set_active(any(getattr(conf, f) != 0.0 for f in fields))

            self.pick_wb_btn.setChecked(self.state.active_tool == ToolMode.WB_PICK)
            self.cast_removal_slider.setValue(conf.cast_removal_strength)
            # C-41 only, in the render as well as here: the solve needs the shadow and
            # neutral-axis refs, and both meters are gated to C-41 (processor.py, and
            # `needs_refs` in gpu_engine.py). Everywhere else the slider moved a value that
            # never reached the arithmetic. Safe to hide rather than merely disable — it is
            # already inert, so nothing is left running invisibly.
            from negpy.features.process.models import ProcessMode

            self.cast_removal_slider.setVisible(self.state.config.process.process_mode == ProcessMode.C41)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.region_global_btn,
            self.region_shadow_btn,
            self.region_highlight_btn,
            self.temp_slider,
            self.temp_lock_btn,
            self.cyan_slider,
            self.magenta_slider,
            self.yellow_slider,
            self.pick_wb_btn,
            self.ring_btn,
            self.cast_removal_slider,
        ):
            w.blockSignals(blocked)
