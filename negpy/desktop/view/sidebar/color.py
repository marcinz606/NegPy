from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.session import ToolMode
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, wrap_tooltip
from negpy.desktop.view.widgets.choice_button import ChoiceButton
from negpy.desktop.view.widgets.sliders import CompactSlider, KelvinSlider
from negpy.features.exposure.logic import kelvin_to_wb, wb_to_kelvin


class ColorSidebar(BaseSidebar):
    """White balance: region CMY, Temperature and Pick WB."""

    def _init_ui(self) -> None:
        conf = self.state.config.exposure

        self.region_btn = ChoiceButton(
            (("fa5s.globe", "Global"), ("fa5s.moon", "Shadows"), ("fa5s.sun", "Highlights")),
            "Region for Temperature, CMY and Pick WB: Global applies to the entire tonal range; "
            "Shadows and Highlights bias toward shadow (low-density) or highlight (high-density) areas",
        )
        self.pick_wb_btn = self._tool_toggle(
            "fa5s.eye-dropper",
            "",
            tooltip_with_shortcut(
                "Pick a neutral gray from the canvas — solves the selected region's CMY so the patch prints neutral",
                "pick_wb",
            ),
        )
        self.ring_btn = self._tool_toggle("mdi.target", "", self._ring_tooltip())
        self.pick_wb_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.ring_btn.setFixedWidth(ICON_BUTTON_WIDTH)

        region_row = QHBoxLayout()
        region_row.addWidget(self.region_btn, 1)
        region_row.addWidget(self.pick_wb_btn)
        region_row.addWidget(self.ring_btn)
        self.layout.addLayout(region_row)

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

        self.layout.addStretch()

    _REGION_FIELDS = (
        ("wb_cyan", "wb_magenta", "wb_yellow"),
        ("shadow_cyan", "shadow_magenta", "shadow_yellow"),
        ("highlight_cyan", "highlight_magenta", "highlight_yellow"),
    )
    _REGION_MY = (
        ("wb_magenta", "wb_yellow"),
        ("shadow_magenta", "shadow_yellow"),
        ("highlight_magenta", "highlight_yellow"),
    )

    def _region_index(self) -> int:
        return self.region_btn.currentIndex()

    def _region_my(self, conf) -> tuple:
        m_field, y_field = self._REGION_MY[self._region_index()]
        return getattr(conf, m_field), getattr(conf, y_field)

    def _connect_signals(self) -> None:
        self.region_btn.currentChanged.connect(lambda _i: self.sync_ui())

        self.temp_slider.dragStarted.connect(self._on_temp_drag_started)
        self.temp_slider.dragEnded.connect(self._on_temp_drag_ended)
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        self.temp_slider.valueCommitted.connect(lambda v: self._on_temp_changed(v, persist=True))

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

    @staticmethod
    def _ring_tooltip(printing: bool = False) -> str:
        if printing:
            return "Printing the color ring-around…"
        return tooltip_with_shortcut(
            "Color Ring-Around: print the frame as a 5×5 mosaic — the center patch neutral, the "
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
        # Anchor (M, Y) for the whole drag: re-projecting an already-clipped pair on every tick
        # would corrupt the tint component.
        self._temp_anchor = self._region_my(self.state.config.exposure)

    def _on_temp_drag_ended(self) -> None:
        self._temp_anchor = None

    def _on_temp_changed(self, kelvin: float, persist: bool = False) -> None:
        m0, y0 = self._temp_anchor or self._region_my(self.state.config.exposure)
        m2, y2 = kelvin_to_wb(kelvin, m0, y0)
        m_field, y_field = self._REGION_MY[self._region_index()]
        self.update_config_section("exposure", render=True, persist=persist, readback_metrics=persist, **{m_field: m2, y_field: y2})

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
            channels = self._REGION_FIELDS[idx]
            self.cyan_slider.setValue(getattr(conf, channels[0]))
            self.magenta_slider.setValue(getattr(conf, channels[1]))
            self.yellow_slider.setValue(getattr(conf, channels[2]))
            self.temp_slider.setValue(wb_to_kelvin(*self._region_my(conf)))

            for i, fields in enumerate(self._REGION_FIELDS):
                self.region_btn.set_edited(i, any(getattr(conf, f) != 0.0 for f in fields))

            self.pick_wb_btn.setChecked(self.state.active_tool == ToolMode.WB_PICK)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.region_btn,
            self.temp_slider,
            self.cyan_slider,
            self.magenta_slider,
            self.yellow_slider,
            self.pick_wb_btn,
            self.ring_btn,
        ):
            w.blockSignals(blocked)
