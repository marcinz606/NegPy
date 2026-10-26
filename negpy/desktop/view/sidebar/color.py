from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from negpy.desktop.session import ToolMode
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, RING_AROUND_SWEEP, field_row, gradient_icon, wrap_tooltip
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.widgets.sliders import CompactSlider, KelvinSlider
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.features.exposure.logic import kelvin_to_wb, wb_to_kelvin
from negpy.features.exposure.models import BwFilter
from negpy.features.process.models import ProcessMode


class ColorSidebar(BaseSidebar):
    """White balance: region CMY, Temperature and Pick WB, and Cast Removal over the whole frame. A B&W print
    from a color negative shows only the contrast Filter."""

    def _init_ui(self) -> None:
        conf = self.state.config.exposure

        # Flat, in the order the arrows darken a blue sky: None, then yellow to red, then the greens.
        self.filter_combo = StepCombo.of(
            [
                tr("None"),
                tr("Yellow #8"),
                tr("Deep Yellow #15"),
                tr("Orange #21"),
                tr("Red #25"),
                tr("Yellow-Green #11"),
                tr("Green #58"),
            ],
            tr("Filter: a colored filter, as on the camera lens. It darkens the colors opposite to its own."),
            list(BwFilter),
        )
        self.filter_block = QWidget()
        self.filter_block.setObjectName("collapsible_content_body")
        filter_col = QVBoxLayout(self.filter_block)
        filter_col.setContentsMargins(0, 0, 0, 0)
        filter_col.addLayout(field_row(tr("Filter"), self.filter_combo))
        self.layout.addWidget(self.filter_block)

        self.color_block = QWidget()
        self.color_block.setObjectName("collapsible_content_body")
        color_col = QVBoxLayout(self.color_block)
        color_col.setContentsMargins(0, 0, 0, 0)

        self.region_btn = SegmentedChoice(
            (("fa5s.globe", tr("Global")), ("fa5s.moon", tr("Shadows")), ("fa5s.sun", tr("Highlights"))),
            tr("Apply the color controls to the whole picture, or mostly to its dark or bright parts."),
        )
        self.pick_wb_btn = self._tool_toggle(
            "fa5s.eye-dropper",
            "",
            tooltip_with_shortcut(
                tr("Click something in the picture that should be gray, and the color is corrected to make it gray."),
                "pick_wb",
            ),
        )
        self.ring_btn = self._tool_toggle("mdi.target", "", self._ring_tooltip())
        self.ring_btn.setIcon(gradient_icon("mdi.target", RING_AROUND_SWEEP))
        self.pick_wb_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.ring_btn.setFixedWidth(ICON_BUTTON_WIDTH)

        region_row = QHBoxLayout()
        region_row.addWidget(self.region_btn, 1)
        region_row.addWidget(self.pick_wb_btn)
        region_row.addWidget(self.ring_btn)
        color_col.addLayout(region_row)

        # Temperature lever over the selected region's M/Y pair (real darkroom: cyan stays 0).
        self.temp_slider = KelvinSlider(tr("Temperature"))
        self.temp_slider.setValue(wb_to_kelvin(conf.wb_magenta, conf.wb_yellow))
        self._temp_anchor = None
        color_col.addWidget(self.temp_slider)

        self.cyan_slider = CompactSlider(tr("Cyan"), -1.0, 1.0, conf.wb_cyan, has_neutral=True)
        self.cyan_slider.slider.setObjectName("cyan_slider")
        self.magenta_slider = CompactSlider(tr("Magenta"), -1.0, 1.0, conf.wb_magenta, has_neutral=True)
        self.magenta_slider.slider.setObjectName("magenta_slider")
        self.yellow_slider = CompactSlider(tr("Yellow"), -1.0, 1.0, conf.wb_yellow, has_neutral=True)
        self.yellow_slider.slider.setObjectName("yellow_slider")
        for slider in (self.cyan_slider, self.magenta_slider, self.yellow_slider):
            color_col.addWidget(slider)

        # Balances each dye layer against the frame's own grays; no region scopes it.
        self.cast_removal_slider = CompactSlider(tr("Cast Removal"), 0.0, 1.0, conf.cast_removal_strength)
        self.cast_removal_slider.setToolTip(
            tr("Cast Removal: removes a color cast that differs between the dark and bright parts of the picture. 0 is off.")
        )
        color_col.addWidget(self.cast_removal_slider)
        self.layout.addWidget(self.color_block)

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
        self.filter_combo.currentIndexChanged.connect(
            lambda _i: self.update_config_section("exposure", persist=True, bw_filter=self.filter_combo.currentData())
        )

        self.temp_slider.dragStarted.connect(self._on_temp_drag_started)
        self.temp_slider.dragEnded.connect(self._on_temp_drag_ended)
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        self.temp_slider.valueCommitted.connect(lambda v: self._on_temp_changed(v, persist=True))

        self.cyan_slider.valueChanged.connect(self._on_cyan_changed)
        self.magenta_slider.valueChanged.connect(self._on_magenta_changed)
        self.yellow_slider.valueChanged.connect(self._on_yellow_changed)
        self.cyan_slider.valueCommitted.connect(lambda v: self._on_cyan_changed(v, persist=True))
        self.cast_removal_slider.valueChanged.connect(
            lambda v: self.update_config_section("exposure", persist=False, readback_metrics=False, cast_removal_strength=v)
        )
        self.cast_removal_slider.valueCommitted.connect(
            lambda v: self.update_config_section("exposure", persist=True, readback_metrics=True, cast_removal_strength=v)
        )
        self.magenta_slider.valueCommitted.connect(lambda v: self._on_magenta_changed(v, persist=True))
        self.yellow_slider.valueCommitted.connect(lambda v: self._on_yellow_changed(v, persist=True))

        self.pick_wb_btn.toggled.connect(self._on_pick_wb_toggled)
        self.ring_btn.clicked.connect(lambda checked: self.controller.toggle_ring_around(force=checked))
        # Session state, not config, so the button follows the controller rather than sync_ui.
        self.controller.test_strip_changed.connect(self._sync_ring_btn)

    @staticmethod
    def _ring_tooltip(printing: bool = False) -> str:
        if printing:
            return tr("Printing the color ring-around…")
        return tooltip_with_shortcut(
            tr("Ring-around: show the picture as a grid of color variations, then click the one that looks right."),
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
            self.cast_removal_slider.setValue(conf.cast_removal_strength)
            self.filter_combo.setCurrentIndex(self.filter_combo.findData(conf.bw_filter))
            bw = self.state.config.process.process_mode == ProcessMode.BW
            self.filter_block.setVisible(bw)
            self.color_block.setVisible(not bw)

            for i, fields in enumerate(self._REGION_FIELDS):
                self.region_btn.set_edited(i, any(getattr(conf, f) != 0.0 for f in fields))

            self.pick_wb_btn.setChecked(self.state.active_tool == ToolMode.WB_PICK)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.filter_combo,
            self.region_btn,
            self.temp_slider,
            self.cyan_slider,
            self.magenta_slider,
            self.yellow_slider,
            self.cast_removal_slider,
            self.pick_wb_btn,
            self.ring_btn,
        ):
            w.blockSignals(blocked)
