from dataclasses import replace

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from negpy.desktop.session import ToolMode
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import (
    ICON_BUTTON_WIDTH,
    field_row,
    header_button,
    header_row,
    hint_label,
    section_subheader,
    wrap_tooltip,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.widgets.shape_list import ShapeList, ShapeRow
from negpy.desktop.view.widgets.sliders import CompactSlider, SliderGroup
from negpy.features.leak.bake import leak_active
from negpy.features.leak.models import LeakColor, LeakKind
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.i18n import tr

_LEAK_TIP = tr("Reduce light leaks under the leaks you draw or Auto places, before any dust repair.")
_BAND_TIP = tr("Band: drag along the leak's strongest line, then drag each dashed edge to where the leak fades out.")
_EDGE_TIP = tr("Edge: drag along the side a leak comes in from; the line snaps to it. Drag the dashed line to where it fades.")
_SPOT_TIP = tr("Spot: drag out an oval over a round leak, as a pinhole or a frame-counter window throws.")
_PAINT_TIP = tr("Paint: brush over a leak with no line or center, wider than the leak. Uses the Retouch card's Brush Size.")
_AUTO_TIP = tr(
    "Auto: place a shape over each leak the rebate supports: an edge where it crosses into the picture, a band along a straight leak."
)
_ROLL_TIP = tr(
    "From Roll: carry this frame's leaks to the selected frames; each reads its own strength. With none selected, select the frames with a leak."
)
_BURNED_TIP = tr("Show Burned: wash the film the leak burned out in every layer, where no picture is left to bring back")
_RECOVER_TIP = tr("Rebuild a color layer the leak drove to its densest from the layers that still hold the picture.")
_KIND_ICONS = {
    LeakKind.BAND: "fa5s.grip-lines-vertical",
    LeakKind.EDGE: "mdi.border-top-variant",
    LeakKind.SPOT: "fa5s.circle",
    LeakKind.PAINT: "fa5s.paint-roller",
}
_KIND_LABELS = {LeakKind.BAND: tr("Band"), LeakKind.EDGE: tr("Edge"), LeakKind.SPOT: tr("Spot"), LeakKind.PAINT: tr("Paint")}
# Order matches the Color choice.
_COLORS = (LeakColor.AUTO, LeakColor.BACK, LeakColor.FRONT)
_COLOR_LABELS = (tr("Auto"), tr("Back"), tr("Front"))
_COLOR_TIPS = (
    tr("Read the leak's color off the film: lit rebate it crosses, else the film under it"),
    tr("Light through the base, as a worn door seal lets in: it reaches the red-sensitive layer first and prints red to orange"),
    tr("Light on the emulsion side, through the lens or a gap at the gate: it reaches the blue-sensitive layer first"),
)


class LeakSidebar(BaseSidebar):
    """Panel for light leak reduction."""

    def _init_ui(self) -> None:
        conf = self.state.config.leak

        self.leak_btn = self._small_toggle("fa5s.sun", tr("Reduce Light Leak"), conf.leak_remove, _LEAK_TIP, align_left=True)
        self.auto_btn = self._icon_action("fa5s.magic", _AUTO_TIP)
        self.auto_btn.plain_tooltip = _AUTO_TIP
        self.roll_btn = self._icon_action("fa5s.film", _ROLL_TIP)
        self.roll_btn.plain_tooltip = _ROLL_TIP
        top_row = QHBoxLayout()
        top_row.addWidget(self.leak_btn, 1)
        top_row.addWidget(self.auto_btn)
        top_row.addWidget(self.roll_btn)
        self.layout.addLayout(top_row)

        self.dmax_slider = CompactSlider(tr("Film Dmax"), 0.0, 4.0, conf.leak_dmax, step=0.05, precision=100)
        self.dmax_slider.spin.setSpecialValueText(tr("Off"))
        self.dmax_slider.setToolTip(
            wrap_tooltip(
                tr(
                    "The film's maximum density over its base. Off treats the curve as a straight line; set it when the film's dense end is saturated."
                )
            )
        )
        self.grain_slider = CompactSlider(tr("Grain"), 0.0, 100.0, conf.leak_grain * 100.0, step=1.0, precision=1, unit="%")
        self.grain_slider.setToolTip(wrap_tooltip(tr("Calms the grain the correction stretches, only where leak came off")))
        self.contrast_slider = CompactSlider(tr("Contrast"), 0.0, 100.0, conf.leak_contrast * 100.0, step=1.0, precision=1, unit="%")
        self.contrast_slider.setToolTip(
            wrap_tooltip(
                tr("Lifts the local contrast the leak flattened, only where leak came off. The correction alone cannot bring it back")
            )
        )
        self.layout.addWidget(SliderGroup(self.dmax_slider, self.grain_slider, self.contrast_slider))
        self.recover_btn = self._small_toggle("fa5s.layer-group", tr("Recover Layers"), conf.leak_recover, _RECOVER_TIP, align_left=True)
        self.layout.addWidget(self.recover_btn)

        self.leaks_subheader = section_subheader(tr("LEAKS · {n}", n=0))
        self.burned_btn = self._tool_toggle("fa5s.fire-alt", "", _BURNED_TIP)
        self.burned_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.undo_btn = self._icon_action("fa5s.undo", tr("Undo Last: remove the most recent leak"))
        self.clear_btn = self._icon_action("fa5s.trash-alt", tr("Clear All: remove all leaks"))
        self.layout.addLayout(header_row(self.leaks_subheader, self.burned_btn, self.undo_btn, self.clear_btn))
        self.band_btn = self._tool_toggle(_KIND_ICONS[LeakKind.BAND], "", _BAND_TIP)
        self.edge_btn = self._tool_toggle(_KIND_ICONS[LeakKind.EDGE], "", _EDGE_TIP)
        self.spot_btn = self._tool_toggle(_KIND_ICONS[LeakKind.SPOT], "", _SPOT_TIP)
        self.paint_btn = self._tool_toggle(_KIND_ICONS[LeakKind.PAINT], "", _PAINT_TIP)
        tools_row = QHBoxLayout()
        for btn in self._tool_modes():
            btn.setFixedWidth(ICON_BUTTON_WIDTH)
            tools_row.addWidget(btn)
        tools_row.addStretch(1)
        self.layout.addLayout(tools_row)

        self.leak_list = ShapeList(
            tr("Click a leak to select it. Click its shape icon to turn it off or on, and the trash icon to delete it.")
        )
        self.layout.addWidget(self.leak_list)

        self.layout.addWidget(section_subheader(tr("SELECTED LEAK")))
        self.strength_slider = CompactSlider(tr("Strength"), 0.0, 400.0, 100.0, step=1.0, precision=1, unit="%")
        self.strength_slider.setToolTip(
            wrap_tooltip(tr("How much of the leak the film reads under the selected shape comes off; nearby thin film limits it."))
        )
        self.falloff_slider = CompactSlider(tr("Falloff"), 0.0, 100.0, 50.0, step=1.0, precision=1, unit="%")
        self.falloff_slider.setToolTip(
            wrap_tooltip(
                tr(
                    "How the selected band, spot or painted region fades toward its edge: low holds a shoulder, high fades early and softly."
                )
            )
        )
        self.layout.addWidget(self.strength_slider)
        self.layout.addWidget(self.falloff_slider)
        self.color_btn = SegmentedChoice(tuple(("", label) for label in _COLOR_LABELS), tr("Where the selected leak's color comes from"))
        for i, tip in enumerate(_COLOR_TIPS):
            self.color_btn.set_choice_tooltip(i, tip)
        color_row = field_row(tr("Color"), self.color_btn)
        color_row.setContentsMargins(0, 0, 0, 0)
        self.color_row = QWidget()
        self.color_row.setLayout(color_row)
        self.layout.addWidget(self.color_row)
        self.trim_slider = CompactSlider(tr("Color Trim"), -50.0, 50.0, 0.0, step=1.0, precision=1, has_neutral=True, unit="%")
        self.trim_slider.setToolTip(
            wrap_tooltip(
                tr("Take out more or less of the selected leak's leading color, where a corrected leak leaves its own tint in the shadows.")
            )
        )
        self.trim_group = SliderGroup(self.trim_slider)
        self.layout.addWidget(self.trim_group)

        self.unsupported_hint = hint_label(
            tr("Light leak reduction works on a single negative or slide, not on a Positive frame or a composite.")
        )
        self.layout.addWidget(self.unsupported_hint)
        self.burned_hint = hint_label(
            tr("Part of the leak burned out every layer, with no picture left to rebuild: Heal or Clone it."), kind="warning"
        )
        self.layout.addWidget(self.burned_hint)
        self.layout.addStretch()

    def _tool_modes(self) -> dict:
        return {
            self.band_btn: ToolMode.LEAK_BAND,
            self.edge_btn: ToolMode.LEAK_EDGE,
            self.spot_btn: ToolMode.LEAK_SPOT,
            self.paint_btn: ToolMode.LEAK_PAINT,
        }

    def _connect_signals(self) -> None:
        for slider, field, cast in (
            (self.dmax_slider, "leak_dmax", float),
            (self.grain_slider, "leak_grain", lambda v: float(v) / 100.0),
            (self.contrast_slider, "leak_contrast", lambda v: float(v) / 100.0),
        ):
            slider.valueChanged.connect(lambda v, f=field, c=cast: self.update_config_section("leak", readback_metrics=False, **{f: c(v)}))
            slider.valueCommitted.connect(lambda v, f=field, c=cast: self.update_config_section("leak", persist=True, **{f: c(v)}))
        for slider, field in ((self.strength_slider, "strength"), (self.falloff_slider, "falloff"), (self.trim_slider, "trim")):
            slider.valueChanged.connect(
                lambda v, f=field: self.controller.update_selected_leak(persist=False, readback_metrics=False, **{f: float(v) / 100.0})
            )
            slider.valueCommitted.connect(lambda v, f=field: self.controller.update_selected_leak(**{f: float(v) / 100.0}))
        self.color_btn.currentChanged.connect(lambda i: self.controller.update_selected_leak(color=_COLORS[i]))
        self.leak_btn.toggled.connect(lambda c: self.update_config_section("leak", persist=True, render=True, leak_remove=c))
        self.recover_btn.toggled.connect(lambda c: self.update_config_section("leak", persist=True, render=True, leak_recover=c))
        for btn, mode in self._tool_modes().items():
            btn.toggled.connect(lambda c, m=mode: self.controller.set_active_tool(m if c else ToolMode.NONE))
        self.auto_btn.clicked.connect(self.controller.auto_leaks)
        self.roll_btn.clicked.connect(self.controller.leaks_from_roll)
        self.burned_btn.toggled.connect(self.controller.set_leak_show_burned)
        self.controller.metrics_available.connect(lambda _metrics: self._sync_burned())
        self.undo_btn.clicked.connect(self.controller.undo_last_leak)
        self.clear_btn.clicked.connect(self.controller.clear_leaks)

    def _build_row(self, i: int, shape) -> ShapeRow:
        color = THEME.text_primary if shape.enabled else THEME.text_muted
        row = ShapeRow()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(THEME.space_md, THEME.space_xs, THEME.space_sm, THEME.space_xs)
        lay.setSpacing(THEME.space_sm)
        kind_btn = header_button(qta.icon(_KIND_ICONS[shape.kind], color=color), tr("Click to turn this leak's correction off or on"))
        values = [] if shape.strength == 1.0 else [f"{shape.strength * 100.0:.0f}%"]
        if shape.color is not LeakColor.AUTO and self._color_film():
            values.append(_COLOR_LABELS[_COLORS.index(shape.color)])
        if shape.trim and self._color_film():
            values.append(tr("Trim {value}", value=f"{shape.trim * 100.0:+.0f}%"))
        label = QLabel(f"{i + 1}.  {_KIND_LABELS[shape.kind]}   " + " · ".join(values))
        label.setStyleSheet(f"color: {color};")
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        delete = header_button(qta.icon("fa5s.trash-alt", color=THEME.text_primary), tr("Delete this leak"))
        lay.addWidget(kind_btn)
        lay.addWidget(label)
        lay.addStretch()
        lay.addWidget(delete)
        row.clicked.connect(lambda i=i: self.controller.select_leak(i))
        kind_btn.clicked.connect(lambda _=False, i=i, on=not shape.enabled: self.controller.set_leak_enabled(i, on))
        delete.clicked.connect(lambda _=False, i=i: self.controller.delete_leak(i))
        return row

    def _sync_burned(self) -> None:
        self.burned_hint.setVisible(self.state.config.leak.leak_remove and self.state.leak_burned)

    def _sync_enabled(self) -> None:
        config = self.state.config
        usable = leak_active(replace(config, leak=replace(config.leak, leak_remove=True)))
        on = usable and config.leak.leak_remove
        for w in (self.leak_btn, self.burned_btn, self.auto_btn, self.roll_btn, *self._tool_modes()):
            w.setEnabled(usable)
        for w in (self.dmax_slider, self.grain_slider, self.contrast_slider, self.recover_btn):
            w.setEnabled(on)
        self.leak_list.setEnabled(usable)
        self.unsupported_hint.setVisible(not usable)

    def sync_ui(self) -> None:
        conf = self.state.config.leak
        self.block_signals(True)
        try:
            self.leak_btn.setChecked(conf.leak_remove)
            self.recover_btn.setChecked(conf.leak_recover)
            for btn, mode in self._tool_modes().items():
                btn.setChecked(self.state.active_tool == mode)
            self.burned_btn.setChecked(self.state.leak_show_burned)
            self._sync_burned()
            count = len(conf.leak_shapes)
            self.leaks_subheader.setText(tr("LEAKS · {n}", n=count))
            self.undo_btn.setEnabled(count > 0)
            self.clear_btn.setEnabled(count > 0)
            self.dmax_slider.setValue(conf.leak_dmax)
            self.grain_slider.setValue(conf.leak_grain * 100.0)
            self.contrast_slider.setValue(conf.leak_contrast * 100.0)
            selected = self.state.leak_selected
            self.leak_list.set_rows([self._build_row(i, s) for i, s in enumerate(conf.leak_shapes)], selected)
            shape = conf.leak_shapes[selected] if 0 <= selected < count else None
            self.strength_slider.setEnabled(shape is not None)
            self.falloff_slider.setEnabled(shape is not None and shape.kind is not LeakKind.EDGE)
            self.color_btn.setEnabled(shape is not None)
            self.color_row.setVisible(self._color_film())
            self.trim_slider.setEnabled(shape is not None)
            self.trim_group.setVisible(self._color_film())
            if shape is not None:
                self.strength_slider.setValue(shape.strength * 100.0)
                self.falloff_slider.setValue(shape.falloff * 100.0)
                self.color_btn.setCurrentIndex(_COLORS.index(shape.color))
                self.trim_slider.setValue(shape.trim * 100.0)
            self._sync_enabled()
        finally:
            self.block_signals(False)

    def _color_film(self) -> bool:
        """A black-and-white film has one layer, so a leak on it has no color to set."""
        return self.state.config.process.process_mode != ProcessMode.BW

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.leak_btn,
            self.recover_btn,
            self.burned_btn,
            *self._tool_modes(),
            self.dmax_slider,
            self.grain_slider,
            self.contrast_slider,
            self.strength_slider,
            self.falloff_slider,
            self.color_btn,
            self.trim_slider,
        ):
            w.blockSignals(blocked)
