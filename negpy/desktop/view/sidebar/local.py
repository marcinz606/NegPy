from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QWidget
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.templates import field_label_qss
from negpy.desktop.view.styles.theme import THEME
from negpy.features.local.models import MaskShape


_MASK_ROW_H = 30
_SHAPE_ICONS = {
    MaskShape.POLYGON: "fa5s.draw-polygon",
    MaskShape.OVAL: "fa5s.circle",
    MaskShape.GRADIENT: "fa5s.grip-lines",
}


class _MaskRow(QWidget):
    """One mask list row; clicking its background (not the buttons) selects the mask."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)


class LocalSidebar(BaseSidebar):
    """
    Polygon-mask dodge/burn local adjustments. Draw a polygon, then tune its
    print exposure, its grade and its feather independently of other masks.
    """

    def _init_ui(self) -> None:
        self.draw_btn = self._tool_toggle(
            "fa5s.draw-polygon",
            "Draw Mask",
            "Draw a new mask: click to place vertices; double-click, Enter, or a click near "
            "the start closes; Esc cancels. Select a mask from the list to edit it (no need to "
            "re-enter this tool): drag a vertex to move it, click an edge '+' dot to add a point, "
            "right-click a vertex to delete it.",
        )
        self.oval_btn = self._tool_toggle(
            "fa5s.circle",
            "Oval",
            "Burn through a hole in the card, or dodge with a wand: drag out an oval. Its three "
            "handles move it (centre) and set each axis, so it can be stretched and tilted.",
        )
        self.gradient_btn = self._tool_toggle(
            "fa5s.grip-lines",
            "Card Edge",
            "The graduated burn a printer makes by moving a card across the paper: drag from the "
            "full-exposure edge (solid line) to where it fades out (dashed). The distance between "
            "the two handles is the softness, so Feather does nothing here.",
        )
        tool_row = QHBoxLayout()
        tool_row.addWidget(self.draw_btn)
        tool_row.addWidget(self.oval_btn)
        tool_row.addWidget(self.gradient_btn)
        self.layout.addLayout(tool_row)

        self.mask_list = QListWidget()
        self.mask_list.setToolTip("Click a mask to select it. Use the eye to show/hide its outline and the trash icon to delete it.")
        self.mask_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.mask_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The row is a custom widget, so drop the app-wide item padding/margin/border
        # that would otherwise squeeze and clip it.
        self.mask_list.setStyleSheet(
            "QListView::item { border: none; margin: 0px; padding: 0px; }QListView::item:selected { background-color: #2A2A2A; }"
        )
        self.layout.addWidget(self.mask_list)

        # Exposure-signed like the frame's Print Density and the Finishing edge burn:
        # positive is more light on the paper, hence darker.
        self.burn_slider = CompactSlider("Burn", -2.0, 2.0, 0.0, step=0.05, precision=100, has_neutral=True, unit=" st")
        self.burn_slider.setToolTip(
            "Print exposure for the selected mask, in stops — positive burns (longer exposure, "
            "darker paper), negative dodges (held back, brighter paper)"
        )

        self.feather_slider = CompactSlider("Feather", 0.0, 0.15, 0.04, step=0.005, precision=1000)
        self.feather_slider.setToolTip("Edge softness for the selected mask")

        # inverted like every other grade slider (Tone's ISO-R Grade, split grade,
        # layer trims): dragging right is harder paper, even though R falls.
        self.grade_slider = CompactSlider("Grade", -40.0, 40.0, 0.0, step=5.0, precision=1, has_neutral=True, unit=" R", inverted=True)
        self.grade_slider.setToolTip(
            "Print the selected mask at its own grade, in ISO-R points off the frame's Grade — "
            "negative is harder, the darkroom's burn-in through the hard filter. The region's own "
            "midtone holds, so this changes its contrast without moving its overall density."
        )

        self.invert_btn = QPushButton("Invert")
        self.invert_btn.setCheckable(True)
        self.invert_btn.setToolTip(
            "Act everywhere except inside the selected mask — the card itself instead of the hole "
            "cut in it. Burn the surround and hold the face, in one mask."
        )

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.burn_slider)
        slider_row.addWidget(self.grade_slider)
        self.layout.addLayout(slider_row)
        self.layout.addWidget(self.feather_slider)
        self.layout.addWidget(self.invert_btn)

        self.mask_count_label = QLabel("0 masks")
        self.mask_count_label.setStyleSheet(field_label_qss())
        self.layout.addWidget(self.mask_count_label)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        for btn, mode in self._tool_modes().items():
            btn.toggled.connect(lambda checked, m=mode: self._on_draw_toggled(checked, m))
        # Drag steps render only; the commit writes history and settings, as in every
        # other sidebar.
        for slider, field in (
            (self.burn_slider, "stops"),
            (self.feather_slider, "feather"),
            (self.grade_slider, "grade"),
        ):
            slider.valueChanged.connect(
                lambda v, f=field: self.controller.update_selected_local_mask(persist=False, readback_metrics=False, **{f: float(v)})
            )
            slider.valueCommitted.connect(lambda v, f=field: self.controller.update_selected_local_mask(**{f: float(v)}))
        self.invert_btn.toggled.connect(lambda v: self.controller.update_selected_local_mask(invert=bool(v)))

    def _tool_modes(self) -> dict:
        return {
            self.draw_btn: ToolMode.LOCAL_DRAW,
            self.oval_btn: ToolMode.LOCAL_OVAL,
            self.gradient_btn: ToolMode.LOCAL_GRADIENT,
        }

    def _on_draw_toggled(self, checked: bool, mode: ToolMode) -> None:
        self.controller.set_active_tool(mode if checked else ToolMode.NONE)

    def _row_icon_btn(self, icon_name: str, checkable: bool) -> QPushButton:
        btn = QPushButton()
        btn.setCheckable(checkable)
        btn.setFlat(True)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary))
        btn.setFixedSize(26, 22)
        btn.setStyleSheet("QPushButton {border: none; padding: 0px;}")
        return btn

    def _build_mask_row(self, i: int, mask) -> _MaskRow:
        if mask.stops > 0:
            kind, color = "Burn", "#4A8FE8"
        elif mask.stops < 0:
            kind, color = "Dodge", "#E8C84A"
        else:
            # A mask that only changes grade is neither: it re-prints the area
            # at its own contrast without adding or holding back exposure.
            kind, color = "Grade", THEME.text_primary
        row = _MaskRow()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 2, 4, 2)
        lay.setSpacing(4)

        values = [f"{mask.stops:+.2f} st"] if mask.stops else []
        if mask.grade:
            values.append(f"{mask.grade:+.0f} R")
        if mask.invert:
            values.append("inv")
        shape_icon = QLabel()
        shape_icon.setPixmap(qta.icon(_SHAPE_ICONS[mask.shape], color=color).pixmap(12, 12))
        shape_icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        label = QLabel(f"{i + 1}.  {kind}   " + "  ".join(values))
        label.setStyleSheet(f"color: {color};")
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        visible = i not in self.state.local_hidden_masks
        eye = self._row_icon_btn("fa5s.eye" if visible else "fa5s.eye-slash", checkable=True)
        eye.setChecked(visible)
        eye.setToolTip("Show or hide this mask's outline on the canvas")
        delete = self._row_icon_btn("fa5s.trash-alt", checkable=False)
        delete.setToolTip("Delete this mask")

        lay.addWidget(shape_icon)
        lay.addWidget(label)
        lay.addStretch()
        lay.addWidget(eye)
        lay.addWidget(delete)

        row.clicked.connect(lambda i=i: self.controller.select_local_mask(i))
        eye.toggled.connect(lambda checked, i=i, b=eye: self._on_eye_toggled(i, checked, b))
        delete.clicked.connect(lambda _=False, i=i: self.controller.delete_local_mask(i))
        return row

    def _on_eye_toggled(self, i: int, checked: bool, btn: QPushButton) -> None:
        btn.setIcon(qta.icon("fa5s.eye" if checked else "fa5s.eye-slash", color=THEME.text_primary))
        self.controller.set_local_mask_visible(i, checked)

    def sync_ui(self) -> None:
        conf = self.state.config.local
        self.block_signals(True)
        try:
            for btn, mode in self._tool_modes().items():
                btn.setChecked(self.state.active_tool == mode)

            n = len(conf.masks)
            self.mask_count_label.setText(f"{n} mask{'s' if n != 1 else ''}")

            idx = self.state.local_selected_mask
            has_selection = 0 <= idx < n

            self.mask_list.blockSignals(True)
            self.mask_list.clear()
            for i, mask in enumerate(conf.masks):
                item = QListWidgetItem()
                row = self._build_mask_row(i, mask)
                item.setSizeHint(QSize(0, _MASK_ROW_H))
                self.mask_list.addItem(item)
                self.mask_list.setItemWidget(item, row)
            if has_selection:
                self.mask_list.setCurrentRow(idx)
            else:
                self.mask_list.clearSelection()
            self.mask_list.setVisible(n > 0)
            if n:
                self.mask_list.setFixedHeight(_MASK_ROW_H * n + 2 * self.mask_list.frameWidth())
            self.mask_list.blockSignals(False)
            mask = conf.masks[idx] if has_selection else None
            self.burn_slider.setEnabled(has_selection)
            # The distance between the handles sets the card-edge softness, not a blur.
            self.feather_slider.setEnabled(has_selection and mask.shape != MaskShape.GRADIENT)
            self.grade_slider.setEnabled(has_selection)
            self.invert_btn.setEnabled(has_selection)
            if mask is not None:
                self.burn_slider.setValue(mask.stops)
                self.feather_slider.setValue(mask.feather)
                self.grade_slider.setValue(mask.grade)
                self.invert_btn.setChecked(mask.invert)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in [*self._tool_modes(), self.burn_slider, self.feather_slider, self.grade_slider, self.invert_btn]:
            w.blockSignals(blocked)
