from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QWidget
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.templates import field_label_qss
from negpy.desktop.view.styles.theme import THEME


_MASK_ROW_H = 30


class _MaskRow(QWidget):
    """One mask list row; clicking its background (not the buttons) selects the mask."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)


class LocalSidebar(BaseSidebar):
    """
    Polygon-mask dodge/burn local adjustments. Draw a polygon, then tune
    its strength (dodge/burn EV) and feather independently of other masks.
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
        self.layout.addWidget(self.draw_btn)

        self.mask_list = QListWidget()
        self.mask_list.setToolTip(
            "Click a mask to select it. Use the eye to show/hide its outline and the trash icon to delete it."
        )
        self.mask_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.mask_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The row is a custom widget, so drop the app-wide item padding/margin/border
        # that would otherwise squeeze and clip it.
        self.mask_list.setStyleSheet(
            "QListView::item { border: none; margin: 0px; padding: 0px; }"
            "QListView::item:selected { background-color: #2A2A2A; }"
        )
        self.layout.addWidget(self.mask_list)

        self.strength_slider = CompactSlider("Strength", -1.0, 1.0, 0.3, step=0.05, precision=100, has_neutral=True, unit=" EV")
        self.strength_slider.setToolTip("EV adjustment for the selected mask — positive brightens (dodge), negative darkens (burn)")

        self.feather_slider = CompactSlider("Feather", 0.0, 0.15, 0.04, step=0.005, precision=1000)
        self.feather_slider.setToolTip("Edge softness for the selected mask")

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.strength_slider)
        slider_row.addWidget(self.feather_slider)
        self.layout.addLayout(slider_row)

        self.mask_count_label = QLabel("0 masks")
        self.mask_count_label.setStyleSheet(field_label_qss())
        self.layout.addWidget(self.mask_count_label)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.draw_btn.toggled.connect(self._on_draw_toggled)
        self.strength_slider.valueChanged.connect(lambda v: self.controller.update_selected_local_mask(strength=float(v)))
        self.feather_slider.valueChanged.connect(lambda v: self.controller.update_selected_local_mask(feather=float(v)))

    def _on_draw_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.LOCAL_DRAW if checked else ToolMode.NONE)

    def _row_icon_btn(self, icon_name: str, checkable: bool) -> QPushButton:
        btn = QPushButton()
        btn.setCheckable(checkable)
        btn.setFlat(True)
        btn.setIcon(qta.icon(icon_name, color=THEME.text_primary))
        btn.setFixedSize(26, 22)
        btn.setStyleSheet("QPushButton {border: none; padding: 0px;}")
        return btn

    def _build_mask_row(self, i: int, mask) -> _MaskRow:
        kind = "Dodge" if mask.strength >= 0 else "Burn"
        row = _MaskRow()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 2, 4, 2)
        lay.setSpacing(4)

        label = QLabel(f"{i + 1}.  {kind}   {mask.strength:+.2f} EV")
        label.setStyleSheet(f"color: {'#E8C84A' if mask.strength >= 0 else '#4A8FE8'};")
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        visible = i not in self.state.local_hidden_masks
        eye = self._row_icon_btn("fa5s.eye" if visible else "fa5s.eye-slash", checkable=True)
        eye.setChecked(visible)
        eye.setToolTip("Show or hide this mask's outline on the canvas")
        delete = self._row_icon_btn("fa5s.trash-alt", checkable=False)
        delete.setToolTip("Delete this mask")

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
            self.draw_btn.setChecked(self.state.active_tool == ToolMode.LOCAL_DRAW)

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
            self.strength_slider.setEnabled(has_selection)
            self.feather_slider.setEnabled(has_selection)
            if has_selection:
                mask = conf.masks[idx]
                self.strength_slider.setValue(mask.strength)
                self.feather_slider.setValue(mask.feather)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in [self.draw_btn, self.strength_slider, self.feather_slider]:
            w.blockSignals(blocked)
