from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QLabel
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.theme import THEME


class LocalSidebar(BaseSidebar):
    """
    Polygon-mask dodge/burn local adjustments. Draw a polygon, then tune
    its strength (dodge/burn EV) and feather independently of other masks.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)

        self.draw_btn = QPushButton(" Draw Mask")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setIcon(qta.icon("fa5s.draw-polygon", color=THEME.text_primary))
        self.draw_btn.setToolTip(
            "Click to place vertices, double-click or click near the start to close. "
            "Click inside an existing mask to select it. Esc cancels the current shape."
        )
        self.show_btn = QPushButton(" Show Masks")
        self.show_btn.setCheckable(True)
        self.show_btn.setIcon(qta.icon("fa5s.eye", color=THEME.text_primary))
        self.show_btn.setToolTip("Show or hide the mask outlines on the canvas")

        button_row = QHBoxLayout()
        button_row.addWidget(self.draw_btn)
        button_row.addWidget(self.show_btn)
        self.layout.addLayout(button_row)

        self.strength_slider = CompactSlider("Strength", -1.0, 1.0, 0.3, step=0.05, precision=100, has_neutral=True, unit=" EV")
        self.strength_slider.setToolTip("EV adjustment for the selected mask — positive brightens (dodge), negative darkens (burn)")

        self.feather_slider = CompactSlider("Feather", 0.0, 0.15, 0.02, step=0.005, precision=1000)
        self.feather_slider.setToolTip("Edge softness for the selected mask")

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.strength_slider)
        slider_row.addWidget(self.feather_slider)
        self.layout.addLayout(slider_row)

        status_row = QHBoxLayout()
        self.mask_count_label = QLabel("0 masks")
        self.mask_count_label.setStyleSheet(f"font-size: {THEME.font_size_base}px; color: {THEME.text_secondary};")
        self.delete_btn = QPushButton(" Delete")
        self.delete_btn.setIcon(qta.icon("fa5s.times", color=THEME.text_primary))
        self.delete_btn.setToolTip("Delete the selected mask")
        self.delete_btn.setEnabled(False)
        self.clear_btn = QPushButton(" Clear All")
        self.clear_btn.setIcon(qta.icon("fa5s.trash-alt", color=THEME.text_primary))
        self.clear_btn.setToolTip("Remove all dodge/burn masks")
        status_row.addWidget(self.mask_count_label)
        status_row.addStretch()
        status_row.addWidget(self.delete_btn)
        status_row.addWidget(self.clear_btn)
        self.layout.addLayout(status_row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.draw_btn.toggled.connect(self._on_draw_toggled)
        self.show_btn.toggled.connect(self.controller.set_local_overlay_visible)
        self.strength_slider.valueChanged.connect(lambda v: self.controller.update_selected_local_mask(strength=float(v)))
        self.feather_slider.valueChanged.connect(lambda v: self.controller.update_selected_local_mask(feather=float(v)))
        self.delete_btn.clicked.connect(self.controller.delete_selected_local_mask)
        self.clear_btn.clicked.connect(self.controller.clear_local)

    def _on_draw_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.LOCAL_DRAW if checked else ToolMode.NONE)

    def sync_ui(self) -> None:
        conf = self.state.config.local
        self.block_signals(True)
        try:
            self.draw_btn.setChecked(self.state.active_tool == ToolMode.LOCAL_DRAW)
            self.show_btn.setChecked(self.state.show_local_overlay)

            n = len(conf.masks)
            self.mask_count_label.setText(f"{n} mask{'s' if n != 1 else ''}")
            self.clear_btn.setEnabled(n > 0)

            idx = self.state.local_selected_mask
            has_selection = 0 <= idx < n
            self.delete_btn.setEnabled(has_selection)
            self.strength_slider.setEnabled(has_selection)
            self.feather_slider.setEnabled(has_selection)
            if has_selection:
                mask = conf.masks[idx]
                self.strength_slider.setValue(mask.strength)
                self.feather_slider.setValue(mask.feather)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in [self.draw_btn, self.show_btn, self.strength_slider, self.feather_slider]:
            w.blockSignals(blocked)
