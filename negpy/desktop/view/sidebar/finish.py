from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QColorDialog, QHBoxLayout, QPushButton

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.widgets.sliders import CompactSlider


class FinishSidebar(BaseSidebar):
    """
    Panel for post-crop finishing effects: vignette, border.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.finish

        self.layout.addWidget(section_subheader("VIGNETTE"))

        row1 = QHBoxLayout()
        self.vignette_strength_slider = CompactSlider("Strength", -1.0, 1.0, conf.vignette_strength)
        self.vignette_size_slider = CompactSlider("Size", 0.0, 1.0, conf.vignette_size)
        row1.addWidget(self.vignette_strength_slider)
        row1.addWidget(self.vignette_size_slider)
        self.layout.addLayout(row1)

        self.layout.addWidget(section_subheader("BORDER"))

        row2 = QHBoxLayout()
        self.border_slider = CompactSlider("Width", 0.0, 2.5, conf.border_size)
        self.color_btn = QPushButton()
        self.color_btn.setFixedHeight(30)
        self.color_btn.setFixedWidth(30)
        self.color_btn.setToolTip("Click to pick a border colour")
        self._update_color_btn(conf.border_color)
        row2.addWidget(self.border_slider)
        row2.addWidget(self.color_btn)
        self.layout.addLayout(row2)

        self.layout.addStretch()

    def _update_color_btn(self, hex_color: str) -> None:
        self.color_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")

    def _connect_signals(self) -> None:
        self.vignette_strength_slider.valueChanged.connect(
            lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, vignette_strength=v)
        )
        self.vignette_strength_slider.valueCommitted.connect(
            lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, vignette_strength=v)
        )

        self.vignette_size_slider.valueChanged.connect(
            lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, vignette_size=v)
        )
        self.vignette_size_slider.valueCommitted.connect(
            lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, vignette_size=v)
        )

        self.border_slider.valueChanged.connect(
            lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, border_size=v)
        )
        self.border_slider.valueCommitted.connect(
            lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, border_size=v)
        )

        self.color_btn.clicked.connect(self._on_color_clicked)

    def _on_color_clicked(self) -> None:
        color = QColorDialog.getColor(QColor(self.state.config.finish.border_color))
        if color.isValid():
            hex_color = color.name()
            self._update_color_btn(hex_color)
            self.update_config_section("finish", persist=True, render=True, border_color=hex_color)

    def sync_ui(self) -> None:
        conf = self.state.config.finish
        self.block_signals(True)
        try:
            self.vignette_strength_slider.setValue(conf.vignette_strength)
            self.vignette_size_slider.setValue(conf.vignette_size)
            self.border_slider.setValue(conf.border_size)
            self._update_color_btn(conf.border_color)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.vignette_strength_slider,
            self.vignette_size_slider,
            self.border_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
