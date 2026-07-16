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
        conf = self.state.config.finish

        self.layout.addWidget(section_subheader("VIGNETTE"))

        row1 = QHBoxLayout()
        self.vignette_burn_slider = CompactSlider("Burn", -2.0, 2.0, conf.vignette_stops, unit=" st")
        self.vignette_size_slider = CompactSlider("Size", 0.0, 1.0, conf.vignette_size)
        row1.addWidget(self.vignette_burn_slider)
        row1.addWidget(self.vignette_size_slider)
        self.layout.addLayout(row1)

        row1b = QHBoxLayout()
        self.vignette_roundness_slider = CompactSlider("Roundness", 0.0, 1.0, conf.vignette_roundness)
        row1b.addWidget(self.vignette_roundness_slider)
        self.layout.addLayout(row1b)

        self.layout.addWidget(section_subheader("FILED CARRIER"))

        self.carrier_btn = QPushButton("Print the rebate")
        self.carrier_btn.setCheckable(True)
        self.carrier_btn.setChecked(conf.carrier_enabled)
        self.carrier_btn.setToolTip("Filed-out negative carrier: a black rebate frame with a rough inner edge")
        self.layout.addWidget(self.carrier_btn)

        row_carrier = QHBoxLayout()
        self.carrier_width_slider = CompactSlider("Width", 0.5, 5.0, conf.carrier_width, unit=" mm")
        self.carrier_rough_slider = CompactSlider("Rough", 0.0, 1.0, conf.carrier_rough)
        row_carrier.addWidget(self.carrier_width_slider)
        row_carrier.addWidget(self.carrier_rough_slider)
        self.layout.addLayout(row_carrier)

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

        row3 = QHBoxLayout()
        self.bottom_weight_slider = CompactSlider("Bottom weight", 1.0, 2.0, conf.border_bottom_weight)
        row3.addWidget(self.bottom_weight_slider)
        self.layout.addLayout(row3)

        row4 = QHBoxLayout()
        self.match_paper_btn = QPushButton("Match paper white")
        self.match_paper_btn.setCheckable(True)
        self.match_paper_btn.setChecked(conf.border_match_paper)
        self.match_paper_btn.setToolTip("Tint the mat with the toned paper white instead of the picked colour")
        row4.addWidget(self.match_paper_btn)
        self.layout.addLayout(row4)

        self.layout.addStretch()

    def _update_color_btn(self, hex_color: str) -> None:
        self.color_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")

    def _connect_signals(self) -> None:
        self.vignette_burn_slider.valueChanged.connect(
            lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, vignette_stops=v)
        )
        self.vignette_burn_slider.valueCommitted.connect(
            lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, vignette_stops=v)
        )

        self.vignette_roundness_slider.valueChanged.connect(
            lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, vignette_roundness=v)
        )
        self.vignette_roundness_slider.valueCommitted.connect(
            lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, vignette_roundness=v)
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

        self.carrier_btn.toggled.connect(lambda checked: self.update_config_section("finish", persist=True, carrier_enabled=bool(checked)))
        self.carrier_width_slider.valueChanged.connect(
            lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, carrier_width=v)
        )
        self.carrier_width_slider.valueCommitted.connect(
            lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, carrier_width=v)
        )
        self.carrier_rough_slider.valueChanged.connect(
            lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, carrier_rough=v)
        )
        self.carrier_rough_slider.valueCommitted.connect(
            lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, carrier_rough=v)
        )

        self.bottom_weight_slider.valueChanged.connect(
            lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, border_bottom_weight=v)
        )
        self.bottom_weight_slider.valueCommitted.connect(
            lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, border_bottom_weight=v)
        )

        self.match_paper_btn.toggled.connect(
            lambda checked: self.update_config_section("finish", persist=True, border_match_paper=bool(checked))
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
            self.vignette_burn_slider.setValue(conf.vignette_stops)
            self.vignette_size_slider.setValue(conf.vignette_size)
            self.vignette_roundness_slider.setValue(conf.vignette_roundness)
            self.carrier_btn.setChecked(conf.carrier_enabled)
            self.carrier_width_slider.setValue(conf.carrier_width)
            self.carrier_rough_slider.setValue(conf.carrier_rough)
            self.border_slider.setValue(conf.border_size)
            self.bottom_weight_slider.setValue(conf.border_bottom_weight)
            self.match_paper_btn.setChecked(conf.border_match_paper)
            self._update_color_btn(conf.border_color)
            self.color_btn.setEnabled(not conf.border_match_paper)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.vignette_burn_slider,
            self.vignette_size_slider,
            self.vignette_roundness_slider,
            self.carrier_btn,
            self.carrier_width_slider,
            self.carrier_rough_slider,
            self.border_slider,
            self.bottom_weight_slider,
            self.match_paper_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
