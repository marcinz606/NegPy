from PyQt6.QtWidgets import QHBoxLayout, QLabel
from src.desktop.view.widgets.sliders import CompactSlider
from src.desktop.view.sidebar.base import BaseSidebar
from src.desktop.view.styles.theme import THEME
from src.domain.models import ProcessMode


class LabSidebar(BaseSidebar):
    """
    Panel for color separation and sharpening with high-density horizontal layout.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)
        conf = self.state.config.lab

        # Color Calibration
        self.label_color = QLabel("Color Calibration")
        self.label_color.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold;"
        )
        self.layout.addWidget(self.label_color)

        color_row = QHBoxLayout()
        self.sep_slider = CompactSlider("Separation", 1.0, 2.0, conf.color_separation)
        self.sat_slider = CompactSlider("Saturation", 0.0, 2.0, conf.saturation)
        color_row.addWidget(self.sep_slider)
        color_row.addWidget(self.sat_slider)
        self.layout.addLayout(color_row)

        # Detail & Clarity
        label_detail = QLabel("Detail & Clarity")
        label_detail.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        self.layout.addWidget(label_detail)

        detail_row = QHBoxLayout()
        self.clahe_slider = CompactSlider("CLAHE", 0.0, 1.0, conf.clahe_strength)
        self.sharp_slider = CompactSlider("Sharpen", 0.0, 1.0, conf.sharpen)
        detail_row.addWidget(self.clahe_slider)
        detail_row.addWidget(self.sharp_slider)
        self.layout.addLayout(detail_row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.sep_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "lab", readback_metrics=False, color_separation=v
            )
        )
        self.sat_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "lab", readback_metrics=False, saturation=v
            )
        )
        self.clahe_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "lab", readback_metrics=False, clahe_strength=v
            )
        )
        self.sharp_slider.valueChanged.connect(
            lambda v: self.update_config_section(
                "lab", readback_metrics=False, sharpen=v
            )
        )

    def sync_ui(self) -> None:
        conf = self.state.config.lab
        self.block_signals(True)
        try:
            self.sep_slider.setValue(conf.color_separation)
            self.sat_slider.setValue(conf.saturation)
            self.clahe_slider.setValue(conf.clahe_strength)
            self.sharp_slider.setValue(conf.sharpen)

            # Dynamic Visibility (Hide color controls in B&W)
            is_color = self.state.config.process_mode == ProcessMode.C41
            self.label_color.setVisible(is_color)
            self.sep_slider.setVisible(is_color)
            self.sat_slider.setVisible(is_color)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.sep_slider,
            self.sat_slider,
            self.clahe_slider,
            self.sharp_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
