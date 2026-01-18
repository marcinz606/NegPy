from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from src.desktop.view.widgets.sliders import CompactSlider
from src.desktop.controller import AppController


class LabSidebar(QWidget):
    """
    Panel for color separation and sharpening with high-density horizontal layout.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)
        layout.setSpacing(10)

        conf = self.state.config.lab

        # 1. Color Calibration Row
        layout.addWidget(QLabel("<b>Color Calibration</b>"))
        color_row = QHBoxLayout()
        self.sep_slider = CompactSlider("Separation", 0.5, 2.0, conf.color_separation)
        self.sat_slider = CompactSlider("Saturation", 0.0, 2.0, conf.saturation)
        color_row.addWidget(self.sep_slider)
        color_row.addWidget(self.sat_slider)
        layout.addLayout(color_row)

        # 2. Detail & Clarity Row
        layout.addWidget(QLabel("<b>Detail & Clarity</b>"))
        detail_row = QHBoxLayout()
        self.clahe_slider = CompactSlider("CLAHE", 0.0, 1.0, conf.clahe_strength)
        self.sharp_slider = CompactSlider("Sharpen", 0.0, 1.0, conf.sharpen)
        detail_row.addWidget(self.clahe_slider)
        detail_row.addWidget(self.sharp_slider)
        layout.addLayout(detail_row)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.sep_slider.valueChanged.connect(
            lambda v: self._update_lab("color_separation", v)
        )
        self.sat_slider.valueChanged.connect(
            lambda v: self._update_lab("saturation", v)
        )
        self.clahe_slider.valueChanged.connect(
            lambda v: self._update_lab("clahe_strength", v)
        )
        self.sharp_slider.valueChanged.connect(lambda v: self._update_lab("sharpen", v))

    def _update_lab(self, field: str, val: float) -> None:
        from dataclasses import replace

        new_lab = replace(self.state.config.lab, **{field: val})
        self.controller.session.update_config(replace(self.state.config, lab=new_lab))
        self.controller.request_render()

    def sync_ui(self) -> None:
        """
        Updates widgets from current state.
        """
        conf = self.state.config.lab
        self.sep_slider.blockSignals(True)
        self.sat_slider.blockSignals(True)
        self.clahe_slider.blockSignals(True)
        self.sharp_slider.blockSignals(True)

        try:
            self.sep_slider.setValue(conf.color_separation)
            self.sat_slider.setValue(conf.saturation)
            self.clahe_slider.setValue(conf.clahe_strength)
            self.sharp_slider.setValue(conf.sharpen)
        finally:
            self.sep_slider.blockSignals(False)
            self.sat_slider.blockSignals(False)
            self.clahe_slider.blockSignals(False)
            self.sharp_slider.blockSignals(False)
