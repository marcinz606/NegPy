from PyQt6.QtWidgets import QHBoxLayout
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.features.process.models import ProcessMode


class LabSidebar(BaseSidebar):
    """
    Panel for color separation, sharpening, and contrast.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.lab

        row1 = QHBoxLayout()
        self.separation_slider = CompactSlider("Separation", 1.0, 2.0, conf.color_separation)
        self.saturation_slider = CompactSlider("Saturation", 0.0, 2.0, conf.saturation)
        row1.addWidget(self.separation_slider)
        row1.addWidget(self.saturation_slider)
        self.layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.clahe_slider = CompactSlider("CLAHE", 0.0, 1.0, conf.clahe_strength)
        self.sharpen_slider = CompactSlider("Sharpening", 0.0, 2.0, conf.sharpen)
        row2.addWidget(self.clahe_slider)
        row2.addWidget(self.sharpen_slider)
        self.layout.addLayout(row2)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.clahe_slider.valueChanged.connect(lambda v: self.update_config_section("lab", readback_metrics=False, clahe_strength=v))
        self.sharpen_slider.valueChanged.connect(lambda v: self.update_config_section("lab", readback_metrics=False, sharpen=v))
        self.saturation_slider.valueChanged.connect(lambda v: self.update_config_section("lab", readback_metrics=False, saturation=v))
        self.separation_slider.valueChanged.connect(lambda v: self.update_config_section("lab", readback_metrics=False, color_separation=v))

    def sync_ui(self) -> None:
        conf = self.state.config.lab
        is_bw = self.state.config.process.process_mode == ProcessMode.BW

        self.block_signals(True)
        try:
            self.clahe_slider.setValue(conf.clahe_strength)
            self.sharpen_slider.setValue(conf.sharpen)
            self.saturation_slider.setValue(conf.saturation)
            self.separation_slider.setValue(conf.color_separation)

            self.separation_slider.setEnabled(not is_bw)
            self.saturation_slider.setEnabled(not is_bw)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.clahe_slider,
            self.sharpen_slider,
            self.saturation_slider,
            self.separation_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
