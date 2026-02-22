from PyQt6.QtWidgets import QComboBox
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.features.process.models import ProcessMode
from negpy.features.toning.logic import PAPER_PROFILES


class ToningSidebar(BaseSidebar):
    """
    Panel for chemical toning simulation and paper substrate.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.toning
        self.paper_combo = QComboBox()
        self.paper_combo.addItems(list(PAPER_PROFILES.keys()))
        self.paper_combo.setCurrentText(conf.paper_profile)
        self.layout.addWidget(self.paper_combo)

        self.selenium_slider = CompactSlider("Selenium", 0.0, 2.0, conf.selenium_strength, color="#444466")
        self.sepia_slider = CompactSlider("Sepia", 0.0, 2.0, conf.sepia_strength, color="#664422")

        self.layout.addWidget(self.selenium_slider)
        self.layout.addWidget(self.sepia_slider)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.paper_combo.currentTextChanged.connect(lambda v: self.update_config_section("toning", paper_profile=v))
        self.selenium_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", readback_metrics=False, selenium_strength=v)
        )
        self.sepia_slider.valueChanged.connect(lambda v: self.update_config_section("toning", readback_metrics=False, sepia_strength=v))

    def sync_ui(self) -> None:
        conf = self.state.config.toning
        is_bw = self.state.config.process.process_mode == ProcessMode.BW

        self.block_signals(True)
        try:
            self.paper_combo.setCurrentText(conf.paper_profile)
            self.selenium_slider.setValue(conf.selenium_strength)
            self.sepia_slider.setValue(conf.sepia_strength)

            self.selenium_slider.setEnabled(is_bw)
            self.sepia_slider.setEnabled(is_bw)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.paper_combo,
            self.selenium_slider,
            self.sepia_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
