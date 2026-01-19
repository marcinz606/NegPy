from PyQt6.QtWidgets import QComboBox
from src.desktop.view.widgets.sliders import SignalSlider
from src.desktop.view.sidebar.base import BaseSidebar


class ToningSidebar(BaseSidebar):
    """
    Panel for paper simulation and chemical toning.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.toning

        # Paper Profile
        self.paper_combo = QComboBox()
        self.paper_combo.addItems(
            ["None", "Generic Glossy", "Generic Matte", "Warmtone"]
        )
        self.paper_combo.setCurrentText(conf.paper_profile)

        # Toning Sliders
        self.selenium_slider = SignalSlider(
            "Selenium", 0.0, 1.0, conf.selenium_strength
        )
        self.sepia_slider = SignalSlider("Sepia", 0.0, 1.0, conf.sepia_strength)

        self.layout.addWidget(self.paper_combo)
        self.layout.addWidget(self.selenium_slider)
        self.layout.addWidget(self.sepia_slider)

    def _connect_signals(self) -> None:
        self.paper_combo.currentTextChanged.connect(
            lambda v: self.update_config_section("toning", paper_profile=v)
        )
        self.selenium_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", selenium_strength=v)
        )
        self.sepia_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", sepia_strength=v)
        )

    def sync_ui(self) -> None:
        conf = self.state.config.toning
        self.block_signals(True)
        try:
            self.paper_combo.setCurrentText(conf.paper_profile)
            self.selenium_slider.setValue(conf.selenium_strength)
            self.sepia_slider.setValue(conf.sepia_strength)

            # Dynamic Visibility (Only B&W)
            is_bw = self.state.config.process_mode == "B&W"
            self.selenium_slider.setVisible(is_bw)
            self.sepia_slider.setVisible(is_bw)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        self.paper_combo.blockSignals(blocked)
        self.selenium_slider.blockSignals(blocked)
        self.sepia_slider.blockSignals(blocked)
