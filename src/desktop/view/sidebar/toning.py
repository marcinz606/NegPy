from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox
from src.desktop.view.widgets.sliders import SignalSlider
from src.desktop.controller import AppController


class ToningSidebar(QWidget):
    """
    Panel for paper simulation and chemical toning.
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

        layout.addWidget(self.paper_combo)
        layout.addWidget(self.selenium_slider)
        layout.addWidget(self.sepia_slider)

    def _connect_signals(self) -> None:
        self.paper_combo.currentTextChanged.connect(self._on_paper_changed)
        self.selenium_slider.valueChanged.connect(
            lambda v: self._update_toning("selenium_strength", v)
        )
        self.sepia_slider.valueChanged.connect(
            lambda v: self._update_toning("sepia_strength", v)
        )

    def _on_paper_changed(self, text: str) -> None:
        from dataclasses import replace

        new_toning = replace(self.state.config.toning, paper_profile=text)
        self.controller.session.update_config(
            replace(self.state.config, toning=new_toning)
        )
        self.controller.request_render()

    def _update_toning(self, field: str, val: float) -> None:
        from dataclasses import replace

        new_toning = replace(self.state.config.toning, **{field: val})
        self.controller.session.update_config(
            replace(self.state.config, toning=new_toning)
        )
        self.controller.request_render()

    def sync_ui(self) -> None:
        """
        Updates widgets and visibility based on process mode.
        """
        conf = self.state.config.toning
        self.paper_combo.blockSignals(True)
        self.selenium_slider.blockSignals(True)
        self.sepia_slider.blockSignals(True)

        try:
            self.paper_combo.setCurrentText(conf.paper_profile)
            self.selenium_slider.setValue(conf.selenium_strength)
            self.sepia_slider.setValue(conf.sepia_strength)

            # Dynamic Visibility (Only B&W)
            is_bw = self.state.config.process_mode == "B&W"
            self.selenium_slider.setVisible(is_bw)
            self.sepia_slider.setVisible(is_bw)
        finally:
            self.paper_combo.blockSignals(False)
            self.selenium_slider.blockSignals(False)
            self.sepia_slider.blockSignals(False)
