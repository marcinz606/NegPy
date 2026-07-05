from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.widgets.sliders import CompactSlider, HueSlider
from negpy.features.process.models import ProcessMode


class ToningSidebar(BaseSidebar):
    """
    Panel for chemical and split toning simulation.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.toning

        self.layout.addWidget(section_subheader("TONERS"))

        self.selenium_slider = CompactSlider("Selenium", 0.0, 2.0, conf.selenium_strength, color="#444466")
        self.selenium_slider.setToolTip(
            "Simulates selenium toning — converts the densest silver first: deeper blacks, cool eggplant shadows (B&W only)"
        )
        self.sepia_slider = CompactSlider("Sepia", 0.0, 2.0, conf.sepia_strength, color="#664422")
        self.sepia_slider.setToolTip(
            "Simulates sepia bleach-redevelop toning — warms the highlights first, shadows hold; partial strength gives the classic split-sepia look (B&W only)"
        )
        self.layout.addWidget(self.selenium_slider)
        self.layout.addWidget(self.sepia_slider)

        self.layout.addWidget(section_subheader("SPLIT TONE"))

        row_sh = QHBoxLayout()
        self.shadow_hue_slider = HueSlider("Shadow Hue", conf.shadow_tint_hue)
        self.shadow_hue_slider.setToolTip("Hue of the shadow split-toning color")
        self.shadow_str_slider = CompactSlider("Shadow Strength", 0.0, 1.0, conf.shadow_tint_strength)
        self.shadow_str_slider.setToolTip("Strength of the shadow split-tone color")
        row_sh.addWidget(self.shadow_hue_slider)
        row_sh.addWidget(self.shadow_str_slider)
        self.layout.addLayout(row_sh)

        row_hl = QHBoxLayout()
        self.highlight_hue_slider = HueSlider("Highlight Hue", conf.highlight_tint_hue)
        self.highlight_hue_slider.setToolTip("Hue of the highlight split-toning color")
        self.highlight_str_slider = CompactSlider("Highlight Strength", 0.0, 1.0, conf.highlight_tint_strength)
        self.highlight_str_slider.setToolTip("Strength of the highlight split-tone color")
        row_hl.addWidget(self.highlight_hue_slider)
        row_hl.addWidget(self.highlight_str_slider)
        self.layout.addLayout(row_hl)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.selenium_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, selenium_strength=v)
        )
        self.selenium_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, selenium_strength=v)
        )
        self.sepia_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, sepia_strength=v)
        )
        self.sepia_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, sepia_strength=v)
        )
        self.shadow_hue_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, shadow_tint_hue=v)
        )
        self.shadow_hue_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, shadow_tint_hue=v)
        )
        self.shadow_str_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, shadow_tint_strength=v)
        )
        self.shadow_str_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, shadow_tint_strength=v)
        )
        self.highlight_hue_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, highlight_tint_hue=v)
        )
        self.highlight_hue_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, highlight_tint_hue=v)
        )
        self.highlight_str_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, highlight_tint_strength=v)
        )
        self.highlight_str_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, highlight_tint_strength=v)
        )

    def sync_ui(self) -> None:
        conf = self.state.config.toning
        is_bw = self.state.config.process.process_mode == ProcessMode.BW

        self.block_signals(True)
        try:
            self.selenium_slider.setValue(conf.selenium_strength)
            self.sepia_slider.setValue(conf.sepia_strength)
            self.shadow_hue_slider.setValue(conf.shadow_tint_hue)
            self.shadow_str_slider.setValue(conf.shadow_tint_strength)
            self.highlight_hue_slider.setValue(conf.highlight_tint_hue)
            self.highlight_str_slider.setValue(conf.highlight_tint_strength)

            self.selenium_slider.setVisible(is_bw)
            self.sepia_slider.setVisible(is_bw)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.selenium_slider,
            self.sepia_slider,
            self.shadow_hue_slider,
            self.shadow_str_slider,
            self.highlight_hue_slider,
            self.highlight_str_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
