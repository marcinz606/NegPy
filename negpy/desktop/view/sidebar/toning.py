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
        conf = self.state.config.toning

        self.chemical_header = section_subheader("CHEMICAL TONING")
        self.chemical_header.setToolTip(
            "Toners apply as sequential baths in the order shown — silver toned by an earlier bath is locked to the later ones"
        )
        self.layout.addWidget(self.chemical_header)

        self.selenium_slider = CompactSlider("Selenium", 0.0, 2.0, conf.selenium_strength)
        self.sepia_slider = CompactSlider("Sepia", 0.0, 2.0, conf.sepia_strength)
        self.gold_slider = CompactSlider("Gold", 0.0, 2.0, conf.gold_strength)
        self.gold_slider.setToolTip(
            "Simulates gold toning — cool blue-black on untoned silver, slight Dmax boost; over sepia it shifts the highlights orange-red (B&W only)"
        )
        self.blue_slider = CompactSlider("Iron Blue", 0.0, 2.0, conf.blue_strength)
        self.blue_slider.setToolTip(
            "Simulates iron blue (Prussian blue) toning — blues the image shadows-first and intensifies: deeper navy blacks (B&W only)"
        )
        self.copper_slider = CompactSlider("Copper", 0.0, 2.0, conf.copper_strength)
        self.copper_slider.setToolTip(
            "Simulates copper toning — pink to brick-red shift with the classic Dmax loss: blacks weaken as the bath bleaches (B&W only)"
        )
        self.vanadium_slider = CompactSlider("Vanadium", 0.0, 2.0, conf.vanadium_strength)
        self.vanadium_slider.setToolTip(
            "Simulates vanadium green toning — bleach-then-tone greens the mids and highlights while deep shadows keep their black silver (B&W only)"
        )
        for left, right in (
            (self.selenium_slider, self.sepia_slider),
            (self.gold_slider, self.blue_slider),
            (self.copper_slider, self.vanadium_slider),
        ):
            row = QHBoxLayout()
            row.addWidget(left)
            row.addWidget(right)
            self.layout.addLayout(row)

        self.layout.addWidget(section_subheader("SPLIT TONING"))

        row_sh = QHBoxLayout()
        self.shadow_hue_slider = HueSlider("Shadow Hue", conf.shadow_tint_hue)
        self.shadow_str_slider = CompactSlider("Shadow Strength", 0.0, 1.0, conf.shadow_tint_strength)
        row_sh.addWidget(self.shadow_hue_slider)
        row_sh.addWidget(self.shadow_str_slider)
        self.layout.addLayout(row_sh)

        row_hl = QHBoxLayout()
        self.highlight_hue_slider = HueSlider("Highlight Hue", conf.highlight_tint_hue)
        self.highlight_str_slider = CompactSlider("Highlight Strength", 0.0, 1.0, conf.highlight_tint_strength)
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
        self.gold_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, gold_strength=v)
        )
        self.gold_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, gold_strength=v)
        )
        self.blue_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, blue_strength=v)
        )
        self.blue_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, blue_strength=v)
        )
        self.copper_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, copper_strength=v)
        )
        self.copper_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, copper_strength=v)
        )
        self.vanadium_slider.valueChanged.connect(
            lambda v: self.update_config_section("toning", persist=False, readback_metrics=False, vanadium_strength=v)
        )
        self.vanadium_slider.valueCommitted.connect(
            lambda v: self.update_config_section("toning", persist=True, readback_metrics=True, vanadium_strength=v)
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
            self.gold_slider.setValue(conf.gold_strength)
            self.blue_slider.setValue(conf.blue_strength)
            self.copper_slider.setValue(conf.copper_strength)
            self.vanadium_slider.setValue(conf.vanadium_strength)
            self.shadow_hue_slider.setValue(conf.shadow_tint_hue)
            self.shadow_str_slider.setValue(conf.shadow_tint_strength)
            self.highlight_hue_slider.setValue(conf.highlight_tint_hue)
            self.highlight_str_slider.setValue(conf.highlight_tint_strength)

            self.chemical_header.setVisible(is_bw)
            self.selenium_slider.setVisible(is_bw)
            self.sepia_slider.setVisible(is_bw)
            self.gold_slider.setVisible(is_bw)
            self.blue_slider.setVisible(is_bw)
            self.copper_slider.setVisible(is_bw)
            self.vanadium_slider.setVisible(is_bw)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.selenium_slider,
            self.sepia_slider,
            self.gold_slider,
            self.blue_slider,
            self.copper_slider,
            self.vanadium_slider,
            self.shadow_hue_slider,
            self.shadow_str_slider,
            self.highlight_hue_slider,
            self.highlight_str_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
