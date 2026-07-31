from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.features.lab.models import SharpenMethod
from negpy.features.process.models import ProcessMode


class LabSidebar(BaseSidebar):
    """
    Panel for colour, sharpening, and contrast. Spectral crosstalk lives in the
    Process sidebar (capture-side, negative-density domain).
    """

    def _init_ui(self) -> None:
        conf = self.state.config.lab

        self.color_header = section_subheader("COLOUR")
        self.layout.addWidget(self.color_header)

        self.saturation_slider = CompactSlider("Chroma", 0.0, 2.0, conf.saturation, has_neutral=True)
        self.layout.addWidget(self.saturation_slider)

        row1 = QHBoxLayout()
        self.skin_protection_slider = CompactSlider("Skin Protection", 0.0, 1.0, conf.skin_protection)
        self.chroma_denoise_slider = CompactSlider("Chroma Denoise", 0.0, 5.0, conf.chroma_denoise)
        row1.addWidget(self.skin_protection_slider)
        row1.addWidget(self.chroma_denoise_slider)
        self.layout.addLayout(row1)

        self.layout.addWidget(section_subheader("SHARPEN"))

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.sharpen_method_combo = QComboBox()
        self.sharpen_method_combo.addItem("Unsharp Mask", SharpenMethod.USM.value)
        self.sharpen_method_combo.addItem("Deconvolution", SharpenMethod.RL.value)
        self.sharpen_method_combo.setCurrentIndex(self.sharpen_method_combo.findData(str(conf.sharpen_method)))
        method_row.addWidget(self.sharpen_method_combo, 1)
        self.layout.addLayout(method_row)

        self.sharpen_slider = CompactSlider("Sharpening", 0.0, 1.0, conf.sharpen)
        self.layout.addWidget(self.sharpen_slider)

        row_sharpen = QHBoxLayout()
        self.sharpen_radius_slider = CompactSlider("Radius (px)", 0.5, 3.0, conf.sharpen_radius)
        self.sharpen_masking_slider = CompactSlider("Masking", 0.0, 1.0, conf.sharpen_masking)
        row_sharpen.addWidget(self.sharpen_radius_slider)
        row_sharpen.addWidget(self.sharpen_masking_slider)
        self.layout.addLayout(row_sharpen)

        self.layout.addWidget(section_subheader("DETAIL"))

        self.clahe_slider = CompactSlider("CLAHE", 0.0, 1.0, conf.clahe_strength)
        self.layout.addWidget(self.clahe_slider)

        self.layout.addWidget(section_subheader("EFFECTS"))

        row4 = QHBoxLayout()
        self.glow_slider = CompactSlider("Glow", 0.0, 1.0, conf.glow_amount)
        self.halation_slider = CompactSlider("Halation", 0.0, 1.0, conf.halation_strength)
        row4.addWidget(self.glow_slider)
        row4.addWidget(self.halation_slider)
        self.layout.addLayout(row4)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.sharpen_method_combo.currentIndexChanged.connect(
            lambda idx: self.update_config_section("lab", persist=True, sharpen_method=self.sharpen_method_combo.itemData(idx))
        )

        self.clahe_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, clahe_strength=v)
        )
        self.clahe_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, clahe_strength=v)
        )

        self.sharpen_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, sharpen=v)
        )
        self.sharpen_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, sharpen=v)
        )

        self.sharpen_radius_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, sharpen_radius=v)
        )
        self.sharpen_radius_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, sharpen_radius=v)
        )

        self.sharpen_masking_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, sharpen_masking=v)
        )
        self.sharpen_masking_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, sharpen_masking=v)
        )

        self.saturation_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, saturation=v)
        )
        self.saturation_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, saturation=v)
        )

        self.skin_protection_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, skin_protection=v)
        )
        self.skin_protection_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, skin_protection=v)
        )

        self.chroma_denoise_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, chroma_denoise=v)
        )
        self.chroma_denoise_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, chroma_denoise=v)
        )

        self.glow_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, glow_amount=v)
        )
        self.glow_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, glow_amount=v)
        )

        self.halation_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, halation_strength=v)
        )
        self.halation_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, halation_strength=v)
        )

    def sync_ui(self) -> None:
        conf = self.state.config.lab
        is_bw = self.state.config.process.process_mode == ProcessMode.BW

        self.block_signals(True)
        try:
            self.clahe_slider.setValue(conf.clahe_strength)
            self.sharpen_method_combo.setCurrentIndex(self.sharpen_method_combo.findData(str(conf.sharpen_method)))
            self.sharpen_slider.setValue(conf.sharpen)
            self.sharpen_radius_slider.setValue(conf.sharpen_radius)
            self.sharpen_masking_slider.setValue(conf.sharpen_masking)
            self.saturation_slider.setValue(conf.saturation)
            self.skin_protection_slider.setValue(conf.skin_protection)
            self.chroma_denoise_slider.setValue(conf.chroma_denoise)
            self.glow_slider.setValue(conf.glow_amount)
            self.halation_slider.setValue(conf.halation_strength)

            # COLOR is entirely colour controls — hide the header with them in B&W.
            self.color_header.setVisible(not is_bw)
            self.saturation_slider.setVisible(not is_bw)
            self.skin_protection_slider.setVisible(not is_bw)
            self.chroma_denoise_slider.setVisible(not is_bw)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.sharpen_method_combo,
            self.clahe_slider,
            self.sharpen_slider,
            self.sharpen_radius_slider,
            self.sharpen_masking_slider,
            self.saturation_slider,
            self.skin_protection_slider,
            self.chroma_denoise_slider,
            self.glow_slider,
            self.halation_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
