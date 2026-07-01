from PyQt6.QtWidgets import QComboBox, QHBoxLayout
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, section_subheader
from negpy.features.process.models import ProcessMode
from negpy.services.assets.crosstalk import CrosstalkProfiles


class LabSidebar(BaseSidebar):
    """
    Panel for color separation, sharpening, and contrast.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.lab

        self.layout.addWidget(section_subheader("COLOR"))

        matrix_row = QHBoxLayout()
        self.crosstalk_label = field_label("Crosstalk Matrix")
        self.crosstalk_combo = QComboBox()
        self.crosstalk_combo.addItems(CrosstalkProfiles.list_profiles())
        self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)
        self.crosstalk_combo.setToolTip(
            "Crosstalk matrix used by the Crosstalk slider. 'Default' is built-in; "
            "drop custom .toml matrices in the NegPy/crosstalk folder (see docs/CROSSTALK.md)"
        )
        matrix_row.addWidget(self.crosstalk_label, 1)
        matrix_row.addWidget(self.crosstalk_combo, 1)
        self.layout.addLayout(matrix_row)

        row1 = QHBoxLayout()
        self.separation_slider = CompactSlider("Crosstalk", 1.0, 2.0, conf.color_separation)
        self.separation_slider.setToolTip("Spectral crosstalk correction: unmixes R, G, B channels for richer color separation")
        self.chroma_denoise_slider = CompactSlider("Denoise", 0.0, 5.0, conf.chroma_denoise)
        self.chroma_denoise_slider.setToolTip("Chroma noise reduction in Lab space — smooths color noise while preserving luminance grain")
        row1.addWidget(self.separation_slider)
        row1.addWidget(self.chroma_denoise_slider)
        self.layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.saturation_slider = CompactSlider("Saturation", 0.0, 2.0, conf.saturation, has_neutral=True)
        self.vibrance_slider = CompactSlider("Vibrance", 0.0, 2.0, conf.vibrance, has_neutral=True)
        self.vibrance_slider.setToolTip("Selectively boosts muted colors while protecting already-saturated tones")
        row2.addWidget(self.saturation_slider)
        row2.addWidget(self.vibrance_slider)
        self.layout.addLayout(row2)

        self.layout.addWidget(section_subheader("DETAIL"))

        row3 = QHBoxLayout()
        self.clahe_slider = CompactSlider("CLAHE", 0.0, 1.0, conf.clahe_strength)
        self.clahe_slider.setToolTip("Contrast Limited Adaptive Histogram Equalization — local contrast enhancement")
        self.sharpen_slider = CompactSlider("Sharpening", 0.0, 1.0, conf.sharpen)
        row3.addWidget(self.clahe_slider)
        row3.addWidget(self.sharpen_slider)
        self.layout.addLayout(row3)

        self.layout.addWidget(section_subheader("EFFECTS"))

        row4 = QHBoxLayout()
        self.glow_slider = CompactSlider("Glow", 0.0, 1.0, conf.glow_amount)
        self.halation_slider = CompactSlider("Halation", 0.0, 1.0, conf.halation_strength)
        row4.addWidget(self.glow_slider)
        row4.addWidget(self.halation_slider)
        self.layout.addLayout(row4)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.crosstalk_combo.currentTextChanged.connect(self._on_crosstalk_changed)

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

        self.saturation_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, saturation=v)
        )
        self.saturation_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, saturation=v)
        )

        self.vibrance_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, vibrance=v)
        )
        self.vibrance_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, vibrance=v)
        )

        self.separation_slider.valueChanged.connect(
            lambda v: self.update_config_section("lab", persist=False, readback_metrics=False, color_separation=v)
        )
        self.separation_slider.valueCommitted.connect(
            lambda v: self.update_config_section("lab", persist=True, readback_metrics=True, color_separation=v)
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

    def _on_crosstalk_changed(self, name: str) -> None:
        matrix = CrosstalkProfiles.get_matrix(name)
        self.update_config_section("lab", persist=True, crosstalk_profile=name, crosstalk_matrix=matrix)

    def sync_ui(self) -> None:
        conf = self.state.config.lab
        is_bw = self.state.config.process.process_mode == ProcessMode.BW

        self.block_signals(True)
        try:
            profiles = CrosstalkProfiles.list_profiles()
            if profiles != [self.crosstalk_combo.itemText(i) for i in range(self.crosstalk_combo.count())]:
                self.crosstalk_combo.clear()
                self.crosstalk_combo.addItems(profiles)
            self.crosstalk_combo.setCurrentText(conf.crosstalk_profile)

            self.clahe_slider.setValue(conf.clahe_strength)
            self.sharpen_slider.setValue(conf.sharpen)
            self.saturation_slider.setValue(conf.saturation)
            self.vibrance_slider.setValue(conf.vibrance)
            self.separation_slider.setValue(conf.color_separation)
            self.chroma_denoise_slider.setValue(conf.chroma_denoise)
            self.glow_slider.setValue(conf.glow_amount)
            self.halation_slider.setValue(conf.halation_strength)

            self.separation_slider.setVisible(not is_bw)
            self.crosstalk_combo.setVisible(not is_bw)
            self.crosstalk_label.setVisible(not is_bw)
            self.saturation_slider.setVisible(not is_bw)
            self.vibrance_slider.setVisible(not is_bw)
            self.chroma_denoise_slider.setVisible(not is_bw)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.clahe_slider,
            self.sharpen_slider,
            self.saturation_slider,
            self.vibrance_slider,
            self.separation_slider,
            self.crosstalk_combo,
            self.chroma_denoise_slider,
            self.glow_slider,
            self.halation_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
