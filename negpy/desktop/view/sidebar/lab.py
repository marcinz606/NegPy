from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QHBoxLayout
from negpy.desktop.view.widgets.fold import group_fold, rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.auto_sliders import auto_masking_share, shown_masking, stored_masking
from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, field_label, hint_label, section_subheader
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.features.lab.logic import NEUTRAL_MASKING
from negpy.features.lab.models import SharpenMethod
from negpy.features.process.models import ProcessMode


class LabSidebar(BaseSidebar):
    """
    Panel for color, sharpening, and contrast. Spectral crosstalk lives in the
    Process sidebar (capture-side, negative-density domain).
    """

    def _init_ui(self) -> None:
        conf = self.state.config.lab

        self.color_header = section_subheader(tr("COLOR"))
        self.layout.addWidget(self.color_header)

        self.saturation_slider = CompactSlider(tr("Chroma"), 0.0, 2.0, conf.saturation, has_neutral=True)
        self.layout.addWidget(self.saturation_slider)

        self.skin_protection_slider = CompactSlider(tr("Skin Protection"), 0.0, 1.0, conf.skin_protection)
        self.chroma_denoise_slider = CompactSlider(tr("Chroma Denoise"), 0.0, 5.0, conf.chroma_denoise)
        self.chroma_fold = rail_fold("lab_chroma", self.saturation_slider, self.skin_protection_slider, self.chroma_denoise_slider)
        self.layout.addWidget(self.chroma_fold)

        self.layout.addWidget(section_subheader(tr("DETAIL")))

        self.sharpen_slider = CompactSlider(tr("Sharpening"), 0.0, 1.0, conf.sharpen)
        self.layout.addWidget(self.sharpen_slider)

        method_row = QHBoxLayout()
        method_row.addWidget(field_label(tr("Method")))
        self.sharpen_method_combo = StepCombo()
        self.sharpen_method_combo.addItem(tr("Unsharp Mask"), SharpenMethod.USM.value)
        self.sharpen_method_combo.addItem(tr("Deconvolution"), SharpenMethod.RL.value)
        self.sharpen_method_combo.setCurrentIndex(self.sharpen_method_combo.findData(str(conf.sharpen_method)))
        method_row.addWidget(self.sharpen_method_combo, 1)
        self.auto_masking_btn = self._small_toggle(
            "fa5s.magic",
            "",
            conf.sharpen_auto_masking,
            tr("Auto Masking: sets Masking from the grain of each frame, so grain stays smooth."),
        )
        self.auto_masking_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.auto_masking_btn.setProperty("value_column", True)
        method_row.addWidget(self.auto_masking_btn)
        self.sharpen_radius_slider = CompactSlider(tr("Radius"), 0.5, 3.0, conf.sharpen_radius, unit=" px")
        self.sharpen_masking_slider = CompactSlider(tr("Masking"), 0.0, 1.0, conf.sharpen_masking)
        self.auto_masking_hint = hint_label("")
        self.auto_masking_hint.setVisible(False)
        self.layout.addWidget(
            rail_fold(
                "lab_sharpen",
                self.sharpen_slider,
                method_row,
                self.sharpen_radius_slider,
                self.sharpen_masking_slider,
                self.auto_masking_hint,
            )
        )

        self.clahe_slider = CompactSlider(tr("CLAHE"), 0.0, 1.0, conf.clahe_strength)
        self.layout.addWidget(self.clahe_slider)

        self.glow_slider = CompactSlider(tr("Glow"), 0.0, 1.0, conf.glow_amount)
        self.halation_slider = CompactSlider(tr("Halation"), 0.0, 1.0, conf.halation_strength)
        effects_header, effects_fold = group_fold("lab_effects", tr("Effects"), self.glow_slider, self.halation_slider)
        self.layout.addLayout(effects_header)
        self.layout.addWidget(effects_fold)

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

        self.sharpen_masking_slider.valueChanged.connect(lambda v: self._set_masking(v, persist=False))
        self.sharpen_masking_slider.valueCommitted.connect(lambda v: self._set_masking(v, persist=True))
        self.auto_masking_btn.clicked.connect(
            lambda checked: self.update_config_section("lab", persist=True, readback_metrics=True, sharpen_auto_masking=checked)
        )
        self.controller.image_updated.connect(self._sync_masking)

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

    def _meters(self) -> dict:
        return self.state.auto_meters.get(self.state.current_file_hash or "", {})

    def _sync_masking(self) -> None:
        """With Auto Masking on, Masking shows the measured value plus the offset and resets to the measured value."""
        config, meters = self.state.config, self._meters()
        auto = auto_masking_share(config, meters)
        self.auto_masking_hint.setVisible(auto is not None)
        if auto is not None:
            self.auto_masking_hint.setText(tr("Auto: {value}, from the grain of this view.", value=f"{auto:.2f}"))
        self.sharpen_masking_slider.blockSignals(True)
        self.sharpen_masking_slider.set_default(NEUTRAL_MASKING if auto is None else min(auto, 1.0))
        self.sharpen_masking_slider.setValue(shown_masking(config, meters))
        self.sharpen_masking_slider.blockSignals(False)

    def _set_masking(self, value: float, persist: bool) -> None:
        # A value at the slider's default is a reset: back to the neutral offset exactly.
        if self.sharpen_masking_slider.is_default(value):
            stored = NEUTRAL_MASKING
        else:
            stored = stored_masking(self.state.config, self._meters(), value)
        self.update_config_section("lab", persist=persist, readback_metrics=persist, sharpen_masking=stored)

    def sync_ui(self) -> None:
        conf = self.state.config.lab
        is_bw = self.state.config.process.process_mode == ProcessMode.BW

        self.block_signals(True)
        try:
            self.clahe_slider.setValue(conf.clahe_strength)
            self.sharpen_method_combo.setCurrentIndex(self.sharpen_method_combo.findData(str(conf.sharpen_method)))
            self.sharpen_slider.setValue(conf.sharpen)
            self.sharpen_radius_slider.setValue(conf.sharpen_radius)
            self.auto_masking_btn.setChecked(conf.sharpen_auto_masking)
            self.saturation_slider.setValue(conf.saturation)
            self.skin_protection_slider.setValue(conf.skin_protection)
            self.chroma_denoise_slider.setValue(conf.chroma_denoise)
            self.glow_slider.setValue(conf.glow_amount)
            self.halation_slider.setValue(conf.halation_strength)

            # COLOR is entirely color controls, so hide the header with them in B&W.
            self.color_header.setVisible(not is_bw)
            self.saturation_slider.setVisible(not is_bw)
            self.chroma_fold.content.setVisible(not is_bw)
        finally:
            self.block_signals(False)
        self._sync_masking()

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.sharpen_method_combo,
            self.clahe_slider,
            self.sharpen_slider,
            self.sharpen_radius_slider,
            self.sharpen_masking_slider,
            self.auto_masking_btn,
            self.saturation_slider,
            self.skin_protection_slider,
            self.chroma_denoise_slider,
            self.glow_slider,
            self.halation_slider,
        ]
        for w in widgets:
            w.blockSignals(blocked)
