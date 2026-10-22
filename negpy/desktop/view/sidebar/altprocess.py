from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label
from negpy.desktop.view.widgets.sliders import CompactSlider, SliderGroup
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.features.altprocess.models import AltProcess, Sensitizer
from negpy.features.process.models import ProcessMode

_SENSITIZER_LABELS = {
    Sensitizer.CLASSIC: tr("Classic (Herschel)"),
    Sensitizer.NEW: tr("New (Ware)"),
}


class AltProcessSidebar(BaseSidebar):
    """Alternative printing processes, one at a time: lith, cyanotype or Sabattier. Lith's paper
    comes from the Exposure panel, the cyanotype takes its color from the sensitizer, and a
    Sabattier print is plain silver."""

    def _init_ui(self) -> None:
        conf = self.state.config.altproc

        self._modes = (
            (AltProcess.NONE, "fa5s.ban", tr("None"), tr("Print normally, with no alternative process.")),
            (
                AltProcess.LITH,
                "fa5s.fire",
                tr("Lith"),
                tr("Lith print: creamy warm highlights and hard, deep blacks, in the paper's color, with only Selenium and Gold toners."),
            ),
            (
                AltProcess.CYANOTYPE,
                "fa5s.sun",
                tr("Cyanotype"),
                tr("Cyanotype: a blue print with green highlights, made in UV light, toned with Bleach and Tannin in place of toners."),
            ),
            (
                AltProcess.SABATTIER,
                "fa5s.adjust",
                tr("Sabattier"),
                tr("Sabattier: a flash of light during development reverses the light tones and draws light lines along edges."),
            ),
        )
        self.mode_btn = SegmentedChoice(tuple((icon, label) for _m, icon, label, _t in self._modes), tr("Alternative printing process"))
        for i, (*_rest, tip) in enumerate(self._modes):
            self.mode_btn.set_choice_tooltip(i, tip)
        self.layout.addWidget(self.mode_btn)
        self.color_hint = hint_label(tr("Black and white film only"))
        self.layout.addWidget(self.color_hint)

        self.lith_block = self._build_lith(conf)
        self.cyano_block = self._build_cyanotype(conf)
        self.sabattier_block = self._build_sabattier(conf)
        self.layout.addWidget(self.lith_block)
        self.layout.addWidget(self.cyano_block)
        self.layout.addWidget(self.sabattier_block)

        self.layout.addStretch()

    def _build_lith(self, conf) -> QWidget:
        block = QWidget()
        col = QVBoxLayout(block)
        col.setContentsMargins(0, 0, 0, 0)

        self.exposure_slider = CompactSlider(tr("Exposure"), 0.0, 5.0, conf.lith_exposure, step=0.1, unit=" st")
        self.exposure_slider.setToolTip(tr("Extra exposure: more gives warmer, more colorful highlights and softer tones."))
        self.snatch_slider = CompactSlider(tr("Snatch Point"), 0.0, 1.0, conf.lith_snatch)
        self.snatch_slider.setToolTip(
            tr("When the print leaves the developer: higher gives deeper, colder blacks, and lower stays light and warm.")
        )
        self.abruptness_slider = CompactSlider(tr("Abruptness"), 0.0, 1.0, conf.lith_abruptness)
        self.abruptness_slider.setToolTip(tr("How suddenly the shadows turn black."))

        col.addWidget(self.exposure_slider)
        col.addWidget(self.snatch_slider)
        col.addWidget(SliderGroup(self.abruptness_slider))
        return block

    def _build_cyanotype(self, conf) -> QWidget:
        block = QWidget()
        col = QVBoxLayout(block)
        col.setContentsMargins(0, 0, 0, 0)

        sens_row = QHBoxLayout()
        self.sensitizer_combo = StepCombo()
        for s in Sensitizer:
            self.sensitizer_combo.addItem(_SENSITIZER_LABELS[s], s.value)
        self._select_sensitizer(conf.cyano_sensitizer)
        self.sensitizer_combo.setToolTip(
            tr("Classic gives a light blue with green highlights, and New gives deeper, cleaner blues and more range.")
        )
        sens_row.addWidget(field_label(tr("Sensitizer")))
        sens_row.addWidget(self.sensitizer_combo, stretch=1)
        col.addLayout(sens_row)

        self.cyano_exposure_slider = CompactSlider(tr("Exposure"), -2.0, 4.0, conf.cyano_exposure, step=0.1, unit=" st")
        self.cyano_exposure_slider.setToolTip(tr("UV exposure time: more makes the print bluer, and less makes it paler."))
        self.cyano_scale_slider = CompactSlider(tr("Exposure Scale"), 0.8, 2.8, conf.cyano_scale, step=0.05)
        self.cyano_scale_slider.setToolTip(
            tr("Contrast: a shorter scale gives more contrast but can lose detail in shadows and highlights.")
        )
        self.cyano_bleach_slider = CompactSlider(tr("Bleach"), 0.0, 0.5, conf.cyano_bleach)
        self.cyano_bleach_slider.setToolTip(tr("Washing soda bleach: removes blue, highlights first."))
        self.cyano_tannin_slider = CompactSlider(tr("Tannin"), 0.0, 0.5, conf.cyano_tannin)
        self.cyano_tannin_slider.setToolTip(tr("Tea or tannin toner: turns bleached areas brown, so use Bleach first for a full brown."))

        col.addWidget(self.cyano_exposure_slider)
        col.addWidget(self.cyano_scale_slider)

        col.addWidget(self.cyano_bleach_slider)
        col.addWidget(self.cyano_tannin_slider)
        return block

    def _build_sabattier(self, conf) -> QWidget:
        block = QWidget()
        col = QVBoxLayout(block)
        col.setContentsMargins(0, 0, 0, 0)

        self.sabattier_strength_slider = CompactSlider(tr("Strength"), 0.0, 2.0, conf.sabattier_strength, step=0.05)
        self.sabattier_strength_slider.setToolTip(tr("Length of the second exposure: above 1 the light tones reverse."))
        self.sabattier_reexposure_slider = CompactSlider(tr("Re-exposure"), 0.1, 0.9, conf.sabattier_reexposure, step=0.05)
        self.sabattier_reexposure_slider.setToolTip(tr("Where the reversal starts: tones darker than this stay as they are."))
        self.sabattier_agitation_slider = CompactSlider(tr("Agitation"), 0.0, 1.0, conf.sabattier_agitation, step=0.05)
        self.sabattier_agitation_slider.setToolTip(tr("Tray rocking after the flash: less rocking draws wider light lines along edges."))

        col.addWidget(self.sabattier_strength_slider)
        col.addWidget(self.sabattier_reexposure_slider)
        col.addWidget(self.sabattier_agitation_slider)
        return block

    def _select_sensitizer(self, sensitizer) -> None:
        """Items store the plain str; findData(StrEnum) misses and would snap the
        combo back to the first entry on every re-sync."""
        self.sensitizer_combo.setCurrentIndex(max(self.sensitizer_combo.findData(str(sensitizer)), 0))

    def _connect_signals(self) -> None:
        self.mode_btn.currentChanged.connect(lambda i: self.update_config_section("altproc", persist=True, alt_process=self._modes[i][0]))

        self.sensitizer_combo.currentIndexChanged.connect(
            lambda i: self.update_config_section("altproc", persist=True, cyano_sensitizer=Sensitizer(self.sensitizer_combo.itemData(i)))
        )

        for slider, field in (
            (self.exposure_slider, "lith_exposure"),
            (self.snatch_slider, "lith_snatch"),
            (self.abruptness_slider, "lith_abruptness"),
            (self.cyano_exposure_slider, "cyano_exposure"),
            (self.cyano_scale_slider, "cyano_scale"),
            (self.cyano_bleach_slider, "cyano_bleach"),
            (self.cyano_tannin_slider, "cyano_tannin"),
            (self.sabattier_strength_slider, "sabattier_strength"),
            (self.sabattier_reexposure_slider, "sabattier_reexposure"),
            (self.sabattier_agitation_slider, "sabattier_agitation"),
        ):
            slider.valueChanged.connect(
                lambda v, f=field: self.update_config_section("altproc", persist=False, readback_metrics=False, **{f: v})
            )
            slider.valueCommitted.connect(
                lambda v, f=field: self.update_config_section("altproc", persist=True, readback_metrics=True, **{f: v})
            )

    def sync_ui(self) -> None:
        conf = self.state.config.altproc
        is_bw = self.state.config.process.process_mode == ProcessMode.BW
        mode = conf.alt_process if is_bw else AltProcess.NONE

        self.block_signals(True)
        try:
            self.mode_btn.setCurrentIndex(next(i for i, (m, *_rest) in enumerate(self._modes) if m == conf.alt_process))
            self.mode_btn.setEnabled(is_bw)
            self.color_hint.setVisible(not is_bw)

            self.exposure_slider.setValue(conf.lith_exposure)
            self.snatch_slider.setValue(conf.lith_snatch)
            self.abruptness_slider.setValue(conf.lith_abruptness)
            self._select_sensitizer(conf.cyano_sensitizer)
            self.cyano_exposure_slider.setValue(conf.cyano_exposure)
            self.cyano_scale_slider.setValue(conf.cyano_scale)
            self.cyano_bleach_slider.setValue(conf.cyano_bleach)
            self.cyano_tannin_slider.setValue(conf.cyano_tannin)
            self.sabattier_strength_slider.setValue(conf.sabattier_strength)
            self.sabattier_reexposure_slider.setValue(conf.sabattier_reexposure)
            self.sabattier_agitation_slider.setValue(conf.sabattier_agitation)

            self.lith_block.setVisible(mode == AltProcess.LITH)
            self.cyano_block.setVisible(mode == AltProcess.CYANOTYPE)
            self.sabattier_block.setVisible(mode == AltProcess.SABATTIER)
        finally:
            self.block_signals(False)

    def _sliders(self) -> list:
        return [
            self.exposure_slider,
            self.snatch_slider,
            self.abruptness_slider,
            self.cyano_exposure_slider,
            self.cyano_scale_slider,
            self.cyano_bleach_slider,
            self.cyano_tannin_slider,
            self.sabattier_strength_slider,
            self.sabattier_reexposure_slider,
            self.sabattier_agitation_slider,
        ]

    def block_signals(self, blocked: bool) -> None:
        for w in [self.mode_btn, self.sensitizer_combo, *self._sliders()]:
            w.blockSignals(blocked)
