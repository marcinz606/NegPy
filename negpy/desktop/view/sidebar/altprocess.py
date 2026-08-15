from PyQt6.QtWidgets import QButtonGroup, QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.altprocess.models import AltProcess, Sensitizer
from negpy.features.process.models import ProcessMode

_SENSITIZER_LABELS = {
    Sensitizer.CLASSIC: "Classic (Herschel)",
    Sensitizer.NEW: "New (Ware)",
}


class AltProcessSidebar(BaseSidebar):
    """
    Alternative printing processes. One at a time — a print is either lith-developed
    or a cyanotype, never both. Lith's paper comes from the Exposure panel; the
    cyanotype is on rag paper and takes its color from the sensitiser.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.altproc

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons = {}
        mode_row = QHBoxLayout()
        for mode, icon, label, tip in (
            (AltProcess.NONE, "fa5s.ban", "None", "Print normally — no alternative process."),
            (
                AltProcess.LITH,
                "fa5s.fire",
                "Lith",
                "Develop the print in dilute lith developer: creamy warm highlights and a "
                "near-vertical drop into hard, sooty blacks.\n"
                "The color — peach highlights through an olive transition to neutral blacks — "
                "comes from the paper chosen in Exposure.\n"
                "Only Selenium and Gold do anything distinctive on a lith print, so the other toners "
                "are disabled while this is on",
            ),
            (
                AltProcess.CYANOTYPE,
                "fa5s.sun",
                "Cyanotype",
                "Contact-print the negative in UV onto iron-sensitised rag paper. The image "
                "substance is Prussian blue, so the print never goes black — it goes blue, with "
                "green highlights where the residual yellow sensitiser mixes in.\n"
                "There is no silver in a cyanotype, so every chemical toner is disabled while "
                "this is on; use Bleach and Tannin instead",
            ),
        ):
            btn = self._tool_toggle(icon, label, tip)
            btn.setChecked(conf.alt_process == mode)
            self.mode_group.addButton(btn)
            self.mode_buttons[mode] = btn
            mode_row.addWidget(btn, 1)
        self.layout.addLayout(mode_row)

        self.lith_block = self._build_lith(conf)
        self.cyano_block = self._build_cyanotype(conf)
        self.layout.addWidget(self.lith_block)
        self.layout.addWidget(self.cyano_block)

        self.layout.addStretch()

    def _build_lith(self, conf) -> QWidget:
        block = QWidget()
        col = QVBoxLayout(block)
        col.setContentsMargins(0, 0, 0, 0)

        self.exposure_slider = CompactSlider("Exposure", 0.0, 5.0, conf.lith_exposure, step=0.1, unit=" stops")
        self.exposure_slider.setToolTip(
            "Print over-exposure. Lith printing runs on two to four stops more light than a normal print: "
            "more light gives warmer, more colorful highlights and softer gradation"
        )
        self.snatch_slider = CompactSlider("Snatch Point", 0.0, 1.0, conf.lith_snatch)
        self.snatch_slider.setToolTip(
            "How long the print stays in the developer before it is snatched out. "
            "Later (higher) drops the knee further up the tonal scale: deeper, colder blacks and a wider "
            "band of undifferentiated shadow. Earlier keeps the print high-key, warm and weak in the blacks"
        )
        self.abruptness_slider = CompactSlider("Abruptness", 0.0, 1.0, conf.lith_abruptness)
        self.abruptness_slider.setToolTip(
            "How abruptly the shadows go black — a seasoned, low-sulphite developer makes the knee a step. "
            "Low leaves a gentle roll into the blacks"
        )

        col.addWidget(self.exposure_slider)
        row_dev = QHBoxLayout()
        row_dev.addWidget(self.snatch_slider)
        row_dev.addWidget(self.abruptness_slider)
        col.addLayout(row_dev)
        return block

    def _build_cyanotype(self, conf) -> QWidget:
        block = QWidget()
        col = QVBoxLayout(block)
        col.setContentsMargins(0, 0, 0, 0)

        sens_row = QHBoxLayout()
        self.sensitizer_combo = QComboBox()
        for s in Sensitizer:
            self.sensitizer_combo.addItem(_SENSITIZER_LABELS[s], s.value)
        self._select_sensitizer(conf.cyano_sensitizer)
        self.sensitizer_combo.setToolTip(
            "Sensitiser. Classic is Herschel's ammonium ferric citrate: it loses much of its "
            "pigment in the wash, so it tops out around a red-channel density of 1.0 and keeps a "
            "strong green highlight stain. New is Ware's ferric oxalate — deeper, cleaner and "
            "able to hold a far longer scale"
        )
        sens_row.addWidget(field_label("Sensitiser"))
        sens_row.addWidget(self.sensitizer_combo, stretch=1)
        col.addLayout(sens_row)

        self.cyano_exposure_slider = CompactSlider("Exposure", -2.0, 4.0, conf.cyano_exposure, step=0.1, unit=" stops")
        self.cyano_exposure_slider.setToolTip(
            "Time under the UV source. More light drives more of the scale into Prussian blue; less leaves the print pale and high-key"
        )
        self.cyano_scale_slider = CompactSlider("Exposure Scale", 0.8, 2.8, conf.cyano_scale, step=0.05)
        self.cyano_scale_slider.setToolTip(
            "The negative density range the sensitiser can print, in log D — the contrast control. "
            "Ware measures about 1.0 to 1.2 for the traditional formula against 2.4 for the new one, "
            "and his Simple Cyanotype ships as three variants at 1.8, 2.3 and 2.7. "
            "Short scale means a contrastier print that clips both ends of a normal negative"
        )
        self.cyano_bleach_slider = CompactSlider("Bleach", 0.0, 0.5, conf.cyano_bleach)
        self.cyano_bleach_slider.setToolTip(
            "Washing soda. Strips Prussian blue out of the print, highlights first — take it far "
            "enough and only the deepest shadows keep any pigment"
        )
        self.cyano_tannin_slider = CompactSlider("Tannin", 0.0, 0.5, conf.cyano_tannin)
        self.cyano_tannin_slider.setToolTip(
            "Tea, coffee or tannic acid. Re-develops the bleached iron as a brown iron tannate that "
            "covers more than the pigment it replaced, so the print goes browner and a little deeper. "
            "Bleach first for a full brown, on its own for a split blue-brown"
        )

        row_exp = QHBoxLayout()
        row_exp.addWidget(self.cyano_exposure_slider)
        row_exp.addWidget(self.cyano_scale_slider)
        col.addLayout(row_exp)

        row_tone = QHBoxLayout()
        row_tone.addWidget(self.cyano_bleach_slider)
        row_tone.addWidget(self.cyano_tannin_slider)
        col.addLayout(row_tone)
        return block

    def _select_sensitizer(self, sensitizer) -> None:
        """Items store the plain str; findData(StrEnum) misses and would snap the
        combo back to the first entry on every re-sync."""
        self.sensitizer_combo.setCurrentIndex(max(self.sensitizer_combo.findData(str(sensitizer)), 0))

    def _connect_signals(self) -> None:
        for mode, btn in self.mode_buttons.items():
            btn.clicked.connect(lambda _c, m=mode: self.update_config_section("altproc", persist=True, alt_process=m))

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
            for m, btn in self.mode_buttons.items():
                btn.setChecked(conf.alt_process == m)
                btn.setEnabled(is_bw)

            self.exposure_slider.setValue(conf.lith_exposure)
            self.snatch_slider.setValue(conf.lith_snatch)
            self.abruptness_slider.setValue(conf.lith_abruptness)
            self._select_sensitizer(conf.cyano_sensitizer)
            self.cyano_exposure_slider.setValue(conf.cyano_exposure)
            self.cyano_scale_slider.setValue(conf.cyano_scale)
            self.cyano_bleach_slider.setValue(conf.cyano_bleach)
            self.cyano_tannin_slider.setValue(conf.cyano_tannin)

            self.lith_block.setVisible(mode == AltProcess.LITH)
            self.cyano_block.setVisible(mode == AltProcess.CYANOTYPE)
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
        ]

    def block_signals(self, blocked: bool) -> None:
        for w in [*self.mode_buttons.values(), self.sensitizer_combo, *self._sliders()]:
            w.blockSignals(blocked)
