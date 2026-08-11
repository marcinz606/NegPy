from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.process.models import ProcessMode


class LithSidebar(BaseSidebar):
    """
    Lith (infectious) development. The paper comes from the Exposure panel.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.lith

        self.enable_btn = self._small_toggle(
            "fa5s.fire",
            "Lith Printing",
            conf.lith_enabled,
            "Develop the print in dilute lith developer: creamy warm highlights and a "
            "near-vertical drop into hard, sooty blacks.\n"
            "The colour — peach highlights through an olive transition to neutral blacks — "
            "comes from the paper chosen in Exposure.\n"
            "Only Selenium and Gold do anything distinctive on a lith print, so the other toners "
            "are disabled while this is on (B&W only)",
        )
        self.layout.addWidget(self.enable_btn)

        self.exposure_slider = CompactSlider("Exposure", 0.0, 5.0, conf.lith_exposure, step=0.1, unit=" stops")
        self.exposure_slider.setToolTip(
            "Print over-exposure. Lith printing runs on two to four stops more light than a normal print: "
            "more light gives warmer, more colourful highlights and softer gradation"
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

        self.layout.addWidget(self.exposure_slider)

        row_dev = QHBoxLayout()
        row_dev.addWidget(self.snatch_slider)
        row_dev.addWidget(self.abruptness_slider)
        self.layout.addLayout(row_dev)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(lambda v: self.update_config_section("lith", persist=True, lith_enabled=v))
        for slider, field in (
            (self.exposure_slider, "lith_exposure"),
            (self.snatch_slider, "lith_snatch"),
            (self.abruptness_slider, "lith_abruptness"),
        ):
            slider.valueChanged.connect(
                lambda v, f=field: self.update_config_section("lith", persist=False, readback_metrics=False, **{f: v})
            )
            slider.valueCommitted.connect(
                lambda v, f=field: self.update_config_section("lith", persist=True, readback_metrics=True, **{f: v})
            )

    def sync_ui(self) -> None:
        conf = self.state.config.lith
        is_bw = self.state.config.process.process_mode == ProcessMode.BW

        self.block_signals(True)
        try:
            self.enable_btn.setChecked(conf.lith_enabled)
            self.exposure_slider.setValue(conf.lith_exposure)
            self.snatch_slider.setValue(conf.lith_snatch)
            self.abruptness_slider.setValue(conf.lith_abruptness)

            self.enable_btn.setEnabled(is_bw)
            for w in self._sliders():
                w.setEnabled(is_bw and conf.lith_enabled)
        finally:
            self.block_signals(False)

    def _sliders(self) -> list:
        return [
            self.exposure_slider,
            self.snatch_slider,
            self.abruptness_slider,
        ]

    def block_signals(self, blocked: bool) -> None:
        for w in [self.enable_btn, *self._sliders()]:
            w.blockSignals(blocked)
