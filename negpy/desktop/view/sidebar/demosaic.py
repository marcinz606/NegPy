from PyQt6.QtWidgets import QComboBox, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, hint_label
from negpy.features.process.models import DemosaicMode
from negpy.infrastructure.loaders.helpers import supported_demosaic_modes

_TIP = (
    "<table width='280'><tr><td>"
    "How the sensor's color-filter mosaic is interpolated into RGB. <b>Auto</b> keeps NegPy's own "
    "choice: a fast half-size decode on screen, AHD for export.<br><br>"
    "<b>Linear</b> bilinear, fastest and softest, with color fringes on hard edges. "
    "<b>VNG</b> smooth, low zipper artifacts, slightly soft. "
    "<b>PPG</b> fast, with clean edges. "
    "<b>AHD</b> LibRaw's own default, the balanced choice. "
    "<b>DCB</b> more fine detail than AHD, can ring on texture. "
    "<b>DHT</b> the most detail, and the most willing to turn grain into a maze pattern. "
    "<b>AAHD</b> anti-aliased AHD, softer edges and fewer artifacts.<br><br>"
    "Judge these on grain: the detail-seeking algorithms read film grain as structure."
    "</td></tr></table>"
)


class DemosaicSidebar(BaseSidebar):
    """CFA interpolation, chosen separately for what you see and what you get."""

    def _init_ui(self) -> None:
        conf = self.state.config.process
        modes = [str(m) for m in supported_demosaic_modes()]

        self.preview_combo = QComboBox()
        self.preview_combo.addItems(modes)
        self.preview_combo.setToolTip(_TIP)
        self.export_combo = QComboBox()
        self.export_combo.addItems(modes)
        self.export_combo.setToolTip(_TIP)

        for label, combo in (("Preview", self.preview_combo), ("Export", self.export_combo)):
            row = QHBoxLayout()
            row.addWidget(field_label(label))
            row.addWidget(combo, 1)
            self.layout.addLayout(row)

        self.hint = hint_label(
            "Bayer and X-Trans RAW only — a scanner TIFF or a linear DNG arrives already de-mosaiced. "
            "Auto and Linear are the fastest for the preview."
        )
        self.layout.addWidget(self.hint)

        self.preview_combo.setCurrentText(str(DemosaicMode(conf.demosaic_preview)))
        self.export_combo.setCurrentText(str(DemosaicMode(conf.demosaic_export)))

    def _connect_signals(self) -> None:
        self.preview_combo.currentTextChanged.connect(lambda name: self._on_changed("demosaic_preview", name))
        self.export_combo.currentTextChanged.connect(lambda name: self._on_changed("demosaic_export", name))

    def _on_changed(self, field: str, name: str) -> None:
        # apply_config: source_token carries the preview choice, so changing it decodes again.
        self.update_config_section("process", persist=True, render=True, **{field: DemosaicMode(name)})

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            # Through the enum: an unrecognised stored value reads back as Auto rather than
            # leaving the combo on whatever it showed.
            self.preview_combo.setCurrentText(str(DemosaicMode(conf.demosaic_preview)))
            self.export_combo.setCurrentText(str(DemosaicMode(conf.demosaic_export)))
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.preview_combo, self.export_combo):
            w.blockSignals(blocked)
