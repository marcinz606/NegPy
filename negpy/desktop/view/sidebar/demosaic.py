from PyQt6.QtWidgets import QComboBox, QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import FIELD_LABEL_WIDTH, field_label, hint_label, section_subheader, wrap_tooltip
from negpy.features.hdr.models import hdr_active
from negpy.features.process.logic import VALID_HIGHLIGHT_LEVELS
from negpy.features.process.models import DemosaicMode, ProcessMode
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
    "Judge these on grain: the detail-seeking algorithms read film grain as structure. "
    "Auto and Linear are the fastest for the preview.<br><br>"
    "Bayer and X-Trans RAW only: a scanner TIFF or a linear DNG arrives already de-mosaiced."
    "</td></tr></table>"
)

# Highlight Reconstruction dropdown: rawpy's HighlightMode, collapsed to the three settings
# worth choosing between (see effective_highlight_reconstruction) — Reconstruct pinned to
# libraw's own default level rather than exposing all seven numbered levels.
_HIGHLIGHT_LEVELS = (
    (0, "Off"),
    (2, "Blend"),
    (5, "Reconstruct"),
)

_HIGHLIGHT_TIP = (
    "Camera RAW only, and only useful when a highlight actually clipped.<br><br>"
    "<b>Off</b> (default) — a blown highlight stays flat white, or magenta if one channel "
    "clipped first.<br><br>"
    "<b>Blend</b> — a plausible neutral color from the unclipped channels. Best for a "
    "near-neutral highlight: sun, sky, chrome, glass.<br><br>"
    "<b>Reconstruct</b> — libraw's more aggressive default. Can miscolor a highlight that "
    "was actually a saturated light source, since a clipped channel alone can't tell the "
    "two apart."
)


def _highlight_bucket(level: int) -> int:
    """Which of the three entries a stored value belongs under. Off and Blend are exact;
    any other valid level is some Reconstruct level (3-9), so it buckets there.

    A value outside `VALID_HIGHLIGHT_LEVELS` (a hand-edited sidecar) buckets to Off,
    matching `effective_highlight_reconstruction`'s own resolution — the panel must never
    show Reconstruct armed while the decode actually clips.
    """
    if level not in VALID_HIGHLIGHT_LEVELS:
        return 0
    if level == 2:
        return 1
    return 2 if level else 0


class DemosaicSidebar(BaseSidebar):
    """How a camera RAW decodes: CFA interpolation, chosen separately for what you see
    and what you get, and a slide's highlight reconstruction."""

    def _init_ui(self) -> None:
        conf = self.state.config.process
        modes = [str(m) for m in supported_demosaic_modes()]

        self.preview_combo = QComboBox()
        self.preview_combo.addItems(modes)
        self.preview_combo.setToolTip(_TIP)
        self.export_combo = QComboBox()
        self.export_combo.addItems(modes)
        self.export_combo.setToolTip(_TIP)

        self.layout.addWidget(section_subheader("DEMOSAIC"))
        for label, combo in (("Preview", self.preview_combo), ("Export", self.export_combo)):
            row = QHBoxLayout()
            lbl = field_label(label)
            lbl.setFixedWidth(FIELD_LABEL_WIDTH)
            row.addWidget(lbl)
            row.addWidget(combo, 1)
            self.layout.addLayout(row)

        # Shown only on a frame these do not reach; the general note lives in the tooltip.
        self.hint = hint_label("This frame is not a Bayer or X-Trans RAW, so it arrives already de-mosaiced.")
        self.hint.setVisible(False)
        self.layout.addWidget(self.hint)

        self.preview_combo.setCurrentText(str(DemosaicMode(conf.demosaic_preview)))
        self.export_combo.setCurrentText(str(DemosaicMode(conf.demosaic_export)))

        self.highlight_header = section_subheader("HIGHLIGHTS")
        self.layout.addWidget(self.highlight_header)
        highlight_row = QHBoxLayout()
        self.highlight_label = field_label("Recovery")
        self.highlight_label.setFixedWidth(FIELD_LABEL_WIDTH)
        highlight_row.addWidget(self.highlight_label)
        self.highlight_combo = QComboBox()
        self.highlight_combo.addItems([label for _level, label in _HIGHLIGHT_LEVELS])
        self.highlight_combo.setToolTip(wrap_tooltip(_HIGHLIGHT_TIP))
        self.highlight_combo.setCurrentIndex(_highlight_bucket(conf.highlight_reconstruction))
        highlight_row.addWidget(self.highlight_combo, 1)
        self.layout.addLayout(highlight_row)

        self.highlight_merged_hint = hint_label("Not applied to a merged bracket.")
        self.highlight_merged_hint.setToolTip(
            wrap_tooltip(
                "A reconstructed pixel no longer reads near the sensor ceiling, so the merge's "
                "own highlight recovery would trust a per-frame guess as real signal and blend "
                "inconsistent guesses across frames. A bracket already recovers a genuine "
                "highlight from a shorter, unclipped exposure, which reconstruction's guess "
                "cannot improve on. Unmerge the frame if you need it."
            )
        )
        self.highlight_merged_hint.setVisible(False)
        self.layout.addWidget(self.highlight_merged_hint)

    def _connect_signals(self) -> None:
        self.preview_combo.currentTextChanged.connect(lambda name: self._on_changed("demosaic_preview", name))
        self.export_combo.currentTextChanged.connect(lambda name: self._on_changed("demosaic_export", name))
        self.highlight_combo.currentIndexChanged.connect(self._on_highlight_reconstruction_changed)

    def _on_changed(self, field: str, name: str) -> None:
        # apply_config (inside set_roll_default): source_token carries the preview
        # choice, so changing it decodes again.
        self.controller.set_roll_default("demosaic", **{field: DemosaicMode(name)})

    def _on_highlight_reconstruction_changed(self, bucket: int) -> None:
        level, _label = _HIGHLIGHT_LEVELS[bucket]
        self.controller.set_roll_default("demosaic", highlight_reconstruction=level)

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            # Through the enum: an unrecognised stored value reads back as Auto rather than
            # leaving the combo on whatever it showed.
            self.preview_combo.setCurrentText(str(DemosaicMode(conf.demosaic_preview)))
            self.export_combo.setCurrentText(str(DemosaicMode(conf.demosaic_export)))

            # Reconstruction only means anything against a slide's own blown highlights (see
            # effective_highlight_reconstruction), so it hides off Slide. Greyed instead of hidden
            # when the source has no camera matrix (a scanner TIFF, JPEG, or other
            # already-rendered file), and on a merge, which the hint explains.
            is_e6 = conf.process_mode == ProcessMode.E6
            merged = hdr_active(self.state.config.hdr)
            # No camera matrix: not a camera RAW decode, so nothing here is demosaiced.
            self.hint.setVisible(bool(self.state.current_file_hash) and self.state.preview_cam_xyz is None)
            self.highlight_header.setVisible(is_e6)
            self.highlight_label.setVisible(is_e6)
            self.highlight_combo.setVisible(is_e6)
            self.highlight_combo.setEnabled(self.state.preview_cam_xyz is not None and not merged)
            self.highlight_combo.setCurrentIndex(_highlight_bucket(conf.highlight_reconstruction))
            self.highlight_merged_hint.setVisible(is_e6 and merged)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.preview_combo, self.export_combo, self.highlight_combo):
            w.blockSignals(blocked)
