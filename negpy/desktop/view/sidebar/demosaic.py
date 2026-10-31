from typing import Optional

from negpy.kernel.system.i18n import tr

from negpy.desktop.settings_catalog import HIGHLIGHT_LEVELS, highlight_bucket
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import raw_shots_details, raw_shots_summary, raw_shots_warning
from negpy.desktop.view.styles.templates import field_row, hint_label, section_subheader, set_hint_kind, wrap_tooltip
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.features.hdr.models import hdr_active
from negpy.features.process.models import DemosaicMode, ProcessMode
from negpy.features.rgbscan.models import is_rgb_triplet
from negpy.infrastructure.loaders.helpers import supported_demosaic_modes
from negpy.infrastructure.loaders.raw_shots import ShotKind, ShotsInfo

_TIP = tr("How the camera's raw data becomes pixels: Auto is fast on screen and uses AHD for export.")

_SHOTS_TIP = tr("A RAW that holds several shots opens one of them, or Pixel Shift combines four shots into measured colors at every pixel.")

_HIGHLIGHT_TIP = tr(
    "Rebuilds blown highlights in a camera RAW: Blend suits sun and sky, and Reconstruct is stronger but can get the color wrong."
)


def _shot_choices(info: ShotsInfo) -> tuple[list[str], list[int]]:
    """Menu labels and stored raw_shot values: Pixel Shift (Auto, 0) where the shots combine,
    then each shot by number."""
    labels = [tr("Shot {n}", n=n) for n in range(1, info.count + 1)]
    values = list(range(1, info.count + 1))
    if info.kind is ShotKind.PIXEL_SHIFT:
        labels.insert(0, tr("Pixel Shift"))
        values.insert(0, 0)
    return labels, values


class DemosaicSidebar(BaseSidebar):
    """How a camera RAW decodes: the shot of a multi-shot RAW, its white balance, CFA
    interpolation chosen separately for what you see and what you get, and a slide's
    highlight reconstruction."""

    def _init_ui(self) -> None:
        conf = self.state.config.process
        modes = [str(m) for m in supported_demosaic_modes()]

        # Shown only for a RAW that holds several shots; the choices come from that file.
        self.shots_header = section_subheader(tr("SHOTS"))
        self.shot_combo = StepCombo.of([], _SHOTS_TIP)
        shots_row = field_row(tr("Shots"), self.shot_combo)
        self.shots_label = shots_row.itemAt(0).widget()
        self.shots_hint = hint_label("")
        self.layout.addWidget(self.shots_header)
        self.layout.addLayout(shots_row)
        self.layout.addWidget(self.shots_hint)
        self._shot_choices_for: Optional[tuple[int, ShotKind]] = None

        self.preview_combo = StepCombo()
        self.preview_combo.addItems(modes)
        self.preview_combo.setToolTip(_TIP)
        self.export_combo = StepCombo()
        self.export_combo.addItems(modes)
        self.export_combo.setToolTip(_TIP)

        self.capture_header = section_subheader(tr("White Balance"))

        self.linear_raw_btn = SegmentedChoice(
            (("fa5s.palette", tr("Camera WB")), ("fa5s.sliders-h", tr("Linear RAW"))),
            tr("Camera WB uses the camera's white balance, and Linear RAW decodes neutral, for narrowband light."),
            data=(False, True),
        )
        self.linear_raw_btn.setCurrentIndex(self.linear_raw_btn.findData(conf.linear_raw))
        self.layout.addWidget(self.capture_header)
        self.layout.addWidget(self.linear_raw_btn)

        # Greyed rather than hidden: the capture setting is sticky, and a hidden sticky setting
        # follows a frame into Transparency unseen.
        self.capture_hint = hint_label("")
        self.capture_hint.setVisible(False)  # text and tooltip are set per film process in sync_ui
        self.layout.addWidget(self.capture_hint)

        self.layout.addWidget(section_subheader(tr("DEMOSAIC")))
        self.layout.addLayout(field_row(tr("Preview"), self.preview_combo))
        self.layout.addLayout(field_row(tr("Export"), self.export_combo))

        # Shown only on a frame these do not reach; the general note lives in the tooltip.
        self.hint = hint_label(tr("This frame is not a Bayer or X-Trans RAW, so it arrives already de-mosaiced."))
        self.hint.setVisible(False)
        self.layout.addWidget(self.hint)
        self.combined_hint = hint_label(tr("Not used for Pixel Shift: every pixel has measured colors."))
        self.combined_hint.setVisible(False)
        self.layout.addWidget(self.combined_hint)

        self.preview_combo.setCurrentText(str(DemosaicMode(conf.demosaic_preview)))
        self.export_combo.setCurrentText(str(DemosaicMode(conf.demosaic_export)))

        self.highlight_header = section_subheader(tr("HIGHLIGHTS"))
        self.layout.addWidget(self.highlight_header)
        self.highlight_combo = StepCombo()
        self.highlight_combo.addItems([label for _level, label in HIGHLIGHT_LEVELS])
        self.highlight_combo.setToolTip(wrap_tooltip(_HIGHLIGHT_TIP))
        self.highlight_combo.setCurrentIndex(highlight_bucket(conf.highlight_reconstruction))
        highlight_row = field_row(tr("Recovery"), self.highlight_combo)
        self.highlight_label = highlight_row.itemAt(0).widget()
        self.layout.addLayout(highlight_row)

        self.highlight_merged_hint = hint_label(tr("Not applied to a merged bracket."))
        self.highlight_merged_hint.setToolTip(
            wrap_tooltip(tr("A merged bracket already recovers highlights from its shorter exposures, so unmerge the frame to use this."))
        )
        self.highlight_merged_hint.setVisible(False)
        self.layout.addWidget(self.highlight_merged_hint)
        self.highlight_combined_hint = hint_label(tr("Not applied to Pixel Shift."))
        self.highlight_combined_hint.setVisible(False)
        self.layout.addWidget(self.highlight_combined_hint)

    def _connect_signals(self) -> None:
        self.preview_combo.currentTextChanged.connect(lambda name: self._on_changed("demosaic_preview", name))
        self.export_combo.currentTextChanged.connect(lambda name: self._on_changed("demosaic_export", name))
        self.highlight_combo.currentIndexChanged.connect(self._on_highlight_reconstruction_changed)
        self.linear_raw_btn.currentChanged.connect(self._on_linear_raw_changed)
        self.shot_combo.currentIndexChanged.connect(self._on_shot_changed)

    def _on_shot_changed(self, _index: int) -> None:
        # raw_shot is a meter input and part of source_token: the edit decodes again and meters again.
        value = self.shot_combo.currentData()
        if value is not None:
            self.controller.set_roll_default("demosaic", raw_shot=int(value))

    def _current_shots(self) -> Optional[ShotsInfo]:
        info = self.state.preview_raw_shots
        if info is None or self.state.preview_raw_shots_path != self.state.current_file_path:
            return None
        return info

    def _sync_shots(self, raw_shot: int) -> Optional[ShotsInfo]:
        info = self._current_shots()
        for widget in (self.shots_header, self.shots_label, self.shot_combo, self.shots_hint):
            widget.setVisible(info is not None)
        if info is None:
            return None
        if self._shot_choices_for != (info.count, info.kind):
            labels, values = _shot_choices(info)
            self.shot_combo.set_choices(labels, values)
            self._shot_choices_for = (info.count, info.kind)
        index = self.shot_combo.findData(raw_shot)
        self.shot_combo.setCurrentIndex(index if index >= 0 else self.shot_combo.findData(1 if info.kind is ShotKind.SHOTS else 0))
        warning = raw_shots_warning(info)
        self.shots_hint.setText(warning or raw_shots_summary(info))
        set_hint_kind(self.shots_hint, "warning" if warning else "muted")
        details = raw_shots_details(info)
        self.shots_hint.setToolTip(wrap_tooltip(details) if details else "")
        return info

    def _on_linear_raw_changed(self, _index: int) -> None:
        # linear_raw switches use_camera_wb, so it is a source change: set_roll_default's
        # apply_config re-decodes and suppresses the bounds analysis over the stale buffer.
        self.controller.set_roll_default("demosaic", linear_raw=bool(self.linear_raw_btn.currentData()))

    def _on_changed(self, field: str, name: str) -> None:
        # apply_config (inside set_roll_default): source_token carries the preview
        # choice, so changing it decodes again.
        self.controller.set_roll_default("demosaic", **{field: DemosaicMode(name)})

    def _on_highlight_reconstruction_changed(self, bucket: int) -> None:
        level, _label = HIGHLIGHT_LEVELS[bucket]
        self.controller.set_roll_default("demosaic", highlight_reconstruction=level)

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            # Through the enum: an unrecognised stored value reads back as Auto rather than
            # leaving the combo on whatever it showed.
            self.linear_raw_btn.setCurrentIndex(self.linear_raw_btn.findData(conf.linear_raw))
            # Linear RAW is inert on the *transfer*, where the camera matrix folds the as-shot
            # multipliers back in (with Positive on it decides the decode again, so it stays live
            # there), and on an RGB-scan triplet, where a narrowband exposure has no full-spectrum
            # scene for a WB gain to describe in the first place — every exposure decodes neutral
            # regardless. Scanning setup describes a negative rig.
            from negpy.features.process.path import RenderPath, render_path

            e6 = conf.process_mode == ProcessMode.E6
            triplet = is_rgb_triplet(self.state.config.rgbscan)
            self.linear_raw_btn.setEnabled(render_path(conf) is not RenderPath.TRANSFER and not triplet)
            self.capture_hint.setVisible(e6 or triplet)
            if e6:
                self.capture_hint.setText(
                    tr("Scanning setup is not used for slides.")
                    if conf.positive_source
                    else tr("Not applied to an as-captured transparency.")
                )
                self.capture_hint.setToolTip(
                    wrap_tooltip(
                        tr("With Positive on, Linear RAW would undo Positive, so leave it off.")
                        if conf.positive_source
                        else tr("Linear RAW has no effect on a slide.")
                    )
                )
            elif triplet:
                self.capture_hint.setText(tr("Linear RAW is locked for a Trichrome triplet."))
                self.capture_hint.setToolTip(
                    wrap_tooltip(tr("Each Trichrome exposure holds one color only, so camera white balance has nothing to correct."))
                )

            self.preview_combo.setCurrentText(str(DemosaicMode(conf.demosaic_preview)))
            self.export_combo.setCurrentText(str(DemosaicMode(conf.demosaic_export)))

            # The combined shots carry no mosaic, so the demosaic choices and LibRaw's
            # reconstruction have nothing to act on there.
            shots = self._sync_shots(conf.raw_shot)
            combined = shots is not None and shots.shot == 0
            self.preview_combo.setEnabled(not combined)
            self.export_combo.setEnabled(not combined)
            self.combined_hint.setVisible(combined)

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
            self.highlight_combo.setEnabled(self.state.preview_cam_xyz is not None and not merged and not combined)
            self.highlight_combo.setCurrentIndex(highlight_bucket(conf.highlight_reconstruction))
            self.highlight_merged_hint.setVisible(is_e6 and merged)
            self.highlight_combined_hint.setVisible(is_e6 and combined and not merged)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.linear_raw_btn, self.preview_combo, self.export_combo, self.highlight_combo, self.shot_combo):
            w.blockSignals(blocked)
