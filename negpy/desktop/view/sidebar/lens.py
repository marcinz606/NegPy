from dataclasses import replace

from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, section_subheader, wrap_tooltip
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.lens.models import LensMetadata
from negpy.infrastructure.loaders.lens_metadata import read_lens_metadata
from negpy.services.rendering.lens import metadata_lens_corrections


class LensSidebar(BaseSidebar):
    """
    Roll-wide correction of the scanning lens: radial distortion by hand, or the
    distortion and lateral CA the source file carries.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.geometry

        self.distortion_slider = CompactSlider(
            "Distortion Correction", -0.10, 0.10, conf.distortion_k1, step=0.001, precision=1000, has_neutral=True
        )
        # Nothing derives the readout's decimals from `precision`, so a 0.001 step needs both.
        self.distortion_slider.spin.setDecimals(3)
        self.distortion_slider.setToolTip(
            wrap_tooltip(
                "Radial lens distortion. Positive corrects barrel, negative pincushion. Use the film rebate as a straight reference."
            )
        )
        self.layout.addWidget(self.distortion_slider)

        self.layout.addWidget(section_subheader("EMBEDDED PROFILE"))

        self.metadata_distortion_btn = self._labeled_toggle(
            "fa5s.camera",
            " Distortion",
            conf.lens_distortion_from_metadata,
            "Apply embedded scanning-lens distortion correction. Replaces manual distortion.",
        )
        self.metadata_ca_btn = self._labeled_toggle(
            "fa5s.camera",
            " CA",
            conf.lens_ca_from_metadata,
            "Apply embedded lateral chromatic aberration correction. Can be used with manual distortion.",
        )
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.metadata_distortion_btn, 1)
        btn_row.addWidget(self.metadata_ca_btn, 1)
        self.layout.addLayout(btn_row)

        self.lens_hint = hint_label("")
        self.lens_hint.setWordWrap(True)
        self.layout.addWidget(self.lens_hint)

    def _connect_signals(self) -> None:
        self.metadata_distortion_btn.toggled.connect(lambda enabled: self._set_metadata_lens("lens_distortion_from_metadata", enabled))
        self.metadata_ca_btn.toggled.connect(lambda enabled: self._set_metadata_lens("lens_ca_from_metadata", enabled))
        self.distortion_slider.valueChanged.connect(lambda _v: self.controller.show_rotation_guide())
        self.distortion_slider.valueChanged.connect(
            lambda v: self.controller.set_roll_default("lens", persist=False, readback_metrics=False, distortion_k1=v)
        )
        self.distortion_slider.valueCommitted.connect(lambda v: self.controller.set_roll_default("lens", distortion_k1=v))

    def _set_metadata_lens(self, field: str, enabled: bool) -> None:
        button = self.metadata_distortion_btn if field == "lens_distortion_from_metadata" else self.metadata_ca_btn
        if enabled and not button.isEnabled():
            return
        self.controller.set_roll_default("lens", **{field: enabled})

    def _sync_metadata_lens(self) -> None:
        config = self.state.config
        lens = read_lens_metadata(self.state.current_file_path)
        if self.state.preview_lens_path == self.state.current_file_path and self.state.preview_lens is not None:
            lens = self.state.preview_lens
        requested = replace(config, geometry=replace(config.geometry, lens_distortion_from_metadata=True, lens_ca_from_metadata=True))
        if not metadata_lens_corrections(requested):
            lens = LensMetadata(reason="Embedded lens correction is unavailable for composites.")
        if self.state.has_ir:
            lens = LensMetadata(reason="Embedded lens correction is unavailable for RGB+IR sources.")
        for button, enabled, available in (
            (self.metadata_distortion_btn, config.geometry.lens_distortion_from_metadata, lens.distortion),
            (self.metadata_ca_btn, config.geometry.lens_ca_from_metadata, lens.ca),
        ):
            button.setChecked(enabled)
            button.setEnabled(available or enabled)
            button.edited_dot.set_active(enabled)
        self.lens_hint.setText(lens.description if lens.available else f"Unavailable: {lens.reason}")
        self.distortion_slider.setEnabled(not config.geometry.lens_distortion_from_metadata)

    def sync_ui(self) -> None:
        self.block_signals(True)
        try:
            self.distortion_slider.setValue(self.state.config.geometry.distortion_k1)
            self._sync_metadata_lens()
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.distortion_slider, self.metadata_distortion_btn, self.metadata_ca_btn):
            w.blockSignals(blocked)
