from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, wrap_tooltip
from negpy.domain.models import CROP_RATIO_CHOICES, canonical_crop_ratio
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.geometry.models import AutocropMode


class AutocropSidebar(BaseSidebar):
    """
    Roll-wide auto crop: what the frame detector looks for, and the batch run over the
    whole roll. The rect it finds stays each frame's own, on the Geometry card.
    """

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = field_label(text)
        lbl.setFixedWidth(42)
        return lbl

    def _init_ui(self) -> None:
        conf = self.state.config.geometry

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(self._field_label("Ratio"))
        self.ratio_combo = QComboBox()
        # One entry per shape (see CROP_RATIO_CHOICES). The crop tool auto-orients to match the
        # current drag, so a separate portrait entry for every ratio would duplicate the same
        # shape twice.
        self.ratio_combo.addItems([r.value for r in CROP_RATIO_CHOICES])
        self.ratio_combo.setCurrentText(canonical_crop_ratio(conf.autocrop_ratio))
        self.ratio_combo.setPlaceholderText("Select Ratio…")
        self.ratio_combo.setToolTip(wrap_tooltip("Aspect ratio the auto crop and the crop tool snap to"))
        ratio_row.addWidget(self.ratio_combo, 1)

        self.detect_ratio_btn = self._icon_action("fa5s.crosshairs", "Detect closest aspect ratio from the film frame")
        ratio_row.addWidget(self.detect_ratio_btn)

        self.layout.addLayout(ratio_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._field_label("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Image only", AutocropMode.IMAGE.value)
        self.mode_combo.addItem("Film edge", AutocropMode.FILM.value)
        self._select_mode(conf.autocrop_mode)
        self.mode_combo.setToolTip(wrap_tooltip("Auto crop target: exposed image only, or full film including rebate/sprockets"))
        mode_row.addWidget(self.mode_combo, 1)
        self.layout.addLayout(mode_row)

        self.offset_slider = CompactSlider(
            "Crop Offset",
            -5.0,
            100.0,
            float(conf.autocrop_offset),
            step=1.0,
            precision=1,
            unit=" px",
        )
        self.offset_slider.setToolTip(wrap_tooltip("How far inside the detected edge the crop lands"))
        self.rebate_trim_slider = CompactSlider(
            "Rebate Trim",
            0.0,
            150.0,
            conf.autocrop_rebate_trim * 100.0,
            step=5.0,
            precision=1,
            unit="%",
        )
        self.rebate_trim_slider.setToolTip(
            wrap_tooltip(
                "How far into the detected rebate auto crop cuts: 0% stops at the film edge, "
                "100% lands on the image edge, above 100% bites in to clear a white border"
            )
        )
        self.rebate_trim_slider.setEnabled(conf.autocrop_mode == AutocropMode.IMAGE)

        trim_row = QHBoxLayout()
        trim_row.addWidget(self.offset_slider, 1)
        trim_row.addWidget(self.rebate_trim_slider, 1)
        self.layout.addLayout(trim_row)

        self.auto_crop_all_btn = self._labeled_action(
            "fa5s.layer-group",
            " Batch Autocrop",
            "Analyze all visible landscape frames as one roll. Confident frames calibrate weak ones; "
            "manual and ambiguous crops are preserved. Runs before Roll Analysis.",
        )
        self.auto_crop_all_btn.setEnabled(conf.autocrop_mode == AutocropMode.IMAGE)
        self.layout.addWidget(self.auto_crop_all_btn)

    def _connect_signals(self) -> None:
        self.ratio_combo.currentTextChanged.connect(self.controller.set_crop_ratio)
        self.detect_ratio_btn.clicked.connect(self.controller.detect_aspect_ratio)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.auto_crop_all_btn.clicked.connect(self.controller.request_batch_auto_crop)

        self.offset_slider.valueChanged.connect(
            lambda v: self.controller.set_roll_default("autocrop", persist=False, readback_metrics=False, autocrop_offset=int(v))
        )
        self.offset_slider.valueCommitted.connect(lambda v: self.controller.set_roll_default("autocrop", autocrop_offset=int(v)))
        self.rebate_trim_slider.valueChanged.connect(
            lambda v: self.controller.set_roll_default("autocrop", persist=False, readback_metrics=False, autocrop_rebate_trim=v / 100.0)
        )
        self.rebate_trim_slider.valueCommitted.connect(
            lambda v: self.controller.set_roll_default("autocrop", autocrop_rebate_trim=v / 100.0)
        )

    def _on_mode_changed(self, idx: int) -> None:
        mode = self.mode_combo.itemData(idx)
        self._sync_mode_enabled(mode)
        self.controller.set_roll_default("autocrop", autocrop_mode=mode)

    def _select_mode(self, mode) -> None:
        """findData matches on the stored str, and a config that has never been edited
        still holds the AutocropMode itself, which does not compare equal through a
        QVariant."""
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(str(mode)))

    def _sync_mode_enabled(self, mode) -> None:
        self.auto_crop_all_btn.setEnabled(mode == AutocropMode.IMAGE)
        self.rebate_trim_slider.setEnabled(mode == AutocropMode.IMAGE)

    def sync_ui(self) -> None:
        conf = self.state.config.geometry
        self.block_signals(True)
        try:
            self.ratio_combo.setCurrentText(canonical_crop_ratio(conf.autocrop_ratio))
            self._select_mode(conf.autocrop_mode)
            self.offset_slider.setValue(float(conf.autocrop_offset))
            self.rebate_trim_slider.setValue(conf.autocrop_rebate_trim * 100.0)
            self._sync_mode_enabled(conf.autocrop_mode)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.ratio_combo,
            self.detect_ratio_btn,
            self.mode_combo,
            self.offset_slider,
            self.rebate_trim_slider,
            self.auto_crop_all_btn,
        ):
            w.blockSignals(blocked)
