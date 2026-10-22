from negpy.desktop.view.styles.templates import add_choice_items, select_choice
from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.session import ToolMode
from negpy.desktop.view.canvas.crop_guides import GUIDE_LABELS, ORIENTATION_COUNT, CropGuide
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import clearable_tool_toggle, field_row, wrap_tooltip
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.domain.models import CROP_RATIO_CHOICES, canonical_crop_ratio
from negpy.desktop.view.widgets.fold import group_fold, rail_fold
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.geometry.logic import has_manual_crop
from negpy.features.geometry.models import AutocropMode


class AutocropSidebar(BaseSidebar):
    """The crop of this frame: drawn by hand or found by Auto Crop, and what the frame detector looks for."""

    def _init_ui(self) -> None:
        conf = self.state.config.geometry

        self.manual_crop_btn = clearable_tool_toggle(
            "fa5s.crop-alt", tr("Draw"), tr("Draw the crop on the picture."), tr("Clear the crop you drew.")
        )
        auto_tip = tr("Auto Crop: find the edges of the picture and crop to them.")
        self.auto_frame_btn = self._small_toggle("fa5s.magic", tr("Auto Crop"), conf.crop_from_auto, auto_tip)
        tool_row = QHBoxLayout()
        tool_row.addWidget(self.manual_crop_btn, 1)
        tool_row.addWidget(self.auto_frame_btn, 1)
        self.layout.addLayout(tool_row)

        self.ratio_combo = StepCombo()
        # One entry per shape (see CROP_RATIO_CHOICES). The crop tool auto-orients to match the
        # current drag, so a separate portrait entry for every ratio would duplicate the same
        # shape twice.
        add_choice_items(self.ratio_combo, [r.value for r in CROP_RATIO_CHOICES])
        select_choice(self.ratio_combo, canonical_crop_ratio(conf.autocrop_ratio))
        self.ratio_combo.setPlaceholderText(tr("Select Ratio…"))
        self.ratio_combo.setToolTip(wrap_tooltip(tr("Shape of the crop, for Auto Crop and the crop tool.")))
        self.detect_ratio_btn = self._icon_action("fa5s.crosshairs", tr("Detect closest aspect ratio from the film frame"))
        self.layout.addLayout(field_row(tr("Ratio"), self.ratio_combo, self.detect_ratio_btn))

        self.guide_combo = StepCombo()
        for guide, label in GUIDE_LABELS.items():
            self.guide_combo.addItem(label, guide.value)
        self.guide_combo.setCurrentIndex(self.guide_combo.findData(self.state.crop_guide))
        self.guide_combo.setToolTip(
            tooltip_with_shortcut(tr("Composition guide shown in the crop tool"), ("crop_guide_next", "crop_guide_orient"))
        )
        self.guide_orient_btn = self._icon_action(
            "fa5s.redo", tooltip_with_shortcut(tr("Rotate the guide orientation (spiral, triangles)"), "crop_guide_orient")
        )
        self._sync_guide_orient_btn()
        self.layout.addLayout(field_row(tr("Guide"), self.guide_combo, self.guide_orient_btn))

        self.mode_combo = StepCombo()
        self.mode_combo.addItem(tr("Image only"), AutocropMode.IMAGE.value)
        self.mode_combo.addItem(tr("Film edge"), AutocropMode.FILM.value)
        self._select_mode(conf.autocrop_mode)
        self.mode_combo.setToolTip(wrap_tooltip(tr("Crop to the picture only, or keep the whole film with its edges and holes.")))
        mode_row = field_row(tr("Mode"), self.mode_combo)

        self.offset_slider = CompactSlider(
            tr("Crop Offset"),
            -5.0,
            100.0,
            float(conf.autocrop_offset),
            step=1.0,
            precision=1,
            unit=" px",
        )
        self.offset_slider.setToolTip(wrap_tooltip(tr("How far inside the detected edge the crop lands")))
        self.rebate_trim_slider = CompactSlider(
            tr("Rebate Trim"),
            0.0,
            150.0,
            conf.autocrop_rebate_trim * 100.0,
            step=5.0,
            precision=1,
            unit="%",
        )
        self.rebate_trim_slider.setToolTip(
            wrap_tooltip(tr("How far the crop cuts into the film border: 0% at the film edge, 100% at the picture edge."))
        )
        self.rebate_trim_slider.setEnabled(conf.autocrop_mode == AutocropMode.IMAGE)

        square_tip = tr("Square First: straighten the frame to its film edges before Auto Crop crops it.")
        self.square_btn = self._small_toggle("fa5s.ruler-combined", tr("Square First"), conf.autocrop_square, square_tip)
        all_tip = tr("Crop all frames together: clear frames help unclear ones, and crops that you drew stay.")
        self.auto_crop_all_btn = self._labeled_action("fa5s.layer-group", tr("Crop All Frames"), all_tip)
        self.auto_crop_all_btn.setEnabled(conf.autocrop_mode == AutocropMode.IMAGE)
        run_row = QHBoxLayout()
        run_row.addWidget(self.square_btn, 1)
        run_row.addWidget(self.auto_crop_all_btn, 1)

        header, self.settings_fold = group_fold(
            "autocrop_settings",
            tr("Auto Crop Settings"),
            mode_row,
            rail_fold("autocrop_offset", mode_row, self.offset_slider, self.rebate_trim_slider),
            run_row,
        )
        self.layout.addLayout(header)
        self.layout.addWidget(self.settings_fold)

    def cycle_guide(self) -> None:
        self.guide_combo.setCurrentIndex((self.guide_combo.currentIndex() + 1) % self.guide_combo.count())

    def _sync_guide_orient_btn(self) -> None:
        guide = self.guide_combo.currentData()
        self.guide_orient_btn.setEnabled(ORIENTATION_COUNT.get(CropGuide(guide), 1) > 1 if guide else False)

    def _connect_signals(self) -> None:
        self.manual_crop_btn.toggled.connect(
            lambda checked: self.controller.set_active_tool(ToolMode.CROP_MANUAL if checked else ToolMode.NONE)
        )
        self.manual_crop_btn.clear_btn.clicked.connect(self.controller.reset_crop)
        self.auto_frame_btn.toggled.connect(self._on_auto_frame_toggled)
        self.ratio_combo.currentIndexChanged.connect(lambda _: self.controller.set_crop_ratio(self.ratio_combo.currentData()))
        self.detect_ratio_btn.clicked.connect(self.controller.detect_aspect_ratio)
        self.guide_combo.currentIndexChanged.connect(lambda _i: self.controller.set_crop_guide(self.guide_combo.currentData()))
        self.guide_combo.currentIndexChanged.connect(lambda _i: self._sync_guide_orient_btn())
        self.guide_orient_btn.clicked.connect(self.controller.cycle_crop_guide_orientation)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.square_btn.toggled.connect(lambda on: self.controller.set_crop_settings(autocrop_square=bool(on)))
        self.auto_crop_all_btn.clicked.connect(self.controller.request_batch_auto_crop)

        self.offset_slider.valueChanged.connect(
            lambda v: self.controller.set_crop_settings(persist=False, readback_metrics=False, autocrop_offset=int(v))
        )
        self.offset_slider.valueCommitted.connect(lambda v: self.controller.set_crop_settings(autocrop_offset=int(v)))
        self.rebate_trim_slider.valueChanged.connect(
            lambda v: self.controller.set_crop_settings(persist=False, readback_metrics=False, autocrop_rebate_trim=v / 100.0)
        )
        self.rebate_trim_slider.valueCommitted.connect(lambda v: self.controller.set_crop_settings(autocrop_rebate_trim=v / 100.0))

    def _on_auto_frame_toggled(self, checked: bool) -> None:
        if checked:
            self.controller.apply_auto_crop()
        else:
            self.controller.reset_crop()

    def _on_mode_changed(self, idx: int) -> None:
        mode = self.mode_combo.itemData(idx)
        self._sync_mode_enabled(mode)
        self.controller.set_crop_settings(autocrop_mode=mode)

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
            self.manual_crop_btn.setChecked(self.state.active_tool == ToolMode.CROP_MANUAL)
            self.manual_crop_btn.clear_btn.setVisible(has_manual_crop(conf))
            select_choice(self.ratio_combo, canonical_crop_ratio(conf.autocrop_ratio))
            self.guide_combo.setCurrentIndex(self.guide_combo.findData(self.state.crop_guide))
            self._sync_guide_orient_btn()
            self._select_mode(conf.autocrop_mode)
            self.offset_slider.setValue(float(conf.autocrop_offset))
            self.rebate_trim_slider.setValue(conf.autocrop_rebate_trim * 100.0)
            self._sync_mode_enabled(conf.autocrop_mode)
            self.auto_frame_btn.setChecked(conf.crop_from_auto)
            self.auto_frame_btn.edited_dot.set_active(conf.crop_from_auto)
            self.square_btn.setChecked(conf.autocrop_square)
            self.square_btn.edited_dot.set_active(conf.autocrop_square)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.manual_crop_btn,
            self.ratio_combo,
            self.detect_ratio_btn,
            self.guide_combo,
            self.guide_orient_btn,
            self.mode_combo,
            self.offset_slider,
            self.rebate_trim_slider,
            self.auto_frame_btn,
            self.square_btn,
            self.auto_crop_all_btn,
        ):
            w.blockSignals(blocked)
