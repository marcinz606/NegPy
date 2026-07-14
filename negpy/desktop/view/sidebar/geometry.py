from dataclasses import replace

import qtawesome as qta
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)

from negpy.desktop.session import ToolMode
from negpy.desktop.view.canvas.crop_guides import GUIDE_LABELS, ORIENTATION_COUNT, CropGuide
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import EditedDot, default_button_height, field_label, section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.domain.models import AspectRatio
from negpy.features.geometry.models import FINE_ROTATION_LIMIT, AutocropMode
from negpy.features.process.models import invalidate_local_bounds


class CropToolButton(QPushButton):
    """Checkable button with a small corner dot indicating an active crop."""

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._dot = EditedDot(self)

    def set_crop_active(self, active: bool) -> None:
        self._dot.set_active(active)


class GeometrySidebar(BaseSidebar):
    """
    Panel for cropping and fine adjustments.
    """

    @staticmethod
    def _field_label(text: str) -> QLabel:
        # fixed width so the Ratio/Guide combos left-align
        lbl = field_label(text)
        lbl.setFixedWidth(42)
        return lbl

    def _init_ui(self) -> None:
        conf = self.state.config.geometry

        self.layout.addWidget(section_subheader("CROP"))

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(self._field_label("Ratio"))
        self.ratio_combo = QComboBox()
        # Filter out 'Original' as it's not a crop ratio (usually 'Free' is used for no constraint)
        ratios = [r.value for r in AspectRatio if r != AspectRatio.ORIGINAL]
        self.ratio_combo.addItems(ratios)
        self.ratio_combo.setCurrentText(conf.autocrop_ratio)
        self.ratio_combo.setPlaceholderText("Select Ratio...")
        ratio_row.addWidget(self.ratio_combo, 1)

        self.detect_ratio_btn = self._icon_action("fa5s.crosshairs", "Detect closest aspect ratio from the film frame")
        ratio_row.addWidget(self.detect_ratio_btn)

        self.layout.addLayout(ratio_row)

        btn_row = QHBoxLayout()
        self.manual_crop_btn = CropToolButton(" Crop")
        self.manual_crop_btn.setCheckable(True)
        self.manual_crop_btn.setIcon(qta.icon("fa5s.crop-alt", color=THEME.text_primary, color_on="#FFFFFF"))

        self.clear_crop_btn = QPushButton(" Reset")
        self.clear_crop_btn.setIcon(qta.icon("fa5s.undo", color=THEME.text_primary))
        self.clear_crop_btn.setToolTip("Reset crop: clear the manual crop and disable auto crop")

        btn_row.addWidget(self.manual_crop_btn, 1)
        btn_row.addWidget(self.clear_crop_btn, 1)
        self.layout.addLayout(btn_row)

        guide_row = QHBoxLayout()
        guide_row.addWidget(self._field_label("Guide"))
        self.guide_combo = QComboBox()
        for guide, label in GUIDE_LABELS.items():
            self.guide_combo.addItem(label, guide.value)
        self.guide_combo.setCurrentIndex(self.guide_combo.findData(self.state.crop_guide))
        self.guide_combo.setToolTip(
            tooltip_with_shortcut("Composition guide shown in the crop tool", ("crop_guide_next", "crop_guide_orient"))
        )
        guide_row.addWidget(self.guide_combo, 1)
        self.guide_orient_btn = self._icon_action(
            "fa5s.redo", tooltip_with_shortcut("Rotate the guide orientation (spiral, triangles)", "crop_guide_orient")
        )
        guide_row.addWidget(self.guide_orient_btn)
        self._sync_guide_orient_btn()
        self.layout.addLayout(guide_row)

        self.layout.addWidget(section_subheader("AUTO CROP"))

        # Auto crop toggle + mode: crop to exposed image, or keep full film incl. rebate
        auto_row = QHBoxLayout()
        self.reset_crop_btn = CropToolButton(" Auto")
        self.reset_crop_btn.setCheckable(True)
        self.reset_crop_btn.setIcon(qta.icon("fa5s.magic", color=THEME.text_primary, color_on="#FFFFFF", color_disabled=THEME.text_muted))
        self.reset_crop_btn.setFixedHeight(default_button_height())

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Image only", AutocropMode.IMAGE.value)
        self.mode_combo.addItem("Film edge", AutocropMode.FILM.value)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(conf.autocrop_mode))
        self.mode_combo.setToolTip("Auto crop target: exposed image only, or full film including rebate/sprockets")
        self.reset_crop_btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.mode_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        auto_row.addWidget(self.reset_crop_btn, 1)
        auto_row.addWidget(self.mode_combo, 1)
        self.layout.addLayout(auto_row)

        self.offset_slider = CompactSlider(
            "Crop Offset",
            -5.0,
            100.0,
            float(conf.autocrop_offset),
            step=1.0,
            precision=1,
            unit=" px",
        )
        self.layout.addWidget(self.offset_slider)

        self.layout.addWidget(section_subheader("ALIGNMENT"))

        align_row = QHBoxLayout()
        self.straighten_btn = self._tool_toggle("fa5s.ruler", "", "")
        self.straighten_btn.setFixedWidth(36)

        # Slider shows the photographer's convention — positive = clockwise on screen.
        # Internally geometry.fine_rotation keeps the cv2/warp convention (positive =
        # counter-clockwise, flip-independent because flips apply before fine rotation),
        # so saved edits keep their meaning: display = -stored at this boundary.
        self.fine_rot_slider = CompactSlider("Fine Rotation", -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT, -conf.fine_rotation, unit="°")
        align_row.addWidget(self.fine_rot_slider, 1)
        align_row.addWidget(self.straighten_btn, 0)
        self.layout.addLayout(align_row)

    def cycle_guide(self) -> None:
        self.guide_combo.setCurrentIndex((self.guide_combo.currentIndex() + 1) % self.guide_combo.count())

    def _sync_guide_orient_btn(self) -> None:
        guide = self.guide_combo.currentData()
        self.guide_orient_btn.setEnabled(ORIENTATION_COUNT.get(CropGuide(guide), 1) > 1 if guide else False)

    def _connect_signals(self) -> None:
        self.guide_combo.currentIndexChanged.connect(lambda _i: self.controller.set_crop_guide(self.guide_combo.currentData()))
        self.guide_combo.currentIndexChanged.connect(lambda _i: self._sync_guide_orient_btn())
        self.guide_orient_btn.clicked.connect(self.controller.cycle_crop_guide_orientation)
        self.ratio_combo.currentTextChanged.connect(self._on_ratio_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.detect_ratio_btn.clicked.connect(self.controller.detect_aspect_ratio)
        self.manual_crop_btn.toggled.connect(self._on_manual_crop_toggled)
        self.clear_crop_btn.clicked.connect(self.controller.reset_crop)
        self.reset_crop_btn.toggled.connect(self._on_auto_crop_toggled)

        self.offset_slider.valueChanged.connect(
            lambda v: self.update_config_section("geometry", render=True, persist=False, readback_metrics=False, autocrop_offset=int(v))
        )
        self.offset_slider.valueCommitted.connect(self._on_offset_committed)

        self.straighten_btn.toggled.connect(self._on_straighten_toggled)

        # Display convention is CW-positive; negate crossing into the stored convention.
        self.fine_rot_slider.valueChanged.connect(
            lambda v: self.update_config_section("geometry", render=True, persist=False, readback_metrics=False, fine_rotation=-v)
        )
        self.fine_rot_slider.valueChanged.connect(lambda _v: self.controller.show_rotation_guide())
        self.fine_rot_slider.valueCommitted.connect(
            lambda v: self.update_config_section("geometry", render=True, persist=True, readback_metrics=True, fine_rotation=-v)
        )

    def _on_ratio_changed(self, ratio: str) -> None:
        new_config = replace(
            self.state.config,
            geometry=replace(self.state.config.geometry, autocrop_ratio=ratio),
            process=replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process)),
        )
        self.controller.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def _on_mode_changed(self, idx: int) -> None:
        new_config = replace(
            self.state.config,
            geometry=replace(self.state.config.geometry, autocrop_mode=self.mode_combo.itemData(idx)),
            process=replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process)),
        )
        self.controller.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def _on_offset_committed(self, v: float) -> None:
        new_config = replace(
            self.state.config,
            geometry=replace(self.state.config.geometry, autocrop_offset=int(v)),
            process=replace(self.state.config.process, **invalidate_local_bounds(self.state.config.process)),
        )
        self.controller.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def _on_manual_crop_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.CROP_MANUAL if checked else ToolMode.NONE)

    def _on_straighten_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.STRAIGHTEN if checked else ToolMode.NONE)

    def _on_auto_crop_toggled(self, checked: bool) -> None:
        if checked:
            self.controller.apply_auto_crop()
        else:
            self.controller.reset_crop()

    def sync_ui(self) -> None:
        conf = self.state.config.geometry

        self.block_signals(True)
        try:
            self.guide_combo.setCurrentIndex(self.guide_combo.findData(self.state.crop_guide))
            self._sync_guide_orient_btn()
            self.ratio_combo.setCurrentText(conf.autocrop_ratio)
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(conf.autocrop_mode))

            self.offset_slider.setValue(float(conf.autocrop_offset))
            self.fine_rot_slider.setValue(-conf.fine_rotation)

            self.manual_crop_btn.setChecked(self.state.active_tool == ToolMode.CROP_MANUAL)
            self.straighten_btn.setChecked(self.state.active_tool == ToolMode.STRAIGHTEN)
            self.reset_crop_btn.setChecked(conf.auto_crop_enabled)
            self.manual_crop_btn.set_crop_active(conf.manual_crop_rect is not None)
            self.reset_crop_btn.set_crop_active(conf.auto_crop_enabled)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        self.guide_combo.blockSignals(blocked)
        self.guide_orient_btn.blockSignals(blocked)
        self.ratio_combo.blockSignals(blocked)
        self.mode_combo.blockSignals(blocked)
        self.detect_ratio_btn.blockSignals(blocked)
        self.offset_slider.blockSignals(blocked)
        self.fine_rot_slider.blockSignals(blocked)
        self.manual_crop_btn.blockSignals(blocked)
        self.straighten_btn.blockSignals(blocked)
        self.reset_crop_btn.blockSignals(blocked)
