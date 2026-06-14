from dataclasses import replace

import qtawesome as qta
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
)

from negpy.desktop.session import ToolMode
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.domain.models import AspectRatio
from negpy.features.geometry.models import AutocropMode
from negpy.features.process.models import invalidate_local_bounds


class GeometrySidebar(BaseSidebar):
    """
    Panel for cropping and fine adjustments.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.geometry

        # First row: Ratio combo + detect button
        ratio_row = QHBoxLayout()
        self.ratio_combo = QComboBox()
        # Filter out 'Original' as it's not a crop ratio (usually 'Free' is used for no constraint)
        ratios = [r.value for r in AspectRatio if r != AspectRatio.ORIGINAL]
        self.ratio_combo.addItems(ratios)
        self.ratio_combo.setCurrentText(conf.autocrop_ratio)
        self.ratio_combo.setPlaceholderText("Select Ratio...")
        self.ratio_combo.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 4px;")
        ratio_row.addWidget(self.ratio_combo, 1)

        self.detect_ratio_btn = QPushButton()
        self.detect_ratio_btn.setIcon(qta.icon("fa5s.crosshairs", color=THEME.text_primary))
        self.detect_ratio_btn.setToolTip("Detect closest aspect ratio from the film frame")
        self.detect_ratio_btn.setFixedWidth(36)
        ratio_row.addWidget(self.detect_ratio_btn)

        self.layout.addLayout(ratio_row)

        # Buttons side by side
        btn_row = QHBoxLayout()
        self.manual_crop_btn = QPushButton(" Manual")
        self.manual_crop_btn.setCheckable(True)
        self.manual_crop_btn.setIcon(qta.icon("fa5s.crop-alt", color=THEME.text_primary))
        self.manual_crop_btn.setToolTip(tooltip_with_shortcut("Manual crop", "manual_crop"))

        self.move_crop_btn = QPushButton(" Move")
        self.move_crop_btn.setCheckable(True)
        self.move_crop_btn.setIcon(qta.icon("fa5s.arrows-alt", color=THEME.text_primary))
        self.move_crop_btn.setToolTip("Translate the existing crop rectangle (preserves size)")

        self.reset_crop_btn = QPushButton(" Auto")
        self.reset_crop_btn.setCheckable(True)
        self.reset_crop_btn.setIcon(qta.icon("fa5s.magic", color=THEME.text_primary))
        self.reset_crop_btn.setToolTip(tooltip_with_shortcut("Apply automatic crop using the current ratio and offset", "auto_crop"))
        btn_row.addWidget(self.manual_crop_btn)
        btn_row.addWidget(self.move_crop_btn)
        self.layout.addLayout(btn_row)

        # Auto crop toggle + mode: crop to exposed image, or keep full film incl. rebate
        auto_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Image only", AutocropMode.IMAGE.value)
        self.mode_combo.addItem("Film edge", AutocropMode.FILM.value)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(conf.autocrop_mode))
        self.mode_combo.setToolTip("Auto crop target: exposed image only, or full film including rebate/sprockets")
        self.mode_combo.setStyleSheet(f"font-size: {THEME.font_size_base}px; padding: 4px;")
        self.reset_crop_btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.mode_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        auto_row.addWidget(self.reset_crop_btn, 1)
        auto_row.addWidget(self.mode_combo, 1)
        self.layout.addLayout(auto_row)

        # Sliders (2 columns)
        slider_row = QHBoxLayout()
        self.offset_slider = CompactSlider(
            "Crop Offset",
            -5.0,
            100.0,
            float(conf.autocrop_offset),
            step=1.0,
            precision=1,
            unit=" px",
        )
        self.offset_slider.setToolTip(tooltip_with_shortcut("Insets the crop border from the auto-detected film edge (px)", "offset_inc"))
        self.fine_rot_slider = CompactSlider("Fine Rot", -5.0, 5.0, conf.fine_rotation, unit="°")
        self.fine_rot_slider.setToolTip("Fine-tunes rotation to correct slight tilt (degrees)")
        slider_row.addWidget(self.offset_slider)
        slider_row.addWidget(self.fine_rot_slider)
        self.layout.addLayout(slider_row)

    def _connect_signals(self) -> None:
        self.ratio_combo.currentTextChanged.connect(self._on_ratio_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.detect_ratio_btn.clicked.connect(self.controller.detect_aspect_ratio)
        self.manual_crop_btn.toggled.connect(self._on_manual_crop_toggled)
        self.move_crop_btn.toggled.connect(self._on_move_crop_toggled)
        self.reset_crop_btn.toggled.connect(self._on_auto_crop_toggled)

        self.offset_slider.valueChanged.connect(
            lambda v: self.update_config_section("geometry", render=True, persist=False, readback_metrics=False, autocrop_offset=int(v))
        )
        self.offset_slider.valueCommitted.connect(self._on_offset_committed)

        self.fine_rot_slider.valueChanged.connect(
            lambda v: self.update_config_section("geometry", render=True, persist=False, readback_metrics=False, fine_rotation=v)
        )
        self.fine_rot_slider.valueCommitted.connect(
            lambda v: self.update_config_section("geometry", render=True, persist=True, readback_metrics=True, fine_rotation=v)
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

    def _on_move_crop_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.CROP_MOVE if checked else ToolMode.NONE)

    def _on_auto_crop_toggled(self, checked: bool) -> None:
        if checked:
            self.controller.apply_auto_crop()
        else:
            self.controller.reset_crop()

    def sync_ui(self) -> None:
        conf = self.state.config.geometry

        self.block_signals(True)
        try:
            self.ratio_combo.setCurrentText(conf.autocrop_ratio)
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(conf.autocrop_mode))

            self.offset_slider.setValue(float(conf.autocrop_offset))
            self.fine_rot_slider.setValue(conf.fine_rotation)

            self.manual_crop_btn.setChecked(self.state.active_tool == ToolMode.CROP_MANUAL)
            self.move_crop_btn.setChecked(self.state.active_tool == ToolMode.CROP_MOVE)
            self.move_crop_btn.setEnabled(conf.manual_crop_rect is not None)
            self.reset_crop_btn.setChecked(conf.auto_crop_enabled)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        self.ratio_combo.blockSignals(blocked)
        self.mode_combo.blockSignals(blocked)
        self.detect_ratio_btn.blockSignals(blocked)
        self.offset_slider.blockSignals(blocked)
        self.fine_rot_slider.blockSignals(blocked)
        self.manual_crop_btn.blockSignals(blocked)
        self.move_crop_btn.blockSignals(blocked)
        self.reset_crop_btn.blockSignals(blocked)
