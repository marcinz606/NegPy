from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QHBoxLayout,
)
from src.desktop.view.widgets.sliders import CompactSlider
from src.desktop.view.styles.theme import THEME
from src.desktop.controller import AppController
from src.desktop.session import ToolMode
from src.domain.constants import SUPPORTED_ASPECT_RATIOS, VERTICAL_ASPECT_RATIOS


class GeometrySidebar(QWidget):
    """
    Panel for cropping and fine adjustments.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)
        layout.setSpacing(10)

        conf = self.state.config.geometry

        # Ratio
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(SUPPORTED_ASPECT_RATIOS + VERTICAL_ASPECT_RATIOS)
        self.ratio_combo.setCurrentText(conf.autocrop_ratio)
        self.ratio_combo.setStyleSheet(
            f"font-size: {THEME.font_size_base}px; padding: 4px;"
        )

        # Toggles
        self.keep_borders_btn = QPushButton("Keep Borders")
        self.keep_borders_btn.setCheckable(True)
        self.keep_borders_btn.setChecked(conf.keep_full_frame)
        self._update_keep_borders_style(conf.keep_full_frame)

        # Buttons side by side
        btn_row = QHBoxLayout()
        self.manual_crop_btn = QPushButton("Manual Crop")
        self.manual_crop_btn.setCheckable(True)
        self.reset_crop_btn = QPushButton("Auto Crop")
        btn_row.addWidget(self.manual_crop_btn)
        btn_row.addWidget(self.reset_crop_btn)

        # Sliders (2 columns)
        slider_row = QHBoxLayout()
        self.offset_slider = CompactSlider(
            "Crop Offset", -20.0, 100.0, float(conf.autocrop_offset)
        )
        self.fine_rot_slider = CompactSlider("Fine Rot", -5.0, 5.0, conf.fine_rotation)
        slider_row.addWidget(self.offset_slider)
        slider_row.addWidget(self.fine_rot_slider)

        layout.addWidget(self.ratio_combo)
        layout.addWidget(self.keep_borders_btn)
        layout.addLayout(btn_row)
        layout.addLayout(slider_row)

    def _connect_signals(self) -> None:
        self.ratio_combo.currentTextChanged.connect(self._on_ratio_changed)
        self.keep_borders_btn.toggled.connect(self._on_keep_borders_changed)
        self.manual_crop_btn.toggled.connect(self._on_manual_crop_toggled)
        self.reset_crop_btn.clicked.connect(self.controller.reset_crop)
        self.offset_slider.valueChanged.connect(self._on_offset_changed)
        self.fine_rot_slider.valueChanged.connect(self._on_fine_rot_changed)

    def _on_ratio_changed(self, text: str) -> None:
        from dataclasses import replace

        new_geo = replace(self.state.config.geometry, autocrop_ratio=text)
        self.controller.session.update_config(
            replace(self.state.config, geometry=new_geo)
        )
        self.controller.request_render()

    def _on_keep_borders_changed(self, checked: bool) -> None:
        from dataclasses import replace

        self._update_keep_borders_style(checked)
        new_geo = replace(self.state.config.geometry, keep_full_frame=checked)
        self.controller.session.update_config(
            replace(self.state.config, geometry=new_geo)
        )
        self.controller.request_render()

    def _update_keep_borders_style(self, checked: bool) -> None:
        if checked:
            self.keep_borders_btn.setStyleSheet(
                f"background-color: {THEME.accent_primary}; color: white; font-weight: bold;"
            )
        else:
            self.keep_borders_btn.setStyleSheet("")

    def _on_offset_changed(self, val: float) -> None:
        from dataclasses import replace

        new_geo = replace(self.state.config.geometry, autocrop_offset=int(val))
        self.controller.session.update_config(
            replace(self.state.config, geometry=new_geo)
        )
        self.controller.request_render()

    def _on_fine_rot_changed(self, val: float) -> None:
        from dataclasses import replace

        new_geo = replace(self.state.config.geometry, fine_rotation=val)
        self.controller.session.update_config(
            replace(self.state.config, geometry=new_geo)
        )
        self.controller.request_render()

    def sync_ui(self) -> None:
        """
        Updates widgets to match global state.
        """
        conf = self.state.config.geometry
        self.ratio_combo.blockSignals(True)
        self.keep_borders_btn.blockSignals(True)
        self.offset_slider.blockSignals(True)
        self.fine_rot_slider.blockSignals(True)

        try:
            self.ratio_combo.setCurrentText(conf.autocrop_ratio)
            self.keep_borders_btn.setChecked(conf.keep_full_frame)
            self._update_keep_borders_style(conf.keep_full_frame)
            self.offset_slider.setValue(float(conf.autocrop_offset))
            self.fine_rot_slider.setValue(conf.fine_rotation)
        finally:
            self.ratio_combo.blockSignals(False)
            self.keep_borders_btn.blockSignals(False)
            self.offset_slider.blockSignals(False)
            self.fine_rot_slider.blockSignals(False)

    def _on_manual_crop_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(
            ToolMode.CROP_MANUAL if checked else ToolMode.NONE
        )
