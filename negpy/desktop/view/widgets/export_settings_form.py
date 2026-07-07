import os
from typing import Any, Dict

import qtawesome as qta
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.domain.models import (
    JXL_TAGGABLE_SPACES,
    AspectRatio,
    ColorSpace,
    ExportFormat,
    ExportPresetOutputMode,
    ExportResolutionMode,
    export_blocked,
)
from negpy.infrastructure.display.color_mgmt import ColorService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry

_LABEL_WIDTH = 90


def constrain_combo(combo: QComboBox, min_chars: int = 6) -> None:
    """Stop long item text from stretching the panel: size the combo to a small
    minimum and elide overflow, filling its row via the layout's spare space."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class ExportSettingsForm(QWidget):
    """Shared FORMAT / SIZE / COLOR / DESTINATION rows for the export sidebar and
    the presets dialog. Emits ``changed`` on any edit; the parent owns persistence.
    Read/write the rows via ``values()`` / ``load()`` keyed by the shared field
    names used by both ExportConfig and ExportPreset."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._flat_mode = False
        self._init_ui()

    @staticmethod
    def _row_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setFixedWidth(_LABEL_WIDTH)
        return label

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self._build_format(root)
        self._build_size(root)
        self._build_color(root)
        self._build_destination(root)

    # --- FORMAT --------------------------------------------------------------

    def _build_format(self, root: QVBoxLayout) -> None:
        self._format_section = QWidget()
        format_box = QVBoxLayout(self._format_section)
        format_box.setContentsMargins(0, 0, 0, 0)
        format_box.setSpacing(10)

        format_box.addWidget(section_subheader("FORMAT"))

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(self._row_label("Format"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems([f.value for f in ExportFormat])
        constrain_combo(self.fmt_combo)
        self.fmt_combo.currentTextChanged.connect(self._on_fmt_changed)
        fmt_row.addWidget(self.fmt_combo)
        format_box.addLayout(fmt_row)

        self._quality_container = QWidget()
        quality_box = QVBoxLayout(self._quality_container)
        quality_box.setContentsMargins(0, 0, 0, 0)
        self.quality_spin = CompactSlider("JPEG Quality", 1, 100, 90, step=1, precision=1)
        self.quality_spin.valueChanged.connect(self._on_changed)
        quality_box.addWidget(self.quality_spin)
        format_box.addWidget(self._quality_container)

        self._build_jxl(format_box)
        self._build_webp(format_box)
        root.addWidget(self._format_section)

    def _build_jxl(self, root: QVBoxLayout) -> None:
        self._jxl_container = QWidget()
        jxl_box = QVBoxLayout(self._jxl_container)
        jxl_box.setContentsMargins(0, 0, 0, 0)

        self.jxl_lossless_check = QCheckBox("Lossless")
        self.jxl_lossless_check.setChecked(True)
        self.jxl_lossless_check.toggled.connect(self._on_jxl_lossless_toggled)
        jxl_box.addWidget(self.jxl_lossless_check)

        self.jxl_distance_spin = CompactSlider("Distance", 0.0, 15.0, 1.0, step=0.1)
        self.jxl_distance_spin.label.setToolTip("libjxl distance: ~1.0 ≈ visually lossless, higher = more loss")
        self.jxl_distance_spin.valueChanged.connect(self._on_changed)
        jxl_box.addWidget(self.jxl_distance_spin)

        self.jxl_effort_spin = CompactSlider("Effort", 1, 9, 7, step=1, precision=1)
        self.jxl_effort_spin.label.setToolTip("Encoder effort: higher = slower, smaller file")
        self.jxl_effort_spin.valueChanged.connect(self._on_changed)
        jxl_box.addWidget(self.jxl_effort_spin)

        self.jxl_cs_warning = QLabel()
        self.jxl_cs_warning.setWordWrap(True)
        self.jxl_cs_warning.setStyleSheet(f"color: {THEME.accent_edited}; font-size: 10px;")
        jxl_box.addWidget(self.jxl_cs_warning)

        root.addWidget(self._jxl_container)

    def _build_webp(self, root: QVBoxLayout) -> None:
        self._webp_container = QWidget()
        webp_box = QVBoxLayout(self._webp_container)
        webp_box.setContentsMargins(0, 0, 0, 0)

        self.webp_lossless_check = QCheckBox("Lossless")
        self.webp_lossless_check.setChecked(False)
        self.webp_lossless_check.toggled.connect(self._on_changed)
        webp_box.addWidget(self.webp_lossless_check)

        self.webp_quality_spin = CompactSlider("Quality", 1, 100, 90, step=1, precision=1)
        self.webp_quality_spin.label.setToolTip("Lossy: visual quality. Lossless: compression effort.")
        self.webp_quality_spin.valueChanged.connect(self._on_changed)
        webp_box.addWidget(self.webp_quality_spin)

        self.webp_method_spin = CompactSlider("Method", 0, 6, 4, step=1, precision=1)
        self.webp_method_spin.label.setToolTip("Encoder effort: higher = slower, smaller file")
        self.webp_method_spin.valueChanged.connect(self._on_changed)
        webp_box.addWidget(self.webp_method_spin)

        root.addWidget(self._webp_container)

    # --- SIZE ----------------------------------------------------------------

    def _build_size(self, root: QVBoxLayout) -> None:
        root.addWidget(section_subheader("SIZE"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.mode_original_btn = QPushButton("Original")
        self.mode_print_btn = QPushButton("Print")
        self.mode_target_px_btn = QPushButton("Pixels")
        btn_style = f"font-size: {THEME.font_size_base}px; padding: 8px;"
        for btn in (self.mode_original_btn, self.mode_print_btn, self.mode_target_px_btn):
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            mode_row.addWidget(btn)
        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.setExclusive(True)
        self.mode_btn_group.addButton(self.mode_original_btn, 0)
        self.mode_btn_group.addButton(self.mode_print_btn, 1)
        self.mode_btn_group.addButton(self.mode_target_px_btn, 2)
        self.mode_btn_group.idToggled.connect(self._on_mode_toggled)
        root.addLayout(mode_row)

        # PRINT mode: cm + DPI
        self._print_container = QWidget()
        print_inner = QHBoxLayout(self._print_container)
        print_inner.setContentsMargins(0, 0, 0, 0)
        vbox_size = QVBoxLayout()
        vbox_size.addWidget(QLabel('Size <span style="color: #666666; font-size: 10px;">cm</span>'))
        self.size_input = QDoubleSpinBox()
        self.size_input.setRange(1.0, 500.0)
        self.size_input.setValue(30.0)
        self.size_input.valueChanged.connect(self._on_changed)
        vbox_size.addWidget(self.size_input)
        vbox_dpi = QVBoxLayout()
        vbox_dpi.addWidget(QLabel("DPI"))
        self.dpi_input = QSpinBox()
        self.dpi_input.setRange(72, 4800)
        self.dpi_input.setValue(300)
        self.dpi_input.valueChanged.connect(self._on_changed)
        vbox_dpi.addWidget(self.dpi_input)
        print_inner.addLayout(vbox_size)
        print_inner.addLayout(vbox_dpi)
        root.addWidget(self._print_container)

        # TARGET_PX mode: long edge in pixels
        self._target_px_container = QWidget()
        target_px_inner = QVBoxLayout(self._target_px_container)
        target_px_inner.setContentsMargins(0, 0, 0, 0)
        target_px_inner.addWidget(QLabel('Long edge <span style="color: #666666; font-size: 10px;">px</span>'))
        self.target_px_input = QSpinBox()
        self.target_px_input.setRange(256, 32768)
        self.target_px_input.setValue(2000)
        self.target_px_input.valueChanged.connect(self._on_changed)
        target_px_inner.addWidget(self.target_px_input)
        root.addWidget(self._target_px_container)

        self._ratio_row_widget = QWidget()
        ratio_row = QHBoxLayout(self._ratio_row_widget)
        ratio_row.setContentsMargins(0, 0, 0, 0)
        ratio_row.addWidget(self._row_label("Paper ratio"))
        self.ratio_combo = QComboBox()
        ratios = [AspectRatio.ORIGINAL] + [r.value for r in AspectRatio if r != AspectRatio.ORIGINAL]
        self.ratio_combo.addItems(ratios)
        constrain_combo(self.ratio_combo)
        self.ratio_combo.currentTextChanged.connect(self._on_changed)
        ratio_row.addWidget(self.ratio_combo)
        root.addWidget(self._ratio_row_widget)

    # --- COLOR ---------------------------------------------------------------

    def _build_color(self, root: QVBoxLayout) -> None:
        root.addWidget(section_subheader("COLOR"))

        # Drop bundled profiles already backed by a color-space enum so the ICC
        # lists don't duplicate the color-space selector.
        enum_mapped = {ColorSpaceRegistry.get_icc_path(cs.value) for cs in ColorSpace}
        enum_mapped.discard(None)
        custom_profiles = [p for p in ColorService.get_available_profiles() if p not in enum_mapped]

        self._icc_input_profiles = ["None"] + custom_profiles
        input_row = QHBoxLayout()
        input_row.addWidget(self._row_label("Input ICC"))
        self.input_combo = QComboBox()
        self.input_combo.addItems([os.path.basename(p) for p in self._icc_input_profiles])
        constrain_combo(self.input_combo)
        self.input_combo.setToolTip("Source/input ICC profile")
        self.input_combo.currentIndexChanged.connect(self._on_changed)
        input_row.addWidget(self.input_combo)
        root.addLayout(input_row)

        cs_row = QHBoxLayout()
        cs_row.addWidget(self._row_label("Color space"))
        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems([cs.value for cs in ColorSpace])
        constrain_combo(self.color_space_combo)
        self.color_space_combo.currentTextChanged.connect(self._on_changed)
        self.color_space_combo.currentTextChanged.connect(self._refresh_jxl_warning)
        cs_row.addWidget(self.color_space_combo)
        root.addLayout(cs_row)

        self._icc_output_profiles = ["None"] + custom_profiles
        output_row = QHBoxLayout()
        output_row.addWidget(self._row_label("Output ICC"))
        self.icc_output_combo = QComboBox()
        self.icc_output_combo.addItems([os.path.basename(p) for p in self._icc_output_profiles])
        constrain_combo(self.icc_output_combo)
        self.icc_output_combo.setToolTip("Custom output ICC profile (overrides color space)")
        self.icc_output_combo.currentIndexChanged.connect(self._on_changed)
        output_row.addWidget(self.icc_output_combo)
        root.addLayout(output_row)

    # --- DESTINATION ---------------------------------------------------------

    def _build_destination(self, root: QVBoxLayout) -> None:
        root.addWidget(section_subheader("DESTINATION"))

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._row_label("Folder"))
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItem("Subfolder of source", ExportPresetOutputMode.SUBFOLDER_OF_SOURCE)
        self.output_mode_combo.addItem("Same as source", ExportPresetOutputMode.SAME_AS_SOURCE)
        self.output_mode_combo.addItem("Absolute path", ExportPresetOutputMode.ABSOLUTE)
        constrain_combo(self.output_mode_combo)
        self.output_mode_combo.currentIndexChanged.connect(self._on_output_mode_changed)
        mode_row.addWidget(self.output_mode_combo)
        root.addLayout(mode_row)

        self._subfolder_container = QWidget()
        sf_inner = QHBoxLayout(self._subfolder_container)
        sf_inner.setContentsMargins(0, 0, 0, 0)
        sf_inner.addWidget(self._row_label("Subfolder"))
        self.subfolder_edit = QLineEdit()
        self.subfolder_edit.setPlaceholderText("e.g. TIFF")
        self.subfolder_edit.textChanged.connect(self._on_changed)
        sf_inner.addWidget(self.subfolder_edit)
        root.addWidget(self._subfolder_container)

        self._abspath_container = QWidget()
        ap_inner = QHBoxLayout(self._abspath_container)
        ap_inner.setContentsMargins(0, 0, 0, 0)
        ap_inner.addWidget(self._row_label("Path"))
        self.abspath_edit = QLineEdit()
        self.abspath_edit.setToolTip("Export folder")
        self.abspath_edit.textChanged.connect(self._on_changed)
        self.abspath_browse_btn = QPushButton()
        self.abspath_browse_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.abspath_browse_btn.setFixedWidth(40)
        self.abspath_browse_btn.setToolTip("Choose export folder")
        self.abspath_browse_btn.clicked.connect(self._browse_output_path)
        ap_inner.addWidget(self.abspath_edit)
        ap_inner.addWidget(self.abspath_browse_btn)
        root.addWidget(self._abspath_container)

        filename_row = QHBoxLayout()
        filename_row.addWidget(self._row_label("Filename"))
        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("Filename Pattern...")
        self.filename_edit.setToolTip(
            "Jinja2 template. Variables:\n"
            "{{ original_name }}, {{ colorspace }}, {{ format }},\n"
            "{{ paper_ratio }}, {{ size }}, {{ dpi }}, {{ target_px }},\n"
            "{{ border }}, {{ date }}"
        )
        self.filename_edit.textChanged.connect(self._on_changed)
        filename_row.addWidget(self.filename_edit)
        root.addLayout(filename_row)

        self.overwrite_check = QCheckBox("Overwrite existing files")
        self.overwrite_check.stateChanged.connect(self._on_changed)
        root.addWidget(self.overwrite_check)

    # --- Change handling -----------------------------------------------------

    def _on_changed(self, *_) -> None:
        if not self._loading:
            self.changed.emit()

    def _on_fmt_changed(self, fmt: str) -> None:
        self._quality_container.setVisible(fmt == ExportFormat.JPEG)
        self._jxl_container.setVisible(fmt == ExportFormat.JXL)
        self._webp_container.setVisible(fmt == ExportFormat.WEBP)
        self._apply_jxl_constraints()
        self._refresh_jxl_warning()
        self._on_changed()

    def _apply_jxl_constraints(self) -> None:
        """For JXL, grey out colour spaces it can't tag and disable the output ICC
        override (a custom profile would land pixels in an un-enumerable space while
        we still tag enumeratively — a silent mistag)."""
        is_jxl = self.fmt_combo.currentText() == ExportFormat.JXL

        model = self.color_space_combo.model()
        for i in range(self.color_space_combo.count()):
            item = model.item(i)
            if item is not None:
                supported = self.color_space_combo.itemText(i) in JXL_TAGGABLE_SPACES
                item.setEnabled(supported or not is_jxl)
        if is_jxl and self.color_space_combo.currentText() not in JXL_TAGGABLE_SPACES:
            self.color_space_combo.setCurrentText(ColorSpace.SRGB.value)

        if is_jxl:
            self.icc_output_combo.setCurrentIndex(0)  # None — no custom output profile
        self.icc_output_combo.setEnabled(not is_jxl)

    def _on_jxl_lossless_toggled(self, lossless: bool) -> None:
        self.jxl_distance_spin.setEnabled(not lossless)
        self._on_changed()

    def is_export_blocked(self) -> bool:
        """True when the current JXL + colour space pairing can't be tagged."""
        if self._flat_mode:
            return False
        return export_blocked(self.fmt_combo.currentText(), self.color_space_combo.currentText())

    def _refresh_jxl_warning(self) -> None:
        blocked = self.is_export_blocked()
        if blocked:
            self.jxl_cs_warning.setText(
                f"JPEG XL can't tag {self.color_space_combo.currentText()} — "
                "choose sRGB, P3 D65, Rec 2020, or Greyscale, or a different format."
            )
        self.jxl_cs_warning.setVisible(blocked)

    def _on_mode_toggled(self, _id: int, checked: bool) -> None:
        if not checked:
            return
        mode = self._current_mode_value()
        self._update_mode_visibility(mode)
        self._update_ratio_visibility(mode)
        self._on_changed()

    def _on_output_mode_changed(self, _idx: int) -> None:
        self._update_output_mode_visibility(self.output_mode_combo.currentData())
        self._on_changed()

    def _browse_output_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.abspath_edit.text())
        if path:
            self.abspath_edit.setText(path)

    # --- Mode helpers --------------------------------------------------------

    _MODE_BY_ID = {
        0: ExportResolutionMode.ORIGINAL.value,
        1: ExportResolutionMode.PRINT.value,
        2: ExportResolutionMode.TARGET_PX.value,
    }
    _ID_BY_MODE = {v: k for k, v in _MODE_BY_ID.items()}

    def _current_mode_value(self) -> str:
        return self._MODE_BY_ID.get(self.mode_btn_group.checkedId(), ExportResolutionMode.PRINT.value)

    def _select_mode_button(self, mode_value: str) -> None:
        btn = self.mode_btn_group.button(self._ID_BY_MODE.get(mode_value, 1))
        if btn is not None:
            btn.setChecked(True)

    def _update_mode_visibility(self, mode_value: str) -> None:
        self._print_container.setVisible(mode_value == ExportResolutionMode.PRINT.value)
        self._target_px_container.setVisible(mode_value == ExportResolutionMode.TARGET_PX.value)

    def _update_ratio_visibility(self, mode_value: str | None = None) -> None:
        """Paper ratio applies to print-style sizing; flat + Original hides it."""
        if mode_value is None:
            mode_value = self._current_mode_value()
        if self._flat_mode and mode_value == ExportResolutionMode.ORIGINAL.value:
            self._ratio_row_widget.setVisible(False)
        else:
            self._ratio_row_widget.setVisible(True)

    def set_flat_mode(self, enabled: bool, *, preset_editor: bool = False) -> None:
        """Toggle flat-master export UI: hide delivery formats, adjust size rows.

        When ``preset_editor`` is True (Manage Presets dialog), the format row stays
        visible but limited to TIFF/DNG instead of hiding entirely.
        """
        enabled = bool(enabled)
        if enabled == self._flat_mode:
            return
        self._flat_mode = enabled

        if preset_editor:
            self._format_section.setVisible(True)
            current = self.fmt_combo.currentText()
            self.fmt_combo.blockSignals(True)
            self.fmt_combo.clear()
            if enabled:
                flat_formats = [ExportFormat.TIFF.value, ExportFormat.DNG.value]
                self.fmt_combo.addItems(flat_formats)
                self.fmt_combo.setCurrentText(current if current in flat_formats else ExportFormat.TIFF.value)
                self._quality_container.setVisible(False)
                self._jxl_container.setVisible(False)
                self._webp_container.setVisible(False)
            else:
                all_formats = [f.value for f in ExportFormat]
                self.fmt_combo.addItems(all_formats)
                self.fmt_combo.setCurrentText(current if current in all_formats else ExportFormat.JPEG.value)
            self.fmt_combo.blockSignals(False)
            if not enabled:
                self._on_fmt_changed(self.fmt_combo.currentText())
            self._update_ratio_visibility()
            return

        self._format_section.setVisible(not enabled)
        if enabled:
            self._update_ratio_visibility()
        else:
            self._on_fmt_changed(self.fmt_combo.currentText())
            self._update_ratio_visibility()

    def flat_mode(self) -> bool:
        return self._flat_mode

    def _update_output_mode_visibility(self, mode) -> None:
        self._subfolder_container.setVisible(mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE)
        self._abspath_container.setVisible(mode == ExportPresetOutputMode.ABSOLUTE)

    # --- Load / read ---------------------------------------------------------

    def load(self, v: Dict[str, Any]) -> None:
        """Populate all rows from a dict of shared field values."""
        self._loading = True
        try:
            self.fmt_combo.setCurrentText(v["export_fmt"])
            self.quality_spin.setValue(v.get("jpeg_quality", 90))
            self._quality_container.setVisible(v["export_fmt"] == ExportFormat.JPEG)

            self.jxl_lossless_check.setChecked(v.get("jxl_lossless", True))
            self.jxl_distance_spin.setValue(v.get("jxl_distance", 1.0))
            self.jxl_distance_spin.setEnabled(not v.get("jxl_lossless", True))
            self.jxl_effort_spin.setValue(v.get("jxl_effort", 7))
            self._jxl_container.setVisible(v["export_fmt"] == ExportFormat.JXL)

            self.webp_quality_spin.setValue(v.get("webp_quality", 90))
            self.webp_lossless_check.setChecked(v.get("webp_lossless", False))
            self.webp_method_spin.setValue(v.get("webp_method", 4))
            self._webp_container.setVisible(v["export_fmt"] == ExportFormat.WEBP)

            self._select_mode_button(v["export_resolution_mode"])
            self._update_mode_visibility(v["export_resolution_mode"])
            self._update_ratio_visibility(v["export_resolution_mode"])
            self.size_input.setValue(v["export_print_size"])
            self.dpi_input.setValue(v["export_dpi"])
            self.target_px_input.setValue(v["export_target_long_edge_px"])
            self.ratio_combo.setCurrentText(v["paper_aspect_ratio"])

            in_path = v.get("icc_input_path")
            self.input_combo.setCurrentText(os.path.basename(in_path) if in_path else "None")
            self.color_space_combo.setCurrentText(v["export_color_space"])
            out_path = v.get("icc_output_path")
            self.icc_output_combo.setCurrentText(os.path.basename(out_path) if out_path else "None")

            mode = v.get("output_mode", ExportPresetOutputMode.ABSOLUTE)
            idx = self.output_mode_combo.findData(mode)
            if idx >= 0:
                self.output_mode_combo.setCurrentIndex(idx)
            self._update_output_mode_visibility(mode)
            self.subfolder_edit.setText(v.get("output_subfolder", ""))
            self.abspath_edit.setText(v.get("output_path", ""))
            self.filename_edit.setText(v["filename_pattern"])
            self.overwrite_check.setChecked(v["overwrite"])
            self._apply_jxl_constraints()
            self._refresh_jxl_warning()
        finally:
            self._loading = False

    def values(self) -> Dict[str, Any]:
        """Read all rows back into a dict of shared field values."""
        in_idx = self.input_combo.currentIndex()
        out_idx = self.icc_output_combo.currentIndex()
        return {
            "export_fmt": self.fmt_combo.currentText(),
            "jpeg_quality": int(self.quality_spin.value()),
            "jxl_lossless": self.jxl_lossless_check.isChecked(),
            "jxl_distance": self.jxl_distance_spin.value(),
            "jxl_effort": int(self.jxl_effort_spin.value()),
            "webp_quality": int(self.webp_quality_spin.value()),
            "webp_lossless": self.webp_lossless_check.isChecked(),
            "webp_method": int(self.webp_method_spin.value()),
            "export_resolution_mode": self._current_mode_value(),
            "paper_aspect_ratio": self.ratio_combo.currentText(),
            "export_print_size": self.size_input.value(),
            "export_dpi": self.dpi_input.value(),
            "export_target_long_edge_px": self.target_px_input.value(),
            "output_mode": self.output_mode_combo.currentData() or ExportPresetOutputMode.ABSOLUTE,
            "output_subfolder": self.subfolder_edit.text(),
            "output_path": self.abspath_edit.text(),
            "filename_pattern": self.filename_edit.text(),
            "overwrite": self.overwrite_check.isChecked(),
            "export_color_space": self.color_space_combo.currentText(),
            "icc_input_path": self._icc_input_profiles[in_idx] if in_idx > 0 else None,
            "icc_output_path": self._icc_output_profiles[out_idx] if out_idx > 0 else None,
        }
