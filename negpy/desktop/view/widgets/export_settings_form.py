import os
from typing import Any, Dict, Optional

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import ICON_BUTTON_WIDTH, hint_label, labeled_toggle_qss, section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.domain.models import (
    EXPORT_COLOR_SPACES,
    JXL_TAGGABLE_SPACES,
    AspectRatio,
    ColorSpace,
    ExportFormat,
    ExportPresetOutputMode,
    ExportResolutionMode,
    TiffCompression,
    export_blocked,
)
from negpy.infrastructure.display.color_mgmt import ColorService, import_icc_profile
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
from negpy.kernel.system.config import APP_CONFIG

_LABEL_WIDTH = 90
_EXPORT_SPACES = frozenset(EXPORT_COLOR_SPACES)


def _select_data(combo: QComboBox, data: Any) -> None:
    idx = combo.findData(data)
    combo.setCurrentIndex(max(idx, 0))


def _coerce_output_mode(mode: Any) -> ExportPresetOutputMode:
    """Persisted configs come back from JSON as plain strings. The destination
    combo holds StrEnum members as item data, and findData() compares across the
    QVariant boundary — a plain string never matches, so the row must be
    normalized to the enum before lookup."""
    try:
        return ExportPresetOutputMode(mode)
    except ValueError:
        return ExportPresetOutputMode.ABSOLUTE


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
        self._linear_mode = False
        # Retained while an ICC profile is selected, so switching back to a space restores
        # the last one rather than a default.
        self._export_space = ColorSpace.SRGB.value
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
        for f in ExportFormat:
            self.fmt_combo.addItem(f.value, f.value)
        constrain_combo(self.fmt_combo)
        self.fmt_combo.currentIndexChanged.connect(self._on_fmt_changed)
        fmt_row.addWidget(self.fmt_combo)
        format_box.addLayout(fmt_row)

        self._depth_container = QWidget()
        depth_row = QHBoxLayout(self._depth_container)
        depth_row.setContentsMargins(0, 0, 0, 0)
        depth_row.addWidget(self._row_label("Bit Depth"))
        self.bit_depth_combo = QComboBox()
        for label, data in (("8-bit", 8), ("16-bit", 16)):
            self.bit_depth_combo.addItem(label, data)
        self.bit_depth_combo.setToolTip("JPEG and WebP are 8-bit formats and ignore this")
        constrain_combo(self.bit_depth_combo)
        self.bit_depth_combo.currentIndexChanged.connect(self._on_changed)
        depth_row.addWidget(self.bit_depth_combo)
        format_box.addWidget(self._depth_container)

        self._quality_container = QWidget()
        quality_box = QVBoxLayout(self._quality_container)
        quality_box.setContentsMargins(0, 0, 0, 0)
        self.quality_spin = CompactSlider("JPEG Quality", 1, 100, 90, step=1, precision=1)
        self.quality_spin.valueChanged.connect(self._on_changed)
        quality_box.addWidget(self.quality_spin)
        self.jpeg_progressive_check = QCheckBox("Progressive")
        self.jpeg_progressive_check.setToolTip("Renders in passes while downloading; slightly smaller on large images")
        self.jpeg_progressive_check.toggled.connect(self._on_changed)
        quality_box.addWidget(self.jpeg_progressive_check)
        format_box.addWidget(self._quality_container)

        self._build_tiff(format_box)
        self._build_png(format_box)
        self._build_jxl(format_box)
        self._build_webp(format_box)
        root.addWidget(self._format_section)

    def _build_tiff(self, root: QVBoxLayout) -> None:
        self._tiff_container = QWidget()
        tiff_row = QHBoxLayout(self._tiff_container)
        tiff_row.setContentsMargins(0, 0, 0, 0)
        tiff_row.addWidget(self._row_label("Compression"))
        self.tiff_compression_combo = QComboBox()
        for label, data in (
            ("Uncompressed", TiffCompression.NONE),
            ("LZW", TiffCompression.LZW),
            ("ZIP", TiffCompression.ZIP),
        ):
            self.tiff_compression_combo.addItem(label, data)
        self.tiff_compression_combo.setToolTip("All three are lossless; ZIP is usually the smallest")
        constrain_combo(self.tiff_compression_combo)
        self.tiff_compression_combo.currentIndexChanged.connect(self._on_changed)
        tiff_row.addWidget(self.tiff_compression_combo)
        root.addWidget(self._tiff_container)

    def _build_png(self, root: QVBoxLayout) -> None:
        self._png_container = QWidget()
        png_box = QVBoxLayout(self._png_container)
        png_box.setContentsMargins(0, 0, 0, 0)
        self.png_level_spin = CompactSlider("Compression", 0, 9, 6, step=1, precision=1)
        self.png_level_spin.setToolTip("Lossless either way: higher = slower, smaller file")
        self.png_level_spin.valueChanged.connect(self._on_changed)
        png_box.addWidget(self.png_level_spin)
        root.addWidget(self._png_container)

    def _build_jxl(self, root: QVBoxLayout) -> None:
        self._jxl_container = QWidget()
        jxl_box = QVBoxLayout(self._jxl_container)
        jxl_box.setContentsMargins(0, 0, 0, 0)

        self.jxl_lossless_check = QCheckBox("Lossless")
        self.jxl_lossless_check.setChecked(True)
        self.jxl_lossless_check.toggled.connect(self._on_jxl_lossless_toggled)
        jxl_box.addWidget(self.jxl_lossless_check)

        self.jxl_distance_spin = CompactSlider("Distance", 0.0, 15.0, 1.0, step=0.1)
        self.jxl_distance_spin.setToolTip("libjxl distance: ~1.0 ≈ visually lossless, higher = more loss")
        self.jxl_distance_spin.valueChanged.connect(self._on_changed)
        jxl_box.addWidget(self.jxl_distance_spin)

        self.jxl_effort_spin = CompactSlider("Effort", 1, 9, 7, step=1, precision=1)
        self.jxl_effort_spin.setToolTip("Encoder effort: higher = slower, smaller file")
        self.jxl_effort_spin.valueChanged.connect(self._on_changed)
        jxl_box.addWidget(self.jxl_effort_spin)

        self.jxl_cs_warning = hint_label(kind="warning")
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
        self.webp_quality_spin.setToolTip("Lossy: visual quality. Lossless: compression effort.")
        self.webp_quality_spin.valueChanged.connect(self._on_changed)
        webp_box.addWidget(self.webp_quality_spin)

        self.webp_method_spin = CompactSlider("Method", 0, 6, 4, step=1, precision=1)
        self.webp_method_spin.setToolTip("Encoder effort: higher = slower, smaller file")
        self.webp_method_spin.valueChanged.connect(self._on_changed)
        webp_box.addWidget(self.webp_method_spin)

        root.addWidget(self._webp_container)

    # --- SIZE ----------------------------------------------------------------

    def _build_size(self, parent: QVBoxLayout) -> None:
        self._size_section = QWidget()
        root = QVBoxLayout(self._size_section)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        parent.addWidget(self._size_section)

        root.addWidget(section_subheader("SIZE"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.mode_original_btn = QPushButton("Original")
        self.mode_print_btn = QPushButton("Print")
        self.mode_target_px_btn = QPushButton("Pixels")
        for btn in (self.mode_original_btn, self.mode_print_btn, self.mode_target_px_btn):
            btn.setCheckable(True)
            btn.setStyleSheet(labeled_toggle_qss())
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
        vbox_size.addWidget(QLabel(f'Size <span style="color: {THEME.text_hint}; font-size: {THEME.font_size_small}px;">cm</span>'))
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
        target_px_inner.addWidget(
            QLabel(f'Long edge <span style="color: {THEME.text_hint}; font-size: {THEME.font_size_small}px;">px</span>')
        )
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

    def _build_color(self, parent: QVBoxLayout) -> None:
        self._color_section = QWidget()
        root = QVBoxLayout(self._color_section)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        parent.addWidget(self._color_section)

        header_row = QHBoxLayout()
        header_row.addWidget(section_subheader("COLOR MANAGEMENT"))
        header_row.addStretch()
        self.icc_import_btn = QPushButton()
        self.icc_import_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.icc_import_btn.setFixedWidth(ICON_BUTTON_WIDTH)
        self.icc_import_btn.setToolTip(f"Import an ICC profile into {APP_CONFIG.user_icc_dir}")
        self.icc_import_btn.clicked.connect(self._import_icc)
        header_row.addWidget(self.icc_import_btn, alignment=Qt.AlignmentFlag.AlignBottom)
        root.addLayout(header_row)

        root.addWidget(hint_label("Processing is scene-linear (Adobe RGB primaries)"))

        input_row = QHBoxLayout()
        input_row.addWidget(self._row_label("Input ICC"))
        self.input_combo = QComboBox()
        constrain_combo(self.input_combo)
        self.input_combo.setToolTip("Treat the source as this profile, for a scan whose profile is known but untagged")
        self.input_combo.currentIndexChanged.connect(self._on_changed)
        input_row.addWidget(self.input_combo)
        root.addLayout(input_row)

        # One destination, one control: a custom ICC overrides a color space everywhere
        # downstream (encode_export, effective_output_icc), so offering both as separate
        # rows shows a live-looking selector that the other one silently voids.
        profile_row = QHBoxLayout()
        profile_row.addWidget(self._row_label("Export profile"))
        self.export_profile_combo = QComboBox()
        constrain_combo(self.export_profile_combo)
        self.export_profile_combo.setToolTip(
            "Color space the exported file is converted to and tagged with. Pick an imported "
            "ICC profile to target a printer or paper instead."
        )
        self.export_profile_combo.currentIndexChanged.connect(self._on_export_profile_changed)
        profile_row.addWidget(self.export_profile_combo)
        root.addLayout(profile_row)

        self._color_extra = QVBoxLayout()
        self._color_extra.setContentsMargins(0, 0, 0, 0)
        self._color_extra.setSpacing(6)
        root.addLayout(self._color_extra)

        self._reload_icc_profiles()

    def add_color_widget(self, widget: QWidget) -> None:
        """Park a caller's widget at the foot of the COLOR block. The sidebar's soft-proof
        toggle and its warning belong beside the profile they describe; the presets dialog
        adds nothing."""
        self._color_extra.addWidget(widget)

    def _reload_icc_profiles(self, select: Optional[str] = None) -> None:
        """Rebuild both ICC lists from disk, restoring the selections by payload.

        Signals stay blocked throughout, so the caller decides whether the rebuild
        counts as a user edit.
        """
        # Drop bundled profiles already backed by a color-space enum, so an ICC entry never
        # duplicates a space entry.
        enum_mapped = {ColorSpaceRegistry.get_icc_path(cs.value) for cs in ColorSpace}
        enum_mapped.discard(None)
        custom = [p for p in ColorService.get_available_profiles() if p not in enum_mapped]

        prev_input = self.input_combo.currentData()
        prev_profile = select or self.export_profile_combo.currentData() or ColorSpace.SRGB.value

        self.input_combo.blockSignals(True)
        self.export_profile_combo.blockSignals(True)
        try:
            self.input_combo.clear()
            self.input_combo.addItem("None", None)
            self.export_profile_combo.clear()
            for cs in EXPORT_COLOR_SPACES:
                self.export_profile_combo.addItem(cs, cs)
            if custom:
                self.export_profile_combo.insertSeparator(self.export_profile_combo.count())
            for path in custom:
                name = os.path.basename(path)
                self.input_combo.addItem(name, path)
                self.export_profile_combo.addItem(name, path)
            _select_data(self.input_combo, prev_input)
            _select_data(self.export_profile_combo, prev_profile)
        finally:
            self.input_combo.blockSignals(False)
            self.export_profile_combo.blockSignals(False)

    def set_source_space(self, color_space: str) -> None:
        """Name the space `Same as Source` resolves to, so it does not read as
        'NegPy ignored my profile'. Item lookups go by payload, not text."""
        idx = self.export_profile_combo.findData(ColorSpace.SAME_AS_SOURCE.value)
        if idx >= 0:
            label = ColorSpace.SAME_AS_SOURCE.value
            self.export_profile_combo.setItemText(idx, f"{label} ({color_space})" if color_space else label)

    def _on_export_profile_changed(self) -> None:
        data = self.export_profile_combo.currentData()
        if data in _EXPORT_SPACES:
            self._export_space = data
        self._refresh_jxl_warning()
        self._on_changed()

    def _import_icc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import ICC profile", "", "ICC profiles (*.icc *.icm)")
        if not path:
            return
        # ColorSpaceRegistry prefers a user file named after a space, so such an import
        # replaces that bundled profile everywhere instead of adding a choice.
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in _EXPORT_SPACES:
            confirm = QMessageBox.question(
                self,
                "Replace a built-in space?",
                f"A profile named '{stem}' replaces NegPy's own {stem} profile everywhere, "
                "instead of appearing as a separate choice. Import anyway?",
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        try:
            stored = import_icc_profile(path, APP_CONFIG.user_icc_dir)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Import failed", str(e))
            return
        self._reload_icc_profiles(select=stored)
        self._on_export_profile_changed()

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
        self.abspath_browse_btn.setFixedWidth(ICON_BUTTON_WIDTH)
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
            "{{ border }}, {{ date }},\n"
            "{{ roll }}, {{ frame }}, {{ frame|pad(3) }}, {{ frame_padded }},\n"
            "{{ camera }}, {{ lens }}, {{ film }}, {{ film_iso }}, {{ film_format }},\n"
            "{{ developer }}, {{ push_pull }}, {{ scanning }}, {{ exposure }}\n"
            "(see docs/TEMPLATING.md for the full list)"
        )
        self.filename_edit.textChanged.connect(self._on_changed)
        filename_row.addWidget(self.filename_edit)
        root.addLayout(filename_row)

        self.overwrite_check = QCheckBox("Overwrite existing files")
        self.overwrite_check.setToolTip(
            "Checked: replace files that already exist, without asking. "
            "Unchecked: ask before overwriting (Overwrite / Rename / Cancel) when a file already exists."
        )
        self.overwrite_check.stateChanged.connect(self._on_changed)
        root.addWidget(self.overwrite_check)

    # --- Change handling -----------------------------------------------------

    def _on_changed(self, *_) -> None:
        if not self._loading:
            self.changed.emit()

    def _update_format_visibility(self, fmt: Any) -> None:
        self._quality_container.setVisible(fmt == ExportFormat.JPEG)
        self._tiff_container.setVisible(fmt == ExportFormat.TIFF)
        self._png_container.setVisible(fmt == ExportFormat.PNG)
        self._jxl_container.setVisible(fmt == ExportFormat.JXL)
        self._webp_container.setVisible(fmt == ExportFormat.WEBP)
        # A flat master is always 16-bit, so the row would name a choice the export overrides.
        depth_formats = (ExportFormat.TIFF, ExportFormat.PNG, ExportFormat.JXL)
        self._depth_container.setVisible(fmt in depth_formats and not self._flat_mode)

    def _on_fmt_changed(self, *_ignored: Any) -> None:
        self._update_format_visibility(self.fmt_combo.currentData())
        self._apply_jxl_constraints()
        self._refresh_jxl_warning()
        self._on_changed()

    def _apply_jxl_constraints(self) -> None:
        """For JXL, grey out every export profile it can't tag. Custom ICC entries fall out
        by the same rule: a custom profile would land pixels in an un-enumerable space while
        we still tag enumeratively — a silent mistag."""
        is_jxl = self.fmt_combo.currentData() == ExportFormat.JXL

        model = self.export_profile_combo.model()
        for i in range(self.export_profile_combo.count()):
            item = model.item(i)
            if item is not None:
                supported = self.export_profile_combo.itemData(i) in JXL_TAGGABLE_SPACES
                item.setEnabled(supported or not is_jxl)
        if is_jxl and self.export_profile_combo.currentData() not in JXL_TAGGABLE_SPACES:
            _select_data(self.export_profile_combo, ColorSpace.SRGB.value)

        # flat_export_config() always forces jxl_lossless=True for a flat master, so hide the
        # lossy toggle and distance rather than show a control the export silently overrides.
        flat_locked_lossless = self._flat_mode and is_jxl
        if flat_locked_lossless:
            self.jxl_lossless_check.setChecked(True)
        self.jxl_lossless_check.setVisible(not flat_locked_lossless)
        self.jxl_distance_spin.setVisible(not flat_locked_lossless)

    def _on_jxl_lossless_toggled(self, lossless: bool) -> None:
        self.jxl_distance_spin.setEnabled(not lossless)
        self._on_changed()

    def is_export_blocked(self) -> bool:
        """True when the current JXL + color space pairing can't be tagged."""
        if self._flat_mode:
            return False
        return export_blocked(self.fmt_combo.currentData(), self.export_profile_combo.currentData() or "")

    def _refresh_jxl_warning(self) -> None:
        blocked = self.is_export_blocked()
        if blocked:
            self.jxl_cs_warning.setText(
                f"JPEG XL can't tag {self.export_profile_combo.currentText()} — "
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

    def set_flat_mode(self, enabled: bool) -> None:
        """Toggle flat-master export UI: constrain FORMAT to TIFF/JXL (a flat master
        is always 16-bit lossless), adjust size rows. Same behaviour in the export
        sidebar and the presets dialog's flat preset editor — both need the format
        choice, not just sizing."""
        enabled = bool(enabled)
        if enabled == self._flat_mode:
            return
        self._flat_mode = enabled

        self._format_section.setVisible(not self._linear_mode)
        current = self.fmt_combo.currentData()
        self.fmt_combo.blockSignals(True)
        self.fmt_combo.clear()
        if enabled:
            # "(lossless)" on the JXL entry: a flat master is always lossless, since
            # flat_export_config() forces jxl_lossless=True, so the label should never leave that in
            # doubt the way the general "JXL" entry does.
            flat_items = [(ExportFormat.TIFF.value, ExportFormat.TIFF.value), ("JXL (lossless)", ExportFormat.JXL.value)]
            for label, data in flat_items:
                self.fmt_combo.addItem(label, data)
            target = current if current in (ExportFormat.TIFF.value, ExportFormat.JXL.value) else ExportFormat.TIFF.value
        else:
            for f in ExportFormat:
                self.fmt_combo.addItem(f.value, f.value)
            all_formats = [f.value for f in ExportFormat]
            target = current if current in all_formats else ExportFormat.JPEG.value
        idx = self.fmt_combo.findData(target)
        self.fmt_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.fmt_combo.blockSignals(False)
        self._on_fmt_changed()
        self._update_ratio_visibility()

    def flat_mode(self) -> bool:
        return self._flat_mode

    def set_linear_mode(self, enabled: bool) -> None:
        """Toggle Linear Output export UI: hide FORMAT, SIZE and COLOR, which a raw dump
        has no use for, and keep DESTINATION, which it needs exactly as much as print does.
        The Linear panel owns its own format row, so FORMAT would be a second, conflicting one."""
        enabled = bool(enabled)
        if enabled == self._linear_mode:
            return
        self._linear_mode = enabled
        self._format_section.setVisible(not enabled)
        self._size_section.setVisible(not enabled)
        self._color_section.setVisible(not enabled)
        if not enabled:
            # Restore the rows the format and size modes were hiding on their own.
            self._on_fmt_changed()
            self._update_mode_visibility(self._current_mode_value())
            self._update_ratio_visibility()

    def linear_mode(self) -> bool:
        return self._linear_mode

    def _update_output_mode_visibility(self, mode) -> None:
        self._subfolder_container.setVisible(mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE)
        self._abspath_container.setVisible(mode == ExportPresetOutputMode.ABSOLUTE)

    # --- Load / read ---------------------------------------------------------

    def load(self, v: Dict[str, Any]) -> None:
        """Populate all rows from a dict of shared field values."""
        self._loading = True
        try:
            # findData() compares across the QVariant boundary. Item data is a plain str, but a
            # caller may hand back an ExportFormat (StrEnum) member, which never matches there even
            # though it is == the str in Python. Normalize.
            fmt_idx = self.fmt_combo.findData(str(v["export_fmt"]))
            if fmt_idx >= 0:
                self.fmt_combo.setCurrentIndex(fmt_idx)
            depth_idx = self.bit_depth_combo.findData(int(v.get("export_bit_depth", 16)))
            self.bit_depth_combo.setCurrentIndex(depth_idx if depth_idx >= 0 else 1)

            self.quality_spin.setValue(v.get("jpeg_quality", 90))
            self.jpeg_progressive_check.setChecked(v.get("jpeg_progressive", False))

            comp_idx = self.tiff_compression_combo.findData(TiffCompression(v.get("tiff_compression", TiffCompression.ZIP)))
            self.tiff_compression_combo.setCurrentIndex(comp_idx if comp_idx >= 0 else 0)

            self.png_level_spin.setValue(v.get("png_compress_level", 6))

            self.jxl_lossless_check.setChecked(v.get("jxl_lossless", True))
            self.jxl_distance_spin.setValue(v.get("jxl_distance", 1.0))
            self.jxl_distance_spin.setEnabled(not v.get("jxl_lossless", True))
            self.jxl_effort_spin.setValue(v.get("jxl_effort", 7))

            self.webp_quality_spin.setValue(v.get("webp_quality", 90))
            self.webp_lossless_check.setChecked(v.get("webp_lossless", False))
            self.webp_method_spin.setValue(v.get("webp_method", 4))

            self._update_format_visibility(v["export_fmt"])

            self._select_mode_button(v["export_resolution_mode"])
            self._update_mode_visibility(v["export_resolution_mode"])
            self._update_ratio_visibility(v["export_resolution_mode"])
            self.size_input.setValue(v["export_print_size"])
            self.dpi_input.setValue(v["export_dpi"])
            self.target_px_input.setValue(v["export_target_long_edge_px"])
            self.ratio_combo.setCurrentText(v["paper_aspect_ratio"])

            self._export_space = v["export_color_space"]
            _select_data(self.input_combo, v.get("icc_input_path"))
            out_path = v.get("icc_output_path")
            # A stale path (the profile was deleted) falls back to the color space.
            if not (out_path and self.export_profile_combo.findData(out_path) >= 0):
                out_path = None
            _select_data(self.export_profile_combo, out_path or v["export_color_space"])

            mode = _coerce_output_mode(v.get("output_mode"))
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
        profile = self.export_profile_combo.currentData()
        is_space = profile in _EXPORT_SPACES
        return {
            "export_fmt": self.fmt_combo.currentData(),
            "export_bit_depth": int(self.bit_depth_combo.currentData()),
            "jpeg_quality": int(self.quality_spin.value()),
            "jpeg_progressive": self.jpeg_progressive_check.isChecked(),
            "tiff_compression": self.tiff_compression_combo.currentData(),
            "png_compress_level": int(self.png_level_spin.value()),
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
            "export_color_space": profile if is_space else self._export_space,
            "icc_input_path": self.input_combo.currentData(),
            "icc_output_path": None if is_space else profile,
        }
