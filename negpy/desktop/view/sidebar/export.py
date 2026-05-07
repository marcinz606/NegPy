import qtawesome as qta
from PyQt6.QtCore import QTimer

from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.domain.models import AspectRatio, ColorSpace, ExportFormat, ExportResolutionMode


class ExportSidebar(BaseSidebar):
    """
    Panel for export settings and batch processing.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)
        conf = self.state.config.export

        # Debounce timer for all export settings
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(1000)
        self.update_timer.timeout.connect(self._persist_all_export_settings)

        self.layout.addWidget(section_subheader("FORMAT"))

        fmt_row = QHBoxLayout()
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems([f.value for f in ExportFormat])
        self.fmt_combo.setCurrentText(conf.export_fmt)
        self.fmt_combo.setToolTip("File format")

        self.cs_combo = QComboBox()
        self.cs_combo.addItems([cs.value for cs in ColorSpace])
        self.cs_combo.setCurrentText(conf.export_color_space)
        self.cs_combo.setToolTip("Output color space")
        fmt_row.addWidget(self.fmt_combo)
        fmt_row.addWidget(self.cs_combo)
        self.layout.addLayout(fmt_row)

        self.ratio_combo = QComboBox()
        # "Original" is first, then the rest
        ratios = [AspectRatio.ORIGINAL] + [r.value for r in AspectRatio if r != AspectRatio.ORIGINAL]
        self.ratio_combo.addItems(ratios)
        self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
        self.ratio_combo.setToolTip("Paper aspect ratio")
        self.layout.addWidget(self.ratio_combo)

        self.layout.addWidget(section_subheader("RESOLUTION"))

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
        self.layout.addLayout(mode_row)

        # PRINT mode: cm + DPI
        self.print_size_container = QWidget()
        print_layout = QVBoxLayout(self.print_size_container)
        print_layout.setContentsMargins(0, 0, 0, 0)
        print_row = QHBoxLayout()

        vbox_size = QVBoxLayout()
        size_label = QLabel('Size <span style="color: #666666; font-size: 10px;">cm</span>')
        vbox_size.addWidget(size_label)
        self.size_input = QDoubleSpinBox()
        self.size_input.setRange(1.0, 500.0)
        self.size_input.setValue(conf.export_print_size)
        vbox_size.addWidget(self.size_input)

        vbox_dpi = QVBoxLayout()
        vbox_dpi.addWidget(QLabel("DPI"))
        self.dpi_input = QSpinBox()
        self.dpi_input.setRange(72, 4800)
        self.dpi_input.setValue(conf.export_dpi)
        vbox_dpi.addWidget(self.dpi_input)

        print_row.addLayout(vbox_size)
        print_row.addLayout(vbox_dpi)
        print_layout.addLayout(print_row)
        self.layout.addWidget(self.print_size_container)

        # TARGET_PX mode: long edge in pixels
        self.target_px_container = QWidget()
        target_px_layout = QVBoxLayout(self.target_px_container)
        target_px_layout.setContentsMargins(0, 0, 0, 0)
        target_px_layout.addWidget(QLabel('Long edge <span style="color: #666666; font-size: 10px;">px</span>'))
        self.target_px_input = QSpinBox()
        self.target_px_input.setRange(256, 32768)
        self.target_px_input.setValue(conf.export_target_long_edge_px)
        target_px_layout.addWidget(self.target_px_input)
        self.layout.addWidget(self.target_px_container)

        self._select_mode_button(conf.export_resolution_mode)
        self._update_mode_visibility(conf.export_resolution_mode)

        self.layout.addWidget(section_subheader("DESTINATION"))

        self.pattern_input = QLineEdit(conf.filename_pattern)
        self.pattern_input.setPlaceholderText("Filename Pattern...")
        self.pattern_input.setToolTip(
            "Jinja2 Template. Available variables:\n"
            "- {{ original_name }}\n"
            "- {{ colorspace }}\n"
            "- {{ format }} (JPEG/TIFF)\n"
            "- {{ paper_ratio }}\n"
            "- {{ size }} (e.g. 20cm; PRINT mode only)\n"
            "- {{ dpi }} (PRINT mode only)\n"
            "- {{ target_px }} (e.g. 2000px; TARGET_PX mode only)\n"
            "- {{ border }} ('border' or empty)\n"
            "- {{ date }} (YYYYMMDD)"
        )
        self.layout.addWidget(self.pattern_input)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit(conf.export_path)
        self.path_input.setToolTip("Export folder")
        self.browse_btn = QPushButton()
        self.browse_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.browse_btn.setFixedWidth(40)
        self.browse_btn.setToolTip("Choose export folder")
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        self.layout.addLayout(path_layout)

        self.layout.addWidget(section_subheader("BATCH"))

        batch_row = QHBoxLayout()
        self.batch_export_btn = QPushButton(" Export All")
        self.batch_export_btn.setObjectName("batch_export_btn")
        self.batch_export_btn.setFixedHeight(40)
        self.batch_export_btn.setIcon(qta.icon("fa5s.images", color=THEME.text_primary))

        self.apply_all_btn = QPushButton(" Sync export settings")
        self.apply_all_btn.setFixedHeight(40)
        self.apply_all_btn.setCheckable(True)
        self.apply_all_btn.setChecked(True)
        self.apply_all_btn.setToolTip("Apply current export settings (Size, DPI, Border) to all files")
        self._update_apply_all_style(True)

        batch_row.addWidget(self.batch_export_btn)
        batch_row.addWidget(self.apply_all_btn)
        self.layout.addLayout(batch_row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        # All changes trigger the same debounce timer
        self.fmt_combo.currentTextChanged.connect(lambda _: self.update_timer.start())
        self.cs_combo.currentTextChanged.connect(lambda _: self.update_timer.start())
        self.ratio_combo.currentTextChanged.connect(lambda _: self.update_timer.start())
        self.mode_btn_group.idToggled.connect(self._on_mode_toggled)

        self.size_input.valueChanged.connect(lambda _: self.update_timer.start())
        self.dpi_input.valueChanged.connect(lambda _: self.update_timer.start())
        self.target_px_input.valueChanged.connect(lambda _: self.update_timer.start())

        self.browse_btn.clicked.connect(self._on_browse_clicked)
        self.pattern_input.textChanged.connect(lambda _: self.update_timer.start())
        self.path_input.textChanged.connect(lambda _: self.update_timer.start())

        self.apply_all_btn.toggled.connect(self._update_apply_all_style)
        self.batch_export_btn.clicked.connect(
            lambda: self.controller.request_batch_export(override_settings=self.apply_all_btn.isChecked())
        )

    def _update_apply_all_style(self, checked: bool) -> None:
        """Toggle checked appearance for the Sync export settings button."""
        if checked:
            self.apply_all_btn.setStyleSheet("""
                QPushButton {
                    background-color: #222222;
                    color: white;
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 4px;
                }
            """)
            self.apply_all_btn.setIcon(qta.icon("fa5s.clone", color="white"))
        else:
            self.apply_all_btn.setStyleSheet("font-weight: bold;")
            self.apply_all_btn.setIcon(qta.icon("fa5s.clone", color=THEME.text_primary))

    def _persist_all_export_settings(self) -> None:
        """Collects all UI values and performs a single debounced config update."""
        self.update_config_section(
            "export",
            persist=True,
            render=True,
            export_fmt=self.fmt_combo.currentText(),
            export_color_space=self.cs_combo.currentText(),
            paper_aspect_ratio=self.ratio_combo.currentText(),
            export_resolution_mode=self._current_mode_value(),
            export_print_size=self.size_input.value(),
            export_dpi=self.dpi_input.value(),
            export_target_long_edge_px=self.target_px_input.value(),
            filename_pattern=self.pattern_input.text(),
            export_path=self.path_input.text(),
        )

    _MODE_BY_ID = {
        0: ExportResolutionMode.ORIGINAL.value,
        1: ExportResolutionMode.PRINT.value,
        2: ExportResolutionMode.TARGET_PX.value,
    }
    _ID_BY_MODE = {v: k for k, v in _MODE_BY_ID.items()}

    def _current_mode_value(self) -> str:
        return self._MODE_BY_ID.get(self.mode_btn_group.checkedId(), ExportResolutionMode.PRINT.value)

    def _select_mode_button(self, mode_value: str) -> None:
        idx = self._ID_BY_MODE.get(mode_value, 1)
        btn = self.mode_btn_group.button(idx)
        if btn is not None:
            btn.setChecked(True)

    def _update_mode_visibility(self, mode_value: str) -> None:
        self.print_size_container.setVisible(mode_value == ExportResolutionMode.PRINT.value)
        self.target_px_container.setVisible(mode_value == ExportResolutionMode.TARGET_PX.value)

    def _on_mode_toggled(self, _id: int, checked: bool) -> None:
        if not checked:
            return
        self._update_mode_visibility(self._current_mode_value())
        self.update_timer.start()

    def _on_browse_clicked(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.state.config.export.export_path)
        if path:
            self.path_input.setText(path)

    def sync_ui(self) -> None:
        conf = self.state.config.export
        self.block_signals(True)
        try:
            self.fmt_combo.setCurrentText(conf.export_fmt)
            self.cs_combo.setCurrentText(conf.export_color_space)
            self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
            self._select_mode_button(conf.export_resolution_mode)
            self._update_mode_visibility(conf.export_resolution_mode)
            self.size_input.setValue(conf.export_print_size)
            self.dpi_input.setValue(conf.export_dpi)
            self.target_px_input.setValue(conf.export_target_long_edge_px)
            self.pattern_input.setText(conf.filename_pattern)
            self.path_input.setText(conf.export_path)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.fmt_combo,
            self.cs_combo,
            self.ratio_combo,
            self.mode_original_btn,
            self.mode_print_btn,
            self.mode_target_px_btn,
            self.size_input,
            self.dpi_input,
            self.target_px_input,
            self.pattern_input,
            self.path_input,
        ]
        for w in widgets:
            w.blockSignals(blocked)
