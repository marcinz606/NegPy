from PyQt6.QtWidgets import (
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QLineEdit,
    QColorDialog,
    QDoubleSpinBox,
    QSpinBox,
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import QTimer
import qtawesome as qta
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.domain.models import ColorSpace, AspectRatio, ExportFormat


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

        fmt_row = QHBoxLayout()
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems([f.value for f in ExportFormat])
        self.fmt_combo.setCurrentText(conf.export_fmt)

        self.cs_combo = QComboBox()
        self.cs_combo.addItems([cs.value for cs in ColorSpace] + ["Same as Source"])
        self.cs_combo.setCurrentText(conf.export_color_space)
        fmt_row.addWidget(self.fmt_combo)
        fmt_row.addWidget(self.cs_combo)
        self.layout.addLayout(fmt_row)

        self.ratio_combo = QComboBox()
        # "Original" is first, then the rest
        ratios = [AspectRatio.ORIGINAL] + [r.value for r in AspectRatio if r != AspectRatio.ORIGINAL]
        self.ratio_combo.addItems(ratios)
        self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
        self.layout.addWidget(self.ratio_combo)

        self.orig_res_btn = QPushButton(" Use Original Resolution")
        self.orig_res_btn.setCheckable(True)
        self.orig_res_btn.setChecked(conf.use_original_res)
        self.orig_res_btn.setIcon(qta.icon("fa5s.compress-arrows-alt", color=THEME.text_primary))
        self.layout.addWidget(self.orig_res_btn)

        self.size_container = QWidget()
        size_layout = QVBoxLayout(self.size_container)
        size_layout.setContentsMargins(0, 0, 0, 0)
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
        size_layout.addLayout(print_row)
        self.layout.addWidget(self.size_container)
        self.size_container.setVisible(not conf.use_original_res)

        border_row = QHBoxLayout()
        vbox_border = QVBoxLayout()
        border_label = QLabel('Width <span style="color: #666666; font-size: 10px;">cm</span>')
        vbox_border.addWidget(border_label)
        self.border_input = QDoubleSpinBox()
        self.border_input.setRange(0.0, 10.0)
        self.border_input.setSingleStep(0.1)
        self.border_input.setValue(conf.export_border_size)
        vbox_border.addWidget(self.border_input)

        vbox_color = QVBoxLayout()
        vbox_color.addWidget(QLabel("Color"))
        self.color_btn = QPushButton()
        self.color_btn.setFixedHeight(30)
        self._update_color_btn(conf.export_border_color)
        vbox_color.addWidget(self.color_btn)

        border_row.addLayout(vbox_border)
        border_row.addLayout(vbox_color)
        self.layout.addLayout(border_row)

        self.pattern_input = QLineEdit(conf.filename_pattern)
        self.pattern_input.setPlaceholderText("Filename Pattern...")
        self.pattern_input.setToolTip(
            "Jinja2 Template. Available variables:\n"
            "- {{ original_name }}\n"
            "- {{ colorspace }}\n"
            "- {{ format }} (JPEG/TIFF)\n"
            "- {{ paper_ratio }}\n"
            "- {{ size }} (e.g. 20cm)\n"
            "- {{ dpi }}\n"
            "- {{ border }} ('border' or empty)\n"
            "- {{ date }} (YYYYMMDD)"
        )
        self.layout.addWidget(self.pattern_input)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit(conf.export_path)
        self.browse_btn = QPushButton()
        self.browse_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.browse_btn.setFixedWidth(40)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        self.layout.addLayout(path_layout)

        batch_row = QHBoxLayout()
        self.batch_export_btn = QPushButton(" EXPORT ALL LOADED")
        self.batch_export_btn.setObjectName("batch_export_btn")
        self.batch_export_btn.setFixedHeight(40)
        self.batch_export_btn.setIcon(qta.icon("fa5s.images", color="white"))

        self.apply_all_btn = QPushButton(" Apply to all")
        self.apply_all_btn.setFixedHeight(40)
        self.apply_all_btn.setCheckable(True)
        self.apply_all_btn.setToolTip("Apply current export settings (Size, DPI, Border) to all files")
        self._update_apply_all_style(False)

        batch_row.addWidget(self.batch_export_btn)
        batch_row.addWidget(self.apply_all_btn)
        self.layout.addLayout(batch_row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        # All changes trigger the same debounce timer
        self.fmt_combo.currentTextChanged.connect(lambda _: self.update_timer.start())
        self.cs_combo.currentTextChanged.connect(lambda _: self.update_timer.start())
        self.ratio_combo.currentTextChanged.connect(lambda _: self.update_timer.start())
        self.orig_res_btn.toggled.connect(self._on_orig_res_toggled)

        self.size_input.valueChanged.connect(lambda _: self.update_timer.start())
        self.dpi_input.valueChanged.connect(lambda _: self.update_timer.start())
        self.border_input.valueChanged.connect(lambda _: self.update_timer.start())

        self.color_btn.clicked.connect(self._on_color_clicked)
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        self.pattern_input.textChanged.connect(lambda _: self.update_timer.start())
        self.path_input.textChanged.connect(lambda _: self.update_timer.start())

        self.apply_all_btn.toggled.connect(self._update_apply_all_style)
        self.batch_export_btn.clicked.connect(
            lambda: self.controller.request_batch_export(override_settings=self.apply_all_btn.isChecked())
        )

    def _update_apply_all_style(self, checked: bool) -> None:
        """
        Toggles button highlighting to match the Export All button when active.
        """
        if checked:
            self.apply_all_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME.accent_primary};
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {THEME.accent_secondary};
                }}
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
            use_original_res=self.orig_res_btn.isChecked(),
            export_print_size=self.size_input.value(),
            export_dpi=self.dpi_input.value(),
            export_border_size=self.border_input.value(),
            filename_pattern=self.pattern_input.text(),
            export_path=self.path_input.text(),
        )

    def _on_orig_res_toggled(self, checked: bool) -> None:
        self.size_container.setVisible(not checked)
        self.update_timer.start()

    def _on_color_clicked(self) -> None:
        color = QColorDialog.getColor(QColor(self.state.config.export.export_border_color))
        if color.isValid():
            hex_color = color.name()
            self._update_color_btn(hex_color)
            # Let the timer do it
            self.update_config_section("export", persist=True, render=True, export_border_color=hex_color)

    def _on_browse_clicked(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.state.config.export.export_path)
        if path:
            self.path_input.setText(path)

    def _update_color_btn(self, hex_color: str) -> None:
        self.color_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")

    def sync_ui(self) -> None:
        conf = self.state.config.export
        self.block_signals(True)
        try:
            self.fmt_combo.setCurrentText(conf.export_fmt)
            self.cs_combo.setCurrentText(conf.export_color_space)
            self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
            self.orig_res_btn.setChecked(conf.use_original_res)
            self.size_container.setVisible(not conf.use_original_res)
            self.size_input.setValue(conf.export_print_size)
            self.dpi_input.setValue(conf.export_dpi)
            self.border_input.setValue(conf.export_border_size)
            self._update_color_btn(conf.export_border_color)
            self.pattern_input.setText(conf.filename_pattern)
            self.path_input.setText(conf.export_path)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.fmt_combo,
            self.cs_combo,
            self.ratio_combo,
            self.orig_res_btn,
            self.size_input,
            self.dpi_input,
            self.border_input,
            self.pattern_input,
            self.path_input,
        ]
        for w in widgets:
            w.blockSignals(blocked)
