from PyQt6.QtWidgets import (
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QLineEdit,
    QColorDialog,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtGui import QColor
import qtawesome as qta
from src.desktop.view.styles.theme import THEME
from src.desktop.view.sidebar.base import BaseSidebar
from src.domain.models import ColorSpace, AspectRatio, ExportFormat


class ExportSidebar(BaseSidebar):
    """
    Panel for export settings and batch processing.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)
        conf = self.state.config.export

        label_fmt = QLabel("Format & Color")
        label_fmt.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold;"
        )
        self.layout.addWidget(label_fmt)

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

        label_size = QLabel("Sizing & Ratio")
        label_size.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        self.layout.addWidget(label_size)

        self.ratio_combo = QComboBox()
        # "Original" is first, then the rest
        ratios = [AspectRatio.ORIGINAL] + [
            r.value for r in AspectRatio if r != AspectRatio.ORIGINAL
        ]
        self.ratio_combo.addItems(ratios)
        self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
        self.layout.addWidget(self.ratio_combo)

        self.orig_res_btn = QPushButton(" Use Original Resolution")
        self.orig_res_btn.setCheckable(True)
        self.orig_res_btn.setChecked(conf.use_original_res)
        self.orig_res_btn.setIcon(
            qta.icon("fa5s.compress-arrows-alt", color=THEME.text_primary)
        )
        self._update_orig_res_style(conf.use_original_res)
        self.layout.addWidget(self.orig_res_btn)

        self.size_container = QWidget()
        size_layout = QVBoxLayout(self.size_container)
        size_layout.setContentsMargins(0, 0, 0, 0)
        print_row = QHBoxLayout()

        vbox_size = QVBoxLayout()
        vbox_size.addWidget(QLabel("Size (cm)"))
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

        label_border = QLabel("Border")
        label_border.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        self.layout.addWidget(label_border)

        border_row = QHBoxLayout()
        vbox_border = QVBoxLayout()
        vbox_border.addWidget(QLabel("Width (cm)"))
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

        # Output Path & Batch
        label_out = QLabel("Output")
        label_out.setStyleSheet(
            f"font-size: {THEME.font_size_header}px; font-weight: bold; margin-top: 5px;"
        )
        self.layout.addWidget(label_out)

        self.pattern_input = QLineEdit(conf.filename_pattern)
        self.pattern_input.setPlaceholderText("Filename Pattern...")
        self.layout.addWidget(self.pattern_input)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit(conf.export_path)
        self.browse_btn = QPushButton()
        self.browse_btn.setIcon(qta.icon("fa5s.folder-open", color=THEME.text_primary))
        self.browse_btn.setFixedWidth(40)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        self.layout.addLayout(path_layout)

        self.batch_export_btn = QPushButton(" EXPORT ALL LOADED")
        self.batch_export_btn.setFixedHeight(40)
        self.batch_export_btn.setIcon(qta.icon("fa5s.images", color="white"))
        self.batch_export_btn.setStyleSheet(f"""
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
        self.layout.addWidget(self.batch_export_btn)

    def _connect_signals(self) -> None:
        self.fmt_combo.currentTextChanged.connect(
            lambda v: self.update_config_section("export", export_fmt=v)
        )
        self.cs_combo.currentTextChanged.connect(
            lambda v: self.update_config_section("export", export_color_space=v)
        )
        self.ratio_combo.currentTextChanged.connect(
            lambda v: self.update_config_section("export", paper_aspect_ratio=v)
        )
        self.orig_res_btn.toggled.connect(self._on_orig_res_toggled)

        self.size_input.valueChanged.connect(
            lambda v: self.update_config_section("export", export_print_size=v)
        )
        self.dpi_input.valueChanged.connect(
            lambda v: self.update_config_section("export", export_dpi=v)
        )
        self.border_input.valueChanged.connect(
            lambda v: self.update_config_section("export", export_border_size=v)
        )

        self.color_btn.clicked.connect(self._on_color_clicked)
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        self.pattern_input.textChanged.connect(
            lambda v: self.update_config_section("export", filename_pattern=v)
        )
        self.path_input.textChanged.connect(
            lambda v: self.update_config_section("export", export_path=v)
        )
        self.batch_export_btn.clicked.connect(self.controller.request_batch_export)

    def _on_orig_res_toggled(self, checked: bool) -> None:
        self._update_orig_res_style(checked)
        self.size_container.setVisible(not checked)
        self.update_config_section("export", use_original_res=checked)

    def _update_orig_res_style(self, checked: bool) -> None:
        if checked:
            self.orig_res_btn.setStyleSheet(
                f"background-color: {THEME.accent_primary}; color: white; font-weight: bold;"
            )
        else:
            self.orig_res_btn.setStyleSheet("")

    def _on_color_clicked(self) -> None:
        color = QColorDialog.getColor(
            QColor(self.state.config.export.export_border_color)
        )
        if color.isValid():
            hex_color = color.name()
            self.update_config_section("export", export_border_color=hex_color)
            self._update_color_btn(hex_color)

    def _on_browse_clicked(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", self.state.config.export.export_path
        )
        if path:
            self.path_input.setText(path)

    def _update_color_btn(self, hex_color: str) -> None:
        self.color_btn.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #555;"
        )

    def sync_ui(self) -> None:
        conf = self.state.config.export
        self.block_signals(True)
        try:
            self.fmt_combo.setCurrentText(conf.export_fmt)
            self.cs_combo.setCurrentText(conf.export_color_space)
            self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
            self.orig_res_btn.setChecked(conf.use_original_res)
            self._update_orig_res_style(conf.use_original_res)
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
