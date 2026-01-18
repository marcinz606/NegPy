from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QLineEdit,
    QColorDialog,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
)
from PyQt6.QtGui import QColor
from src.desktop.view.styles.theme import THEME
from src.desktop.controller import AppController
from src.domain.models import ColorSpace
from src.domain.constants import SUPPORTED_ASPECT_RATIOS


class ExportSidebar(QWidget):
    """
    Panel for export settings and batch processing.
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

        conf = self.state.config.export

        # 1. Format & Color Space
        layout.addWidget(QLabel("<b>Format & Color</b>"))
        fmt_row = QHBoxLayout()
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["JPEG", "TIFF"])
        self.fmt_combo.setCurrentText(conf.export_fmt)

        self.cs_combo = QComboBox()
        self.cs_combo.addItems([cs.value for cs in ColorSpace] + ["Same as Source"])
        self.cs_combo.setCurrentText(conf.export_color_space)
        fmt_row.addWidget(self.fmt_combo)
        fmt_row.addWidget(self.cs_combo)
        layout.addLayout(fmt_row)

        # 2. Paper & Resolution Toggle
        layout.addWidget(QLabel("<b>Sizing & Ratio</b>"))
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["Original"] + SUPPORTED_ASPECT_RATIOS)
        self.ratio_combo.setCurrentText(conf.paper_aspect_ratio)
        layout.addWidget(self.ratio_combo)

        self.orig_res_btn = QPushButton("Use Original Resolution")
        self.orig_res_btn.setCheckable(True)
        self.orig_res_btn.setChecked(conf.use_original_res)
        self._update_orig_res_style(conf.use_original_res)
        layout.addWidget(self.orig_res_btn)

        # 3. Print Size & DPI (Hidden if Original Res)
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

        layout.addWidget(self.size_container)
        self.size_container.setVisible(not conf.use_original_res)

        # 4. Border Settings
        layout.addWidget(QLabel("<b>Border</b>"))
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
        layout.addLayout(border_row)

        # 5. File/Path
        layout.addWidget(QLabel("<b>Output</b>"))
        layout.addWidget(QLabel("Filename Pattern:"))
        self.pattern_input = QLineEdit(conf.filename_pattern)
        layout.addWidget(self.pattern_input)

        layout.addWidget(QLabel("Export Directory:"))
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit(conf.export_path)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(40)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)

        self.batch_export_btn = QPushButton("EXPORT ALL LOADED")
        self.batch_export_btn.setFixedHeight(40)
        self.batch_export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME.accent_green};
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #388e3c;
            }}
        """)
        layout.addWidget(self.batch_export_btn)

    def _connect_signals(self) -> None:
        self.fmt_combo.currentTextChanged.connect(
            lambda v: self._update_export("export_fmt", v)
        )
        self.cs_combo.currentTextChanged.connect(
            lambda v: self._update_export("export_color_space", v)
        )
        self.ratio_combo.currentTextChanged.connect(
            lambda v: self._update_export("paper_aspect_ratio", v)
        )
        self.orig_res_btn.toggled.connect(self._on_orig_res_toggled)

        self.size_input.valueChanged.connect(
            lambda v: self._update_export("export_print_size", v)
        )
        self.dpi_input.valueChanged.connect(
            lambda v: self._update_export("export_dpi", v)
        )
        self.border_input.valueChanged.connect(
            lambda v: self._update_export("export_border_size", v)
        )

        self.color_btn.clicked.connect(self._on_color_clicked)
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        self.pattern_input.textChanged.connect(
            lambda v: self._update_export("filename_pattern", v)
        )
        self.path_input.textChanged.connect(
            lambda v: self._update_export("export_path", v)
        )

        self.batch_export_btn.clicked.connect(self.controller.request_batch_export)

    def _on_orig_res_toggled(self, checked: bool) -> None:
        self._update_orig_res_style(checked)
        self.size_container.setVisible(not checked)
        self._update_export("use_original_res", checked)

    def _update_orig_res_style(self, checked: bool) -> None:
        if checked:
            self.orig_res_btn.setStyleSheet(
                f"background-color: {THEME.accent_primary}; color: white; font-weight: bold;"
            )
        else:
            self.orig_res_btn.setStyleSheet("")

    def _update_export(self, field: str, val: any) -> None:
        from dataclasses import replace

        new_export = replace(self.state.config.export, **{field: val})
        self.controller.session.update_config(
            replace(self.state.config, export=new_export)
        )
        self.controller.request_render()

    def _on_color_clicked(self) -> None:
        color = QColorDialog.getColor(
            QColor(self.state.config.export.export_border_color)
        )
        if color.isValid():
            hex_color = color.name()
            self._update_export("export_border_color", hex_color)
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
        """
        Updates widgets from current state.
        """
        conf = self.state.config.export
        self.fmt_combo.blockSignals(True)
        self.cs_combo.blockSignals(True)
        self.ratio_combo.blockSignals(True)
        self.orig_res_btn.blockSignals(True)
        self.size_input.blockSignals(True)
        self.dpi_input.blockSignals(True)
        self.border_input.blockSignals(True)
        self.pattern_input.blockSignals(True)
        self.path_input.blockSignals(True)

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
            self.fmt_combo.blockSignals(False)
            self.cs_combo.blockSignals(False)
            self.ratio_combo.blockSignals(False)
            self.orig_res_btn.blockSignals(False)
            self.size_input.blockSignals(False)
            self.dpi_input.blockSignals(False)
            self.border_input.blockSignals(False)
            self.pattern_input.blockSignals(False)
            self.path_input.blockSignals(False)
