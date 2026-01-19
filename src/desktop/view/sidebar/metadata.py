from PyQt6.QtWidgets import (
    QLineEdit,
    QSpinBox,
    QFormLayout,
    QTextEdit,
)
from src.desktop.view.sidebar.base import BaseSidebar


class MetadataSidebar(BaseSidebar):
    """
    Panel for recording analog film information.
    """

    def _init_ui(self) -> None:
        self.form = QFormLayout()
        conf = self.state.config.metadata

        self.stock_input = QLineEdit(conf.film_stock)
        self.iso_spin = QSpinBox()
        self.iso_spin.setRange(1, 12800)
        self.iso_spin.setValue(conf.iso)

        self.dev_input = QLineEdit(conf.developer)
        self.dil_input = QLineEdit(conf.dilution)
        self.scan_input = QLineEdit(conf.scan_hardware)

        self.notes_input = QTextEdit()
        self.notes_input.setPlainText(conf.notes)
        self.notes_input.setMaximumHeight(80)

        self.form.addRow("Film Stock:", self.stock_input)
        self.form.addRow("ISO:", self.iso_spin)
        self.form.addRow("Developer:", self.dev_input)
        self.form.addRow("Dilution:", self.dil_input)
        self.form.addRow("Scanner:", self.scan_input)
        self.form.addRow("Notes:", self.notes_input)

        self.layout.addLayout(self.form)

    def _connect_signals(self) -> None:
        self.stock_input.textChanged.connect(
            lambda v: self.update_config_section("metadata", render=False, film_stock=v)
        )
        self.iso_spin.valueChanged.connect(
            lambda v: self.update_config_section("metadata", render=False, iso=v)
        )
        self.dev_input.textChanged.connect(
            lambda v: self.update_config_section("metadata", render=False, developer=v)
        )
        self.dil_input.textChanged.connect(
            lambda v: self.update_config_section("metadata", render=False, dilution=v)
        )
        self.scan_input.textChanged.connect(
            lambda v: self.update_config_section(
                "metadata", render=False, scan_hardware=v
            )
        )
        self.notes_input.textChanged.connect(
            lambda: self.update_config_section(
                "metadata", render=False, notes=self.notes_input.toPlainText()
            )
        )

    def sync_ui(self) -> None:
        conf = self.state.config.metadata
        self.block_signals(True)
        try:
            self.stock_input.setText(conf.film_stock)
            self.iso_spin.setValue(conf.iso)
            self.dev_input.setText(conf.developer)
            self.dil_input.setText(conf.dilution)
            self.scan_input.setText(conf.scan_hardware)
            self.notes_input.setPlainText(conf.notes)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.stock_input,
            self.iso_spin,
            self.dev_input,
            self.dil_input,
            self.scan_input,
            self.notes_input,
        ]
        for w in widgets:
            w.blockSignals(blocked)
