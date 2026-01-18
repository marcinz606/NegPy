from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QSpinBox,
    QFormLayout,
    QTextEdit,
)
from src.desktop.controller import AppController


class MetadataSidebar(QWidget):
    """
    Panel for recording analog film information.
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

        form = QFormLayout()

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

        form.addRow("Film Stock:", self.stock_input)
        form.addRow("ISO:", self.iso_spin)
        form.addRow("Developer:", self.dev_input)
        form.addRow("Dilution:", self.dil_input)
        form.addRow("Scanner:", self.scan_input)
        form.addRow("Notes:", self.notes_input)

        layout.addLayout(form)

    def _connect_signals(self) -> None:
        self.stock_input.textChanged.connect(
            lambda v: self._update_meta("film_stock", v)
        )
        self.iso_spin.valueChanged.connect(lambda v: self._update_meta("iso", v))
        self.dev_input.textChanged.connect(lambda v: self._update_meta("developer", v))
        self.dil_input.textChanged.connect(lambda v: self._update_meta("dilution", v))
        self.scan_input.textChanged.connect(
            lambda v: self._update_meta("scan_hardware", v)
        )
        self.notes_input.textChanged.connect(
            lambda: self._update_meta("notes", self.notes_input.toPlainText())
        )

    def _update_meta(self, field: str, val: any) -> None:
        from dataclasses import replace

        new_meta = replace(self.state.config.metadata, **{field: val})
        self.controller.session.update_config(
            replace(self.state.config, metadata=new_meta)
        )
        self.controller.request_render()

    def sync_ui(self) -> None:
        """
        Updates widgets from current state.
        """
        conf = self.state.config.metadata
        self.blockSignals(True)
        self.stock_input.blockSignals(True)
        self.iso_spin.blockSignals(True)
        self.dev_input.blockSignals(True)
        self.dil_input.blockSignals(True)
        self.scan_input.blockSignals(True)
        self.notes_input.blockSignals(True)

        try:
            self.stock_input.setText(conf.film_stock)
            self.iso_spin.setValue(conf.iso)
            self.dev_input.setText(conf.developer)
            self.dil_input.setText(conf.dilution)
            self.scan_input.setText(conf.scan_hardware)
            self.notes_input.setPlainText(conf.notes)
        finally:
            self.stock_input.blockSignals(False)
            self.iso_spin.blockSignals(False)
            self.dev_input.blockSignals(False)
            self.dil_input.blockSignals(False)
            self.scan_input.blockSignals(False)
            self.notes_input.blockSignals(False)
            self.blockSignals(False)
        self.controller.request_render()
