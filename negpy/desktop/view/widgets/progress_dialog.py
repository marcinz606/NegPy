import qtawesome as qta
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.theme import THEME


class ProgressDialog(QDialog):
    """
    Non-modal floating popup showing animated progress for a batch job
    (export, analysis, thumbnails). Sits over the main window without blocking it.
    """

    abort_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Working…")
        self.setModal(False)
        self.setFixedWidth(360)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {THEME.bg_panel}; border: 1px solid {THEME.border_primary}; }}
            QLabel {{ color: {THEME.text_primary}; }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        header = QHBoxLayout()
        header.setSpacing(THEME.space_xl)

        self._spinner = QToolButton()
        self._spinner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._spinner.setStyleSheet("QToolButton { border: none; background: transparent; }")
        self._spin = qta.Spin(self._spinner, interval=12, step=12)
        self._spinner.setIcon(qta.icon("fa5s.circle-notch", color=THEME.text_primary, animation=self._spin))
        self._spinner.setIconSize(QSize(24, 24))
        header.addWidget(self._spinner, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._title = QLabel("")
        self._title.setStyleSheet(f"font-size: {THEME.font_size_header}px; font-weight: {THEME.weight_semibold};")
        header.addWidget(self._title, alignment=Qt.AlignmentFlag.AlignVCenter)
        header.addStretch(1)
        root.addLayout(header)

        self._file_label = QLabel("")
        self._file_label.setStyleSheet(f"font-size: {THEME.font_size_xs}px; color: {THEME.text_secondary};")
        root.addWidget(self._file_label)

        self._bar = QProgressBar()
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(f"""
            QProgressBar {{ background-color: {THEME.border_primary}; border: none; border-radius: {THEME.radius_sm}px; }}
            QProgressBar::chunk {{ background-color: {THEME.accent_primary}; border-radius: {THEME.radius_sm}px; }}
        """)
        root.addWidget(self._bar)

        footer = QHBoxLayout()
        self._count = QLabel("")
        self._count.setStyleSheet(f"font-size: {THEME.font_size_xs}px; color: {THEME.text_muted};")
        footer.addWidget(self._count)
        footer.addStretch(1)

        self._abort = QPushButton("Abort")
        self._abort.setStyleSheet("QPushButton { padding: 6px 14px; }")
        self._abort.clicked.connect(self._on_abort)
        footer.addWidget(self._abort)
        root.addLayout(footer)

    def start(self, title: str, abortable: bool) -> None:
        """Reset and show the popup for a new batch job."""
        self._title.setText(title)
        self._file_label.setText("")
        self._count.setText("")
        self._bar.setRange(0, 0)  # indeterminate until first progress
        self._abort.setVisible(abortable)
        self._abort.setEnabled(True)
        self._abort.setText("Abort")
        self.show()
        self.raise_()

    def set_progress(self, current: int, total: int, label: str) -> None:
        if total > 0:
            if self._bar.maximum() != total:
                self._bar.setRange(0, total)
            self._bar.setValue(current)
            self._count.setText(f"{current}/{total}")
        self._file_label.setText(label)

    def finish(self) -> None:
        self.hide()

    def _on_abort(self) -> None:
        self._abort.setEnabled(False)
        self._abort.setText("Aborting…")
        self.abort_requested.emit()
