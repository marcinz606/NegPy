from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QProgressBar,
)
from PyQt6.QtCore import QTimer


class TopStatusBar(QWidget):
    """Integrated status bar at the top of the viewport."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        self.msg_label = QLabel("Ready")
        self.msg_label.setStyleSheet("color: #aaa; font-size: 16px;")

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(150)
        self.progress.setFixedHeight(12)
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            """
            QProgressBar { background-color: #222; border: 1px solid #333; border-radius: 6px; }
            QProgressBar::chunk { background-color: {THEME.accent_primary}; border-radius: 5px; }
        """
        )

        layout.addWidget(self.msg_label)
        layout.addStretch()
        layout.addWidget(self.progress)

    def showMessage(self, text: str, timeout: int = 0):
        if text == "Image Updated":
            return
        self.msg_label.setText(text)
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.msg_label.setText("Ready"))
