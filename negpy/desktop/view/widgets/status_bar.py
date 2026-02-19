from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar, QFrame
from PyQt6.QtCore import QTimer
from negpy.desktop.view.styles.theme import THEME


class TopStatusBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setObjectName("StatusDashboard")

        self.setStyleSheet(f"""
            QWidget#StatusDashboard {{
                background-color: {THEME.bg_status_bar if hasattr(THEME, "bg_status_bar") else "#111"};
                border-bottom: 1px solid {THEME.border_primary};
            }}
            QLabel {{
                color: {THEME.text_secondary};
                font-size: 11px;
                font-weight: 500;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(16)

        # Left: Activity
        self.msg_label = QLabel("Ready")
        layout.addWidget(self.msg_label)

        # Center: Progress
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setFixedHeight(4)
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #121212;
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {THEME.accent_primary};
                border-radius: 2px;
            }}
        """)
        layout.addStretch()
        layout.addWidget(self.progress)
        layout.addStretch()

        # Right: System Info
        self.system_info = QHBoxLayout()
        self.system_info.setSpacing(12)

        self.gpu_label = QLabel("CPU")
        self.gpu_label.setStyleSheet(f"color: {THEME.text_muted};")

        self.system_info.addWidget(self._create_separator())
        self.system_info.addWidget(self.gpu_label)

        layout.addLayout(self.system_info)

    def _create_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setFixedWidth(1)
        line.setFixedHeight(12)
        line.setStyleSheet(f"background-color: {THEME.border_primary}; border: none;")
        return line

    def showMessage(self, text: str, timeout: int = 0):
        if text == "Image Updated":
            return
        self.msg_label.setText(text.upper())
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.msg_label.setText("READY"))

    def set_gpu_info(self, backend: str, active: bool = False):
        self.gpu_label.setText(backend.upper())
        color = THEME.accent_primary if active else THEME.text_muted
        self.gpu_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def set_progress(self, current: int, total: int):
        if total <= 0:
            self.progress.setVisible(False)
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        if current >= total:
            QTimer.singleShot(1000, lambda: self.progress.setVisible(False))
