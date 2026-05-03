from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar, QSizePolicy
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
                font-size: {THEME.font_size_xs}px;
                font-weight: 500;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(16)

        self.msg_label = QLabel("ready")
        layout.addWidget(self.msg_label)

        self.tool_label = QLabel("")
        layout.addWidget(self.tool_label)

        layout.addStretch()

        self.zoom_label = QLabel("")
        self.dims_label = QLabel("")
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.dims_label)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(3)
        self.progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: transparent;
                border: none;
                border-top: 3px solid {THEME.accent_primary};
                border-radius: 0;
            }}
            QProgressBar::chunk {{
                background-color: {THEME.accent_primary};
                border-radius: 0;
            }}
        """)
        layout.addWidget(self.progress)

    def showMessage(self, text: str, timeout: int = 0):
        if text == "Image Updated":
            return
        self.msg_label.setText(text.lower())
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.msg_label.setText("ready"))

    def set_right_cluster(self, zoom: str, dims: str, tool: str) -> None:
        self.zoom_label.setText(zoom)
        self.dims_label.setText(dims)
        self.tool_label.setText(tool)

    def set_progress(self, current: int, total: int):
        if total <= 0:
            self.progress.setVisible(False)
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        if current >= total:
            QTimer.singleShot(1000, lambda: self.progress.setVisible(False))
