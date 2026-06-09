import qtawesome as qta
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import QToolButton, QVBoxLayout, QWidget

from negpy.desktop.view.styles.theme import THEME


class LoadingOverlay(QWidget):
    """
    Translucent scrim with a centred spinner, shown over the canvas while a file loads.
    Keeps the previous frame visible (dimmed) instead of blanking the canvas.
    """

    # Backstop so the spinner can never get stuck if no render/error arrives.
    _MAX_VISIBLE_MS = 8000

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background-color: rgba(5, 5, 5, 0.45);")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._spinner = QToolButton()
        self._spinner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._spinner.setStyleSheet("QToolButton { border: none; background: transparent; }")
        self._spin = qta.Spin(self._spinner, interval=12, step=12)
        self._spinner.setIcon(qta.icon("fa5s.circle-notch", color=THEME.text_primary, animation=self._spin))
        self._spinner.setIconSize(QSize(36, 36))
        layout.addWidget(self._spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self.stop)

        self.hide()

    def start(self) -> None:
        self.show()
        self.raise_()
        self._timeout.start(self._MAX_VISIBLE_MS)

    def stop(self) -> None:
        self._timeout.stop()
        self.hide()
