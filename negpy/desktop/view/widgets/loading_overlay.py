import qtawesome as qta
from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PyQt6.QtWidgets import QLabel, QToolButton, QVBoxLayout, QWidget

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
        # Integer alpha: QSS float alpha is not reliably parsed across Qt versions.
        self.setStyleSheet("background-color: rgba(5, 5, 5, 115);")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        self._spinner = QToolButton()
        self._spinner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._spinner.setStyleSheet("QToolButton { border: none; background: transparent; }")
        self._spin = qta.Spin(self._spinner, interval=12, step=12)
        self._spinner.setIcon(qta.icon("fa5s.circle-notch", color=THEME.accent_primary, animation=self._spin))
        self._spinner.setIconSize(QSize(52, 52))
        layout.addWidget(self._spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel("processing…")
        self._label.setStyleSheet(
            f"color: {THEME.text_primary}; font-size: {THEME.font_size_lg}px; font-weight: 600; "
            "background-color: rgba(10, 10, 10, 225); border: 1px solid rgba(255, 255, 255, 55); "
            "border-radius: 6px; padding: 5px 14px;"
        )
        layout.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Track the canvas size ourselves: relying on the main window's resizeEvent
        # leaves the overlay at its default (0,0) sizeHint rect until the first
        # window resize — the spinner then huddles in the top-left corner.
        parent.installEventFilter(self)

        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self.stop)

        self.hide()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self.parent() and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.setGeometry(self.parent().rect())
        return False

    def start(self) -> None:
        self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()
        self._timeout.start(self._MAX_VISIBLE_MS)

    def stop(self) -> None:
        self._timeout.stop()
        self.hide()
