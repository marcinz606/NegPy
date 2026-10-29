from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.styles.theme import THEME


class LoadingOverlay(QWidget):
    """
    Translucent scrim over the canvas while a file loads. Keeps the previous frame visible
    (dimmed) instead of blanking the canvas; the canvas strip carries the words.
    """

    # Backstop so the scrim can never get stuck if no render/error arrives.
    MAX_VISIBLE_MS = 8000

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Integer alpha: QSS float alpha is not reliably parsed across Qt versions.
        self.setStyleSheet(f"background-color: {THEME.surface_scrim};")

        # Track the canvas size ourselves. Relying on the main window's resizeEvent leaves the
        # overlay at its default (0,0) sizeHint rect until the first window resize.
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
        self._timeout.start(self.MAX_VISIBLE_MS)

    def stop(self) -> None:
        self._timeout.stop()
        self.hide()
