from negpy.kernel.system.i18n import tr
from typing import Optional

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import with_alpha
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.charts import ON_PLATE

_SIZE = QSize(300, 180)
_REST_ALPHA = 165


class AnalysisPlate(QWidget):
    """The Analysis charts in a canvas corner while their card is closed: translucent at rest,
    solid under the pointer. A click asks for the next chart."""

    cycle_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._charts: Optional[QStackedWidget] = None
        self._blocked = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        self.setFixedSize(_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("Analysis Overlay: click to show the next chart"))
        self.hide()

    def holds_charts(self) -> bool:
        return self._charts is not None

    def hold(self, charts: QStackedWidget) -> None:
        self._charts = charts
        self._mark_pages(True)
        # The app sheet gives every widget an opaque ground, the stack included.
        charts.setStyleSheet("background: transparent;")
        self.layout().addWidget(charts)
        charts.show()
        self._sync()

    def release(self) -> None:
        """Hides the plate. The charts stay its child until their owner takes them back."""
        self._mark_pages(False)
        assert self._charts is not None
        self._charts.setStyleSheet("")
        self._charts = None
        self._sync()

    def set_blocked(self, blocked: bool) -> None:
        self._blocked = blocked
        self._sync()

    def _mark_pages(self, on_plate: bool) -> None:
        assert self._charts is not None
        for i in range(self._charts.count()):
            self._charts.widget(i).setProperty(ON_PLATE, on_plate)

    def _sync(self) -> None:
        self.setVisible(self._charts is not None and not self._blocked)

    def enterEvent(self, event) -> None:  # noqa: N802
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.cycle_requested.emit()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(THEME.border_primary), 1))
        painter.setBrush(with_alpha(THEME.bg_input, 255 if self.underMouse() else _REST_ALPHA))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), THEME.radius_md, THEME.radius_md)
