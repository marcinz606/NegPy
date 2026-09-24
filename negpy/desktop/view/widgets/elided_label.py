from PyQt6.QtCore import QEvent, QSize, Qt
from PyQt6.QtGui import QHelpEvent, QPainter
from PyQt6.QtWidgets import QLabel, QToolTip

from negpy.desktop.view.styles.templates import wrap_tooltip


class ElidedLabel(QLabel):
    """Single-line label that shrinks below its text, ending it in … and showing the
    full text on hover. `text()` stays the full text."""

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def is_elided(self) -> bool:
        return self.fontMetrics().horizontalAdvance(self.text()) > self.contentsRect().width()

    def paintEvent(self, a0) -> None:
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, rect.width())
        QPainter(self).drawText(rect, int(self.alignment()), text)

    def event(self, e) -> bool:
        if isinstance(e, QHelpEvent) and e.type() == QEvent.Type.ToolTip and self.is_elided():
            own = self.toolTip().removeprefix("<qt>").removesuffix("</qt>")
            QToolTip.showText(e.globalPos(), wrap_tooltip(self.text(), footer=f"<br><br>{own}" if own else ""), self)
            return True
        return super().event(e)
