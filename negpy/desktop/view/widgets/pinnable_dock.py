import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QHBoxLayout, QLabel, QToolButton, QWidget

from negpy.desktop.view.styles.theme import THEME


class PinnableDockWidget(QDockWidget):
    """QDockWidget that shows a pin button in the title bar while floating."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None,
        *,
        pin_tooltip: str,
        on_pin,
    ) -> None:
        super().__init__(title, parent)
        self._pin_tooltip = pin_tooltip
        self._on_pin = on_pin
        self._floating_title: QWidget | None = None
        self._title_label: QLabel | None = None
        self.topLevelChanged.connect(self._on_top_level_changed)

    def _on_top_level_changed(self, floating: bool) -> None:
        if floating:
            self._show_pin_title_bar()
        else:
            self.setTitleBarWidget(None)

    def _show_pin_title_bar(self) -> None:
        if self._floating_title is None:
            bar = QWidget()
            layout = QHBoxLayout(bar)
            layout.setContentsMargins(8, 4, 4, 4)
            layout.setSpacing(6)

            label = QLabel(self.windowTitle())
            label.setStyleSheet(f"color: {THEME.text_primary}; font-weight: bold; background: transparent;")

            pin = QToolButton()
            pin.setIcon(qta.icon("fa5s.thumbtack", color=THEME.text_primary))
            pin.setToolTip(self._pin_tooltip)
            pin.setCursor(Qt.CursorShape.PointingHandCursor)
            pin.setStyleSheet("QToolButton { border: none; background: transparent; padding: 2px; }")
            pin.clicked.connect(self._on_pin)

            layout.addWidget(label)
            layout.addStretch()
            layout.addWidget(pin)

            self._floating_title = bar
            self._title_label = label
        elif self._title_label is not None:
            self._title_label.setText(self.windowTitle())

        self.setTitleBarWidget(self._floating_title)
