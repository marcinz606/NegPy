from typing import Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from negpy.desktop.view.styles.theme import THEME
import qtawesome as qta


class CollapsibleSection(QWidget):
    """
    A simple collapsible container with a header button and configurable initial state.
    """

    reset_requested = pyqtSignal()
    expanded_changed = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        icon: Optional[QIcon] = None,
        background_widget: Optional[QWidget] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._title_text = title

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFixedHeight(36)

        # Styled by the QPushButton#collapsible_header rules in modern_dark.qss;
        # overlay="true" = header stacked over a preview widget (translucent bg).
        self.toggle_button.setObjectName("collapsible_header")
        self.toggle_button.setProperty("overlay", "true" if background_widget else "false")

        btn_layout = QHBoxLayout(self.toggle_button)
        btn_layout.setContentsMargins(THEME.space_xl, 8, THEME.space_xl, 8)
        btn_layout.setSpacing(10)

        if icon:
            icon_label = QLabel()
            icon_label.setPixmap(icon.pixmap(14, 14))
            btn_layout.addWidget(icon_label)

        self.title_label = QLabel(self._title_text)
        self.title_label.setStyleSheet(
            f"font-weight: 600; font-size: {THEME.font_size_header}px; letter-spacing: 0.01em; background: transparent;"
        )
        btn_layout.addWidget(self.title_label)

        btn_layout.addStretch()

        self.reset_btn = QPushButton()
        self.reset_btn.setIcon(qta.icon("fa5s.undo", color=THEME.text_muted))
        self.reset_btn.setFixedSize(20, 20)
        self.reset_btn.setIconSize(QSize(10, 10))
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setToolTip(f"Reset {title} to defaults")
        self.reset_btn.setVisible(False)
        self.reset_btn.setObjectName("collapsible_reset_btn")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        btn_layout.addWidget(self.reset_btn)

        self.chevron_label = QLabel()
        self.chevron_label.setStyleSheet("background: transparent;")
        self._update_chevron(expanded)
        btn_layout.addWidget(self.chevron_label)

        if background_widget:
            background_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            header_container = QWidget()
            header_container.setFixedHeight(36)
            stacked = QStackedLayout(header_container)
            stacked.setStackingMode(QStackedLayout.StackingMode.StackAll)
            stacked.setContentsMargins(0, 0, 0, 0)
            stacked.addWidget(background_widget)
            stacked.addWidget(self.toggle_button)
            self.main_layout.addWidget(header_container)
        else:
            self.main_layout.addWidget(self.toggle_button)

        self.content_area = QFrame()
        self.content_area.setObjectName("collapsible_content")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(THEME.space_xl, 4, THEME.space_xl, 8)  # same inset as header
        self.content_layout.setSpacing(4)
        self.content_area.setVisible(expanded)

        self.main_layout.addWidget(self.content_area)

        self.toggle_button.toggled.connect(self._on_toggle)

    def set_content(self, widget: QWidget) -> None:
        # Plain QWidget content gets painted #0D0D0D by the global `QWidget {}`
        # QSS rule, covering the #121212 card frame (custom subclasses aren't
        # auto-painted). The objectName rule forces it transparent either way.
        widget.setObjectName("collapsible_content_body")
        self.content_layout.addWidget(widget)

    def _update_chevron(self, expanded: bool) -> None:
        if expanded:
            self.chevron_label.setPixmap(qta.icon("fa5s.chevron-down", color="#A0A0A0").pixmap(12, 12))
        else:
            self.chevron_label.setPixmap(qta.icon("fa5s.chevron-right", color="#A0A0A0").pixmap(12, 12))

    def set_modified(self, count: int) -> None:
        """Append count to title when non-zero; show reset button."""
        self.modified_count = count
        visible = count > 0
        self.reset_btn.setVisible(visible)
        if visible:
            self.title_label.setText(f"{self._title_text} · {count}")
        else:
            self.title_label.setText(self._title_text)

    def _on_reset_clicked(self) -> None:
        self.reset_requested.emit()

    def _on_toggle(self, checked: bool) -> None:
        self.content_area.setVisible(checked)
        self._update_chevron(checked)
        self.expanded_changed.emit(checked)
