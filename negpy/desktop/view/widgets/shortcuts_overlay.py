from PyQt6.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QHBoxLayout, QScrollArea, QWidget
from PyQt6.QtCore import Qt
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.shortcut_registry import REGISTRY


class ShortcutsOverlay(QDialog):
    """Modal keyboard shortcut reference, opened with '?'. Reads from REGISTRY."""

    def __init__(self, shortcut_manager, parent=None):
        super().__init__(parent)
        self._shortcut_manager = shortcut_manager
        self.setWindowTitle("Keyboard Shortcuts")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(920, 700)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(0)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {THEME.bg_panel};
                border: 1px solid {THEME.border_primary};
            }}
            QLabel {{
                color: {THEME.text_primary};
                font-size: 12px;
            }}
        """)

        bindings = self._shortcut_manager.bindings
        categories: dict[str, list] = {}
        for action_id, entry in REGISTRY.items():
            categories.setdefault(entry.category, []).append((action_id, entry))

        grouped_categories = list(categories.items())
        left_column, right_column = self._split_categories(grouped_categories)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        columns = QHBoxLayout(container)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(20)

        self._add_category_column(columns, left_column, bindings)
        self._add_category_column(columns, right_column, bindings)

        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)
        root.addSpacing(16)

        actions = QHBoxLayout()
        customize_btn = QPushButton("Customize")
        # Padding-only override: a widget-level `background:` would beat the app
        # stylesheet in every state and kill the global hover/pressed feedback.
        customize_btn.setStyleSheet("font-size: 12px; padding: 6px 20px;")
        customize_btn.clicked.connect(self._customize)
        actions.addWidget(customize_btn)
        actions.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("font-size: 12px; padding: 6px 20px;")
        # Primary action styling from the app stylesheet (accent fill with its own
        # hover/pressed shades), same as the Export panel's call-to-action.
        close_btn.setProperty("primary", True)
        close_btn.clicked.connect(self.accept)
        actions.addWidget(close_btn)
        root.addLayout(actions)

    def _customize(self) -> None:
        if self._shortcut_manager.open_editor(self):
            self.accept()

    def _split_categories(self, grouped_categories: list[tuple[str, list]]) -> tuple[list[tuple[str, list]], list[tuple[str, list]]]:
        left_column: list[tuple[str, list]] = []
        right_column: list[tuple[str, list]] = []
        left_weight = 0
        right_weight = 0

        for category in grouped_categories:
            weight = len(category[1]) + 1
            if left_weight <= right_weight:
                left_column.append(category)
                left_weight += weight
            else:
                right_column.append(category)
                right_weight += weight

        return left_column, right_column

    def _add_category_column(
        self, parent_layout: QHBoxLayout, grouped_categories: list[tuple[str, list]], bindings: dict[str, str]
    ) -> None:
        column_widget = QWidget()
        column_layout = QVBoxLayout(column_widget)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(10)

        for index, (category, entries) in enumerate(grouped_categories):
            if index > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet(f"background-color: {THEME.border_primary}; border: none; margin: 4px 0;")
                sep.setFixedHeight(1)
                column_layout.addWidget(sep)

            cat_lbl = QLabel(category)
            cat_lbl.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 10px; font-weight: bold; padding: 6px 0 2px 0;")
            column_layout.addWidget(cat_lbl)

            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)
            grid.setColumnMinimumWidth(0, 90)
            grid.setColumnMinimumWidth(1, 240)

            for row, (action_id, entry) in enumerate(entries):
                key_lbl = QLabel(bindings.get(action_id, ""))
                key_lbl.setStyleSheet(f"""
                    color: {THEME.text_primary};
                    background-color: {THEME.bg_header};
                    border: 1px solid {THEME.border_primary};
                    border-radius: 3px;
                    font-family: monospace;
                    font-size: 11px;
                    padding: 1px 5px;
                """)
                key_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                desc_lbl = QLabel(entry.description)
                desc_lbl.setWordWrap(True)
                desc_lbl.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 12px; padding-left: 4px;")
                grid.addWidget(key_lbl, row, 0, alignment=Qt.AlignmentFlag.AlignTop)
                grid.addWidget(desc_lbl, row, 1)

            column_layout.addLayout(grid)

        column_layout.addStretch()
        parent_layout.addWidget(column_widget, stretch=1)
