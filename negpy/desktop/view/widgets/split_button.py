import qtawesome as qta
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QHBoxLayout, QMenu, QPushButton, QSizePolicy, QToolButton, QWidget

from negpy.desktop.view.styles.theme import THEME


def make_split_button(text: str, icon_name: str, menu: QMenu, *, primary: bool = False) -> tuple[QWidget, QPushButton, QToolButton]:
    """Fused main-action button + menu-arrow pair, styled via the shared
    split_main_btn / split_menu_btn QSS rules. ``primary`` marks both halves
    with the accent [primary="true"] style."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)

    icon_color = "white" if primary else THEME.text_primary
    main_btn = QPushButton(text)
    main_btn.setObjectName("split_main_btn")
    main_btn.setIcon(qta.icon(icon_name, color=icon_color))
    main_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    main_btn.setFixedHeight(36)

    menu_btn = QToolButton()
    menu_btn.setObjectName("split_menu_btn")
    menu_btn.setAutoRaise(False)
    menu_btn.setIcon(qta.icon("fa5s.chevron-down", color=icon_color))
    menu_btn.setIconSize(QSize(18, 18))
    menu_btn.setFixedWidth(36)
    menu_btn.setFixedHeight(36)
    menu_btn.clicked.connect(lambda: menu.exec(menu_btn.mapToGlobal(menu_btn.rect().bottomLeft())))

    if primary:
        main_btn.setProperty("primary", True)
        menu_btn.setProperty("primary", True)

    row.addWidget(main_btn, 1)
    row.addWidget(menu_btn, 0)
    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return container, main_btn, menu_btn
