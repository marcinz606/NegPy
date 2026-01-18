from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QToolButton,
)
from src.desktop.controller import AppController


class ActionToolbar(QWidget):
    """
    Two-row toolbar for file navigation, geometry actions, and session management.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.session = controller.session

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 10, 0, 10)
        main_layout.setSpacing(12)

        # High-visibility styling
        self.setStyleSheet("""
            QToolButton, QPushButton {
                font-size: 14px;
                padding: 8px 16px;
                min-width: 90px;
                background-color: #333;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QToolButton:hover, QPushButton:hover {
                background-color: #444;
                border: 1px solid #555;
            }
            QPushButton#export_btn {
                background-color: #2e7d32;
                font-weight: bold;
                padding: 10px 30px;
            }
            QPushButton#export_btn:hover {
                background-color: #388e3c;
            }
        """)

        # Row 1: Navigation & Geometry
        row1 = QHBoxLayout()
        row1.addStretch()

        self.btn_prev = QToolButton()
        self.btn_prev.setText("Prev")
        self.btn_next = QToolButton()
        self.btn_next.setText("Next")

        self.btn_rot_l = QToolButton()
        self.btn_rot_l.setText("Rotate L")
        self.btn_rot_r = QToolButton()
        self.btn_rot_r.setText("Rotate R")

        self.btn_flip_h = QToolButton()
        self.btn_flip_h.setText("Flip H")
        self.btn_flip_v = QToolButton()
        self.btn_flip_v.setText("Flip V")

        row1.addWidget(self.btn_prev)
        row1.addWidget(self.btn_next)
        row1.addSpacing(20)
        row1.addWidget(self.btn_rot_l)
        row1.addWidget(self.btn_rot_r)
        row1.addSpacing(20)
        row1.addWidget(self.btn_flip_h)
        row1.addWidget(self.btn_flip_v)
        row1.addStretch()

        main_layout.addLayout(row1)

        # Row 2: Clipboard & Session
        row2 = QHBoxLayout()
        row2.addStretch()

        self.btn_copy = QPushButton("Copy")
        self.btn_paste = QPushButton("Paste")
        self.btn_reset = QPushButton("Reset")
        self.btn_export = QPushButton("Export Image")
        self.btn_export.setObjectName("export_btn")

        row2.addWidget(self.btn_copy)
        row2.addWidget(self.btn_paste)
        row2.addSpacing(20)
        row2.addWidget(self.btn_export)
        row2.addSpacing(20)
        row2.addWidget(self.btn_reset)
        row2.addStretch()

        main_layout.addLayout(row2)

    def _connect_signals(self) -> None:
        self.btn_prev.clicked.connect(self.session.prev_file)
        self.btn_next.clicked.connect(self.session.next_file)

        self.btn_rot_l.clicked.connect(lambda: self.rotate(1))
        self.btn_rot_r.clicked.connect(lambda: self.rotate(-1))
        self.btn_flip_h.clicked.connect(lambda: self.flip("horizontal"))
        self.btn_flip_v.clicked.connect(lambda: self.flip("vertical"))

        self.btn_copy.clicked.connect(self.session.copy_settings)
        self.btn_paste.clicked.connect(self.session.paste_settings)
        self.btn_reset.clicked.connect(self.session.reset_settings)
        self.btn_export.clicked.connect(self.controller.request_export)

        # State sync for button enabled/disabled
        self.session.state_changed.connect(self._update_ui_state)

    def rotate(self, direction: int) -> None:
        from dataclasses import replace

        new_rot = (self.session.state.config.geometry.rotation + direction) % 4
        new_geo = replace(self.session.state.config.geometry, rotation=new_rot)
        new_config = replace(self.session.state.config, geometry=new_geo)
        self.session.update_config(new_config)
        self.controller.request_render()

    def flip(self, axis: str) -> None:
        from dataclasses import replace

        geo = self.session.state.config.geometry
        if axis == "horizontal":
            new_geo = replace(geo, flip_horizontal=not geo.flip_horizontal)
        else:
            new_geo = replace(geo, flip_vertical=not geo.flip_vertical)

        new_config = replace(self.session.state.config, geometry=new_geo)
        self.session.update_config(new_config)
        self.controller.request_render()

    def _update_ui_state(self) -> None:
        state = self.session.state
        self.btn_prev.setEnabled(state.selected_file_idx > 0)
        self.btn_next.setEnabled(
            state.selected_file_idx < len(state.uploaded_files) - 1
        )
        self.btn_paste.setEnabled(state.clipboard is not None)
