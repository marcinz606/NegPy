import qtawesome as qta
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.theme import THEME


class ActionToolbar(QWidget):
    """
    Unified toolbar for file navigation, geometry actions, and session management.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.session = controller.session

        self._init_ui()
        self._connect_signals()

    def _create_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setStyleSheet(f"color: {THEME.border_color}; background-color: {THEME.border_color};")
        line.setFixedWidth(1)
        return line

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 10, 0, 10)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QFrame()
        container.setObjectName("toolbar_container")
        container.setStyleSheet(f"""
            QFrame#toolbar_container {{
                background-color: {THEME.bg_panel};
                border: 1px solid {THEME.border_color};
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(6, 4, 6, 4)
        v_layout.setSpacing(8)

        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(10)

        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(10)

        icon_color = THEME.text_primary
        icon_size = QSize(16, 16)
        btn_height = 32

        # 1. Navigation
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(qta.icon("fa5s.chevron-left", color=icon_color))
        self.btn_next = QToolButton()
        self.btn_next.setIcon(qta.icon("fa5s.chevron-right", color=icon_color))

        # 2. History
        self.btn_undo = QPushButton(" Undo")
        self.btn_undo.setIcon(qta.icon("fa5s.arrow-left", color=icon_color))
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_redo = QPushButton(" Redo")
        self.btn_redo.setIcon(qta.icon("fa5s.arrow-right", color=icon_color))
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")

        # 3. Geometry
        self.btn_rot_l = QToolButton()
        self.btn_rot_l.setIcon(qta.icon("fa5s.undo", color=icon_color))
        self.btn_rot_r = QToolButton()
        self.btn_rot_r.setIcon(qta.icon("fa5s.redo", color=icon_color))
        self.btn_flip_h = QToolButton()
        self.btn_flip_h.setIcon(qta.icon("fa5s.arrows-alt-h", color=icon_color))
        self.btn_flip_v = QToolButton()
        self.btn_flip_v.setIcon(qta.icon("fa5s.arrows-alt-v", color=icon_color))

        # 4. Clipboard
        self.btn_copy = QPushButton(" Copy")
        self.btn_copy.setIcon(qta.icon("fa5s.copy", color=icon_color))
        self.btn_paste = QPushButton(" Paste")
        self.btn_paste.setIcon(qta.icon("fa5s.paste", color=icon_color))
        self.btn_reset = QPushButton(" Reset")
        self.btn_reset.setIcon(qta.icon("fa5s.history", color=icon_color))

        # 5. Zoom
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(100, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(80)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(35)
        self.zoom_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 11px;")

        # 6. Session
        self.btn_save = QPushButton(" Save")
        self.btn_save.setIcon(qta.icon("fa5s.save", color=icon_color))
        self.btn_export = QPushButton(" Export")
        self.btn_export.setObjectName("export_btn")
        self.btn_export.setIcon(qta.icon("fa5s.check-circle", color="white"))
        self.btn_unload = QPushButton(" Unload")
        self.btn_unload.setIcon(qta.icon("fa5s.times-circle", color=icon_color))

        all_buttons = [
            self.btn_prev,
            self.btn_next,
            self.btn_undo,
            self.btn_redo,
            self.btn_rot_l,
            self.btn_rot_r,
            self.btn_flip_h,
            self.btn_flip_v,
            self.btn_copy,
            self.btn_paste,
            self.btn_reset,
            self.btn_save,
            self.btn_export,
            self.btn_unload,
        ]

        for btn in all_buttons:
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        row1_layout.addWidget(self.btn_prev)
        row1_layout.addWidget(self.btn_next)
        row1_layout.addWidget(self.zoom_slider)
        row1_layout.addWidget(self.zoom_label)
        row1_layout.addWidget(self.btn_rot_l)
        row1_layout.addWidget(self.btn_rot_r)
        row1_layout.addWidget(self.btn_flip_h)
        row1_layout.addWidget(self.btn_flip_v)

        row2_layout.addWidget(self.btn_undo)
        row2_layout.addWidget(self.btn_redo)
        row2_layout.addWidget(self.btn_copy)
        row2_layout.addWidget(self.btn_paste)
        row2_layout.addWidget(self.btn_reset)
        row2_layout.addWidget(self.btn_save)
        row2_layout.addWidget(self.btn_export)
        row2_layout.addWidget(self.btn_unload)

        # Add rows to the vertical layout container
        v_layout.addLayout(row1_layout)
        v_layout.addLayout(row2_layout)

        main_layout.addWidget(container)

    def _connect_signals(self) -> None:
        self.btn_prev.clicked.connect(self.session.prev_file)
        self.btn_next.clicked.connect(self.session.next_file)

        self.btn_rot_l.clicked.connect(lambda: self.rotate(1))
        self.btn_rot_r.clicked.connect(lambda: self.rotate(-1))
        self.btn_flip_h.clicked.connect(lambda: self.flip("horizontal"))
        self.btn_flip_v.clicked.connect(lambda: self.flip("vertical"))
        self.btn_undo.clicked.connect(self.session.undo)
        self.btn_redo.clicked.connect(self.session.redo)

        self.btn_copy.clicked.connect(self.session.copy_settings)
        self.btn_paste.clicked.connect(self.session.paste_settings)
        self.btn_save.clicked.connect(self.controller.save_current_edits)
        self.btn_reset.clicked.connect(self.session.reset_settings)
        self.btn_unload.clicked.connect(self.session.remove_current_file)
        self.btn_export.clicked.connect(self.controller.request_export)

        # Zoom
        self.zoom_slider.valueChanged.connect(lambda v: self.controller.zoom_requested.emit(float(v / 100.0)))
        self.controller.zoom_changed.connect(self._on_zoom_changed)

        # State sync for button enabled/disabled
        self.session.state_changed.connect(self._update_ui_state)

    def _on_zoom_changed(self, zoom: float) -> None:
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(zoom * 100))
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def rotate(self, direction: int) -> None:
        from dataclasses import replace

        new_rot = (self.session.state.config.geometry.rotation + direction) % 4
        new_geo = replace(self.session.state.config.geometry, rotation=new_rot)
        new_config = replace(self.session.state.config, geometry=new_geo)
        self.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def flip(self, axis: str) -> None:
        from dataclasses import replace

        geo = self.session.state.config.geometry
        if axis == "horizontal":
            new_geo = replace(geo, flip_horizontal=not geo.flip_horizontal)
        else:
            new_geo = replace(geo, flip_vertical=not geo.flip_vertical)

        new_config = replace(self.session.state.config, geometry=new_geo)
        self.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def _update_ui_state(self) -> None:
        state = self.session.state
        self.btn_prev.setEnabled(state.selected_file_idx > 0)
        self.btn_next.setEnabled(state.selected_file_idx < len(state.uploaded_files) - 1)
        self.btn_unload.setEnabled(state.selected_file_idx >= 0)
        self.btn_paste.setEnabled(state.clipboard is not None)

        self.btn_undo.setEnabled(state.undo_index > 0)
        self.btn_redo.setEnabled(state.undo_index < state.max_history_index)
