from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QFrame,
)
from PyQt6.QtCore import QSize, Qt
import qtawesome as qta
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
                padding: 2px;
            }}
        """)
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(6, 6, 6, 6)
        h_layout.setSpacing(12)

        icon_color = THEME.text_primary
        icon_size = QSize(16, 16)
        btn_height = 32

        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(qta.icon("fa5s.chevron-left", color=icon_color))
        self.btn_prev.setToolTip("Previous Image (Left Arrow)")

        self.btn_next = QToolButton()
        self.btn_next.setIcon(qta.icon("fa5s.chevron-right", color=icon_color))
        self.btn_next.setToolTip("Next Image (Right Arrow)")

        self.btn_rot_l = QToolButton()
        self.btn_rot_l.setIcon(qta.icon("fa5s.undo", color=icon_color))
        self.btn_rot_l.setToolTip("Rotate CCW ([)")

        self.btn_rot_r = QToolButton()
        self.btn_rot_r.setIcon(qta.icon("fa5s.redo", color=icon_color))
        self.btn_rot_r.setToolTip("Rotate CW (])")

        self.btn_flip_h = QToolButton()
        self.btn_flip_h.setIcon(qta.icon("fa5s.arrows-alt-h", color=icon_color))
        self.btn_flip_h.setToolTip("Flip Horizontal (H)")

        self.btn_flip_v = QToolButton()
        self.btn_flip_v.setIcon(qta.icon("fa5s.arrows-alt-v", color=icon_color))
        self.btn_flip_v.setToolTip("Flip Vertical (V)")

        for btn in [
            self.btn_prev,
            self.btn_next,
            self.btn_rot_l,
            self.btn_rot_r,
            self.btn_flip_h,
            self.btn_flip_v,
        ]:
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_copy = QPushButton(" Copy")
        self.btn_copy.setIcon(qta.icon("fa5s.copy", color=icon_color))

        self.btn_paste = QPushButton(" Paste")
        self.btn_paste.setIcon(qta.icon("fa5s.paste", color=icon_color))

        self.btn_reset = QPushButton(" Reset")
        self.btn_reset.setIcon(qta.icon("fa5s.history", color=icon_color))

        self.btn_unload = QPushButton(" Unload")
        self.btn_unload.setIcon(qta.icon("fa5s.times-circle", color=icon_color))

        self.btn_save = QPushButton(" Save")
        self.btn_save.setIcon(qta.icon("fa5s.save", color=icon_color))

        self.btn_export = QPushButton(" Export")
        self.btn_export.setObjectName("export_btn")
        self.btn_export.setIcon(qta.icon("fa5s.check-circle", color="white"))
        self.btn_export.setIconSize(icon_size)
        self.btn_export.setToolTip("Export the current image with applied settings (E)")

        for btn in [self.btn_copy, self.btn_paste, self.btn_save, self.btn_export, self.btn_reset, self.btn_unload]:
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # Assemble Row
        h_layout.addWidget(self.btn_prev)
        h_layout.addWidget(self.btn_next)

        h_layout.addWidget(self._create_separator())

        h_layout.addWidget(self.btn_rot_l)
        h_layout.addWidget(self.btn_rot_r)
        h_layout.addWidget(self.btn_flip_h)
        h_layout.addWidget(self.btn_flip_v)

        h_layout.addWidget(self._create_separator())

        h_layout.addWidget(self.btn_copy)
        h_layout.addWidget(self.btn_paste)
        h_layout.addWidget(self.btn_reset)

        h_layout.addWidget(self._create_separator())

        h_layout.addWidget(self.btn_save)
        h_layout.addWidget(self.btn_export)
        h_layout.addWidget(self.btn_unload)

        main_layout.addWidget(container)

    def _connect_signals(self) -> None:
        self.btn_prev.clicked.connect(self.session.prev_file)
        self.btn_next.clicked.connect(self.session.next_file)

        self.btn_rot_l.clicked.connect(lambda: self.rotate(1))
        self.btn_rot_r.clicked.connect(lambda: self.rotate(-1))
        self.btn_flip_h.clicked.connect(lambda: self.flip("horizontal"))
        self.btn_flip_v.clicked.connect(lambda: self.flip("vertical"))

        self.btn_copy.clicked.connect(self.session.copy_settings)
        self.btn_paste.clicked.connect(self.session.paste_settings)
        self.btn_save.clicked.connect(self.controller.save_current_edits)
        self.btn_reset.clicked.connect(self.session.reset_settings)
        self.btn_unload.clicked.connect(self.session.remove_current_file)
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
        self.btn_next.setEnabled(state.selected_file_idx < len(state.uploaded_files) - 1)
        self.btn_unload.setEnabled(state.selected_file_idx >= 0)
        self.btn_paste.setEnabled(state.clipboard is not None)
