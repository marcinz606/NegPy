import qtawesome as qta
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.theme import THEME

CANVAS_COLORS = [
    ("#050505", (0.02, 0.02, 0.02), "Black"),
    ("#1C1C1C", (0.11, 0.11, 0.11), "Dark Grey"),
    ("#404040", (0.25, 0.25, 0.25), "Mid Grey"),
]


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
        v_layout.setSpacing(0)

        row_layout = QHBoxLayout()
        row_layout.setSpacing(6)

        icon_color = THEME.text_primary
        icon_size = QSize(16, 16)
        btn_height = 32

        # 1. Navigation
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(qta.icon("fa5s.chevron-left", color=icon_color))
        self.btn_prev.setToolTip("Previous")
        self.btn_next = QToolButton()
        self.btn_next.setIcon(qta.icon("fa5s.chevron-right", color=icon_color))
        self.btn_next.setToolTip("Next")

        # (kept as internal state holders — not added to layout)
        self.btn_undo = QPushButton()
        self.btn_redo = QPushButton()
        self.btn_copy = QPushButton()
        self.btn_paste = QPushButton()
        self.btn_reset = QPushButton()
        self.btn_unload = QPushButton()

        # 2. Geometry
        self.btn_rot_l = QToolButton()
        self.btn_rot_l.setIcon(qta.icon("fa5s.undo", color=icon_color))
        self.btn_rot_l.setToolTip("Rotate CCW  [")
        self.btn_rot_r = QToolButton()
        self.btn_rot_r.setIcon(qta.icon("fa5s.redo", color=icon_color))
        self.btn_rot_r.setToolTip("Rotate CW  ]")
        self.btn_flip_h = QToolButton()
        self.btn_flip_h.setIcon(qta.icon("fa5s.arrows-alt-h", color=icon_color))
        self.btn_flip_h.setToolTip("Flip Horizontal  H")
        self.btn_flip_v = QToolButton()
        self.btn_flip_v.setIcon(qta.icon("fa5s.arrows-alt-v", color=icon_color))
        self.btn_flip_v.setToolTip("Flip Vertical  V")

        # 3. Zoom
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(80)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(35)
        self.zoom_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 11px;")

        self.btn_hq = QToolButton()
        self.btn_hq.setText("HQ")
        self.btn_hq.setCheckable(True)
        self.btn_hq.setToolTip("Toggle High Quality Preview")

        # 4. Canvas background swatches
        self.canvas_color_btns: list[QToolButton] = []
        self.canvas_color_group = QButtonGroup(self)
        self.canvas_color_group.setExclusive(True)
        for i, (hex_col, _, label) in enumerate(CANVAS_COLORS):
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setToolTip(f"Canvas: {label}")
            btn.setFixedSize(20, 20)
            btn.setStyleSheet(f"""
                QToolButton {{
                    background-color: {hex_col};
                    border: 1px solid #444;
                    border-radius: 3px;
                }}
                QToolButton:checked {{
                    border: 2px solid {THEME.accent_primary};
                }}
                QToolButton:hover {{
                    border: 1px solid #888;
                }}
            """)
            self.canvas_color_group.addButton(btn, i)
            self.canvas_color_btns.append(btn)
        self.canvas_color_btns[self.session.state.canvas_bg_index].setChecked(True)

        # 5. Save / Export
        self.btn_save = QPushButton(" Save")
        self.btn_save.setIcon(qta.icon("fa5s.save", color=icon_color))

        self.btn_export = QPushButton(" Export")
        self.btn_export.setObjectName("export_btn")
        self.btn_export.setIcon(qta.icon("fa5s.check-circle", color=icon_color))
        self.btn_export.setToolTip("Export  Ctrl+E")
        self.btn_export.setFixedHeight(36)

        # 6. Overflow menu
        self.btn_overflow = QToolButton()
        self.btn_overflow.setIcon(qta.icon("fa5s.ellipsis-h", color=icon_color))
        self.btn_overflow.setToolTip("More actions")
        self.btn_overflow.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        overflow_menu = QMenu(self.btn_overflow)
        self._action_undo = overflow_menu.addAction(qta.icon("fa5s.arrow-left", color=icon_color), "Undo  Ctrl+Z", self.session.undo)
        self._action_redo = overflow_menu.addAction(qta.icon("fa5s.arrow-right", color=icon_color), "Redo  Ctrl+Y", self.session.redo)
        overflow_menu.addSeparator()
        self._action_copy = overflow_menu.addAction(
            qta.icon("fa5s.copy", color=icon_color), "Copy Settings  Ctrl+C", self.session.copy_settings
        )
        self._action_paste = overflow_menu.addAction(
            qta.icon("fa5s.paste", color=icon_color), "Paste Settings  Ctrl+V", self.session.paste_settings
        )
        overflow_menu.addSeparator()
        overflow_menu.addAction(qta.icon("fa5s.history", color=icon_color), "Reset Settings", self.session.reset_settings)
        overflow_menu.addSeparator()
        overflow_menu.addAction(qta.icon("fa5s.times-circle", color=icon_color), "Unload", self.session.remove_current_file)
        overflow_menu.addSeparator()
        overflow_menu.addAction(qta.icon("fa5s.keyboard", color=icon_color), "Keyboard Shortcuts  ?", self._show_shortcuts)
        self.btn_overflow.setMenu(overflow_menu)

        standard_buttons = [
            self.btn_prev,
            self.btn_next,
            self.btn_rot_l,
            self.btn_rot_r,
            self.btn_flip_h,
            self.btn_flip_v,
            self.btn_save,
            self.btn_hq,
            self.btn_overflow,
        ]
        for btn in standard_buttons:
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_export.setIconSize(icon_size)
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)

        # Single-row layout: prev · next · sep · zoom+label · hq · swatches · sep · rot_l · rot_r · flip_h · flip_v · sep · save · export · overflow
        row_layout.addWidget(self.btn_prev)
        row_layout.addWidget(self.btn_next)
        row_layout.addWidget(self._create_separator())
        row_layout.addWidget(self.zoom_slider)
        row_layout.addWidget(self.zoom_label)
        row_layout.addWidget(self.btn_hq)
        for btn in self.canvas_color_btns:
            row_layout.addWidget(btn)
        row_layout.addWidget(self._create_separator())
        row_layout.addWidget(self.btn_rot_l)
        row_layout.addWidget(self.btn_rot_r)
        row_layout.addWidget(self.btn_flip_h)
        row_layout.addWidget(self.btn_flip_v)
        row_layout.addWidget(self._create_separator())
        row_layout.addWidget(self.btn_save)
        row_layout.addWidget(self.btn_export)
        row_layout.addWidget(self.btn_overflow)

        v_layout.addLayout(row_layout)
        main_layout.addWidget(container)

    def _connect_signals(self) -> None:
        self.btn_prev.clicked.connect(self.session.prev_file)
        self.btn_next.clicked.connect(self.session.next_file)

        self.btn_rot_l.clicked.connect(lambda: self.rotate(1))
        self.btn_rot_r.clicked.connect(lambda: self.rotate(-1))
        self.btn_flip_h.clicked.connect(lambda: self.flip("horizontal"))
        self.btn_flip_v.clicked.connect(lambda: self.flip("vertical"))

        self.btn_save.clicked.connect(self.controller.save_current_edits)
        self.btn_export.clicked.connect(self.controller.request_export)

        self.canvas_color_group.idToggled.connect(self._on_canvas_color_changed)

        self.zoom_slider.valueChanged.connect(lambda v: self.controller.zoom_requested.emit(float(v / 100.0)))
        self.btn_hq.clicked.connect(self.controller.toggle_hq_preview)
        self.controller.zoom_changed.connect(self._on_zoom_changed)

        self.session.state_changed.connect(self._update_ui_state)

    def _on_canvas_color_changed(self, idx: int, checked: bool) -> None:
        if checked:
            self.session.set_canvas_bg(idx)
            if self.controller.canvas:
                _, (r, g, b), _ = CANVAS_COLORS[idx]
                self.controller.canvas.set_background_color(r, g, b)

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

    def _show_shortcuts(self) -> None:
        from negpy.desktop.view.widgets.shortcuts_overlay import ShortcutsOverlay

        dlg = ShortcutsOverlay(self.window())
        dlg.exec()

    def _update_ui_state(self) -> None:
        state = self.session.state
        self.btn_prev.setEnabled(state.selected_file_idx > 0)
        self.btn_next.setEnabled(state.selected_file_idx < len(state.uploaded_files) - 1)
        self.btn_hq.setChecked(state.hq_preview)

        self._action_undo.setEnabled(state.undo_index > 0)
        self._action_redo.setEnabled(state.undo_index < state.max_history_index)
        self._action_paste.setEnabled(state.clipboard is not None)
