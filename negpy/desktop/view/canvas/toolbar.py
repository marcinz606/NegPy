import qtawesome as qta
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.keyboard_shortcuts import _context_undo
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.styles.templates import swatch_qss
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.system.config import APP_CONFIG

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
        line.setObjectName("toolbar_separator")
        line.setFixedWidth(1)
        return line

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 10, 0, 10)

        container = QFrame()
        container.setObjectName("toolbar_container")
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(6, 4, 6, 4)
        v_layout.setSpacing(0)

        row_layout = QHBoxLayout()
        row_layout.setSpacing(6)

        icon_color = THEME.text_primary
        icon_size = QSize(16, 16)
        btn_height = 32

        # 0. Panel toggles (live at the toolbar's outer edges)
        self.btn_toggle_left = QToolButton()
        self.btn_toggle_left.setCheckable(True)
        self.btn_toggle_left.setChecked(True)
        self.btn_toggle_left.setIcon(qta.icon("fa5s.columns", color=icon_color))
        self.btn_toggle_left.setToolTip(tooltip_with_shortcut("Toggle Session Panel", "toggle_left_panel"))
        self.btn_toggle_right = QToolButton()
        self.btn_toggle_right.setCheckable(True)
        self.btn_toggle_right.setChecked(True)
        self.btn_toggle_right.setIcon(qta.icon("fa5s.sliders-h", color=icon_color))
        self.btn_toggle_right.setToolTip(tooltip_with_shortcut("Toggle Controls Panel", "toggle_right_panel"))

        # 1. Navigation
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(qta.icon("fa5s.chevron-left", color=icon_color))
        self.btn_prev.setToolTip("Previous")
        self.btn_next = QToolButton()
        self.btn_next.setIcon(qta.icon("fa5s.chevron-right", color=icon_color))
        self.btn_next.setToolTip("Next")

        # Undo / Redo live in the main toolbar (mdi arrows, distinct from the
        # circular rotate icons which reuse fa5s.undo/redo).
        self.btn_undo = QToolButton()
        self.btn_undo.setIcon(qta.icon("mdi.undo", color=icon_color))
        self.btn_undo.setToolTip(tooltip_with_shortcut("Undo", "undo"))
        self.btn_redo = QToolButton()
        self.btn_redo.setIcon(qta.icon("mdi.redo", color=icon_color))
        self.btn_redo.setToolTip(tooltip_with_shortcut("Redo", "redo"))

        # (kept as internal state holders — not added to layout)
        self.btn_copy = QPushButton()
        self.btn_paste = QPushButton()
        self.btn_reset = QPushButton()
        self.btn_unload = QPushButton()

        # 2. Geometry
        self.btn_rot_l = QToolButton()
        self.btn_rot_l.setIcon(qta.icon("fa5s.undo", color=icon_color))
        self.btn_rot_l.setToolTip(tooltip_with_shortcut("Rotate CCW", "rotate_ccw"))
        self.btn_rot_r = QToolButton()
        self.btn_rot_r.setIcon(qta.icon("fa5s.redo", color=icon_color))
        self.btn_rot_r.setToolTip(tooltip_with_shortcut("Rotate CW", "rotate_cw"))
        self.btn_flip_h = QToolButton()
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.setIcon(qta.icon("fa5s.arrows-alt-h", color=icon_color))
        self.btn_flip_h.setToolTip(tooltip_with_shortcut("Flip Horizontal", "flip_h"))
        self.btn_flip_v = QToolButton()
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.setIcon(qta.icon("fa5s.arrows-alt-v", color=icon_color))
        self.btn_flip_v.setToolTip(tooltip_with_shortcut("Flip Vertical", "flip_v"))

        # 3. Zoom (range matches APP_CONFIG canvas_zoom_min/max, percent)
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(
            int(APP_CONFIG.canvas_zoom_min * 100),
            int(APP_CONFIG.canvas_zoom_max * 100),
        )
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(80)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(42)
        self.zoom_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px;")

        self.btn_zoom_fit = QToolButton()
        self.btn_zoom_fit.setIcon(qta.icon("fa5s.expand", color=icon_color))
        self.btn_zoom_fit.setToolTip(tooltip_with_shortcut("Fit to Window", "fit_view"))
        self.btn_zoom_original = QToolButton()
        self.btn_zoom_original.setText("1:1")
        self.btn_zoom_original.setToolTip(
            tooltip_with_shortcut(
                "Original size (100%). Displays a lower-resolution preview unless HQ is enabled.",
                "zoom_100",
            )
        )

        self.btn_hq = QToolButton()
        self.btn_hq.setText("HQ")
        self.btn_hq.setCheckable(True)
        self.btn_hq.setToolTip("Toggle High Quality Preview")

        self.btn_compare = QToolButton()
        self.btn_compare.setCheckable(True)
        self.btn_compare.setIcon(qta.icon("fa5s.adjust", color=icon_color))
        self.btn_compare.setToolTip(tooltip_with_shortcut("Before / After — show the auto baseline", "toggle_compare"))

        self.btn_flat_peek = QToolButton()
        self.btn_flat_peek.setCheckable(True)
        self.btn_flat_peek.setIcon(qta.icon("fa5s.eye", color=icon_color))
        self.btn_flat_peek.setToolTip(
            tooltip_with_shortcut("Peek flat scan — temporarily show the flat master (does not change your edit)", "toggle_flat_peek")
        )

        # GPU acceleration toggle (details surfaced via tooltip, refreshed by the dashboard)
        self.btn_gpu = QToolButton()
        self.btn_gpu.setCheckable(True)
        self.btn_gpu.setIcon(qta.icon("fa5s.bolt", color=icon_color))
        self._gpu_available = GPUDevice.get().is_available
        if self._gpu_available:
            self.btn_gpu.setChecked(self.session.state.gpu_enabled)
        else:
            self.btn_gpu.setEnabled(False)
            self.btn_gpu.setChecked(False)
        self.btn_gpu.setToolTip("GPU Acceleration")

        # 4. Canvas background swatches
        self.canvas_color_btns: list[QToolButton] = []
        self.canvas_color_group = QButtonGroup(self)
        self.canvas_color_group.setExclusive(True)
        for i, (hex_col, _, label) in enumerate(CANVAS_COLORS):
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setToolTip(f"Canvas: {label}")
            btn.setFixedSize(20, 20)
            btn.setStyleSheet(swatch_qss(hex_col))
            self.canvas_color_group.addButton(btn, i)
            self.canvas_color_btns.append(btn)
        self.canvas_color_btns[self.session.state.canvas_bg_index].setChecked(True)

        # 6. Overflow menu & responsive groups
        self.btn_overflow = QToolButton()
        self.btn_overflow.setIcon(qta.icon("fa5s.ellipsis-h", color=icon_color))
        self.btn_overflow.setToolTip("More actions")
        self.btn_overflow.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        overflow_menu = QMenu(self.btn_overflow)

        # Overflow: swatches + HQ group (<720px)
        self._ov_hq_action = overflow_menu.addAction("Toggle HQ Preview")
        self._ov_hq_action.setCheckable(True)
        self._ov_hq_action.setVisible(False)
        overflow_menu.addSeparator()
        self._ov_color_actions: list = []
        for i, (hex_col, _, label) in enumerate(CANVAS_COLORS):
            action = overflow_menu.addAction(f"Canvas: {label}")
            action.setVisible(False)
            self._ov_color_actions.append(action)

        # Overflow: flip + rotate group (<580px)
        self._ov_sep_main = overflow_menu.addSeparator()
        self._ov_sep_main.setVisible(False)
        self._ov_rot_l_action = overflow_menu.addAction(qta.icon("fa5s.undo", color=icon_color), "Rotate CCW")
        self._ov_rot_l_action.setVisible(False)
        self._ov_rot_r_action = overflow_menu.addAction(qta.icon("fa5s.redo", color=icon_color), "Rotate CW")
        self._ov_rot_r_action.setVisible(False)
        self._ov_flip_h_action = overflow_menu.addAction(qta.icon("fa5s.arrows-alt-h", color=icon_color), "Flip Horizontal")
        self._ov_flip_h_action.setCheckable(True)
        self._ov_flip_h_action.setVisible(False)
        self._ov_flip_v_action = overflow_menu.addAction(qta.icon("fa5s.arrows-alt-v", color=icon_color), "Flip Vertical")
        self._ov_flip_v_action.setCheckable(True)
        self._ov_flip_v_action.setVisible(False)
        self._ov_sep_rotate = overflow_menu.addSeparator()
        self._ov_sep_rotate.setVisible(False)

        # Edits auto-save to the DB (and surface in History), so an explicit Save
        # lives here in the overflow rather than the main toolbar.
        overflow_menu.addAction(qta.icon("fa5s.save", color=icon_color), "Save Edits", self.controller.save_current_edits)
        overflow_menu.addSeparator()
        self._action_copy = overflow_menu.addAction(
            qta.icon("fa5s.copy", color=icon_color), "Copy Settings  Ctrl+C", self.session.copy_settings
        )
        self._action_copy_bounds = overflow_menu.addAction(
            qta.icon("fa5s.copy", color=icon_color), "Copy Settings + Bounds  Ctrl+Shift+C", self.session.copy_settings_with_bounds
        )
        self._action_paste = overflow_menu.addAction(
            qta.icon("fa5s.paste", color=icon_color), "Paste Settings  Ctrl+V", self.session.paste_settings
        )
        overflow_menu.addSeparator()
        overflow_menu.addAction(qta.icon("fa5s.history", color=icon_color), "Reset Settings", self.session.reset_settings)
        overflow_menu.addSeparator()
        overflow_menu.addAction(qta.icon("fa5s.times-circle", color=icon_color), "Unload", self._on_overflow_unload)
        overflow_menu.addSeparator()
        scale_menu = overflow_menu.addMenu(qta.icon("fa5s.search-plus", color=icon_color), "UI Scale")
        self._ui_scale_group = QActionGroup(self)
        self._ui_scale_group.setExclusive(True)
        current_scale = float(self.session.repo.get_global_setting("ui_scale", 1.0) or 1.0)
        for pct in (80, 90, 100, 110, 120):
            val = pct / 100.0
            act = scale_menu.addAction(f"{pct}%")
            act.setCheckable(True)
            act.setChecked(abs(val - current_scale) < 0.001)
            self._ui_scale_group.addAction(act)
            act.triggered.connect(lambda _checked=False, v=val, p=pct: self._on_ui_scale_selected(v, p))
        overflow_menu.addSeparator()

        overflow_menu.addAction(qta.icon("fa5s.map-signs", color=icon_color), "Take the tour", self._show_tour)
        overflow_menu.addAction(qta.icon("fa5s.keyboard", color=icon_color), "Keyboard Shortcuts  ?", self._show_shortcuts)
        self.btn_overflow.setMenu(overflow_menu)

        standard_buttons = [
            self.btn_toggle_left,
            self.btn_toggle_right,
            self.btn_prev,
            self.btn_next,
            self.btn_rot_l,
            self.btn_rot_r,
            self.btn_flip_h,
            self.btn_flip_v,
            self.btn_undo,
            self.btn_redo,
            self.btn_hq,
            self.btn_compare,
            self.btn_gpu,
            self.btn_overflow,
        ]
        for btn in standard_buttons:
            btn.setIconSize(icon_size)
            btn.setFixedHeight(btn_height)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # Custom-sized buttons skip the standard sizing above but must share the
        # same hover cursor as the rest of the toolbar.
        for btn in (self.btn_zoom_fit, self.btn_zoom_original, *self.canvas_color_btns):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # Single-row layout: toggle_left · prev · next · sep1 · zoom+label · hq · swatches · sep2 · rot_l · rot_r · flip_h · flip_v · sep3 · undo · redo · compare · gpu · overflow · toggle_right
        row_layout.addWidget(self.btn_toggle_left)
        row_layout.addWidget(self.btn_prev)
        row_layout.addWidget(self.btn_next)
        self._sep1 = self._create_separator()
        row_layout.addWidget(self._sep1)
        row_layout.addWidget(self.zoom_slider)
        row_layout.addWidget(self.zoom_label)
        row_layout.addWidget(self.btn_zoom_fit)
        row_layout.addWidget(self.btn_zoom_original)
        row_layout.addWidget(self.btn_hq)
        for btn in self.canvas_color_btns:
            row_layout.addWidget(btn)
        self._sep2 = self._create_separator()
        row_layout.addWidget(self._sep2)
        row_layout.addWidget(self.btn_rot_l)
        row_layout.addWidget(self.btn_rot_r)
        row_layout.addWidget(self.btn_flip_h)
        row_layout.addWidget(self.btn_flip_v)
        self._sep3 = self._create_separator()
        row_layout.addWidget(self._sep3)
        row_layout.addWidget(self.btn_undo)
        row_layout.addWidget(self.btn_redo)
        row_layout.addWidget(self.btn_compare)
        row_layout.addWidget(self.btn_flat_peek)
        row_layout.addWidget(self.btn_gpu)
        row_layout.addWidget(self.btn_overflow)
        row_layout.addWidget(self.btn_toggle_right)

        # Overflow groups for responsive resizeEvent
        self._ov_swatches_hq: list = [self.btn_hq] + self.canvas_color_btns + [self._sep2]
        self._ov_flip_rotate: list = [self.btn_rot_l, self.btn_rot_r, self.btn_flip_h, self.btn_flip_v, self._sep3]

        v_layout.addLayout(row_layout)
        # Size the pill to its controls; don't stretch it across the canvas.
        main_layout.addWidget(container, 0, Qt.AlignmentFlag.AlignCenter)

    def _on_flat_peek_changed(self, active: bool) -> None:
        self.btn_flat_peek.blockSignals(True)
        self.btn_flat_peek.setChecked(active)
        self.btn_flat_peek.blockSignals(False)

    def _connect_signals(self) -> None:
        self.btn_prev.clicked.connect(self.session.prev_file)
        self.btn_next.clicked.connect(self.session.next_file)

        self.btn_rot_l.clicked.connect(lambda: self.rotate(1))
        self.btn_rot_r.clicked.connect(lambda: self.rotate(-1))
        self.btn_flip_h.clicked.connect(lambda: self.flip("horizontal"))
        self.btn_flip_v.clicked.connect(lambda: self.flip("vertical"))

        # Same context routing as Ctrl+Z: heal-undo while a heal tool is in hand.
        self.btn_undo.clicked.connect(lambda: _context_undo(self.controller))
        self.btn_redo.clicked.connect(self.session.redo)

        self.canvas_color_group.idToggled.connect(self._on_canvas_color_changed)

        self.zoom_slider.valueChanged.connect(lambda v: self.controller.zoom_requested.emit(float(v / 100.0)))
        self.btn_zoom_fit.clicked.connect(self._on_fit_clicked)
        self.btn_zoom_original.clicked.connect(self._on_original_clicked)
        self.btn_hq.clicked.connect(self.controller.toggle_hq_preview)
        self.btn_compare.clicked.connect(self.controller.toggle_compare)
        self.controller.compare_changed.connect(self.btn_compare.setChecked)
        self.btn_flat_peek.toggled.connect(lambda checked: self.controller.toggle_flat_peek(force=checked))
        self.controller.flat_peek_changed.connect(self._on_flat_peek_changed)
        self.btn_gpu.toggled.connect(self._on_gpu_toggled)
        self.controller.zoom_changed.connect(self._on_zoom_changed)

        self.session.state_changed.connect(self._update_ui_state)
        self.session.asset_model.layoutChanged.connect(self._update_ui_state)

        # Overflow menu action connections
        self._ov_hq_action.triggered.connect(self.controller.toggle_hq_preview)
        for i, action in enumerate(self._ov_color_actions):
            action.triggered.connect(lambda checked, idx=i: self._on_canvas_color_changed(idx, True))
        self._ov_rot_l_action.triggered.connect(lambda: self.rotate(1))
        self._ov_rot_r_action.triggered.connect(lambda: self.rotate(-1))
        self._ov_flip_h_action.triggered.connect(lambda: self.flip("horizontal"))
        self._ov_flip_v_action.triggered.connect(lambda: self.flip("vertical"))

    def _on_overflow_unload(self) -> None:
        from negpy.desktop.view.confirm import confirm_unload

        if self.session.state.selected_file_idx < 0:
            return
        if confirm_unload(self):
            self.session.remove_current_file()

    def _on_gpu_toggled(self, checked: bool) -> None:
        if checked != self.session.state.gpu_enabled:
            self.session.set_gpu_enabled(checked)

    def refresh_gpu_status(self) -> None:
        """Reflect current GPU on/off state and active backend in the toolbar button."""
        enabled = self.session.state.gpu_enabled

        self.btn_gpu.blockSignals(True)
        self.btn_gpu.setChecked(enabled and self._gpu_available)
        self.btn_gpu.blockSignals(False)

        icon_color = THEME.accent_primary if (enabled and self._gpu_available) else THEME.text_primary
        self.btn_gpu.setIcon(qta.icon("fa5s.bolt", color=icon_color))

        if not self._gpu_available:
            self.btn_gpu.setToolTip("GPU not available on this hardware")
        elif enabled:
            try:
                backend = self.controller.render_worker.processor.backend_name
            except Exception:
                backend = "GPU"
            self.btn_gpu.setToolTip(f"GPU Acceleration: ON — {backend}\nClick to force the CPU pipeline.")
        else:
            self.btn_gpu.setToolTip("GPU Acceleration: OFF — CPU pipeline\nClick to enable WebGPU for near-instant previews.")

    def _on_ui_scale_selected(self, value: float, pct: int) -> None:
        self.session.repo.save_global_setting("ui_scale", value)
        QMessageBox.information(
            self,
            "UI Scale",
            f"UI scale set to {pct}%.\n\nRestart NegPy to apply the change.",
        )

    def _on_canvas_color_changed(self, idx: int, checked: bool) -> None:
        if checked:
            self.session.set_canvas_bg(idx)
            if self.controller.canvas:
                _, (r, g, b), _ = CANVAS_COLORS[idx]
                self.controller.canvas.set_background_color(r, g, b)

    def _on_zoom_changed(self, zoom: float) -> None:
        # The slider tracks the internal fit-relative zoom_level; the label shows the
        # true pixel zoom (zoom_level x fit_scale), which is what the user cares about.
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(round(max(0.0, zoom) * 100.0)))
        self.zoom_slider.blockSignals(False)
        canvas = getattr(self.controller, "canvas", None)
        pct = canvas.current_zoom_percent() if canvas is not None else int(round(max(0.0, zoom) * 100.0))
        self.zoom_label.setText(f"{pct}%")

    def _on_fit_clicked(self) -> None:
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            canvas.fit_to_window()

    def _on_original_clicked(self) -> None:
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            canvas.zoom_to_original()

    def rotate(self, direction: int) -> None:
        from dataclasses import replace

        from negpy.features.geometry.logic import rotate_normalized_rect

        config = self.session.state.config
        geo = config.geometry
        # The button's labelled direction is the visual rotation the user sees (the
        # handedness fix below only keeps that promise under a flip). Crop/analysis
        # rects live in display space, so they rotate by that visual quarter-turn.
        visual_turns_ccw = direction
        # Pipeline applies rotate-then-flip; a single mirror inverts rotation handedness.
        if geo.flip_horizontal != geo.flip_vertical:
            direction = -direction
        new_rot = (geo.rotation + direction) % 4
        new_geo = replace(geo, rotation=new_rot)
        # Rotate the manual crop rect with the content so it keeps framing the same area
        # (without this it stayed put and misaligned after a 90°/180° turn).
        if geo.manual_crop_rect is not None:
            new_geo = replace(new_geo, manual_crop_rect=rotate_normalized_rect(geo.manual_crop_rect, visual_turns_ccw))
        new_config = replace(config, geometry=new_geo)
        # The freehand analysis region is display-space too; rotate it alongside.
        if config.process.analysis_rect is not None:
            new_rect = rotate_normalized_rect(config.process.analysis_rect, visual_turns_ccw)
            new_config = replace(new_config, process=replace(config.process, analysis_rect=new_rect))
        self.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def flip(self, axis: str) -> None:
        from dataclasses import replace

        from negpy.features.geometry.logic import mirror_normalized_rect, toggle_flip

        horizontal = axis == "horizontal"
        config = self.session.state.config
        # toggle_flip negates fine rotation and mirrors the crop rect so the
        # result is a true mirror of the current render (see its docstring).
        new_config = replace(config, geometry=toggle_flip(config.geometry, horizontal))
        # The freehand analysis region is transformed-space like the crop rect;
        # mirroring it keeps the meters reading the same picture content.
        if config.process.analysis_rect is not None:
            new_rect = mirror_normalized_rect(config.process.analysis_rect, horizontal)
            new_config = replace(new_config, process=replace(config.process, analysis_rect=new_rect))
        self.session.update_config(new_config, persist=True)
        self.controller.request_render()

    def _show_tour(self) -> None:
        from negpy.desktop.view.main_window import MainWindow

        win = self.window()
        if isinstance(win, MainWindow):
            win.show_tutorial()

    def _show_shortcuts(self) -> None:
        from negpy.desktop.view.widgets.shortcuts_overlay import ShortcutsOverlay

        dlg = ShortcutsOverlay(self.window().shortcut_manager, self.window())
        dlg.exec()

    def _update_ui_state(self) -> None:
        state = self.session.state
        model = self.session.asset_model
        display_idx = model.actual_to_display(state.selected_file_idx)
        self.btn_prev.setEnabled(display_idx > 0)
        self.btn_next.setEnabled(0 <= display_idx < model.rowCount() - 1)
        self.btn_hq.setChecked(state.hq_preview)
        self._ov_hq_action.setChecked(state.hq_preview)

        geo = state.config.geometry
        self.btn_flip_h.setChecked(geo.flip_horizontal)
        self.btn_flip_v.setChecked(geo.flip_vertical)
        self._ov_flip_h_action.setChecked(geo.flip_horizontal)
        self._ov_flip_v_action.setChecked(geo.flip_vertical)

        self.btn_undo.setEnabled(state.undo_index > 0)
        self.btn_redo.setEnabled(state.undo_index < state.max_history_index)
        self._action_paste.setEnabled(state.clipboard is not None)

    def set_available_width(self, w: int) -> None:
        """Responsive overflow keyed on the hosting canvas width (the toolbar itself
        floats at sizeHint size, so its own width no longer reflects available space)."""
        show_swatches_hq = w >= 720
        show_flip_rotate = w >= 580

        for widget in self._ov_swatches_hq:
            widget.setVisible(show_swatches_hq)
        self._ov_hq_action.setVisible(not show_swatches_hq)
        for action in self._ov_color_actions:
            action.setVisible(not show_swatches_hq)

        for widget in self._ov_flip_rotate:
            widget.setVisible(show_flip_rotate)
        self._ov_sep_main.setVisible(not show_flip_rotate)
        self._ov_rot_l_action.setVisible(not show_flip_rotate)
        self._ov_rot_r_action.setVisible(not show_flip_rotate)
        self._ov_flip_h_action.setVisible(not show_flip_rotate)
        self._ov_flip_v_action.setVisible(not show_flip_rotate)
        self._ov_sep_rotate.setVisible(not show_flip_rotate)
