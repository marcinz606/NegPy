from collections.abc import Callable

from PyQt6.QtGui import QKeySequence, QShortcut

from negpy.desktop.view.shortcut_registry import REGISTRY, load_bindings, save_bindings, set_current_bindings


def _show_shortcuts(window) -> None:
    from negpy.desktop.view.widgets.shortcuts_overlay import ShortcutsOverlay

    dlg = ShortcutsOverlay(window.shortcut_manager, window)
    dlg.exec()


class ShortcutManager:
    def __init__(self, window):
        self.window = window
        self.bindings = load_bindings(window.controller.session.repo)
        self._shortcuts: list[QShortcut] = []
        self._actions = self._build_actions()
        self.apply_bindings(self.bindings)

    def _slider_adjuster(self, getter: Callable[[], object], direction: float) -> Callable[[], None]:
        def _adjust() -> None:
            slider = getter()
            step = slider.spin.singleStep()
            slider.adjust_by(step * direction)

        return _adjust

    def _build_actions(self) -> dict[str, Callable[[], None]]:
        controller = self.window.controller
        toolbar = self.window.toolbar
        controls = self.window.controls_panel
        right = self.window.right_panel

        actions: dict[str, Callable[[], None]] = {
            "prev_file": controller.session.prev_file,
            "next_file": controller.session.next_file,
            "toggle_compare": controller.toggle_compare,
            "rotate_ccw": lambda: toolbar.rotate(1),
            "rotate_cw": lambda: toolbar.rotate(-1),
            "flip_h": lambda: toolbar.flip("horizontal"),
            "flip_v": lambda: toolbar.flip("vertical"),
            "lock_bounds_toggle": lambda: controls.process_sidebar.lock_bounds_btn.toggle(),
            "pick_wb": lambda: controls.colour_sidebar.pick_wb_btn.toggle(),
            "manual_crop": lambda: controls.geometry_sidebar.manual_crop_btn.toggle(),
            "auto_crop": lambda: controls.geometry_sidebar.reset_crop_btn.toggle(),
            "pick_dust": lambda: controls.retouch_sidebar.pick_dust_btn.toggle(),
            "cancel_tool": controller.cancel_active_tool,
            "toggle_left_panel": self.window.toggle_session_dock,
            "toggle_right_panel": self.window.toggle_controls_dock,
            "tab_setup": lambda: right.show_tab_by_key("setup"),
            "tab_geometry": lambda: right.show_tab_by_key("geometry"),
            "tab_tone": lambda: right.show_tab_by_key("tone"),
            "tab_color": lambda: right.show_tab_by_key("color"),
            "tab_finish": lambda: right.show_tab_by_key("finish"),
            "tab_export": lambda: right.show_tab_by_key("export"),
            "tab_metadata": lambda: right.show_tab_by_key("metadata"),
            "tab_history": lambda: right.show_tab_by_key("history"),
            "tab_scan": lambda: right.show_tab_by_key("scan"),
            "fit_view": self.window.canvas.fit_to_window,
            "zoom_100": lambda: controller.zoom_requested.emit(1.0),
            "zoom_200": lambda: controller.zoom_requested.emit(2.0),
            "export": controller.request_export,
            "copy": controller.session.copy_settings,
            "copy_with_bounds": controller.session.copy_settings_with_bounds,
            "paste": controller.session.paste_settings,
            "undo": controller.session.undo,
            "redo": controller.session.redo,
            "show_shortcuts": lambda: _show_shortcuts(self.window),
        }

        slider_targets: dict[str, tuple[Callable[[], object], float]] = {
            "cyan_inc": (lambda: controls.colour_sidebar.cyan_slider, 1.0),
            "cyan_dec": (lambda: controls.colour_sidebar.cyan_slider, -1.0),
            "magenta_up": (lambda: controls.colour_sidebar.magenta_slider, 1.0),
            "magenta_down": (lambda: controls.colour_sidebar.magenta_slider, -1.0),
            "yellow_up": (lambda: controls.colour_sidebar.yellow_slider, 1.0),
            "yellow_down": (lambda: controls.colour_sidebar.yellow_slider, -1.0),
            "density_up": (lambda: controls.tone_sidebar.density_slider, 1.0),
            "density_down": (lambda: controls.tone_sidebar.density_slider, -1.0),
            # ISO R scale is inverted: harder grade = lower R.
            "grade_up": (lambda: controls.tone_sidebar.grade_slider, -10.0),
            "grade_down": (lambda: controls.tone_sidebar.grade_slider, 10.0),
            "toe_inc": (lambda: controls.tone_sidebar.toe_slider, 1.0),
            "toe_dec": (lambda: controls.tone_sidebar.toe_slider, -1.0),
            "toe_width_inc": (lambda: controls.tone_sidebar.toe_w_slider, 1.0),
            "toe_width_dec": (lambda: controls.tone_sidebar.toe_w_slider, -1.0),
            "shoulder_inc": (lambda: controls.tone_sidebar.sh_slider, 1.0),
            "shoulder_dec": (lambda: controls.tone_sidebar.sh_slider, -1.0),
            "shoulder_width_inc": (lambda: controls.tone_sidebar.sh_w_slider, 1.0),
            "shoulder_width_dec": (lambda: controls.tone_sidebar.sh_w_slider, -1.0),
            "offset_inc": (lambda: controls.geometry_sidebar.offset_slider, 1.0),
            "offset_dec": (lambda: controls.geometry_sidebar.offset_slider, -1.0),
            "fine_rot_inc": (lambda: controls.geometry_sidebar.fine_rot_slider, 1.0),
            "fine_rot_dec": (lambda: controls.geometry_sidebar.fine_rot_slider, -1.0),
            "analysis_buffer_inc": (lambda: controls.process_sidebar.analysis_buffer_slider, 1.0),
            "analysis_buffer_dec": (lambda: controls.process_sidebar.analysis_buffer_slider, -1.0),
            "luma_range_clip_inc": (lambda: controls.process_sidebar.luma_range_clip_slider, 1.0),
            "luma_range_clip_dec": (lambda: controls.process_sidebar.luma_range_clip_slider, -1.0),
            "color_range_clip_inc": (lambda: controls.process_sidebar.color_range_clip_slider, 1.0),
            "color_range_clip_dec": (lambda: controls.process_sidebar.color_range_clip_slider, -1.0),
            "white_point_inc": (lambda: controls.process_sidebar.white_point_slider, 1.0),
            "white_point_dec": (lambda: controls.process_sidebar.white_point_slider, -1.0),
            "black_point_inc": (lambda: controls.process_sidebar.black_point_slider, 1.0),
            "black_point_dec": (lambda: controls.process_sidebar.black_point_slider, -1.0),
            "separation_inc": (lambda: controls.process_sidebar.crosstalk_strength_slider, 1.0),
            "separation_dec": (lambda: controls.process_sidebar.crosstalk_strength_slider, -1.0),
            "chroma_denoise_inc": (lambda: controls.lab_sidebar.chroma_denoise_slider, 1.0),
            "chroma_denoise_dec": (lambda: controls.lab_sidebar.chroma_denoise_slider, -1.0),
            "saturation_inc": (lambda: controls.lab_sidebar.saturation_slider, 1.0),
            "saturation_dec": (lambda: controls.lab_sidebar.saturation_slider, -1.0),
            "vibrance_inc": (lambda: controls.lab_sidebar.vibrance_slider, 1.0),
            "vibrance_dec": (lambda: controls.lab_sidebar.vibrance_slider, -1.0),
            "clahe_inc": (lambda: controls.lab_sidebar.clahe_slider, 1.0),
            "clahe_dec": (lambda: controls.lab_sidebar.clahe_slider, -1.0),
            "sharpen_inc": (lambda: controls.lab_sidebar.sharpen_slider, 1.0),
            "sharpen_dec": (lambda: controls.lab_sidebar.sharpen_slider, -1.0),
            "glow_inc": (lambda: controls.lab_sidebar.glow_slider, 1.0),
            "glow_dec": (lambda: controls.lab_sidebar.glow_slider, -1.0),
            "halation_inc": (lambda: controls.lab_sidebar.halation_slider, 1.0),
            "halation_dec": (lambda: controls.lab_sidebar.halation_slider, -1.0),
            "threshold_inc": (lambda: controls.retouch_sidebar.threshold_slider, 1.0),
            "threshold_dec": (lambda: controls.retouch_sidebar.threshold_slider, -1.0),
            "auto_size_inc": (lambda: controls.retouch_sidebar.auto_size_slider, 1.0),
            "auto_size_dec": (lambda: controls.retouch_sidebar.auto_size_slider, -1.0),
            "manual_size_inc": (lambda: controls.retouch_sidebar.manual_size_slider, 1.0),
            "manual_size_dec": (lambda: controls.retouch_sidebar.manual_size_slider, -1.0),
            "selenium_inc": (lambda: controls.toning_sidebar.selenium_slider, 1.0),
            "selenium_dec": (lambda: controls.toning_sidebar.selenium_slider, -1.0),
            "sepia_inc": (lambda: controls.toning_sidebar.sepia_slider, 1.0),
            "sepia_dec": (lambda: controls.toning_sidebar.sepia_slider, -1.0),
            "shadow_hue_inc": (lambda: controls.toning_sidebar.shadow_hue_slider, 1.0),
            "shadow_hue_dec": (lambda: controls.toning_sidebar.shadow_hue_slider, -1.0),
            "shadow_strength_inc": (lambda: controls.toning_sidebar.shadow_str_slider, 1.0),
            "shadow_strength_dec": (lambda: controls.toning_sidebar.shadow_str_slider, -1.0),
            "highlight_hue_inc": (lambda: controls.toning_sidebar.highlight_hue_slider, 1.0),
            "highlight_hue_dec": (lambda: controls.toning_sidebar.highlight_hue_slider, -1.0),
            "highlight_strength_inc": (lambda: controls.toning_sidebar.highlight_str_slider, 1.0),
            "highlight_strength_dec": (lambda: controls.toning_sidebar.highlight_str_slider, -1.0),
            "vignette_str_inc": (lambda: controls.finish_sidebar.vignette_strength_slider, 1.0),
            "vignette_str_dec": (lambda: controls.finish_sidebar.vignette_strength_slider, -1.0),
            "vignette_size_inc": (lambda: controls.finish_sidebar.vignette_size_slider, 1.0),
            "vignette_size_dec": (lambda: controls.finish_sidebar.vignette_size_slider, -1.0),
            "border_size_inc": (lambda: controls.finish_sidebar.border_slider, 1.0),
            "border_size_dec": (lambda: controls.finish_sidebar.border_slider, -1.0),
        }
        for action_id, (getter, direction) in slider_targets.items():
            actions[action_id] = self._slider_adjuster(getter, direction)
        return actions

    def apply_bindings(self, bindings: dict[str, str]) -> None:
        self.bindings = dict(bindings)
        set_current_bindings(self.bindings)
        for shortcut in self._shortcuts:
            shortcut.setParent(None)
        self._shortcuts.clear()

        for action_id, callback in self._actions.items():
            key = self.bindings.get(action_id, "")
            if not key:
                continue
            shortcut = QShortcut(QKeySequence(key), self.window)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

        self.window.controls_panel.apply_shortcut_tooltips()
        self.window.right_panel.apply_shortcut_tooltips()

    def update_bindings(self, bindings: dict[str, str]) -> None:
        save_bindings(self.window.controller.session.repo, bindings)
        self.apply_bindings(bindings)

    def open_editor(self, parent=None) -> bool:
        from negpy.desktop.view.widgets.shortcut_editor import ShortcutEditorDialog

        dlg = ShortcutEditorDialog(self.bindings, parent or self.window)
        if dlg.exec():
            self.update_bindings(dlg.bindings())
            return True
        return False


def setup_keyboard_shortcuts(window) -> ShortcutManager:
    manager = ShortcutManager(window)
    missing = [action_id for action_id in REGISTRY if action_id not in manager._actions]
    if missing:
        raise RuntimeError(f"Shortcut actions missing handlers: {missing}")
    return manager
