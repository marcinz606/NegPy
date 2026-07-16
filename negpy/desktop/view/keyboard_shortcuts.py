from collections.abc import Callable

from PyQt6.QtGui import QKeySequence, QShortcut

from negpy.desktop.session import ToolMode
from negpy.desktop.view.shortcut_registry import (
    REGISTRY,
    load_bindings,
    load_slider_steps,
    save_bindings,
    save_slider_steps,
    set_current_bindings,
    slider_step_for,
)
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUP_BY_ACTION, sign_for_action


def _context_undo(controller) -> None:
    """Ctrl+Z targets what the user is working on: while a heal/scratch tool is
    active it removes the last placed heal; otherwise it's the normal edit undo."""
    if controller.session.state.active_tool in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK):
        controller.undo_last_retouch()
    else:
        controller.session.undo()


def _context_cancel(controller, window) -> None:
    """Esc ladder: the first press clears in-progress tool geometry (polyline
    points, straighten line), the second puts the tool down."""
    if not window.canvas.overlay.cancel_in_progress():
        controller.cancel_active_tool()


def _toggle_tool_button(window, tab_key: str, button) -> None:
    """Reveal the tool's tab first: the tab-switch suspend/restore logic only runs
    on switches, so activating a tool while its tab is hidden would leave it live
    with its controls off-screen."""
    window.right_panel.show_tab_by_key(tab_key)
    button.toggle()


def _show_shortcuts(window) -> None:
    from negpy.desktop.view.widgets.shortcuts_overlay import ShortcutsOverlay

    dlg = ShortcutsOverlay(window.shortcut_manager, window)
    dlg.exec()


class ShortcutManager:
    def __init__(self, window):
        self.window = window
        self.bindings = load_bindings(window.controller.session.repo)
        self.slider_steps = load_slider_steps(window.controller.session.repo)
        self._shortcuts: list[QShortcut] = []
        self._actions = self._build_actions()
        self.apply_bindings(self.bindings)

    def _slider_adjuster(self, getter: Callable[[], object], action_id: str) -> Callable[[], None]:
        group = SLIDER_GROUP_BY_ACTION[action_id]

        def _adjust() -> None:
            step = slider_step_for(group.id, self.slider_steps)
            getter().adjust_by(step * sign_for_action(action_id))

        return _adjust

    def _build_actions(self) -> dict[str, Callable[[], None]]:
        controller = self.window.controller
        toolbar = self.window.toolbar
        controls = self.window.controls_panel
        right = self.window.right_panel

        actions: dict[str, Callable[[], None]] = {
            "prev_file": controller.session.prev_file,
            "next_file": controller.session.next_file,
            "toggle_keep": lambda: controller.session.toggle_mark("keeper"),
            "toggle_reject": lambda: controller.session.toggle_mark("excluded"),
            "toggle_compare": controller.toggle_compare,
            "rotate_ccw": lambda: toolbar.rotate(1),
            "rotate_cw": lambda: toolbar.rotate(-1),
            "flip_h": lambda: toolbar.flip("horizontal"),
            "flip_v": lambda: toolbar.flip("vertical"),
            "lock_bounds_toggle": lambda: controls.process_sidebar.lock_bounds_btn.toggle(),
            "pick_wb": lambda: controls.colour_sidebar.pick_wb_btn.toggle(),
            "manual_crop": lambda: controls.geometry_sidebar.manual_crop_btn.toggle(),
            "straighten": lambda: controls.geometry_sidebar.straighten_btn.toggle(),
            "crop_guide_next": lambda: controls.geometry_sidebar.cycle_guide(),
            "crop_guide_orient": controller.cycle_crop_guide_orientation,
            "auto_crop": lambda: controls.geometry_sidebar.reset_crop_btn.toggle(),
            "pick_dust": lambda: _toggle_tool_button(self.window, "finish", controls.retouch_sidebar.pick_dust_btn),
            "pick_scratch": lambda: _toggle_tool_button(self.window, "finish", controls.retouch_sidebar.pick_scratch_btn),
            "local_draw": lambda: _toggle_tool_button(self.window, "tone", controls.local_sidebar.draw_btn),
            "analysis_draw": lambda: _toggle_tool_button(self.window, "setup", controls.process_sidebar.analysis_region_btn),
            "toggle_flat_peek": controller.toggle_flat_peek,
            "cancel_tool": lambda: _context_cancel(controller, self.window),
            "toggle_left_panel": self.window.toggle_session_dock,
            "toggle_right_panel": self.window.toggle_controls_dock,
            "reset_panel_layout": self.window.reset_panel_layout,
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
            "zoom_100": self.window.canvas.zoom_to_original,
            "zoom_200": lambda: self.window.canvas.zoom_to_percent(200.0),
            "export": controller.request_export,
            "copy": controller.session.copy_settings,
            "copy_with_bounds": controller.session.copy_settings_with_bounds,
            "paste": controller.session.paste_settings,
            "undo": lambda: _context_undo(controller),
            "redo": controller.session.redo,
            "show_shortcuts": lambda: _show_shortcuts(self.window),
        }

        slider_targets: dict[str, Callable[[], object]] = {
            "cyan_inc": lambda: controls.colour_sidebar.cyan_slider,
            "cyan_dec": lambda: controls.colour_sidebar.cyan_slider,
            "magenta_up": lambda: controls.colour_sidebar.magenta_slider,
            "magenta_down": lambda: controls.colour_sidebar.magenta_slider,
            "yellow_up": lambda: controls.colour_sidebar.yellow_slider,
            "yellow_down": lambda: controls.colour_sidebar.yellow_slider,
            "temp_warm": lambda: controls.colour_sidebar.temp_slider,
            "temp_cool": lambda: controls.colour_sidebar.temp_slider,
            "density_up": lambda: controls.tone_sidebar.density_slider,
            "density_down": lambda: controls.tone_sidebar.density_slider,
            "grade_up": lambda: controls.tone_sidebar.grade_slider,
            "grade_down": lambda: controls.tone_sidebar.grade_slider,
            "toe_inc": lambda: controls.tone_sidebar.toe_slider,
            "toe_dec": lambda: controls.tone_sidebar.toe_slider,
            "toe_width_inc": lambda: controls.tone_sidebar.toe_w_slider,
            "toe_width_dec": lambda: controls.tone_sidebar.toe_w_slider,
            "shoulder_inc": lambda: controls.tone_sidebar.sh_slider,
            "shoulder_dec": lambda: controls.tone_sidebar.sh_slider,
            "shoulder_width_inc": lambda: controls.tone_sidebar.sh_w_slider,
            "shoulder_width_dec": lambda: controls.tone_sidebar.sh_w_slider,
            "snap_inc": lambda: controls.tone_sidebar.midtone_gamma_slider,
            "snap_dec": lambda: controls.tone_sidebar.midtone_gamma_slider,
            "shadow_density_inc": lambda: controls.tone_sidebar.shadow_density_slider,
            "shadow_density_dec": lambda: controls.tone_sidebar.shadow_density_slider,
            "highlight_density_inc": lambda: controls.tone_sidebar.highlight_density_slider,
            "highlight_density_dec": lambda: controls.tone_sidebar.highlight_density_slider,
            "shadow_grade_inc": lambda: controls.tone_sidebar.shadow_grade_slider,
            "shadow_grade_dec": lambda: controls.tone_sidebar.shadow_grade_slider,
            "highlight_grade_inc": lambda: controls.tone_sidebar.highlight_grade_slider,
            "highlight_grade_dec": lambda: controls.tone_sidebar.highlight_grade_slider,
            "offset_inc": lambda: controls.geometry_sidebar.offset_slider,
            "offset_dec": lambda: controls.geometry_sidebar.offset_slider,
            "fine_rot_inc": lambda: controls.geometry_sidebar.fine_rot_slider,
            "fine_rot_dec": lambda: controls.geometry_sidebar.fine_rot_slider,
            "analysis_buffer_inc": lambda: controls.process_sidebar.analysis_buffer_slider,
            "analysis_buffer_dec": lambda: controls.process_sidebar.analysis_buffer_slider,
            "luma_range_clip_inc": lambda: controls.process_sidebar.luma_range_clip_slider,
            "luma_range_clip_dec": lambda: controls.process_sidebar.luma_range_clip_slider,
            "color_range_clip_inc": lambda: controls.process_sidebar.color_range_clip_slider,
            "color_range_clip_dec": lambda: controls.process_sidebar.color_range_clip_slider,
            "white_point_inc": lambda: controls.process_sidebar.white_point_slider,
            "white_point_dec": lambda: controls.process_sidebar.white_point_slider,
            "black_point_inc": lambda: controls.process_sidebar.black_point_slider,
            "black_point_dec": lambda: controls.process_sidebar.black_point_slider,
            "separation_inc": lambda: controls.process_sidebar.crosstalk_strength_slider,
            "separation_dec": lambda: controls.process_sidebar.crosstalk_strength_slider,
            "chroma_denoise_inc": lambda: controls.lab_sidebar.chroma_denoise_slider,
            "chroma_denoise_dec": lambda: controls.lab_sidebar.chroma_denoise_slider,
            "saturation_inc": lambda: controls.lab_sidebar.saturation_slider,
            "saturation_dec": lambda: controls.lab_sidebar.saturation_slider,
            "vibrance_inc": lambda: controls.lab_sidebar.vibrance_slider,
            "vibrance_dec": lambda: controls.lab_sidebar.vibrance_slider,
            "clahe_inc": lambda: controls.lab_sidebar.clahe_slider,
            "clahe_dec": lambda: controls.lab_sidebar.clahe_slider,
            "sharpen_inc": lambda: controls.lab_sidebar.sharpen_slider,
            "sharpen_dec": lambda: controls.lab_sidebar.sharpen_slider,
            "glow_inc": lambda: controls.lab_sidebar.glow_slider,
            "glow_dec": lambda: controls.lab_sidebar.glow_slider,
            "halation_inc": lambda: controls.lab_sidebar.halation_slider,
            "halation_dec": lambda: controls.lab_sidebar.halation_slider,
            "threshold_inc": lambda: controls.retouch_sidebar.threshold_slider,
            "threshold_dec": lambda: controls.retouch_sidebar.threshold_slider,
            "auto_size_inc": lambda: controls.retouch_sidebar.auto_size_slider,
            "auto_size_dec": lambda: controls.retouch_sidebar.auto_size_slider,
            "manual_size_inc": lambda: controls.retouch_sidebar.manual_size_slider,
            "manual_size_dec": lambda: controls.retouch_sidebar.manual_size_slider,
            "selenium_inc": lambda: controls.toning_sidebar.selenium_slider,
            "selenium_dec": lambda: controls.toning_sidebar.selenium_slider,
            "sepia_inc": lambda: controls.toning_sidebar.sepia_slider,
            "sepia_dec": lambda: controls.toning_sidebar.sepia_slider,
            "shadow_hue_inc": lambda: controls.toning_sidebar.shadow_hue_slider,
            "shadow_hue_dec": lambda: controls.toning_sidebar.shadow_hue_slider,
            "shadow_strength_inc": lambda: controls.toning_sidebar.shadow_str_slider,
            "shadow_strength_dec": lambda: controls.toning_sidebar.shadow_str_slider,
            "highlight_hue_inc": lambda: controls.toning_sidebar.highlight_hue_slider,
            "highlight_hue_dec": lambda: controls.toning_sidebar.highlight_hue_slider,
            "highlight_strength_inc": lambda: controls.toning_sidebar.highlight_str_slider,
            "highlight_strength_dec": lambda: controls.toning_sidebar.highlight_str_slider,
            "vignette_str_inc": lambda: controls.finish_sidebar.vignette_strength_slider,
            "vignette_str_dec": lambda: controls.finish_sidebar.vignette_strength_slider,
            "vignette_size_inc": lambda: controls.finish_sidebar.vignette_size_slider,
            "vignette_size_dec": lambda: controls.finish_sidebar.vignette_size_slider,
            "border_size_inc": lambda: controls.finish_sidebar.border_slider,
            "border_size_dec": lambda: controls.finish_sidebar.border_slider,
        }
        for action_id, getter in slider_targets.items():
            actions[action_id] = self._slider_adjuster(getter, action_id)
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

    def update_slider_steps(self, steps: dict[str, float]) -> None:
        save_slider_steps(self.window.controller.session.repo, steps)
        self.slider_steps = dict(steps)

    def open_editor(self, parent=None) -> bool:
        from negpy.desktop.view.widgets.shortcut_editor import ShortcutEditorDialog

        dlg = ShortcutEditorDialog(
            self.bindings,
            self.slider_steps,
            parent or self.window,
            session=self.window.controller.session,
        )
        if dlg.exec():
            self.update_bindings(dlg.bindings())
            self.update_slider_steps(dlg.slider_steps())
            return True
        return False


def setup_keyboard_shortcuts(window) -> ShortcutManager:
    manager = ShortcutManager(window)
    missing = [action_id for action_id in REGISTRY if action_id not in manager._actions]
    if missing:
        raise RuntimeError(f"Shortcut actions missing handlers: {missing}")
    return manager
