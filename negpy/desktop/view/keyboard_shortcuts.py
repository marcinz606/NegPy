from collections.abc import Callable

from PyQt6.QtGui import QKeySequence, QShortcut

from negpy.desktop.session import ToolMode
from negpy.desktop.view.widgets.granular_settings_dialog import open_paste_dialog
from negpy.desktop.view.shortcut_registry import (
    REGISTRY,
    load_bindings,
    load_slider_steps,
    save_bindings,
    save_slider_steps,
    set_current_bindings,
    slider_step_for,
)
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUP_BY_ACTION, SLIDER_GROUPS, sign_for_action
from negpy.desktop.view.slider_targets import slider_widget_map


def _context_undo(controller) -> None:
    """Ctrl+Z targets what the user is working on: while a heal/scratch tool is
    active it removes the last placed heal; otherwise it's the normal edit undo."""
    if controller.session.state.active_tool in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK):
        controller.undo_last_retouch()
    else:
        controller.session.undo()


def _context_cancel(controller, window) -> None:
    """Esc ladder: a test strip is dismissed first (it owns the canvas while up), then the
    grain focuser loupe, then an armed zone, then in-progress tool geometry (polyline
    points, straighten line, zone pins), then the tool itself."""
    if controller.state.test_strip or controller.state.test_strip_pending:
        controller.toggle_test_strip(force=False)
        return
    if controller.state.grain_focuser:
        controller.toggle_grain_focuser(force=False)
        return
    if controller.state.zone_arm_target is not None:
        controller._disarm_zone_target()
        return
    if controller.state.zone_pins:
        controller.clear_zone_pins()
        return
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
            "scan_setup": lambda: controls.process_sidebar.scan_setup_btn.click(),
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
            "toggle_zones": controller.toggle_zones_overlay,
            "toggle_test_strip": controller.toggle_test_strip,
            "toggle_ring_around": controller.toggle_ring_around,
            "toggle_grain_focuser": controller.toggle_grain_focuser,
            "toggle_printing_notes": controller.toggle_printing_notes,
            "cancel_tool": lambda: _context_cancel(controller, self.window),
            "show_library": self.window.session_panel.show_library,
            "browse_parent": self.window.session_panel.browse_parent,
            "focus_search": self.window.session_panel.file_browser.focus_search,
            "search_library": self.window.session_panel.file_browser.search_library,
            "toggle_library_tree": self.window.session_panel.toggle_library_tree,
            "toggle_immersive_canvas": lambda: controller.session.set_immersive_canvas(not controller.session.state.immersive_canvas),
            "toggle_sticky_zoom": lambda: controller.session.set_sticky_zoom(not controller.session.state.sticky_zoom),
            "toggle_left_panel": self.window.toggle_session_dock,
            "toggle_right_panel": self.window.toggle_controls_dock,
            "reset_panel_layout": self.window.reset_panel_layout,
            "tab_favourites": lambda: right.show_tab_by_key("favourites"),
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
            "export_linear_output": controller.request_linear_output_export,
            "copy": controller.session.copy_settings,
            "copy_with_bounds": controller.session.copy_settings_with_bounds,
            "paste": lambda: open_paste_dialog(self.window, controller),
            "save_work_print": self.window.right_panel.history_panel.save_work_print,
            "undo": lambda: _context_undo(controller),
            "redo": controller.session.redo,
            "show_shortcuts": lambda: _show_shortcuts(self.window),
            "show_analysis_help": self.window.right_panel.show_analysis_help,
        }

        widgets = slider_widget_map(controls)
        for group in SLIDER_GROUPS:
            getter = widgets[group.id]
            actions[group.inc_action] = self._slider_adjuster(getter, group.inc_action)
            actions[group.dec_action] = self._slider_adjuster(getter, group.dec_action)
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
