from collections.abc import Callable
from typing import Optional

from PyQt6.QtGui import QKeySequence, QShortcut

from negpy.desktop.session import ToolMode
from negpy.desktop.view.confirm import confirm_reset_frames
from negpy.desktop.view.widgets.granular_settings_dialog import open_paste_dialog, open_sticky_dialog, open_sync_bounds_dialog
from negpy.desktop.view.shortcut_registry import (
    REGISTRY,
    load_bindings,
    load_slider_steps,
    save_bindings,
    save_slider_steps,
    set_current_bindings,
    slider_step_for,
)
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUP_BY_ACTION, SLIDER_GROUPS, SliderShortcutGroup, sign_for_action
from negpy.desktop.view.slider_targets import slider_widget_map
from negpy.desktop.view.widgets.collapsible import hidden_by_gating


def _context_undo(controller) -> None:
    """Ctrl+Z targets what the user is working on: while a heal/scratch tool is
    active it removes the last placed heal; otherwise it's the normal edit undo."""
    if controller.session.state.active_tool in (ToolMode.DUST_PICK, ToolMode.SCRATCH_PICK, ToolMode.SCRATCH_LINE):
        controller.undo_last_retouch()
    else:
        controller.session.undo()


def _fire_tab_header(right, action: str) -> None:
    """The tab in front answers for its own cards; a tab that owns no settings has no
    header and nothing happens."""
    header = right.active_tab_header()
    if header is None:
        return
    if action == "cards":
        header.cards_btn.click()
        return
    signal = {"reset": header.reset_requested, "revert": header.roll_revert_requested}.get(action, header.apply_requested)
    signal.emit()


def _reset_roll(window, controller) -> None:
    count = len(controller.session.asset_model.visible_actual_indices_ordered())
    if count and confirm_reset_frames(window, count, roll=True) and controller.session.reset_roll_settings(scope="roll"):
        controller.request_render()


def _reset_selected(window, controller) -> None:
    count = len(controller.session.state.selected_indices)
    if count and confirm_reset_frames(window, count) and controller.session.reset_roll_settings(scope="selection"):
        controller.request_render()


def _context_cancel(controller, window) -> None:
    """Esc ladder: whatever has taken the canvas over goes first — a test strip, any peek,
    the before/after split — then the grain focuser loupe, then an armed zone, then
    in-progress tool geometry (polyline points, straighten line, zone pins), then the tool
    itself."""
    if controller.state.test_strip or controller.state.test_strip_pending:
        controller.toggle_test_strip(force=False)
        return
    if controller.state.negative_peek:
        controller.toggle_negative_peek(force=False)
        return
    if controller.state.embedded_peek:
        controller.toggle_embedded_peek(force=False)
        return
    if controller.state.flat_peek:
        controller.toggle_flat_peek(force=False)
        return
    if controller.state.compare_mode:
        controller.toggle_compare()
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


def _slider_name(slider: object, group: SliderShortcutGroup) -> str:
    label = getattr(slider, "label", None)
    return label.text() if label is not None else group.label.replace(" ↑/↓", "")


def _show_shortcuts(window) -> None:
    from negpy.desktop.view.widgets.shortcuts_overlay import ShortcutsOverlay

    dlg = ShortcutsOverlay(window.shortcut_manager, window)
    dlg.exec()


def _open_preferences(window, controller) -> None:
    # Deferred: the dialog reaches the toolbar for the canvas palette, which imports this module.
    from negpy.desktop.view.widgets.preferences_dialog import open_preferences

    open_preferences(window, controller)


class ShortcutManager:
    def __init__(self, window):
        self.window = window
        self.bindings = load_bindings(window.controller.session.repo)
        self.slider_steps = load_slider_steps(window.controller.session.repo)
        self._shortcuts: list[QShortcut] = []
        self._actions = self._build_actions()
        self.apply_bindings(self.bindings)

    def _toggle_slider_values(self) -> None:
        from negpy.desktop.view.widgets.sliders import apply_slider_value_visibility

        repo = self.window.controller.session.repo
        pinned = not bool(repo.get_global_setting("show_slider_values", default=False))
        repo.save_global_setting("show_slider_values", pinned)
        apply_slider_value_visibility(self.window, pinned)

    def _slider_adjuster(self, getter: Callable[[], object], action_id: str) -> Callable[[], None]:
        group = SLIDER_GROUP_BY_ACTION[action_id]

        def _adjust() -> None:
            slider = getter()
            # A window-wide QShortcut fires from any tab, so the gating a mouse gets for free on a
            # disabled or mode-hidden control has to be applied here by hand.
            if not slider.isEnabled() or hidden_by_gating(slider):
                self.window.controller.set_status(f"{_slider_name(slider, group)} not available", 1500, kind="warning")
                return
            step = slider_step_for(group.id, self.slider_steps)
            slider.adjust_by(step * sign_for_action(action_id))
            self._announce(slider, group)

        return _adjust

    def _announce(self, slider: object, group: SliderShortcutGroup) -> None:
        """Report the new value in the HUD. Nothing else does: the slider may sit on a
        hidden tab, and its value box only appears under the pointer."""
        name = _slider_name(slider, group)
        spin = getattr(slider, "spin", None)
        # The spin box already carries this slider's decimals and unit suffix.
        value = spin.text().strip() if spin is not None else f"{slider.value():.2f}"
        self.window.controller.set_status(f"{name} {value}", 1500)

    def _build_actions(self) -> dict[str, Callable[[], None]]:
        controller = self.window.controller
        toolbar = self.window.toolbar
        controls = self.window.controls_panel
        right = self.window.right_panel

        actions: dict[str, Callable[[], None]] = {
            "prev_file": controller.session.prev_file,
            "next_file": controller.session.next_file,
            "toggle_keep": lambda: controller.session.toggle_mark("keeper"),
            "toggle_scene_overlay": lambda: self.window.session_panel.file_browser.scenes_btn.click(),
            "hdr_merge": controller.request_hdr_merge_selected,
            "hdr_unmerge": controller.request_unmerge_hdr,
            # The view method, not the controller's: it carries the confirm the deletion needs.
            "half_frame_undiptych": self.window.session_panel.file_browser.prompt_undiptych,
            "update_thumbnails_selection": (
                lambda: controller.cancel_thumbnail_refresh()
                if controller.thumbnail_refresh_running
                else controller.request_thumbnail_refresh("selection")
            ),
            "update_thumbnails_roll": (
                lambda: controller.cancel_thumbnail_refresh()
                if controller.thumbnail_refresh_running
                else controller.request_thumbnail_refresh("roll")
            ),
            "toggle_reject": lambda: controller.session.toggle_mark("excluded"),
            "toggle_compare": controller.toggle_compare,
            "rotate_ccw": lambda: toolbar.rotate(1),
            "rotate_cw": lambda: toolbar.rotate(-1),
            "flip_h": lambda: toolbar.flip("horizontal"),
            "flip_v": lambda: toolbar.flip("vertical"),
            "lock_bounds_toggle": lambda: controls.process_sidebar.lock_bounds_btn.toggle(),
            "reanalyze_frame": lambda: controls.process_sidebar.reanalyze_frame_btn.click(),
            "cast_average_toggle": lambda: controls.process_sidebar.use_cast_avg_btn.toggle(),
            "metadata_preset_load": lambda: right.metadata_sidebar.metadata_preset_load_btn.click(),
            "metadata_clear_gear": lambda: right.metadata_sidebar.gear_clear_btn.click(),
            "metadata_clear_process": lambda: right.metadata_sidebar.process_clear_btn.click(),
            "metadata_clear_scanning": lambda: right.metadata_sidebar.scan_clear_btn.click(),
            "scan_setup": lambda: controls.sensor_sidebar.scan_setup_btn.click(),
            "scan_prescan": (lambda: right.scan_sidebar.prescan_btn.click() if getattr(right, "scan_sidebar", None) is not None else None),
            "mode_color_negative": lambda: controls.process_sidebar.mode_btns[0].click(),
            "mode_bw_negative": lambda: controls.process_sidebar.mode_btns[1].click(),
            "mode_transparency": lambda: controls.process_sidebar.mode_btns[2].click(),
            "pick_wb": lambda: controls.color_sidebar.pick_wb_btn.toggle(),
            "manual_crop": lambda: controls.geometry_sidebar.manual_crop_btn.toggle(),
            "straighten": lambda: controls.geometry_sidebar.straighten_btn.toggle(),
            "crop_guide_next": lambda: controls.geometry_sidebar.cycle_guide(),
            "crop_guide_orient": controller.cycle_crop_guide_orientation,
            "auto_crop": lambda: controls.geometry_sidebar.reset_crop_btn.toggle(),
            "crop_to_valid": lambda: controls.geometry_sidebar.crop_to_valid_btn.toggle(),
            "lens_distortion_from_metadata": lambda: controls.lens_sidebar.metadata_distortion_btn.click(),
            "lens_ca_from_metadata": lambda: controls.lens_sidebar.metadata_ca_btn.click(),
            "pick_dust": lambda: _toggle_tool_button(self.window, "finish", controls.retouch_sidebar.pick_dust_btn),
            "pick_scratch": lambda: _toggle_tool_button(self.window, "finish", controls.retouch_sidebar.pick_scratch_btn),
            "pick_scratch_line": lambda: _toggle_tool_button(self.window, "finish", controls.retouch_sidebar.pick_line_btn),
            "local_draw": lambda: _toggle_tool_button(self.window, "tone", controls.local_sidebar.draw_btn),
            "local_oval": lambda: _toggle_tool_button(self.window, "tone", controls.local_sidebar.oval_btn),
            "local_gradient": lambda: _toggle_tool_button(self.window, "tone", controls.local_sidebar.gradient_btn),
            "analysis_draw": lambda: _toggle_tool_button(self.window, "roll", controls.process_sidebar.analysis_region_btn),
            "toggle_flat_peek": controller.toggle_flat_peek,
            "toggle_negative_peek": controller.toggle_negative_peek,
            "toggle_embedded_peek": controller.toggle_embedded_peek,
            "toggle_zones": controller.toggle_zones_overlay,
            "toggle_test_strip": controller.toggle_test_strip,
            "toggle_ring_around": controller.toggle_ring_around,
            "toggle_grain_focuser": controller.toggle_grain_focuser,
            "toggle_lith": controller.toggle_lith,
            "toggle_cyanotype": controller.toggle_cyanotype,
            "toggle_printing_notes": controller.toggle_printing_notes,
            "toggle_soft_proof": lambda: controller.set_soft_proof(not controller.state.soft_proof_enabled),
            "cancel_tool": lambda: _context_cancel(controller, self.window),
            "show_library": self.window.session_panel.show_library,
            "focus_search": self.window.session_panel.file_browser.focus_search,
            "search_library": self.window.session_panel.file_browser.search_library,
            "toggle_library_tree": self.window.session_panel.toggle_library_tree,
            "library_discovery_filters": self.window.session_panel.library_tree.edit_discovery_filters,
            "toggle_immersive_canvas": lambda: controller.session.set_immersive_canvas(not controller.session.state.immersive_canvas),
            "toggle_sticky_zoom": lambda: controller.session.set_sticky_zoom(not controller.session.state.sticky_zoom),
            "toggle_sticky_settings": lambda: controller.session.set_sticky_settings_enabled(
                not controller.session.state.sticky_settings_enabled
            ),
            "toggle_slider_values": self._toggle_slider_values,
            "toggle_invert_zoom_scroll": lambda: controller.session.set_invert_zoom_scroll(not controller.session.state.invert_zoom_scroll),
            "toggle_left_panel": self.window.toggle_session_dock,
            "toggle_right_panel": self.window.toggle_controls_dock,
            "reset_panel_layout": self.window.reset_panel_layout,
            "edit_toolbar": toolbar.open_toolbar_editor,
            "tab_favourites": lambda: right.show_tab_by_key("favourites"),
            "tab_roll": lambda: right.show_tab_by_key("roll"),
            "tab_geometry": lambda: right.show_tab_by_key("geometry"),
            "tab_tone": lambda: right.show_tab_by_key("tone"),
            "tab_color": lambda: right.show_tab_by_key("color"),
            "tab_finish": lambda: right.show_tab_by_key("finish"),
            "tab_export": lambda: right.show_tab_by_key("export"),
            "tab_metadata": lambda: right.show_tab_by_key("metadata"),
            "tab_history": lambda: right.show_tab_by_key("history"),
            "tab_gear": lambda: right.show_tab_by_key("gear"),
            "tab_gear_items": lambda: right.show_gear_section_by_key("items"),
            "tab_gear_presets": lambda: right.show_gear_section_by_key("presets"),
            "tab_scan": lambda: right.show_tab_by_key("scan"),
            "fit_view": self.window.canvas.fit_to_window,
            "zoom_100": self.window.canvas.zoom_to_original,
            "zoom_200": lambda: self.window.canvas.zoom_to_percent(200.0),
            "export": controller.request_export,
            "export_linear_output": controller.request_linear_output_export,
            "copy": controller.session.copy_settings,
            "copy_with_bounds": controller.session.copy_settings_with_bounds,
            "paste": lambda: open_paste_dialog(self.window, controller),
            "sync_bounds": lambda: open_sync_bounds_dialog(self.window, controller.session),
            "reset_roll": lambda: _reset_roll(self.window, controller),
            "reset_tab": lambda: _fire_tab_header(right, "reset"),
            "reset_tab_to_roll": lambda: _fire_tab_header(right, "revert"),
            "reset_to_roll": controller.revert_frame_to_roll,
            "apply_tab": lambda: _fire_tab_header(right, "apply"),
            "toggle_tab_cards": lambda: _fire_tab_header(right, "cards"),
            "roll_batch_analysis": controller.request_batch_normalization,
            "analyze_all_scenes": controller.request_analyze_all_scenes,
            "roll_settings": lambda: self.window.session_panel.file_browser.roll_settings_btn.click(),
            "save_as_roll": lambda: self.window.session_panel.file_browser.save_roll_btn.click(),
            "import_roll": lambda: self.window.session_panel.library_tree.prompt_import_folder(),
            "index_library": lambda: self.window.session_panel.library_tree.index_btn.click(),
            "metadata_infer_gear": lambda: right.metadata_sidebar.gear_infer_btn.click(),
            "toggle_gear_catalog": lambda: right.gear_panel.show_catalog_btn.click(),
            "toggle_semantic_search": lambda: self.window.session_panel.file_browser.semantic_btn.click(),
            "toggle_positive_source": lambda: controls.process_sidebar.positive_source_btn.click(),
            "persistent_settings": lambda: open_sticky_dialog(self.window, controller),
            "open_preferences": lambda: _open_preferences(self.window, controller),
            "save_work_print": self.window.right_panel.history_panel.save_work_print,
            "undo": lambda: _context_undo(controller),
            "redo": controller.session.redo,
            "show_shortcuts": lambda: _show_shortcuts(self.window),
            "show_analysis_help": self.window.right_panel.show_analysis_help,
            "check_for_updates": lambda: self.window.session_panel.check_for_updates(),
            # Button clicks, so the shortcut runs the same gating and toast the mouse gets.
            "toggle_hq": toolbar.btn_hq.click,
            "toggle_optical_removal": controls.retouch_sidebar.auto_dust_btn.click,
            "toggle_right_click_excludes": controls.retouch_sidebar.right_click_btn.click,
            "toggle_ir_removal": controls.retouch_sidebar.ir_dust_btn.click,
            "toggle_flat_field": controls.flatfield_sidebar.enable_btn.click,
            "batch_autocrop": controls.autocrop_sidebar.auto_crop_all_btn.click,
            "toggle_auto_density": controls.tone_sidebar.auto_density_btn.click,
            "toggle_auto_grade": controls.tone_sidebar.auto_grade_btn.click,
            "preset_apply": controls.presets_sidebar.apply_btn.click,
            "preset_save": controls.presets_sidebar.save_btn.click,
        }

        widgets = slider_widget_map(controls)
        for group in SLIDER_GROUPS:
            getter = widgets[group.id]
            actions[group.inc_action] = self._slider_adjuster(getter, group.inc_action)
            actions[group.dec_action] = self._slider_adjuster(getter, group.dec_action)
        return actions

    def action_for(self, action_id: str) -> Optional[Callable[[], None]]:
        """The handler a shortcut runs. The macOS menu bar dispatches through this, so a
        menu item and its key can never drift apart."""
        return self._actions.get(action_id)

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
        self.window.toolbar.apply_shortcut_tooltips()
        # The macOS menu bar carries key equivalents of its own; a rebind has to reach them
        # or the retired key keeps working from the menu.
        menus = getattr(self.window, "mac_menus", None)
        if menus is not None:
            menus.sync_shortcuts()

    def update_bindings(self, bindings: dict[str, str]) -> None:
        save_bindings(self.window.controller.session.repo, bindings)
        self.apply_bindings(bindings)

    def update_slider_steps(self, steps: dict[str, float]) -> None:
        save_slider_steps(self.window.controller.session.repo, steps)
        self.slider_steps = dict(steps)

    def open_editor(self, parent=None) -> bool:
        from negpy.desktop.view.widgets.shortcut_editor import ShortcutEditorDialog

        dlg = ShortcutEditorDialog(self.bindings, self.slider_steps, parent or self.window)
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
