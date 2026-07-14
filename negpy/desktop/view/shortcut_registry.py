from dataclasses import dataclass
from typing import Iterable

from negpy.desktop.view.slider_shortcut_groups import (
    SLIDER_GROUPS,
    SLIDER_GROUP_BY_ACTION,
    SLIDER_GROUP_BY_ID,
    SliderShortcutGroup,
)


@dataclass(frozen=True)
class ShortcutEntry:
    default_key: str
    description: str
    category: str


REGISTRY: dict[str, ShortcutEntry] = {
    "prev_file": ShortcutEntry("Left", "Previous file", "Navigation"),
    "next_file": ShortcutEntry("Right", "Next file", "Navigation"),
    "toggle_keep": ShortcutEntry("K", "Mark frame as keeper", "Triage"),
    "toggle_reject": ShortcutEntry("Shift+X", "Reject frame (skipped by batch export)", "Triage"),
    "toggle_compare": ShortcutEntry("\\", "Before/after (auto baseline)", "Tools"),
    "rotate_cw": ShortcutEntry("]", "Rotate 90° CW", "Geometry"),
    "rotate_ccw": ShortcutEntry("[", "Rotate 90° CCW", "Geometry"),
    "flip_h": ShortcutEntry("H", "Flip horizontal", "Geometry"),
    "flip_v": ShortcutEntry("V", "Flip vertical", "Geometry"),
    "offset_dec": ShortcutEntry("Z", "Crop offset down", "Geometry"),
    "offset_inc": ShortcutEntry("X", "Crop offset up", "Geometry"),
    "fine_rot_dec": ShortcutEntry("Alt+Shift+R", "Fine rotation counter-clockwise", "Geometry"),
    "fine_rot_inc": ShortcutEntry("Alt+R", "Fine rotation clockwise", "Geometry"),
    "straighten": ShortcutEntry("L", "Toggle straighten line tool", "Geometry"),
    "pick_wb": ShortcutEntry("Shift+W", "Toggle WB picker", "Tools"),
    "manual_crop": ShortcutEntry("Shift+C", "Toggle manual crop", "Tools"),
    "crop_guide_next": ShortcutEntry("O", "Next crop guide overlay", "Geometry"),
    "crop_guide_orient": ShortcutEntry("Shift+O", "Rotate crop guide orientation", "Geometry"),
    "auto_crop": ShortcutEntry("Shift+A", "Toggle autocrop", "Geometry"),
    "pick_dust": ShortcutEntry("Shift+D", "Toggle heal tool", "Tools"),
    "pick_scratch": ShortcutEntry("Shift+S", "Toggle scratch tool", "Tools"),
    "local_draw": ShortcutEntry("Shift+B", "Toggle dodge & burn mask draw", "Tools"),
    "analysis_draw": ShortcutEntry("Shift+R", "Toggle analysis region draw", "Tools"),
    "toggle_flat_peek": ShortcutEntry("|", "Peek flat scan (digital intermediate)", "Tools"),
    "cancel_tool": ShortcutEntry("Esc", "Cancel active tool (first press clears in-progress points)", "Tools"),
    "cyan_dec": ShortcutEntry("", "Cyan down", "Exposure"),
    "cyan_inc": ShortcutEntry("", "Cyan up", "Exposure"),
    "magenta_down": ShortcutEntry("D", "Magenta down", "Exposure"),
    "magenta_up": ShortcutEntry("E", "Magenta up", "Exposure"),
    "yellow_down": ShortcutEntry("F", "Yellow down", "Exposure"),
    "yellow_up": ShortcutEntry("R", "Yellow up", "Exposure"),
    "temp_warm": ShortcutEntry("T", "Temperature warmer", "Exposure"),
    "temp_cool": ShortcutEntry("G", "Temperature cooler", "Exposure"),
    "density_down": ShortcutEntry("A", "Density down", "Exposure"),
    "density_up": ShortcutEntry("Q", "Density up", "Exposure"),
    "grade_down": ShortcutEntry("S", "Grade down", "Exposure"),
    "grade_up": ShortcutEntry("W", "Grade up", "Exposure"),
    "toe_dec": ShortcutEntry("Alt+Shift+T", "Toe down", "Exposure"),
    "toe_inc": ShortcutEntry("Alt+T", "Toe up", "Exposure"),
    "toe_width_dec": ShortcutEntry("Alt+Shift+Y", "Toe width down", "Exposure"),
    "toe_width_inc": ShortcutEntry("Alt+Y", "Toe width up", "Exposure"),
    "shoulder_dec": ShortcutEntry("Alt+Shift+U", "Shoulder down", "Exposure"),
    "shoulder_inc": ShortcutEntry("Alt+U", "Shoulder up", "Exposure"),
    "shoulder_width_dec": ShortcutEntry("Alt+Shift+I", "Shoulder width down", "Exposure"),
    "shoulder_width_inc": ShortcutEntry("Alt+I", "Shoulder width up", "Exposure"),
    "snap_dec": ShortcutEntry("", "Snap (midtone) down", "Exposure"),
    "snap_inc": ShortcutEntry("", "Snap (midtone) up", "Exposure"),
    "shadow_density_dec": ShortcutEntry("", "Shadows density down", "Exposure"),
    "shadow_density_inc": ShortcutEntry("", "Shadows density up", "Exposure"),
    "highlight_density_dec": ShortcutEntry("", "Highlights density down", "Exposure"),
    "highlight_density_inc": ShortcutEntry("", "Highlights density up", "Exposure"),
    "shadow_grade_dec": ShortcutEntry("", "Shadows grade down", "Exposure"),
    "shadow_grade_inc": ShortcutEntry("", "Shadows grade up", "Exposure"),
    "highlight_grade_dec": ShortcutEntry("", "Highlights grade down", "Exposure"),
    "highlight_grade_inc": ShortcutEntry("", "Highlights grade up", "Exposure"),
    "lock_bounds_toggle": ShortcutEntry("Alt+Q", "Toggle bounds lock", "Process"),
    "analysis_buffer_dec": ShortcutEntry("Alt+Shift+B", "Analysis buffer down", "Process"),
    "analysis_buffer_inc": ShortcutEntry("Alt+B", "Analysis buffer up", "Process"),
    "luma_range_clip_dec": ShortcutEntry("Alt+Shift+N", "Luma range clip down", "Process"),
    "luma_range_clip_inc": ShortcutEntry("Alt+N", "Luma range clip up", "Process"),
    "color_range_clip_dec": ShortcutEntry("Alt+Shift+E", "Colour range clip down", "Process"),
    "color_range_clip_inc": ShortcutEntry("Alt+E", "Colour range clip up", "Process"),
    "white_point_dec": ShortcutEntry("Alt+Shift+P", "White point down", "Process"),
    "white_point_inc": ShortcutEntry("Alt+P", "White point up", "Process"),
    "black_point_dec": ShortcutEntry("Alt+Shift+O", "Black point down", "Process"),
    "black_point_inc": ShortcutEntry("Alt+O", "Black point up", "Process"),
    "separation_dec": ShortcutEntry("Alt+Shift+1", "Crosstalk down", "Process"),
    "separation_inc": ShortcutEntry("Alt+1", "Crosstalk up", "Process"),
    "chroma_denoise_dec": ShortcutEntry("Alt+Shift+2", "Denoise down", "Lab"),
    "chroma_denoise_inc": ShortcutEntry("Alt+2", "Denoise up", "Lab"),
    "saturation_dec": ShortcutEntry("Alt+Shift+3", "Saturation down", "Lab"),
    "saturation_inc": ShortcutEntry("Alt+3", "Saturation up", "Lab"),
    "vibrance_dec": ShortcutEntry("Alt+Shift+4", "Vibrance down", "Lab"),
    "vibrance_inc": ShortcutEntry("Alt+4", "Vibrance up", "Lab"),
    "clahe_dec": ShortcutEntry("Alt+Shift+5", "CLAHE down", "Lab"),
    "clahe_inc": ShortcutEntry("Alt+5", "CLAHE up", "Lab"),
    "sharpen_dec": ShortcutEntry("Alt+Shift+6", "Sharpening down", "Lab"),
    "sharpen_inc": ShortcutEntry("Alt+6", "Sharpening up", "Lab"),
    "glow_dec": ShortcutEntry("Alt+Shift+7", "Glow down", "Lab"),
    "glow_inc": ShortcutEntry("Alt+7", "Glow up", "Lab"),
    "halation_dec": ShortcutEntry("Alt+Shift+8", "Halation down", "Lab"),
    "halation_inc": ShortcutEntry("Alt+8", "Halation up", "Lab"),
    "threshold_dec": ShortcutEntry("Alt+Shift+9", "Threshold down", "Retouch"),
    "threshold_inc": ShortcutEntry("Alt+9", "Threshold up", "Retouch"),
    "auto_size_dec": ShortcutEntry("Alt+Shift+0", "Auto size down", "Retouch"),
    "auto_size_inc": ShortcutEntry("Alt+0", "Auto size up", "Retouch"),
    "manual_size_dec": ShortcutEntry("Alt+Shift+M", "Brush size down", "Retouch"),
    "manual_size_inc": ShortcutEntry("Alt+M", "Brush size up", "Retouch"),
    "selenium_dec": ShortcutEntry("Alt+Shift+J", "Selenium down", "Toning"),
    "selenium_inc": ShortcutEntry("Alt+J", "Selenium up", "Toning"),
    "sepia_dec": ShortcutEntry("Alt+Shift+K", "Sepia down", "Toning"),
    "sepia_inc": ShortcutEntry("Alt+K", "Sepia up", "Toning"),
    "shadow_hue_dec": ShortcutEntry("Alt+Shift+H", "Shadow hue down", "Toning"),
    "shadow_hue_inc": ShortcutEntry("Alt+H", "Shadow hue up", "Toning"),
    "shadow_strength_dec": ShortcutEntry("Alt+Shift+G", "Shadow strength down", "Toning"),
    "shadow_strength_inc": ShortcutEntry("Alt+G", "Shadow strength up", "Toning"),
    "highlight_hue_dec": ShortcutEntry("Alt+Shift+L", "Highlight hue down", "Toning"),
    "highlight_hue_inc": ShortcutEntry("Alt+L", "Highlight hue up", "Toning"),
    "highlight_strength_dec": ShortcutEntry("Alt+Shift+Semicolon", "Highlight strength down", "Toning"),
    "highlight_strength_inc": ShortcutEntry("Alt+Semicolon", "Highlight strength up", "Toning"),
    "vignette_str_dec": ShortcutEntry("Alt+Shift+V", "Vignette strength down", "Finishing"),
    "vignette_str_inc": ShortcutEntry("Alt+V", "Vignette strength up", "Finishing"),
    "vignette_size_dec": ShortcutEntry("Alt+Shift+S", "Vignette size down", "Finishing"),
    "vignette_size_inc": ShortcutEntry("Alt+S", "Vignette size up", "Finishing"),
    "border_size_dec": ShortcutEntry("Alt+Shift+D", "Border width down", "Finishing"),
    "border_size_inc": ShortcutEntry("Alt+D", "Border width up", "Finishing"),
    "toggle_left_panel": ShortcutEntry("Ctrl+[", "Toggle session panel", "View"),
    "toggle_right_panel": ShortcutEntry("Ctrl+]", "Toggle controls panel", "View"),
    "tab_setup": ShortcutEntry("Ctrl+1", "Setup tab", "Tabs"),
    "tab_geometry": ShortcutEntry("Ctrl+2", "Geometry tab", "Tabs"),
    "tab_tone": ShortcutEntry("Ctrl+3", "Tone tab", "Tabs"),
    "tab_color": ShortcutEntry("Ctrl+4", "Colour tab (Lab & Toning)", "Tabs"),
    "tab_finish": ShortcutEntry("Ctrl+5", "Finish tab", "Tabs"),
    "tab_history": ShortcutEntry("Ctrl+6", "History tab", "Tabs"),
    "tab_export": ShortcutEntry("Ctrl+7", "Export tab", "Tabs"),
    "tab_metadata": ShortcutEntry("Ctrl+8", "Metadata tab", "Tabs"),
    "tab_scan": ShortcutEntry("Ctrl+9", "Scan tab", "Tabs"),
    "fit_view": ShortcutEntry("0", "Fit to window", "View"),
    "zoom_100": ShortcutEntry("1", "Zoom 100%", "View"),
    "zoom_200": ShortcutEntry("2", "Zoom 200%", "View"),
    "export": ShortcutEntry("Ctrl+E", "Export", "Actions"),
    "copy": ShortcutEntry("Ctrl+C", "Copy settings", "Actions"),
    "copy_with_bounds": ShortcutEntry("Ctrl+Shift+C", "Copy settings (with bounds)", "Actions"),
    "paste": ShortcutEntry("Ctrl+V", "Paste settings", "Actions"),
    "undo": ShortcutEntry("Ctrl+Z", "Undo", "Actions"),
    "redo": ShortcutEntry("Ctrl+Y", "Redo", "Actions"),
    "show_shortcuts": ShortcutEntry("?", "Show shortcuts", "Help"),
}

_CURRENT_BINDINGS: dict[str, str] = {}
_CURRENT_SLIDER_STEPS: dict[str, float] = {}


def default_slider_steps() -> dict[str, float]:
    return {group.id: group.default_step for group in SLIDER_GROUPS}


def merge_slider_steps(overrides: dict[str, float] | None = None) -> dict[str, float]:
    steps = default_slider_steps()
    if overrides:
        for group_id, value in overrides.items():
            if group_id in SLIDER_GROUP_BY_ID:
                steps[group_id] = float(value)
    return steps


def load_slider_steps(repo) -> dict[str, float]:
    saved = repo.get_global_setting("shortcut_slider_steps", {}) or {}
    return merge_slider_steps(saved if isinstance(saved, dict) else {})


def save_slider_steps(repo, steps: dict[str, float]) -> None:
    defaults = default_slider_steps()
    overrides = {group_id: value for group_id, value in steps.items() if group_id in defaults and float(value) != defaults[group_id]}
    repo.save_global_setting("shortcut_slider_steps", overrides)


def set_current_slider_steps(steps: dict[str, float]) -> None:
    global _CURRENT_SLIDER_STEPS
    _CURRENT_SLIDER_STEPS = merge_slider_steps(steps)


def current_slider_steps() -> dict[str, float]:
    if not _CURRENT_SLIDER_STEPS:
        set_current_slider_steps(default_slider_steps())
    return dict(_CURRENT_SLIDER_STEPS)


def slider_step_for(group_id: str, steps: dict[str, float] | None = None) -> float:
    resolved = steps or current_slider_steps()
    return resolved.get(group_id, default_slider_steps()[group_id])


@dataclass(frozen=True)
class EditorRowSingle:
    action_id: str
    entry: ShortcutEntry


@dataclass(frozen=True)
class EditorRowSlider:
    group: SliderShortcutGroup


EditorRow = EditorRowSingle | EditorRowSlider


def categories_in_order() -> list[tuple[str, list[tuple[str, ShortcutEntry]]]]:
    ordered: list[tuple[str, list[tuple[str, ShortcutEntry]]]] = []
    index: dict[str, int] = {}
    for action_id, entry in REGISTRY.items():
        if entry.category not in index:
            index[entry.category] = len(ordered)
            ordered.append((entry.category, []))
        ordered[index[entry.category]][1].append((action_id, entry))
    return ordered


def category_editor_rows(items: list[tuple[str, ShortcutEntry]]) -> list[EditorRow]:
    """Merge paired slider shortcuts into one editor row, preserving registry order."""
    rows: list[EditorRow] = []
    seen_groups: set[str] = set()
    for action_id, entry in items:
        group = SLIDER_GROUP_BY_ACTION.get(action_id)
        if group is not None:
            if group.id in seen_groups:
                continue
            seen_groups.add(group.id)
            rows.append(EditorRowSlider(group))
            continue
        rows.append(EditorRowSingle(action_id, entry))
    return rows


def default_bindings() -> dict[str, str]:
    return {action_id: entry.default_key for action_id, entry in REGISTRY.items()}


def merge_bindings(overrides: dict[str, str] | None = None) -> dict[str, str]:
    bindings = default_bindings()
    if overrides:
        for action_id, key in overrides.items():
            if action_id in REGISTRY:
                bindings[action_id] = str(key)
    return bindings


def load_bindings(repo) -> dict[str, str]:
    saved = repo.get_global_setting("shortcut_bindings", {}) or {}
    return merge_bindings(saved if isinstance(saved, dict) else {})


def save_bindings(repo, bindings: dict[str, str]) -> None:
    defaults = default_bindings()
    overrides = {action_id: key for action_id, key in bindings.items() if action_id in defaults and key != defaults[action_id]}
    repo.save_global_setting("shortcut_bindings", overrides)


def set_current_bindings(bindings: dict[str, str]) -> None:
    global _CURRENT_BINDINGS
    _CURRENT_BINDINGS = merge_bindings(bindings)


def current_bindings() -> dict[str, str]:
    if not _CURRENT_BINDINGS:
        set_current_bindings(default_bindings())
    return dict(_CURRENT_BINDINGS)


def key_for(action_id: str, bindings: dict[str, str] | None = None) -> str:
    return (bindings or current_bindings()).get(action_id, "")


def tooltip_with_shortcut(text: str, action_ids: str | Iterable[str] | None = None, bindings: dict[str, str] | None = None) -> str:
    if action_ids is None:
        return text
    if isinstance(action_ids, str):
        ids = [action_ids]
    else:
        ids = list(action_ids)
    keys = [key_for(action_id, bindings) for action_id in ids if action_id in REGISTRY and key_for(action_id, bindings)]
    if not keys:
        return text
    # Each key is a bordered table cell so it reads like a physical keycap: a thin,
    # lighter-than-background border boxing the label. Qt's rich-text engine ignores
    # `border` on inline <span> elements (background/padding render, the outline does
    # not) but honours it on table cells, so the chips must be <td>s, not spans.
    cells = [
        f'<td style="border:1px solid #5A5A5A;background:#242424;color:#C8C8C8;padding:1px 6px;font-size:10px;">{key}</td>' for key in keys
    ]
    # The " & " separator sits in its own borderless cell so it doesn't inherit a
    # keycap outline. The whole row is right-aligned on its own line below the text.
    row = "<td>&nbsp;&amp;&nbsp;</td>".join(cells)
    return f'{text}<table align="right" cellspacing="0" cellpadding="0"><tr>{row}</tr></table>'
