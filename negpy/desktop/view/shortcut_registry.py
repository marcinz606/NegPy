from negpy.kernel.system.i18n import tr
import re
from dataclasses import dataclass
from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence

from negpy.desktop.view.slider_shortcut_groups import (
    SLIDER_GROUPS,
    SLIDER_GROUP_BY_ACTION,
    SLIDER_GROUP_BY_ID,
    SliderShortcutGroup,
)
from negpy.desktop.view.styles.theme import THEME


@dataclass(frozen=True)
class ShortcutEntry:
    default_key: str
    description: str
    category: str
    # The window that dispatches the key; keys clash only within one window.
    window: str = "main"


LIVE_VIEW = "live_view"


REGISTRY: dict[str, ShortcutEntry] = {
    "prev_file": ShortcutEntry("Left", tr("Previous file"), tr("Navigation")),
    "next_file": ShortcutEntry("Right", tr("Next file"), tr("Navigation")),
    "first_file": ShortcutEntry("Home", tr("First file"), tr("Navigation")),
    "last_file": ShortcutEntry("End", tr("Last file"), tr("Navigation")),
    "toggle_keep": ShortcutEntry("K", tr("Mark frame as keeper"), tr("Triage")),
    "toggle_reject": ShortcutEntry("Shift+X", tr("Reject frame (skipped by batch export)"), tr("Triage")),
    "toggle_advance_after_mark": ShortcutEntry("", tr("Advance to the next frame after marking"), tr("Triage")),
    # No default key: nothing obvious is free, and an invented binding that collides makes Qt
    # fire activatedAmbiguously and kills both actions.
    "hdr_merge": ShortcutEntry("", tr("Merge selected exposures into one HDR frame"), tr("Triage")),
    "hdr_unmerge": ShortcutEntry("", tr("Unmerge an HDR frame back into its exposures"), tr("Triage")),
    "half_frame_undiptych": ShortcutEntry("", tr("Unsplit a diptych back into one plain frame"), tr("Triage")),
    "update_thumbnails_selection": ShortcutEntry("", tr("Update selected frames' thumbnails"), tr("Triage")),
    "update_thumbnails_roll": ShortcutEntry("", tr("Update thumbnails for the whole roll"), tr("Triage")),
    "toggle_compare": ShortcutEntry(
        "\\", tr("Before/after split against the default conversion; hold to see the default alone"), tr("Tools")
    ),
    "rotate_cw": ShortcutEntry("]", tr("Rotate 90° CW"), tr("Geometry")),
    "rotate_ccw": ShortcutEntry("[", tr("Rotate 90° CCW"), tr("Geometry")),
    "flip_h": ShortcutEntry("H", tr("Flip horizontal"), tr("Geometry")),
    "flip_v": ShortcutEntry("V", tr("Flip vertical"), tr("Geometry")),
    "offset_dec": ShortcutEntry("Z", tr("Crop offset down"), tr("Geometry")),
    "offset_inc": ShortcutEntry("X", tr("Crop offset up"), tr("Geometry")),
    "fine_rot_dec": ShortcutEntry("Alt+Shift+R", tr("Fine rotation counter-clockwise"), tr("Geometry")),
    "fine_rot_inc": ShortcutEntry("Alt+R", tr("Fine rotation clockwise"), tr("Geometry")),
    "converge_v_dec": ShortcutEntry("", tr("Easel tilt down"), tr("Geometry")),
    "converge_v_inc": ShortcutEntry("", tr("Easel tilt up"), tr("Geometry")),
    "converge_h_dec": ShortcutEntry("", tr("Easel swing down"), tr("Geometry")),
    "converge_h_inc": ShortcutEntry("", tr("Easel swing up"), tr("Geometry")),
    "auto_skew": ShortcutEntry("", tr("Square to Film: square the frame to its edges"), tr("Geometry")),
    "straighten": ShortcutEntry("L", tr("Toggle the Level tool"), tr("Geometry")),
    "keystone_lines": ShortcutEntry("", tr("Toggle tilt/swing reference-lines tool"), tr("Geometry")),
    "pick_wb": ShortcutEntry("Shift+W", tr("Toggle WB picker"), tr("Tools")),
    "manual_crop": ShortcutEntry("Shift+C", tr("Toggle manual crop"), tr("Tools")),
    "crop_guide_next": ShortcutEntry("O", tr("Next crop guide overlay"), tr("Geometry")),
    "crop_guide_orient": ShortcutEntry("Shift+O", tr("Rotate crop guide orientation"), tr("Geometry")),
    "lens_distortion_from_metadata": ShortcutEntry("", tr("Embedded Distortion"), tr("Geometry")),
    "lens_ca_from_metadata": ShortcutEntry("", tr("Embedded CA"), tr("Geometry")),
    "auto_crop": ShortcutEntry("Shift+A", tr("Toggle Auto Crop"), tr("Geometry")),
    "auto_crop_square": ShortcutEntry("", tr("Toggle Square First"), tr("Geometry")),
    "crop_to_valid": ShortcutEntry("", tr("Toggle Crop Empty Corners"), tr("Geometry")),
    "pick_dust": ShortcutEntry("Shift+D", tr("Toggle heal tool"), tr("Tools")),
    "pick_scratch": ShortcutEntry("Shift+S", tr("Toggle scratch tool"), tr("Tools")),
    "pick_scratch_line": ShortcutEntry("Shift+K", tr("Toggle transport-scratch line tool"), tr("Tools")),
    "clone_tool": ShortcutEntry("Shift+J", tr("Toggle clone tool"), tr("Tools")),
    "local_draw": ShortcutEntry("Shift+B", tr("Toggle dodge & burn mask draw"), tr("Tools")),
    "local_oval": ShortcutEntry("", tr("Toggle dodge & burn oval mask draw"), tr("Tools")),
    "local_gradient": ShortcutEntry("", tr("Toggle dodge & burn card-edge mask draw"), tr("Tools")),
    "analysis_draw": ShortcutEntry("Ctrl+R", tr("Toggle analysis region draw"), tr("Tools")),
    "film_base": ShortcutEntry("", tr("Toggle film base pick"), tr("Tools")),
    "open_tools_menu": ShortcutEntry("", tr("Open the Tools menu: every canvas tool"), tr("Tools")),
    "toggle_flat_peek": ShortcutEntry("M", tr("Peek flat scan (digital intermediate); hold to peek only while down"), tr("Tools")),
    "toggle_negative_peek": ShortcutEntry("N", tr("Peek negative (source as loaded); hold to peek only while down"), tr("Tools")),
    "toggle_embedded_peek": ShortcutEntry("P", tr("Peek the camera's embedded preview; hold to peek only while down"), tr("Tools")),
    "toggle_zones": ShortcutEntry("Shift+Z", tr("Adams zone overlay"), tr("Tools")),
    "toggle_clipping_overlay": ShortcutEntry("", tr("Clipping Overlay: paper black and paper white on the frame"), tr("Tools")),
    "toggle_zone_hover_highlight": ShortcutEntry(
        "", tr("Zone Highlight: show the zone strip's hovered zone alone on the frame"), tr("Tools")
    ),
    "toggle_parade_overlay": ShortcutEntry("", tr("Parade Overlay: R, G and B on one plot, white where they agree"), tr("Tools")),
    "toggle_scan_parade_base": ShortcutEntry(
        "", tr("Base Relative: the scan Parade's density above each channel's film base"), tr("Tools")
    ),
    "toggle_hq": ShortcutEntry("", tr("Toggle high-quality (full-resolution) preview"), tr("View")),
    "toggle_optical_removal": ShortcutEntry("", tr("Toggle Optical Removal"), tr("Retouch")),
    "toggle_right_click_excludes": ShortcutEntry("", tr("Toggle right-click excludes from Optical Removal"), tr("Retouch")),
    "toggle_ir_removal": ShortcutEntry("", tr("Toggle IR Removal"), tr("Retouch")),
    "toggle_light_leak": ShortcutEntry("", tr("Toggle Reduce Light Leak"), tr("Light Leak")),
    "tool_leak_band": ShortcutEntry("", tr("Toggle light leak band draw"), tr("Light Leak")),
    "tool_leak_edge": ShortcutEntry("", tr("Toggle light leak edge draw"), tr("Light Leak")),
    "tool_leak_spot": ShortcutEntry("", tr("Toggle light leak spot draw"), tr("Light Leak")),
    "tool_leak_paint": ShortcutEntry("", tr("Toggle light leak paint brush"), tr("Light Leak")),
    "leak_auto": ShortcutEntry("", tr("Find light leaks and place shapes"), tr("Light Leak")),
    "leak_from_roll": ShortcutEntry("", tr("Carry the light leak to the selected frames"), tr("Light Leak")),
    "toggle_leak_burned": ShortcutEntry("", tr("Show the film a light leak burned out"), tr("Light Leak")),
    "toggle_flat_field": ShortcutEntry("", tr("Toggle Apply Flat Field"), tr("Geometry")),
    "toggle_flatfield_peek": ShortcutEntry("", tr("Check Flat Field (how well the profile corrects its reference)"), tr("Geometry")),
    "batch_autocrop": ShortcutEntry("", tr("Auto-crop all frames"), tr("Geometry")),
    "toggle_auto_density": ShortcutEntry("", tr("Toggle Auto Density"), tr("Exposure")),
    "toggle_auto_grade": ShortcutEntry("", tr("Toggle Auto Grade"), tr("Exposure")),
    "toggle_auto_both": ShortcutEntry("", tr("Toggle Auto Density and Auto Grade together"), tr("Exposure")),
    "toggle_hold_black": ShortcutEntry("", tr("Toggle Hold Black (a shadow lift keeps the black point)"), tr("Exposure")),
    "preset_apply": ShortcutEntry("", tr("Apply the selected preset"), tr("Actions")),
    "preset_save": ShortcutEntry("", tr("Save a preset from the current settings"), tr("Actions")),
    "toggle_test_strip": ShortcutEntry("Shift+T", tr("Density × grade test strip"), tr("Tools")),
    "toggle_paper_strip": ShortcutEntry("", tr("Paper test strip: the frame on every print stock"), tr("Tools")),
    "toggle_pick_better": ShortcutEntry("", tr("Pick Better: pick the better of two images"), tr("Exposure")),
    "finish_pick_better": ShortcutEntry("", tr("Finish Pick Better"), tr("Exposure")),
    "undo_pick_better": ShortcutEntry("", tr("Undo Pick Better"), tr("Exposure")),
    "toggle_ring_around": ShortcutEntry("Shift+F", tr("Color ring-around (M/Y filtration)"), tr("Tools")),
    "toggle_grain_focuser": ShortcutEntry("Shift+L", tr("Grain focuser loupe"), tr("Tools")),
    "toggle_printing_notes": ShortcutEntry("Shift+N", tr("Printing notes (dodge/burn map + print recipe)"), tr("Tools")),
    "toggle_cursor_probe": ShortcutEntry("Shift+I", tr("Probe at cursor (zone, print density and ΔD beside the pointer)"), tr("Tools")),
    "toggle_soft_proof": ShortcutEntry("Shift+P", tr("Soft proof the print on screen"), tr("Tools")),
    "cancel_tool": ShortcutEntry(
        "Esc", tr("Leave the current view (peek, split, strip, zone overlay) or cancel the active tool"), tr("Tools")
    ),
    "cyan_dec": ShortcutEntry("", tr("Cyan down"), tr("Exposure")),
    "cyan_inc": ShortcutEntry("", tr("Cyan up"), tr("Exposure")),
    "magenta_down": ShortcutEntry("D", tr("Magenta down"), tr("Exposure")),
    "magenta_up": ShortcutEntry("E", tr("Magenta up"), tr("Exposure")),
    "yellow_down": ShortcutEntry("F", tr("Yellow down"), tr("Exposure")),
    "yellow_up": ShortcutEntry("R", tr("Yellow up"), tr("Exposure")),
    "temp_warm": ShortcutEntry("T", tr("Temperature warmer"), tr("Exposure")),
    "temp_cool": ShortcutEntry("G", tr("Temperature cooler"), tr("Exposure")),
    "density_down": ShortcutEntry("A", tr("Density down"), tr("Exposure")),
    "density_up": ShortcutEntry("Q", tr("Density up"), tr("Exposure")),
    "grade_down": ShortcutEntry("S", tr("Grade down"), tr("Exposure")),
    "grade_up": ShortcutEntry("W", tr("Grade up"), tr("Exposure")),
    "toe_dec": ShortcutEntry("Alt+Shift+T", tr("Toe down"), tr("Exposure")),
    "toe_inc": ShortcutEntry("Alt+T", tr("Toe up"), tr("Exposure")),
    "toe_width_dec": ShortcutEntry("Alt+Shift+Y", tr("Toe width down"), tr("Exposure")),
    "toe_width_inc": ShortcutEntry("Alt+Y", tr("Toe width up"), tr("Exposure")),
    "shoulder_dec": ShortcutEntry("Alt+Shift+U", tr("Shoulder down"), tr("Exposure")),
    "shoulder_inc": ShortcutEntry("Alt+U", tr("Shoulder up"), tr("Exposure")),
    "shoulder_width_dec": ShortcutEntry("Alt+Shift+I", tr("Shoulder width down"), tr("Exposure")),
    "shoulder_width_inc": ShortcutEntry("Alt+I", tr("Shoulder width up"), tr("Exposure")),
    "snap_dec": ShortcutEntry("", tr("Snap (midtone) down"), tr("Exposure")),
    "snap_inc": ShortcutEntry("", tr("Snap (midtone) up"), tr("Exposure")),
    "shadow_density_dec": ShortcutEntry("", tr("Shadows density down"), tr("Exposure")),
    "shadow_density_inc": ShortcutEntry("", tr("Shadows density up"), tr("Exposure")),
    "highlight_density_dec": ShortcutEntry("", tr("Highlights density down"), tr("Exposure")),
    "highlight_density_inc": ShortcutEntry("", tr("Highlights density up"), tr("Exposure")),
    "shadow_grade_dec": ShortcutEntry("", tr("Shadows grade down"), tr("Exposure")),
    "shadow_grade_inc": ShortcutEntry("", tr("Shadows grade up"), tr("Exposure")),
    "highlight_grade_dec": ShortcutEntry("", tr("Highlights grade down"), tr("Exposure")),
    "highlight_grade_inc": ShortcutEntry("", tr("Highlights grade up"), tr("Exposure")),
    "dye_separation_dec": ShortcutEntry("", tr("Dye Separation down"), tr("Exposure")),
    "dye_separation_inc": ShortcutEntry("", tr("Dye Separation up"), tr("Exposure")),
    "separation_damping_dec": ShortcutEntry("", tr("Separation Damping down"), tr("Exposure")),
    "separation_damping_inc": ShortcutEntry("", tr("Separation Damping up"), tr("Exposure")),
    "contrast_mask_dec": ShortcutEntry("", tr("Contrast Mask down"), tr("Exposure")),
    "contrast_mask_inc": ShortcutEntry("", tr("Contrast Mask up"), tr("Exposure")),
    "mask_spacer_dec": ShortcutEntry("", tr("Mask Spacer down"), tr("Exposure")),
    "mask_spacer_inc": ShortcutEntry("", tr("Mask Spacer up"), tr("Exposure")),
    "diffusion_dec": ShortcutEntry("", tr("Diffusion down"), tr("Exposure")),
    "diffusion_inc": ShortcutEntry("", tr("Diffusion up"), tr("Exposure")),
    "preflash_dec": ShortcutEntry("", tr("Preflash down"), tr("Exposure")),
    "preflash_inc": ShortcutEntry("", tr("Preflash up"), tr("Exposure")),
    "lock_bounds_toggle": ShortcutEntry("Alt+Q", tr("Toggle Lock Bounds"), tr("Process")),
    "reanalyze_frame": ShortcutEntry("", tr("Reanalyze this frame's bounds"), tr("Process")),
    "cast_average_toggle": ShortcutEntry("", tr("Toggle cast average"), tr("Process")),
    "metadata_preset_load": ShortcutEntry("", tr("Apply the selected metadata preset to this frame"), tr("Actions")),
    "metadata_clear_gear": ShortcutEntry("", tr("Clear the frame's camera, lens and film stock"), tr("Actions")),
    "metadata_clear_process": ShortcutEntry("", tr("Clear the frame's development process"), tr("Actions")),
    "metadata_clear_scanning": ShortcutEntry("", tr("Clear the frame's scan setup"), tr("Actions")),
    "assign_sequential_frame": ShortcutEntry("", tr("Assign sequential frame numbers"), tr("Actions")),
    "measure_chart": ShortcutEntry("", tr("Measure a crosstalk matrix from a color chart"), tr("Process")),
    "scan_prescan": ShortcutEntry("", tr("Prescan and set crop (Plustek)"), tr("Process")),
    "scan_meter_frame": ShortcutEntry("", tr("Meter a frame and lock the scan exposure (Coolscan)"), tr("Process")),
    "scan_as_roll": ShortcutEntry("", tr("Toggle Scan as Roll"), tr("Process")),
    "scan_folder_as_roll": ShortcutEntry("", tr("Toggle Folder as Roll"), tr("Process")),
    "scan_new_roll": ShortcutEntry("", tr("Start the next scan roll"), tr("Process")),
    "scanlight_light": ShortcutEntry("", tr("Turn the Scanlight on or off"), tr("Process")),
    "scanlight_keep_light_on": ShortcutEntry("", tr("Toggle Keep Light On (Scanlight)"), tr("Process")),
    "scanlight_infrared": ShortcutEntry("", tr("Toggle Infrared: an infrared exposure after every frame (Scanlight)"), tr("Process")),
    "mode_color_negative": ShortcutEntry("", tr("Mode: Color Negative"), tr("Process")),
    "mode_bw_negative": ShortcutEntry("", tr("Mode: B&W Negative"), tr("Process")),
    "mode_transparency": ShortcutEntry("", tr("Mode: Transparency"), tr("Process")),
    "analysis_buffer_dec": ShortcutEntry("Alt+Shift+B", tr("Analysis buffer down"), tr("Process")),
    "analysis_buffer_inc": ShortcutEntry("Alt+B", tr("Analysis buffer up"), tr("Process")),
    "toggle_positive_source": ShortcutEntry("", tr("Toggle Positive (source is already a finished positive)"), tr("Process")),
    "luma_range_clip_dec": ShortcutEntry("Alt+Shift+N", tr("Luma range clip down"), tr("Process")),
    "luma_range_clip_inc": ShortcutEntry("Alt+N", tr("Luma range clip up"), tr("Process")),
    "color_range_clip_dec": ShortcutEntry("Alt+Shift+E", tr("Color range clip down"), tr("Process")),
    "color_range_clip_inc": ShortcutEntry("Alt+E", tr("Color range clip up"), tr("Process")),
    "white_point_dec": ShortcutEntry("Alt+Shift+P", tr("White point down"), tr("Exposure")),
    "white_point_inc": ShortcutEntry("Alt+P", tr("White point up"), tr("Exposure")),
    "black_point_dec": ShortcutEntry("Alt+Shift+O", tr("Black point down"), tr("Exposure")),
    "black_point_inc": ShortcutEntry("Alt+O", tr("Black point up"), tr("Exposure")),
    "render_ev_dec": ShortcutEntry("", tr("Render exposure down"), tr("Process")),
    "render_ev_inc": ShortcutEntry("", tr("Render exposure up"), tr("Process")),
    "separation_dec": ShortcutEntry("Alt+Shift+1", tr("Crosstalk down"), tr("Process")),
    "separation_inc": ShortcutEntry("Alt+1", tr("Crosstalk up"), tr("Process")),
    "chroma_denoise_dec": ShortcutEntry("Alt+Shift+2", tr("Denoise down"), tr("Lab")),
    "chroma_denoise_inc": ShortcutEntry("Alt+2", tr("Denoise up"), tr("Lab")),
    "saturation_dec": ShortcutEntry("Alt+Shift+3", tr("Chroma down"), tr("Lab")),
    "saturation_inc": ShortcutEntry("Alt+3", tr("Chroma up"), tr("Lab")),
    "clahe_dec": ShortcutEntry("Alt+Shift+5", tr("CLAHE down"), tr("Lab")),
    "clahe_inc": ShortcutEntry("Alt+5", tr("CLAHE up"), tr("Lab")),
    "sharpen_dec": ShortcutEntry("Alt+Shift+6", tr("Sharpening down"), tr("Lab")),
    "sharpen_inc": ShortcutEntry("Alt+6", tr("Sharpening up"), tr("Lab")),
    "glow_dec": ShortcutEntry("Alt+Shift+7", tr("Glow down"), tr("Lab")),
    "glow_inc": ShortcutEntry("Alt+7", tr("Glow up"), tr("Lab")),
    "halation_dec": ShortcutEntry("Alt+Shift+8", tr("Halation down"), tr("Lab")),
    "halation_inc": ShortcutEntry("Alt+8", tr("Halation up"), tr("Lab")),
    "threshold_dec": ShortcutEntry("Alt+Shift+9", tr("Spot threshold down"), tr("Retouch")),
    "threshold_inc": ShortcutEntry("Alt+9", tr("Spot threshold up"), tr("Retouch")),
    "hair_threshold_dec": ShortcutEntry("", tr("Hair threshold down"), tr("Retouch")),
    "hair_threshold_inc": ShortcutEntry("", tr("Hair threshold up"), tr("Retouch")),
    "auto_size_dec": ShortcutEntry("Alt+Shift+0", tr("Auto size down"), tr("Retouch")),
    "auto_size_inc": ShortcutEntry("Alt+0", tr("Auto size up"), tr("Retouch")),
    "manual_size_dec": ShortcutEntry("Alt+Shift+M", tr("Brush size down"), tr("Retouch")),
    "manual_size_inc": ShortcutEntry("Alt+M", tr("Brush size up"), tr("Retouch")),
    "toggle_lith": ShortcutEntry("", tr("Lith printing on/off"), tr("Alt Process")),
    "lith_snatch_dec": ShortcutEntry("", tr("Snatch point earlier"), tr("Alt Process")),
    "lith_snatch_inc": ShortcutEntry("", tr("Snatch point later"), tr("Alt Process")),
    "lith_exposure_dec": ShortcutEntry("", tr("Lith exposure down"), tr("Alt Process")),
    "lith_exposure_inc": ShortcutEntry("", tr("Lith exposure up"), tr("Alt Process")),
    "lith_abruptness_dec": ShortcutEntry("", tr("Lith abruptness down"), tr("Alt Process")),
    "lith_abruptness_inc": ShortcutEntry("", tr("Lith abruptness up"), tr("Alt Process")),
    "toggle_cyanotype": ShortcutEntry("", tr("Cyanotype printing on/off"), tr("Alt Process")),
    "cyano_exposure_dec": ShortcutEntry("", tr("Cyanotype exposure down"), tr("Alt Process")),
    "cyano_exposure_inc": ShortcutEntry("", tr("Cyanotype exposure up"), tr("Alt Process")),
    "cyano_scale_dec": ShortcutEntry("", tr("Exposure scale shorter"), tr("Alt Process")),
    "cyano_scale_inc": ShortcutEntry("", tr("Exposure scale longer"), tr("Alt Process")),
    "cyano_bleach_dec": ShortcutEntry("", tr("Bleach down"), tr("Alt Process")),
    "cyano_bleach_inc": ShortcutEntry("", tr("Bleach up"), tr("Alt Process")),
    "cyano_tannin_dec": ShortcutEntry("", tr("Tannin down"), tr("Alt Process")),
    "cyano_tannin_inc": ShortcutEntry("", tr("Tannin up"), tr("Alt Process")),
    "toggle_sabattier": ShortcutEntry("", tr("Sabattier printing on/off"), tr("Alt Process")),
    "sabattier_strength_dec": ShortcutEntry("", tr("Sabattier strength down"), tr("Alt Process")),
    "sabattier_strength_inc": ShortcutEntry("", tr("Sabattier strength up"), tr("Alt Process")),
    "sabattier_reexposure_dec": ShortcutEntry("", tr("Re-exposure lower"), tr("Alt Process")),
    "sabattier_reexposure_inc": ShortcutEntry("", tr("Re-exposure higher"), tr("Alt Process")),
    "sabattier_agitation_dec": ShortcutEntry("", tr("Agitation down"), tr("Alt Process")),
    "sabattier_agitation_inc": ShortcutEntry("", tr("Agitation up"), tr("Alt Process")),
    "selenium_dec": ShortcutEntry("Alt+Shift+J", tr("Selenium down"), tr("Toning")),
    "selenium_inc": ShortcutEntry("Alt+J", tr("Selenium up"), tr("Toning")),
    "sepia_dec": ShortcutEntry("Alt+Shift+K", tr("Sepia down"), tr("Toning")),
    "sepia_inc": ShortcutEntry("Alt+K", tr("Sepia up"), tr("Toning")),
    "shadow_hue_dec": ShortcutEntry("Alt+Shift+H", tr("Shadow hue down"), tr("Toning")),
    "shadow_hue_inc": ShortcutEntry("Alt+H", tr("Shadow hue up"), tr("Toning")),
    "shadow_strength_dec": ShortcutEntry("Alt+Shift+G", tr("Shadow strength down"), tr("Toning")),
    "shadow_strength_inc": ShortcutEntry("Alt+G", tr("Shadow strength up"), tr("Toning")),
    "highlight_hue_dec": ShortcutEntry("Alt+Shift+L", tr("Highlight hue down"), tr("Toning")),
    "highlight_hue_inc": ShortcutEntry("Alt+L", tr("Highlight hue up"), tr("Toning")),
    "highlight_strength_dec": ShortcutEntry("Alt+Shift+;", tr("Highlight strength down"), tr("Toning")),
    "highlight_strength_inc": ShortcutEntry("Alt+;", tr("Highlight strength up"), tr("Toning")),
    "vignette_str_dec": ShortcutEntry("Alt+Shift+V", tr("Vignette burn down"), tr("Finishing")),
    "vignette_str_inc": ShortcutEntry("Alt+V", tr("Vignette burn up"), tr("Finishing")),
    "vignette_size_dec": ShortcutEntry("Alt+Shift+S", tr("Vignette size down"), tr("Finishing")),
    "vignette_size_inc": ShortcutEntry("Alt+S", tr("Vignette size up"), tr("Finishing")),
    "border_size_dec": ShortcutEntry("Alt+Shift+D", tr("Border width down"), tr("Finishing")),
    "border_size_inc": ShortcutEntry("Alt+D", tr("Border width up"), tr("Finishing")),
    "show_library": ShortcutEntry("Ctrl+L", tr("Open the Library tab"), tr("Navigation")),
    "command_palette": ShortcutEntry("Ctrl+K", tr("Find a control, card or action"), tr("Navigation")),
    "focus_search": ShortcutEntry("Ctrl+F", tr("Focus the Film Strip search box"), tr("Navigation")),
    "search_library": ShortcutEntry("Ctrl+Shift+F", tr("Search the whole library and load the matches"), tr("Navigation")),
    "toggle_semantic_search": ShortcutEntry("", tr("Toggle search by meaning"), tr("Navigation")),
    "library_discovery_filters": ShortcutEntry("", tr("Edit the folder names library discovery skips"), tr("View")),
    "cycle_frame_info": ShortcutEntry("I", tr("Frame info on the canvas: off, basic or full"), tr("View")),
    "toggle_sticky_zoom": ShortcutEntry("", tr("Sticky zoom (keep zoom level when switching images)"), tr("View")),
    "toggle_sticky_settings": ShortcutEntry("", tr("Carry settings between frames (apply Persistent Settings to a new file)"), tr("View")),
    "toggle_invert_zoom_scroll": ShortcutEntry("", tr("Reverse scroll-to-zoom direction (scroll up zooms out)"), tr("View")),
    "toggle_shortcut_hints": ShortcutEntry("", tr("Shortcut hints (show a control's keys after a mouse edit)"), tr("View")),
    "toggle_film_strip": ShortcutEntry("Ctrl+[", tr("Collapse or expand the Film Strip"), tr("View")),
    "toggle_toolbar_mode": ShortcutEntry("", tr("Switch the toolbar between the photo's tools and the roll's"), tr("View")),
    "toggle_right_panel": ShortcutEntry("Ctrl+]", tr("Toggle controls panel (re-docks when floating)"), tr("View")),
    "reset_panel_layout": ShortcutEntry("Ctrl+Shift+L", tr("Dock the controls panel and expand the Film Strip"), tr("View")),
    "toggle_side_panels": ShortcutEntry(
        "Shift+H", tr("Hide the controls panel and collapse the Film Strip, or bring both back"), tr("View")
    ),
    "toggle_light_table": ShortcutEntry("Shift+G", tr("Light Table: the whole roll in rows, in place of the canvas"), tr("View")),
    "light_table_zoom_in": ShortcutEntry("+", tr("Light Table: larger frames, fewer to a row"), tr("View")),
    "light_table_zoom_out": ShortcutEntry("-", tr("Light Table: smaller frames, more to a row"), tr("View")),
    "toggle_reference": ShortcutEntry("Shift+R", tr("Reference view: pin this frame beside the canvas to match others to it"), tr("View")),
    "edit_toolbar": ShortcutEntry("", tr("Edit the toolbar (which controls it shows, in what order)"), tr("View")),
    "tab_roll": ShortcutEntry("Ctrl+1", tr("Roll tab"), tr("Tabs")),
    "tab_geometry": ShortcutEntry("Ctrl+2", tr("Geometry tab"), tr("Tabs")),
    "tab_tone": ShortcutEntry("Ctrl+3", tr("Exposure tab"), tr("Tabs")),
    "tab_color": ShortcutEntry("Ctrl+4", tr("Look tab"), tr("Tabs")),
    "tab_finish": ShortcutEntry("Ctrl+5", tr("Finish tab"), tr("Tabs")),
    "tab_history": ShortcutEntry("Ctrl+6", tr("History tab"), tr("Tabs")),
    "tab_export": ShortcutEntry("Ctrl+7", tr("Export tab"), tr("Tabs")),
    "tab_metadata": ShortcutEntry("Ctrl+8", tr("Metadata tab"), tr("Tabs")),
    "tab_gear_items": ShortcutEntry("", tr("Metadata: My Gear tab"), tr("Tabs")),
    "tab_gear_presets": ShortcutEntry("", tr("Metadata: Presets tab"), tr("Tabs")),
    "tab_scan": ShortcutEntry("Ctrl+9", tr("Scan tab"), tr("Tabs")),
    "tab_favourites": ShortcutEntry("Ctrl+0", tr("Favorites tab"), tr("Tabs")),
    "fit_view": ShortcutEntry("0", tr("Fit to window"), tr("View")),
    "zoom_100": ShortcutEntry("1", tr("Zoom 100%"), tr("View")),
    "zoom_200": ShortcutEntry("2", tr("Zoom 200%"), tr("View")),
    "export": ShortcutEntry("Ctrl+E", tr("Export"), tr("Actions")),
    "export_linear_output": ShortcutEntry("", tr("Export Linear Output"), tr("Actions")),
    "merge_frame": ShortcutEntry("", tr("Merge Frame to TIFF Negative…"), tr("Actions")),
    "merge_selected": ShortcutEntry("", tr("Merge Selected to TIFF Negative…"), tr("Actions")),
    "merge_roll": ShortcutEntry("", tr("Merge Roll to TIFF Negative…"), tr("Actions")),
    "contact_sheet": ShortcutEntry("", tr("Contact Sheet…"), tr("Actions")),
    "copy": ShortcutEntry("Ctrl+C", tr("Copy settings"), tr("Actions")),
    "copy_with_bounds": ShortcutEntry("Ctrl+Shift+C", tr("Copy settings (with bounds)"), tr("Actions")),
    "paste": ShortcutEntry("Ctrl+V", tr("Paste settings"), tr("Actions")),
    "sync_bounds": ShortcutEntry("", tr("Sync bounds to other frames…"), tr("Actions")),
    "reset_roll": ShortcutEntry("", tr("Reset roll to defaults"), tr("Actions")),
    "close_roll": ShortcutEntry("", tr("Close the roll, or unload every frame"), tr("Actions")),
    "remove_from_roll": ShortcutEntry("", tr("Remove the selected frames from the roll…"), tr("Actions")),
    "reset_tab": ShortcutEntry("", tr("Reset this tab to defaults"), tr("Actions")),
    "reset_tab_to_roll": ShortcutEntry("", tr("Reset this tab to the roll's settings"), tr("Actions")),
    "reset_to_roll": ShortcutEntry("", tr("Reset this frame to the roll's settings"), tr("Actions")),
    "undo_roll_push": ShortcutEntry("", tr("Undo Apply to Roll"), tr("Actions")),
    "load_sidecar": ShortcutEntry("", tr("Load this frame's edit from a sidecar…"), tr("Actions")),
    "show_in_folder": ShortcutEntry("", tr("Show the frame's file in its folder"), tr("Actions")),
    "move_to_trash": ShortcutEntry("", tr("Move the selected frames' files to the Trash…"), tr("Actions")),
    "apply_tab": ShortcutEntry("", tr("Apply this tab to the roll…"), tr("Actions")),
    "toggle_tab_cards": ShortcutEntry("", tr("Expand or collapse this tab's cards"), tr("Actions")),
    "roll_batch_analysis": ShortcutEntry("", tr("Roll Analysis (measure the roll's baseline)"), tr("Actions")),
    "analyze_all_scenes": ShortcutEntry("", tr("Analyze every scene of the roll"), tr("Actions")),
    "roll_settings": ShortcutEntry("", tr("Open Roll Settings"), tr("Actions")),
    "save_as_roll": ShortcutEntry("", tr("Save the loaded frames as a roll"), tr("Actions")),
    "import_roll": ShortcutEntry("", tr("Import a folder as a roll"), tr("Actions")),
    "index_library": ShortcutEntry("", tr("Index the library for search by meaning"), tr("Actions")),
    "metadata_infer_gear": ShortcutEntry("", tr("Infer gear from the folder name"), tr("Actions")),
    "toggle_gear_catalog": ShortcutEntry("", tr("Toggle the gear catalog"), tr("Actions")),
    "persistent_settings": ShortcutEntry("", tr("Choose which settings carry to the next file"), tr("Actions")),
    "open_preferences": ShortcutEntry("Ctrl+,", tr("Open Preferences"), tr("Actions")),
    "save_work_print": ShortcutEntry("Ctrl+Shift+S", tr("Save the current edit as a named work print"), tr("Actions")),
    "apply_work_print_metadata": ShortcutEntry("", tr("When a work print loads, restore its saved metadata"), tr("Actions")),
    "undo": ShortcutEntry("Ctrl+Z", tr("Undo"), tr("Actions")),
    "redo": ShortcutEntry("Ctrl+Y", tr("Redo"), tr("Actions")),
    "live_view_scan": ShortcutEntry("S", tr("Scan, or stop the capture"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_scan_space": ShortcutEntry("Space", tr("Scan, or stop the capture (second key)"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_retake": ShortcutEntry("R", tr("Retake the current frame"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_meter": ShortcutEntry("M", tr("Meter: set the shutter from one RAW probe"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_histogram": ShortcutEntry("H", tr("Show the live histogram"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_zebra": ShortcutEntry("Z", tr("Show clipping zebras"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_capture_red": ShortcutEntry("", tr("Toggle the red exposure for Retake"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_capture_green": ShortcutEntry("", tr("Toggle the green exposure for Retake"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_capture_blue": ShortcutEntry("", tr("Toggle the blue exposure for Retake"), tr("Camera Live View"), LIVE_VIEW),
    "live_view_capture_ir": ShortcutEntry("", tr("Toggle the infrared exposure for Scan and Retake"), tr("Camera Live View"), LIVE_VIEW),
    "show_shortcuts": ShortcutEntry("?", tr("Show shortcuts"), tr("Help")),
    "show_tour": ShortcutEntry("", tr("Take the tour: a guided walk through NegPy, one chapter at a time"), tr("Help")),
    "show_analysis_help": ShortcutEntry("", tr("Analysis panel guide"), tr("Help")),
    "check_for_updates": ShortcutEntry("", tr("Check for updates"), tr("Help")),
    "show_about": ShortcutEntry("", tr("About NegPy: version and project page"), tr("Help")),
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


_RENAMED_ACTIONS = {"toggle_left_panel": "toggle_film_strip"}


def merge_bindings(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The defaults with the user's keys over them. A default on a key the user gave another action
    is left unbound, as two shortcuts on one key both go dead."""
    bindings = default_bindings()
    chosen = {}
    for action_id, key in (overrides or {}).items():
        action_id = _RENAMED_ACTIONS.get(action_id, action_id)
        if action_id in REGISTRY:
            chosen[action_id] = str(key)
    bindings.update(chosen)
    taken = {(clash_scope(a, k), QKeySequence(k).toString()) for a, k in chosen.items() if k}
    for action_id, key in bindings.items():
        if action_id not in chosen and key and (clash_scope(action_id, key), QKeySequence(key).toString()) in taken:
            bindings[action_id] = ""
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


def clash_scope(action_id: str, key: str) -> str:
    """The window whose other bindings a key clashes with. A Ctrl or Cmd chord passes a
    floating panel's key guard to the main window, so it clashes there too."""
    seq = QKeySequence(key)
    command = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
    if not seq.isEmpty() and seq[0].keyboardModifiers() & command:
        return "main"
    return REGISTRY[action_id].window


def key_for(action_id: str, bindings: dict[str, str] | None = None) -> str:
    return (bindings or current_bindings()).get(action_id, "")


def display_key(key: str) -> str:
    """A binding written the way the platform writes it: ``⇧⌘C`` on macOS, ``Ctrl+Shift+C``
    elsewhere. Qt binds portable ``Ctrl`` to ⌘ on macOS, so every label that shows a key
    goes through here."""
    if not key:
        return ""
    return QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText) or key


def label_with_shortcut(text: str, action_id: str, bindings: dict[str, str] | None = None) -> str:
    """A menu label with its binding appended, or the plain text when the action has none."""
    key = display_key(key_for(action_id, bindings))
    return f"{text}  {key}" if key else text


def tooltip_with_shortcut(text: str, action_ids: str | Iterable[str] | None = None, bindings: dict[str, str] | None = None) -> str:
    if action_ids is None:
        return text
    if isinstance(action_ids, str):
        ids = [action_ids]
    else:
        ids = list(action_ids)
    bound = [action_id for action_id in ids if action_id in REGISTRY and key_for(action_id, bindings)]
    if not bound:
        return text
    row = keycap_cells([key_for(action_id, bindings) for action_id in bound], "&amp;")
    # The ids ride along in a comment, which rich text never renders: shortcut_hints reads them back.
    mark = _IDS_MARK.format(",".join(bound))
    return f'{text}<table align="right" cellspacing="0" cellpadding="0"><tr>{row}</tr></table>{mark}'


_IDS_MARK = "<!--shortcut:{}-->"
_IDS_RE = re.compile(r"<!--shortcut:([\w,]+)-->")


def shortcut_ids_in(tooltip: str) -> list[str]:
    """The action ids tooltip_with_shortcut put in a tooltip, or none."""
    found = _IDS_RE.search(tooltip)
    return found.group(1).split(",") if found else []


def keycap_cells(keys: Iterable[str], separator: str) -> str:
    """Table cells that read like physical keycaps. Qt's rich text draws `border` on table
    cells only, never on an inline <span>; the separator cell has no border of its own."""
    cells = [
        f'<td style="border:1px solid {THEME.border_indicator};background:{THEME.bg_selected};color:{THEME.text_primary};padding:1px 6px;font-size:{THEME.font_size_small}px;">{display_key(key)}</td>'
        for key in keys
    ]
    return f"<td>&nbsp;{separator}&nbsp;</td>".join(cells)
