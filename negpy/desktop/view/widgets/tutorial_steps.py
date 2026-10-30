from __future__ import annotations

from negpy.kernel.system.i18n import tr

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.shortcut_registry import display_key, key_for
from negpy.desktop.view.widgets.tutorial_overlay import Offer, TutorialStep

if TYPE_CHECKING:
    from negpy.desktop.view.main_window import MainWindow

BASICS = tr("Basics")
ROLL = tr("The Roll")
FRAMING = tr("Framing")
PRINTING = tr("Printing")
LOOK = tr("Look & Finish")
SEEING = tr("Seeing the Print")
OUTPUT = tr("Output & Records")
SCANNING = tr("Scanning")


def _k(action_id: str) -> str:
    """The action's current key, bold, so a rebind never leaves the tour teaching a dead key."""
    return f"<b>{display_key(key_for(action_id))}</b>"


def _load_demo(w: "MainWindow") -> None:
    from negpy.desktop.view.widgets.tutorial_demo import write_demo
    from negpy.kernel.system.config import BASE_USER_DIR

    w.controller.request_asset_discovery([str(write_demo(Path(BASE_USER_DIR) / "tour"))], auto_open=True)


_DEMO = Offer(tr("Load Demo Negative"), _load_demo, lambda w: not w.state.current_file_path)


def _marked(w: "MainWindow", mark: str) -> bool:
    files, i = w.state.uploaded_files, w.state.selected_file_idx
    return 0 <= i < len(files) and bool(files[i].get(mark))


def _palette_open(_w: "MainWindow") -> bool:
    from negpy.desktop.view.widgets.command_palette import CommandPalette

    return isinstance(QApplication.activePopupWidget(), CommandPalette)


def _put_down(w: "MainWindow") -> None:
    from negpy.desktop.session import ToolMode

    st = w.state
    esc = w.shortcut_manager.action_for("cancel_tool")
    for _ in range(12):
        busy = (
            st.active_tool != ToolMode.NONE,
            st.test_strip or st.test_strip_pending,
            st.negative_peek or st.embedded_peek or st.flat_peek or st.flatfield_peek,
            st.compare_mode,
            st.grain_focuser,
            st.zone_arm_target is not None,
            w.light_table_active(),
        )
        if esc is None or not any(busy):
            return
        esc()


def _resume_offer() -> Offer:
    def saved(w: "MainWindow") -> int:
        value = w.controller.session.repo.get_global_setting("tutorial_resume", 0)
        return value if isinstance(value, int) else 0

    return Offer(
        tr("Resume Where You Left Off"),
        lambda w: w.tutorial_overlay.goto(saved(w)),
        lambda w: w.tutorial_overlay.index == 0 and saved(w) > 0,
    )


def build(window: "MainWindow") -> list[TutorialStep]:
    """Return the ordered list of tutorial steps for *window*."""

    def canvas(w: "MainWindow"):
        return w.canvas

    def cp(w: "MainWindow"):
        return w.controls_panel

    def rp(w: "MainWindow"):
        return w.right_panel

    def fb(w: "MainWindow"):
        return w.file_browser

    step = partial(TutorialStep, also=canvas, post_hook=_put_down)

    return [
        step(
            BASICS,
            tr("Welcome to NegPy"),
            tr(
                "NegPy turns scans of film into finished pictures. It works like a darkroom: you pick a paper, "
                "then set brightness, contrast and color.<br><br>"
                "The chapter menu jumps anywhere. A circle marks a task to try; Undo reverses it."
            ),
            lambda w: None,
            offer=_resume_offer(),
        ),
        step(
            BASICS,
            tr("The Screen"),
            tr(
                "The picture is in the middle, the <b>Film Strip</b> under it, and the controls on the right. "
                "The <b>Roll</b> tab holds what a roll shares, the <b>Frame</b> tab one picture's print. "
                "Edits save by themselves."
            ),
            lambda w: rp(w).group_switcher,
        ),
        step(
            BASICS,
            tr("Find Anything"),
            tr("Every slider, card and action is one search away. Type what you want in your own words, and NegPy opens the control."),
            lambda w: w.drawer.titleBarWidget(),
            task=tr("Press {value} and type a word such as “grade”.", value=_k("command_palette")),
            watch=_palette_open,
        ),
        step(
            BASICS,
            tr("The Library: Rolls"),
            tr(
                "A <b>roll</b> is a group of frames, usually one film. <b>+</b> imports a folder as a roll. "
                "NegPy never moves or changes your files."
            ),
            lambda w: rp(w).library_tree.tree,
            task=tr("Open a frame: double-click a roll, or load the demo negative."),
            watch=lambda w: w.state.current_file_path,
            guide=("library", "Library"),
            offer=_DEMO,
        ),
        step(
            BASICS,
            tr("Keep and Reject"),
            tr(
                "Mark the frames you want. {value} keeps a frame and {value1} rejects it; batch exports skip rejects. "
                "The filter box finds frames by name or by details such as <code>film:portra</code>.",
                value=_k("toggle_keep"),
                value1=_k("toggle_reject"),
            ),
            lambda w: fb(w).list_view,
            task=tr("Press {value} to keep this frame.", value=_k("toggle_keep")),
            watch=lambda w: _marked(w, "keeper"),
            guide=("frames", "Film Strip"),
            offer=_DEMO,
        ),
        step(
            BASICS,
            tr("Light Table"),
            tr(
                "{value} shows the whole roll as film on a light table, for sorting. Right-click a selection to "
                "stitch parts of a large negative or merge a bracket (HDR).",
                value=_k("toggle_light_table"),
            ),
            lambda w: w.toolbar.btn_light_table,
            guide=("frames", "Film Strip"),
        ),
        step(
            ROLL,
            tr("Frame or Roll"),
            tr(
                "Each card has a <b>Frame / Roll</b> pair. Roll cards follow the roll until you change them on one "
                "frame. <b>Roll</b> copies a card to the roll; <b>Reset to Roll</b> brings the roll's values back."
            ),
            lambda w: cp(w).tone_section.roll_btn or cp(w).tone_section,
        ),
        step(
            ROLL,
            tr("Setup"),
            tr(
                "Start here. Pick the film (<b>Color</b>, <b>B&amp;W</b> or <b>Slide</b>), what you scanned with, "
                "and the light. NegPy then shows only the controls your setup uses."
            ),
            lambda w: cp(w).film_section,
            task=tr("Change the film mode, then set it back."),
            watch=lambda w: w.state.config.process.process_mode,
            guide=("film", "Setup"),
            offer=_DEMO,
        ),
        step(
            ROLL,
            tr("Your Scanning Setup"),
            tr(
                "These cards fit your scanner. <b>Frame Assembly</b> joins three-color shots or splits half frames. "
                "<b>Calibration</b> fixes color errors from your light and camera. <b>Optics</b> fixes lens "
                "distortion and uneven light."
            ),
            lambda w: cp(w).sensor_section,
            guide=("sensor", "Calibration"),
        ),
        step(
            ROLL,
            tr("Metering"),
            tr(
                "To invert a negative, NegPy measures its darkest and brightest parts. Keep the film border and "
                "holder out: crop tight, or draw the area to measure ({value}).",
                value=_k("analysis_draw"),
            ),
            lambda w: cp(w).process_section,
            guide=("process", "Metering"),
        ),
        step(
            ROLL,
            tr("Roll Analysis"),
            tr(
                "The frames of one film should print alike. <b>Reanalyze</b> measures the whole roll and shares "
                "the result, so the frames match. It crops the frames that wait for Auto Crop first. <b>Scenes</b> "
                "get their own."
            ),
            lambda w: cp(w).baseline_section,
            guide=("baseline", "Roll Analysis"),
        ),
        step(
            FRAMING,
            tr("Crop"),
            tr(
                "<b>Draw</b>: drag a corner to resize the crop, the inside to move it, or outside the box to rotate. "
                "<b>Auto Crop</b> finds the edges by itself. {value} changes the composition guide.",
                value=_k("crop_guide_next"),
            ),
            lambda w: cp(w).autocrop_section,
            focus=lambda w: cp(w).autocrop_sidebar.manual_crop_btn,
            task=tr("Click Draw and drag a new rectangle on the picture. Enter confirms."),
            watch=lambda w: w.state.config.geometry.crop_rect,
            guide=("autocrop", "Crop"),
            offer=_DEMO,
        ),
        step(
            FRAMING,
            tr("Straighten"),
            tr(
                "<b>Level</b>: draw a line along the horizon, and the picture turns to level it. <b>Tilt</b> and "
                "<b>Swing</b> fix buildings that lean. <b>Square to Film</b> lines the frame up with the film edges."
            ),
            lambda w: cp(w).geometry_section,
            focus=lambda w: cp(w).geometry_sidebar.straighten_btn,
            guide=("geometry", "Straighten"),
        ),
        step(
            PRINTING,
            tr("Print Stock"),
            tr(
                "First pick the paper, as a darkroom printer does. Each real paper has its own contrast and colors. "
                "<b>Digital (Adobe RGB)</b>, the default, is an ideal paper."
            ),
            lambda w: cp(w).paper_section,
            focus=lambda w: cp(w).paper_sidebar.paper_combo,
            task=tr("Pick a print stock other than Digital and compare the print."),
            watch=lambda w: w.state.config.exposure.paper_profile,
            guide=("paper", "Paper Response"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            tr("Print Density"),
            tr(
                "<b>Print Density</b> sets how light or dark the print is, like the exposure time under an enlarger. "
                "Lower is lighter. Auto Density sets it for each frame; your change adjusts from there."
            ),
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.density_slider,
            task=tr("Drag Print Density and watch the print."),
            watch=lambda w: w.state.config.exposure.density,
            guide=("tone", "Tone"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            tr("ISO-R Grade"),
            tr(
                "<b>ISO-R Grade</b> is contrast, like the grade of darkroom paper. A lower number gives more "
                "contrast, and about 110 is normal. Auto Grade sets it for each frame."
            ),
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.grade_slider,
            task=tr("Drag ISO-R Grade."),
            watch=lambda w: w.state.config.exposure.grade,
            guide=("tone", "Tone"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            tr("Test Strip"),
            tr("A <b>test strip</b> prints the frame as a grid of brightness and contrast steps. Click the one you like."),
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.test_strip_btn,
            task=tr("Press {value} to make a test strip.", value=_k("toggle_test_strip")),
            watch=lambda w: w.state.test_strip,
            offer=_DEMO,
        ),
        step(
            PRINTING,
            tr("Shadows and Highlights"),
            tr(
                "The folds under Print Density and ISO-R Grade change only the shadows or only the highlights: "
                "lighter, darker, or more or less contrast."
            ),
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.shadow_density_slider,
            task=tr("Drag one of the four zone sliders."),
            watch=lambda w: tuple(
                getattr(w.state.config.exposure, f) for f in ("shadow_density", "highlight_density", "shadow_grade", "highlight_grade")
            ),
            guide=("tone", "Tone"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            tr("Filtration"),
            tr(
                "Correct the color, like the filters of an enlarger. Click <b>Pick WB</b>, then something that "
                "should be gray, or drag a color slider. {value} opens a grid of color choices.",
                value=_k("toggle_ring_around"),
            ),
            lambda w: cp(w).color_section,
            task=tr("Click Pick WB, then a gray area in the picture, or drag a filter slider."),
            watch=lambda w: (w.state.config.exposure.wb_cyan, w.state.config.exposure.wb_magenta, w.state.config.exposure.wb_yellow),
            guide=("color", "Filtration"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            tr("Dodge and Burn"),
            tr(
                "Make one area lighter (dodge) or darker (burn). Draw a shape, an oval or a fading card edge, then "
                "drag its Burn slider. {value} shows your marks on the picture.",
                value=_k("toggle_printing_notes"),
            ),
            lambda w: cp(w).local_section,
            task=tr("Draw a mask on the picture with any of the three tools, then drag its Burn or Grade."),
            watch=lambda w: tuple((m.stops, m.grade) for m in w.state.config.local.masks if m.stops or m.grade),
            guide=("local", "Dodge & Burn"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            tr("Pick Better"),
            tr(
                "Not sure which slider to move? <b>Pick Better</b>, the balance scale, shows two versions beside "
                "your edit. Click the better one; each round tunes one control."
            ),
            lambda w: cp(w).pick_better_btn,
        ),
        step(
            LOOK,
            tr("Lab"),
            tr(
                "Finishing as a lab scanner does it. <b>Chroma</b> sets color strength, <b>Sharpening</b> sharpens "
                "detail, and <b>Glow</b> and <b>Halation</b> add a soft or a red glow around bright areas."
            ),
            lambda w: cp(w).lab_section,
            task=tr("Drag any Lab slider and watch the print."),
            watch=lambda w: w.state.config.lab,
            guide=("lab", "Lab"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            tr("Toning"),
            tr(
                "<b>Toning</b> changes the color of a B&amp;W print, as darkroom toners do; <b>Split Toning</b> "
                "tints any print. <b>Alternative Processes</b> gives lith, cyanotype and Sabattier looks."
            ),
            lambda w: cp(w).toning_section,
            task=tr("Drag Shadow Strength, then turn Shadow Hue."),
            watch=lambda w: w.state.config.toning,
            guide=("toning", "Toning"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            tr("Dust and Scratches"),
            tr(
                "<b>Optical Removal</b> and <b>IR Removal</b> find dust by themselves. <b>Heal</b> paints dust away, "
                "<b>Scratch</b> and <b>Line</b> repair scratches, and <b>Clone</b> copies film over a defect."
            ),
            lambda w: cp(w).retouch_section,
            focus=lambda w: cp(w).retouch_sidebar.pick_dust_btn,
            task=tr("Pick Heal and click a dust speck on the picture."),
            watch=lambda w: w.state.config.retouch,
            guide=("retouch", "Retouch"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            tr("Finishing"),
            tr(
                "<b>Vignette</b> darkens the edges, <b>Filed Carrier</b> adds a rough black frame, and <b>Border</b> puts a mat around the print."
            ),
            lambda w: cp(w).finish_section,
            task=tr("Drag Vignette Burn, or give the print a Border."),
            watch=lambda w: w.state.config.finish,
            guide=("finish", "Finishing"),
            offer=_DEMO,
        ),
        step(
            SEEING,
            tr("The Analysis Readout"),
            tr(
                "The readout above the tabs describes the print: the paper curve with histograms, a strip of zones "
                "from black to white, and warnings when parts go pure black or white."
            ),
            lambda w: rp(w).analysis_section,
            guide=("analysis", "Analysis"),
        ),
        step(
            SEEING,
            tr("Before and After"),
            tr(
                "{value} splits the canvas between the default conversion and your edit. Hold {value1} to see the scan as a negative.",
                value=_k("toggle_compare"),
                value1=_k("toggle_negative_peek"),
            ),
            lambda w: w.toolbar.source_choice,
            task=tr("Press {value} for the split. Press it again to close it.", value=_k("toggle_compare")),
            watch=lambda w: w.state.compare_mode,
            offer=_DEMO,
        ),
        step(
            SEEING,
            tr("Soft Proof"),
            tr(
                "<b>Soft Proof</b> ({value}) shows on screen how the picture will look on a given printer and paper.",
                value=_k("toggle_soft_proof"),
            ),
            lambda w: rp(w).export_sidebar._soft_proof_section,
            guide=("soft_proof", "Soft Proof"),
        ),
        step(
            OUTPUT,
            tr("History and Work Prints"),
            tr(
                "Every change is a step in <b>History</b>; click one to go back. <b>Work prints</b> ({value}) are "
                "versions you keep under a name.",
                value=_k("save_work_print"),
            ),
            lambda w: rp(w).history_panel,
            pre_hook=lambda w: rp(w).show_tab_by_key("history"),
        ),
        step(
            OUTPUT,
            tr("Metadata"),
            tr("Camera, lens and film details go into every export, so photo apps show your film gear. Set them once for the roll."),
            lambda w: rp(w).metadata_sidebar,
            pre_hook=lambda w: rp(w).show_tab_by_key("metadata"),
            guide=("metadata_gear", "Analog Gear"),
        ),
        step(
            OUTPUT,
            tr("Export"),
            tr(
                "Pick a format and size on <b>File</b>, then click <b>Export</b> at the bottom; its arrow picks this "
                "frame, the selection or all frames. <b>Flat</b> exports a flat master for other editors."
            ),
            lambda w: rp(w).export_sidebar,
            focus=lambda w: rp(w).export_sidebar.intent_btn,
            task=tr("Click Flat or Linear to change the output intent."),
            watch=lambda w: rp(w).export_sidebar.intent_btn.currentIndex(),
            pre_hook=lambda w: rp(w).show_tab_by_key("export"),
        ),
        step(
            SCANNING,
            tr("Scanning"),
            tr(
                "Scan straight into NegPy, from a film scanner (Coolscan, Plustek or SANE) or a camera on a copy "
                "stand with a Scanlight. The frames open as a roll while they scan."
            ),
            lambda w: rp(w).scan_device_section,
            pre_hook=lambda w: (rp(w).show_tab_by_key("scan"), rp(w).show_scan_source("film")),
            guide=("scan_device", "Device"),
        ),
        step(
            SCANNING,
            tr("You Are Ready"),
            tr(
                "{value} lists every shortcut and {value1} finds any control. Each card's ⓘ opens its guide. "
                "Replay this tour from the ⋯ menu.",
                value=_k("show_shortcuts"),
                value1=_k("command_palette"),
            ),
            lambda w: None,
            pre_hook=lambda w: rp(w).show_tab_by_key("roll"),
        ),
    ]
