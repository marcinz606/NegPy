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

    return isinstance(QApplication.activeModalWidget(), CommandPalette)


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
        return w.session_panel.file_browser

    step = partial(TutorialStep, also=canvas, post_hook=_put_down)

    return [
        step(
            BASICS,
            tr("Welcome to NegPy"),
            tr(
                "NegPy is a <b>virtual darkroom</b>: it reads a scan as film density and prints it on a "
                "model of real paper.<br><br>"
                "The chapter menu jumps anywhere in the tour. A circle marks a task; Undo reverses its edit."
            ),
            lambda w: None,
            offer=_resume_offer(),
        ),
        step(
            BASICS,
            tr("The Screen"),
            tr(
                "Left: the <b>Library</b> and the <b>Film Strip</b>. Center: the canvas, where tools act. "
                "Right: the controls. The <b>Roll</b> tab holds what a roll shares, the <b>Frame</b> tab one "
                "picture's print. Edits save automatically, keyed to the image content."
            ),
            lambda w: rp(w).group_switcher,
        ),
        step(
            BASICS,
            tr("Find Anything"),
            tr(
                "Every slider, card and action is one search away. Type what you want in your own words and "
                "the result opens the right tab and shows the control."
            ),
            lambda w: w.drawer.titleBarWidget(),
            task=tr("Press {value} and type a word such as “grade”.", value=_k("command_palette")),
            watch=_palette_open,
        ),
        step(
            BASICS,
            tr("The Library: Rolls"),
            tr(
                "A <b>roll</b> is a named group of frames. <b>+</b> imports a folder, or each subfolder, as a "
                "roll. NegPy never moves, renames or deletes your files. <b>Save as Roll…</b> in the Film "
                "Strip makes a roll of any frames."
            ),
            lambda w: w.session_panel.library_tree.tree,
            task=tr("Open a frame: double-click a roll, or load the demo negative."),
            watch=lambda w: w.state.current_file_path,
            guide=("library", "Library"),
            offer=_DEMO,
        ),
        step(
            BASICS,
            tr("Filter and Search"),
            tr(
                "{value} filters the open frames by file name or by <code>field:value</code> terms, such as <code>film:portra iso:&gt;=400</code>. {value1} searches the whole library. <b>Search by meaning</b> (Preferences → Performance) finds “a dog on a beach”.",
                value=_k("focus_search"),
                value1=_k("search_library"),
            ),
            lambda w: fb(w).search_input,
            guide=("frames", "Film Strip"),
        ),
        step(
            BASICS,
            tr("Keep and Reject"),
            tr(
                "Cull on the contact sheet. {value} marks a keeper, {value1} rejects a frame. A reject stays on the sheet but drops out of batch exports. The funnel filters the grid to <b>Keepers Only</b> or <b>Hide Rejected</b>.",
                value=_k("toggle_keep"),
                value1=_k("toggle_reject"),
            ),
            lambda w: fb(w).list_view,
            task=tr("Press {value} to keep this frame.", value=_k("toggle_keep")),
            watch=lambda w: _marked(w, "keeper"),
            offer=_DEMO,
        ),
        step(
            BASICS,
            tr("Light Table, Stitch and HDR"),
            tr(
                "{value} opens the <b>Light Table</b>: large thumbnails over the window. Right-click a selection: <b>Stitch Selected Frames</b> joins parts of a large negative, <b>Merge Exposures (HDR)</b> joins a bracket, and <b>Scene</b> groups frames of one subject.",
                value=_k("toggle_light_table"),
            ),
            lambda w: fb(w).light_table_btn,
            guide=("frames", "Film Strip"),
        ),
        step(
            ROLL,
            tr("Frame or Roll"),
            tr(
                "Each card header has a <b>Frame / Roll</b> pair. Roll tab cards follow the roll until you "
                "edit one. Roll pushes a card's settings to the roll or the selection; <b>Reset to Roll</b> "
                "pulls the roll's value back."
            ),
            lambda w: cp(w).tone_section.roll_btn or cp(w).tone_section,
        ),
        step(
            ROLL,
            tr("Film Mode"),
            tr(
                "The first choice: <b>Color</b> negative, <b>B&amp;W</b> negative or <b>Slide</b>. The wand "
                "detects the mode when a file opens. <b>Positive</b>, on Slide, is for a file that is already "
                "a positive, such as a scanned print."
            ),
            lambda w: cp(w).film_section,
            task=tr("Change the film mode, then set it back."),
            watch=lambda w: w.state.config.process.process_mode,
            guide=("film", "Film Mode"),
            offer=_DEMO,
        ),
        step(
            ROLL,
            tr("Frame Assembly"),
            tr(
                "<b>Trichrome</b> merges three exposures of one negative, under red, green and blue light, "
                "into one clean color scan. <b>Half Frame</b> splits each scan from a half-frame camera into "
                "two frames, each with its own edit and export."
            ),
            lambda w: cp(w).assembly_section,
            guide=("assembly", "Frame Assembly"),
        ),
        step(
            ROLL,
            tr("Calibration"),
            tr(
                "This card corrects the capture, not the look. The <b>bulb</b> asks how you scan, then sets "
                "<b>Camera WB</b> or <b>Linear RAW</b>. <b>Single-Shot Narrowband Calibration</b> "
                "removes sensor leak between bands. <b>Hue Trim</b> rotates hues for an unusual lamp."
            ),
            lambda w: cp(w).sensor_section,
            focus=lambda w: cp(w).sensor_sidebar.scan_setup_btn,
            guide=("sensor", "Calibration"),
        ),
        step(
            ROLL,
            tr("Crosstalk"),
            tr(
                "Film dyes absorb outside their own bands, which mutes color. <b>Crosstalk</b> unmixes them "
                "before metering. Pick the matrix for your process and set Strength; <b>+</b> edits a "
                "matrix. A color chart shot on the roll? <b>Measure from Chart…</b> fits one to it. "
                "Run Roll Analysis again after a change."
            ),
            lambda w: cp(w).sensor_section,
            focus=lambda w: cp(w).sensor_sidebar.crosstalk_combo,
            guide=("sensor", "Calibration"),
        ),
        step(
            ROLL,
            tr("Cast Removal"),
            tr(
                "A negative's cast changes with density, so one white balance cannot fix it. "
                "<b>Cast Removal</b> gives each channel its own slope, set from the frame's neutrals, so "
                "grays stay neutral from shadow to highlight."
            ),
            lambda w: cp(w).sensor_section,
            focus=lambda w: cp(w).sensor_sidebar.cast_removal_slider,
            guide=("sensor", "Calibration"),
        ),
        step(
            ROLL,
            tr("Metering"),
            tr(
                "NegPy meters the black point, white point and orange mask. Keep rebate and holder out: crop tight, set an <b>Analysis Buffer</b> or draw a region ({value}). <b>Lock Bounds</b> keeps the result; White and Black Point trim it. <b>Film Base</b>, one click on the clear rebate, lets Analysis read Dmax and Dmin above the base.",
                value=_k("analysis_draw"),
            ),
            lambda w: cp(w).process_section,
            guide=("process", "Metering"),
        ),
        step(
            ROLL,
            tr("Roll Analysis"),
            tr(
                "One enlarger setting for the roll. <b>Reanalyze</b> meters every frame into a roll baseline; "
                "<b>Use average</b> (Luma, Color, Cast) applies it, so frames print alike. <b>Use This "
                "Frame</b> makes one frame the reference. <b>Scenes</b> get their own baseline."
            ),
            lambda w: cp(w).baseline_section,
            guide=("baseline", "Roll Analysis"),
        ),
        step(
            ROLL,
            tr("Raw Decode"),
            tr("<b>Raw Decode</b> picks the demosaic method for preview and export, and recovers clipped raw highlights."),
            lambda w: cp(w).demosaic_section,
            guide=("demosaic", "Raw Decode"),
        ),
        step(
            ROLL,
            tr("Optics"),
            tr(
                "<b>Optics</b> corrects the rig: Distortion straightens curved edges, a file with a lens "
                "profile uses its embedded correction, and <b>Flat Field</b> divides out uneven light with a "
                "shot of the bare light source."
            ),
            lambda w: cp(w).optics_section,
            guide=("optics", "Optics"),
        ),
        step(
            FRAMING,
            tr("Crop"),
            tr(
                "The crop sets the framing and what the meter reads. Drag a corner to resize, inside to move, a handle outside the box to rotate. The wand finds the film edge. {value} changes the guides; {value1} turns them.",
                value=_k("crop_guide_next"),
                value1=_k("crop_guide_orient"),
            ),
            lambda w: cp(w).geometry_section,
            focus=lambda w: cp(w).geometry_sidebar.manual_crop_btn,
            task=tr("Click the crop tool and drag a new rectangle on the picture. Enter confirms."),
            watch=lambda w: w.state.config.geometry.crop_rect,
            guide=("geometry", "Geometry"),
            offer=_DEMO,
        ),
        step(
            FRAMING,
            tr("Straighten, Tilt and Swing"),
            tr(
                "<b>Straighten</b>: draw a line along a horizon and the frame levels to it; Fine Rotation "
                "trims by hand. <b>Tilt</b> and <b>Swing</b> are easel movements for converging verticals "
                "and horizontals. <b>Crop by Default</b> trims the wedge they leave. <b>Auto Skew</b> squares "
                "the frame to its film edges."
            ),
            lambda w: cp(w).geometry_section,
            focus=lambda w: cp(w).geometry_sidebar.straighten_btn,
            guide=("geometry", "Geometry"),
        ),
        step(
            FRAMING,
            tr("The Roll's Crop"),
            tr(
                "Auto Crop, Ratio, Mode (image or full film edge), Crop Offset and Rebate Trim belong to the roll. "
                "<b>Auto-crop all frames</b> crops them together, so weak detections borrow from "
                "strong ones. Crops you drew stay."
            ),
            lambda w: cp(w).autocrop_section,
            guide=("autocrop", "Crop"),
        ),
        step(
            PRINTING,
            tr("Print Stock"),
            tr(
                "Pick the paper first, as a printer does. <b>Print stock</b> prints on a real RA4 paper or "
                "print film, such as Kodak Endura Premier, Fujicolor Crystal Archive or Kodak Vision 2383, "
                "with the curve and dye colors from its datasheet. <b>Digital (Adobe RGB)</b>, the default, "
                "is an idealized print. A B&amp;W negative lists B&amp;W papers."
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
                "<b>Print Density</b> is enlarger exposure: it slides the negative along the paper curve. A "
                "lower value prints lighter. Auto Density meters each frame, but only part of the way, so a "
                "low-key frame stays dark."
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
                "<b>ISO-R Grade</b> is paper contrast on the ISO-R scale, 50 to 180. A lower R is harder, a "
                "higher R is softer, and about 110 is classic grade 2. Auto Grade meters contrast per frame."
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
            tr("Auto and Set Targets"),
            tr(
                "The <b>Auto</b> menu turns Auto Density and Auto Grade on or off; <b>Hold Black</b> keeps "
                "the black point under a shadow lift. <b>Set Targets…</b> moves their aim (density, contrast, "
                "<b>Shadow Reach</b>, <b>Highlight Hold</b>) for every image. <b>Bake Auto Density/Grade</b> "
                "freezes a frame's metered values, so a later target change leaves it alone."
            ),
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.auto_btn,
            guide=("tone", "Tone"),
        ),
        step(
            PRINTING,
            tr("Pick Better"),
            tr(
                "Not sure which slider to move? <b>Pick Better</b>, the balance scale on the Exposure tab's "
                "bar, prints two versions beside your edit. Click the better one, or the center to skip. "
                "Each round tunes one control, then the rounds repeat in finer steps. <b>Custom Routing</b> "
                "sets the order."
            ),
            lambda w: cp(w).pick_better_btn,
        ),
        step(
            PRINTING,
            tr("Test Strip"),
            tr(
                "A <b>test strip</b> prints bands of the frame at stepped exposure and grade, as on an easel. "
                "Click the band you like and its values apply."
            ),
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.test_strip_btn,
            task=tr("Press {value} to make a test strip.", value=_k("toggle_test_strip")),
            watch=lambda w: w.state.test_strip,
            offer=_DEMO,
        ),
        step(
            PRINTING,
            tr("Zone Density and Split Grade"),
            tr(
                "<b>Shadows Density</b> and <b>Highlights Density</b> burn or hold one zone and roll into "
                "paper black and white, without a clip. <b>Shadows Grade</b> and <b>Highlights Grade</b> "
                "change one zone's contrast, as a split-grade print does: harder shadows without harsh "
                "highlights."
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
            tr("Preflash and Contrast Mask"),
            tr(
                "<b>Preflash</b>, a short even exposure before the print, brings in thin highlight detail and "
                "keeps shadows clean. <b>Contrast Mask</b>, the darkroom unsharp mask, lowers overall "
                "contrast and keeps local detail. <b>Mask Spacer</b> sets its softness."
            ),
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.preflash_slider,
            guide=("tone", "Tone"),
        ),
        step(
            PRINTING,
            tr("Paper Response"),
            tr(
                "These controls trim the print stock's curve. <b>Toe</b> and "
                "<b>Shoulder</b> shape the curve's ends, <b>Snap</b> the midtones, <b>Dye Separation</b> the "
                "color strength. <b>Paper White</b> and <b>Paper Black</b> show the paper's base and deepest "
                "black."
            ),
            lambda w: cp(w).paper_section,
            guide=("paper", "Paper Response"),
        ),
        step(
            PRINTING,
            tr("One Dye Layer"),
            tr(
                "Pick <b>R</b>, <b>G</b> or <b>B</b> on Tone or Paper Response and that card's curve controls "
                "trim one dye layer. Filtration only shifts a layer; a trim changes its shape, for a cast that "
                "differs in shadows and highlights. A dot marks a trimmed layer."
            ),
            lambda w: cp(w).paper_section,
            focus=lambda w: cp(w).paper_sidebar.ch_btn,
            guide=("paper", "Paper Response"),
        ),
        step(
            PRINTING,
            tr("Filtration"),
            tr(
                "White balance is CC filtration, as on a color enlarger. The CMY sliders move the pack; Temperature warms or cools ({value} / {value1}). The region menu limits it to shadows or highlights. {value2} opens a <b>Ring-around</b>.",
                value=_k("temp_warm"),
                value1=_k("temp_cool"),
                value2=_k("toggle_ring_around"),
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
                "<b>Draw Mask</b> is a cut card, <b>Oval</b> a hole in a card, <b>Card Edge</b> a graded burn. Burn is in stops (negative dodges); Grade sets contrast. <b>Tone Limit</b> keeps a mask to one zone. {value} shows the printing map.",
                value=_k("toggle_printing_notes"),
            ),
            lambda w: cp(w).local_section,
            task=tr("Draw a mask on the picture with any of the three tools, then drag its Burn or Grade."),
            watch=lambda w: tuple((m.stops, m.grade) for m in w.state.config.local.masks if m.stops or m.grade),
            guide=("local", "Dodge & Burn"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            tr("Lab"),
            tr(
                "The look after the print. <b>Chroma</b> scales color; <b>Skin Protection</b> keeps faces "
                "natural. <b>Chroma Denoise</b> removes color noise, not grain. <b>Sharpening</b> acts on "
                "lightness only. <b>CLAHE</b> adds local contrast. <b>Glow</b> and <b>Halation</b> add lens "
                "bloom and film's red halo."
            ),
            lambda w: cp(w).lab_section,
            task=tr("Drag any Lab slider and watch the print."),
            watch=lambda w: w.state.config.lab,
            guide=("lab", "Lab"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            tr("Alternative Processes"),
            tr(
                "For B&amp;W negatives. <b>Lith</b> gives creamy warm highlights and sudden hard blacks. "
                "<b>Cyanotype</b> prints in Prussian blue, with bleach and tannin toning. <b>Sabattier</b> "
                "re-exposes the print part-way through development, for partial reversal and Mackie lines. One selector picks "
                "one process, because they exclude each other."
            ),
            lambda w: cp(w).altproc_section,
            guide=("altproc", "Alternative Processes"),
        ),
        step(
            LOOK,
            tr("Toning"),
            tr(
                "<b>Split Toning</b> moves shadows and highlights toward their own hues and keeps lightness, "
                "so the grain stays. <b>Chemical Toning</b> (B&amp;W) runs six baths on the print's silver, "
                "in the order shown: Selenium, Sepia, Gold, Iron Blue, Copper and Vanadium."
            ),
            lambda w: cp(w).toning_section,
            task=tr("Drag Shadow Strength, then turn Shadow Hue."),
            watch=lambda w: w.state.config.toning,
            guide=("toning", "Toning"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            tr("Dust Removal"),
            tr(
                "<b>Optical Removal</b> finds dust by local contrast. <b>IR Removal</b> uses the scanner's "
                "infrared channel, where dust shows and the dyes do not. The <b>Overlay</b> menu shows what "
                "each pass marked, so you can set a threshold by eye."
            ),
            lambda w: cp(w).retouch_section,
            focus=lambda w: cp(w).retouch_sidebar.auto_dust_btn,
            guide=("retouch", "Retouch"),
        ),
        step(
            LOOK,
            tr("Heal, Clone, Scratch and Line"),
            tr(
                "<b>Heal</b> marks a search area, not a stamp: only pixels that stand out are rebuilt, with "
                "the frame's own grain. <b>Clone</b> copies film from another area where Heal cannot rebuild. "
                "<b>Scratch</b> follows a line you click along a hair. <b>Line</b> "
                "traces a whole straight transport scratch from one click."
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
                "Presentation, after the crop. <b>Vignette</b> is an edge burn in stops. <b>Filed "
                "Carrier</b> prints the rough black edge of a filed-out carrier. <b>Border</b> puts a mat "
                "around the print; <b>Paper White</b> matches it to the toned paper."
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
                "The densitometer above the tabs: an <b>H&amp;D chart</b> with this frame on the paper curve, "
                "or a <b>Waveform</b> or <b>Vectorscope</b>; histograms with clip marks, and a strip of print "
                "zones. <b>Probe</b> reads the density under the cursor."
            ),
            lambda w: rp(w).analysis_section,
            guide=("analysis", "Analysis"),
        ),
        step(
            SEEING,
            tr("Zone Placement"),
            tr(
                "Place zones as in the Zone System. Pin a point in the picture, give it a target zone, and "
                "<b>Place zones</b> solves Print Density and Grade so the pins print where you want them."
            ),
            lambda w: rp(w).analysis_section,
            guide=("analysis", "Analysis"),
        ),
        step(
            SEEING,
            tr("Around the Canvas"),
            tr(
                "Double-click the picture to switch between Fit and 100% at that spot; zoomed in, the <b>Navigator</b> moves the view. {value} cycles the frame info and {value1} puts the density probe beside the pointer. <b>Tools</b> on the toolbar arms any canvas tool: crop, pickers, masks and retouching.",
                value=_k("cycle_frame_info"),
                value1=_k("toggle_cursor_probe"),
            ),
            lambda w: w.toolbar.btn_tools,
        ),
        step(
            SEEING,
            tr("Before and After"),
            tr(
                "Split the canvas between this print and the default conversion. Drag the divider to move it. "
                "The toolbar's <b>Print</b> menu holds it, with the peeks."
            ),
            lambda w: w.toolbar.source_choice,
            task=tr("Press {value} for the split. Press it again to close it.", value=_k("toggle_compare")),
            watch=lambda w: w.state.compare_mode,
            offer=_DEMO,
        ),
        step(
            SEEING,
            tr("Peeks, Reference, Loupe and Zones"),
            tr(
                "Hold {value} for the scan as captured, {value1} for the flat master, {value2} for the camera's preview. {value3} pins a frame beside the canvas, to match a roll by eye. {value4} is a grain loupe and {value5} maps the print's zones in color, gray steps or a grid, both under <b>Overlays</b> on the toolbar.",
                value=_k("toggle_negative_peek"),
                value1=_k("toggle_flat_peek"),
                value2=_k("toggle_embedded_peek"),
                value3=_k("toggle_reference"),
                value4=_k("toggle_grain_focuser"),
                value5=_k("toggle_zones"),
            ),
            lambda w: w.toolbar.source_choice,
            task=tr("Hold {value} to see the negative.", value=_k("toggle_negative_peek")),
            watch=lambda w: w.state.negative_peek,
            offer=_DEMO,
        ),
        step(
            SEEING,
            tr("Soft Proof"),
            tr(
                "<b>Soft Proof</b> shows the print as a printer and paper will make it: paper white, ink black and a gamut warning. {value} turns it on and off. The display profile keeps the canvas true on your monitor.",
                value=_k("toggle_soft_proof"),
            ),
            lambda w: rp(w).export_sidebar._soft_proof_section,
            guide=("soft_proof", "Soft Proof"),
        ),
        step(
            OUTPUT,
            tr("History and Work Prints"),
            tr(
                "Each edit is a step in <b>History</b>; click one to go back, then edit on to branch. <b>Work prints</b> are named versions you keep: {value} saves one, to compare or export later. History stays after a restart.",
                value=_k("save_work_print"),
            ),
            lambda w: rp(w).history_panel,
            pre_hook=lambda w: rp(w).show_tab_by_key("history"),
        ),
        step(
            OUTPUT,
            tr("Favorites and Presets"),
            tr(
                "<b>Favorites</b> is your own tab: pin the controls you use most. <b>Presets</b> keep chosen "
                "settings and apply them to other frames, over your edit or as a new look."
            ),
            lambda w: rp(w).favourites_sidebar,
            pre_hook=lambda w: rp(w).show_tab_by_key("favourites"),
            guide=("presets", "Presets"),
        ),
        step(
            OUTPUT,
            tr("Metadata and Gear"),
            tr(
                "Film, camera, lens and scan details go into the exported EXIF and XMP; the cards follow the "
                "roll. <b>Protect Original Metadata</b> exports the source's own tags instead. The "
                "<b>Gear</b> tab keeps your cameras, lenses and films for every picker."
            ),
            lambda w: rp(w).metadata_sidebar,
            focus=lambda w: rp(w).metadata_sidebar.protect_btn,
            pre_hook=lambda w: rp(w).show_tab_by_key("metadata"),
            guide=("metadata_gear", "Analog Gear"),
        ),
        step(
            OUTPUT,
            tr("Roll Settings"),
            tr(
                "One dialog tags gear, capture, place, process and scan fields for this frame, a selection "
                "or the roll. A new roll whose folder name matches your gear, such as “penf” for “Pen F”, "
                "opens it filled in."
            ),
            lambda w: fb(w).roll_settings_btn,
        ),
        step(
            OUTPUT,
            tr("Output Intent"),
            tr(
                "<b>Print</b> is the finished print. <b>Flat</b> is a flat log master for other editors, with no print look or effects. <b>Linear</b> is scene-linear data for a raw workflow. {value} shows Flat on the canvas.",
                value=_k("toggle_flat_peek"),
            ),
            lambda w: rp(w).export_sidebar.intent_btn,
            task=tr("Open the intent menu and choose another intent."),
            watch=lambda w: rp(w).export_sidebar.intent_btn.currentIndex(),
            pre_hook=lambda w: rp(w).show_tab_by_key("export"),
        ),
        step(
            OUTPUT,
            tr("Export"),
            tr(
                "Pick a format, color space and size. The button's arrow chooses this frame, the selection "
                "or all visible frames. <b>Presets</b> run several recipes at once. <b>Contact Sheet</b> prints "
                "the roll as film strips; <b>Sidecars</b> save edits beside the scans."
            ),
            lambda w: rp(w).export_sidebar,
            focus=lambda w: rp(w).export_sidebar.export_main_btn,
            pre_hook=lambda w: rp(w).show_tab_by_key("export"),
            guide=("export_presets", "Export Presets"),
        ),
        step(
            SCANNING,
            tr("Film Scanners"),
            tr(
                "Scan straight into NegPy. Pick <b>Film Scanner</b> in the Scanner card for SANE scanners, the "
                "Nikon Coolscan and Plustek. Prescan a strip, set an offset per frame, and the frames open as a "
                "roll while they are scanned."
            ),
            lambda w: rp(w).scan_device_section,
            pre_hook=lambda w: (rp(w).show_tab_by_key("scan"), rp(w).show_scan_source("film")),
            guide=("scan_device", "Device"),
        ),
        step(
            SCANNING,
            tr("Camera Scanning"),
            tr(
                "Tether a camera through gphoto2, with live view, and drive a Scanlight for white or red, "
                "green and blue exposures. A Trichrome capture arrives grouped, ready to merge."
            ),
            lambda w: rp(w).scan_camera_section,
            pre_hook=lambda w: (rp(w).show_tab_by_key("scan"), rp(w).show_scan_source("camera")),
            guide=("scan_camera", "Camera"),
        ),
        step(
            SCANNING,
            tr("You're All Set"),
            tr(
                "{value} lists every shortcut and {value1} finds any control. <b>Preferences</b> ({value2}) holds the interface, performance and storage settings. Each panel's ⓘ opens its full guide. Replay this tour, one chapter at a time, from the ⋯ menu.",
                value=_k("show_shortcuts"),
                value1=_k("command_palette"),
                value2=_k("open_preferences"),
            ),
            lambda w: None,
            pre_hook=lambda w: rp(w).show_tab_by_key("roll"),
        ),
    ]
