from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.shortcut_registry import display_key, key_for
from negpy.desktop.view.widgets.tutorial_overlay import Offer, TutorialStep

if TYPE_CHECKING:
    from negpy.desktop.view.main_window import MainWindow

BASICS = "Basics"
ROLL = "The Roll"
FRAMING = "Framing"
PRINTING = "Printing"
LOOK = "Look & Finish"
SEEING = "Seeing the Print"
OUTPUT = "Output & Records"
SCANNING = "Scanning"


def _k(action_id: str) -> str:
    """The action's current key, bold, so a rebind never leaves the tour teaching a dead key."""
    return f"<b>{display_key(key_for(action_id))}</b>"


def _load_demo(w: "MainWindow") -> None:
    from negpy.desktop.view.widgets.tutorial_demo import write_demo
    from negpy.kernel.system.config import BASE_USER_DIR

    w.controller.request_asset_discovery([str(write_demo(Path(BASE_USER_DIR) / "tour"))], auto_open=True)


_DEMO = Offer("Load Demo Negative", _load_demo, lambda w: not w.state.current_file_path)


def _marked(w: "MainWindow", mark: str) -> bool:
    files, i = w.state.uploaded_files, w.state.selected_file_idx
    return 0 <= i < len(files) and bool(files[i].get(mark))


def _palette_open(_w: "MainWindow") -> bool:
    from negpy.desktop.view.widgets.command_palette import CommandPalette

    return isinstance(QApplication.activeModalWidget(), CommandPalette)


def _put_down(w: "MainWindow") -> None:
    """Esc until the canvas is plain, so a tool or view picked up on one step never rides into the next."""
    from negpy.desktop.session import ToolMode

    st = w.state
    esc = w.shortcut_manager.action_for("cancel_tool")
    for _ in range(12):
        busy = (
            st.active_tool != ToolMode.NONE,
            st.test_strip or st.test_strip_pending,
            st.negative_peek or st.embedded_peek or st.flat_peek,
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
        "Resume Where You Left Off",
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

    # The picture stays undimmed on every step, so an edit shows as it will print, and each step
    # leaves the canvas as it found it.
    step = partial(TutorialStep, also=canvas, post_hook=_put_down)

    return [
        step(
            BASICS,
            "Welcome to NegPy",
            "NegPy prints your film scans through a <b>virtual darkroom</b>: it reads a scan as film "
            "density and prints it on a model of real paper. The controls are darkroom controls: "
            "exposure, grade, filtration, dodge and burn.<br><br>"
            "The chapter menu above jumps anywhere in the tour. A circle marks a task to try; Undo "
            "reverses its edit.",
            lambda w: None,
            offer=_resume_offer(),
        ),
        step(
            BASICS,
            "The Screen",
            "Left: the <b>Library</b> and the <b>Film Strip</b>. Center: the canvas, where tools act. "
            "Right: the controls. The <b>Roll</b> tab holds what a whole roll shares; the <b>Frame</b> "
            "tab holds one picture's print, from Geometry to Finish. Export, Metadata, Gear and Scan "
            "sit beside them. Edits save by themselves, keyed to the image content.",
            lambda w: rp(w).group_switcher,
        ),
        step(
            BASICS,
            "Find Anything",
            "Every slider, card and action is one search away. Type what you want in your own words and "
            "the result opens the right tab and shows the control.",
            lambda w: w.drawer.titleBarWidget(),
            task=f"Press {_k('command_palette')} and type a word such as “grade”.",
            watch=_palette_open,
        ),
        step(
            BASICS,
            "The Library: Rolls",
            "A <b>roll</b> is a named group of frames. <b>+</b> imports a folder as a roll, or each "
            "folder inside a parent as a roll. NegPy never moves, renames or deletes your files. "
            "Double-click a roll to open it. Any hand-picked set of frames can also be a roll: "
            "<b>Save as Roll…</b> in the Film Strip.",
            lambda w: w.session_panel.library_tree.tree,
            task="Open a frame: double-click a roll, or load the demo negative.",
            watch=lambda w: w.state.current_file_path,
            guide=("library", "Library"),
            offer=_DEMO,
        ),
        step(
            BASICS,
            "Filter and Search",
            f"{_k('focus_search')} filters the open frames. A word matches the file name; "
            "<code>field:value</code> terms match what a frame is, for example "
            "<code>film:portra iso:&gt;=400</code>. "
            f"{_k('search_library')} runs the same search over the whole library. "
            "<b>Search by meaning</b> (Preferences → Performance) finds “a dog on a beach”.",
            lambda w: fb(w).search_input,
            guide=("frames", "Film Strip"),
        ),
        step(
            BASICS,
            "Keep and Reject",
            f"Cull on the contact sheet. {_k('toggle_keep')} marks a keeper, {_k('toggle_reject')} "
            "rejects a frame. A reject stays on the sheet but drops out of batch exports. The funnel "
            "filters the grid to <b>Keepers Only</b> or <b>Hide Rejected</b>.",
            lambda w: fb(w).list_view,
            task=f"Press {_k('toggle_keep')} to keep this frame.",
            watch=lambda w: _marked(w, "keeper"),
            offer=_DEMO,
        ),
        step(
            BASICS,
            "Light Table, Stitch and HDR",
            f"{_k('toggle_light_table')} opens the <b>Light Table</b>: large thumbnails over the whole "
            "window. Select frames and right-click for more: <b>Stitch Selected Frames</b> joins the "
            "parts of a large negative, <b>Merge Exposures (HDR)</b> joins a bracket into one frame, and "
            "<b>Scene</b> groups frames of one subject. The files on disk stay as they are.",
            lambda w: fb(w).light_table_btn,
            guide=("frames", "Film Strip"),
        ),
        step(
            ROLL,
            "Frame or Roll",
            "Each card header has a <b>Frame / Roll</b> pair. A Roll tab card follows the roll; move a "
            "slider and it flips to Frame, click Roll to give the roll this value. On a Frame card, Roll "
            "copies chosen settings to the selected frames or the roll. <b>Reset to Roll</b> pulls the "
            "roll's value back.",
            lambda w: cp(w).tone_section.roll_btn or cp(w).tone_section,
        ),
        step(
            ROLL,
            "Film Mode",
            "The first choice: <b>Color</b> negative, <b>B&amp;W</b> negative or <b>Slide</b>. Each one "
            "changes the conversion. The wand detects the mode when a file opens. <b>Positive</b>, on "
            "Slide, is for a file that is already a positive, such as a scanned print.",
            lambda w: cp(w).film_section,
            task="Change the film mode, then set it back.",
            watch=lambda w: w.state.config.process.process_mode,
            guide=("film", "Film Mode"),
            offer=_DEMO,
        ),
        step(
            ROLL,
            "Frame Assembly",
            "<b>Trichrome</b> merges three exposures of one negative, under red, green and blue light, "
            "into one clean color scan. <b>Half Frame</b> splits each scan from a half-frame camera into "
            "two frames, each with its own edit and export.",
            lambda w: cp(w).assembly_section,
            guide=("assembly", "Frame Assembly"),
        ),
        step(
            ROLL,
            "Calibration",
            "This card corrects the capture, not the look. The <b>bulb</b> asks how you scan and what "
            "light you use, then sets Linear RAW and Narrowband. <b>Narrowband</b> corrects RGB-LED "
            "light. <b>Single-Shot Narrowband Calibration</b> removes the sensor's leak between bands. "
            "<b>Hue Trim</b> turns all hues back for an unusual lamp.",
            lambda w: cp(w).sensor_section,
            focus=lambda w: cp(w).sensor_sidebar.scan_setup_btn,
            guide=("sensor", "Calibration"),
        ),
        step(
            ROLL,
            "Crosstalk",
            "Each film dye also absorbs outside its own band, which mutes color. A <b>Crosstalk</b> "
            "matrix unmixes the dyes in density, before metering. Pick the matrix for your process and "
            "blend it with Strength; <b>+</b> edits a matrix. Run Roll Analysis again after a change.",
            lambda w: cp(w).sensor_section,
            focus=lambda w: cp(w).sensor_sidebar.crosstalk_combo,
            guide=("sensor", "Calibration"),
        ),
        step(
            ROLL,
            "Cast Removal",
            "A negative's cast changes with density, so one white balance leaves shadows and highlights "
            "off. <b>Cast Removal</b> gives each channel its own slope, so grays stay neutral from "
            "shadow to highlight. It adapts to the neutrals in each frame. It starts on for color "
            "negatives and off for slides.",
            lambda w: cp(w).sensor_section,
            focus=lambda w: cp(w).sensor_sidebar.cast_removal_slider,
            guide=("sensor", "Calibration"),
        ),
        step(
            ROLL,
            "Metering",
            "NegPy meters the negative to find its black and white points and its orange mask. Rebate, "
            "sprocket holes and holder are not picture: crop tight, set an <b>Analysis Buffer</b>, or "
            f"draw a region ({_k('analysis_draw')}). <b>Lock Bounds</b> keeps the result once it looks "
            "right. White Point and Black Point trim it.",
            lambda w: cp(w).process_section,
            guide=("process", "Metering"),
        ),
        step(
            ROLL,
            "Roll Analysis",
            "One enlarger setting for the roll. <b>Reanalyze</b> meters every frame and keeps a roll "
            "baseline. <b>Use average</b> (Luma, Color, Cast) lets frames borrow it, so exposure and "
            "color do not jump. <b>Use This Frame</b> makes one frame the reference. <b>Scenes</b> get "
            "a baseline of their own.",
            lambda w: cp(w).baseline_section,
            guide=("baseline", "Roll Analysis"),
        ),
        step(
            ROLL,
            "Raw Decode",
            "<b>Raw Decode</b> picks the demosaic method for preview and export, and recovers clipped raw highlights.",
            lambda w: cp(w).demosaic_section,
            guide=("demosaic", "Raw Decode"),
        ),
        step(
            ROLL,
            "Optics",
            "<b>Optics</b> corrects the rig: Distortion straightens curved edges, a file with a lens "
            "profile uses its embedded correction, and <b>Flat Field</b> divides out uneven light with a "
            "shot of the bare light source.",
            lambda w: cp(w).optics_section,
            guide=("optics", "Optics"),
        ),
        step(
            FRAMING,
            "Crop",
            "The crop sets the framing and what the meter reads. Drag a corner to resize, drag inside to "
            "move. The wand finds the film edge. "
            f"{_k('crop_guide_next')} changes the guide lines and {_k('crop_guide_orient')} turns them. "
            "The handles outside the box rotate the frame by hand.",
            lambda w: cp(w).geometry_section,
            focus=lambda w: cp(w).geometry_sidebar.manual_crop_btn,
            task="Click the crop tool and drag a new rectangle on the picture. Enter confirms.",
            watch=lambda w: w.state.config.geometry.crop_rect,
            guide=("geometry", "Geometry"),
            offer=_DEMO,
        ),
        step(
            FRAMING,
            "Straighten, Tilt and Swing",
            "<b>Straighten</b>: draw a line along a horizon and the frame levels to it. Fine Rotation "
            "trims by hand. <b>Tilt</b> and <b>Swing</b> are easel movements for converging verticals "
            "and horizontals. <b>Crop by Default</b> trims the wedge they leave, so no edge shows "
            "invented pixels.",
            lambda w: cp(w).geometry_section,
            focus=lambda w: cp(w).geometry_sidebar.straighten_btn,
            guide=("geometry", "Geometry"),
        ),
        step(
            FRAMING,
            "The Roll's Crop",
            "The crop's shape and what the detector looks for belong to the roll: Ratio, Mode (image or "
            "full film edge), Crop Offset and Rebate Trim. <b>Auto-crop the roll</b> analyzes all "
            "frames together, so weak detections borrow from strong ones. Crops you drew stay.",
            lambda w: cp(w).autocrop_section,
            guide=("autocrop", "Crop"),
        ),
        step(
            PRINTING,
            "Print Density",
            "<b>Print Density</b> is enlarger exposure: it slides the negative along the paper curve. A "
            "lower value prints lighter. Auto Density meters each frame, but only part of the way, so a "
            "low-key frame stays dark.",
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.density_slider,
            task="Drag Print Density and watch the print.",
            watch=lambda w: w.state.config.exposure.density,
            guide=("tone", "Tone"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            "ISO-R Grade",
            "<b>ISO-R Grade</b> is paper contrast on the ISO-R scale, 50 to 180. A lower R is harder, a "
            "higher R is softer, and about 110 is classic grade 2. Auto Grade meters contrast per frame.",
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.grade_slider,
            task="Drag ISO-R Grade.",
            watch=lambda w: w.state.config.exposure.grade,
            guide=("tone", "Tone"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            "Auto and Set Targets",
            "The <b>Auto</b> menu turns Auto Density and Auto Grade on or off. <b>Set Targets…</b> moves "
            "their aim: print density, contrast, how far each meter is trusted, the metering band, "
            "<b>Shadow Reach</b> and <b>Highlight Hold</b>. Targets are a calibration for every image, "
            "not an edit.",
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.auto_btn,
            guide=("tone", "Tone"),
        ),
        step(
            PRINTING,
            "Test Strip",
            "A <b>test strip</b> prints bands of the frame at stepped exposure and grade, as on an easel. "
            "Click the band you like and its values apply.",
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.test_strip_btn,
            task=f"Press {_k('toggle_test_strip')} to make a test strip.",
            watch=lambda w: w.state.test_strip,
            offer=_DEMO,
        ),
        step(
            PRINTING,
            "Zone Density and Split Grade",
            "<b>Shadows Density</b> and <b>Highlights Density</b> burn or hold one zone and roll into "
            "paper black and white, without a clip. <b>Shadows Grade</b> and <b>Highlights Grade</b> "
            "change one zone's contrast, as a split-grade print does: harder shadows without harsh "
            "highlights.",
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.shadow_density_slider,
            task="Drag one of the four zone sliders.",
            watch=lambda w: tuple(
                getattr(w.state.config.exposure, f) for f in ("shadow_density", "highlight_density", "shadow_grade", "highlight_grade")
            ),
            guide=("tone", "Tone"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            "One Dye Layer",
            "Pick <b>R</b>, <b>G</b> or <b>B</b> and the curve controls act on one dye layer. "
            "Filtration can only shift a layer; these trims change its shape, which removes a cast that "
            "is different in shadows and highlights. A dot marks a trimmed layer.",
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.ch_btn,
            guide=("tone", "Tone"),
        ),
        step(
            PRINTING,
            "Paper Response",
            "A <b>paper profile</b> gives the print the character of a real paper, from its datasheet. "
            "<b>Toe</b> and <b>Shoulder</b> shape the ends of the curve, <b>Snap</b> the midtones, and "
            "<b>Dye Separation</b> the color strength, in density. <b>Paper White</b> and <b>Paper "
            "Black</b> show the real paper base and maximum black.",
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.paper_combo,
            guide=("tone", "Tone"),
        ),
        step(
            PRINTING,
            "Preflash and Contrast Mask",
            "<b>Preflash</b> is a short even exposure before the print: thin highlight detail prints "
            "and the shadows stay clean. <b>Contrast Mask</b> is the darkroom unsharp mask, a soft copy "
            "of the negative that lowers overall contrast and keeps local detail. <b>Mask Spacer</b> "
            "sets its softness.",
            lambda w: cp(w).tone_section,
            focus=lambda w: cp(w).tone_sidebar.preflash_slider,
            guide=("tone", "Tone"),
        ),
        step(
            PRINTING,
            "Filtration",
            "White balance here is CC filtration, as on a color enlarger head. Cyan, Magenta and Yellow "
            f"move the pack; Temperature turns it warm or cool ({_k('temp_warm')} / {_k('temp_cool')}). "
            "The region menu limits it to shadows or highlights. "
            f"{_k('toggle_ring_around')} opens a <b>Ring-around</b> of filter steps.",
            lambda w: cp(w).color_section,
            task="Click Pick WB, then a gray area in the picture, or drag a filter slider.",
            watch=lambda w: (w.state.config.exposure.wb_cyan, w.state.config.exposure.wb_magenta, w.state.config.exposure.wb_yellow),
            guide=("color", "Filtration"),
            offer=_DEMO,
        ),
        step(
            PRINTING,
            "Dodge and Burn",
            "<b>Draw Mask</b> is a cut card, <b>Oval</b> the hole in a card, <b>Card Edge</b> a graded "
            "burn. Burn is in stops; a negative value dodges. Grade prints the area at another contrast. "
            "<b>Tone Limit</b> keeps a mask to one zone, so a sky burn leaves the trees. "
            f"{_k('toggle_printing_notes')} shows the printing map.",
            lambda w: cp(w).local_section,
            task="Draw a mask on the picture with any of the three tools, then drag its Burn or Grade.",
            watch=lambda w: tuple((m.stops, m.grade) for m in w.state.config.local.masks if m.stops or m.grade),
            guide=("local", "Dodge & Burn"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            "Lab",
            "The look after the print. <b>Chroma</b> scales color evenly; <b>Skin Protection</b> keeps "
            "faces natural. <b>Chroma Denoise</b> removes color noise, not grain. <b>Sharpening</b> acts "
            "on lightness only. <b>CLAHE</b> adds local contrast. <b>Glow</b> and <b>Halation</b> copy "
            "lens bloom and the red halo of film.",
            lambda w: cp(w).lab_section,
            task="Drag any Lab slider and watch the print.",
            watch=lambda w: w.state.config.lab,
            guide=("lab", "Lab"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            "Alternative Processes",
            "For B&amp;W negatives. <b>Lith</b> gives creamy warm highlights and sudden hard blacks. "
            "<b>Cyanotype</b> prints in Prussian blue, with bleach and tannin toning. One selector picks "
            "one process, because they exclude each other.",
            lambda w: cp(w).altproc_section,
            guide=("altproc", "Alternative Processes"),
        ),
        step(
            LOOK,
            "Toning",
            "<b>Split Toning</b> moves shadows and highlights toward their own hues and keeps lightness, "
            "so the grain stays. <b>Chemical Toning</b> (B&amp;W) runs six baths on the print's silver, "
            "in the order shown: Selenium, Sepia, Gold, Iron Blue, Copper and Vanadium.",
            lambda w: cp(w).toning_section,
            task="Drag Shadow Strength, then turn Shadow Hue.",
            watch=lambda w: w.state.config.toning,
            guide=("toning", "Toning"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            "Dust Removal",
            "<b>Optical Removal</b> finds dust by local contrast. <b>IR Removal</b> uses the scanner's "
            "infrared channel, where dust shows and the dyes do not. The <b>Overlay</b> menu shows what "
            "each pass marked, so you can set a threshold by eye.",
            lambda w: cp(w).retouch_section,
            focus=lambda w: cp(w).retouch_sidebar.auto_dust_btn,
            guide=("retouch", "Retouch"),
        ),
        step(
            LOOK,
            "Heal, Scratch and Line",
            "<b>Heal</b> marks a search area, not a stamp: only pixels that stand out are rebuilt, with "
            "the frame's own grain. <b>Scratch</b> follows a line you click along a hair. <b>Line</b> "
            "traces a whole straight transport scratch from one click.",
            lambda w: cp(w).retouch_section,
            focus=lambda w: cp(w).retouch_sidebar.pick_dust_btn,
            task="Pick Heal and click a dust speck on the picture.",
            watch=lambda w: w.state.config.retouch,
            guide=("retouch", "Retouch"),
            offer=_DEMO,
        ),
        step(
            LOOK,
            "Finishing",
            "Presentation, after the crop. <b>Vignette</b> is an edge burn in stops. <b>Filed "
            "Carrier</b> prints the black edge of a filed-out carrier, with rough edges and flare. "
            "<b>Border</b> puts a mat around the print; <b>Paper White</b> makes it match the toned "
            "paper.",
            lambda w: cp(w).finish_section,
            task="Drag Vignette Burn, or give the print a Border.",
            watch=lambda w: w.state.config.finish,
            guide=("finish", "Finishing"),
            offer=_DEMO,
        ),
        step(
            SEEING,
            "The Analysis Readout",
            "The densitometer above the tabs. The <b>H&amp;D chart</b> shows the paper curve and where "
            "this frame sits on it. The histograms show output or density, with clip marks. The zone "
            "strip maps the print to Zones 0 to X. <b>Probe</b> reads the density under the cursor.",
            lambda w: rp(w).analysis_section,
            guide=("analysis", "Analysis"),
        ),
        step(
            SEEING,
            "Zone Placement",
            "Place zones as in the Zone System. Pin a point in the picture, give it a target zone, and "
            "<b>Place zones</b> solves Print Density and Grade so the pins print where you want them.",
            lambda w: rp(w).analysis_section,
            guide=("analysis", "Analysis"),
        ),
        step(
            SEEING,
            "Before and After",
            "Split the canvas between this print and the default conversion. Drag the divider to move it.",
            lambda w: w.toolbar.btn_compare,
            task=f"Press {_k('toggle_compare')} for the split. Press it again to close it.",
            watch=lambda w: w.state.compare_mode,
            offer=_DEMO,
        ),
        step(
            SEEING,
            "Peeks, Reference, Loupe and Zones",
            f"Hold {_k('toggle_negative_peek')} for the scan as captured, {_k('toggle_flat_peek')} for "
            f"the flat master, {_k('toggle_embedded_peek')} for the camera's preview. "
            f"{_k('toggle_reference')} pins a frame beside the canvas, to match a roll by eye. "
            f"{_k('toggle_grain_focuser')} is a grain loupe; {_k('toggle_zones')} shows Adams zones.",
            lambda w: w.toolbar.btn_negative_peek,
            task=f"Hold {_k('toggle_negative_peek')} to see the negative.",
            watch=lambda w: w.state.negative_peek,
            offer=_DEMO,
        ),
        step(
            SEEING,
            "Soft Proof",
            "<b>Soft Proof</b> shows the print as a printer and paper will make it: paper white, ink "
            f"black and a gamut warning. {_k('toggle_soft_proof')} turns it on and off. The display "
            "profile keeps the canvas true on your monitor.",
            lambda w: rp(w).export_sidebar._soft_proof_section,
            guide=("soft_proof", "Soft Proof"),
        ),
        step(
            OUTPUT,
            "History and Work Prints",
            "Each edit is a step in <b>History</b>; click one to go back, then edit on to branch. "
            f"<b>Work prints</b> are named versions you keep: {_k('save_work_print')} saves one, to "
            "compare or export later. History stays after a restart.",
            lambda w: rp(w).history_panel,
            pre_hook=lambda w: rp(w).show_tab_by_key("history"),
        ),
        step(
            OUTPUT,
            "Favorites and Presets",
            "<b>Favorites</b> is your own tab: pin the controls you use most. <b>Presets</b> keep chosen "
            "settings and apply them to other frames, over your edit or as a new look.",
            lambda w: rp(w).favourites_sidebar,
            pre_hook=lambda w: rp(w).show_tab_by_key("favourites"),
            guide=("presets", "Presets"),
        ),
        step(
            OUTPUT,
            "Metadata and Gear",
            "Film, camera, lens, developer and scan details go into the exported EXIF and XMP. The "
            "cards follow the roll, as on the Roll tab. <b>Protect Original Metadata</b> leaves the "
            "source file's tags alone. The <b>Gear</b> tab keeps your cameras, lenses and film stocks "
            "for every picker.",
            lambda w: rp(w).metadata_sidebar,
            focus=lambda w: rp(w).metadata_sidebar.protect_btn,
            pre_hook=lambda w: rp(w).show_tab_by_key("metadata"),
            guide=("metadata_gear", "Analog Gear"),
        ),
        step(
            OUTPUT,
            "Roll Settings",
            "Tag a whole roll in one dialog: gear, capture, place, process and scan fields, for this "
            "frame, a selection or the roll. A new roll whose folder name matches your gear, such as "
            "“penf” for “Pen F”, opens this dialog filled in.",
            lambda w: fb(w).roll_settings_btn,
        ),
        step(
            OUTPUT,
            "Output Intent",
            "<b>Print</b> is the finished print. <b>Flat</b> is a flat log master for other editors, "
            "with no print look or effects. <b>Linear</b> is scene-linear data for a raw workflow. "
            f"{_k('toggle_flat_peek')} shows Flat on the canvas.",
            lambda w: rp(w).export_sidebar.intent_btn,
            task="Open the intent menu and choose another intent.",
            watch=lambda w: rp(w).export_sidebar.intent_btn.currentIndex(),
            pre_hook=lambda w: rp(w).show_tab_by_key("export"),
        ),
        step(
            OUTPUT,
            "Export",
            "Pick a format (JPEG, 16-bit TIFF, PNG, WebP, JPEG XL), a color space and a size. The "
            "arrow on the button picks this frame, the selection or all visible frames. <b>Presets</b> "
            "run several recipes in one pass. <b>Contact Sheet</b> prints the roll as film strips on photographic paper, and "
            "<b>Sidecars</b> write each edit to a file beside its scan.",
            lambda w: rp(w).export_sidebar,
            focus=lambda w: rp(w).export_sidebar.export_main_btn,
            pre_hook=lambda w: rp(w).show_tab_by_key("export"),
            guide=("export_presets", "Export Presets"),
        ),
        step(
            SCANNING,
            "Film Scanners",
            "Scan straight into NegPy. <b>Film Scanner</b> drives SANE scanners, the Nikon Coolscan "
            "(with ICE, multi-sampling and Superfine) and Plustek. Prescan a strip, set an offset per "
            "frame, and each scan opens as a frame.",
            lambda w: rp(w).scan_sane_section,
            pre_hook=lambda w: rp(w).show_tab_by_key("scan"),
            guide=("scan_sane", "Film Scanner"),
        ),
        step(
            SCANNING,
            "Camera Scanning",
            "Tether a camera through gphoto2, with live view, and drive a Scanlight for white or red, "
            "green and blue exposures. A Trichrome capture arrives grouped, ready to merge.",
            lambda w: rp(w).scan_rgb_section,
            pre_hook=lambda w: rp(w).show_tab_by_key("scan"),
            guide=("scan_rgb", "Camera Scanning"),
        ),
        step(
            SCANNING,
            "You're All Set",
            f"{_k('show_shortcuts')} lists every shortcut and {_k('command_palette')} finds any control. "
            f"<b>Preferences</b> ({_k('open_preferences')}) holds the interface, performance and storage "
            "settings. Each panel's ⓘ opens its full guide. Replay this tour, one chapter at a time, "
            "from the ⋯ menu.",
            lambda w: None,
            pre_hook=lambda w: rp(w).show_tab_by_key("roll"),
        ),
    ]
