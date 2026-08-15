from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.widgets.tutorial_overlay import TutorialStep

if TYPE_CHECKING:
    from negpy.desktop.view.main_window import MainWindow


def build(window: "MainWindow") -> list[TutorialStep]:
    """Return the ordered list of tutorial steps for *window*."""

    def _process(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.process_sidebar

    def _mode(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.process_sidebar.mode_bar

    def _density(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.density_slider

    def _toe(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.toe_slider

    def _channel_selector(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.ch_global_btn

    def _region_btn(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.color_sidebar.region_global_btn

    def _lab(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.lab_sidebar

    def _retouch(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.retouch_sidebar

    def _export(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.export_sidebar

    def _rgbscan(w: "MainWindow") -> Optional[QWidget]:
        return w.session_panel.file_browser.rgb_scan_btn

    def _half_frame(w: "MainWindow") -> Optional[QWidget]:
        return w.session_panel.file_browser.half_frame_btn

    def _flatfield(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.flatfield_sidebar.enable_btn

    def _crop(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.geometry_sidebar.manual_crop_btn

    def _paper(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.paper_combo

    def _toning(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.toning_sidebar

    def _altproc(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.altproc_sidebar

    def _local(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.local_sidebar.draw_btn

    def _history(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.history_panel.list

    def _flat_master(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.export_sidebar.intent_flat_btn

    def _analysis_buffer(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.process_sidebar.analysis_buffer_slider

    def _crosstalk(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.sensor_sidebar.crosstalk_combo

    def _calibration(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.sensor_sidebar

    def _roll(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.roll_sidebar.analyze_roll_btn

    def _cast_removal(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.color_sidebar.cast_removal_slider

    def _auto_targets(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.targets_btn

    def _split_grade(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.shadow_grade_slider

    def _zone_density(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.shadow_density_slider

    def _gear_manage(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.metadata_sidebar.manage_btn

    def _narrowband(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.sensor_sidebar.narrowband_scan_btn

    def _triage(w: "MainWindow") -> Optional[QWidget]:
        return w.session_panel.file_browser.sheet_btn

    def _dust_overlay(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.retouch_sidebar.overlay_btn

    def _edge_burn(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.finish_sidebar.vignette_burn_slider

    return [
        TutorialStep(
            title="Welcome to NegPy",
            body=(
                "NegPy is a non-destructive RAW film scanner built as a "
                "<b>virtual darkroom</b>. Your scan is treated as a physical measurement "
                "of film transmittance. It is converted to log density, film's native "
                "scale, and printed through a model of real photographic paper "
                "(the H&amp;D curve). This is not a curves-and-levels editor.<br><br>"
                "Edits follow a fixed pipeline:<br><br>"
                "<b>Import → Process → Exposure → Lab → Export</b><br><br>"
                "Everything runs on the GPU for near-instant previews. "
                "All edits are stored in a local database keyed by file hash, so you can "
                "move or rename files freely without losing your work."
            ),
            target=lambda w: None,
        ),
        TutorialStep(
            title="Session Panel: Loading Files",
            body=(
                "Load RAW files or folders here. "
                "The filmstrip lets you flip through your roll quickly. "
                "All loaded files can be batch-processed or batch-exported at once."
            ),
            target=lambda w: w.session_panel,
        ),
        TutorialStep(
            title="RGB Scan: Trichromatic Capture",
            body=(
                "Shot a negative as three separate frames under red, green and blue light? "
                "<b>RGB Scan</b> merges them into one clean, low-noise color scan.<br><br>"
                "Toggle the <b>RGB Scan</b> button in the Files toolbar. Folders are grouped "
                "into triplets automatically, and <b>Edit RGB Triplet…</b> (right-click a frame) "
                "fixes the grouping. Frames are sub-pixel aligned to kill color fringing, then "
                "run through the normal conversion."
            ),
            target=_rgbscan,
        ),
        TutorialStep(
            title="Half Frame: Two Photos per Scan",
            body=(
                "Shooting a half-frame camera, a Pentax 17 or an Olympus Pen? Each scan "
                "holds <b>two photos side by side</b>.<br><br>"
                "Toggle <b>Half Frame</b> in the Files toolbar and every scan appears as "
                "two frames on the contact sheet, split automatically at the gutter "
                "between them. Each half is a full citizen: its own exposure metering, "
                "its own edits and history, its own sidecar, and exports as "
                "<code>name_1</code> / <code>name_2</code>.<br><br>"
                "Toggling off puts the scans back together. Per-half edits are kept and "
                "return when you switch it back on."
            ),
            target=_half_frame,
        ),
        TutorialStep(
            title="Keep & Reject: Culling the Roll",
            body=(
                "Cull the roll where you see it, on the contact sheet. <b>K</b> marks a frame as a "
                "keeper (a small check badge); <b>Shift+X</b> rejects it (a cross badge, and the "
                "thumbnail dims).<br><br>"
                "Rejected frames stay on the sheet, and nothing is deleted or moved, but they drop "
                "out of batch exports and sidecar writes, so a reject cannot sneak into a "
                "delivery.<br><br>"
                "The <b>Sheet</b> menu filters the grid: <b>All</b>, <b>Keepers only</b> or "
                "<b>Hide rejected</b>. A tally beside it counts the roll, and the marks persist "
                "across sessions."
            ),
            target=_triage,
        ),
        TutorialStep(
            title="Flat-Field Correction",
            body=(
                "Corrects uneven illumination, meaning vignetting or falloff from your light "
                "source or scanner, using a reference scan of the bare light.<br><br>"
                "Save named reference profiles, pick the active one, and toggle correction "
                "per image. Off by default."
            ),
            target=_flatfield,
            section_attr="flatfield_section",
        ),
        TutorialStep(
            title="Geometry: Crop & Straighten",
            body=(
                "The unified <b>Crop</b> tool: drag corners to resize, drag inside to move, "
                "click outside to draw a fresh rectangle. <b>Auto</b> detects the film edge, "
                "<b>Fine Rot</b> straightens tilted scans, and <b>Detect Aspect Ratio</b> snaps "
                "to the nearest standard ratio.<br><br>"
                "The <b>Guide</b> dropdown swaps the overlay grid: Thirds, Phi Grid, Diagonals, "
                "Golden Spiral and more (<b>O</b> cycles guides, <b>Shift+O</b> flips "
                "orientation). Four <b>rotation handles</b> just outside the crop box spin "
                "the frame freehand (±45°), composing with Fine Rot for fine-tuning.<br><br>"
                "Crop matters for more than framing, because the conversion <b>meters what is "
                "inside the crop</b> to find the black and white points. Unexposed rebate sits at "
                "film-base density, a false brightest highlight, while sprocket holes and scanner "
                "bed sit at the opposite extreme. None of it is picture. Left in frame, it drags "
                "the detected bounds, giving milky blacks and a wrong mask estimate.<br><br>"
                "Crop tight to the image, or use the <b>Analysis Buffer</b> (next) when you "
                "want to keep a border.<br><br>"
                "<b>Batch Autocrop</b> does the whole roll at once. It analyses every visible "
                "landscape frame together, letting the confident detections calibrate the weak "
                "ones, so camera-scan crops come out consistent instead of frame-by-frame. It "
                "runs in the background with progress and cancel, and leaves your manual crops "
                "alone. Available in Image-only autocrop mode."
            ),
            target=_crop,
            section_attr="geometry_section",
        ),
        TutorialStep(
            title="Analysis Buffer: Keep the Meter on the Image",
            body=(
                "Insets the metering window from the frame edge, up to 25% per side, so the "
                "bounds analysis reads <b>only the picture</b>.<br><br>"
                "The meter is statistical. It cannot tell film rebate, sprocket holes or holder "
                "from scene tones, and densities that never occurred in the scene skew the "
                "percentile black and white points.<br><br>"
                "Rule of thumb: the analysis area should contain image and nothing else. Use "
                "the buffer when you deliberately keep a border in frame; a tight crop is the "
                "cleaner fix.<br><br>"
                "For odd frames, the <b>draw region</b> tool beside it goes further. Draw the "
                "metering area freehand on the canvas and the meter reads exactly that, with "
                "no centered inset."
            ),
            target=_analysis_buffer,
            section_attr="process_section",
        ),
        TutorialStep(
            title="Film Mode & Decoding",
            body=(
                "The first choice of every edit sits above the panels: <b>what kind of film "
                "this is</b>.<br><br>"
                "The three buttons pick the chemistry, <b>Color</b> (C-41 negative), "
                "<b>B&amp;W</b> (panchromatic negative) or <b>Slide</b> (transparency/reversal). "
                "Each swaps the core conversion math and re-runs the pipeline from scratch, and "
                "the wand beside them <b>auto-detects</b> the mode when a file loads.<br><br>"
                "In <b>Calibration</b> below, <b>Linear RAW</b> decodes with neutral multipliers, "
                "bypassing the camera's as-shot white balance so the orange mask arrives untouched. "
                "Toggling it reloads the file. Off (the default) decodes with the as-shot balance "
                "instead, which is what a camera scan under white light wants.<br><br>"
                "You do not have to guess. The <b>bulb</b> button asks two questions about your "
                "rig, camera or scanner and white light or narrowband RGB, then sets Linear RAW "
                "and Narrowband from the answer. It opens by itself once this tour is done.<br><br>"
                "In the <b>Normalization</b> panel, <b>Lock Bounds</b> freezes this frame's analysed "
                "bounds, so cropping or moving a slider no longer re-meters it. Lock in once the "
                "conversion looks right. In Slide mode a <b>Normalize</b> button appears at the "
                "bottom, stretching a faded or expired slide back to the full range."
            ),
            target=_mode,
        ),
        TutorialStep(
            title="Normalization Panel: Bounds Analysis",
            body=(
                "Film dyes follow Beer–Lambert absorption, so density is logarithmic. NegPy "
                "converts the raw signal to log space and meters it there, on two independent "
                "axes: a <b>luma</b> pass sets the black/white-point span, and a per-channel "
                "<b>color</b> pass <b>measures the orange mask from the actual negative</b>, "
                "with no hardcoded mask constants.<br><br>"
                "<b>Luma Range Clip</b> tunes the tonal span. Neutral already applies a small "
                "robust clip, positive tightens it (good for dense or fogged negatives where a "
                "few stray pixels drag the bounds to extremes), and negative pushes them outward "
                "for lifted blacks and unclipped highlights. <b>Color Clip</b> does the same "
                "for the per-channel balance, the orange-mask removal, independently of the "
                "tonal range.<br><br>"
                "<b>White Point</b> / <b>Black Point</b> fine-tune the detected bounds without "
                "re-analysis, for highlight recovery or shadow crush. Their <b>Global / R / G / "
                "B</b> selector scopes them: Global moves both bounds together, while R, G and "
                "B become per-dye-layer film-base (Dmin) and Dmax trims, like a scanner's "
                "per-channel levels. That is the tool for a mask that reads slightly off in one "
                "layer. Hidden in B&amp;W Negative.<br><br>"
                "The stretch is <b>unclamped</b>: tones outside the bounds survive and roll "
                "off later in the print curve's toe and shoulder."
            ),
            target=_process,
            section_attr="process_section",
        ),
        TutorialStep(
            title="Calibration: What Your Rig Does to the Colors",
            body=(
                "Everything in this panel corrects the <b>capture</b>, not the look. Three "
                "separate things sit between the scene and your file, and each gets its own "
                "control: the camera's color filters, the film's dyes, and the light source. "
                "They are not interchangeable, and none substitutes for another.<br><br>"
                "<b>Capture</b> decides how the file is decoded. You do not have to work it out "
                "yourself: the <b>bulb</b> button runs a two-question wizard, how you scan and "
                "what light you use, and sets <b>Linear RAW</b> and <b>Narrowband</b> from the "
                "answer.<br><br>"
                "<b>Trichrome Calibration</b> is for single-shot narrowband camera scans. Your "
                "camera's color filters overlap the light's bands, so a pure red exposure leaks "
                "into green and blue. That leak belongs to your sensor and light together and has "
                "nothing to do with the film, so it is corrected on the linear capture before "
                "inversion. Build a <b>Profile</b> from three bare-light R/G/B exposures with the "
                "calibrate button. It needs Linear RAW on.<br><br>"
                "<b>Crosstalk</b> handles the film's own dye absorptions, in negative density. It "
                "has its own step in a moment.<br><br>"
                "<b>Hue Trim</b> is for the light source. An unusual lamp turns <i>every</i> color "
                "by roughly the same angle, so yellows read orange and greens go olive while "
                "neutrals stay put. White balance cannot fix that, because a rotation is not a "
                "cast and there is no grey to correct. Judge it on something whose color you know, "
                "and leave it at 0 for an ordinary broadband light. It is sticky, since a light "
                "source is a property of your rig.<br><br>"
                "Narrowband and Trichrome Calibration are greyed out on <b>Transparency</b>: both "
                "describe negative dyes, which a slide does not have. Hue Trim still applies."
            ),
            target=_calibration,
            section_attr="sensor_section",
        ),
        TutorialStep(
            title="Narrowband Scan: Correcting LED Light",
            body=(
                "A trichrome scan lit by <b>narrowband RGB LEDs</b>, a Scanlight or most RGB-LED "
                "sources, hits each dye layer with a much purer band than white light does. "
                "The layers separate further than the film intends, and the conversion can come "
                "out over-saturated.<br><br>"
                "The <b>Narrowband Scan</b> toggle corrects for that light source. It applies "
                "to the preview <i>and</i> every export, so what you judge is what you "
                "deliver.<br><br>"
                "Turning on <b>RGB Scan</b> mode switches it on for you, on the current frame "
                "and as the default for new ones. If you have set a custom <b>Input ICC</b> "
                "profile, that takes precedence and this toggle steps aside."
            ),
            target=_narrowband,
            section_attr="sensor_section",
        ),
        TutorialStep(
            title="Crosstalk: Dye Unmixing",
            body=(
                "Each film dye layer also absorbs outside its own band. These <b>secondary "
                "absorptions</b> leak one channel into another and mute color. They are "
                "linear in negative dye density (Beer–Lambert), so NegPy unmixes them "
                "with a per-stock matrix in log-density space, <b>before any analysis</b>.<br><br>"
                "Pick a profile matching your film stock and blend it in with the "
                "<b>Strength</b> slider.<br><br>"
                "Changed the matrix or strength? <b>Re-run Batch Analysis</b>, because bounds "
                "measured under a different matrix are invalid."
            ),
            target=_crosstalk,
            section_attr="sensor_section",
        ),
        TutorialStep(
            title="Roll Consistency: Batch Analysis",
            body=(
                "One enlarger setting for the whole roll. <b>Batch Analysis</b> meters every "
                "loaded frame and builds a roll-wide baseline, then two buttons lock frames "
                "to it on independent axes: <b>Use Luma Average</b> takes the roll-wide tonal "
                "range, <b>Use Color Average</b> takes the roll-wide color balance. Turn on "
                "either, or both, so exposure and color do not jump from frame to "
                "frame.<br><br>"
                "Roll presets save and load the baseline for later sessions. A locked "
                "baseline is also what keeps <b>Flat masters</b> consistent across a roll."
            ),
            target=_roll,
            section_attr="roll_section",
        ),
        TutorialStep(
            title="Exposure: Density & Grade",
            body=(
                "<b>Density</b> slides the negative's log exposure along the paper curve, "
                "exactly like enlarger exposure time. Lower values print brighter.<br><br>"
                "<b>Grade</b> sets contrast on the photographic <b>ISO-R paper scale</b> "
                "(50–180, default 115), which is the range of log exposure the paper accepts. "
                "Lower R is harder (more contrast and punch), higher R is softer, and R110 is "
                "roughly classic paper grade 2. The resulting slope is the literal H&amp;D gamma: "
                "negative density range over paper exposure range.<br><br>"
                "<b>Dye Separation</b> pushes the print's three dye densities apart in the "
                "same matrix the paper's own dye crosstalk uses. It works in density space "
                "rather than as a post-hoc color boost, so it stays in step with the curve and "
                "takes per-layer R/G/B trims for crossover. 1.0 is off.<br><br>"
                "<b>Auto Density</b> and <b>Auto Grade</b> meter each frame for sensible "
                "brightness and contrast out of the box. They correct only <i>partially</i>, "
                "so low-key and high-key shots keep their mood. Turn them off to let the "
                "conversion follow the negative honestly."
            ),
            target=_density,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Set Targets: Retune the Autos",
            body=(
                "Auto Density and Auto Grade aim at a fixed idea of a good print. Your scanner, "
                "your film and your taste may disagree, and <b>Set Targets</b> moves the aim.<br><br>"
                "<b>Print Density Target</b> is how bright the metered midtone prints; "
                "<b>Contrast Target</b> is the printed contrast every frame is aimed at. The two "
                "strength sliders decide how far each meter is trusted: at 0 you get a fixed "
                "setting for every frame, at 1 every frame is forced to the same key or the same "
                "contrast. <b>Metering Band</b> caps how far Auto Density may swing between "
                "frames.<br><br>"
                "These are a <b>calibration, not an edit</b>. They apply to every image, "
                "including ones you have already worked on, and are remembered between sessions. "
                "The preview follows the sliders live, <b>Cancel</b> puts them back, and "
                "<b>Restore Defaults</b> returns to the shipped values."
            ),
            target=_auto_targets,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Exposure: H&D Curve (Toe & Shoulder)",
            body=(
                "The <b>Toe</b> and <b>Shoulder</b> controls shape the shadow and highlight roll-off "
                "of the H&D characteristic curve, a model of how photographic paper responds to "
                "light rather than a generic tone curve.<br><br>"
                "<b>Toe</b>: lifts the paper-black ceiling. Positive gives film's gentle "
                "shadow toe; negative deepens it.<br>"
                "<b>Shoulder</b>: compresses highlights toward paper white. Negative extends "
                "them instead, and can clip.<br>"
                "<b>Width</b>: how far each knee's roll-off reaches up (Toe) or down "
                "(Shoulder) the tonal scale. Each knee has its own.<br>"
                "<b>Snap</b>: the paper's variable midtone gamma. Endpoints and anchor stay put.<br><br>"
                "Two toggles set where the print's ends actually land. <b>Paper White</b> "
                "simulates paper base density, so whites print at about 0.93 rather than pure "
                "white, like a real print on real stock. <b>Paper Black</b> shows the paper's real "
                "D-max as a lifted, slightly milky black. Leave it off (the default) for black "
                "point compensation, in effect an ICC relative-colorimetric soft-proof, which "
                "maps D-max to display black so the adapted eye reads black as black. With it "
                "off, pull Toe negative to clip deep shadows to exact black.<br><br>"
                "All of these sit under the <b>Paper Response</b> header, below the paper "
                "profile: the deeper print-curve controls, grouped away from the everyday "
                "Density and Grade above."
            ),
            target=_toe,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Split Grade: Zone Contrast",
            body=(
                "Split-grade printing. <b>Shadows Grade</b> and <b>Highlights Grade</b> trim "
                "each zone's contrast in ISO-R points on top of the main Grade, giving harder "
                "shadows without blowing the highlights, or softer highlights without "
                "flattening the shadows, like a second enlarger exposure through a different "
                "filter.<br><br>"
                "Both trims spare the midtones and stay bounded by the paper's black and "
                "white, and they scope per color layer through the <b>Global / R / G / B</b> "
                "selector like the main Grade."
            ),
            target=_split_grade,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Zone Density: Shadows & Highlights",
            body=(
                "Where Split Grade is zone <i>contrast</i>, these are zone <i>brightness</i>. "
                "<b>Shadows Density</b> and <b>Highlights Density</b> darken or brighten each "
                "zone while rolling into the paper's black and white limits instead of "
                "clipping, so you can burn in a sky without blocking it up.<br><br>"
                "They sit right under <b>Print Density</b>, as the pair you reach for after "
                "setting overall brightness. Shadows Density has the wider travel of the two, "
                "since shadow burns are what a print usually needs. Both stay <b>global</b>: "
                "they are print exposure, not curve shape, so they do not split per dye layer."
            ),
            target=_zone_density,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Per-Layer Trims: Crossover Correction",
            body=(
                "The <b>Global / Red / Green / Blue</b> selector scopes the curve controls to a "
                "single dye layer. Pick R, G or B and Grade, the Split Grades, Toe, Shoulder, "
                "the Widths and Snap all retarget to that layer, their labels gaining an "
                "<b>R</b>, <b>G</b> or <b>B</b>. Grade and the Widths swap to dedicated trim "
                "sliders centred on zero, because you are nudging that layer away from the shared "
                "curve, not setting it from scratch.<br><br>"
                "Color filtration can only <i>shift</i> a layer's curve; trims change its "
                "<i>shape</i>. That fixes crossover casts that differ between shadows, mids and "
                "highlights, the correction a real color darkroom never had. The H&D chart "
                "draws the diverged per-layer curves live.<br><br>"
                "What is not per-layer greys out while a channel is selected: Print Density, "
                "the zone densities, the autos and the paper toggles are properties of the "
                "print, not of one emulsion. A dot on a channel button marks a layer you have "
                "already trimmed, so casts you fixed weeks ago stay findable. The whole "
                "selector disappears in B&amp;W Negative, which has one emulsion and one curve."
            ),
            target=_channel_selector,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Exposure: Filtration",
            body=(
                "White balance is real CC filtration: ±1.0 on a slider is ±20cc of dichroic "
                "density. The <b>Global / Shadows / Highlights</b> buttons on top scope the "
                "CMY sliders to a region for precise split-toning control.<br><br>"
                "The <b>Temperature</b> slider re-dials the filter pack along the warm-to-cool "
                "axis. Magenta and Yellow move together in the right ratio while your "
                "green-magenta tint stays put. Travel is mired-linear (equal drag, equal "
                "perceived shift), <b>T</b>/<b>G</b> nudge it, and the thermometer button "
                "locks the temperature for the whole roll.<br><br>"
                "<b>Pick WB</b>: click a neutral area in the preview and the filtration is "
                "calculated for you."
            ),
            target=_region_btn,
            section_attr="color_section",
        ),
        TutorialStep(
            title="Cast Removal: Neutral Greys End to End",
            body=(
                "A negative's color cast is not constant. It varies with density, so a "
                "midtone-only white balance leaves shadows and highlights drifting "
                "off-color.<br><br>"
                "<b>Cast Removal</b> measures each channel's deep-shadow reference and gives "
                "it its own slope, pivoting on the midtone, so greys read neutral from deep "
                "shadows through highlights rather than at one point only.<br><br>"
                "Its strength adapts per frame to how confidently the neutral greys read. "
                "Clean greys get the full correction and few-neutral scenes get a gentler one, "
                "and the slider (default 0.5) trims on top. 0 turns it off."
            ),
            target=_cast_removal,
            section_attr="color_section",
        ),
        TutorialStep(
            title="Exposure: Paper Profiles",
            body=(
                "A <b>paper profile</b> sets the print character, the H&D curve shape, "
                "without touching contrast or exposure. Each profile carries its paper's "
                "tone, per-channel gamma and base tint, mapped from Ilford / Kodak / Foma / "
                "Fuji datasheets.<br><br>"
                "Profiles are mode-aware (RA4 color papers in Color Negative, tonal papers in "
                "B&W Negative) and sticky roll-wide. The dropdown steps aside entirely in "
                "Transparency, where a slide is the final image and no paper is involved. "
                "<b>Neutral</b> reproduces the defaults exactly, and Grade and Density still "
                "trim on top."
            ),
            target=_paper,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Dodge & Burn",
            body=(
                "Darkroom-style local lighten and darken, with one shape per darkroom move. "
                "<b>Draw Mask</b> is the cut card: click to drop vertices, double-click to close. "
                "<b>Oval</b> is the hole in the card, stretched and tilted by its two axis "
                "handles. <b>Card Edge</b> is the graduated burn, dragged from the edge that "
                "gets full exposure to where it fades out.<br><br>"
                "<b>Burn</b> sets each mask's print exposure in <b>stops</b>, signed like the "
                "rest of NegPy: positive burns, negative dodges. <b>Feather</b> softens the edge "
                "(not on a Card Edge, where the handle spacing is the softness), <b>Invert</b> "
                "acts everywhere except inside the mask, and <b>Grade</b> prints that region "
                "through a different contrast, the hard-filter burn of a real darkroom.<br><br>"
                "Masks change the <b>print exposure</b> ahead of the paper curve, so burns roll "
                "into paper black through the toe and dodges lift toward paper white, like "
                "holding back light under the enlarger. Masks are stored in raw-image space, so "
                "they survive rotation, flip and crop."
            ),
            target=_local,
            section_attr="local_section",
        ),
        TutorialStep(
            title="Lab Panel: Film Aesthetics",
            body=(
                "<b>Color:</b> "
                "<b>Chroma</b> scales color evenly across every tone, the flat, post-decode "
                "counterpart to <b>Dye Separation</b> over in Tone, which works on the print's dye "
                "densities instead. "
                "<b>Skin Protection</b> holds skin-hued color under a chroma ceiling so faces "
                "do not go sunburnt, and works even at Chroma 1.0. "
                "<b>Chroma Denoise</b> smooths chroma noise in Lab space without touching "
                "luminance grain.<br><br>"
                "<b>Sharpen:</b> "
                "<b>Method</b> picks <b>Unsharp Mask</b> or <b>Deconvolution</b> "
                "(Richardson-Lucy, which reverses the scanner's optical blur). Both run on the "
                "L channel, so there are no color halos, and share <b>Sharpening</b>, "
                "<b>Radius</b> and <b>Masking</b>.<br><br>"
                "<b>Detail:</b> "
                "<b>CLAHE</b> applies local contrast enhancement that lifts midtone detail "
                "without blowing highlights.<br><br>"
                "<b>Effects:</b> "
                "<b>Glow</b> simulates lens bloom. "
                "<b>Halation</b> mimics red scatter caused by light bouncing back through the "
                "film base, strongly red-dominant, exactly like real film halation."
            ),
            target=_lab,
            section_attr="lab_section",
        ),
        TutorialStep(
            title="Alternative Processes: Lith & Cyanotype",
            body=(
                "Two printing processes that are not ordinary silver-gelatin enlarging. They are "
                "mutually exclusive, so the <b>None / Lith / Cyanotype</b> buttons pick one and "
                "only that process's controls appear. Both are <b>B&amp;W Negative only</b> and "
                "off by default.<br><br>"
                "<b>Lith</b> is the darkroom trick of massively over-exposing a lith-capable "
                "paper, developing in dilute low-sulphite developer and pulling the print "
                "part-way through: creamy warm highlights, then an abrupt drop into hard, sooty "
                "blacks. <b>Exposure</b> is the over-exposure in stops, <b>Snatch Point</b> is how "
                "long the print stays in the developer before you pull it, and <b>Abruptness</b> "
                "is how suddenly the shadows go black. There is no color control, because the "
                "paper you picked in the Exposure panel sets the whole path, from peach highlights "
                "through an olive transition to neutral blacks.<br><br>"
                "<b>Cyanotype</b> is contact-printed in UV onto paper brushed with iron salts. "
                "There is no silver and no development to time: the image is Prussian blue, so "
                "the print never goes black, it goes blue. <b>Sensitiser</b> picks the classic "
                "Herschel mix or Ware's deeper modern formula, <b>Exposure</b> is time under UV, "
                "and <b>Exposure Scale</b> is the density range the sensitiser can hold, which is "
                "the contrast control. <b>Bleach</b> strips pigment highlights-first, and "
                "<b>Tannin</b> re-develops the bleached iron brown, so a partial bleach leaves a "
                "split blue-brown.<br><br>"
                "Either one changes the Toning panel below, since the toners act on the "
                "alternative print rather than the other way round."
            ),
            target=_altproc,
            section_attr="altproc_section",
        ),
        TutorialStep(
            title="Toning",
            body=(
                "<b>Split Toning</b> (all modes) pushes shadows and highlights toward independent "
                "hue angles with their own strength. It works in Lab space, so luminance, and "
                "with it grain and detail, is preserved exactly.<br><br>"
                "<b>Chemical Toning</b> (B&W Negative only) simulates six baths on the print's "
                "silver density, applied in the order shown. <b>Selenium</b> converts the densest "
                "silver first, for deeper blacks and cool eggplant shadows. <b>Sepia</b> "
                "bleach-redevelops the thinnest silver first, for warm highlights that hold the "
                "shadows, and partial strength gives the classic split-sepia look. <b>Gold</b> is "
                "the archival bath: cool blue-black on untoned silver, but run over Sepia it "
                "pushes the toned highlights from yellow-brown toward orange-red.<br><br>"
                "<b>Iron Blue</b> gives Prussian-blue shadows deepening to navy, <b>Copper</b> a "
                "pink-to-brick shift with the classic Dmax loss, and <b>Vanadium</b> greens the "
                "mids and highlights while the deep shadows keep their black.<br><br>"
                "An alternative process changes what is available: with <b>Lith</b> on, only "
                "Selenium and Gold stay enabled, and both bite harder on lith's fine silver. "
                "With <b>Cyanotype</b> on, all six grey out, because a cyanotype holds no silver "
                "for a bath to react with. Split Toning keeps working in every case."
            ),
            target=_toning,
            section_attr="toning_section",
        ),
        TutorialStep(
            title="Retouch Panel: Dust Removal",
            body=(
                "<b>Optical Removal</b> detects and removes small particles on the visible scan by "
                "local contrast. Lower the threshold to be more aggressive.<br><br>"
                "<b>IR Removal</b> works from the scanner's infrared channel, where dust blocks "
                "light but the color dyes do not, catching what the eye cannot separate from "
                "grain. Detection is ratio-normalized rather than a raw-IR threshold, so the "
                "slider responds smoothly instead of flipping the whole frame at a cliff. "
                "<b>Method</b> picks how the film underneath is rebuilt: <b>NegPy</b> divides "
                "semi-transparent dust back out and fills the opaque cores, while <b>OpenICE</b> "
                "works in log density and restores detail rather than averaging it away. "
                "B&amp;W and Kodachrome scans are skipped.<br><br>"
                "However a mark is found, the repair is the same: the film under it is rebuilt "
                "from the clean film around it, with the frame's own grain transplanted back.<br><br>"
                "<b>Heal Tool</b>: click or drag over dust spots. The brush marks a <i>search "
                "area</i>, not a stamp, so only the pixels that stand out from the film around "
                "them are rewritten and clean grain under a generous brush is left alone.<br><br>"
                "<b>Scratch Tool</b>: click a polyline along a hair or scratch, then double-click "
                "or press <b>Enter</b> to commit.<br><br>"
                "<b>Transport Line</b>: for the long straight marks a roll picks up in a camera "
                "or lab. Click once anywhere on the scratch and the whole line is traced and "
                "repaired. <b>Line Sensitivity</b> tunes how readily one is followed.<br><br>"
                "<b>Brush Size</b> sets the manual brush, and <b>Undo Last</b> / <b>Clear All</b> "
                "manage the spots."
            ),
            target=_retouch,
            section_attr="retouch_section",
        ),
        TutorialStep(
            title="Dust Overlay: See What Is Detected",
            body=(
                "Dust thresholds are hard to set blind. The <b>Overlay</b> button cycles the "
                "detection inspector so you can tune by eye: <b>Off → Marked → IR</b>. The IR "
                "state appears only on scans that have an infrared channel.<br><br>"
                "<b>Marked</b> paints every spot the detector is about to fix; <b>IR</b> shows "
                "the raw infrared read behind it. Turn Optical or IR Removal on first, because "
                "the overlay draws what those passes found, so with both off there is nothing to "
                "show.<br><br>"
                "Watch the overlay while you drag a threshold: too aggressive lights up grain, "
                "too conservative leaves specks unmarked."
            ),
            target=_dust_overlay,
            section_attr="retouch_section",
        ),
        TutorialStep(
            title="Finish: Edge Burn, Carrier & Mats",
            body=(
                "Print-presentation touches, applied at the very end of the pipeline, after "
                "the crop, on the finished print.<br><br>"
                "<b>Edge Burn</b> is a real exposure burn measured in <b>stops</b> rather than a "
                "darkening overlay: the darkroom printer's edge burn that holds the eye inside "
                "the frame. <b>Size</b> sets how far in it reaches, and <b>Roundness</b> morphs "
                "it from a radial falloff to a straight-edged card burn.<br><br>"
                "<b>Filed Carrier</b> prints the black rebate of a filed-out negative carrier. "
                "<b>Width</b> sets the frame (0 = off), <b>Roughness</b> breaks up its filed edge "
                "the way a real carrier looks, <b>Flare</b> adds the glow off the bared metal of "
                "the bevel, and <b>Corners</b> rounds the aperture, since no file cuts a sharp "
                "inside corner.<br><br>"
                "<b>Border</b> lays a mat around the print: <b>Width</b>, plus <b>Bottom "
                "weight</b> for the window-mat proportion where the bottom margin runs deeper. "
                "Pick its color from the swatch, or turn on <b>Paper white</b> to tie the mat "
                "to the toned paper white so it matches the print instead of fighting it."
            ),
            target=_edge_burn,
            section_attr="finish_section",
        ),
        TutorialStep(
            title="History",
            body=(
                "The <b>History</b> tab lists every edit step for the current photo. Click any "
                "step to jump back to that state, the preview updates instantly, then carry on "
                "editing from there to branch.<br><br>"
                "Right-click a step to <b>Export this version</b>. Up to 100 steps per file, and "
                "the history survives restarts."
            ),
            target=_history,
            pre_hook=lambda w: w.right_panel.show_tab_by_key("history"),
        ),
        TutorialStep(
            title="Metadata & Gear Library",
            body=(
                "The <b>Metadata</b> tab writes film and scan info, meaning stock, format, "
                "developer, push/pull and scanner, into the EXIF/XMP of exported files.<br><br>"
                "<b>Manage…</b> opens the <b>Gear Library</b>: a searchable, user-extendable "
                "library of cameras, lenses and film stocks. Gear picked for a frame rides "
                "into the exported XMP.<br><br>"
                "<b>Protect original metadata</b> keeps the source file's EXIF/XMP untouched "
                "instead of NegPy rewriting it."
            ),
            target=_gear_manage,
            pre_hook=lambda w: w.right_panel.show_tab_by_key("metadata"),
        ),
        TutorialStep(
            title="Export",
            body=(
                "The <b>Export</b> tab (right panel, now active) is where you save your "
                "results.<br><br>"
                "Choose a format (<b>JPEG</b>, high-bit-depth <b>TIFF</b>, PNG, WebP, JPEG XL), "
                "pick a color space, and set resolution or print size. The <b>ICC</b> section adds "
                "monitor-profile display and soft-proofing.<br><br>"
                "The <b>Export</b> and <b>Export Presets</b> buttons are triggers. Each button's "
                "menu arrow picks what it exports (current frame, selected frames, or all visible "
                "frames) and remembers the choice. Presets run every enabled preset per frame. "
                "<b>Contact Sheet</b> renders all frames into one sheet. "
                "Export always runs at full RAW resolution."
            ),
            target=_export,
            pre_hook=lambda w: w.right_panel.show_tab_by_key("export"),
        ),
        TutorialStep(
            title="Export: Flat Master",
            body=(
                "The <b>Flat</b> output intent exports a flat, neutral, low-contrast "
                "digital-intermediate master for Lightroom / Darktable / Photoshop: a "
                "high-bit-depth TIFF, or lossless JPEG XL where the chosen color space can be "
                "tagged.<br><br>"
                "It replaces the print curve with a true log encoding and skips the creative "
                "stages, so there is no print look, no effects, no toning and no vignette. The "
                "color space follows your export selection. <b>Preview Flat</b> peeks at the "
                "master on the canvas, also on the toolbar and on <b>|</b>, and "
                "<b>Roll Baseline</b> keeps flat masters consistent across a roll. Standard "
                "<b>Print</b> output is unaffected."
            ),
            target=_flat_master,
            pre_hook=lambda w: w.right_panel.show_tab_by_key("export"),
        ),
        TutorialStep(
            title="You're all set!",
            body=(
                "That is the core workflow. A few more things worth knowing:<br><br>"
                "• Press <b>?</b> or use the ⋯ menu for keyboard shortcuts.<br>"
                "• Canvas tools share one grammar: the first <b>Esc</b> clears the points "
                "you are placing, the second puts the tool down. <b>Shift+S</b> Scratch, "
                "<b>Shift+B</b> Dodge &amp; Burn, <b>Shift+R</b> Analysis Region, "
                "<b>|</b> flat-master peek.<br>"
                "• Scanning with a tethered camera? The <b>Camera Scanning</b> section on the "
                "Scan tab drives the body and Scanlight directly (macOS/Linux). See "
                "<code>docs/CAMERA_SCANNING.md</code>.<br>"
                "• See <code>docs/USER_GUIDE.md</code> for the full reference.<br>"
                "• Having GPU or rendering issues? Edit "
                "<code>Documents/NegPy/override.toml</code> to switch backends "
                "without touching code.<br>"
                "• Edits auto-save to a local database, so there is no manual save needed "
                "between files."
            ),
            target=lambda w: None,
            pre_hook=lambda w: w.right_panel.show_tab_by_key("setup"),
        ),
    ]
