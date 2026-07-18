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

    def _density(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.density_slider

    def _toe(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.toe_slider

    def _channel_selector(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.ch_global_btn

    def _region_btn(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.colour_sidebar.region_global_btn

    def _lab(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.lab_sidebar

    def _retouch(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.retouch_sidebar

    def _export(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.export_sidebar

    def _rgbscan(w: "MainWindow") -> Optional[QWidget]:
        return w.session_panel.file_browser.rgb_scan_btn

    def _flatfield(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.flatfield_sidebar.enable_btn

    def _crop(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.geometry_sidebar.manual_crop_btn

    def _paper(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.paper_combo

    def _toning(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.toning_sidebar

    def _local(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.local_sidebar.draw_btn

    def _history(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.history_panel.list

    def _flat_master(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.export_sidebar.intent_flat_btn

    def _analysis_buffer(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.process_sidebar.analysis_buffer_slider

    def _crosstalk(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.process_sidebar.crosstalk_combo

    def _roll(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.roll_sidebar.analyze_roll_btn

    def _cast_removal(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.colour_sidebar.cast_removal_slider

    def _auto_targets(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.targets_btn

    def _split_grade(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.shadow_grade_slider

    def _zone_density(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.tone_sidebar.shadow_density_slider

    def _gear_manage(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.metadata_sidebar.manage_btn

    def _narrowband(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.process_sidebar.narrowband_scan_btn

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
                "of film transmittance: it's converted to log density — film's native "
                "scale — and printed through a model of real photographic paper "
                "(the H&amp;D curve). Not a curves-and-levels editor.<br><br>"
                "Edits follow a fixed pipeline:<br><br>"
                "<b>Import → Process → Exposure → Lab → Export</b><br><br>"
                "Everything runs on the GPU for near-instant previews. "
                "All edits are stored in a local database keyed by file hash — "
                "move or rename files freely without losing your work."
            ),
            target=lambda w: None,
        ),
        TutorialStep(
            title="Session Panel — Loading Files",
            body=(
                "Load RAW files or folders here. "
                "The filmstrip lets you flip through your roll quickly. "
                "All loaded files can be batch-processed or batch-exported at once."
            ),
            target=lambda w: w.session_panel,
        ),
        TutorialStep(
            title="RGB Scan — Trichromatic Capture",
            body=(
                "Shot a negative as three separate frames under red, green and blue light? "
                "<b>RGB Scan</b> merges them into one clean, low-noise colour scan.<br><br>"
                "Toggle the <b>RGB Scan</b> button in the Files toolbar — folders are grouped "
                "into triplets automatically, and <b>Edit RGB Triplet…</b> (right-click a frame) "
                "fixes the grouping. Frames are sub-pixel aligned to kill colour fringing, then "
                "run through the normal conversion."
            ),
            target=_rgbscan,
        ),
        TutorialStep(
            title="Keep & Reject — Culling the Roll",
            body=(
                "Cull the roll where you see it, on the contact sheet. <b>K</b> marks a frame as a "
                "keeper (a small check badge); <b>Shift+X</b> rejects it (a cross badge, and the "
                "thumbnail dims).<br><br>"
                "Rejected frames stay on the sheet — nothing is deleted or moved — but they drop "
                "out of batch exports and sidecar writes, so a reject can't sneak into a "
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
                "Corrects uneven illumination — vignetting or falloff from your light source "
                "or scanner — using a reference scan of the bare light.<br><br>"
                "Save named reference profiles, pick the active one, and toggle correction "
                "per image. Off by default."
            ),
            target=_flatfield,
            section_attr="flatfield_section",
        ),
        TutorialStep(
            title="Geometry — Crop & Straighten",
            body=(
                "The unified <b>Crop</b> tool: drag corners to resize, drag inside to move, "
                "click outside to draw a fresh rectangle. <b>Auto</b> detects the film edge, "
                "<b>Fine Rot</b> straightens tilted scans, and <b>Detect Aspect Ratio</b> snaps "
                "to the nearest standard ratio.<br><br>"
                "The <b>Guide</b> dropdown swaps the overlay grid — Thirds, Phi Grid, Diagonals, "
                "Golden Spiral and more (<b>O</b> cycles guides, <b>Shift+O</b> flips "
                "orientation) — and four <b>rotation handles</b> just outside the crop box spin "
                "the frame freehand (±45°), composing with Fine Rot for fine-tuning.<br><br>"
                "Crop matters for more than framing: the conversion <b>meters what's inside "
                "the crop</b> to find the black and white points. Unexposed rebate sits at "
                "film-base density (a false brightest highlight); sprocket holes and scanner "
                "bed at the opposite extreme. None of it is picture — left in frame, it drags "
                "the detected bounds, giving milky blacks and a wrong mask estimate.<br><br>"
                "Crop tight to the image, or use the <b>Analysis Buffer</b> (next) when you "
                "want to keep a border.<br><br>"
                "<b>Batch Autocrop</b> does the whole roll at once: it analyses every visible "
                "landscape frame together, letting the confident detections calibrate the weak "
                "ones so camera-scan crops come out consistent instead of frame-by-frame. It "
                "runs in the background with progress and cancel, and leaves your manual crops "
                "alone. Available in Image-only autocrop mode."
            ),
            target=_crop,
            section_attr="geometry_section",
        ),
        TutorialStep(
            title="Analysis Buffer — Keep the Meter on the Image",
            body=(
                "Insets the metering window from the frame edge — up to 25% per side — so the "
                "bounds analysis reads <b>only the picture</b>.<br><br>"
                "The meter is statistical: it can't tell film rebate, sprocket holes or holder "
                "from scene tones, and densities that never occurred in the scene skew the "
                "percentile black/white points.<br><br>"
                "Rule of thumb: the analysis area should contain image and nothing else. Use "
                "the buffer when you deliberately keep a border in frame; a tight crop is the "
                "cleaner fix.<br><br>"
                "For odd frames, the <b>draw region</b> tool next to it goes further: draw the "
                "metering area freehand on the canvas, and the meter reads exactly that — "
                "no centered inset."
            ),
            target=_analysis_buffer,
            section_attr="process_section",
        ),
        TutorialStep(
            title="Process Panel — Bounds Analysis",
            body=(
                "Film dyes follow Beer–Lambert absorption — density is logarithmic — so NegPy "
                "converts the raw signal to log space and meters it there, on two independent "
                "axes: a <b>luma</b> pass sets the black/white-point span, and a per-channel "
                "<b>colour</b> pass <b>measures the orange mask from the actual negative</b> — "
                "no hardcoded mask constants.<br><br>"
                "<b>Luma Range Clip</b> tunes the tonal span (positive = tighter recovery, "
                "negative = outward headroom); <b>Colour Clip</b> sets the per-channel balance "
                "independently. <b>White Point</b> / <b>Black Point</b> fine-tune the detected "
                "bounds without re-analysis — highlight recovery or shadow crush — globally or "
                "per dye layer via their <b>Global / R / G / B</b> selector, like a scanner's "
                "per-channel levels.<br><br>"
                "The stretch is <b>unclamped</b>: tones outside the bounds survive and roll "
                "off later in the print curve's toe and shoulder.<br><br>"
                "<b>Linear RAW</b> also lives here: it decodes the RAW with neutral "
                "multipliers, bypassing the camera's as-shot white balance."
            ),
            target=_process,
            section_attr="process_section",
        ),
        TutorialStep(
            title="Narrowband Scan — Correcting LED Light",
            body=(
                "A scan lit by <b>narrowband RGB LEDs</b> — a Scanlight, or most RGB-LED "
                "sources — hits each dye layer with a much purer band than white light does. "
                "The layers separate further than the film intends, and the conversion comes "
                "out over-saturated.<br><br>"
                "The <b>Narrowband Scan</b> toggle corrects for that light source. It applies "
                "to the preview <i>and</i> every export, so what you judge is what you "
                "deliver.<br><br>"
                "Turning on <b>RGB Scan</b> mode switches it on for you — on the current frame "
                "and as the default for new ones. If you've set a custom <b>Input ICC</b> "
                "profile, that takes precedence and this toggle steps aside."
            ),
            target=_narrowband,
            section_attr="process_section",
        ),
        TutorialStep(
            title="Crosstalk — Dye Unmixing",
            body=(
                "Each film dye layer also absorbs outside its own band — <b>secondary "
                "absorptions</b> that leak one channel into another and mute colour. These "
                "are linear in negative dye density (Beer–Lambert), so NegPy unmixes them "
                "with a per-stock matrix in log-density space, <b>before any analysis</b>.<br><br>"
                "Pick a profile matching your film stock and blend it in with the "
                "<b>Separation</b> strength.<br><br>"
                "Changed the matrix or strength? <b>Re-run Batch Analysis</b> — bounds "
                "measured under a different matrix are invalid."
            ),
            target=_crosstalk,
            section_attr="process_section",
        ),
        TutorialStep(
            title="Roll Consistency — Batch Analysis",
            body=(
                "One enlarger setting for the whole roll. <b>Batch Analysis</b> meters every "
                "loaded frame and builds a roll-wide baseline; <b>Use Roll Average</b> then "
                "locks frames to it, so exposure and colour don't jump from frame to "
                "frame.<br><br>"
                "Roll presets save and load the baseline for later sessions. A locked "
                "baseline is also what keeps <b>Flat masters</b> consistent across a roll."
            ),
            target=_roll,
            section_attr="roll_section",
        ),
        TutorialStep(
            title="Exposure — Density & Grade",
            body=(
                "<b>Density</b> slides the negative's log exposure along the paper curve — "
                "exactly enlarger exposure time. Lower values print brighter.<br><br>"
                "<b>Grade</b> sets contrast on the photographic <b>ISO-R paper scale</b> "
                "(50–180, default 115): the range of log exposure the paper accepts. Lower R "
                "is harder (more contrast and punch); higher R is softer — R110 is roughly "
                "classic paper grade 2. The resulting slope is the literal H&amp;D gamma: "
                "negative density range over paper exposure range.<br><br>"
                "<b>Auto Density</b> and <b>Auto Grade</b> meter each frame for sensible "
                "brightness and contrast out of the box — they correct only <i>partially</i>, "
                "so low-key and high-key shots keep their mood. Turn them off to let the "
                "conversion follow the negative honestly."
            ),
            target=_density,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Set Targets — Retune the Autos",
            body=(
                "Auto Density and Auto Grade aim at a fixed idea of a good print. Your scanner, "
                "your film and your taste may disagree — <b>Set Targets</b> moves the aim.<br><br>"
                "<b>Print Density Target</b> is how bright the metered midtone prints; "
                "<b>Contrast Target</b> is the printed contrast every frame is aimed at. The two strength "
                "sliders decide how far each meter is trusted: at 0 you get a fixed setting for "
                "every frame, at 1 every frame is forced to the same key or the same contrast. "
                "<b>Metering Band</b> caps how far Auto Density may swing between frames.<br><br>"
                "These are a <b>calibration, not an edit</b>: they apply to every image, "
                "including ones you've already worked on, and are remembered between sessions. "
                "The preview follows the sliders live, <b>Cancel</b> puts them back, and "
                "<b>Restore Defaults</b> returns to the shipped values."
            ),
            target=_auto_targets,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Exposure — H&D Curve (Toe & Shoulder)",
            body=(
                "The <b>Toe</b> and <b>Shoulder</b> controls shape the shadow and highlight roll-off "
                "of the H&D characteristic curve — a model of how photographic paper responds to "
                "light, not a generic tone curve.<br><br>"
                "<b>Toe</b>: lifts the paper-black ceiling.<br>"
                "<b>Shoulder</b>: compresses highlights toward paper white.<br>"
                "<b>Width</b>: how far each knee's roll-off reaches.<br>"
                "<b>Snap</b>: the paper's variable midtone gamma — endpoints and anchor stay put.<br><br>"
                "<b>Paper Black</b> shows the paper's real D-max as a lifted black; leave it off "
                "(the default) for black point compensation that maps D-max to display black — then "
                "pull Toe negative to clip deep shadows to exact black.<br><br>"
                "Snap and Paper Black sit with the zone controls (next) under the "
                "<b>Paper Response</b> header."
            ),
            target=_toe,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Split Grade — Zone Contrast",
            body=(
                "Split-grade printing: <b>Shadows Grade</b> and <b>Highlights Grade</b> trim "
                "each zone's contrast in ISO-R points on top of the main Grade — harder "
                "shadows without blowing the highlights, or softer highlights without "
                "flattening the shadows, like a second enlarger exposure through a different "
                "filter.<br><br>"
                "Both trims spare the midtones and stay bounded by the paper's black and "
                "white, and they scope per colour layer through the <b>Global / R / G / B</b> "
                "selector like the main Grade."
            ),
            target=_split_grade,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Zone Density — Shadows & Highlights",
            body=(
                "Where Split Grade is zone <i>contrast</i>, these are zone <i>brightness</i>: "
                "<b>Shadows Density</b> and <b>Highlights Density</b> darken or brighten each "
                "zone while rolling into the paper's black and white limits instead of "
                "clipping — burning in a sky without blocking it up.<br><br>"
                "They live under the <b>Paper Response</b> header with Snap and Paper Black — "
                "the deeper print-curve controls."
            ),
            target=_zone_density,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Per-Layer Trims — Crossover Correction",
            body=(
                "The <b>Global / Red / Green / Blue</b> selector scopes the curve controls to a "
                "single dye layer: in a channel mode, Grade, the Split Grades, Toe, Shoulder, "
                "Width and Snap become that layer's trims.<br><br>"
                "Colour filtration can only <i>shift</i> a layer's curve; trims change its "
                "<i>shape</i> — fixing crossover casts that differ between shadows, mids and "
                "highlights, the correction a real colour darkroom never had. The H&D chart "
                "draws the diverged per-layer curves live."
            ),
            target=_channel_selector,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Exposure — Filtration",
            body=(
                "White balance is real CC filtration — ±1.0 on a slider is ±20cc of dichroic "
                "density. The <b>Global / Shadows / Highlights</b> buttons on top scope the "
                "CMY sliders to a region for precise split-toning control.<br><br>"
                "The <b>Temperature</b> slider re-dials the filter pack along the warm–cool "
                "axis: Magenta and Yellow move together in the right ratio while your "
                "green–magenta tint stays put. Travel is mired-linear (equal drag, equal "
                "perceived shift), <b>T</b>/<b>G</b> nudge it, and the thermometer button "
                "locks the temperature for the whole roll.<br><br>"
                "<b>Pick WB</b>: click a neutral area in the preview and the filtration is "
                "calculated for you."
            ),
            target=_region_btn,
            section_attr="colour_section",
        ),
        TutorialStep(
            title="Cast Removal — Neutral Greys End to End",
            body=(
                "A negative's colour cast isn't constant: it varies with density, so a "
                "midtone-only white balance leaves shadows and highlights drifting "
                "off-colour.<br><br>"
                "<b>Cast Removal</b> measures each channel's deep-shadow reference and gives "
                "it its own slope, pivoting on the midtone — greys read neutral from deep "
                "shadows through highlights, not just at one point.<br><br>"
                "Its strength adapts per frame to how confidently the neutral greys read — "
                "clean greys get the full correction, few-neutral scenes gentler — and the "
                "slider (default 0.5) trims on top; 0 turns it off."
            ),
            target=_cast_removal,
            section_attr="colour_section",
        ),
        TutorialStep(
            title="Exposure — Paper Profiles",
            body=(
                "A <b>paper profile</b> sets the print character — the H&D curve shape — "
                "without touching contrast or exposure. Each profile carries its paper's "
                "tone, per-channel gamma and base tint, mapped from Ilford / Kodak / Foma / "
                "Fuji datasheets.<br><br>"
                "Profiles are mode-aware (RA4 colour papers in C-41, tonal papers in B&W) and "
                "sticky roll-wide. <b>Neutral</b> reproduces the defaults exactly — Grade and "
                "Density still trim on top."
            ),
            target=_paper,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Dodge & Burn",
            body=(
                "Darkroom-style local lighten/darken with freehand <b>polygon masks</b>. "
                "<b>Draw Mask</b>, click to drop vertices, double-click to close; each mask has "
                "its own EV <b>Strength</b> and <b>Feather</b>.<br><br>"
                "Masks change the <b>print exposure</b> ahead of the paper curve — burns roll "
                "into paper black through the toe and dodges lift toward paper white, like "
                "holding back light under the enlarger. Masks are stored in raw-image space, so "
                "they survive rotation, flip and crop. <b>Show Masks</b> toggles their overlay."
            ),
            target=_local,
            section_attr="local_section",
        ),
        TutorialStep(
            title="Lab Panel — Film Aesthetics",
            body=(
                "<b>Colour:</b> "
                "<b>Separation</b> amplifies R/G/B channel differences for richer colour. "
                "<b>Saturation</b> boosts all tones equally; "
                "<b>Vibrance</b> lifts muted tones while protecting already-saturated ones. "
                "<b>Dye Mute</b> pulls the other way: it mutes colour in step with print "
                "contrast, so a hard grade doesn't run away into poster colour. It's on by "
                "default — set it to 0 for the fully saturated look. "
                "<b>Denoise</b> smooths chroma noise in Lab space without touching luminance grain.<br><br>"
                "<b>Detail:</b> "
                "<b>CLAHE</b> applies local contrast enhancement that lifts midtone detail without blowing highlights. "
                "<b>Sharpening</b> uses L-channel unsharp masking — no colour halos.<br><br>"
                "<b>Effects:</b> "
                "<b>Glow</b> simulates lens bloom. "
                "<b>Halation</b> mimics red scatter caused by light bouncing back through the film base — "
                "strongly red-dominant, exactly like real film halation."
            ),
            target=_lab,
            section_attr="lab_section",
        ),
        TutorialStep(
            title="Toning",
            body=(
                "<b>Split Toning</b> (all modes) pushes shadows and highlights toward independent "
                "hue angles with their own strength. It works in Lab space, so luminance — and "
                "therefore grain and detail — is preserved exactly.<br><br>"
                "<b>Selenium</b> and <b>Sepia</b> simulate classic chemical toners on the print's "
                "silver density (B&W mode only): selenium converts the densest silver first — "
                "deeper blacks and cool eggplant shadows; sepia bleach-redevelops the thinnest "
                "silver first — warm highlights that hold the shadows (partial strength gives the "
                "classic split-sepia look).<br><br>"
                "<b>Gold</b> is the archival gold bath: a cool blue-black shift in the "
                "highlights and mids with a slight density boost, while dense shadows hold. "
                "Run it over Sepia for the classic combination — toned highlights pushed from "
                "yellow-brown toward orange-red."
            ),
            target=_toning,
            section_attr="toning_section",
        ),
        TutorialStep(
            title="Retouch Panel — Dust Removal",
            body=(
                "<b>Optical Removal</b> detects and removes small particles on the visible scan by "
                "local contrast. Lower the threshold to be more aggressive.<br><br>"
                "<b>IR Removal</b> works from the scanner's infrared channel, where dust blocks "
                "light but the colour dyes don't — catching what the eye can't separate from "
                "grain. Detection is ratio-normalized rather than a raw-IR threshold, so the "
                "slider responds smoothly instead of flipping the whole frame at a cliff. "
                "Semi-transparent specks are divided back out to recover the image hiding "
                "underneath, and only the opaque cores get cloned. B&amp;W and Kodachrome scans "
                "are skipped.<br><br>"
                "<b>Heal Tool</b>: click individual dust spots in the preview — each heal "
                "clones a matching patch from elsewhere in the frame and blends the seam, so "
                "grain stays intact.<br><br>"
                "<b>Scratch Tool</b>: click a polyline along a hair or scratch, double-click "
                "or <b>Enter</b> to commit. <b>Undo Last</b> / <b>Clear All</b> manage the "
                "spots."
            ),
            target=_retouch,
            section_attr="retouch_section",
        ),
        TutorialStep(
            title="Dust Overlay — See What's Detected",
            body=(
                "Dust thresholds are hard to set blind. The <b>Overlay</b> button cycles the "
                "detection inspector so you can tune by eye: <b>Off → Marked → IR</b> (the IR "
                "state appears only on scans that have an infrared channel).<br><br>"
                "<b>Marked</b> paints every spot the detector is about to fix; <b>IR</b> shows "
                "the raw infrared read behind it. Turn Optical or IR Removal on first — the "
                "overlay draws what those passes found, so with both off there's nothing to "
                "show.<br><br>"
                "Watch the overlay while you drag a threshold: too aggressive lights up grain, "
                "too conservative leaves specks unmarked."
            ),
            target=_dust_overlay,
            section_attr="retouch_section",
        ),
        TutorialStep(
            title="Finish — Edge Burn, Carrier & Mats",
            body=(
                "Print-presentation touches, applied at the very end of the pipeline — after "
                "the crop, on the finished print.<br><br>"
                "<b>Edge Burn</b> is a real exposure burn measured in <b>stops</b>, not a "
                "darkening overlay: the darkroom printer's edge burn that holds the eye inside "
                "the frame. <b>Size</b> sets how far in it reaches, and <b>Roundness</b> morphs "
                "it from a radial falloff to a straight-edged card burn.<br><br>"
                "<b>Filed Carrier</b> prints the black rebate of a filed-out negative carrier — "
                "<b>Width</b> sets the frame (0 = off), <b>Roughness</b> breaks up its inner "
                "edge the way a filed carrier actually looks.<br><br>"
                "<b>Border</b> lays a mat around the print: <b>Width</b>, plus <b>Bottom "
                "weight</b> for the window-mat proportion where the bottom margin runs deeper. "
                "Pick its colour from the swatch, or turn on <b>Paper white</b> to tie the mat "
                "to the toned paper white so it matches the print instead of fighting it."
            ),
            target=_edge_burn,
            section_attr="finish_section",
        ),
        TutorialStep(
            title="History",
            body=(
                "The <b>History</b> tab lists every edit step for the current photo. Click any "
                "step to jump back to that state — the preview updates instantly — then carry on "
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
                "The <b>Metadata</b> tab writes film and scan info — stock, format, "
                "developer, push/pull, scanner — into the EXIF/XMP of exported files.<br><br>"
                "<b>Manage…</b> opens the <b>Gear Library</b>: a searchable, user-extendable "
                "library of cameras, lenses and film stocks; gear picked for a frame rides "
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
                "The <b>Export</b> tab (right panel, now active) is where you save your results.<br><br>"
                "Choose a format (<b>JPEG</b>, high-bit-depth <b>TIFF</b>, PNG, WebP, JPEG XL, DNG), "
                "pick a colour space, and set resolution or print size. The <b>ICC</b> section adds "
                "monitor-profile display and soft-proofing.<br><br>"
                "The <b>Export</b> and <b>Export Presets</b> buttons are triggers; each button's "
                "menu arrow picks what it exports (current frame, selected frames, or all visible "
                "frames) and remembers the choice. Presets run every enabled preset per frame. "
                "<b>Contact Sheet</b> renders all frames into one sheet. "
                "Export always runs at full RAW resolution."
            ),
            target=_export,
            pre_hook=lambda w: w.right_panel.show_tab_by_key("export"),
        ),
        TutorialStep(
            title="Export — Flat Master",
            body=(
                "The <b>Flat — for editing elsewhere</b> output intent exports a flat, neutral, "
                "wide-gamut <b>16-bit TIFF</b> (or linear <b>DNG</b>) digital-intermediate master "
                "for Lightroom / Darktable / Photoshop.<br><br>"
                "It skips the creative print look and maps camera RAWs to ProPhoto via the camera's "
                "own matrix. <b>Preview Flat</b> peeks at the master on the canvas — also on the "
                "toolbar and on <b>|</b> — and "
                "<b>Roll Baseline</b> keeps flat masters consistent across a roll. Standard "
                "<b>Print</b> output is unaffected."
            ),
            target=_flat_master,
            pre_hook=lambda w: w.right_panel.show_tab_by_key("export"),
        ),
        TutorialStep(
            title="You're all set!",
            body=(
                "That's the core workflow. A few more things worth knowing:<br><br>"
                "• Press <b>?</b> or use the ⋯ menu for keyboard shortcuts.<br>"
                "• Canvas tools share one grammar: the first <b>Esc</b> clears the points "
                "you're placing, the second puts the tool down. <b>Shift+S</b> Scratch, "
                "<b>Shift+B</b> Dodge &amp; Burn, <b>Shift+R</b> Analysis Region, "
                "<b>|</b> flat-master peek.<br>"
                "• Scanning with a tethered camera? The <b>Camera Scanning</b> section on the "
                "Scan tab drives the body and Scanlight directly (macOS/Linux) — see "
                "<code>docs/CAMERA_SCANNING.md</code>.<br>"
                "• See <code>docs/USER_GUIDE.md</code> for the full reference.<br>"
                "• Having GPU or rendering issues? Edit "
                "<code>Documents/NegPy/override.toml</code> to switch backends "
                "without touching code.<br>"
                "• Edits auto-save to a local database — no manual save needed between files."
            ),
            target=lambda w: None,
            pre_hook=lambda w: w.right_panel.show_tab_by_key("setup"),
        ),
    ]
