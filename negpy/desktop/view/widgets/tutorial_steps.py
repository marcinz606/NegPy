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

    return [
        TutorialStep(
            title="Welcome to NegPy",
            body=(
                "NegPy is a non-destructive RAW film scanner. "
                "Your edits follow a fixed pipeline:<br><br>"
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
                "click outside to draw a fresh rectangle.<br><br>"
                "<b>Auto</b> detects the film edge (with a mode selector for image-only vs. full "
                "film extent). <b>Fine Rot</b> straightens tilted scans against a rule-of-thirds "
                "grid, and <b>Detect Aspect Ratio</b> snaps the crop to the nearest standard ratio."
            ),
            target=_crop,
            section_attr="geometry_section",
        ),
        TutorialStep(
            title="Process Panel — Bounds Analysis",
            body=(
                "Unlike converters that use hardcoded orange mask constants, "
                "NegPy <b>measures the orange mask directly from each negative</b> "
                "using statistical bounds analysis.<br><br>"
                "The clip is split into two independent controls: <b>Luma Range Clip</b> sets the "
                "tonal black/white-point span (positive = tighter recovery, negative = outward "
                "headroom), while <b>Colour Clip</b> sets the per-channel balance for orange-mask "
                "cast removal — independent of the tonal range. Find good settings for your scanning setup.<br><br>"
                "<b>Batch Analysis</b> normalises all loaded frames to a roll-wide average — "
                "enable <b>Use Roll Average</b> afterwards for consistent exposure across the entire roll."
            ),
            target=_process,
            section_attr="process_section",
        ),
        TutorialStep(
            title="Exposure — Density & Grade",
            body=(
                "<b>Density</b> controls overall brightness — lower values produce a brighter image, "
                "just like longer exposure in an analog darkroom.<br><br>"
                "<b>Grade</b> sets contrast on the photographic <b>ISO-R paper scale</b> "
                "(50–180, default 115). Lower R is harder (more contrast and punch); higher R "
                "is softer (flatter) — R110 is roughly classic paper grade 2.<br><br>"
                "<b>Auto Density</b> and <b>Auto Grade</b> meter each frame for sensible "
                "brightness and contrast out of the box; turn them off to let the conversion "
                "follow the negative honestly."
            ),
            target=_density,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Exposure — H&D Curve (Toe & Shoulder)",
            body=(
                "The <b>Toe</b> and <b>Shoulder</b> controls shape the shadow and highlight roll-off "
                "of the H&D characteristic curve — not a generic tone curve, but a model of how "
                "photographic paper responds to light, with independent softplus knees at each end.<br><br>"
                "<b>Toe</b>: lifts the paper-black ceiling, adding depth to the darkest areas.<br>"
                "<b>Shoulder</b>: compresses highlights for a softer fade to paper white.<br>"
                "<b>Width</b>: how sharply each transition knee bends."
            ),
            target=_toe,
            section_attr="tone_section",
        ),
        TutorialStep(
            title="Exposure — Color Balance",
            body=(
                "Three CMY sliders operate in three <b>regions</b> — Global, Shadows, and Highlights — "
                "giving you precise split-toning control over colour balance.<br><br>"
                "<b>Pick WB</b>: click a neutral area in the preview to auto-calculate white balance shifts.<br><br>"
                "<b>Linear RAW</b>: bypasses the camera's as-shot white balance and starts from neutral "
                "multipliers. Leave it off for a sensible default starting point."
            ),
            target=_region_btn,
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
            title="Lab Panel — Film Aesthetics",
            body=(
                "<b>Color:</b> "
                "<b>Separation</b> amplifies R/G/B channel differences for richer colour. "
                "<b>Saturation</b> boosts all tones equally; "
                "<b>Vibrance</b> lifts muted tones while protecting already-saturated ones. "
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
                "<b>Selenium</b> and <b>Sepia</b> simulate classic chemical toners (B&W mode only): "
                "selenium cools the shadows, sepia warms the midtones."
            ),
            target=_toning,
            section_attr="toning_section",
        ),
        TutorialStep(
            title="Retouch Panel — Dust Removal",
            body=(
                "<b>Auto Dust</b> detects and removes small particles based on a density threshold. "
                "Lower the threshold to be more aggressive.<br><br>"
                "<b>Heal Tool</b>: click to enable, then click individual dust spots in the preview "
                "for manual removal. Use <b>Undo Last</b> or <b>Clear All</b> to manage spots."
            ),
            target=_retouch,
            section_attr="retouch_section",
        ),
        TutorialStep(
            title="Dodge & Burn",
            body=(
                "Darkroom-style local lighten/darken with freehand <b>polygon masks</b>. "
                "<b>Draw Mask</b>, click to drop vertices, double-click to close; each mask has "
                "its own EV <b>Strength</b> and <b>Feather</b>.<br><br>"
                "Masks are stored in raw-image space, so they survive rotation, flip and crop. "
                "<b>Show Masks</b> toggles their overlay. Runs on the GPU with bit-for-bit CPU "
                "parity."
            ),
            target=_local,
            section_attr="local_section",
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
            title="Export",
            body=(
                "The <b>Export</b> tab (right panel, now active) is where you save your results.<br><br>"
                "Choose a format (<b>JPEG</b>, high-bit-depth <b>TIFF</b>, PNG, WebP, JPEG XL, DNG), "
                "pick a colour space, and set resolution or print size. The <b>ICC</b> section adds "
                "monitor-profile display and soft-proofing.<br><br>"
                "<b>Export Presets</b> save named configurations for one-click batch output, and "
                "<b>Contact Sheet</b> renders all frames into one sheet. Export always runs at full "
                "RAW resolution; <b>Export All</b> processes every loaded file."
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
                "own matrix. <b>Preview Flat</b> peeks at the master on the canvas, and "
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
