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
        return w.controls_panel.exposure_sidebar.density_slider

    def _toe(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.exposure_sidebar.toe_slider

    def _region_btn(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.exposure_sidebar.region_global_btn

    def _lab(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.lab_sidebar

    def _retouch(w: "MainWindow") -> Optional[QWidget]:
        return w.controls_panel.retouch_sidebar

    def _export(w: "MainWindow") -> Optional[QWidget]:
        return w.right_panel.export_sidebar

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
            title="Process Panel — Bounds Analysis",
            body=(
                "Unlike converters that use hardcoded orange mask constants, "
                "NegPy <b>measures the orange mask directly from each negative</b> "
                "using statistical bounds analysis.<br><br>"
                "<b>D-Range Clip</b> controls how aggressively outlier pixels are excluded "
                "before calculating white/black points. Find good setting for yor scaning setup.<br><br>"
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
                "<b>Grade</b> sets contrast, mirroring photographic paper grades: "
                "0 is flat, higher values add punch."
            ),
            target=_density,
            section_attr="exposure_section",
        ),
        TutorialStep(
            title="Exposure — Sigmoid Curve",
            body=(
                "The <b>Toe</b> and <b>Shoulder</b> controls shape the shadow and highlight roll-off. "
                "This is not a generic tone curve — it models how analogue paper responds to light.<br><br>"
                "<b>Toe</b>: lifts shadow density, adding depth to the darkest areas.<br>"
                "<b>Shoulder</b>: gently compresses highlights for a softer fade to white.<br>"
                "<b>Width</b>: how broadly each transition region extends."
            ),
            target=_toe,
            section_attr="exposure_section",
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
            section_attr="exposure_section",
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
            title="Export",
            body=(
                "The <b>Export</b> tab (right panel, now active) is where you save your results.<br><br>"
                "Choose <b>JPEG</b> or high-bit-depth <b>TIFF</b>, pick a colour space "
                "(sRGB, Adobe RGB, Greyscale, and others), and set resolution or print size.<br><br>"
                "Export always runs at full RAW resolution through the complete pipeline. "
                "<b>Batch Export</b> processes all loaded files at once."
            ),
            target=_export,
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
