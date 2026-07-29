"""Explains every read-out in the Analysis panel. Reached from the ⓘ in its header.

The prose lives here rather than in docs/ because docs/ isn't bundled into the frozen
build (see NegPy.spec); the stats and probe paragraphs are imported from their tooltips
so the two surfaces can't drift.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout

from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.stats import PROBE_TOOLTIP, STAT_TOOLTIPS

_TOPICS: list[tuple[str, str]] = [
    (
        "Photometric curve",
        "The chart is the paper characteristic (H&amp;D) curve NegPy is printing through right now — "
        "not a curves editor, a model of how a sheet of photographic paper responds. "
        "Left to right is <b>negative density</b>, the exposure the paper receives: dense parts of the "
        "negative (the scene's highlights) sit to the right. Bottom to top is the <b>print tone</b> that "
        "comes out. A steeper curve means more contrast — that is what Grade moves. The flattening at "
        "each end is the toe (shadows) and shoulder (highlights), where the paper runs out of range.<br><br>"
        "The crosshair marks the <b>pivot</b>: the density the curve rotates around when you change "
        "contrast, so the midtone stays put. While you drag a slider a faint <b>ghost</b> of the previous "
        "curve stays behind for comparison. If cast removal pulls the channels apart you get three "
        "separate R/G/B traces instead of one grey curve — that spread <i>is</i> the colour correction.",
    ),
    (
        "The two histograms",
        "Two different histograms share the chart. Behind the curve, rising from the bottom, is the "
        "<b>output histogram</b> — the tones of the print you're looking at, in R, G, B and luminance. "
        "Along the bottom axis is the <b>negative density histogram</b> — what the scan actually "
        "contains, before the curve.<br><br>"
        "Read them against each other: the density histogram tells you which part of the horizontal "
        "axis your negative occupies, and the curve tells you what happens to it. If the negative's "
        "data sits entirely on the flat toe, no amount of contrast will pull those shadows apart — "
        "move the exposure so the data lands on the steep middle instead.",
    ),
    (
        "LIN / LOG toggle",
        "Bottom-right of the chart. It switches the histogram's <i>height</i> axis (how many pixels), "
        "not the tone axis. <b>LIN</b> is literal — a big flat sky dwarfs everything else. <b>LOG</b> "
        "compresses the tall peaks so the thin tails become visible, which is where the few hundred "
        "pixels of deep shadow or specular highlight live. Use LOG when hunting for clipping, LIN when "
        "judging where the bulk of the frame sits. The choice is remembered between sessions.",
    ),
    (
        "Clipping triangles",
        "Small R, G and B triangles in the top corners of the chart: <b>top-left</b> = shadows crushed "
        "to pure black, <b>top-right</b> = highlights blown to pure white. They only appear once a "
        "channel passes 0.5% of the frame. A little is normal — a real print has a black. Watch for a "
        "single channel clipping alone, which is a colour cast pushing one dye off the end rather than "
        "an exposure problem.",
    ),
    (
        "Zone shading and zone ticks",
        "The amber wash on the left and the blue wash on the right mark the curve's toe and shoulder — "
        "the compressed ends where tonal separation is being lost. The ticks along the bottom are "
        "Adams zones I–IX, so you can read straight off the axis which zone a given negative density "
        "prints as.",
    ),
    (
        "Step wedge",
        "A 21-step Stouffer-style grey wedge printed through your current curve, in even density "
        "increments labelled in the scan's own density units. It's a ruler for the curve: where "
        "neighbouring patches are clearly different, you have tonal separation; where they merge into "
        "one flat black or white block, those tones are gone. The brackets mark the usable span. "
        "It hides while peeking the flat scan, since there's no print curve to wedge.",
    ),
    (
        "Zone strip",
        "Ten cells, zone 0 through IX — <b>0 is paper black, V is 18% mid-grey, X is paper white</b>. "
        "The brightness of each cell is the zone's tone; how solid it looks is how much of the frame "
        "lands there. This is the fastest read of whether a frame is low-key, high-key or sitting "
        "sensibly in the middle. The end cells tint <b>red</b> when shadows are blocked up or "
        "highlights are blown. Hover a cell for its exact percentage.",
    ),
    ("Probe", PROBE_TOOLTIP),
]

_STAT_INTRO = (
    "The four numeric rows at the bottom. Each one has the same explanation on hover, and each is a "
    "measurement of the negative rather than of your edit:"
)


def _build_html() -> str:
    heading_css = f"color:{THEME.text_primary}; font-size:{THEME.font_size_header}px; font-weight:{THEME.weight_semibold};"
    body_css = f"color:{THEME.text_secondary}; font-size:{THEME.font_size_base}px;"
    parts = [
        f'<p style="{body_css}">The Analysis panel is your feedback while printing. '
        "Everything in it describes the frame you're on and updates as you edit; nothing in it is a "
        "control. Top to bottom:</p>"
    ]
    for title, text in _TOPICS:
        parts.append(f'<p style="{heading_css}">{title}</p><p style="{body_css}">{text}</p>')

    parts.append(f'<p style="{heading_css}">Negative stats</p><p style="{body_css}">{_STAT_INTRO}</p><ul>')
    for name, text in STAT_TOOLTIPS.items():
        parts.append(f'<li style="{body_css}"><b>{name}</b> — {text}</li>')
    parts.append("</ul>")
    return "".join(parts)


class AnalysisHelpDialog(QDialog):
    """Modal reference for the Analysis panel's charts and read-outs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reading the Analysis panel")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(720, 720)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("Reading the Analysis panel")
        title.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_title}px; font-weight: bold;")
        root.addWidget(title)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {THEME.border_color};")
        root.addWidget(divider)

        self.body = QTextBrowser()
        self.body.setReadOnly(True)
        self.body.setOpenExternalLinks(False)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setStyleSheet("QTextBrowser { background: transparent; border: none; }")
        self.body.setHtml(_build_html())
        root.addWidget(self.body, stretch=1)

        actions = QHBoxLayout()
        actions.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setProperty("primary", True)
        close_btn.clicked.connect(self.accept)
        actions.addWidget(close_btn)
        root.addLayout(actions)
