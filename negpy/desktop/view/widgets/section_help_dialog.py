"""Panel guides: the ⓘ in a section header opens that panel's slice of the user guide.

docs/USER_GUIDE.md is the single source — it ships in the frozen build (build.py) and the
slices are anchored on `<!-- panel:<key> -->` markers keyed by the section's persistence key,
so renumbering or retitling a heading can't orphan a guide.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from negpy.desktop.view.styles.theme import THEME
from negpy.kernel.system.paths import get_resource_path

_MARKER = re.compile(r"^<!--\s*panel:([a-z_]+)\s*-->\s*$")
_HEADING = re.compile(r"^(#{1,6})\s")
# Cross-doc links ([CROSSTALK.md](CROSSTALK.md)) have nowhere to go from a modal, and Qt's
# markdown importer paints anchors in the app palette's accent red — unreadable at body size.
# Flattened to their text, so they read as the file names they are.
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


@lru_cache(maxsize=1)
def _guides() -> dict[str, str]:
    """Section key → the markdown under its heading. Empty when the doc is unreadable,
    which just means no section offers a guide."""
    try:
        lines = Path(get_resource_path("docs/USER_GUIDE.md")).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    guides: dict[str, str] = {}
    for i, line in enumerate(lines):
        marker = _MARKER.match(line)
        if not marker or i + 1 >= len(lines):
            continue
        heading = _HEADING.match(lines[i + 1])
        if not heading:
            continue
        # Stop at the next heading of the same or higher rank; deeper ones (#### topics
        # inside §3) belong to this slice.
        closing = re.compile(rf"^#{{1,{len(heading.group(1))}}}\s")
        body = lines[i + 2 :]
        for j, later in enumerate(body):
            if closing.match(later):
                body = body[:j]
                break
        guides[marker.group(1)] = _LINK.sub(r"\1", "\n".join(body).strip().removesuffix("---").strip())
    return guides


def has_guide(key: str) -> bool:
    return bool(_guides().get(key))


def guide_markdown(key: str) -> str:
    return _guides().get(key, "")


class SectionHelpDialog(QDialog):
    """Modal reference for one panel, rendered from the user guide."""

    def __init__(self, key: str, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        heading = f"Reading the {title} panel"
        self.setWindowTitle(heading)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(720, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        label = QLabel(heading)
        label.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_title}px; font-weight: bold;")
        root.addWidget(label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {THEME.border_color};")
        root.addWidget(divider)

        self.body = QTextBrowser()
        self.body.setReadOnly(True)
        self.body.setOpenExternalLinks(False)
        # Backstop for a link the loader can't flatten (GitHub-dialect bare-URL autolinks):
        # navigating would replace the guide with a load error.
        self.body.setOpenLinks(False)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setStyleSheet(
            f"QTextBrowser {{ background: transparent; border: none; color: {THEME.text_secondary}; font-size: {THEME.font_size_base}px; }}"
        )
        self.body.setMarkdown(guide_markdown(key))
        root.addWidget(self.body, stretch=1)

        actions = QHBoxLayout()
        actions.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setProperty("primary", True)
        close_btn.clicked.connect(self.accept)
        actions.addWidget(close_btn)
        root.addLayout(actions)
