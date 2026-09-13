"""docs/KEYBOARD.md's tables, rendered from the shortcut registry.

The prose above the first table and everything after the end marker are written by hand;
the block between the markers is generated. ``uv run python -m negpy.desktop.view.keyboard_doc``
rewrites it, and tests/test_keyboard_doc.py fails when the file has drifted from the registry.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QKeySequence

from negpy.desktop.view.shortcut_registry import (
    REGISTRY,
    EditorRowSingle,
    EditorRowSlider,
    categories_in_order,
    category_editor_rows,
)

START = "<!-- shortcuts:start -->"
END = "<!-- shortcuts:end -->"
DOC = Path(__file__).resolve().parents[3] / "docs" / "KEYBOARD.md"


def _key(key: str) -> str:
    """Portable spelling with spaced pluses, the way the hand-written tables read."""
    text = QKeySequence(key).toString(QKeySequence.SequenceFormat.PortableText) or key
    return "`" + text.replace("+", " + ").replace("|", "\\|") + "`"


def _step(group) -> str:
    step = f"{group.default_step:.{group.step_decimals}f}"
    return f"{step}{group.step_suffix}"


def render_tables() -> str:
    lines: list[str] = []
    for category, items in categories_in_order():
        rows: list[str] = []
        for row in category_editor_rows(items):
            if isinstance(row, EditorRowSlider):
                g = row.group
                k_inc, k_dec = REGISTRY[g.inc_action].default_key, REGISTRY[g.dec_action].default_key
                if not (k_inc or k_dec):
                    continue
                name = g.label.replace(" ↑/↓", "")
                rows.append(f"| {_key(k_inc)} / {_key(k_dec)} | Increase / decrease **{name}** (default step {_step(g)}) |")
            else:
                assert isinstance(row, EditorRowSingle)
                if not row.entry.default_key:
                    continue
                rows.append(f"| {_key(row.entry.default_key)} | {row.entry.description} |")
        if not rows:
            continue
        lines += [f"## {category}", "| Key | Action |", "|-----|--------|", *rows, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def render_document(existing: str) -> str:
    head, _, rest = existing.partition(START)
    _, _, tail = rest.partition(END)
    if not rest:
        raise ValueError(f"{DOC} has no {START} marker")
    return f"{head}{START}\n{render_tables()}{END}{tail}"


if __name__ == "__main__":
    DOC.write_text(render_document(DOC.read_text(encoding="utf-8")), encoding="utf-8")
