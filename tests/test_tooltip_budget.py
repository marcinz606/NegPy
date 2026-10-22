"""A tooltip is one short, plain sentence; the detail lives in the card's ⓘ guide."""

import ast
import html
import re
from pathlib import Path

from conftest import FakeController, FakeRepo

DESKTOP = Path(__file__).resolve().parents[1] / "negpy" / "desktop"
# Factories in styles/templates.py and the choice widgets: positional indexes that are tooltips.
_TOOLTIP_ARGS = {
    "tool_toggle": (2,),
    "clearable_tool_toggle": (2, 3),
    "labeled_toggle": (3,),
    "labeled_action": (2,),
    "icon_button": (1,),
    "header_button": (1,),
    "SegmentedChoice": (1,),
}
_TIP_NAME = re.compile(r"(?i)tip(s)?$|tooltip(s)?$")

MAX_WORDS = 25
# Shortcut chips render as a trailing table, and a slider appends its reset hint as a <div>.
_CHROME = re.compile(r"<table.*?</table>|<div[^>]*>.*?</div>", re.S)
_WORD = re.compile(r"[^\W\d_][\w'’.-]*")


def _plain(tip: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", _CHROME.sub(" ", tip))).split())


def test_every_panel_tooltip_is_short_and_has_no_dash_aside(qapp):
    from PyQt6.QtWidgets import QWidget

    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = RightPanel(FakeController(FakeRepo()))
    panel.apply_shortcut_tooltips()
    panel.controls_panel.apply_shortcut_tooltips()

    tips = {_plain(w.toolTip()) for w in panel.findChildren(QWidget)} - {""}
    long = sorted(t for t in tips if len(_WORD.findall(t)) > MAX_WORDS)
    dashed = sorted(t for t in tips if "—" in t)
    assert not long, f"Tooltips over {MAX_WORDS} words; move the detail to docs/USER_GUIDE.md:\n" + "\n".join(long)
    assert not dashed, "Tooltips with an em dash; use a period, colon or comma:\n" + "\n".join(dashed)


def _tr_literals(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) in {"tr", "mark"} and sub.args:
            if isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str):
                yield sub.args[0].value


def _static_tooltips():
    """Every tr() literal that reaches a tooltip directly, in dialogs and windows the panel walk never builds."""
    for path in DESKTOP.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            exprs = []
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in {"setToolTip", "set_choice_tooltip"}:
                    exprs += node.args
                exprs += [node.args[i] for i in _TOOLTIP_ARGS.get(name, ()) if i < len(node.args)]
                exprs += [k.value for k in node.keywords if k.arg and _TIP_NAME.search(k.arg)]
            elif isinstance(node, ast.Assign):
                exprs += [node.value for t in node.targets if _TIP_NAME.search(getattr(t, "id", None) or getattr(t, "attr", None) or "")]
            for expr in exprs:
                for text in _tr_literals(expr):
                    yield path.name, text


def test_every_literal_tooltip_is_short_and_has_no_dash_aside():
    long, dashed = set(), set()
    for where, text in _static_tooltips():
        words = _WORD.findall(_plain(text))
        if len(words) > MAX_WORDS:
            long.add(f"{where}: {text[:80]}")
        if "—" in text:
            dashed.add(f"{where}: {text[:80]}")
    assert not long, f"Tooltips over {MAX_WORDS} words:\n" + "\n".join(sorted(long))
    assert not dashed, "Tooltips with an em dash:\n" + "\n".join(sorted(dashed))
