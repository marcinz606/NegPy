"""User-visible strings in the desktop layer use the ellipsis character and American spelling."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "negpy" / "desktop"
BRITISH = re.compile(r"\b(cancelled|colours?|greys?|centred?|recognised|labelled|neighbouring|favourites?|behaviour|normalised?)\b", re.I)
# Settings keys and tab ids are data, so they keep the spelling they were stored under.
IDENTIFIER = re.compile(r"[a-z0-9_]+")


def _strings(tree: ast.AST):
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def _offenders(test) -> list[str]:
    found = []
    for path in DESKTOP.rglob("*.py"):
        for line, text in _strings(ast.parse(path.read_text(encoding="utf-8"))):
            if not IDENTIFIER.fullmatch(text) and test(text):
                found.append(f"{path.relative_to(ROOT)}:{line}: {text[:80]}")
    return found


def test_no_three_dot_ellipsis():
    assert _offenders(lambda s: "..." in s) == []


def test_american_spelling():
    assert _offenders(lambda s: BRITISH.search(s) is not None) == []
