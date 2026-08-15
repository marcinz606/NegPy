"""Counts read as English, not as "1 file(s)"."""

import pytest

from negpy.kernel.system.text import count_of, plural


@pytest.mark.parametrize(
    "count,expected",
    [(0, "0 files"), (1, "1 file"), (2, "2 files"), (17, "17 files")],
)
def test_count_and_noun_agree(count, expected):
    assert count_of(count, "file") == expected


def test_zero_takes_the_plural():
    """English says "0 files", not "0 file"."""
    assert count_of(0, "frame") == "0 frames"


def test_an_irregular_plural_can_be_given():
    assert count_of(1, "copy", "copies") == "1 copy"
    assert count_of(3, "copy", "copies") == "3 copies"


def test_plural_alone_omits_the_number():
    """For a sentence that names the count elsewhere, or lists the items instead."""
    assert plural(1, "frame") == "frame"
    assert plural(2, "frame") == "frames"


def test_it_carries_verbs_too():
    """The noun is rarely the only word that has to agree."""
    assert plural(1, "is", "are") == "is"
    assert plural(2, "is", "are") == "are"


def test_a_multi_word_noun_pluralises_on_the_last_word():
    assert count_of(2, "edit sidecar") == "2 edit sidecars"


def test_negatives_are_treated_by_magnitude():
    """A count should not go plural because it is signed."""
    assert count_of(-1, "file") == "-1 file"


def test_no_user_facing_string_still_says_s_in_brackets():
    """The report that prompted this: "Reopen your last session (1 file(s))?".

    Parsed rather than grepped. A regex over source lines matches between two separate
    literals — `"sizes": [list(s) for s in f["` reads as one string to it — and cannot tell
    a docstring from a label.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "negpy"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "text.py":
            continue  # the helper's own docstring quotes the pattern it replaces
        tree = ast.parse(path.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings or "(s)" not in node.value:
                continue
            offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, "use count_of/plural instead of '(s)': " + ", ".join(offenders)
