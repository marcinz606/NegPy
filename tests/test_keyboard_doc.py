"""docs/KEYBOARD.md carries the registry's bindings, not a hand-typed copy of them."""

from negpy.desktop.view.keyboard_doc import DOC, _key, render_document
from negpy.desktop.view.shortcut_registry import REGISTRY


def test_keyboard_doc_matches_the_registry():
    text = DOC.read_text(encoding="utf-8")
    assert render_document(text) == text, "run: uv run python -m negpy.desktop.view.keyboard_doc"


def test_every_bound_key_is_in_the_doc():
    """Slider pairs render as one merged row, so the key is the thing every action shares."""
    text = DOC.read_text(encoding="utf-8")
    for action_id, entry in REGISTRY.items():
        if entry.default_key:
            assert _key(entry.default_key) in text, action_id
