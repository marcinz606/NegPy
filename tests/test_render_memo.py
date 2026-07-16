from types import SimpleNamespace

import numpy as np

from negpy.desktop.render_memo import RenderMemo


def _cfg(slots: int = 2) -> SimpleNamespace:
    return SimpleNamespace(preview_cache_max_full_res_entries=slots)


def _payload() -> dict:
    return {"base_positive": np.zeros((2, 2, 3), dtype=np.float32), "content_rect": None}


def test_hit_requires_matching_key() -> None:
    m = RenderMemo(_cfg())
    p = _payload()
    m.store("frameA", "key1", p)
    assert m.get("frameA", "key1") is p
    assert m.get("frameA", "key2") is None  # config/display inputs changed
    assert m.get("frameB", "key1") is None


def test_store_overwrites_per_file() -> None:
    m = RenderMemo(_cfg())
    m.store("frameA", "key1", _payload())
    p2 = _payload()
    m.store("frameA", "key2", p2)  # an edit re-renders under a new key
    assert m.get("frameA", "key1") is None
    assert m.get("frameA", "key2") is p2


def test_budget_evicts_least_recent_file() -> None:
    m = RenderMemo(_cfg(slots=2))
    m.store("A", "k", _payload())
    m.store("B", "k", _payload())
    assert m.get("A", "k") is not None  # A is now most-recent
    m.store("C", "k", _payload())
    assert m.get("B", "k") is None
    assert m.get("A", "k") is not None
    assert m.get("C", "k") is not None


def test_budget_floor_is_two() -> None:
    # Even with the knob at 1, navigate-back needs current + previous.
    m = RenderMemo(_cfg(slots=1))
    m.store("A", "k", _payload())
    m.store("B", "k", _payload())
    assert m.get("A", "k") is not None
    assert m.get("B", "k") is not None


def test_empty_identifiers_are_not_stored() -> None:
    m = RenderMemo(_cfg())
    m.store("", "key1", _payload())
    m.store("frameA", "", _payload())
    assert m.get("", "key1") is None
    assert m.get("frameA", "") is None


def test_rekey_moves_entry_to_new_identity() -> None:
    # Bounds writeback after the first render changes the config (render=False):
    # same pixels, new key — the entry must follow.
    m = RenderMemo(_cfg())
    p = _payload()
    m.store("frameA", "pre-bounds", p)
    m.rekey("frameA", "post-bounds")
    assert m.get("frameA", "pre-bounds") is None
    assert m.get("frameA", "post-bounds") is p
    m.rekey("frameB", "whatever")  # unknown file: no-op
    assert m.get("frameB", "whatever") is None


def test_clear_and_invalidate() -> None:
    m = RenderMemo(_cfg())
    m.store("A", "k", _payload())
    m.store("B", "k", _payload())
    m.invalidate("A")
    assert m.get("A", "k") is None
    assert m.get("B", "k") is not None
    m.clear()
    assert m.get("B", "k") is None
