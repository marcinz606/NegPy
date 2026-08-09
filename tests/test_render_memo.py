from types import SimpleNamespace

import numpy as np

from negpy.desktop.render_memo import RenderMemo


def _cfg(slots: int = 2, memo_slots: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        preview_cache_max_full_res_entries=slots,
        render_memo_max_entries=slots if memo_slots is None else memo_slots,
    )


def _payload() -> dict:
    return {"base_positive": np.zeros((2, 2, 3), dtype=np.float32), "content_rect": None}


class _FakeTexture:
    """Stands in for a GPUTexture: the memo frees by duck-typed destroy()."""

    def __init__(self) -> None:
        self.destroyed = 0

    def destroy(self) -> None:
        self.destroyed += 1


def _gpu_payload(tex: _FakeTexture) -> dict:
    return {"base_positive": tex, "content_rect": None}


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


def test_preview_frames_get_the_bigger_budget() -> None:
    # Several frames along a roll stay instant; HQ falls back to the full-res budget.
    m = RenderMemo(_cfg(slots=2, memo_slots=5))
    for i in range(5):
        m.store(f"f{i}", "k", _payload())
    assert m.get("f0", "k") is not None

    m.large_entries = True
    m.store("f5", "k", _payload())
    assert len([h for h in ("f0", "f1", "f2", "f3", "f4", "f5") if m.get(h, "k") is not None]) == 2


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


def test_owned_textures_are_destroyed_when_the_entry_leaves() -> None:
    # A GPU render is retained out of the engine's pool, so the memo owns it.
    m = RenderMemo(_cfg(slots=2))
    evicted, overwritten, invalidated, cleared = (_FakeTexture() for _ in range(4))

    m.store("A", "k", _gpu_payload(evicted))
    m.store("B", "k", _gpu_payload(overwritten))
    m.store("B", "k2", _gpu_payload(invalidated))  # same frame, new edit
    assert overwritten.destroyed == 1

    m.store("C", "k", _gpu_payload(cleared))  # evicts A, the least recent
    assert evicted.destroyed == 1

    m.invalidate("B")
    assert invalidated.destroyed == 1
    m.clear()
    assert cleared.destroyed == 1


def test_restoring_a_frame_from_the_memo_does_not_free_its_own_texture() -> None:
    # Away, back (served from the memo), away again: the same texture is re-filed.
    m = RenderMemo(_cfg())
    tex = _FakeTexture()
    m.store("A", "k", _gpu_payload(tex))
    payload = m.get("A", "k")
    assert payload is not None
    m.store("A", "k", payload)
    assert tex.destroyed == 0
    assert m.get("A", "k") is payload


def test_rekey_keeps_the_texture_alive() -> None:
    m = RenderMemo(_cfg())
    tex = _FakeTexture()
    m.store("A", "pre-bounds", _gpu_payload(tex))
    m.rekey("A", "post-bounds")
    assert tex.destroyed == 0
    assert m.get("A", "post-bounds") is not None


def test_array_payloads_are_left_alone() -> None:
    # The test-strip memo shares this class and stores plain mosaics.
    m = RenderMemo(_cfg())
    m.store("A", "k", _payload())
    m.store("A", "k2", _payload())
    m.clear()  # must not raise: ndarray has no destroy()
