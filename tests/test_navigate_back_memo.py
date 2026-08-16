"""Navigating back to a frame must paint its last render, with no spinner.

The memo used to hold host arrays, produced as a side effect of the soft-proof bake.
Once the proof moved to the display LUT the store site's ndarray guard stopped
matching and the memo went permanently empty. These pin the identity bookkeeping
that lets a GPU texture be retained out of the engine's pool instead.
"""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.desktop.controller import AppController
from negpy.desktop.render_memo import RenderMemo


class _FakeTexture:
    """Stands in for a GPUTexture; the engine pool owns it until a switch."""

    def __init__(self) -> None:
        self.destroyed = 0

    def destroy(self) -> None:
        self.destroyed += 1


def _stub(memo, **overrides):
    stub = SimpleNamespace(
        _is_rendering=False,
        _clear_busy_toast=MagicMock(),
        _first_render_t0=None,
        _pending_render_task=None,
        _thumb_config=object(),
        _render_memo=memo,
        _last_render_identity=None,
        _spared_texture=None,
        _gpu_fallback_notified=True,
        state=SimpleNamespace(
            config=object(),
            metrics_lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
            last_metrics={},
            current_file_hash="h1",
            compare_mode=False,
        ),
        image_updated=MagicMock(),
        _update_thumbnail_from_state=MagicMock(),
        set_status=MagicMock(),
        render_requested=MagicMock(),
    )
    # Real, not mocked: the frame guard and the queue drain are part of what is pinned here.
    stub._renders_another_frame = partial(AppController._renders_another_frame, stub)
    stub._dispatch_pending_render = partial(AppController._dispatch_pending_render, stub)
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def _finish(stub, result, **metrics):
    with patch("negpy.desktop.controller.GPUTexture", _FakeTexture):
        AppController._on_render_finished(stub, result, {"base_positive": result, "source_hash": "h1", **metrics})


def _retain(stub):
    with patch("negpy.desktop.controller.GPUTexture", _FakeTexture):
        return AppController._retain_displayed_texture(stub)


class TestGpuRenderIsMemoized(unittest.TestCase):
    def test_a_settle_render_is_filed_on_the_way_out(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        tex = _FakeTexture()
        stub = _stub(memo)

        _finish(stub, tex, memo_key="k")
        self.assertEqual(stub._last_render_identity, ("h1", "k", None))
        self.assertIsNone(memo.get("h1", "k"), "the pool still owns it mid-session")

        self.assertIs(_retain(stub), tex, "the engine cleanup must be told to spare it")
        payload = memo.get("h1", "k")
        self.assertIsNotNone(payload)
        self.assertIs(payload["base_positive"], tex)

    def test_a_frame_the_memo_cannot_reproduce_is_not_filed(self):
        # Compare peek, splash, tool preview and drag frames all clear the memo key.
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        stub = _stub(memo)
        _finish(stub, _FakeTexture(), memo_key="")
        self.assertIsNone(stub._last_render_identity)
        # Spared from cleanup so the canvas keeps showing it, but not filed: nothing to
        # file it under. Sparing and filing are separate questions.
        self.assertIsNotNone(_retain(stub))
        self.assertIsNone(memo.get("h1", ""))

    def test_a_late_render_of_the_outgoing_file_is_not_filed(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        stub = _stub(memo)
        _finish(stub, _FakeTexture(), memo_key="k", source_hash="other")
        self.assertIsNone(stub._last_render_identity)

    def test_the_cpu_path_still_files_its_array_directly(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        stub = _stub(memo)
        buf = np.zeros((4, 4, 3), dtype=np.float32)
        _finish(stub, buf, memo_key="k")
        self.assertIsNone(stub._last_render_identity, "nothing to retain from the pool")
        self.assertIs(memo.get("h1", "k")["base_positive"], buf)


class TestRetentionRefusesUnsafeFiling(unittest.TestCase):
    """A render in flight paints into the same pooled texture, so its pixels would stop
    matching the key they are filed under — it must not be *filed*.

    It is still *spared* from the cleanup. Those were one decision until the canvas was
    seen to blank during a reload: refusing to spare tells the canvas to let go, and a
    reload with no splash behind it (a merge suppresses its own) then shows nothing at all
    until the new render lands.
    """

    def test_not_filed_while_a_render_is_in_flight(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        tex = _FakeTexture()
        stub = _stub(memo)
        _finish(stub, tex, memo_key="k")
        stub._is_rendering = True
        self.assertIs(_retain(stub), tex, "still spared, so the canvas keeps showing it")
        self.assertIsNone(memo.get("h1", "k"), "but not filed: the pixels are being overwritten")

    def test_not_filed_while_a_render_is_queued(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        tex = _FakeTexture()
        stub = _stub(memo)
        _finish(stub, tex, memo_key="k")
        stub._pending_render_task = object()
        self.assertIs(_retain(stub), tex)
        self.assertIsNone(memo.get("h1", "k"))

    def test_the_identity_is_spent_once(self):
        # Two switches with no render between them: the second must not re-file the
        # first frame's identity over whatever is on screen now. It may still spare it.
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        stub = _stub(memo)
        tex = _FakeTexture()
        _finish(stub, tex, memo_key="k")
        self.assertIsNotNone(_retain(stub))
        memo._store.clear() if hasattr(memo, "_store") else None
        self.assertIs(_retain(stub), tex, "spared again")
        self.assertIsNone(stub._last_render_identity, "but the identity is gone, so nothing is re-filed")


if __name__ == "__main__":
    unittest.main()


class TestSparingSurvivesBackToBackReloads(unittest.TestCase):
    """Dragging the render exposure reloads faster than renders complete.

    `load_file` pops base_positive, so the second reload finds nothing to spare while the
    canvas is still showing the texture spared on the first. Reporting nothing there tells
    the canvas to let go — and a merge suppresses its own splash, so it then paints nothing
    at all. Measured before the fix: 13 of 24 samples blank mid-drag with HQ preview on.
    """

    def test_the_spared_texture_is_offered_again_when_metrics_are_empty(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        tex = _FakeTexture()
        stub = _stub(memo)
        _finish(stub, tex, memo_key="k")
        self.assertIs(_retain(stub), tex)

        # what load_file does on the way out, before the next reload asks again
        stub.state.last_metrics.pop("base_positive", None)
        self.assertIs(_retain(stub), tex, "the canvas is still showing it — keep sparing it")

    def test_it_is_dropped_once_nothing_is_displayed(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        stub = _stub(memo)
        stub._spared_texture = None
        stub.state.last_metrics.pop("base_positive", None)
        self.assertIsNone(_retain(stub), "nothing on screen, nothing to spare")

    def test_a_new_render_replaces_what_is_spared(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        first, second = _FakeTexture(), _FakeTexture()
        stub = _stub(memo)
        _finish(stub, first, memo_key="k")
        self.assertIs(_retain(stub), first)
        _finish(stub, second, memo_key="k2")
        self.assertIs(_retain(stub), second, "the newer frame is what the canvas shows now")
