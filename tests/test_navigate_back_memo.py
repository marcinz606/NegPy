"""Navigating back to a frame must paint its last render, with no spinner.

The memo used to hold host arrays, produced as a side effect of the soft-proof bake.
Once the proof moved to the display LUT the store site's ndarray guard stopped
matching and the memo went permanently empty. These pin the identity bookkeeping
that lets a GPU texture be retained out of the engine's pool instead.
"""

import unittest
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
        _gpu_fallback_notified=True,
        state=SimpleNamespace(
            config=object(),
            metrics_lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
            last_metrics={},
            current_file_hash="h1",
        ),
        image_updated=MagicMock(),
        _update_thumbnail_from_state=MagicMock(),
        set_status=MagicMock(),
        render_requested=MagicMock(),
    )
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
        self.assertIsNone(_retain(stub))

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


class TestRetentionRefusesUnsafeTextures(unittest.TestCase):
    """A render in flight paints into the same pooled texture, so its pixels would
    stop matching the key they are filed under."""

    def test_refused_while_a_render_is_in_flight(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        tex = _FakeTexture()
        stub = _stub(memo)
        _finish(stub, tex, memo_key="k")
        stub._is_rendering = True
        self.assertIsNone(_retain(stub))
        self.assertIsNone(memo.get("h1", "k"))

    def test_refused_while_a_render_is_queued(self):
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        tex = _FakeTexture()
        stub = _stub(memo)
        _finish(stub, tex, memo_key="k")
        stub._pending_render_task = object()
        self.assertIsNone(_retain(stub))

    def test_the_identity_is_spent_once(self):
        # Two switches with no render between them: the second must not re-file the
        # first frame's identity over whatever is on screen now.
        memo = RenderMemo(SimpleNamespace(preview_cache_max_full_res_entries=2))
        stub = _stub(memo)
        _finish(stub, _FakeTexture(), memo_key="k")
        self.assertIsNotNone(_retain(stub))
        self.assertIsNone(_retain(stub))


if __name__ == "__main__":
    unittest.main()
