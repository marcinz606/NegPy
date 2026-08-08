"""While a gesture is live, the UI thread must not handle a full-resolution frame.

Qt runs widget input and painting on one thread, so "decoupled" can only mean that
nothing expensive runs on that thread while the user is dragging. Interactive frames
render against a preview-resolution proxy and skip every consumer that only has to be
right once the gesture settles.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.controller import _interactive_proxy
from negpy.kernel.system.config import APP_CONFIG


class TestInteractiveProxy(unittest.TestCase):
    def test_hq_buffer_gets_a_preview_sized_proxy(self):
        raw = np.zeros((4000, 6000, 3), dtype=np.float32)
        proxy = _interactive_proxy(raw)
        self.assertIsNotNone(proxy)
        self.assertEqual(max(proxy.shape[:2]), APP_CONFIG.preview_render_size)
        self.assertAlmostEqual(proxy.shape[1] / proxy.shape[0], 6000 / 4000, places=2)

    def test_preview_sized_buffer_needs_no_proxy(self):
        raw = np.zeros((1066, APP_CONFIG.preview_render_size, 3), dtype=np.float32)
        self.assertIsNone(_interactive_proxy(raw), "a proxy of the same size would be pure copying")

    def test_non_array_buffers_are_ignored(self):
        self.assertIsNone(_interactive_proxy(None))
        self.assertIsNone(_interactive_proxy(object()))


class TestSettleOnlyWorkIsSkipped(unittest.TestCase):
    def test_thumbnail_is_not_refreshed_mid_gesture(self):
        from negpy.desktop.controller import AppController

        stub = SimpleNamespace(
            _is_rendering=True,
            _clear_busy_toast=MagicMock(),
            _first_render_t0=None,
            _pending_render_task=None,
            _thumb_config=object(),
            _render_memo=MagicMock(),
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
        AppController._on_render_finished(stub, None, {"interactive": True, "source_hash": "h1"})
        stub._update_thumbnail_from_state.assert_not_called()

        stub._thumb_config = object()
        AppController._on_render_finished(stub, None, {"interactive": False, "source_hash": "h1"})
        stub._update_thumbnail_from_state.assert_called_once()

    def test_analysis_panel_skips_interactive_frames(self):
        from negpy.desktop.view.sidebar.right_panel import RightPanel

        panel = MagicMock()
        panel.controller.session.state.last_metrics = {"interactive": True}
        RightPanel._update_analysis(panel)
        panel._update_histograms.assert_not_called()

    def test_bounds_are_not_persisted_mid_gesture(self):
        from negpy.desktop.controller import AppController

        stub = SimpleNamespace(
            state=SimpleNamespace(
                metrics_lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
                last_metrics={},
                ir_degenerate=False,
                current_file_hash="h1",
                config=MagicMock(),
            ),
            metrics_available=MagicMock(),
            session=MagicMock(),
        )
        AppController._on_metrics_updated(stub, {"interactive": True, "log_bounds": object(), "source_hash": "h1"})
        stub.session.update_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
