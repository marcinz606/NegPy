import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.session import UNCROPPED_PREVIEW_TOOLS, ToolMode
from negpy.domain.models import WorkspaceConfig


def _config_with_crop() -> WorkspaceConfig:
    from dataclasses import replace

    cfg = WorkspaceConfig()
    return replace(cfg, geometry=replace(cfg.geometry, crop_rect=(0.25, 0.25, 0.75, 0.75)))


class TestNegativePeekFramesLikeTheTools(unittest.TestCase):
    def _paint(self, tool: ToolMode) -> dict:
        from negpy.desktop.controller import AppController

        state = SimpleNamespace(
            preview_raw=np.full((40, 60, 3), 0.5, dtype=np.float32),
            config=_config_with_crop(),
            original_res=(40, 60),
            active_tool=tool,
            preview_cam_xyz=None,
            preview_camera_wb=None,
            metrics_lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
            last_metrics={},
        )
        stub = SimpleNamespace(state=state, image_updated=MagicMock())
        AppController._paint_negative_peek(stub)
        return state.last_metrics

    def test_every_uncropped_tool_peeks_the_whole_frame(self):
        for tool in UNCROPPED_PREVIEW_TOOLS:
            metrics = self._paint(tool)
            self.assertTrue(metrics["crop_preview_full"], tool)
            self.assertEqual(metrics["base_positive"].shape[:2], (40, 60), tool)

    def test_a_plain_peek_is_cropped_and_says_so(self):
        metrics = self._paint(ToolMode.NONE)
        self.assertFalse(metrics["crop_preview_full"])
        self.assertLess(metrics["base_positive"].shape[0], 40)


class TestBorderGateReadsTheBuffersFlag(unittest.TestCase):
    def _update(self, metrics: dict) -> MagicMock:
        from dataclasses import replace

        from negpy.desktop.view.main_window import MainWindow

        cfg = WorkspaceConfig()
        cfg = replace(cfg, finish=replace(cfg.finish, border_size=1.0))
        stub = SimpleNamespace(
            state=SimpleNamespace(
                uploaded_files=[{"hash": "h1"}],
                last_metrics=metrics,
                gpu_enabled=False,
                config=cfg,
                # The live tool contradicts the buffer: the gate must not read it.
                active_tool=ToolMode.NONE,
            ),
            empty_state=MagicMock(),
            controller=MagicMock(display_transform_params=MagicMock(return_value=("sRGB", None, False))),
            canvas=MagicMock(),
        )
        MainWindow._on_image_updated(stub)
        return stub.canvas.update_buffer

    def test_a_crop_preview_buffer_is_not_padded(self):
        buf = np.full((200, 300, 3), 0.5, dtype=np.float32)
        update = self._update({"base_positive": buf, "crop_preview_full": True})
        self.assertEqual(update.call_args[0][0].shape, buf.shape)
        self.assertIsNone(update.call_args.kwargs["content_rect"])

    def test_a_plain_buffer_still_gets_the_border(self):
        buf = np.full((200, 300, 3), 0.5, dtype=np.float32)
        update = self._update({"base_positive": buf, "crop_preview_full": False})
        rect = update.call_args.kwargs["content_rect"]
        self.assertIsNotNone(rect)
        self.assertGreater(rect[0], 0)


class TestTheToolSetIsSpelledOnce(unittest.TestCase):
    def test_no_hand_spelled_copy_outside_session(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "negpy"
        offenders = [
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if p.name != "session.py" and "ToolMode.CROP_MANUAL, ToolMode.ANALYSIS_DRAW" in p.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [], "spell the set once: UNCROPPED_PREVIEW_TOOLS in session.py")


if __name__ == "__main__":
    unittest.main()
