"""
TDD test: verify that loading a preview (with splash) opens the RAW file only once.

After the fix, PreviewLoadWorker.process() should call loader_factory.get_loader
exactly once per file, even when use_splash=True triggers both a splash and a
linear decode.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import rawpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_raw(w: int = 100, h: int = 100) -> MagicMock:
    """Return a mock rawpy-like object that satisfies the preview_manager API."""
    raw = MagicMock()
    raw.sizes = MagicMock(iheight=h, iwidth=w)
    raw.postprocess.return_value = np.zeros((h, w, 3), dtype=np.uint16)

    thumb = MagicMock()
    thumb.format = rawpy.ThumbFormat.BITMAP
    thumb.data = np.zeros((h, w, 3), dtype=np.uint8)
    raw.extract_thumb.return_value = thumb

    # Support use as a context manager
    raw.__enter__ = lambda s: s
    raw.__exit__ = MagicMock(return_value=False)
    return raw


def _make_loader_factory_patch(fake_raw: MagicMock, open_count: dict):
    """Return a side_effect callable for loader_factory.get_loader."""

    def fake_get_loader(path, linear_raw=False):
        open_count["n"] += 1
        metadata = {"color_space": "Adobe RGB", "orientation": 0, "raw_flip": 0}
        return fake_raw, metadata

    return fake_get_loader


# ---------------------------------------------------------------------------
# The key assertion
# ---------------------------------------------------------------------------


def test_preview_load_worker_opens_file_once(qapp):
    """
    PreviewLoadWorker.process() with use_splash=True must call
    loader_factory.get_loader exactly once — not once for splash and once for
    the linear decode.
    """
    from negpy.desktop.workers.render import PreviewLoadTask, PreviewLoadWorker
    from negpy.services.rendering.preview_manager import PreviewManager

    fake_raw = _make_fake_raw()
    open_count = {"n": 0}

    patch_target = "negpy.services.rendering.preview_manager.loader_factory.get_loader"
    with patch(patch_target, side_effect=_make_loader_factory_patch(fake_raw, open_count)):
        preview_service = PreviewManager()
        worker = PreviewLoadWorker(preview_service)

        # Wire up a dummy finished slot so the signal doesn't go nowhere
        results = []
        worker.finished.connect(lambda fp, buf, dims: results.append((fp, buf, dims)))
        splash_results = []
        worker.splash.connect(lambda fp, buf, dims: splash_results.append((fp, buf, dims)))

        task = PreviewLoadTask(
            file_path="fake.nef",
            workspace_color_space="Adobe RGB",
            use_camera_wb=False,
            full_resolution=False,
            file_hash=None,
            use_splash=True,
        )
        worker.process(task)

    assert open_count["n"] == 1, (
        f"Expected loader_factory.get_loader to be called exactly once, got {open_count['n']}. "
        "try_splash_preview and load_linear_preview must share a single file open."
    )
    # Sanity: both splash and finished were emitted
    assert len(splash_results) == 1, "Expected splash signal to be emitted once"
    assert len(results) == 1, "Expected finished signal to be emitted once"
