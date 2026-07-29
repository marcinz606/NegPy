"""Contact-sheet tiles must come back small, and a failed tile must be reported.

Regression: render_display_array returned the tile at full source resolution while
the worker holds every tile in memory at once, so peak cost scaled with
frames x full resolution (~72MB/tile on a 24MP roll). Once allocation failed, the
worker silently dropped the tile, so a 20-frame run could produce a sheet with a
single image while the progress dialog still counted every file.
"""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import tifffile

from negpy.domain.models import ExportResolutionMode, WorkspaceConfig
from negpy.desktop.workers.export import ExportTask, ExportWorker
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.services.rendering.image_processor import ImageProcessor, _downsample_to_long_edge


class TestDownsampleHelper(unittest.TestCase):
    def test_shrinks_long_edge_to_target(self):
        out = _downsample_to_long_edge(np.zeros((4000, 6000, 3), dtype=np.uint8), 1200)
        self.assertEqual(max(out.shape[:2]), 1200)

    def test_preserves_aspect_ratio(self):
        out = _downsample_to_long_edge(np.zeros((1000, 2000, 3), dtype=np.uint8), 500)
        self.assertEqual(out.shape[:2], (250, 500))

    def test_never_upscales(self):
        src = np.zeros((300, 400, 3), dtype=np.uint8)
        self.assertIs(_downsample_to_long_edge(src, 1200), src)

    def test_non_positive_target_is_a_noop(self):
        src = np.zeros((300, 400, 3), dtype=np.uint8)
        self.assertIs(_downsample_to_long_edge(src, 0), src)


class TestRenderDisplayArraySize(unittest.TestCase):
    """The real render path must honour target_long_px, not return full res."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls._tmp.name, "frame.tiff")
        h, w = 1600, 2400
        tifffile.imwrite(cls.path, (np.random.default_rng(0).random((h, w, 3)) * 60000).astype(np.uint16), photometric="rgb")
        cls.proc = ImageProcessor()

    @classmethod
    def tearDownClass(cls):
        cls.proc.cleanup()
        cls._tmp.cleanup()

    def _render(self, target_long_px: int, prefer_gpu: bool):
        return self.proc.render_display_array(
            self.path,
            WorkspaceConfig(),
            "hash-1",
            target_long_px=target_long_px,
            prefer_gpu=prefer_gpu,
            working_color_space=WORKING_COLOR_SPACE,
            fast_decode=True,
        )

    def test_tile_respects_target_long_px_cpu(self):
        tile = self._render(600, prefer_gpu=False)
        self.assertIsNotNone(tile)
        self.assertEqual(max(tile.shape[:2]), 600)

    def test_tile_respects_target_long_px_gpu(self):
        # Falls back to the CPU engine when no GPU is present; either way the
        # returned tile must be bounded.
        tile = self._render(600, prefer_gpu=True)
        self.assertIsNotNone(tile)
        self.assertEqual(max(tile.shape[:2]), 600)

    def test_tile_is_far_smaller_than_the_source(self):
        tile = self._render(600, prefer_gpu=False)
        source_px = 1600 * 2400
        self.assertLess(tile.shape[0] * tile.shape[1] * 8, source_px)

    def test_pipeline_runs_at_proof_scale_not_full_res(self):
        """The expensive half: rendering the full-res source and keeping only a tile
        cost ~3.5GB peak RSS per 24MP frame. The engine must receive a buffer already
        shrunk to target_long_px."""
        seen: list = []
        real = self.proc.run_pipeline

        def spy(img, *a, **kw):
            seen.append(img.shape[:2])
            return real(img, *a, **kw)

        with patch.object(self.proc, "run_pipeline", side_effect=spy):
            self._render(600, prefer_gpu=False)

        self.assertTrue(seen, "run_pipeline was never called")
        self.assertLessEqual(max(seen[0]), 600, f"pipeline got a {seen[0]} buffer for a 600px tile")

    def test_print_export_mode_does_not_inflate_the_tile(self):
        """A Print/Target-px export setting sizes the paper from print_size x DPI;
        without bounding it the proof render is blown back up to print resolution."""
        params = replace(
            WorkspaceConfig(),
            export=replace(
                WorkspaceConfig().export,
                export_resolution_mode=ExportResolutionMode.PRINT.value,
                export_print_size=60.0,
                export_dpi=600,  # ~14000px paper if unbounded
            ),
        )
        seen: list = []
        real = self.proc.run_pipeline

        def spy(img, *a, **kw):
            seen.append(img.shape[:2])
            return real(img, *a, **kw)

        with patch.object(self.proc, "run_pipeline", side_effect=spy):
            tile = self.proc.render_display_array(
                self.path,
                params,
                "hash-print",
                target_long_px=600,
                prefer_gpu=False,
                working_color_space=WORKING_COLOR_SPACE,
                fast_decode=True,
            )

        self.assertIsNotNone(tile)
        self.assertLessEqual(max(seen[0]), 600)
        # Bounded end to end: the laid-out result never exceeds the tile size either.
        self.assertLessEqual(max(tile.shape[:2]), 600)


class TestContactSheetReportsDroppedTiles(unittest.TestCase):
    """A tile that fails to render must be surfaced, not silently omitted."""

    def _worker_with(self, tile_results):
        worker = ExportWorker()
        worker._processor = MagicMock()
        worker._processor.render_display_array.side_effect = tile_results
        return worker

    def _tasks(self, n):
        return [
            ExportTask(
                file_info={"path": f"/src/f{i}.raw", "name": f"f{i}.raw", "hash": f"h{i}"},
                params=WorkspaceConfig(),
                export_settings=WorkspaceConfig().export,
            )
            for i in range(n)
        ]

    def test_failed_tiles_emit_an_error_each(self):
        good = np.zeros((10, 15, 3), dtype=np.uint8)
        worker = self._worker_with([good, None, None, good])
        errors: list[str] = []
        worker.error.connect(errors.append)

        with tempfile.TemporaryDirectory() as out:
            worker.run_contact_sheet(self._tasks(4), out, 600, 16, 32, 38, False, "#000000", "#ffffff")

        self.assertEqual(len(errors), 2)
        self.assertTrue(all("contact sheet" in e for e in errors))

    def test_all_tiles_good_reports_no_errors(self):
        good = np.zeros((10, 15, 3), dtype=np.uint8)
        worker = self._worker_with([good, good])
        errors: list[str] = []
        worker.error.connect(errors.append)

        with tempfile.TemporaryDirectory() as out:
            worker.run_contact_sheet(self._tasks(2), out, 600, 16, 32, 38, False, "#000000", "#ffffff")

            written = [f for f in os.listdir(out) if f.endswith(".jpg")]

        self.assertEqual(errors, [])
        self.assertEqual(len(written), 1)

    def test_gpu_resources_released_once_per_batch(self):
        """The texture pool is evacuated per batch, not per tile — rebuilding it
        every frame would cost more than it saves now that tiles are small."""
        good = np.zeros((10, 15, 3), dtype=np.uint8)
        worker = self._worker_with([good, good, good])

        with tempfile.TemporaryDirectory() as out:
            worker.run_contact_sheet(self._tasks(3), out, 600, 16, 32, 38, False, "#000000", "#ffffff")

        self.assertEqual(worker._processor.cleanup.call_count, 1)


if __name__ == "__main__":
    unittest.main()
