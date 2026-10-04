import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import tifffile
from PIL import Image

from negpy.domain.models import ExportResolutionMode, WorkspaceConfig
from negpy.desktop.workers.export import ContactSheetJob, ExportWorker
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.services.export.contact_sheet_layout import ContactSheetSettings, SheetFormat
from negpy.services.export.contact_sheet_roll import FrameFacts, SheetFrame, SheetLook
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


def _job(n: int, out_dir: str, **kwargs) -> ContactSheetJob:
    frames = tuple(
        SheetFrame({"path": f"/src/f{i}.raw", "name": f"f{i}.raw", "hash": f"h{i}"}, WorkspaceConfig(), FrameFacts(scan_size=(3000, 2000)))
        for i in range(n)
    )
    settings = kwargs.pop("settings", ContactSheetSettings(dpi=150))
    return ContactSheetJob(frames, SheetFormat.FULL_FRAME, "6×6", settings, SheetLook(), out_dir, **kwargs)


class TestContactSheetWorker(unittest.TestCase):
    def _worker_with(self, tile_results):
        worker = ExportWorker()
        worker._processor = MagicMock()
        worker._processor.render_display_array.side_effect = tile_results
        return worker

    def _run(self, worker, job):
        signals: dict[str, list] = {"error": [], "finished": [], "cancelled": [], "written": []}
        worker.error.connect(signals["error"].append)
        worker.finished.connect(lambda: signals["finished"].append(True))
        worker.cancelled.connect(lambda: signals["cancelled"].append(True))
        worker.contact_sheet_written.connect(signals["written"].append)
        worker.run_contact_sheet(job)
        return signals

    def test_failed_tiles_emit_an_error_each_and_print_blank(self):
        good = np.full((20, 30, 3), 200, dtype=np.uint8)
        worker = self._worker_with([good, None, None, good])
        with tempfile.TemporaryDirectory() as out:
            signals = self._run(worker, _job(4, out))
            written = [f for f in os.listdir(out) if f.endswith(".jpg")]
        self.assertEqual(len(signals["error"]), 2)
        self.assertTrue(all("contact sheet" in e for e in signals["error"]))
        self.assertEqual(written, ["contact_sheet.jpg"])
        self.assertEqual(signals["finished"], [True])

    def test_tiles_render_without_the_print_layout(self):
        good = np.zeros((20, 30, 3), dtype=np.uint8)
        worker = self._worker_with([good])
        config = WorkspaceConfig()
        config = replace(config, finish=replace(config.finish, border_size=2.0, carrier_width=1.0))
        with tempfile.TemporaryDirectory() as out:
            job = _job(1, out)
            job = replace(job, frames=(replace(job.frames[0], config=config),))
            self._run(worker, job)
        params = worker._processor.render_display_array.call_args.args[1]
        self.assertEqual((params.finish.border_size, params.finish.carrier_width), (0.0, 0.0))
        self.assertEqual(params.export.export_resolution_mode, ExportResolutionMode.ORIGINAL)
        # A 36 mm frame at 150 dpi is 213 px; the render is a little larger than the window.
        self.assertEqual(worker._processor.render_display_array.call_args.kwargs["target_long_px"], 319)

    def test_sheet_carries_its_true_size(self):
        good = np.zeros((20, 30, 3), dtype=np.uint8)
        worker = self._worker_with([good, good])
        with tempfile.TemporaryDirectory() as out:
            signals = self._run(worker, _job(2, out))
            with Image.open(os.path.join(out, "contact_sheet.jpg")) as sheet:
                size, dpi, icc = sheet.size, sheet.info.get("dpi"), sheet.info.get("icc_profile")
        self.assertEqual(size, (1425, 1800))
        self.assertEqual(tuple(round(v) for v in dpi), (150, 150))
        self.assertTrue(icc)
        self.assertEqual(signals["written"], [out])

    def test_a_multi_sheet_run_gets_one_free_suffix(self):
        good = np.zeros((20, 30, 3), dtype=np.uint8)
        small = ContactSheetSettings(paper_width=203.2, paper_height=254.0, dpi=150)
        with tempfile.TemporaryDirectory() as out:
            open(os.path.join(out, "contact_sheet_1of2.jpg"), "w").close()
            worker = self._worker_with([good] * 38)
            self._run(worker, _job(38, out, settings=small))
            written = sorted(f for f in os.listdir(out) if f.endswith(".jpg"))
        self.assertEqual(written, ["contact_sheet_1of2.jpg", "contact_sheet_2_1of2.jpg", "contact_sheet_2_2of2.jpg"])

    def test_cancel_leaves_no_partial_set(self):
        good = np.zeros((20, 30, 3), dtype=np.uint8)
        small = ContactSheetSettings(paper_width=203.2, paper_height=254.0, dpi=150)
        worker = self._worker_with([good] * 38)

        def render(*args, **kwargs):
            if worker._processor.render_display_array.call_count == 34:
                worker.cancel()
            return good

        worker._processor.render_display_array.side_effect = render
        with tempfile.TemporaryDirectory() as out:
            signals = self._run(worker, _job(38, out, settings=small))
            left = os.listdir(out)
        self.assertEqual(signals["cancelled"], [True])
        self.assertEqual(left, [])

    def test_an_unexpected_failure_still_releases_the_lane(self):
        worker = self._worker_with([np.zeros((20, 30, 3), dtype=np.uint8)])
        with (
            tempfile.TemporaryDirectory() as out,
            patch("negpy.desktop.workers.export.ContactSheetService.render_sheet", side_effect=RuntimeError("boom")),
        ):
            signals = self._run(worker, _job(1, out))
            left = os.listdir(out)
        self.assertEqual(signals["error"], ["boom"])
        self.assertEqual(signals["finished"], [True])
        self.assertEqual(left, [])

    def test_the_second_half_of_a_scan_reuses_its_decode(self):
        good = np.zeros((20, 15, 3), dtype=np.uint8)
        worker = self._worker_with([good, good, good])
        frames = tuple(
            SheetFrame({"path": path, "name": name, "hash": name, "half": half}, WorkspaceConfig(), FrameFacts())
            for path, name, half in (("/s/a.tif", "a1", 1), ("/s/a.tif", "a2", 2), ("/s/b.tif", "b1", 1))
        )
        with tempfile.TemporaryDirectory() as out:
            job = replace(_job(1, out), frames=frames, format=SheetFormat.HALF_FRAME)
            self._run(worker, job)
        keeps = [call.kwargs["keep_source"] for call in worker._processor.render_display_array.call_args_list]
        self.assertEqual(keeps, [True, False, False])

    def test_gpu_resources_released_once_per_batch(self):
        """The texture pool is evacuated per batch, not per tile — rebuilding it
        every frame would cost more than it saves now that tiles are small."""
        good = np.zeros((10, 15, 3), dtype=np.uint8)
        worker = self._worker_with([good, good, good])
        with tempfile.TemporaryDirectory() as out:
            self._run(worker, _job(3, out))
        self.assertEqual(worker._processor.cleanup.call_count, 1)


if __name__ == "__main__":
    unittest.main()
