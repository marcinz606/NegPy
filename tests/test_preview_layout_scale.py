"""The preview layout pass must never resample content above its own resolution.

The canvas quotes zoom against the buffer the pipeline was handed
(``render_long_edge``), so a layout that upscales makes 1:1 read closer than one
scan pixel per device pixel.
"""

import unittest
from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.services.rendering.gpu_engine import GPUEngine


def _dims(settings: WorkspaceConfig, cw: int, ch: int, size_ref):
    engine = GPUEngine.__new__(GPUEngine)
    return engine._calculate_layout_dims(settings, cw, ch, size_ref)


class TestPreviewLayoutScale(unittest.TestCase):
    def test_half_size_decode_is_not_upscaled(self):
        """A 3008 px preview of a 6016 px scan stays 3008 px under a 4000 px ref."""
        _, _, cw, ch, _, _, _ = _dims(WorkspaceConfig(), 3008, 2008, 4000.0)
        self.assertEqual((cw, ch), (3008, 2008))

    def test_crop_is_not_upscaled(self):
        _, _, cw, ch, _, _, _ = _dims(WorkspaceConfig(), 800, 533, 1600.0)
        self.assertEqual((cw, ch), (800, 533))

    def test_larger_content_still_scales_down_to_ref(self):
        _, _, cw, _, _, _, _ = _dims(WorkspaceConfig(), 6016, 4016, 1600.0)
        self.assertEqual(cw, 1600)

    def test_border_keeps_its_share_of_the_frame(self):
        settings = replace(WorkspaceConfig(), finish=replace(WorkspaceConfig().finish, border_size=1.0))
        pw, _, cw, _, _, _, _ = _dims(settings, 3008, 2008, 4000.0)
        clamped, _, clamped_cw, _, _, _, _ = _dims(settings, 3008, 2008, 3008.0)
        self.assertEqual(cw, 3008)
        self.assertAlmostEqual((pw - cw) / cw, (clamped - clamped_cw) / clamped_cw, places=2)

    def test_export_path_keeps_its_own_resolution(self):
        """No size_ref: the export math owns the paper, and may upscale to a print size."""
        _, _, cw, ch, _, _, _ = _dims(WorkspaceConfig(), 800, 533, None)
        self.assertEqual((cw, ch), (800, 533))


if __name__ == "__main__":
    unittest.main()
