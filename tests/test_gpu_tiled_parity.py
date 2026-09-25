"""Tiled export must render what the untiled preview path renders.

`_process_tiled` replays geometry on the CPU and hands every global meter to the
tiles, so a setting the replay forgets is dropped from exports only.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskKey, MaskShape
from negpy.features.process.models import ProcessMode
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.rendering.gpu_engine import GPUEngine


def _negative(h: int, w: int) -> np.ndarray:
    """A C-41-ish frame: a ramp plus smooth structure. Band-limited on purpose — the
    tiles resample geometry twice where the shader samples once, so per-pixel noise
    would swamp the parity these tests measure."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (xx / w) * 0.5 + (yy / h) * 0.25
    blobs = 0.08 * np.sin(xx / 90.0) * np.cos(yy / 40.0) + 0.05 * np.sin((xx + yy) / 160.0)
    img = np.empty((h, w, 3), dtype=np.float32)
    img[..., 0] = 0.55 + ramp * 0.30 + blobs
    img[..., 1] = 0.35 + ramp * 0.25 + blobs * 0.7
    img[..., 2] = 0.15 + ramp * 0.20 + blobs * 0.4
    return np.clip(img, 1e-4, 1.0)


def _base() -> WorkspaceConfig:
    s = WorkspaceConfig()
    return replace(
        s,
        process=replace(s.process, process_mode=ProcessMode.C41),
        export=replace(s.export, export_resolution_mode="original"),
    )


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuTiledParity(unittest.TestCase):
    def setUp(self):
        self.engine = GPUEngine()
        # Wider than TILE_SIZE, so the export spans more than one tile.
        self.img = _negative(300, 2400)

    def tearDown(self):
        self.engine.destroy_all()

    def _tiled(self, settings) -> np.ndarray:
        res, _ = self.engine._process_tiled(self.img, settings, scale_factor=1.0)
        return res

    def _direct(self, settings) -> np.ndarray:
        tex, _ = self.engine.process_to_texture(self.img, settings, scale_factor=1.0, apply_layout=False)
        return self.engine._readback_downsampled(tex)

    def _assert_parity(self, settings, msg, tol=0.0005):
        tiled, direct = self._tiled(settings), self._direct(settings)
        self.assertEqual(tiled.shape, direct.shape)
        self.assertLess(float(np.abs(tiled - direct).mean()), tol, msg)

    def _assert_changes_export(self, settings, msg, tol=0.01):
        """The control must move the tiled render, or parity holds trivially."""
        diff = float(np.abs(self._tiled(settings) - self._tiled(_base())).mean())
        self.assertGreater(diff, tol, msg)

    def test_tiled_applies_keystone(self):
        base = _base()
        settings = replace(base, geometry=replace(base.geometry, converge_v=8.0, converge_h=-5.0))
        self._assert_changes_export(settings, "Tilt/Swing did nothing to the tiled export")
        self._assert_parity(settings, "Tiled export dropped Tilt/Swing")

    def test_tiled_applies_contrast_mask(self):
        base = _base()
        settings = replace(base, exposure=replace(base.exposure, contrast_mask=0.4))
        self._assert_changes_export(settings, "Contrast Mask did nothing to the tiled export")
        self._assert_parity(settings, "Tiled export dropped the Contrast Mask")

    def test_tiled_local_mask_follows_keystone(self):
        base = _base()
        settings = replace(
            base,
            geometry=replace(base.geometry, converge_v=8.0),
            local=LocalAdjustmentsConfig(
                masks=(LocalMask(vertices=((0.25, 0.5), (0.6, 0.5)), stops=1.5, shape=MaskShape.GRADIENT),),
            ),
        )
        self._assert_changes_export(settings, "The dodge/burn mask did nothing to the tiled export")
        self._assert_parity(settings, "Tiled export placed the dodge/burn mask on an uncorrected frame")

    def test_tiled_applies_a_tone_limited_mask(self):
        base = _base()
        mask = LocalMask(
            vertices=((0.2, 0.5), (0.7, 0.5)), stops=1.5, grade=-20.0, shape=MaskShape.GRADIENT, key=MaskKey.HIGHLIGHTS, key_zone=5.0
        )
        settings = replace(base, local=LocalAdjustmentsConfig(masks=(mask,)))
        self._assert_changes_export(settings, "The tone-limited mask did nothing to the tiled export")
        self._assert_parity(settings, "Tiled export dropped the tone limit")

    def test_tiled_matches_untiled_with_every_stage_live(self):
        """One frame with a control on in each stage: geometry, exposure, local, mask,
        clahe, lab, toning and finish all have to survive the tiling."""
        base = _base()
        settings = replace(
            base,
            geometry=replace(base.geometry, rotation=1, flip_horizontal=True, fine_rotation=1.5, converge_v=6.0, converge_h=-4.0),
            exposure=replace(base.exposure, contrast_mask=0.35, grade=140.0, wb_cyan=0.1, shadow_density=0.2),
            local=LocalAdjustmentsConfig(
                masks=(LocalMask(vertices=((0.3, 0.3), (0.7, 0.7)), stops=1.0, grade=15.0, shape=MaskShape.GRADIENT),),
            ),
            lab=replace(base.lab, saturation=1.2, clahe_strength=0.4, sharpen=0.6, glow_amount=0.3),
            toning=replace(base.toning, selenium_strength=0.5, shadow_tint_hue=200.0, shadow_tint_strength=0.3),
            finish=replace(base.finish, vignette_stops=0.8, vignette_size=0.4),
        )
        # Looser than the focused tests: the tiles share the CLAHE CDF the preview-sized
        # meter render built, where the untiled reference builds its own at full size.
        self._assert_parity(settings, "Tiled export diverged from the preview pipeline", tol=0.002)

    def test_tiled_clahe_cdf_is_this_image(self):
        """The tiles share one CDF. It has to come from the frame being exported, not
        from whatever the engine rendered last."""
        base = _base()
        settings = replace(base, lab=replace(base.lab, clahe_strength=0.6))
        clean = self._tiled(settings)

        other = np.clip(self.img[::-1, ::-1] * 0.6 + 0.1, 1e-4, 1.0)
        self.engine.process_to_texture(other, settings, scale_factor=1.0, apply_layout=False)
        after = self._tiled(settings)

        np.testing.assert_allclose(after, clean, atol=1e-6)


# A real camera's XYZ->camera rows and as-shot multipliers, so the transfer path's
# working-space meter grid differs from the camera-space one.
_CAM_XYZ = [[0.5271, -0.0712, -0.0347], [-0.6153, 1.3653, 0.2763], [-0.1601, 0.2366, 0.7242]]
_CAM_WB = [1856.0, 1024.0, 1744.0]


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestGpuTiledTransferParity(unittest.TestCase):
    """The transparency transfer curve meters apart from the print path: a raw slide and a
    Positive frame both meter working space against the fixed window. Auto Density and Auto
    Grade are on here, so their terms reach the render."""

    def setUp(self):
        self.engine = GPUEngine()
        # A dim positive, so an Auto Density/Auto Grade push shows in the render.
        self.img = np.clip(1.0 - _negative(300, 2400), 1e-4, 1.0) * 0.3

    def tearDown(self):
        self.engine.destroy_all()

    def _assert_parity(self, settings, msg):
        tiled, _ = self.engine._process_tiled(self.img, settings, scale_factor=1.0, cam_xyz=_CAM_XYZ, camera_wb=_CAM_WB)
        tex, _ = self.engine.process_to_texture(
            self.img, settings, scale_factor=1.0, apply_layout=False, cam_xyz=_CAM_XYZ, camera_wb=_CAM_WB
        )
        direct = self.engine._readback_downsampled(tex)
        self.assertEqual(tiled.shape, direct.shape)
        self.assertLess(float(np.abs(tiled - direct).mean()), 0.0005, msg)

    def _slide(self, **process) -> WorkspaceConfig:
        s = WorkspaceConfig()
        return replace(
            s,
            process=replace(s.process, process_mode=ProcessMode.E6, **process),
            export=replace(s.export, export_resolution_mode="original"),
        )

    def test_tiled_raw_slide_meters_like_the_preview(self):
        settings = self._slide()
        self.assertTrue(settings.exposure.auto_exposure and settings.exposure.auto_normalize_contrast)
        self._assert_parity(settings, "Tiled export metered a raw slide differently")

    def test_tiled_positive_meters_like_the_preview(self):
        self._assert_parity(self._slide(positive_source=True), "Tiled export metered a Positive frame differently")


if __name__ == "__main__":
    unittest.main()
