import json
import logging
import unittest
from dataclasses import replace
from negpy.domain.models import ExportResolutionMode, WorkspaceConfig
from negpy.features.process.models import ProcessMode
from negpy.kernel.caching.logic import calculate_config_hash


class TestConfigDeserialization(unittest.TestCase):
    def test_basic_deserialization(self):
        data = {
            "process_mode": ProcessMode.BW,
            "density": 1.2,
            "grade": 3.0,
            "export_fmt": "TIFF",
        }
        config = WorkspaceConfig.from_flat_dict(data)

        self.assertEqual(config.process.process_mode, ProcessMode.BW)
        self.assertEqual(config.exposure.density, 1.2)
        # Legacy 0-5 paper grade migrates to ISO R (150 - 20*G).
        self.assertEqual(config.exposure.grade, 90.0)
        self.assertEqual(config.export.export_fmt, "TIFF")

    def test_unknown_keys_warning(self):
        data = {
            "process_mode": ProcessMode.BW,
            "density": 0.5,
            "this_is_unknown": 42,
            "also_unknown": "hello",
        }
        with self.assertLogs("negpy.domain.models", level=logging.WARNING) as cm:
            config = WorkspaceConfig.from_flat_dict(data)

        self.assertEqual(config.process.process_mode, ProcessMode.BW)
        self.assertEqual(config.exposure.density, 0.5)
        self.assertTrue(any("Dropping unknown config keys" in msg for msg in cm.output))
        self.assertIn("also_unknown", cm.output[0])
        self.assertIn("this_is_unknown", cm.output[0])

    def test_no_warning_when_all_keys_valid(self):
        data = {"process_mode": ProcessMode.C41, "density": 0.0}
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict(data)

    def test_use_original_res_true_migrates_to_original_mode(self):
        data = {"use_original_res": True, "export_print_size": 30.0}
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(config.export.export_resolution_mode, ExportResolutionMode.ORIGINAL.value)

    def test_use_original_res_false_migrates_to_print_mode(self):
        data = {"use_original_res": False, "export_print_size": 30.0}
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(config.export.export_resolution_mode, ExportResolutionMode.PRINT.value)

    def test_explicit_mode_wins_over_legacy_use_original_res(self):
        data = {
            "use_original_res": True,
            "export_resolution_mode": ExportResolutionMode.TARGET_PX.value,
        }
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(config.export.export_resolution_mode, ExportResolutionMode.TARGET_PX.value)

    def test_flatfield_apply_does_not_collide_with_rgbscan_enabled(self):
        """flatfield.apply and rgbscan.enabled must round-trip independently (#356)."""
        cfg = WorkspaceConfig(
            flatfield=replace(WorkspaceConfig().flatfield, apply=True, reference_path="/r.dng"),
            rgbscan=replace(WorkspaceConfig().rgbscan, enabled=False),
        )
        back = WorkspaceConfig.from_flat_dict(cfg.to_dict())
        self.assertTrue(back.flatfield.apply)
        self.assertFalse(back.rgbscan.enabled)

        cfg2 = WorkspaceConfig(
            flatfield=replace(WorkspaceConfig().flatfield, apply=False),
            rgbscan=replace(WorkspaceConfig().rgbscan, enabled=True),
        )
        back2 = WorkspaceConfig.from_flat_dict(cfg2.to_dict())
        self.assertFalse(back2.flatfield.apply)
        self.assertTrue(back2.rgbscan.enabled)

    def test_legacy_use_original_res_does_not_warn(self):
        data = {"use_original_res": False}
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict(data)

    def test_legacy_use_roll_average_true_splits_to_both_axes(self):
        config = WorkspaceConfig.from_flat_dict({"use_roll_average": True})
        self.assertTrue(config.process.use_luma_average)
        self.assertTrue(config.process.use_colour_average)

    def test_legacy_use_roll_average_false_splits_to_both_axes(self):
        config = WorkspaceConfig.from_flat_dict({"use_roll_average": False})
        self.assertFalse(config.process.use_luma_average)
        self.assertFalse(config.process.use_colour_average)

    def test_legacy_use_roll_average_does_not_warn(self):
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict({"use_roll_average": True})

    def test_manual_crop_rect_survives_db_roundtrip_as_tuple(self):
        """Manual crop saved to JSON reloads as a list, making the frozen
        GeometryConfig unhashable and crashing the pipeline hash. The reloaded
        rect must be a tuple and geometry must stay hashable."""
        config = WorkspaceConfig()
        config = replace(config, geometry=replace(config.geometry, manual_crop_rect=(0.1, 0.2, 0.8, 0.9)))

        # Exactly what repository.save_file_settings / load_file_settings do.
        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))

        self.assertIsInstance(reloaded.geometry.manual_crop_rect, tuple)
        self.assertEqual(reloaded.geometry.manual_crop_rect, (0.1, 0.2, 0.8, 0.9))
        hash(reloaded.geometry)  # must not raise

    def test_manual_crop_rect_hashable_in_engine_base_key(self):
        """DarkroomEngine wraps geometry in a plain tuple (base_key) before
        hashing; an unhashable geometry made calculate_config_hash fall through
        to asdict(tuple) -> 'asdict() should be called on dataclass instances'."""
        config = WorkspaceConfig()
        config = replace(config, geometry=replace(config.geometry, manual_crop_rect=(0.1, 0.2, 0.8, 0.9)))
        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))

        base_key = (
            reloaded.process.process_mode,
            reloaded.process.e6_normalize,
            reloaded.geometry,
            reloaded.process.analysis_buffer,
            reloaded.process.luma_range_clip,
        )
        self.assertIsInstance(calculate_config_hash(base_key), str)

    def test_autocrop_mode_defaults_to_image_for_legacy_dicts(self):
        config = WorkspaceConfig.from_flat_dict({"process_mode": ProcessMode.C41})
        self.assertEqual(config.geometry.autocrop_mode, "image")

    def test_autocrop_mode_survives_roundtrip(self):
        config = WorkspaceConfig()
        config = replace(config, geometry=replace(config.geometry, autocrop_mode="film"))

        reloaded = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict(), default=str)))

        self.assertEqual(reloaded.geometry.autocrop_mode, "film")
        hash(reloaded.geometry)  # must not raise

    def test_autocrop_mode_invalid_value_coerces_to_image(self):
        config = WorkspaceConfig.from_flat_dict({"autocrop_mode": "banana"})
        self.assertEqual(config.geometry.autocrop_mode, "image")


if __name__ == "__main__":
    unittest.main()
