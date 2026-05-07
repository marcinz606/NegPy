import logging
import unittest
from negpy.domain.models import ExportResolutionMode, WorkspaceConfig
from negpy.features.process.models import ProcessMode


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
        self.assertEqual(config.exposure.grade, 3.0)
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

    def test_legacy_use_original_res_does_not_warn(self):
        data = {"use_original_res": False}
        with self.assertNoLogs("negpy.domain.models", level=logging.WARNING):
            WorkspaceConfig.from_flat_dict(data)


if __name__ == "__main__":
    unittest.main()
