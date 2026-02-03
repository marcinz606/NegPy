import unittest
from negpy.domain.models import WorkspaceConfig
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


if __name__ == "__main__":
    unittest.main()
