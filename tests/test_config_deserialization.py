import unittest
from src.domain.models import WorkspaceConfig, ProcessMode


class TestConfigDeserialization(unittest.TestCase):
    def test_basic_deserialization(self):
        # Mock serialized JSON data
        flat_data = {
            "process_mode": ProcessMode.C41,
            "dust_remove": True,
        }

        config = WorkspaceConfig.from_flat_dict(flat_data)
        self.assertEqual(config.process_mode, ProcessMode.C41)
        self.assertTrue(config.retouch.dust_remove)


if __name__ == "__main__":
    unittest.main()
