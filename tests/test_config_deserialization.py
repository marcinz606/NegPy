import unittest
from src.domain.models import WorkspaceConfig
from src.features.retouch.models import LocalAdjustmentConfig


class TestConfigDeserialization(unittest.TestCase):
    def test_local_adjustments_deserialization(self):
        # Mock serialized JSON data
        flat_data = {
            "process_mode": "C41",
            "dust_remove": True,
            "local_adjustments": [
                {
                    "strength": 0.5,
                    "radius": 100,
                    "feather": 0.2,
                    "luma_range": [0.0, 1.0],
                    "luma_softness": 0.1,
                    "points": [(0.1, 0.1), (0.2, 0.2)],
                }
            ],
        }

        config = WorkspaceConfig.from_flat_dict(flat_data)

        self.assertTrue(len(config.retouch.local_adjustments) > 0)
        first_adj = config.retouch.local_adjustments[0]

        self.assertIsInstance(first_adj, LocalAdjustmentConfig)
        self.assertEqual(first_adj.strength, 0.5)
        self.assertEqual(first_adj.radius, 100)
        self.assertEqual(first_adj.points, [(0.1, 0.1), (0.2, 0.2)])


if __name__ == "__main__":
    unittest.main()
