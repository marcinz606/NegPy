import unittest
import os
from src.infrastructure.storage.local_asset_store import LocalAssetStore
from src.kernel.system.config import APP_CONFIG


class TestAssetStore(unittest.TestCase):
    def setUp(self):
        self.store = LocalAssetStore(APP_CONFIG.cache_dir, APP_CONFIG.user_icc_dir)
        self.store.initialize()

    def test_get_session_dir(self):
        s_id = "test_sess"
        s_dir = self.store._get_session_dir(s_id)
        self.assertTrue(os.path.exists(s_dir))
        self.assertIn(s_id, s_dir)


if __name__ == "__main__":
    unittest.main()
