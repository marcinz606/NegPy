import unittest
import os
import tempfile
from PIL import Image
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.kernel.system.config import APP_CONFIG


class TestAssetStore(unittest.TestCase):
    def setUp(self):
        self.store = LocalAssetStore(APP_CONFIG.cache_dir, APP_CONFIG.user_icc_dir)
        self.store.initialize()

    def test_get_session_dir(self):
        s_id = "test_sess"
        s_dir = self.store._get_session_dir(s_id)
        self.assertTrue(os.path.exists(s_dir))
        self.assertIn(s_id, s_dir)


class TestThumbnailCacheSize(unittest.TestCase):
    """The thumbnail cache is keyed on file hash alone, so a thumb written before
    APP_CONFIG.thumbnail_size changed would be served forever and upscaled into the
    current cell. A size mismatch has to read as a cache miss."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = LocalAssetStore(self._tmp.name, os.path.join(self._tmp.name, "icc"))
        self.store.initialize()

    def test_thumbnail_at_current_size_is_returned(self):
        ts = APP_CONFIG.thumbnail_size
        self.store.save_thumbnail("h_current", Image.new("RGB", (ts, ts), (10, 20, 30)))
        got = self.store.get_thumbnail("h_current")
        self.assertIsNotNone(got)
        self.assertEqual(max(got.size), ts)

    def test_thumbnail_cached_at_a_stale_size_is_a_miss(self):
        stale = APP_CONFIG.thumbnail_size // 2
        self.store.save_thumbnail("h_stale", Image.new("RGB", (stale, stale), (10, 20, 30)))
        self.assertIsNone(self.store.get_thumbnail("h_stale"))

    def test_missing_thumbnail_is_a_miss(self):
        self.assertIsNone(self.store.get_thumbnail("h_absent"))


if __name__ == "__main__":
    unittest.main()
