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


class TestThumbnailCacheClearing(unittest.TestCase):
    """The Manage Database dialog reports and empties the thumbnail cache."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = LocalAssetStore(self._tmp.name, os.path.join(self._tmp.name, "icc"))
        self.store.initialize()

    def _save(self, file_hash):
        ts = APP_CONFIG.thumbnail_size
        self.store.save_thumbnail(file_hash, Image.new("RGB", (ts, ts), (10, 20, 30)))

    def test_stats_report_count_and_bytes(self):
        self._save("h1")
        self._save("h2")
        count, size = self.store.thumbnail_stats()
        self.assertEqual(count, 2)
        on_disk = sum(os.path.getsize(os.path.join(self.store.thumb_dir, f)) for f in os.listdir(self.store.thumb_dir))
        self.assertEqual(size, on_disk)

    def test_stats_are_zero_without_a_cache_dir(self):
        os.rmdir(self.store.thumb_dir)
        self.assertEqual(self.store.thumbnail_stats(), (0, 0))

    def test_clear_empties_the_cache_and_leaves_it_usable(self):
        self._save("h1")
        self.store.clear_thumbnails()
        self.assertEqual(self.store.thumbnail_stats(), (0, 0))
        self.assertIsNone(self.store.get_thumbnail("h1"))
        self._save("h2")
        self.assertIsNotNone(self.store.get_thumbnail("h2"))


if __name__ == "__main__":
    unittest.main()
