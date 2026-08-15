import unittest

import numpy as np
from PIL import Image

from negpy.services.assets.thumbnails import preview_positive, thumbnail_cache_key


def _orange_mask_negative() -> Image.Image:
    """A C41-looking negative: orange base (R >> B) with a dark patch where the scene
    was bright, and a light patch where the scene was dark."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :] = (210, 130, 60)
    arr[:32, :] = (60, 34, 16)
    rng = np.random.default_rng(7)
    arr = np.clip(arr.astype(np.int16) + rng.integers(-6, 7, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class TestPreviewPositive(unittest.TestCase):
    def test_negative_is_inverted(self):
        """The dense half of the negative is the scene's highlight, so it must come out
        the brighter half of the positive."""
        source = np.asarray(_orange_mask_negative(), dtype=np.float32)
        result = np.asarray(preview_positive(_orange_mask_negative()), dtype=np.float32)

        self.assertGreater(source[:32].mean(), 0.0)
        self.assertLess(source[:32].mean(), source[32:].mean())
        self.assertGreater(result[:32].mean(), result[32:].mean())

    def test_orange_mask_is_neutralized(self):
        """Per-channel bounds pull the base out: the positive must not stay orange."""
        result = np.asarray(preview_positive(_orange_mask_negative()), dtype=np.float32)
        spread = result.reshape(-1, 3).mean(axis=0)
        self.assertLess(float(spread.max() - spread.min()), 40.0)

    def test_narrowband_exposure_stays_neutral(self):
        """One exposure of an RGB-scan triplet holds its picture in a single channel. The
        log stretch turned the other two channels' noise into a solid green cast."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        rng = np.random.default_rng(11)
        arr[:, :, 0] = rng.integers(40, 220, (64, 64))
        arr[:, :, 1] = rng.integers(0, 3, (64, 64))
        arr[:, :, 2] = rng.integers(0, 3, (64, 64))

        result = np.asarray(preview_positive(Image.fromarray(arr)), dtype=np.float32)

        channel_means = result.reshape(-1, 3).mean(axis=0)
        self.assertLess(float(channel_means.max() - channel_means.min()), 5.0)

    def test_transparency_is_left_alone(self):
        """A slide is already a positive; inverting it would be the bug this replaces."""
        rng = np.random.default_rng(3)
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        slide = Image.fromarray(arr)

        result = preview_positive(slide)

        self.assertTrue(np.array_equal(np.asarray(result), arr))

    def test_cache_key_retires_stored_negatives(self):
        """A library scanned before this change must not keep serving its negatives."""
        self.assertNotEqual(thumbnail_cache_key("abc", False), "abc")
        self.assertNotEqual(thumbnail_cache_key("abc", True), "abc-rgb")
        self.assertNotEqual(thumbnail_cache_key("abc", False), thumbnail_cache_key("abc", True))


if __name__ == "__main__":
    unittest.main()
