import unittest
import numpy as np
import rawpy
from negpy.infrastructure.loaders.tiff_loader import NonStandardFileWrapper
from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm


class _FakeRaw:
    def __init__(self, raw_type: rawpy.RawType, block_size: int) -> None:
        self.raw_type = raw_type
        self.raw_pattern = np.zeros((block_size, block_size), dtype=np.uint8)


class TestRawHandlers(unittest.TestCase):
    def test_pakon_detection(self):
        pass

    def test_non_standard_wrapper(self):
        data = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        wrapper = NonStandardFileWrapper(data)

        with wrapper as raw:
            processed = raw.postprocess(output_bps=16)
            self.assertEqual(processed.dtype, np.uint16)
            self.assertAlmostEqual(np.mean(processed), 32767, delta=100)

    def test_xtrans_full_res_uses_dht_not_vng(self):
        # VNG produces dot/maze artifacts on X-Trans's 6x6 CFA in high-contrast
        # regions (see issue #272). DHT is the LGPL-clean algorithm built for X-Trans.
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=6)
        self.assertEqual(get_best_demosaic_algorithm(raw, for_preview=False), rawpy.DemosaicAlgorithm.DHT)

    def test_xtrans_preview_uses_linear(self):
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=6)
        self.assertEqual(get_best_demosaic_algorithm(raw, for_preview=True), rawpy.DemosaicAlgorithm.LINEAR)

    def test_bayer_full_res_uses_ahd(self):
        raw = _FakeRaw(rawpy.RawType.Flat, block_size=2)
        self.assertEqual(get_best_demosaic_algorithm(raw, for_preview=False), rawpy.DemosaicAlgorithm.AHD)


if __name__ == "__main__":
    unittest.main()
