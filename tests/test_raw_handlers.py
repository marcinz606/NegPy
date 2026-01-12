import unittest
import numpy as np
from src.infrastructure.loaders.tiff_loader import NonStandardFileWrapper


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


if __name__ == "__main__":
    unittest.main()
