import unittest
import numpy as np
from negpy.features.exposure.shadows import analyze_shadow_cast, apply_shadow_cast_correction

class TestShadowCastRemoval(unittest.TestCase):
    def test_analyze_shadow_cast(self):
        img = np.full((10, 10, 3), 0.8, dtype=np.float32)
        img[:, :, 2] = 0.9
        
        cast = analyze_shadow_cast(img, threshold=0.75)
        
        self.assertLess(cast[2], 0)
        self.assertGreater(cast[0], 0)
        self.assertAlmostEqual(sum(cast), 0, places=5)

    def test_apply_shadow_cast_correction(self):
        img = np.full((10, 10, 3), 0.8, dtype=np.float32)
        img[:, :, 2] = 0.9
    
        cast = analyze_shadow_cast(img, threshold=0.75)
        res = apply_shadow_cast_correction(img, cast, strength=1.0)
    
        res_mean = np.mean(res, axis=(0, 1))
        initial_diff = 0.1
        final_diff = abs(res_mean[2] - res_mean[0])
        self.assertLess(final_diff, initial_diff * 0.5)

    def test_correction_weighting(self):
        img_s = np.full((1, 1, 3), 0.9, dtype=np.float32)
        img_h = np.full((1, 1, 3), 0.1, dtype=np.float32)
        
        cast = (0.1, 0.1, 0.1)
        res_s = apply_shadow_cast_correction(img_s, cast, strength=1.0)
        res_h = apply_shadow_cast_correction(img_h, cast, strength=1.0)
        
        diff_s = np.mean(res_s - img_s)
        diff_h = np.mean(res_h - img_h)
        
        self.assertGreater(diff_s, diff_h)

if __name__ == "__main__":
    unittest.main()
