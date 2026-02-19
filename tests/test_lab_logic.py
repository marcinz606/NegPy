import unittest
import numpy as np
import cv2
from negpy.features.lab.logic import (
    apply_output_sharpening,
    apply_saturation,
    apply_spectral_crosstalk,
    apply_clahe,
    apply_vibrance,
    apply_chroma_denoise,
)


class TestLabLogic(unittest.TestCase):
    def test_spectral_crosstalk(self) -> None:
        """Matrix should mix channels."""
        img = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32)
        # Identity matrix
        matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        res = apply_spectral_crosstalk(img, 1.0, matrix)
        assert np.allclose(res, img)

        # Swap R and G
        matrix_swap = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        res_swap = apply_spectral_crosstalk(img, 1.0, matrix_swap)
        assert np.allclose(res_swap[0, 0], [0.5, 1.0, 0.0])

    def test_clahe(self) -> None:
        """CLAHE should modify image."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        res = apply_clahe(img, 1.0)
        assert res.shape == img.shape
        # Should be different
        assert not np.allclose(res, img)

    def test_output_sharpening(self) -> None:
        """Sharpening should increase local variance."""
        # Create a simple square
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[25:75, 25:75, :] = 0.5

        res = apply_output_sharpening(img, amount=1.0, scale_factor=1.0)

        # Sharpening should increase variance on edges
        self.assertGreater(np.var(res), np.var(img))

    def test_saturation(self) -> None:
        """Saturation should modify color intensity."""
        # Pure Red (H=0, S=1, V=1)
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :, 0] = 1.0

        # Reduce saturation to 0 (Greyscale)
        desat = apply_saturation(img, 0.0)

        # Desaturated pure primary should result in high value (white in this context for HSV)
        self.assertAlmostEqual(desat[0, 0, 0], 1.0)
        self.assertAlmostEqual(desat[0, 0, 1], 1.0)
        self.assertAlmostEqual(desat[0, 0, 2], 1.0)

        # Increase saturation of a pale color
        # Pale Red: R=1.0, G=0.5, B=0.5
        img2 = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img2[:, :, 0] = 1.0

        sat = apply_saturation(img2, 2.0)
        # S should become 1.0 -> Pure Red
        self.assertAlmostEqual(sat[0, 0, 0], 1.0, delta=1e-5)
        self.assertAlmostEqual(sat[0, 0, 1], 0.0, delta=1e-5)
        self.assertAlmostEqual(sat[0, 0, 2], 0.0, delta=1e-5)

    def test_vibrance(self) -> None:
        """Vibrance should increase saturation of pale colors more than vibrant ones."""
        # Pale color
        img_pale = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img_pale[:, :, 0] = 0.6
        
        # Vibrant color
        img_vibrant = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        img_vibrant[:, :, 0] = 1.0
        
        res_pale = apply_vibrance(img_pale, 1.5)
        res_vibrant = apply_vibrance(img_vibrant, 1.5)
        
        # Calculate saturation increase
        def get_sat(rgb):
            c = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            return np.mean(c[:, :, 1])
            
        sat_gain_pale = get_sat(res_pale) - get_sat(img_pale)
        sat_gain_vibrant = get_sat(res_vibrant) - get_sat(img_vibrant)
        
        self.assertGreater(sat_gain_pale, sat_gain_vibrant)

    def test_chroma_denoise(self) -> None:
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 1] += np.random.normal(0, 5, (100, 100))
        img_noisy = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        res = apply_chroma_denoise(img_noisy, radius=2.0)
        res_lab = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)
        
        np.testing.assert_array_almost_equal(lab[:, :, 0], res_lab[:, :, 0], decimal=0)
        self.assertLess(float(np.var(res_lab[:, :, 1])), float(np.var(lab[:, :, 1])))


if __name__ == "__main__":
    unittest.main()
