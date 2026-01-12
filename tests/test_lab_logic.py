import unittest
import numpy as np
from src.features.lab.logic import (
    apply_output_sharpening,
)


class TestLabLogic(unittest.TestCase):
    def test_output_sharpening(self) -> None:
        """Sharpening should increase local variance."""
        # Create a simple square
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[25:75, 25:75, :] = 0.5

        res = apply_output_sharpening(img, amount=1.0, scale_factor=1.0)

        # Sharpening should increase variance on edges
        self.assertGreater(np.var(res), np.var(img))


if __name__ == "__main__":
    unittest.main()
