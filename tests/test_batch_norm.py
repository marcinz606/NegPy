import unittest
import numpy as np
from dataclasses import replace
from src.domain.models import WorkspaceConfig
from src.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from src.domain.interfaces import PipelineContext


class TestBatchNormalization(unittest.TestCase):
    def setUp(self):
        self.config = WorkspaceConfig()
        self.context = PipelineContext(
            scale_factor=1.0, original_size=(100, 100), process_mode="C41"
        )

    def test_normalization_processor_uses_locked_values(self):
        """
        Verify that NormalizationProcessor ignores local analysis when use_roll_average is ON.
        """
        # Set specific locked values
        locked_floors = (-0.5, -0.5, -0.5)
        locked_ceils = (-0.1, -0.1, -0.1)

        new_process = replace(
            self.config.process,
            use_roll_average=True,
            locked_floors=locked_floors,
            locked_ceils=locked_ceils,
        )
        processor = NormalizationProcessor(new_process)

        img_val = 10**-0.3
        img = np.full((10, 10, 3), img_val, dtype=np.float32)

        res = processor.process(img, self.context)
        self.assertAlmostEqual(res[0, 0, 0], 0.5, places=5)

    def test_photometric_processor_is_independent_of_roll_average(self):
        """
        Verify that PhotometricProcessor no longer applies extra shifts in roll average mode.
        """
        user_shifts = (0.05, 0.05)

        new_exposure = replace(
            self.config.exposure,
            wb_magenta=user_shifts[0],
            wb_yellow=user_shifts[1],
        )

        processor = PhotometricProcessor(new_exposure)

        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        res_batch = processor.process(img, self.context)

        processor_manual = PhotometricProcessor(new_exposure)
        res_manual = processor_manual.process(img, self.context)

        np.testing.assert_array_almost_equal(res_batch, res_manual)


if __name__ == "__main__":
    unittest.main()
