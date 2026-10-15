import numpy as np

from negpy.features.exposure.analysis import output_histogram


def test_an_8bit_frame_bins_exactly_as_its_float_image():
    rgb = np.random.default_rng(0).integers(0, 256, (120, 160, 3)).astype(np.uint8)

    fast = output_histogram(rgb)
    slow = output_histogram(rgb.astype(np.float32) / 255.0)

    assert fast is not None and slow is not None
    assert np.array_equal(fast, slow)
