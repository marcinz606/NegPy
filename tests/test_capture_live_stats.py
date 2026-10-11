import numpy as np

from negpy.services.capture.live_stats import ZEBRA_HIGH, ZEBRA_LOW, clip_masks


def test_clip_masks_read_the_ends_of_the_range():
    rgb = np.full((1, 4, 3), 128, np.uint8)
    rgb[0, 0] = (ZEBRA_HIGH, 0, 0)  # one channel at the top is a clipped highlight
    rgb[0, 1] = (ZEBRA_LOW, ZEBRA_LOW, ZEBRA_LOW)  # every channel at the bottom is a clipped black
    rgb[0, 2] = (ZEBRA_LOW, 50, ZEBRA_LOW)  # one channel up is not

    highlights, blacks = clip_masks(rgb)

    assert highlights.tolist() == [[True, False, False, False]]
    assert blacks.tolist() == [[False, True, False, False]]
