"""Pins and input checks of the alternative process kernels."""

import negpy_core
import numpy as np
import pytest
from core_pin import digest, linux_pin

from negpy.features.cyanotype.logic import apply_cyanotype


@linux_pin
def test_cyanotype_output_is_pinned():
    rng = np.random.default_rng(23)
    dark = rng.random((40, 50, 3), dtype=np.float32)
    images = [rng.random((97, 131, 3), dtype=np.float32), dark * dark * dark]
    outs = [
        apply_cyanotype(img, args[0], True, *args[1:])
        for img in images
        for args in [("classic", 0.0, 1.4, 0.0, 0.0), ("new", 1.0, 2.3, 0.6, 0.4)]
    ]
    assert digest(*outs) == "ddbdef09e6ec49a030ca26c5dde1fb667a3616a39c0f068c22b30cdb0aa9d57b"


def test_sabattier_rejects_a_bad_image():
    with pytest.raises(ValueError):
        negpy_core.sabattier(np.zeros((4, 4), np.float32), (np.float32(1.0),) * 5, np.zeros(0, np.float32))
