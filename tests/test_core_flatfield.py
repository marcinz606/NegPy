"""Hash pins of the flat-field reference kernels on fixed inputs."""

import numpy as np
from core_pin import digest, linux_pin

from negpy.features.flatfield import logic as F


def _reference(shape=(170, 256), seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x - w / 2) / w) ** 2 + ((y - h / 2) / h) ** 2
    ref = (0.8 * (1.0 - 0.9 * r2))[..., None] * np.array([1.0, 0.95, 0.9], dtype=np.float32)
    ref = ref + 0.01 * rng.standard_normal(ref.shape)
    ref[:, : w // 20] *= 0.1
    ref[5:8, 100:104] = 0.99  # clipped
    return ref.astype(np.float32)


@linux_pin
def test_lit_mask_pin():
    assert digest(F._lit_mask(_reference())) == "385092d5960659d60d6aed91f2b229fb09138531a9b9bf464a8860a57c5412d5"
    dark = np.zeros((40, 40, 3), dtype=np.float32)
    assert F._lit_mask(dark).tobytes() == np.ones((40, 40), np.float32).tobytes()


@linux_pin
def test_evenness_pin():
    ratio, evenness = F._evenness(_reference(seed=1))
    assert digest(ratio, evenness) == "57a7d729bbcc37f80b8247c580f68a95522ec58fc95a857a0f31fd02d4376768"


@linux_pin
def test_compute_gain_pin():
    assert digest(F.compute_gain(_reference((400, 600), seed=2))) == "694f2fb56aa25ce7cfbbeb942bcaef06ea74d254f36454f7bb7942ea1bf1fd48"


@linux_pin
def test_fill_from_nearest_pin():
    rng = np.random.default_rng(3)
    values = rng.random((50, 70, 3), dtype=np.float32)
    valid = rng.random((50, 70)) > 0.6
    assert digest(F._fill_from_nearest(values, valid)) == "7a759c551a1368dd7612e5d839e10b66fc2c6ca8fd9e4fd09c899973921cadbf"
