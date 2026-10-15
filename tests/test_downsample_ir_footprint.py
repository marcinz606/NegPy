import numpy as np

from negpy.features.retouch.logic import downsample_ir


def test_the_erode_follows_the_shape_asked_for_not_the_long_edge_target():
    plane = np.ones((300, 400), dtype=np.float32)
    plane[150, 200] = 0.0
    # A 1.25x resample onto an existing buffer; the target long edge is far smaller.
    out = downsample_ir(plane, 100, dims=(320, 240))
    assert int((out < 0.5).sum()) <= 2
