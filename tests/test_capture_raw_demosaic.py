"""Linear calibration demosaic tests with a fake rawpy file seam."""

import numpy as np
import rawpy

from negpy.infrastructure.capture.raw_demosaic import linear_demosaic


class _FakeRaw:
    raw_type = rawpy.RawType.Flat
    raw_pattern = np.zeros((6, 6), dtype=np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def postprocess(self, **kwargs):
        side = 4 if kwargs["half_size"] else 8
        return np.zeros((side, side, 3), dtype=np.uint16)


def test_linear_demosaic_disables_half_size_for_xtrans(monkeypatch):
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeRaw())

    decoded = linear_demosaic("frame.RAF", half_size=True)

    assert decoded.shape == (8, 8, 3)
