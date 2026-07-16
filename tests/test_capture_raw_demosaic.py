"""Linear calibration demosaic tests with a fake rawpy file seam."""

import numpy as np
import rawpy

from negpy.infrastructure.capture.raw_demosaic import linear_demosaic, raw_channel_clip_fraction


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


class _FullRoi:
    """The whole frame, duck-typing calibration's Roi (no services import in an infra test)."""

    def pixels(self, w, h):
        return 0, 0, w, h


class _FakeBayer:
    """A uniform base patch on a Bayer sensor, with a controllable white level."""

    color_desc = b"RGBG"

    def __init__(self, white_level, base=3000.0, sigma=4.0, clipped_rows=0):
        self.white_level = white_level
        rng = np.random.default_rng(7)
        img = base + rng.normal(0.0, sigma, (64, 64))
        if clipped_rows:  # pin some photosites to the ceiling — genuine clipping
            img[:clipped_rows] = 16383
        self.raw_image_visible = img.astype(np.uint16)
        self.raw_colors_visible = np.zeros((64, 64), dtype=np.uint8)  # every site is "R"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


def test_linear_demosaic_disables_half_size_for_xtrans(monkeypatch):
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeRaw())

    decoded = linear_demosaic("frame.RAF", half_size=True)

    assert decoded.shape == (8, 8, 3)


def test_linear_demosaic_pins_the_scale_to_the_white_level(monkeypatch):
    # THE decode fix: LibRaw's default (adjust_maximum_thr=0.75) silently rescales each frame by its
    # own brightest pixel, so a meter comparing frames compares nothing — on the rig it faked a hard
    # LED plateau. The calibration decode must pin the reference to the camera's white level.
    seen = {}

    class _Spy(_FakeRaw):
        def postprocess(self, **kwargs):
            seen.update(kwargs)
            return super().postprocess(**kwargs)

    monkeypatch.setattr(rawpy, "imread", lambda _path: _Spy())
    linear_demosaic("frame.ARW")
    assert seen["adjust_maximum_thr"] == 0.0


def test_raw_clip_returns_zero_when_the_body_reports_no_white_level(monkeypatch):
    # The documented contract ("Returns 0.0 if the channel/white level can't be resolved") — the code
    # used to guess img.max() instead, an image-dependent reference (the adjust_maximum_thr failure
    # class). On a uniform base the guess sits inside the noise, so the quieter the sensor the more
    # photosites read as clipped (~40 % at σ=4 DN): the probe then halves a frame that clips nowhere.
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeBayer(white_level=None))
    assert raw_channel_clip_fraction("x.ARW", 0, _FullRoi()) == 0.0


def test_raw_clip_still_catches_genuine_clipping(monkeypatch):
    # The positive path must survive the contract fix: photosites at the ceiling are reported.
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeBayer(white_level=16383, clipped_rows=8))
    frac = raw_channel_clip_fraction("x.ARW", 0, _FullRoi())
    assert frac > 0.1  # 8 of 64 rows pinned to the ceiling
    # And a clean frame with a proper white level reads ~0, not noise-tail false positives.
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeBayer(white_level=16383))
    assert raw_channel_clip_fraction("x.ARW", 0, _FullRoi()) == 0.0
