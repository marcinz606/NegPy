"""Linear calibration demosaic tests with a fake rawpy file seam."""

import numpy as np
import rawpy

from negpy.infrastructure.capture.raw_demosaic import linear_demosaic, raw_channel_clip_fraction


class _FakeRaw:
    raw_type = rawpy.RawType.Flat
    raw_pattern = np.zeros((6, 6), dtype=np.uint8)
    white_level = 16383
    camera_white_level_per_channel = None  # most bodies: no calibrated table
    black_level_per_channel = [0, 0, 0, 0]

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

    def __init__(self, white_level, base=3000.0, sigma=4.0, clipped_rows=0, camera_white_level=None):
        self.white_level = white_level
        # A body without a calibrated saturation table (most of them): None, same as rawpy.
        self.camera_white_level_per_channel = [camera_white_level] * 4 if camera_white_level is not None else None
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


class _FakeBayerFromValues:
    """A single-column Bayer sensor whose R-channel raw values are exactly the array given."""

    color_desc = b"RGBG"

    def __init__(self, values, white_level, camera_white_level=None):
        self.white_level = white_level
        self.camera_white_level_per_channel = [camera_white_level] * 4 if camera_white_level is not None else None
        self.raw_image_visible = np.asarray(values, dtype=np.uint16).reshape(-1, 1)
        self.raw_colors_visible = np.zeros((len(values), 1), dtype=np.uint8)  # every site is "R"

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


def test_linear_demosaic_pins_the_scale_to_the_camera_calibrated_limit(monkeypatch):
    # A Nikon D800 reports a generic white_level of 16383 but a calibrated
    # camera_white_level_per_channel of 15311 — the sensor goes non-linear below the
    # format's nominal ADC max. Trusting the generic number lets already non-linear
    # photosites land at ~93 % of the demosaiced scale instead of 100 %, so CLIP_CEILING
    # never sees them (issue #906). user_sat must carry the calibrated value through.
    seen = {}

    class _Spy(_FakeRaw):
        white_level = 16383
        camera_white_level_per_channel = [15311, 15311, 15311, 15311]

        def postprocess(self, **kwargs):
            seen.update(kwargs)
            return super().postprocess(**kwargs)

    monkeypatch.setattr(rawpy, "imread", lambda _path: _Spy())
    linear_demosaic("frame.NEF")
    assert seen["user_sat"] == 15311


def test_linear_demosaic_falls_back_to_the_generic_white_level_without_a_calibrated_table(monkeypatch):
    # Most bodies carry no camera_white_level_per_channel table at all — user_sat must
    # still be set from *something*, not silently omitted.
    seen = {}

    class _Spy(_FakeRaw):
        white_level = 16383
        camera_white_level_per_channel = None

        def postprocess(self, **kwargs):
            seen.update(kwargs)
            return super().postprocess(**kwargs)

    monkeypatch.setattr(rawpy, "imread", lambda _path: _Spy())
    linear_demosaic("frame.NEF")
    assert seen["user_sat"] == 16383


def test_linear_demosaic_corrects_user_sat_for_a_nonzero_black_level(monkeypatch):
    # `user_sat` is compared AFTER LibRaw subtracts the black level from every photosite.
    # A D800's black level happens to be 0, which hid this: on a Fuji/Sony/Canon body with
    # a real black level, passing the raw-scale limit unmodified silently under-scales the
    # whole decode and pushes ETTR's exposure the wrong way.
    seen = {}

    class _Spy(_FakeRaw):
        white_level = 16383
        camera_white_level_per_channel = None
        black_level_per_channel = [1020, 1020, 1020, 1020]

        def postprocess(self, **kwargs):
            seen.update(kwargs)
            return super().postprocess(**kwargs)

    monkeypatch.setattr(rawpy, "imread", lambda _path: _Spy())
    linear_demosaic("frame.RAF")
    assert seen["user_sat"] == 16383 - 1020


def test_linear_demosaic_clamps_a_calibrated_limit_that_exceeds_the_format_ceiling(monkeypatch):
    # A calibrated linearity limit can't legitimately exceed the format's own ADC ceiling —
    # a body reporting one anyway is bad metadata, not a wider range, and must not widen
    # what CLIP_CEILING accepts as clean.
    seen = {}

    class _Spy(_FakeRaw):
        white_level = 16383
        camera_white_level_per_channel = [20000, 20000, 20000, 20000]

        def postprocess(self, **kwargs):
            seen.update(kwargs)
            return super().postprocess(**kwargs)

    monkeypatch.setattr(rawpy, "imread", lambda _path: _Spy())
    linear_demosaic("frame.NEF")
    assert seen["user_sat"] == 16383


def test_raw_clip_returns_zero_when_the_body_reports_no_white_level(monkeypatch):
    # The documented contract ("Returns (0.0, 0.0) if the channel/limit can't be resolved") — the
    # code used to guess img.max() instead, an image-dependent reference (the adjust_maximum_thr
    # failure class). On a uniform base the guess sits inside the noise, so the quieter the sensor
    # the more photosites read as clipped: the probe then halves a frame that clips nowhere.
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeBayer(white_level=None))
    linearity, plateau = raw_channel_clip_fraction("x.ARW", 0, _FullRoi())
    assert linearity == 0.0
    assert plateau == 0.0


def test_raw_clip_still_catches_genuine_clipping(monkeypatch):
    # The positive path must survive the contract fix: photosites at the ceiling are reported by
    # both signals — genuinely pinned rows are past the linearity limit AND form a real plateau.
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeBayer(white_level=16383, clipped_rows=8))
    linearity, plateau = raw_channel_clip_fraction("x.ARW", 0, _FullRoi())
    assert linearity > 0.1  # 8 of 64 rows pinned to the ceiling
    assert plateau > 0.1
    # And a clean frame with a proper white level reads ~0, not noise-tail false positives.
    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeBayer(white_level=16383))
    linearity, plateau = raw_channel_clip_fraction("x.ARW", 0, _FullRoi())
    assert linearity == 0.0
    assert plateau == 0.0


def test_raw_clip_uses_the_calibrated_ceiling_when_the_body_reports_a_higher_generic_one(monkeypatch):
    # A D800-shaped body: generic white_level is 16383, the real calibrated ceiling is 15311. A
    # base exposed to just under the *real* ceiling reads past the linearity limit, but reads
    # clean against the generic one alone (issue #906) — this must be caught. It is an ordinary
    # exposed base, not a real pileup, so the plateau signal correctly stays quiet either way.
    monkeypatch.setattr(
        rawpy,
        "imread",
        lambda _path: _FakeBayer(white_level=16383, base=15320.0, sigma=4.0, camera_white_level=15311),
    )
    linearity, plateau = raw_channel_clip_fraction("x.NEF", 0, _FullRoi())
    assert linearity > 0.9
    assert plateau == 0.0

    # The same photosites, checked against the generic ceiling alone, would have read as clean —
    # this is exactly what let the old code pass a clipped base as "on target".
    monkeypatch.setattr(
        rawpy,
        "imread",
        lambda _path: _FakeBayer(white_level=16383, base=15320.0, sigma=4.0, camera_white_level=None),
    )
    linearity, _plateau = raw_channel_clip_fraction("x.NEF", 0, _FullRoi())
    assert linearity == 0.0


def test_raw_clip_plateau_survives_a_noise_tail_above_the_pile(monkeypatch):
    # Real sensor data: a hard pileup at the true saturation point, with a sparse scatter of
    # noise-elevated photosites (dust, a hot pixel, ordinary shot noise) sitting up to a few
    # dozen counts above it. Anchoring on the ROI's own maximum lands in that scatter and misses
    # the pile beneath it; scanning the top window for the densest bin does not.
    rng = np.random.default_rng(3)
    pile = np.full(5000, 15778, dtype=np.int64)
    tail = 15779 + rng.integers(0, 50, size=40)
    below = np.clip(14000 + rng.normal(0, 200, size=2000), 0, None).astype(np.int64)
    values = np.concatenate([pile, tail, below])
    monkeypatch.setattr(
        rawpy,
        "imread",
        lambda _path: _FakeBayerFromValues(values, white_level=16383, camera_white_level=None),
    )
    _linearity, plateau = raw_channel_clip_fraction("x.NEF", 0, _FullRoi())
    assert plateau > 0.6  # the pile is ~71 % of the ROI; a max()-anchored window would report 0


def test_raw_clip_plateau_needs_no_white_level_at_all(monkeypatch):
    # The shape-based path exists specifically for a body with no usable ceiling metadata, so it
    # must not be gated behind one.
    rng = np.random.default_rng(5)
    pile = np.full(500, 12000, dtype=np.int64)
    below = np.clip(8000 + rng.normal(0, 100, size=2000), 0, None).astype(np.int64)
    values = np.concatenate([pile, below])
    monkeypatch.setattr(
        rawpy,
        "imread",
        lambda _path: _FakeBayerFromValues(values, white_level=None, camera_white_level=None),
    )
    linearity, plateau = raw_channel_clip_fraction("x.NEF", 0, _FullRoi())
    assert linearity == 0.0  # no white level at all → the linearity signal has nothing to check
    assert plateau > 0.15  # found from shape alone, roughly pile / total


def test_raw_clip_plateau_does_not_false_positive_on_a_quiet_clean_base(monkeypatch):
    # An ETTR base sitting close to but under saturation, at realistic read noise, must not
    # false-positive: its own maximum is a noise excursion held by a handful of photosites, and
    # the falling tail below it never clears the plateau's density-ratio gate.
    rng = np.random.default_rng(11)
    values = np.clip(15000 + rng.normal(0.0, 4.0, size=20000), 0, None).astype(np.int64)
    monkeypatch.setattr(
        rawpy,
        "imread",
        lambda _path: _FakeBayerFromValues(values, white_level=16383, camera_white_level=None),
    )
    linearity, plateau = raw_channel_clip_fraction("x.NEF", 0, _FullRoi())
    assert linearity == 0.0
    assert plateau == 0.0
