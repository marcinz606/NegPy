import numpy as np
import pytest

from negpy.services.capture.pixel_shift_merge import measure_wobble, merge_set, shot_positions

RGGB = np.array([[0, 1], [3, 2]], np.uint8)


def _colors(height, width, pattern=RGGB):
    return np.tile(pattern, (height // 2 + 1, width // 2 + 1))[:height, :width]


#: The merge and the check of a 4-shot set, each on a list of mosaics.
_BOTH = (lambda m: merge_set(m, RGGB, shot_positions(4)), lambda m: measure_wobble(m, shot_positions(4)))


def _shoot(truth, height, width, count, moved_shot=None, pattern=RGGB):
    """The mosaics of a `truth` RGB scene on the output grid; `moved_shot` sits one sensor row too low."""
    scale = 1 if count == 4 else 2
    channel = np.where(_colors(height, width, pattern) == 3, 1, _colors(height, width, pattern))
    y, x = np.mgrid[0:height, 0:width] * scale
    mosaics = []
    for shot, (row, col) in enumerate(shot_positions(count)):
        extra = scale if shot == moved_shot else 0
        mosaics.append(truth[y + int(row * scale) + extra, x + int(col * scale), channel])
    return mosaics


def _smooth_scene(height, width, scale=1, blur=12):
    """Detail a few sensor pixels wide: single-pixel noise shares nothing between shots."""
    fy, fx = np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :]
    noise = np.random.default_rng(1).standard_normal((height, width))
    smooth = np.fft.irfft2(np.fft.rfft2(noise) * np.exp(-(fy**2 + fx**2) * blur * scale**2), s=noise.shape)
    plane = (8000 + 2000 * smooth / smooth.std()).clip(0, 65535).astype(np.uint16)
    return np.repeat(plane[:, :, None], 3, axis=2)


def test_16_shot_positions_are_distinct_half_pixel_steps():
    positions = shot_positions(16)
    assert positions[:4] == ((0, 1.5), (1, 1.5), (1, 0.5), (0, 0.5))
    assert positions[4:8] == shot_positions(4)
    assert len(set(positions)) == 16


@pytest.mark.parametrize("count", [4, 16])
def test_merge_returns_the_truth(count):
    height, width, scale = 64, 96, 1 if count == 4 else 2
    truth = np.random.default_rng(0).integers(0, 65536, (scale * (height + 1), scale * (width + 1), 3), np.uint16)
    merged = merge_set(iter(_shoot(truth, height, width, count)), _colors(height, width), shot_positions(count))
    assert merged.dtype == np.uint16
    np.testing.assert_array_equal(merged, truth[scale : scale * height, scale : scale * width])


def test_merge_averages_the_two_greens_rounding_up():
    colors = _colors(8, 8)
    mosaic = np.where(colors == 3, 65535, 65534).astype(np.uint16)
    assert (merge_set([mosaic] * 4, colors, shot_positions(4))[:, :, 1] == 65535).all()


@pytest.mark.parametrize("shots", [3, 5])
def test_wrong_shot_count_raises(shots):
    mosaics = [np.zeros((8, 8), np.uint16)] * shots
    for fn in _BOTH:
        with pytest.raises(ValueError):
            fn(iter(mosaics))


@pytest.mark.parametrize("count", [4, 16])
def test_correct_set_is_steady(count):
    height, width, scale = 1280, 384, 1 if count == 4 else 2
    truth = _smooth_scene(scale * (height + 2), scale * (width + 2), scale)
    report = measure_wobble(iter(_shoot(truth, height, width, count)), shot_positions(count))
    assert report.judged and report.steady, report


def test_moved_frame_is_not_steady():
    truth = _smooth_scene(1282, 386)
    report = measure_wobble(iter(_shoot(truth, 1280, 384, 4, moved_shot=2)), shot_positions(4))
    assert report.judged and not report.steady, report


def test_flat_set_is_not_judged():
    rng = np.random.default_rng(2)
    mosaics = [rng.normal(8000, 60, (1280, 384)).astype(np.uint16) for _ in range(4)]
    report = measure_wobble(iter(mosaics), shot_positions(4))
    assert report.bands == 0 and not report.judged and report.steady


@pytest.mark.parametrize("count", [4, 16])
@pytest.mark.parametrize("pattern", [RGGB, RGGB[::-1], RGGB[:, ::-1], RGGB[::-1, ::-1]], ids=["RGGB", "GBRG", "GRBG", "BGGR"])
@pytest.mark.parametrize("size", [(64, 96), (63, 97), (300, 420)], ids=["small", "odd", "pooled"])
def test_merge_returns_the_truth_for_every_bayer_pattern_and_size(count, pattern, size):
    (height, width), scale = size, 1 if count == 4 else 2
    truth = np.random.default_rng(3).integers(0, 65536, (scale * (height + 1), scale * (width + 1), 3), np.uint16)
    merged = merge_set(_shoot(truth, height, width, count, pattern=pattern), _colors(height, width, pattern), shot_positions(count))
    np.testing.assert_array_equal(merged, truth[scale : scale * height, scale : scale * width])


def test_merge_takes_a_view_that_is_not_contiguous():
    truth = np.random.default_rng(4).integers(0, 65536, (65, 97, 3), np.uint16)
    wide = [np.repeat(mosaic, 2, axis=1)[:, ::2] for mosaic in _shoot(truth, 64, 96, 4)]
    assert not wide[0].flags.c_contiguous
    np.testing.assert_array_equal(merge_set(wide, _colors(64, 96), shot_positions(4)), truth[1:64, 1:96])


@pytest.mark.parametrize("bad", [np.zeros((8, 8), np.float32), np.zeros((8, 8, 3), np.uint16), np.zeros((1, 8), np.uint16)])
def test_a_shot_that_is_not_a_uint16_mosaic_raises(bad):
    for fn in _BOTH:
        with pytest.raises(ValueError):
            fn([bad] * 4)


def test_shots_of_two_sizes_raise():
    mosaics = [np.zeros((1280, 384), np.uint16)] * 3 + [np.zeros((1280, 386), np.uint16)]
    for fn in _BOTH:
        with pytest.raises(ValueError):
            fn(mosaics)


def test_a_pattern_that_is_not_bayer_raises():
    with pytest.raises(ValueError):
        merge_set([np.zeros((8, 8), np.uint16)] * 4, np.zeros((8, 8), np.uint8), shot_positions(4))


def test_a_mosaic_too_small_for_a_band_is_not_judged():
    report = measure_wobble([np.full((64, 64), 9000, np.uint16)] * 4, shot_positions(4))
    assert report.bands == 0 and report.steady


def _noisy(mosaics, seed=9):
    rng = np.random.default_rng(seed)
    return [(mosaic + rng.normal(0, 60, mosaic.shape)).clip(0, 65535).astype(np.uint16) for mosaic in mosaics]


@pytest.mark.parametrize("blur", [12, 48, 200])
def test_a_soft_4_shot_set_is_steady_and_a_slipped_one_is_not(blur):
    """A soft frame holds little detail at the scale of one CFA site, where the peak fit is weakest."""
    truth = _smooth_scene(1284, 1540, blur=blur)
    mosaics = _noisy(_shoot(truth, 1280, 1536, 4))
    report = measure_wobble(mosaics, shot_positions(4))
    assert report.judged and report.rms < 0.15 and report.worst < 0.3, report
    mosaics[2] = np.roll(mosaics[2], 1, axis=0)
    slipped = measure_wobble(mosaics, shot_positions(4))
    assert slipped.judged and not slipped.steady and slipped.worst > 1.0, slipped


def test_a_slip_of_one_site_sample_is_read_exactly():
    truth = _smooth_scene(1284, 1540)
    mosaics = _shoot(truth, 1280, 1536, 4)
    mosaics[2] = np.roll(mosaics[2], 2, axis=0)
    report = measure_wobble(mosaics, shot_positions(4))
    assert report.judged and abs(report.worst - 2.0) < 0.1, report


def test_the_check_reads_the_color_the_light_shows():
    """Under red light the other sites hold noise only."""
    truth = _smooth_scene(1282, 386)
    rng = np.random.default_rng(5)
    colors = _colors(1280, 384)
    mosaics = [np.where(colors == 0, mosaic, rng.integers(0, 40, mosaic.shape)).astype(np.uint16) for mosaic in _shoot(truth, 1280, 384, 4)]
    report = measure_wobble(mosaics, shot_positions(4))
    assert report.judged and report.steady, report


@pytest.mark.parametrize("count", [4, 16])
def test_a_shot_blurred_while_it_was_exposed_is_not_steady(count):
    """Smeared along its own rows, the shot still sits in its place: only its detail gives it away."""
    scale = 1 if count == 4 else 2
    truth = _smooth_scene(scale * 1284, scale * 1540, scale)
    mosaics = _noisy(_shoot(truth, 1280, 1536, count))
    assert not measure_wobble(mosaics, shot_positions(count)).soft_shot
    smeared = mosaics[2].astype(np.uint32)
    mosaics[2] = ((np.roll(smeared, 2, axis=1) + 2 * smeared + np.roll(smeared, -2, axis=1)) // 4).astype(np.uint16)
    report = measure_wobble(mosaics, shot_positions(count))
    assert report.soft_shot == 3 and not report.steady, report


def test_a_blank_set_has_no_soft_shot():
    rng = np.random.default_rng(6)
    mosaics = [rng.normal(8000, 60, (1280, 384)).clip(0, 65535).astype(np.uint16) for _ in range(4)]
    assert measure_wobble(mosaics, shot_positions(4)).steady


@pytest.mark.parametrize("count", [4, 16])
@pytest.mark.parametrize("texture", [True, False])
def test_a_shot_with_another_exposure_is_not_steady(count, texture):
    """A flicker of the light: the merge would show the odd shot as a pattern, on blank film too."""
    height, width, scale = 1280, 384, 1 if count == 4 else 2
    truth = _smooth_scene(scale * (height + 2), scale * (width + 2), scale) if texture else np.full((4000, 1000, 3), 9000, np.uint16)
    mosaics = _shoot(truth, height, width, count)
    assert measure_wobble(mosaics, shot_positions(count)).steady
    mosaics[2] = (mosaics[2] * 1.02).astype(np.uint16)
    report = measure_wobble(mosaics, shot_positions(count))
    assert report.uneven_shot == 3 and not report.steady, report


def test_the_level_check_compares_the_light_above_the_black_level():
    """A thin frame: the black level is half the level, and would halve the difference."""
    mosaics = [np.full((64, 64), 1000, np.uint16) for _ in range(4)]
    mosaics[1] += 8  # 1.6 % of the light, 0.8 % of the level
    assert measure_wobble(mosaics, shot_positions(4)).steady
    assert measure_wobble(mosaics, shot_positions(4), black=512).uneven_shot == 2
    rng = np.random.default_rng(1)
    dark = [rng.normal(512, 2, (256, 256)).round().astype(np.uint16) for _ in range(4)]  # no light: noise around black
    assert measure_wobble(dark, shot_positions(4), black=512).steady


def test_the_report_names_the_sharpest_shot():
    mosaics = _shoot(_smooth_scene(1282, 1538), 1280, 1536, 4)
    blurred = [(m.astype(np.float32) + np.roll(m, 2, axis=1)) / 2 for m in mosaics]
    mosaics = [b.astype(np.uint16) for b in blurred[:3]] + [mosaics[3]]
    assert measure_wobble(mosaics, shot_positions(4)).sharpest_shot == 4


@pytest.mark.parametrize(
    "positions",
    [
        ((0, 1), (1, 1), (1, 0)),  # a site is missing
        ((0, 1), (1, 1), (1, 0), (1, 0)),  # a site twice
        ((0, 1), (1, 1), (1, 0), (0, 0.5)),  # half steps that do not fill the doubled grid
        ((0, 1), (1, 1), (1, 0), (0, 0.25)),
        ((0, 1), (1, 1), (1, 0), (0, 2)),
    ],
)
def test_a_position_table_that_leaves_a_site_empty_raises(positions):
    with pytest.raises(ValueError, match="position"):
        merge_set([np.zeros((8, 8), np.uint16)] * len(positions), RGGB, positions)


def test_only_a_confirmed_maker_has_a_position_table():
    assert len(shot_positions(16, "sony")) == 16
    for count, maker in ((8, "sony"), (4, "nikon"), (4, "")):
        with pytest.raises(ValueError):
            shot_positions(count, maker)
