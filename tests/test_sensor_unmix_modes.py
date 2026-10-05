from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessConfig, SensorUnmix
from negpy.features.process.sensor import (
    apply_sensor_correction,
    density_unmix_matrix,
    film_base,
    sensor_token,
)

# A real single-shot calibration (A7R4 + Big Scanlight): green leaks ~25% into blue.
_M = (1.0133, -0.0848, -0.0108, -0.1535, 1.1168, -0.4224, -0.0011, -0.2706, 1.104)
_BASE = np.array([0.23, 0.56, 0.33], dtype=np.float32)


def _negative(rng, h=256, w=384):
    """Orange-base negative: per-pixel densities above the base, with mild grain, and a
    patch of clear base in the interior, where film_base reads it."""
    density = rng.uniform(0.1, 1.2, (h, w, 1)) + rng.normal(0.0, 0.03, (h, w, 3))
    img = _BASE * 10.0 ** (-density)
    img[96:160, 160:224] = _BASE * np.exp(rng.normal(0.0, 0.01, (64, 64, 3)))
    return img.astype(np.float32)


def _linear(img):
    return np.clip(np.einsum("ck,hwk->hwc", np.asarray(_M, np.float32).reshape(3, 3), img), 0.0, None)


def test_linear_mode_is_the_plain_unmix():
    img = _negative(np.random.default_rng(1))
    assert np.allclose(apply_sensor_correction(img, _M, SensorUnmix.LINEAR), _linear(img), rtol=1e-6, atol=1e-8)


def test_two_scale_matches_linear_where_the_calibration_holds():
    img = _negative(np.random.default_rng(2))
    out = apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE)
    assert np.allclose(out, _linear(img), rtol=1e-5, atol=1e-7)


def _neon_frame(rng, noise=0.03):
    """Clear film base beside a blue light through dense yellow dye, with multiplicative grain:
    nine tenths of the blue record is green leak there."""
    img = np.empty((96, 192, 3), np.float32)
    img[:] = _BASE
    img[:, :96] *= np.exp(rng.normal(0.0, 0.01, (96, 96, 3))).astype(np.float32)  # unclipped base has grain
    neon = np.array([0.23, 0.29, 0.29 * 0.2706 / 1.104 / 0.9], np.float32)
    img[:, 96:] = neon
    img[:, 96:] *= np.exp(rng.normal(0.0, noise, (96, 96, 3))).astype(np.float32)
    return img


def test_two_scale_never_reaches_zero():
    img = _neon_frame(np.random.default_rng(5), noise=0.3)
    assert (_linear(img)[:, 96:, 2] == 0).any()
    out = apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE)
    assert (out > 0).all() and np.isfinite(out).all()


def test_two_scale_does_not_amplify_grain_past_the_film_base():
    img = _neon_frame(np.random.default_rng(6))
    c = density_unmix_matrix(_M, film_base(img))
    core = (slice(16, 80), slice(120, 168))
    linear = np.log(np.maximum(_linear(img)[core][:, :, 2], 1e-6)).std()
    two_scale = np.log(apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE)[core][:, :, 2]).std()
    assert two_scale < 0.3 * linear
    assert two_scale < 1.2 * np.hypot(c[2, 2], c[2, 1]) * 0.03


def test_two_scale_keeps_the_color_of_a_flat_area():
    img = _neon_frame(np.random.default_rng(7), noise=0.0)
    img[:, 96:, 2] *= 1.4  # a third of the blue survives: past the base gain, still above zero
    out = apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE)
    shift = np.abs(np.log10(out[40:56, 130:150]) - np.log10(_linear(img)[40:56, 130:150]))
    assert shift.max() < 0.035  # only the color layer's floor knee


def test_two_scale_falls_back_to_linear_without_a_usable_base():
    img = np.zeros((8, 8, 3), np.float32)
    assert np.allclose(apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE), _linear(img))


def test_density_matrix_rows_sum_to_one():
    c = density_unmix_matrix(_M, _BASE)
    assert np.allclose(c.sum(axis=1), 1.0, atol=1e-12)


def test_density_is_exact_at_the_film_base_and_close_near_it():
    img = _negative(np.random.default_rng(3))
    base = film_base(img)
    out = apply_sensor_correction(img, _M, SensorUnmix.DENSITY)
    m = np.asarray(_M, np.float64).reshape(3, 3)
    ratio = base / _BASE  # only the base's color is used, not its level
    assert np.allclose(ratio, ratio.mean(), rtol=0.005)
    patch = np.median(img[96:160, 160:224], axis=(0, 1)).astype(np.float64)
    assert np.allclose(np.median(out[96:160, 160:224], axis=(0, 1)), m @ patch, rtol=0.005)
    near = np.log10(out) - np.log10(np.maximum(_linear(img), 1e-6))
    near -= np.median(near, axis=(0, 1))
    assert np.median(np.abs(near)) < 0.01


def test_density_never_reaches_zero():
    img = _neon_frame(np.random.default_rng(8), noise=0.3)
    assert (_linear(img)[:, 96:, 2] == 0).any()
    out = apply_sensor_correction(img, _M, SensorUnmix.DENSITY)
    assert (out > 0).all() and np.isfinite(out).all()


def test_density_falls_back_to_linear_without_a_usable_base():
    img = np.zeros((8, 8, 3), np.float32)
    assert np.allclose(apply_sensor_correction(img, _M, SensorUnmix.DENSITY), _linear(img))


def test_sensor_token_names_a_non_linear_mode_only():
    process = replace(ProcessConfig(), linear_raw=True, sensor_matrix=_M, sensor_unmix=SensorUnmix.LINEAR)
    linear = sensor_token(process)
    assert "|two_scale" not in linear and "|density" not in linear
    tokens = {sensor_token(replace(process, sensor_unmix=mode)) for mode in SensorUnmix}
    assert len(tokens) == len(SensorUnmix)


def test_mode_roundtrips_and_an_unknown_value_reads_the_default():
    cfg = replace(WorkspaceConfig(), process=replace(ProcessConfig(), sensor_unmix=SensorUnmix.DENSITY))
    assert WorkspaceConfig.from_flat_dict(cfg.to_dict()).process.sensor_unmix == SensorUnmix.DENSITY
    assert ProcessConfig(sensor_unmix="bogus").sensor_unmix == ProcessConfig().sensor_unmix


def test_a_calibrated_edit_saved_before_the_method_choice_stays_linear():
    calibrated = replace(WorkspaceConfig(), process=replace(ProcessConfig(), sensor_matrix=tuple(np.eye(3).ravel())))
    legacy = calibrated.to_dict()
    del legacy["sensor_unmix"]
    assert WorkspaceConfig.from_flat_dict(legacy).process.sensor_unmix == SensorUnmix.LINEAR

    uncalibrated = WorkspaceConfig().to_dict()
    del uncalibrated["sensor_unmix"]
    assert WorkspaceConfig.from_flat_dict(uncalibrated).process.sensor_unmix == SensorUnmix.TWO_SCALE
    assert WorkspaceConfig.from_flat_dict(calibrated.to_dict()).process.sensor_unmix == SensorUnmix.TWO_SCALE


def test_film_base_reads_one_shared_set_of_cells():
    """No clear base in frame: neutral shadows, plus a deep red and a deep blue area that are
    each thinner than the shadows in one channel only. Per-channel thin ends would mix them."""
    rng = np.random.default_rng(9)
    density = np.full((240, 360, 3), 0.8)
    density[60:180, 40:90] = 0.1  # neutral shadows
    density[60:180, 120:240] = (0.02, 0.8, 0.8)  # thin in red alone
    density[60:180, 260:340] = (0.8, 0.8, 0.02)  # thin in blue alone
    img = (_BASE * 10.0 ** (-density) * np.exp(rng.normal(0.0, 0.01, density.shape))).astype(np.float32)
    ratio = film_base(img) / _BASE
    assert np.allclose(ratio, ratio.mean(), rtol=0.01)


def test_a_clipped_base_falls_back_to_linear():
    img = _negative(np.random.default_rng(10))
    img[96:160, 160:224, 1] = img[:, :, 1].max()  # the thin end of green clipped in the capture
    assert film_base(img) is None
    for mode in (SensorUnmix.TWO_SCALE, SensorUnmix.DENSITY):
        assert np.allclose(apply_sensor_correction(img, _M, mode), _linear(img), rtol=1e-6, atol=1e-8)


def test_render_source_unmixes_a_stitch_once_assembled(monkeypatch, tmp_path):
    from negpy.features.stitch.models import StitchConfig
    from negpy.services.rendering import image_processor as ip

    h, w = 40, 60
    parts = {"a.nef": np.full((h, w, 3), 0.4, np.float32), "b.nef": np.full((h, w, 3), 0.6, np.float32)}
    for name in parts:
        (tmp_path / name).touch()
    unmix_flags: list[bool] = []

    def decode(self, path, params, fast_decode=False, wb_override=None, unmix=True):
        unmix_flags.append(unmix)
        return parts[path.rsplit("/", 1)[-1]], None, "Adobe RGB"

    shapes: list[tuple[int, ...]] = []

    def record(img, matrix, mode=SensorUnmix.LINEAR):
        shapes.append(img.shape)
        return img

    monkeypatch.setattr(ip.ImageProcessor, "_decode_oriented_f32", decode)
    monkeypatch.setattr(ip, "apply_sensor_correction", record)
    stitch = StitchConfig(
        stitch_enabled=True,
        stitch_paths=(str(tmp_path / "b.nef"),),
        stitch_transforms=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 50.0, 0.0, 1.0, 0.0)),
        stitch_canvas=(110, h),
        stitch_sizes=((w, h), (w, h)),
    )
    process = replace(ProcessConfig(), linear_raw=True, sensor_matrix=_M)
    cfg = replace(WorkspaceConfig(), process=process, stitch=stitch)
    ip.ImageProcessor()._load_source_f32(str(tmp_path / "a.nef"), cfg)
    assert unmix_flags == [False, False]
    assert shapes == [(h, 110, 3)]


def test_export_unmixes_a_half_frame_after_slicing_it(monkeypatch, tmp_path):
    """The preview slices a half before the unmix, so the export must too: Two-Scale and
    Density read the film base from the buffer they are given."""
    from negpy.services.rendering import image_processor as ip

    class Stop(Exception):
        pass

    unmix_flags: list[bool] = []

    def decode(self, path, params, fast_decode=False, wb_override=None, unmix=True):
        unmix_flags.append(unmix)
        return np.full((40, 120, 3), 0.3, np.float32), None, "Adobe RGB"

    def record(img, matrix, mode):
        raise Stop(img.shape)

    monkeypatch.setattr(ip.ImageProcessor, "_decode_oriented_f32", decode)
    monkeypatch.setattr(ip, "apply_sensor_correction", record)
    (tmp_path / "a.nef").touch()
    cfg = replace(WorkspaceConfig(), process=replace(ProcessConfig(), linear_raw=True, sensor_matrix=_M))
    try:
        ip.ImageProcessor()._prepare_export_source_locked(str(tmp_path / "a.nef"), cfg, "hash", 1, 0.5, None, 0.0)
    except Stop as unmixed:
        shape = unmixed.args[0]
    assert unmix_flags == [False]
    assert shape[1] < 120


def test_a_dead_pixel_does_not_blow_up_two_scale():
    img = _neon_frame(np.random.default_rng(11))
    img[48, 140, 1] = 0.0  # a dead green pixel inside the light
    out = apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE)
    around = np.median(out[40:56, 130:150], axis=(0, 1))
    assert np.all(out[48, 140] < 2.0 * around)


def test_two_scale_still_acts_when_the_base_is_mostly_leak():
    """A light so unbalanced that most of the base's own blue is green leak: the floor must not
    hide how far Linear amplifies the neon."""
    rng = np.random.default_rng(12)
    base = np.array([0.23, 1.0, 0.3], np.float32)
    assert density_unmix_matrix(_M, base)[2, 2] > 5.0
    img = np.empty((96, 3600, 3), np.float32)  # wide enough for a blur of a few pixels
    img[:] = base * np.exp(rng.normal(0.0, 0.01, img.shape))
    neon = np.array([0.23, 0.5, 0.5 * 0.2706 / 1.104 / 0.9], np.float32)
    img[:, 1800:] = neon * np.exp(rng.normal(0.0, 0.1, (96, 1800, 3)))
    core = (slice(16, 80), slice(2000, 3400))
    linear = _linear(img)[core][:, :, 2]
    out = apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE)[core][:, :, 2]
    assert (linear == 0).any() and (out > 0).all()
    assert np.log(out).std() < 0.5 * np.log(np.maximum(linear, 1e-6)).std()


def test_contact_sheet_unmixes_a_half_frame_after_slicing_it(monkeypatch, tmp_path):
    from negpy.services.rendering import image_processor as ip

    unmix_flags: list[bool] = []
    shapes: list[tuple[int, ...]] = []

    def decode(self, path, params, fast_decode=False, wb_override=None, unmix=True):
        unmix_flags.append(unmix)
        return np.full((40, 120, 3), 0.3, np.float32), None, "Adobe RGB"

    def record(img, matrix, mode):
        shapes.append(img.shape)
        return img

    def stop(*args, **kwargs):
        raise RuntimeError("stop after the unmix")

    monkeypatch.setattr(ip.ImageProcessor, "_decode_oriented_f32", decode)
    monkeypatch.setattr(ip, "apply_sensor_correction", record)
    monkeypatch.setattr(ip, "_downsample_to_long_edge", stop)
    (tmp_path / "a.nef").touch()
    cfg = replace(WorkspaceConfig(), process=replace(ProcessConfig(), linear_raw=True, sensor_matrix=_M))
    ip.ImageProcessor().render_display_array(str(tmp_path / "a.nef"), cfg, "hash", 64, prefer_gpu=False, half=1)
    assert unmix_flags == [False]
    assert len(shapes) == 1 and shapes[0][1] < 120


def test_two_scale_keeps_a_hard_light_edge():
    """A light's core far denser than its surround drops every channel together; that is an
    edge, not a dead photosite, and keeps its detail."""
    import negpy.features.process.sensor as sensor

    img = _neon_frame(np.random.default_rng(13), noise=0.0)
    img = np.repeat(img, 13, axis=1)  # wide enough for a blur of a few pixels
    img[:, 1800:] *= 1e-3  # 3 D denser than the light's rim
    out = apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE)
    saved = sensor._DEFECT_RATIO
    try:
        sensor._DEFECT_RATIO = 1e-30
        no_rule = apply_sensor_correction(img, _M, SensorUnmix.TWO_SCALE)
    finally:
        sensor._DEFECT_RATIO = saved
    assert np.allclose(out[:, 1790:1810], no_rule[:, 1790:1810], rtol=1e-4)


def test_density_reads_no_channel_past_its_range():
    img = _negative(np.random.default_rng(14))
    dead = img.copy()
    dead[30, 30, 1] = 0.0
    held = img.copy()
    held[30, 30, 1] = film_base(img)[1] * 1e-3
    a = apply_sensor_correction(dead, _M, SensorUnmix.DENSITY)[30, 30]
    b = apply_sensor_correction(held, _M, SensorUnmix.DENSITY)[30, 30]
    assert np.allclose(a, b, rtol=1e-4)


def test_no_preview_unmix_for_a_triplet_or_a_stitch_holding_one():
    from negpy.features.rgbscan.models import RgbScanConfig
    from negpy.features.stitch.models import StitchConfig
    from negpy.services.rendering.image_processor import preview_takes_unmix

    plain = WorkspaceConfig()
    triplet = replace(plain, rgbscan=RgbScanConfig(enabled=True, green_path="g.nef", blue_path="b.nef"))
    stitch = StitchConfig(stitch_enabled=True, stitch_paths=("b.nef",))
    single_parts = replace(plain, stitch=stitch)
    triplet_part = replace(plain, stitch=replace(stitch, stitch_triplets=(("", ""), ("g.nef", "b.nef"))))
    assert preview_takes_unmix(plain) and preview_takes_unmix(single_parts)
    assert not preview_takes_unmix(triplet) and not preview_takes_unmix(triplet_part)
