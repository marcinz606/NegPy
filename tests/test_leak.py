import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.leak.bake import leak_active, leak_token, remove_leak
from negpy.features.leak.estimate import estimate_leak
from negpy.features.leak.film import NEG_GAMMA, FilmResponse
from negpy.features.leak.models import LeakConfig
from negpy.features.leak.shapes import new_band
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.stitch.models import StitchConfig
from negpy.services.rendering.image_processor import ImageProcessor

_BASE = np.array([0.8, 0.5, 0.3], dtype=np.float32)
_LEAK_COLOR = np.array([1.0, 0.45, 0.15], dtype=np.float32)


def excess_exposure(t: np.ndarray, base: np.ndarray, shoulder=None) -> np.ndarray:
    return FilmResponse(base=base, shoulder=shoulder).excess(t)


def to_transmittance(u: np.ndarray, base: np.ndarray, shoulder=None) -> np.ndarray:
    return FilmResponse(base=base, shoulder=shoulder).transmittance(u)


def _scene(h: int = 600, w: int = 900, seed: int = 0) -> np.ndarray:
    """Excess exposure of a plain scene: blobs of tone over a dark ground, rebate along top and bottom."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(0.0, 1.0, (12, 18)).astype(np.float32) ** 3 * 8.0
    tone = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC).clip(0.0, None)
    e = np.repeat(tone[..., None], 3, axis=2) * np.array([1.0, 0.9, 1.1], dtype=np.float32)
    band = h // 12
    e[:band] = 0.0
    e[-band:] = 0.0
    return e


def _crop(h: int = 600) -> GeometryConfig:
    """The picture between the rebate bands _scene leaves along top and bottom."""
    band = (h // 12) / h
    return GeometryConfig(crop_rect=(0.0, band, 1.0, 1.0 - band))


def _leak(h: int, w: int, peak: float = 3.0, reach: float = 0.25) -> np.ndarray:
    """A leak in from the left edge that fades out inside the frame."""
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    profile = peak * np.clip((np.exp(-x / reach) - np.exp(-0.7 / reach)) / (1.0 - np.exp(-0.7 / reach)), 0.0, None)
    return np.broadcast_to(profile[None, :, None], (h, w, 1)) * _LEAK_COLOR


def _tilted_band(h: int, w: int, angle_deg: float, center=(0.6, 0.5), width: float = 0.05, peak: float = 3.0) -> np.ndarray:
    """A straight leak band through *center* running at *angle_deg*, as a gap in a camera back throws."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.deg2rad(angle_deg)
    d = np.array([np.cos(t), np.sin(t)])
    across = (xs - center[0] * w) * -d[1] + (ys - center[1] * h) * d[0]
    return (peak * np.exp(-0.5 * (across / (width * max(h, w))) ** 2))[..., None] * _LEAK_COLOR


def _negative(e: np.ndarray, grain: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(1)
    noise = np.exp(rng.normal(0.0, grain, e.shape)).astype(np.float32)
    return (to_transmittance(e, _BASE) * noise).astype(np.float32)


# A crop flush with the scan: no rebate outside it.
_UNREAD = GeometryConfig(crop_rect=(0.0, 0.0, 1.0, 1.0))
_DOWN_THE_BAND = new_band((0.6, 0.0), (0.6, 1.0), 0.12, 0.12)
_EDGE_BAND = new_band((0.0, 0.0), (0.0, 1.0), 0.02, 0.7)
# The removal alone under the band down the frame: the clean-up after it has its own tests.
_ON = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(_DOWN_THE_BAND,))


def _strength(k: float) -> LeakConfig:
    return replace(_ON, leak_shapes=(replace(_DOWN_THE_BAND, strength=k),))


def _apply(neg: np.ndarray, leak: LeakConfig = _ON, geometry: GeometryConfig = _UNREAD) -> np.ndarray:
    return remove_leak(neg, WorkspaceConfig(leak=leak, geometry=geometry))[0]


def _removed(e: np.ndarray, leak_field: np.ndarray, leak: LeakConfig = _ON, geometry: GeometryConfig = _UNREAD) -> float:
    """Share of the leak's excess exposure the bake takes out, against the same grain without it."""
    clean = excess_exposure(_negative(e), _BASE)
    leaked = _negative(e + leak_field)
    before = np.abs(excess_exposure(leaked, _BASE) - clean).mean()
    after = np.abs(excess_exposure(_apply(leaked, leak, geometry), _BASE) - clean).mean()
    return 1.0 - after / before


def _band_angle(band, h: int, w: int) -> float:
    (x0, y0), (x1, y1) = band.points
    return float(np.degrees(np.arctan2((y1 - y0) * h, (x1 - x0) * w))) % 180.0


def test_excess_exposure_round_trips():
    e = _scene()
    np.testing.assert_allclose(excess_exposure(_negative(e, grain=0.0), _BASE), e, rtol=1e-4, atol=1e-4)


def test_gamma_is_a_camera_negative():
    assert 0.4 < NEG_GAMMA < 0.8


def test_removes_a_band_across_the_frame():
    e = _scene()
    assert _removed(e, _tilted_band(*e.shape[:2], 90.0)) > 0.4


def test_removes_an_edge_leak_under_a_band_along_that_edge():
    e = _scene()
    assert _removed(e, _leak(*e.shape[:2]), replace(_ON, leak_shapes=(_EDGE_BAND,))) > 0.15


def test_nothing_is_removed_outside_a_shape():
    """With no shape, a lit rebate is not read as a leak in the picture: edge fog never reaches it."""
    e = _scene()
    fogged = e.copy()
    rows = e.shape[0] // 12
    fogged[:rows] += 4.0 * _LEAK_COLOR
    neg = _negative(fogged)
    assert _apply(neg, replace(_ON, leak_shapes=()), _crop()) is neg


def test_more_strength_never_removes_less():
    e = _scene()
    leak_field = _tilted_band(*e.shape[:2], 90.0)
    shares = [_removed(e, leak_field, _strength(k)) for k in (0.25, 0.5, 1.0, 2.0)]
    assert all(b >= a - 0.01 for a, b in zip(shares, shares[1:]))
    assert shares[0] < shares[2]


def test_the_cap_stops_strength_short_of_over_subtracting():
    e = _scene()
    leaked = _negative(e + _tilted_band(*e.shape[:2], 90.0))
    pushed = excess_exposure(_apply(leaked, _strength(4.0)), _BASE)
    clean = excess_exposure(_negative(e), _BASE)
    assert float(np.mean(clean - pushed)) < 0.1


def test_estimate_does_not_depend_on_resolution():
    e = _scene()
    neg = _negative(e + _tilted_band(*e.shape[:2], 90.0))
    big = cv2.resize(neg, (1800, 1200), interpolation=cv2.INTER_CUBIC)
    small_shape = estimate_leak(neg, _ON.leak_shapes, geometry=_UNREAD).field
    big_shape = estimate_leak(big, _ON.leak_shapes, geometry=_UNREAD).field
    big_shape = cv2.resize(big_shape, small_shape.shape[1::-1], interpolation=cv2.INTER_AREA)
    across_small, across_big = small_shape.mean(axis=(0, 2)), big_shape.mean(axis=(0, 2))
    np.testing.assert_allclose(across_small, across_big, atol=0.1 * float(across_small.max()))


def test_bed_brighter_than_base_keeps_its_value():
    e = _scene()
    neg = _negative(e + _tilted_band(*e.shape[:2], 90.0))
    neg[:, 520:540] = 1.0
    out = _apply(neg)
    np.testing.assert_array_equal(out[:, 520:540], neg[:, 520:540])


def test_a_band_on_a_clean_frame_changes_little():
    neg = _negative(_scene(seed=3))
    out = _apply(neg)
    assert float(np.abs(np.log10(out / neg)).mean()) < 0.005


@pytest.mark.parametrize("angle", [90.0, 75.0, 60.0])
def test_auto_finds_a_straight_leak(angle):
    from negpy.features.leak.auto import auto_bands

    e = _scene()
    h, w = e.shape[:2]
    bands = auto_bands(_negative(e + _tilted_band(h, w, angle)), _crop())
    assert len(bands) == 1
    assert abs(_band_angle(bands[0], h, w) - angle) < 8.0
    assert abs((bands[0].points[0][0] + bands[0].points[1][0]) / 2.0 - 0.6) < 0.06


def test_auto_finds_nothing_on_a_clean_frame():
    from negpy.features.leak.auto import auto_bands

    for seed in (0, 3, 7):
        assert auto_bands(_negative(_scene(seed=seed)), _crop()) == []


def test_the_canvas_outline_matches_the_measured_sides():
    from negpy.features.leak.shapes import band_frame, band_outline, band_with_handle

    band, shape = new_band((0.6, 0.2), (0.6, 0.8), 0.05, 0.1), (600, 900)
    outline = band_outline(band, shape)
    _, side, _, _ = band_frame(band, shape)
    for point, expected in ((outline[2], 1.0), (outline[4], -1.0)):
        assert side[round(point[1] * 600), round(point[0] * 900)] == pytest.approx(expected, abs=0.05)
    assert band_with_handle(band, 2, outline[2], shape).width[0] == pytest.approx(0.05)
    assert band_with_handle(band, 3, outline[4], shape).width[1] == pytest.approx(0.1)


def test_falloff_is_full_on_the_line_and_gone_at_the_edges():
    from negpy.features.leak.shapes import band_falloff

    side = np.linspace(-1.0, 1.0, 201)
    for falloff in (0.0, 0.5, 1.0):
        g = band_falloff(side, falloff)
        assert g[100] == pytest.approx(1.0) and g[0] == pytest.approx(0.0, abs=1e-9) and g[-1] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.diff(g[100:]) <= 1e-12)
    half = 150
    assert band_falloff(side, 0.0)[half] > band_falloff(side, 0.5)[half] > band_falloff(side, 1.0)[half]


def test_a_band_leaves_no_step_across_its_falloff():
    e = _scene()
    h, w = e.shape[:2]
    leaked = _negative(e + _tilted_band(h, w, 90.0))
    fixed = _apply(leaked)
    lift = np.log10(fixed / leaked).mean(axis=(0, 2))
    assert float(np.abs(np.diff(lift)).max()) < 0.01


def test_two_bands_over_one_leak_remove_it_once():
    e = _scene()
    leaked = _negative(e + _tilted_band(*e.shape[:2], 90.0))
    twice = replace(_ON, leak_shapes=(_DOWN_THE_BAND, new_band((0.6, 0.0), (0.6, 1.0), 0.1, 0.1)))
    out = excess_exposure(_apply(leaked, twice), _BASE)
    clean = excess_exposure(_negative(e), _BASE)
    assert float(np.mean(clean - out)) < 0.1


def test_the_shoulder_round_trips_and_matches_the_toe_in_shadows():
    shoulder = np.array([2.5, 2.5, 2.5], dtype=np.float32)
    u = np.array([[0.0, 0.5, 5.0, 200.0]], dtype=np.float32).T * np.ones(3, dtype=np.float32)
    t = to_transmittance(u, _BASE, shoulder=shoulder)
    np.testing.assert_allclose(excess_exposure(t, _BASE, shoulder=shoulder), u, rtol=1e-3)
    np.testing.assert_allclose(t[:2], to_transmittance(u[:2], _BASE), rtol=1e-3)


def _dense_scene(h: int = 600, w: int = 900) -> np.ndarray:
    """Excess exposure reaching high on a film with a shoulder: a leak there lands on its flat part."""
    return _scene(h, w) * 300.0 + 1.0


def test_the_shoulder_restores_a_leak_on_dense_film():
    shoulder = np.array([2.4, 2.4, 2.4], dtype=np.float32)
    e = _dense_scene()
    leak_field = _tilted_band(*e.shape[:2], 90.0, peak=400.0)
    rng = np.random.default_rng(2)

    def negative(x):
        return (to_transmittance(x, _BASE, shoulder=shoulder) * np.exp(rng.normal(0.0, 0.005, x.shape))).astype(np.float32)

    clean = negative(e)
    leaked = negative(e + leak_field)

    def error(dmax):
        out = _apply(leaked, replace(_ON, leak_dmax=dmax))
        return float(np.abs(np.log10(_band_region(out) / _band_region(clean))).mean())

    assert error(2.4) <= error(0.0)


def test_a_film_dmax_under_the_frames_densest_film_keeps_the_picture():
    e = _dense_scene()
    clean = _negative(e)
    leaked = _negative(e + _tilted_band(600, 900, 90.0, peak=400.0))

    def error(dmax):
        return float(np.abs(np.log10(_apply(leaked, replace(_ON, leak_dmax=dmax)) / clean)).mean())

    assert error(0.07) < 2.0 * error(0.0) + 0.01


def _grainy_band_frame():
    e = _scene()
    rng = np.random.default_rng(5)
    leaked = (_negative(e + _tilted_band(600, 900, 90.0)) * np.exp(rng.normal(0.0, 0.03, e.shape))).astype(np.float32)
    return e, leaked


def _band_region(img):
    return img[100:500, 470:610]


def test_grain_reduction_acts_only_where_leak_came_off():
    _e, leaked = _grainy_band_frame()
    plain = _apply(leaked, replace(_ON, leak_grain=0.0, leak_contrast=0.0))
    smooth = _apply(leaked, replace(_ON, leak_grain=1.0, leak_contrast=0.0))

    def grain(x):
        d = -np.log10(_band_region(x).mean(-1))
        return float(np.std(d - cv2.GaussianBlur(d, (0, 0), 3)))

    assert grain(smooth) < 0.9 * grain(plain)
    np.testing.assert_array_equal(smooth[:, :100], plain[:, :100])


def test_contrast_lifts_detail_where_leak_came_off():
    _e, leaked = _grainy_band_frame()
    flat = _apply(leaked, replace(_ON, leak_grain=0.0, leak_contrast=0.0))
    lifted = _apply(leaked, replace(_ON, leak_grain=0.0, leak_contrast=1.0))

    def detail(x):
        d = -np.log10(_band_region(x).mean(-1))
        return float(np.std(d - cv2.GaussianBlur(d, (0, 0), 18)))

    assert detail(lifted) > detail(flat)


def test_a_leak_saturates_only_the_layers_its_color_drives():
    """Red-led leak at the densest film: red holds no picture there, green and blue still do, so the
    frame is rebuilt from them rather than burned."""
    from negpy.features.leak.estimate import estimate_leak
    from negpy.features.leak.film import work_copy
    from negpy.features.leak.recover import layer_saturation

    leaked = _negative(_scene() + _tilted_band(600, 900, 90.0, peak=400.0))
    found = estimate_leak(leaked, _ON.leak_shapes, geometry=_UNREAD)
    saturated = layer_saturation(work_copy(leaked), found.response, found.field)
    columns = np.flatnonzero(saturated[..., 0].any(axis=0)) / saturated.shape[1]
    assert abs(float(np.median(columns)) - 0.6) < 0.05
    assert not saturated[..., 2].any()
    _, burned = remove_leak(leaked, WorkspaceConfig(leak=_ON, geometry=_UNREAD))
    assert burned is None or not burned.any()


def test_a_leak_over_shadows_burns_nothing():
    e = _scene() * 0.0 + 0.2
    neg = _negative(e + _tilted_band(600, 900, 90.0, peak=0.5))
    _, burned = remove_leak(neg, WorkspaceConfig(leak=_ON, geometry=_UNREAD))
    assert burned is None or not burned.any()


def test_spike_floor_finds_the_shadow_pile_up():
    from negpy.features.leak.estimate import spike_floor

    rng = np.random.default_rng(0)
    shadows = 2.0 * np.exp(rng.normal(0.0, 0.02, (400, 1)))
    scene = rng.uniform(5.0, 40.0, (1200, 1))
    piled = spike_floor(np.vstack([shadows, scene]))
    assert piled[0] == pytest.approx(2.0, rel=0.1)
    assert np.isnan(spike_floor(rng.uniform(1.0, 60.0, (1600, 1)))[0])


def test_a_single_frame_of_film_takes_it():
    on = WorkspaceConfig(leak=_ON)
    assert leak_active(on)
    assert not leak_active(WorkspaceConfig())
    assert leak_active(replace(on, process=ProcessConfig(process_mode=ProcessMode.E6)))
    assert not leak_active(replace(on, process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=True)))
    assert not leak_active(replace(on, stitch=StitchConfig(stitch_enabled=True)))


def test_token_tracks_the_settings():
    on = WorkspaceConfig(leak=_ON)
    assert leak_token(WorkspaceConfig()) == ""
    assert leak_token(on) != ""
    assert leak_token(on) != leak_token(replace(on, leak=_strength(0.5)))
    assert leak_token(on) != leak_token(replace(on, geometry=_crop()))
    assert leak_token(on) != leak_token(replace(on, leak=replace(_ON, leak_shapes=(_EDGE_BAND,))))
    assert leak_token(on) != leak_token(replace(on, leak=replace(_ON, leak_grain=0.5)))
    assert leak_token(on) != leak_token(replace(on, process=ProcessConfig(process_mode=ProcessMode.E6)))
    assert leak_token(on) != leak_token(replace(on, process=ProcessConfig(process_mode=ProcessMode.BW)))
    auto = replace(on, geometry=GeometryConfig(crop_rect=None))
    assert leak_token(auto) != leak_token(replace(auto, geometry=replace(auto.geometry, autocrop_rebate_trim=0.5)))


def test_shapes_round_trip_through_a_saved_edit():
    from negpy.features.leak.shapes import new_paint

    trimmed = replace(_EDGE_BAND, trim=0.2)
    config = WorkspaceConfig(leak=replace(_ON, leak_shapes=(_DOWN_THE_BAND, trimmed, new_paint([(0.2, 0.3), (0.4, 0.5)], 40.0))))
    back = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(config.to_dict())))
    assert back.leak == config.leak
    assert hash(back.leak) == hash(config.leak)


def test_a_leak_edit_changes_the_thumbnail_fingerprint():
    from negpy.services.assets.thumbnail_fingerprint import thumbnail_fingerprint

    def fingerprint(config):
        return thumbnail_fingerprint(config, workspace_color_space="Adobe RGB", input_icc_path=None)

    plain = WorkspaceConfig()
    assert fingerprint(plain) != fingerprint(WorkspaceConfig(leak=_ON))
    assert fingerprint(WorkspaceConfig(leak=_ON)) != fingerprint(WorkspaceConfig(leak=replace(_ON, leak_shapes=(_EDGE_BAND,))))


def test_processor_bake_is_gated_and_cached():
    e = _scene()
    neg = _negative(e + _tilted_band(*e.shape[:2], 90.0))
    neg.setflags(write=False)
    proc = ImageProcessor()
    on = WorkspaceConfig(leak=_ON, geometry=_UNREAD)
    assert proc._leak_bake(neg, WorkspaceConfig()) is neg
    assert proc._leak_bake(neg, replace(on, process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=True))) is neg
    out = proc._leak_bake(neg, on)
    assert out is not neg
    np.testing.assert_array_equal(out, _apply(neg))
    assert proc._leak_bake(neg, on) is out
    assert proc._leak_bake(neg, replace(on, leak=_strength(0.5))) is not out


def test_a_flat_master_keeps_leak_removal():
    from negpy.domain.models import flat_master_config

    e = _scene()
    neg = _negative(e + _tilted_band(*e.shape[:2], 90.0))
    flat = flat_master_config(WorkspaceConfig(leak=_ON, geometry=_UNREAD))
    np.testing.assert_array_equal(ImageProcessor()._leak_bake(neg, flat), _apply(neg))


def test_a_cpu_export_removes_the_leak_once(monkeypatch):
    from negpy.domain.models import ExportConfig
    from negpy.services.rendering import image_processor

    e = _scene()
    neg = _negative(e + _tilted_band(*e.shape[:2], 90.0))
    calls = []

    def counted(img, settings):
        calls.append(img)
        return remove_leak(img, settings)

    proc = ImageProcessor()
    monkeypatch.setattr(image_processor, "remove_leak", counted)
    monkeypatch.setattr(proc, "_load_source_f32", lambda *a, **k: (neg, None, "Adobe RGB"))
    config = WorkspaceConfig(leak=_ON, geometry=_UNREAD)
    proc._render_export_buffer("frame.tif", config, ExportConfig(), "h", prefer_gpu=False)
    assert len(calls) == 1
    assert calls[0] is neg


def test_the_broad_blur_matches_the_full_kernel_up_to_the_frame_border():
    from negpy.features.leak.recover import _broad_blur

    rng = np.random.default_rng(3)
    d = cv2.resize(rng.uniform(0.0, 3.0, (24, 36)).astype(np.float32), (1500, 1000), interpolation=cv2.INTER_NEAREST)
    d[:, :60] = 3.0
    sigma = 0.02 * 1500 * 2
    np.testing.assert_allclose(_broad_blur(d, sigma), cv2.GaussianBlur(d, (0, 0), sigma), atol=0.01)
