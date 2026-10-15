import numpy as np
import pytest

from negpy.features.flatfield import logic as ff
from negpy.features.flatfield.models import FlatFieldConfig


def _radial_falloff(h: int, w: int) -> np.ndarray:
    """Smooth center-bright / edge-dark illumination map, 3 channels."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    falloff = 1.0 - 0.4 * np.clip(r, 0.0, 1.0)  # 1.0 center → 0.6 corner
    return np.repeat(falloff[:, :, None], 3, axis=2).astype(np.float32)


@pytest.fixture
def gain_store():
    """Back the provider with an in-memory {id: (gain, token)} map; restore on exit."""
    store: dict[str, tuple[np.ndarray, str]] = {}

    def register(profile_id: str, reference: np.ndarray) -> None:
        gain = ff.compute_gain(reference)
        store[profile_id] = (gain, ff.gain_token(gain))

    ff.set_gain_provider(lambda pid: store.get(pid))
    try:
        yield register
    finally:
        ff.set_gain_provider(None)


def test_disabled_is_noop():
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(apply=False, profile_id="anything"), orientation=1)
    assert out is img


def test_empty_profile_is_noop():
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(apply=True, profile_id=""), orientation=1)
    assert out is img


def test_unknown_profile_is_noop(gain_store):
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(apply=True, profile_id="ghost"), orientation=1)
    assert out is img
    assert ff.flatfield_token(FlatFieldConfig(apply=True, profile_id="ghost")) == ""


def test_correction_flattens_uneven_illumination(gain_store):
    h, w = 128, 192
    falloff = _radial_falloff(h, w)
    gain_store("rig", falloff)

    # A uniform scene captured under this illumination is just the falloff map.
    captured = falloff.copy()
    corrected = ff.apply_flatfield(captured, FlatFieldConfig(apply=True, profile_id="rig"), orientation=1)

    # Before: clearly uneven. After: near-flat across the field.
    assert captured.std() > 0.05
    assert corrected.std() < 0.02
    assert corrected.dtype == np.float32


def test_carrier_edge_in_reference_does_not_overcorrect():
    h, w = 128, 192
    clean = _radial_falloff(h, w)
    banded = clean.copy()
    banded[-6:] = 0.01  # dark carrier band along the bottom edge

    expected = clean * ff.compute_gain(clean)
    corrected = banded * ff.compute_gain(banded)
    ratio = corrected[:-10] / expected[:-10]
    assert np.abs(ratio / np.median(ratio) - 1.0).max() < 0.05


def test_half_lit_carrier_on_the_image_border_is_masked():
    h, w = 128, 192
    clean = _radial_falloff(h, w)
    edged = clean.copy()
    edged[0] *= 0.3  # carrier lip on the outermost row, partly lit

    ratio = (edged * ff.compute_gain(edged))[2:] / (clean * ff.compute_gain(clean))[2:]
    assert np.abs(ratio / np.median(ratio) - 1.0).max() < 0.01


def test_soft_carrier_edge_stays_out_of_the_gain():
    h, w = 128, 192
    clean = _radial_falloff(h, w)
    soft = clean.copy()
    soft[-6:] *= 0.15  # carrier rail, lit by stray light
    soft[-9:-6] *= np.linspace(1.0, 0.15, 5, dtype=np.float32)[1:-1, None, None]  # its shadow edge

    ratio = (soft * ff.compute_gain(soft))[:-13] / (clean * ff.compute_gain(clean))[:-13]
    assert np.abs(ratio / np.median(ratio) - 1.0).max() < 0.01


def test_strong_corner_falloff_stays_lit():
    yy, xx = np.mgrid[0:128, 0:192].astype(np.float32)
    r = np.sqrt(((yy - 63.5) / 63.5) ** 2 + ((xx - 95.5) / 95.5) ** 2) / np.sqrt(2)
    reference = np.repeat((1.0 - 0.6 * r)[:, :, None], 3, axis=2)  # corners at 0.4 of the center

    assert ff._lit_mask(reference)[2:-2, 2:-2].all()


def test_carrier_filling_most_of_the_reference_is_masked():
    h, w = 128, 192
    reference = np.full((h, w, 3), 0.01, dtype=np.float32)
    reference[20:108, 40:120] = _radial_falloff(88, 80)  # opening covers under half the frame

    corrected = reference * ff.compute_gain(reference)
    opening = corrected[24:104, 44:116]
    assert opening.max() / opening.min() < 1.5


def test_gain_far_from_the_opening_stays_bounded():
    h, w = 128, 192
    reference = np.full((h, w, 3), 0.01, dtype=np.float32)
    reference[40:88, 60:132] = 1.0

    gain = ff.compute_gain(reference)
    assert gain.max() < 1.5


def test_border_free_reference_matches_the_unmasked_gain():
    import cv2

    reference = _radial_falloff(128, 192)
    sigma = 192 / ff._BLUR_DIVISOR
    blur = cv2.GaussianBlur(reference, (0, 0), sigmaX=sigma, sigmaY=sigma)
    unmasked = blur.reshape(-1, 3).mean(axis=0) / blur

    np.testing.assert_allclose(ff.compute_gain(reference), unmasked, rtol=0.01)


def test_almost_all_dark_reference_uses_the_whole_frame():
    reference = np.full((128, 192, 3), 0.01, dtype=np.float32)
    reference[60:63, 90:93] = 1.0

    gain = ff.compute_gain(reference)
    assert np.isfinite(gain).all()
    assert gain.min() >= 0.25 and gain.max() <= 4.0


def test_gain_resized_to_image(gain_store):
    # Gain baked at one size must resize to a differently-sized working image.
    gain_store("rig", _radial_falloff(64, 64))
    img = np.full((100, 140, 3), 0.5, dtype=np.float32)
    out = ff.apply_flatfield(img, FlatFieldConfig(apply=True, profile_id="rig"), orientation=1)
    assert out.shape == img.shape


def test_token_is_stable_and_profile_scoped(gain_store):
    gain_store("a", _radial_falloff(64, 64))
    gain_store("b", _radial_falloff(48, 72))
    cfg_a = FlatFieldConfig(apply=True, profile_id="a")

    tok = ff.flatfield_token(cfg_a)
    assert tok.startswith("|ff:a:")
    assert tok == ff.flatfield_token(cfg_a)  # deterministic
    assert tok != ff.flatfield_token(FlatFieldConfig(apply=True, profile_id="b"))
    assert ff.flatfield_token(FlatFieldConfig(apply=False, profile_id="a")) == ""


def test_invalidate_drops_cache(gain_store):
    gain_store("rig", _radial_falloff(64, 64))
    cfg = FlatFieldConfig(apply=True, profile_id="rig")
    assert ff.flatfield_token(cfg) != ""  # populates the cache

    ff.set_gain_provider(lambda pid: None)  # provider now yields nothing
    ff.invalidate_gain("rig")
    assert ff.flatfield_token(cfg) == ""


def test_a_half_frame_takes_its_own_half_of_the_gain(gain_store):
    """The preview flat-fields after slicing and the export before; both must match."""
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.assets.half_frame import slice_chain
    from negpy.services.rendering.image_processor import ImageProcessor

    gain_store("p", _radial_falloff(120, 200))
    cfg = FlatFieldConfig(apply=True, profile_id="p")
    scan = np.full((120, 200, 3), 0.4, np.float32) * _radial_falloff(120, 200)
    cut = ((1, 0.5, None, 0.04, "x"),)

    expected = slice_chain(ff.apply_flatfield(scan, cfg, orientation=1), cut)

    half = np.ascontiguousarray(slice_chain(scan, cut))
    np.testing.assert_allclose(ff.apply_flatfield(half, cfg, lambda g: slice_chain(g, cut), orientation=1), expected, rtol=0.02)

    processor = ImageProcessor()
    processor.run_pipeline(
        half,
        replace(WorkspaceConfig(), flatfield=cfg),
        "half",
        render_size_ref=100.0,
        prefer_gpu=False,
        readback_metrics=False,
        gain_slices=cut,
    )
    np.testing.assert_allclose(processor._precorrect_value, expected, rtol=0.02)


def test_without_the_cut_the_gain_would_be_stretched(gain_store):
    from negpy.services.assets.half_frame import slice_chain

    gain_store("p", _radial_falloff(120, 200))
    cfg = FlatFieldConfig(apply=True, profile_id="p")
    scan = np.full((120, 200, 3), 0.4, np.float32) * _radial_falloff(120, 200)
    cut = ((1, 0.5, None, 0.04, "x"),)
    half = np.ascontiguousarray(slice_chain(scan, cut))
    expected = slice_chain(ff.apply_flatfield(scan, cfg, orientation=1), cut)
    assert not np.allclose(ff.apply_flatfield(half, cfg, orientation=1), expected, rtol=0.02)


def test_thumbnail_ir_planes_are_cut_like_the_buffer():
    from negpy.desktop.workers.render import _slice_meta_planes

    ir = np.arange(100 * 200, dtype=np.float32).reshape(100, 200)
    out = _slice_meta_planes({"ir_preview": ir, "detect_preview": None, "x": 1}, {"half": 2, "split_x": 0.5})
    assert out["ir_preview"].shape == (100, 100)
    assert out["ir_preview"][0, 0] == 100
    assert out["detect_preview"] is None and out["x"] == 1


def test_correction_follows_the_light_source_pattern():
    h, w = 160, 240
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Soft LED hot spots behind a diffuser, a few percent deep.
    pattern = 1.0 + 0.03 * np.cos(2 * np.pi * xx / 64.0) * np.cos(2 * np.pi * yy / 64.0)
    reference = np.repeat(pattern[..., None], 3, axis=2)

    corrected = reference * ff.compute_gain(reference)
    interior = corrected[8:-8, 8:-8]
    assert np.abs(interior / np.median(interior) - 1.0).max() < 0.01


def _smooth_falloff(h: int = 128, w: int = 192) -> np.ndarray:
    """Quadratic falloff at half brightness: smooth enough for the gain to follow, never clipped."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (((yy - (h - 1) / 2) / ((h - 1) / 2)) ** 2 + ((xx - (w - 1) / 2) / ((w - 1) / 2)) ** 2) / 2
    return np.repeat((0.5 * (1.0 - 0.3 * r2))[:, :, None], 3, axis=2).astype(np.float32)


def test_check_reference_measures_its_own_self_correction():
    reference = _smooth_falloff()
    check = ff.check_reference(reference, ff.compute_gain(reference))
    assert check.spread < ff.UNEVEN_LIMIT
    assert not check.clipped


def test_check_reference_flags_a_clipped_reference():
    reference = np.clip(_smooth_falloff() * 2.5, 0.0, 1.0)
    assert ff.check_reference(reference, ff.compute_gain(reference)).clipped


def test_check_reference_reads_a_carrier_overcorrection():
    reference = _smooth_falloff()
    reference[-6:] = 0.005
    bad_gain = ff.compute_gain(_smooth_falloff()) * np.linspace(1.0, 1.15, 128)[:, None, None]
    assert ff.check_reference(reference, bad_gain).spread > ff.UNEVEN_LIMIT


def test_evenness_reads_falloff_and_color_shading():
    reference = _smooth_falloff()
    reference[..., 0] *= np.linspace(1.0, 0.9, 192)[None, :]  # red fades to one side
    check = ff.evenness(reference)
    assert check.low < -0.1
    assert check.color > 0.02
    assert not check.clipped


def test_evenness_view_is_gray_where_even_and_black_on_carrier():
    image = np.full((64, 96, 3), 0.5, dtype=np.float32)
    image[:4] = 0.0
    view, check = ff.evenness_view(image)
    np.testing.assert_allclose(view[10:-10, 10:-10], 0.5, atol=1e-6)
    assert view[:4].max() == 0.0
    assert check.spread < 1e-6


def test_a_gain_view_shows_the_light_it_corrects():
    gain = ff.compute_gain(_smooth_falloff())
    view, _ = ff.evenness_view(1.0 / gain, span=ff.GAIN_VIEW_RANGE)
    assert view[64, 96].mean() > view[2, 2].mean()  # brighter center, darker corner


def test_a_profile_check_copy_is_finer_than_the_gain():
    reference = _smooth_falloff(683, 1024)
    assert max(ff.check_copy(reference).shape[:2]) == 512
    assert max(ff.compute_gain(reference).shape[:2]) == 256
    assert ff.check_reference(reference, ff.compute_gain(reference)).spread < ff.UNEVEN_LIMIT


def _sensor_falloff(h: int = 120, w: int = 180) -> np.ndarray:
    """Falloff with no symmetry, so a wrong turn or flip shows."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = (0.6 + 0.4 * xx / w) * (1.0 - 0.2 * yy / h)
    return np.repeat(light[:, :, None], 3, axis=2).astype(np.float32)


def _spread(image: np.ndarray) -> float:
    lum = image.mean(axis=2)
    return float((lum.max() - lum.min()) / np.median(lum))


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_to_sensor_layout_undoes_every_exif_orientation(orientation):
    from negpy.kernel.image.logic import apply_exif_orientation

    sensor = _sensor_falloff(12, 18)
    np.testing.assert_array_equal(ff.to_sensor_layout(apply_exif_orientation(sensor, orientation), orientation), sensor)


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_a_frame_of_any_exif_orientation_is_corrected_where_the_light_fell(gain_store, orientation):
    from negpy.kernel.image.logic import apply_exif_orientation

    light = _sensor_falloff()
    gain_store("rig", light)
    frame = apply_exif_orientation(np.full_like(light, 0.5) * light, orientation)

    out = ff.apply_flatfield(frame, FlatFieldConfig(apply=True, profile_id="rig"), orientation=orientation)

    assert out.shape == frame.shape
    assert _spread(out) < 0.06


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_a_turned_frame_read_as_upright_is_corrected_worse_than_not_at_all(gain_store, orientation):
    from negpy.kernel.image.logic import apply_exif_orientation

    light = _sensor_falloff()
    gain_store("rig", light)
    frame = apply_exif_orientation(np.full_like(light, 0.5) * light, orientation)

    assert _spread(ff.apply_flatfield(frame, FlatFieldConfig(apply=True, profile_id="rig"), orientation=1)) > _spread(frame)


def test_a_turned_half_frame_takes_the_turned_gain_before_its_cut(gain_store):
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.kernel.image.logic import apply_exif_orientation
    from negpy.services.assets.half_frame import slice_chain
    from negpy.services.rendering.image_processor import ImageProcessor

    light = _sensor_falloff(120, 200)
    gain_store("p", light)
    cfg = FlatFieldConfig(apply=True, profile_id="p")
    scan = apply_exif_orientation(np.full_like(light, 0.4) * light, 6)
    cut = ((1, 0.5, None, 0.04, "y"),)
    expected = slice_chain(ff.apply_flatfield(scan, cfg, orientation=6), cut)

    half = np.ascontiguousarray(slice_chain(scan, cut))
    np.testing.assert_allclose(ff.apply_flatfield(half, cfg, lambda g: slice_chain(g, cut), orientation=6), expected, rtol=0.02)

    processor = ImageProcessor()
    processor.run_pipeline(
        half,
        replace(WorkspaceConfig(), flatfield=cfg),
        "half",
        render_size_ref=100.0,
        prefer_gpu=False,
        readback_metrics=False,
        gain_slices=cut,
        source_orientation=6,
    )
    np.testing.assert_allclose(processor._precorrect_value, expected, rtol=0.02)
