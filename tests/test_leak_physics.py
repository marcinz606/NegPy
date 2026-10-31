"""Light leak reduction against a synthetic three-layer negative with a known leak (leak_sim)."""

from dataclasses import replace

import cv2
import leak_sim as sim
import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.leak.bake import leak_scan, remove_leak
from negpy.features.leak.estimate import estimate_leak
from negpy.features.leak.models import LeakColor, LeakConfig
from negpy.features.leak.shapes import new_band, new_edge, new_paint, new_spot
from negpy.features.process.models import ProcessConfig, ProcessMode

_BAND = new_band((0.6, 0.0), (0.6, 1.0), 0.12, 0.12)
_ON = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(_BAND,))


def _density(t: np.ndarray) -> np.ndarray:
    return -np.log10(t)


def _error(frame: sim.Frame, region) -> tuple[np.ndarray, np.ndarray]:
    """Mean density per channel the removal leaves over the clean frame, and the leak's own."""
    out = remove_leak(frame.neg, WorkspaceConfig(leak=_ON, geometry=frame.geometry))[0]
    return (_density(out) - _density(frame.clean))[region].mean(axis=(0, 1)), (_density(frame.neg) - _density(frame.clean))[region].mean(
        axis=(0, 1)
    )


_UNDER_BAND = (slice(sim.REBATE + 10, sim.H - sim.REBATE - 10), slice(500, 580))


def test_a_back_leak_over_gray_comes_off_without_a_cast():
    after, before = _error(sim.leaked(sim.gray_scene(), sim.band(3.0), sim.BACK), _UNDER_BAND)
    assert np.all(np.abs(after) < 0.25 * before)
    # What is left is the leak's own color, not one the removal adds.
    assert np.all(after >= -0.005)


def test_a_dark_subject_in_a_sky_leaves_no_halo():
    frame = sim.leaked(sim.sky_scene(), sim.band(3.0), sim.BACK)
    beside, _ = _error(frame, (slice(60, 250), slice(505, 530)))
    above, before = _error(frame, (slice(55, 100), slice(525, 555)))
    assert beside[0] < 0.3 * before[0] and above[0] < 0.3 * before[0]
    assert abs(beside[0] - above[0]) < 0.25 * before[0]


def test_the_color_comes_from_lit_rebate():
    frame = sim.leaked(sim.sky_scene(seed=3), sim.band(3.0), sim.BACK)
    found = estimate_leak(frame.neg, _ON.leak_shapes, geometry=frame.geometry)
    np.testing.assert_allclose(found.readings[0].color, sim.BACK, atol=0.03)


def test_a_front_leak_reads_about_neutral():
    frame = sim.leaked(sim.sky_scene(), sim.band(3.0), sim.FRONT, front=True)
    color = estimate_leak(frame.neg, _ON.leak_shapes, geometry=frame.geometry).readings[0].color
    assert float(color.min()) > 0.7


def test_the_unmix_reads_the_layers_through_crosstalk():
    frame = sim.leaked(sim.gray_scene(), sim.band(3.0), sim.BACK, mix=sim.MIX)
    unmixed = estimate_leak(frame.neg, _ON.leak_shapes, geometry=frame.geometry, unmix=np.linalg.inv(sim.MIX)).readings[0].color
    mixed = estimate_leak(frame.neg, _ON.leak_shapes, geometry=frame.geometry).readings[0].color
    np.testing.assert_allclose(unmixed, sim.BACK, atol=0.03)
    assert np.abs(mixed - sim.BACK).max() > 0.05


@pytest.mark.parametrize("color, expected", [(LeakColor.BACK, sim.BACK), (LeakColor.FRONT, np.array([0.5, 0.8, 1.0]))])
def test_a_set_color_wins_over_the_film(color, expected):
    from dataclasses import replace

    frame = sim.leaked(sim.gray_scene(), sim.band(3.0), sim.BACK)
    found = estimate_leak(frame.neg, (replace(_BAND, color=color),), geometry=frame.geometry)
    np.testing.assert_allclose(found.readings[0].color / found.readings[0].color.max(), expected / expected.max(), atol=0.05)


@pytest.mark.parametrize("color", [LeakColor.BACK, LeakColor.FRONT])
def test_black_and_white_film_reads_a_set_color_off_the_film(color):
    """One layer takes any light alike, so a black-and-white leak has no color to set."""
    frame = sim.leaked(sim.gray_scene(), sim.band(3.0), sim.FRONT)
    shapes = (replace(_BAND, color=color),)
    found = estimate_leak(frame.neg, shapes, geometry=frame.geometry, color_film=False)
    np.testing.assert_allclose(found.readings[0].color, sim.FRONT, atol=0.05)
    bw = WorkspaceConfig(leak=replace(_ON, leak_shapes=shapes), geometry=frame.geometry, process=ProcessConfig(process_mode=ProcessMode.BW))
    out = remove_leak(frame.neg, bw)[0]
    removed = (_density(frame.neg) - _density(out))[_UNDER_BAND].mean(axis=(0, 1))
    assert float(removed.max() - removed.min()) < 0.1 * float(removed.max())


def test_color_trim_takes_more_of_the_leading_layer_alone():
    frame = sim.leaked(sim.gray_scene(), sim.band(3.0), sim.GATE)
    plain = estimate_leak(frame.neg, (_BAND,), geometry=frame.geometry).readings[0]
    trimmed = estimate_leak(frame.neg, (replace(_BAND, trim=0.2),), geometry=frame.geometry).readings[0]
    lead = int(np.argmax(plain.color))
    ratio = trimmed.field.sum(axis=(0, 1)) / plain.field.sum(axis=(0, 1))
    assert lead == 2
    np.testing.assert_allclose(ratio, [1.0, 1.0, 1.2], rtol=1e-4)


def test_black_and_white_film_ignores_a_color_trim():
    frame = sim.leaked(sim.gray_scene(), sim.band(3.0), sim.FRONT)
    plain = estimate_leak(frame.neg, (_BAND,), geometry=frame.geometry, color_film=False).field
    trimmed = estimate_leak(frame.neg, (replace(_BAND, trim=0.3),), geometry=frame.geometry, color_film=False).field
    np.testing.assert_array_equal(plain, trimmed)


def test_edge_fog_alone_is_not_read_as_a_leak():
    neg = sim.negative(sim.edge_fog(sim.gray_scene(), 6.0))
    config = WorkspaceConfig(leak=LeakConfig(leak_remove=True), geometry=sim.crop())
    assert remove_leak(neg, config)[0] is neg


_TOP = sim.REBATE / sim.H
_TOP_EDGE = new_edge((0.0, _TOP), (1.0, _TOP), 0.3)
_BELOW_TOP = (slice(sim.REBATE, sim.REBATE + 200), slice(0, sim.W))


def _edge_removed(frame: sim.Frame, shapes) -> float:
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=tuple(shapes))
    out = remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=frame.geometry))[0]
    before = (_density(frame.neg) - _density(frame.clean))[_BELOW_TOP][..., 0].mean()
    after = (_density(out) - _density(frame.clean))[_BELOW_TOP][..., 0].mean()
    return 1.0 - after / before


def test_an_edge_takes_off_a_leak_with_tongues():
    frame = sim.leaked(sim.gray_scene(), sim.edge_leak(3.0), sim.BACK)
    assert 0.5 < _edge_removed(frame, (_TOP_EDGE,)) < 1.1


def test_an_edge_takes_off_a_steep_tip_and_leaves_the_picture_past_it():
    """Far steeper than a depth bin: the rebate's own fade, carried across the frame line, reads it."""
    frame = sim.leaked(sim.gray_scene(), sim.edge_tip(20.0), sim.BACK)
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(_TOP_EDGE,))
    out = remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=frame.geometry))[0]
    after = _density(out) - _density(frame.clean)
    before = _density(frame.neg) - _density(frame.clean)
    for top, bottom in ((0, 4), (4, 8), (8, 16)):
        band = slice(sim.REBATE + top, sim.REBATE + bottom)
        left = after[band][..., 0].mean() / before[band][..., 0].mean()
        assert 0.0 <= left < 0.5, (top, bottom, left)
    assert float(np.abs(after[sim.REBATE + 30 : sim.REBATE + 120]).mean()) < 0.01


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_an_edge_never_takes_more_than_the_leak_where_its_tongues_are_short(seed):
    """Bright scene with no shadows by the edge reads as leak; the rebate's profile sets the shape there."""
    frame = sim.leaked(sim.gray_scene(seed=seed), sim.edge_leak(3.0), sim.BACK)
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(_TOP_EDGE,))
    out = remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=frame.geometry))[0]
    rows = slice(sim.REBATE + 2, sim.REBATE + 120)
    for x in range(0, sim.W, 150):
        stretch = (rows, slice(x, x + 150))
        before = (_density(frame.neg) - _density(frame.clean))[stretch][..., 0].mean()
        after = (_density(out) - _density(frame.clean))[stretch][..., 0].mean()
        assert -0.1 < after / before < 0.6


def test_an_edge_reads_a_gap_leak_at_the_frame_line():
    """The rebate's thin end lies past the gap, where the leak has faded; at the line it is whole."""
    frame = sim.leaked(sim.gray_scene(), sim.gap_leak(3.0), sim.GATE)
    assert 0.8 < _edge_removed(frame, (_TOP_EDGE,)) < 1.1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_crop_set_past_the_frame_line_clears_its_gap_and_reads_the_picture_from_the_line(seed):
    """Half the rebate inside the crop: the gap there goes to bare base, and the picture's leak is read
    from the frame line, not from the crop's edge."""
    frame = sim.leaked(sim.gray_scene(seed=seed), sim.gap_leak(3.0), sim.GATE)
    x0, y0, x1, y1 = frame.geometry.crop_rect
    top = y0 - 0.5 * _TOP
    geometry = replace(frame.geometry, crop_rect=(x0, top, x1, y1))
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(new_edge((0.0, top), (1.0, top), 0.3),))
    out = remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=geometry))[0]
    after = _density(out) - _density(frame.clean)
    before = _density(frame.neg) - _density(frame.clean)
    gap = slice(sim.REBATE - sim.REBATE // 2 + 2, sim.REBATE - 2)
    assert float(after[gap][..., 2].mean()) < 0.03
    assert 0.75 < 1.0 - after[_BELOW_TOP][..., 0].mean() / before[_BELOW_TOP][..., 0].mean() < 1.1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auto_clears_lit_rebate_inside_a_wide_crop_where_no_leak_crosses(seed):
    """Fog on the rebate alone: inside a crop set past the frame line Auto clears it to bare base and
    leaves the picture, and a crop at the line gets no shape."""
    from negpy.features.leak.auto import auto_edges

    scene = sim.gray_scene(seed=seed)
    detail = cv2.GaussianBlur(np.random.default_rng(seed + 10).uniform(0.0, 1.0, scene.shape[:2]).astype(np.float32), (0, 0), 1.5)
    scene = scene * (0.4 + 1.2 * detail)[..., None]
    neg, clean = sim.negative(sim.edge_fog(scene, 6.0, sim.GATE)), sim.negative(scene)
    x0, y0, x1, y1 = sim.crop().crop_rect
    assert auto_edges(neg, sim.crop()) == []
    geometry = replace(sim.crop(), crop_rect=(x0, y0 - 0.5 * _TOP, x1, y1))
    edges = auto_edges(neg, geometry)
    assert edges
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=tuple(edges))
    out = remove_leak(neg, WorkspaceConfig(leak=leak, geometry=geometry))[0]
    gap = slice(sim.REBATE - sim.REBATE // 2 + 2, sim.REBATE - 2)
    assert abs(float((_density(out) - _density(clean))[gap].mean())) < 0.02
    assert float(np.abs(_density(out) - _density(neg))[sim.REBATE + 4 : sim.REBATE + 200].mean()) < 0.005


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auto_clears_both_stretches_of_rebate_that_steps_down_toward_the_frame_line(seed):
    """Light-struck rebate beyond a dimmer gap, as at the start of a roll: the edge between them reads as
    texture, yet the rebate runs on to the frame line and both stretches go to bare base."""
    from negpy.features.leak.auto import auto_edges

    scene = sim.gray_scene(seed=seed)
    detail = cv2.GaussianBlur(np.random.default_rng(seed + 10).uniform(0.0, 1.0, scene.shape[:2]).astype(np.float32), (0, 0), 1.5)
    scene = scene * (0.4 + 1.2 * detail)[..., None]
    half = sim.REBATE // 2
    lit = scene.copy()
    lit[:half] += 8.0 * sim.GATE
    lit[half : sim.REBATE] += 1.0 * sim.GATE
    neg, clean = sim.negative(lit), sim.negative(scene)
    x0, _, x1, y1 = sim.crop().crop_rect
    geometry = replace(sim.crop(), crop_rect=(x0, 0.25 * _TOP, x1, y1))
    edges = auto_edges(neg, geometry)
    assert edges
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=tuple(edges))
    out = remove_leak(neg, WorkspaceConfig(leak=leak, geometry=geometry))[0]
    left = _density(out) - _density(clean)
    for rows in (slice(sim.REBATE // 4 + 2, half - 3), slice(half + 3, sim.REBATE - 3)):
        assert abs(float(left[rows].mean())) < 0.02, rows
    assert float(np.abs(_density(out) - _density(neg))[sim.REBATE + 4 : sim.REBATE + 200].mean()) < 0.005


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_an_edge_holds_the_shoulder_of_a_gate_shadow(seed):
    """The soft shadow of a gap holds a shoulder by the line, which an exponential fade drops too soon."""
    frame = sim.leaked(0.1 * sim.gray_scene(seed), sim.gate_shadow(4.0), sim.GATE)
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(_TOP_EDGE,))
    out = remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=frame.geometry))[0]
    near = (slice(sim.REBATE + 10, sim.REBATE + 60), slice(0, sim.W))
    left = (_density(out) - _density(frame.clean))[near][..., 2].mean() / (_density(frame.neg) - _density(frame.clean))[near][..., 2].mean()
    assert -0.1 < left < 0.4


def test_an_edge_reads_a_gap_leak_color_at_the_frame_line():
    """Edge print past the gap glows in its own color, which the rebate's median would mix in."""
    leak = sim.gap_leak(3.0)[..., None] * sim.GATE
    leak[:15] += np.array([2.0, 0.3, 0.1], dtype=np.float32)
    neg = sim.negative(sim.gray_scene() + leak)
    color = estimate_leak(neg, (_TOP_EDGE,), geometry=sim.crop()).readings[0].color
    np.testing.assert_allclose(color, sim.GATE, atol=0.05)


def test_an_edge_over_sky_never_takes_the_sky_for_a_leak():
    """Smooth sky holds no shadows to read a fade by, and its own gradient fades toward an edge too."""
    frame = sim.leaked(sim.sky_scene(), sim.edge_leak(3.0), sim.BACK)
    assert -0.05 < _edge_removed(frame, (_TOP_EDGE,)) < 1.1


@pytest.mark.parametrize("color", [sim.BACK, sim.GATE], ids=["through the base", "through the gate"])
def test_an_edge_reads_the_color_from_the_lit_rebate(color):
    frame = sim.leaked(sim.gray_scene(), sim.edge_leak(3.0), color)
    np.testing.assert_allclose(estimate_leak(frame.neg, (_TOP_EDGE,), geometry=frame.geometry).readings[0].color, color, atol=0.03)


def test_an_edge_over_edge_fog_leaves_the_picture():
    neg = sim.negative(sim.edge_fog(sim.gray_scene(), 6.0))
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(_TOP_EDGE,))
    out = remove_leak(neg, WorkspaceConfig(leak=leak, geometry=sim.crop()))[0]
    assert float(np.abs(_density(out) - _density(neg))[_BELOW_TOP].mean()) < 0.01


# Fog from the cassette or the bath, on the rebate alone.
_FOG = np.array([0.4, 0.5, 1.0], dtype=np.float32)


def test_an_edge_beside_fogged_rebate_reads_no_leak_from_the_picture():
    """A bright scene with no shadows by the edge would otherwise read as leak."""
    neg = sim.negative(sim.edge_fog(sim.gray_scene(seed=2), 6.0, _FOG))
    leak = LeakConfig(leak_remove=True, leak_shapes=(_TOP_EDGE,))
    out = remove_leak(neg, WorkspaceConfig(leak=leak, geometry=sim.crop()))[0]
    np.testing.assert_array_equal(out, neg)


def test_a_set_color_still_reads_a_lens_side_leak_beside_fogged_rebate():
    front = replace(_TOP_EDGE, color=LeakColor.FRONT)
    bare = _edge_removed(sim.leaked(sim.gray_scene(), sim.edge_leak(3.0), sim.FRONT, front=True), (front,))
    frame = sim.leaked(sim.edge_fog(sim.gray_scene(), 6.0, _FOG), sim.edge_leak(3.0), sim.FRONT, front=True)
    assert _edge_removed(frame, (_TOP_EDGE,)) == pytest.approx(0.0, abs=1e-6)
    assert bare > 0.0 and _edge_removed(frame, (front,)) >= 0.5 * bare


@pytest.mark.parametrize("color", [sim.BACK, sim.GATE], ids=["through the base", "through the gate"])
def test_auto_finds_an_edge_leak_the_rebate_carries(color):
    from negpy.features.leak.auto import auto_edges

    frame = sim.leaked(sim.gray_scene(), sim.edge_leak(3.0), color)
    edges = auto_edges(frame.neg, frame.geometry)
    assert len(edges) == 1
    (x0, y0), (x1, y1) = edges[0].points
    assert abs(y0 - _TOP) < 0.02 and abs(y1 - _TOP) < 0.02 and abs(x1 - x0) > 0.6
    assert _edge_removed(frame, edges) > 0.5


@pytest.mark.parametrize(
    "neg",
    [
        sim.negative(sim.edge_fog(sim.gray_scene(), 6.0)),
        sim.negative(sim.edge_fog(sim.gray_scene(seed=2), 6.0, _FOG)),
        sim.negative(sim.gray_scene()),
        sim.negative(sim.sky_scene()),
        sim.leaked(sim.gray_scene(), sim.band(3.0), sim.BACK).neg,
    ],
    ids=["edge fog", "blue edge fog", "clean", "clean sky", "a band"],
)
def test_auto_finds_no_edge_where_the_rebate_shows_none_crossing(neg):
    from negpy.features.leak.auto import auto_edges

    assert auto_edges(neg, sim.crop()) == []


def _bottom_side(strip: np.ndarray, line):
    from negpy.features.leak.rebate import RebateSide

    return RebateSide("bottom", strip, np.median(strip, axis=0), 0.05, line)


def test_a_crossing_lit_past_the_holes_alone_is_edge_print():
    """Edge print past the sprocket holes lifts the rebate in its own color and never reaches the
    frame line, which a leak crossing it lights."""
    from negpy.features.leak.rebate import rebate_crossing

    strip = np.full((40, 3), 0.1)
    strip[18:24] = (8.0, 0.3, 0.1)
    rect = (50, 550, 0, 900)
    crossing = ((450.0, 300.0), (0.0, 1.0))
    bare = np.zeros((40, 3))
    assert rebate_crossing(*crossing, rect, [_bottom_side(strip, None)]).size
    assert rebate_crossing(*crossing, rect, [_bottom_side(strip, bare)]).size == 0
    lit = bare.copy()
    lit[18:24] = (2.0, 3.6, 4.5)
    np.testing.assert_allclose(rebate_crossing(*crossing, rect, [_bottom_side(strip, lit)]), (2.0, 3.6, 4.5))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auto_finds_no_edge_past_a_crop_set_inside_the_picture(seed):
    """Past a crop set inside soft picture lies more of it, and a bright stretch inside the crop line
    over darker picture further in carries across the line as a leak would; it stands no brighter
    over the picture's own shadows than the picture past the crop does."""
    from negpy.features.geometry.models import GeometryConfig
    from negpy.features.leak.auto import auto_edges

    top, soft = sim.REBATE + 90, sim.H - sim.REBATE - 50
    scene = sim.gray_scene(seed)
    scene[sim.REBATE : soft] = 0.1 * scene[sim.REBATE : soft] + 1.0
    scene[top : top + 60] += 1.0
    assert auto_edges(sim.negative(scene), GeometryConfig(crop_rect=(0.0, top / sim.H, 1.0, 1.0 - sim.REBATE / sim.H))) == []


def _spot_over(center=(0.4, 0.45), radius: float = 0.1):
    """A spot drawn over sim.spot: its oval reaches where the leak has faded."""
    cx, cy = center
    rx, ry = radius, radius * sim.W / sim.H
    return new_spot((cx, cy), (cx + rx, cy), (cx, cy + ry))


def _region_removed(frame: sim.Frame, shapes, region) -> float:
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=tuple(shapes))
    out = remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=frame.geometry))[0]
    before = (_density(frame.neg) - _density(frame.clean))[region].mean(axis=(0, 1))
    after = (_density(out) - _density(frame.clean))[region].mean(axis=(0, 1))
    return 1.0 - after / before


_AT_SPOT = (slice(int(0.45 * sim.H) - 40, int(0.45 * sim.H) + 40), slice(int(0.4 * sim.W) - 40, int(0.4 * sim.W) + 40))


@pytest.mark.parametrize("color, front", [(sim.FRONT, True), (sim.BACK, True)])
def test_a_spot_takes_off_a_round_leak(color, front):
    frame = sim.leaked(sim.gray_scene(), sim.spot(3.0), color, front=front)
    removed = _region_removed(frame, (_spot_over(),), _AT_SPOT)
    assert np.all(removed > 0.5) and np.all(removed < 1.2)


def test_a_spot_on_a_clean_frame_changes_little():
    neg = sim.negative(sim.gray_scene())
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(_spot_over(),))
    out = remove_leak(neg, WorkspaceConfig(leak=leak, geometry=sim.crop()))[0]
    assert float(np.abs(_density(out) - _density(neg)).mean()) < 0.005


def test_a_disabled_shape_removes_nothing():
    from dataclasses import replace

    frame = sim.leaked(sim.gray_scene(), sim.spot(3.0), sim.FRONT, front=True)
    leak = LeakConfig(leak_remove=True, leak_shapes=(replace(_spot_over(), enabled=False),))
    assert remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=frame.geometry))[0] is frame.neg


# A leak that drives the red-sensitive layer to the shoulder while green and blue keep picture.
_RED_ONLY = np.array([1.0, 0.002, 0.001], dtype=np.float32)
_SHOULDER = 1.0
_CORE = (slice(sim.REBATE + 10, sim.H - sim.REBATE - 10), slice(535, 545))


def _saturating(color=_RED_ONLY) -> sim.Frame:
    return sim.leaked(sim.gray_scene() * 0.5, sim.band(3000.0, width=0.03), color, shoulder=_SHOULDER)


def _recovered(frame: sim.Frame, recover: bool, dmax: float = 0.0):
    leak = LeakConfig(leak_remove=True, leak_shapes=(_BAND,), leak_dmax=dmax, leak_recover=recover, leak_grain=0.0, leak_contrast=0.0)
    return remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=frame.geometry))


def test_a_layer_the_leak_saturated_is_rebuilt_from_the_others():
    frame = _saturating()
    plain, _ = _recovered(frame, recover=False)
    rebuilt, _ = _recovered(frame, recover=True)
    clean = _density(frame.clean)[_CORE]

    def red(out):
        d = _density(out)[_CORE][..., 0]
        return float(np.abs(d - clean[..., 0]).mean()), float(np.corrcoef(d.ravel(), clean[..., 0].ravel())[0, 1])

    (err_plain, corr_plain), (err_rebuilt, corr_rebuilt) = red(plain), red(rebuilt)
    assert err_rebuilt < 0.7 * err_plain
    assert corr_rebuilt > 0.7 > corr_plain
    np.testing.assert_allclose(_density(rebuilt)[_CORE][..., 1:], _density(plain)[_CORE][..., 1:], atol=1e-4)


def test_a_leak_through_every_layer_is_burned_and_left():
    frame = _saturating(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    plain, burned = _recovered(frame, recover=False)
    rebuilt, _ = _recovered(frame, recover=True)
    assert burned is not None and burned[_CORE].mean() > 0.5
    # No layer holds picture to rebuild from: the burned core stays as removal left it.
    np.testing.assert_allclose(_density(rebuilt)[_CORE], _density(plain)[_CORE], atol=1e-3)


def test_film_dmax_sets_where_a_layer_saturates():
    from negpy.features.leak.film import FilmResponse
    from negpy.features.leak.recover import layer_saturation

    base = sim.BASE
    density = np.zeros((40, 40, 3), np.float32)
    density[:, :20] = 0.97
    density[:, 20:] = 0.9
    small = (base * 10.0 ** (-density)).astype(np.float32)
    leak = np.ones((40, 40, 3), np.float32)
    shoulder = FilmResponse(base, shoulder=np.full(3, _SHOULDER, np.float32))
    saturated = layer_saturation(small, shoulder, leak)
    assert saturated[:, 5:15].all() and not saturated[:, 25:35].any()
    straight = layer_saturation(small, FilmResponse(base), leak)
    assert straight[:, 5:15].all(), "with no Film Dmax, the densest film is the ceiling"


def test_recovery_off_leaves_the_saturated_layer_as_removal_left_it():
    frame = _saturating()
    plain, _ = _recovered(frame, recover=False)
    other, _ = _recovered(frame, recover=False)
    np.testing.assert_array_equal(plain, other)


def test_shapes_carry_by_the_picture_box_and_back():
    from negpy.features.leak.roll import map_shapes

    a_box, a_hw = (50, 550, 100, 800), (600, 900)
    b_box, b_hw = (80, 520, 40, 700), (600, 760)
    paint = new_paint([(0.2, 0.3), (0.4, 0.5)], 40.0)
    shapes = (new_band((0.6, 0.1), (0.6, 0.9), 0.1, 0.05), new_edge((0.2, 50 / 600), (0.7, 50 / 600), 0.2), _spot_over(), paint)
    there = map_shapes(shapes, (a_box, a_hw), (b_box, b_hw))
    (x0, y0), _ = there[1].points
    assert abs(y0 * 600 - 80) < 1e-3, "an edge on the picture's top stays on it"
    assert there[3].strokes != paint.strokes, "a painted leak moves with the picture box"
    back = map_shapes(there, (b_box, b_hw), (a_box, a_hw))
    for s, b in zip(shapes, back):
        np.testing.assert_allclose(np.array(b.points), np.array(s.points), atol=1e-6)
        np.testing.assert_allclose(b.width, s.width, rtol=1e-6)
        for (sp, ss), (bp, bs) in zip(s.strokes, b.strokes):
            np.testing.assert_allclose(np.array(bp), np.array(sp), atol=1e-6)
            assert abs(bs - ss) < 1e-6


def test_a_carried_leak_is_read_again_on_each_frame():
    """The leak sits at the same place on the film frame, at a strength set by how long the film sat
    behind the gap; carried by the picture box, each frame reads its own strength."""
    from negpy.features.leak.roll import map_shapes, picture_box

    first = sim.leaked(sim.gray_scene(seed=0), sim.band(2.0), sim.BACK)
    second = sim.leaked(sim.gray_scene(seed=7), sim.band(5.0), sim.BACK)
    carried = map_shapes((_BAND,), picture_box(first.neg, first.geometry), picture_box(second.neg, second.geometry))
    levels = []
    for frame, shapes in ((first, (_BAND,)), (second, carried)):
        assert _region_removed(frame, shapes, _UNDER_BAND)[0] > 0.6
        levels.append(float(estimate_leak(frame.neg, shapes, geometry=frame.geometry).field[..., 0].max()))
    assert levels[1] > 1.5 * levels[0]


# A generous zig-zag over sim.blotch, as a hand paints a leak: wider than the leak itself.
_PAINTED = new_paint([(0.22, 0.3), (0.48, 0.3), (0.22, 0.42), (0.48, 0.42), (0.22, 0.55), (0.48, 0.55), (0.22, 0.68), (0.48, 0.68)], 160.0)
_IN_BLOTCH = (slice(int(0.4 * sim.H), int(0.6 * sim.H)), slice(int(0.3 * sim.W), int(0.4 * sim.W)))


@pytest.mark.parametrize("color", [sim.FRONT, sim.BACK])
def test_paint_takes_off_a_blotch_with_no_line_or_center(color):
    frame = sim.leaked(sim.gray_scene(), sim.blotch(3.0), color, front=True)
    removed = _region_removed(frame, (_PAINTED,), _IN_BLOTCH)
    assert 0.75 < removed[0] < 1.2


def test_paint_reads_the_leak_color_where_the_leak_is_strong():
    frame = sim.leaked(sim.gray_scene(), sim.blotch(3.0), sim.BACK, front=True)
    color = estimate_leak(frame.neg, (_PAINTED,), geometry=frame.geometry).readings[0].color
    np.testing.assert_allclose(color, sim.BACK, atol=0.06)


def test_paint_on_a_clean_frame_changes_little():
    neg = sim.negative(sim.gray_scene())
    leak = LeakConfig(leak_remove=True, leak_grain=0.0, leak_contrast=0.0, leak_shapes=(_PAINTED,))
    out = remove_leak(neg, WorkspaceConfig(leak=leak, geometry=sim.crop()))[0]
    assert float(np.abs(_density(out) - _density(neg)).mean()) < 0.01


_E6 = ProcessConfig(process_mode=ProcessMode.E6)


def test_a_slide_leak_comes_off_on_the_reversed_curve():
    frame = sim.leaked(sim.gray_scene(), sim.band(3.0), sim.BACK, on_slide=True)
    config = WorkspaceConfig(leak=_ON, geometry=frame.geometry, process=_E6)
    out, _ = remove_leak(frame.neg, config)
    after = (_density(out) - _density(frame.clean))[_UNDER_BAND].mean(axis=(0, 1))
    before = (_density(frame.neg) - _density(frame.clean))[_UNDER_BAND].mean(axis=(0, 1))
    # A leak thins a slide: it reads as density lost, and removal puts it back.
    assert np.all(before < 0.0) and np.all(np.abs(after) < 0.25 * np.abs(before))
    color = estimate_leak(leak_scan(frame.neg, config), _ON.leak_shapes, geometry=frame.geometry).readings[0].color
    np.testing.assert_allclose(color, sim.BACK, atol=0.03)


def test_a_slide_burned_to_clear_film_is_marked():
    frame = sim.leaked(sim.gray_scene() * 0.5, sim.band(3000.0, width=0.03), sim.FRONT, on_slide=True)
    leak = LeakConfig(leak_remove=True, leak_shapes=(_BAND,), leak_grain=0.0, leak_contrast=0.0)
    _, burned = remove_leak(frame.neg, WorkspaceConfig(leak=leak, geometry=frame.geometry, process=_E6))
    assert burned is not None
    burned = cv2.resize(burned.astype(np.uint8), (sim.W, sim.H), interpolation=cv2.INTER_NEAREST).astype(bool)
    assert burned[_CORE].mean() > 0.5 and not burned[:, :400].any()


@pytest.mark.parametrize(
    "pattern, run",
    [
        ("################...#.#.....#", (0, 28)),
        ("#.....#.#...################", (0, 28)),
        ("..##..............#####.....", (18, 23)),
        ("#....#....##....#....#......", None),
    ],
)
def test_auto_joins_stretches_of_one_edge_leak_the_same_way_in_either_direction(monkeypatch, pattern, run):
    """A lone cell beside a long run joins it whichever end of the side it lies at, and short runs
    alone are no leak."""
    from types import SimpleNamespace

    import negpy.features.leak.auto as auto

    strong = np.array([c == "#" for c in pattern])
    n = strong.size
    rows = SimpleNamespace(n=n, supported=strong, amplitude=np.where(strong, 1.0, 0.0), extent=np.full(n, 0.5))
    monkeypatch.setattr(auto, "rebate_along", lambda *a: np.ones((8, 3)))
    monkeypatch.setattr(auto, "edge_rows", lambda reading, edge: rows)
    reading = SimpleNamespace(small=np.zeros((100, 100, 3)), rect=None, profiles=None, response=None)
    found = auto._side_edges(reading, ((0.0, 0.0), (0.0, 1.0)))
    if run is None:
        assert found == []
        return
    assert len(found) == 1
    (_, y0), (_, y1) = found[0].points
    assert (round(y0 * n), round(y1 * n)) == run
