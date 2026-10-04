from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.clone import apply_clone_strokes, clone_token
from negpy.features.retouch.models import HEAL_SIZE_REF
from negpy.services.rendering import image_processor as ip_mod
from negpy.services.rendering.image_processor import ImageProcessor

H, W = 200, 300


def _scene() -> np.ndarray:
    rng = np.random.default_rng(3)
    img = (0.4 + rng.normal(0, 0.02, (H, W, 3))).astype(np.float32)
    img[:, 150:] *= 1.5
    img[95:105, 70:80] = 0.95
    return img


def _stroke(x, y, dx, dy, *, diameter=24.0, strength=1.0, feather=0.0, match=False, points=None):
    pts = points or [[x / W, y / H]]
    return (pts, diameter * HEAL_SIZE_REF / W, dx / W, dy / H, strength, feather, match)


def test_copies_the_source_over_the_destination():
    img = _scene()
    out = apply_clone_strokes(img, [_stroke(75, 100, -50, 0)])
    np.testing.assert_allclose(out[95:105, 70:80], img[95:105, 20:30])
    assert np.array_equal(out[:, 120:], img[:, 120:])


def test_strength_blends_with_the_original():
    img = _scene()
    out = apply_clone_strokes(img, [_stroke(75, 100, -50, 0, strength=0.25)])
    np.testing.assert_allclose(out[100, 75], 0.75 * img[100, 75] + 0.25 * img[100, 25], rtol=1e-5)


def test_feather_fades_the_edge():
    img = _scene()
    hard = apply_clone_strokes(img, [_stroke(75, 100, -50, 0)])
    soft = apply_clone_strokes(img, [_stroke(75, 100, -50, 0, feather=1.0)])
    edge = (100, 75 + 10)
    assert np.allclose(hard[edge], img[100, 35])
    assert not np.allclose(soft[edge], img[100, 35])
    np.testing.assert_allclose(soft[100, 75], img[100, 25], rtol=1e-5)


def test_match_tone_takes_the_destination_level():
    img = _scene()
    plain = apply_clone_strokes(img, [_stroke(75, 100, 150, 0)])
    matched = apply_clone_strokes(img, [_stroke(75, 100, 150, 0, match=True)])
    surround = img[100, 40:60].mean()
    assert abs(plain[97:103, 72:78].mean() - surround) > 0.15
    assert abs(matched[97:103, 72:78].mean() - surround) < 0.02


def test_a_source_off_the_frame_changes_nothing():
    img = _scene()
    assert np.array_equal(apply_clone_strokes(img, [_stroke(75, 100, -200, 0)]), img)


def test_strokes_apply_in_order():
    img = _scene()
    first = _stroke(75, 100, -50, 0)
    second = _stroke(140, 100, -65, 0)
    out = apply_clone_strokes(img, [first, second])
    np.testing.assert_allclose(out[100, 140], apply_clone_strokes(img, [first])[100, 75])


def test_token_tracks_the_strokes():
    base = WorkspaceConfig().retouch
    assert clone_token(base) == ""
    one = replace(base, clone_strokes=[_stroke(75, 100, -50, 0)])
    two = replace(base, clone_strokes=[_stroke(75, 100, -50, 0), _stroke(10, 10, 5, 5)])
    assert clone_token(one) and clone_token(one) != clone_token(two)


def _config(strokes) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, retouch=replace(base.retouch, clone_strokes=strokes))


def test_pipeline_bake_is_incremental_and_matches_a_full_recompute():
    img = _scene()
    img.setflags(write=False)
    strokes = [_stroke(75, 100, -50, 0), _stroke(140, 100, -65, 0, match=True), _stroke(200, 50, 40, 30, strength=0.5)]
    proc = ImageProcessor()
    seen: list[int] = []
    real = ip_mod.apply_clone_strokes

    def counting(image, s):
        seen.append(len(tuple(s)))
        return real(image, s)

    with patch.object(ip_mod, "apply_clone_strokes", side_effect=counting):
        for n in range(1, len(strokes) + 1):
            out = proc._clone_bake(img, _config(strokes[:n]))
    assert seen == [1, 1, 1]
    np.testing.assert_allclose(out, apply_clone_strokes(img, strokes))
    assert proc._clone_bake(img, _config(strokes[:1])) is not out


def test_run_pipeline_renders_the_clone():
    img = _scene()
    cfg = _config([_stroke(75, 100, -50, 0)])
    proc = ImageProcessor()
    with patch.object(ip_mod, "apply_clone_strokes", wraps=ip_mod.apply_clone_strokes) as spy:
        proc.run_pipeline(img, cfg, "src", render_size_ref=300, prefer_gpu=False, readback_metrics=False)
        proc.run_pipeline(img, cfg, "src", render_size_ref=300, prefer_gpu=False, readback_metrics=False)
    assert spy.call_count == 1


def _controller(source=None, offset=None):
    from negpy.desktop.controller import AppController

    state = SimpleNamespace(
        config=WorkspaceConfig(),
        clone_picking=False,
        clone_source=source,
        clone_offset=offset,
        metrics_lock=MagicMock(),
        last_metrics={"uv_grid": _uv_grid()},
    )
    ctrl = SimpleNamespace(state=state, session=MagicMock(), request_render=MagicMock(), set_status=MagicMock())
    ctrl.session.update_config.side_effect = lambda cfg, persist: setattr(state, "config", cfg)
    commit = AppController.handle_clone_stroke_completed.__get__(ctrl)
    return ctrl, commit


def _uv_grid(h=101, w=101) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([xx / (w - 1), yy / (h - 1)], -1)


def test_first_stroke_fixes_the_offset_and_later_strokes_keep_it():
    ctrl, commit = _controller(source=(0.2, 0.3))
    commit([(0.5, 0.5), (0.6, 0.5)])
    commit([(0.7, 0.8)])
    strokes = ctrl.state.config.retouch.clone_strokes
    assert len(strokes) == 2
    for s in strokes:
        assert abs(s[2] - (-0.3)) < 0.02 and abs(s[3] - (-0.2)) < 0.02


def test_a_stroke_without_a_source_asks_for_one():
    ctrl, commit = _controller()
    ctrl.arm_clone_source = MagicMock()
    commit([(0.5, 0.5)])
    assert ctrl.state.config.retouch.clone_strokes == []
    ctrl.arm_clone_source.assert_called_once_with(True)
    ctrl.set_status.assert_called_once()


def test_a_stroke_while_picking_sets_the_source_and_paints_nothing():
    ctrl, commit = _controller(source=(0.2, 0.3), offset=(0.1, 0.1))
    ctrl.state.clone_picking = True
    ctrl.config_updated = MagicMock()
    from negpy.desktop.controller import AppController

    ctrl.set_clone_source = AppController.set_clone_source.__get__(ctrl)
    commit([(0.7, 0.4), (0.8, 0.4)])
    assert ctrl.state.config.retouch.clone_strokes == []
    assert ctrl.state.clone_picking is False
    assert ctrl.state.clone_offset is None
    assert abs(ctrl.state.clone_source[0] - 0.7) < 0.02 and abs(ctrl.state.clone_source[1] - 0.4) < 0.02
