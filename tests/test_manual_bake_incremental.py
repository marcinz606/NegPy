"""_manual_bake's incremental re-bake: painting one more stroke must cost that stroke
alone, not redo every earlier one, while producing exactly the same pixels as a
from-scratch recompute at every step.
"""

from dataclasses import replace
from unittest.mock import patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.logic import manual_bake_token
from negpy.features.retouch.models import HEAL_SIZE_REF
from negpy.services.rendering import image_processor as ip_mod
from negpy.services.rendering.image_processor import ImageProcessor


def _size_at_ref(diameter_px, shape):
    return diameter_px * HEAL_SIZE_REF / max(shape)


def _grainy(h, w, level=0.5, sigma=0.01, seed=21):
    rng = np.random.default_rng(seed)
    return (np.full((h, w, 3), level) + rng.normal(0, sigma, (h, w, 3))).astype(np.float32)


def _stroke(cx, cy, shape, diameter_px=15.0):
    return ([[cx, cy]], _size_at_ref(diameter_px, shape), 0.0, 0.0)


def _scene():
    """Three well-separated specks, the common case a heal session looks like."""
    img = _grainy(240, 240, seed=7)
    img[30:33, 30:33] = 0.92
    img[120:123, 120:123] = 0.92
    img[190:193, 60:63] = 0.92
    shape = img.shape[:2]
    strokes = [
        _stroke(31.5 / shape[1], 31.5 / shape[0], shape),
        _stroke(121.5 / shape[1], 121.5 / shape[0], shape),
        _stroke(61.5 / shape[1], 191.5 / shape[0], shape),
    ]
    return img, strokes


def _config(strokes, lines=()) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, retouch=replace(base.retouch, manual_heal_strokes=strokes, scratch_lines=list(lines)))


def _source_key(settings: WorkspaceConfig) -> str:
    """Mirrors how run_pipeline folds the stroke identity into source_key: _manual_bake's own
    cache is keyed on it, so a fixed key would mask the re-bake this module tests."""
    return "src" + manual_bake_token(settings.retouch)


def _bake(proc: ImageProcessor, img, settings: WorkspaceConfig, key=None):
    return proc._manual_bake(img, settings, key or _source_key(settings))


class TestIncrementalMatchesFullRecompute:
    def test_appending_one_stroke_at_a_time(self):
        img, strokes = _scene()
        proc = ImageProcessor()
        for n in range(1, len(strokes) + 1):
            settings = _config(strokes[:n])
            got, got_wide = _bake(proc, img, settings)
            want, want_wide = _bake(ImageProcessor(), img, settings)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want), err_msg=f"mismatch at {n} strokes")
            assert len(got_wide) == len(want_wide)

    def test_removing_a_stroke_falls_back_and_still_matches(self):
        """Not a strict append (an undo): recompute, never serve a stale result."""
        img, strokes = _scene()
        proc = ImageProcessor()
        _bake(proc, img, _config(strokes))
        settings2 = _config(strokes[:2])
        got, _ = _bake(proc, img, settings2)
        want, _ = _bake(ImageProcessor(), img, settings2)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_a_new_source_falls_back_and_still_matches(self):
        img, strokes = _scene()
        proc = ImageProcessor()
        settings = _config(strokes[:2])
        _bake(proc, img.copy(), settings)
        got, _ = _bake(proc, img, settings)
        want, _ = _bake(ImageProcessor(), img, settings)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


class TestIncrementalSkipsUnchangedWork:
    def test_appending_a_stroke_heals_only_the_new_one(self):
        img, strokes = _scene()
        proc = ImageProcessor()
        seen: list[int] = []
        real = ip_mod.apply_heal_strokes

        def counting(image, s, *a):
            seen.append(len(tuple(s)))
            return real(image, s, *a)

        with patch.object(ip_mod, "apply_heal_strokes", side_effect=counting):
            for n in range(1, len(strokes) + 1):
                _bake(proc, img, _config(strokes[:n]))
        assert seen == [1, 1, 1], f"expected one new stroke healed per call, got {seen}"

    def test_dust_removal_on_detects_once_and_stays_incremental(self):
        img, strokes = _scene()
        proc = ImageProcessor()
        stats_calls: list[int] = []
        seen: list[int] = []
        real_stats, real_heal = ip_mod.compute_dust_stats, ip_mod.apply_heal_strokes

        def counting_stats(*a, **k):
            stats_calls.append(1)
            return real_stats(*a, **k)

        def counting_heal(image, s, *a):
            seen.append(len(tuple(s)))
            return real_heal(image, s, *a)

        with (
            patch.object(ip_mod, "compute_dust_stats", side_effect=counting_stats),
            patch.object(ip_mod, "apply_heal_strokes", side_effect=counting_heal),
        ):
            for n in range(1, len(strokes) + 1):
                base = _config(strokes[:n])
                settings = replace(base, retouch=replace(base.retouch, dust_remove=True))
                proc.run_pipeline(img, settings, "src", render_size_ref=240, prefer_gpu=False, readback_metrics=False)

        assert len(stats_calls) == 1
        assert seen == [1, 1, 1]


class TestManualBake:
    def test_every_stroke_repairs_its_speck(self):
        img, strokes = _scene()
        out, _ = _bake(ImageProcessor(), img, _config(strokes))
        for y, x in ((30, 30), (120, 120), (190, 60)):
            assert abs(float(out[y : y + 3, x : x + 3].mean()) - 0.5) < 0.05
        assert np.array_equal(out[200:, 150:], img[200:, 150:]), "film far from every stroke is untouched"

    def test_a_line_and_a_stroke_bake_together(self):
        img = _grainy(400, 1200, level=0.45, sigma=0.012, seed=3)
        for x in range(1200):
            img[200:202, x] *= 1.12
        img[60:63, 300:303] = 0.92
        from negpy.features.retouch.logic import trace_scratch

        line = trace_scratch(img, 0.5, 200.5 / 400)
        assert line is not None
        settings = _config([_stroke(301.5 / 1200, 61.5 / 400, img.shape[:2])], lines=[line])
        out, _ = _bake(ImageProcessor(), img, settings)
        assert abs(float(out[60:63, 300:303].mean()) - 0.45) < 0.05, "the stroke repaired"
        before = float(np.abs(img[200:202, 100:1100] - 0.45).mean())
        after = float(np.abs(out[200:202, 100:1100] - 0.45).mean())
        assert after < 0.5 * before, "the line repaired"

    def test_an_already_baked_source_is_not_baked_again(self):
        """The export fallback hands run_pipeline a buffer with the heals in it."""
        img, strokes = _scene()
        settings = _config(strokes[:1])
        proc = ImageProcessor()
        with patch.object(ImageProcessor, "_manual_bake", side_effect=AssertionError("baked twice")):
            proc.run_pipeline(img, settings, "src", render_size_ref=240, prefer_gpu=False, readback_metrics=False, source_baked=True)

    def test_auto_hairs_inpaint_once_before_manual_heals(self):
        """Auto-detected hairs heal before PatchMatch runs and stay cached across manual clicks."""
        img = _grainy(240, 240, level=0.5, sigma=0.01, seed=11)
        # Long dark dust hair (low transmittance -> high density) plus two manual dust specks nearby.
        img[80:82, 40:200] = 0.04
        img[90:93, 120:123] = 0.92
        img[140:143, 120:123] = 0.92
        s1 = _stroke(121.5 / 240, 91.5 / 240, (240, 240), diameter_px=14.0)
        s2 = _stroke(121.5 / 240, 141.5 / 240, (240, 240), diameter_px=14.0)

        proc = ImageProcessor()
        proc.engine_gpu = None
        hair_calls: list[int] = []
        real_hair = ip_mod.apply_hair_inpaint

        def counting_hair(*a, **k):
            hair_calls.append(1)
            return real_hair(*a, **k)

        with patch.object(ip_mod, "apply_hair_inpaint", side_effect=counting_hair):
            for cur in ([s1], [s1, s2]):
                base = _config(cur)
                cfg = replace(base, retouch=replace(base.retouch, dust_remove=True, dust_hair_threshold=0.1))
                proc.run_pipeline(img, cfg, "src", render_size_ref=240, prefer_gpu=False, readback_metrics=False)

        assert len(hair_calls) == 1, f"auto hair re-inpainted on manual stroke append ({len(hair_calls)} calls)"
        assert proc._manual_inc_out is not None
        assert float(proc._manual_inc_out[88:95, 118:125].min()) > 0.25, "unhealed hair leaked into manual heal donor"

    def test_scratch_threshold_change_invalidates_incremental_heal_cache(self):
        """Changing scratch_threshold re-runs lines and re-applies heal strokes on the updated line buffer."""
        img = _grainy(300, 900, level=0.45, sigma=0.012, seed=5)
        img[150:152, :] *= 1.14
        img[60:63, 200:203] = 0.92
        from negpy.features.retouch.logic import trace_scratch

        line = trace_scratch(img, 0.5, 150.5 / 300)
        assert line is not None
        stroke = _stroke(201.5 / 900, 61.5 / 300, img.shape[:2])
        proc = ImageProcessor()

        cfg_lo = replace(_config([stroke], lines=[line]), retouch=replace(_config([stroke], lines=[line]).retouch, scratch_threshold=0.1))
        cfg_hi = replace(cfg_lo, retouch=replace(cfg_lo.retouch, scratch_threshold=0.9))
        out_lo, _ = _bake(proc, img, cfg_lo)
        out_hi, _ = _bake(proc, img, cfg_hi)
        want_hi, _ = _bake(ImageProcessor(), img, cfg_hi)
        np.testing.assert_array_equal(np.asarray(out_hi), np.asarray(want_hi))
        assert not np.array_equal(np.asarray(out_lo), np.asarray(out_hi))

    def test_appending_heal_stroke_invalidates_downstream_clone_cache(self):
        """Clone sits downstream of manual heal and must re-bake when a new heal stroke lands under its source."""
        img, strokes = _scene()
        clone = ([[0.75, 0.75]], 18.0, -61.5 / 240 + 0.75, -191.5 / 240 + 0.75, 1.0, 0.3, False)
        # Note: clone stroke tuple is (points, size, dx, dy, strength, feather, match_tone) where source = dest + (dx, dy)
        clone_from_speck = ([[0.75, 0.75]], 18.0, (61.5 / 240) - 0.75, (191.5 / 240) - 0.75, 1.0, 0.2, False)
        proc = ImageProcessor()
        proc.engine_gpu = None

        cfg1 = replace(_config(strokes[:2]), retouch=replace(_config(strokes[:2]).retouch, clone_strokes=[clone_from_speck]))
        proc.run_pipeline(img, cfg1, "src", render_size_ref=240, prefer_gpu=False, readback_metrics=False)
        before_clone = proc._clone_out.copy()

        # Third heal stroke repairs the speck at (61.5, 191.5) that clone copies from.
        cfg2 = replace(_config(strokes), retouch=replace(_config(strokes).retouch, clone_strokes=[clone_from_speck]))
        proc.run_pipeline(img, cfg2, "src", render_size_ref=240, prefer_gpu=False, readback_metrics=False)
        after_clone = proc._clone_out.copy()
        assert not np.array_equal(before_clone, after_clone), "clone cache served stale unhealed source"
        _ = clone

    def test_wide_scratch_line_inpaints_before_manual_heal_and_caches_across_renders(self):
        """Wide scratch cores hair-inpaint ahead of PatchMatch heals and stay cached across slider drags."""
        img = _grainy(320, 640, level=0.42, sigma=0.01, seed=31)
        # Wide bright scratch band (14 px thick -> Chebyshev radius >= 5 -> routed to apply_hair_inpaint)
        # plus a dark manual defect spot right on top of the wide scratch at x=320.
        img[153:167, 40:600] = 0.92
        line = (40.0 / 640, 160.0 / 320, 600.0 / 640, 160.0 / 320, _size_at_ref(20.0, img.shape[:2]))
        touchup = _stroke(320.5 / 640, 160.0 / 320, img.shape[:2], diameter_px=18.0)

        proc = ImageProcessor()
        proc.engine_gpu = None
        call_order: list[str] = []
        real_hair = ip_mod.apply_hair_inpaint
        real_heal = ip_mod.apply_heal_strokes

        def track_hair(*a, **k):
            call_order.append("hair")
            return real_hair(*a, **k)

        def track_heal(*a, **k):
            call_order.append("heal")
            return real_heal(*a, **k)

        cfg1 = _config([touchup], lines=[line])
        cfg2 = replace(cfg1, exposure=replace(cfg1.exposure, density=cfg1.exposure.density + 0.25))
        with (
            patch.object(ip_mod, "apply_hair_inpaint", side_effect=track_hair),
            patch.object(ip_mod, "apply_heal_strokes", side_effect=track_heal),
        ):
            proc.run_pipeline(img, cfg1, "src", render_size_ref=320, prefer_gpu=False, readback_metrics=False)
            proc.run_pipeline(img, cfg2, "src", render_size_ref=320, prefer_gpu=False, readback_metrics=False)

        assert call_order == ["hair", "heal"], f"expected cached hair->heal order, got {call_order}"
        assert proc._manual_inc_routed is not None
        assert float(proc._manual_inc_out[156:164, 316:325].mean()) < 0.55
