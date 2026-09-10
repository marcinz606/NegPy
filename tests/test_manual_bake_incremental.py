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
    """Three well-separated specks, far enough apart that no two strokes' padded
    fill supports ever touch — the common case a heal session actually looks like."""
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


def _config(strokes) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, retouch=replace(base.retouch, manual_heal_strokes=strokes))


def _source_key(settings: WorkspaceConfig) -> str:
    """Mirrors how run_pipeline folds the stroke identity into source_key — _manual_bake's
    own cache is keyed on this, so a fixed key across different stroke lists would mask
    the very re-bake this module tests."""
    return "src" + manual_bake_token(settings.retouch)


class TestIncrementalMatchesFullRecompute:
    def test_appending_one_stroke_at_a_time(self):
        img, strokes = _scene()
        proc = ImageProcessor()
        for n in range(1, len(strokes) + 1):
            settings = _config(strokes[:n])
            key = _source_key(settings)
            got, got_wide = proc._manual_bake(img, settings, key)
            want, want_wide = ImageProcessor()._manual_bake(img, settings, key)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want), err_msg=f"mismatch at {n} strokes")
            assert (got_wide is None) == (want_wide is None)

    def test_appending_all_at_once_matches_one_at_a_time(self):
        """The end state must not depend on how you got there."""
        img, strokes = _scene()
        stepwise = ImageProcessor()
        for n in range(1, len(strokes) + 1):
            settings = _config(strokes[:n])
            out, _ = stepwise._manual_bake(img, settings, _source_key(settings))

        settings_all = _config(strokes)
        direct_out, _ = ImageProcessor()._manual_bake(img, settings_all, _source_key(settings_all))
        np.testing.assert_array_equal(np.asarray(out), np.asarray(direct_out))

    def test_a_stroke_that_finds_nothing_still_matches(self):
        """Painting on clean film contributes no score — the incremental path must not
        treat that as if the earlier strokes' bake had changed."""
        img, strokes = _scene()
        shape = img.shape[:2]
        clean_stroke = _stroke(0.5, 0.5, shape, diameter_px=6.0)  # nothing there
        proc = ImageProcessor()
        settings1 = _config(strokes[:1])
        proc._manual_bake(img, settings1, _source_key(settings1))
        settings2 = _config(strokes[:1] + [clean_stroke])
        got, _ = proc._manual_bake(img, settings2, _source_key(settings2))
        want, _ = ImageProcessor()._manual_bake(img, settings2, _source_key(settings2))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_removing_a_stroke_falls_back_and_still_matches(self):
        """Not a strict append (an undo) — must recompute correctly, not serve a stale
        3-stroke result trimmed to 2."""
        img, strokes = _scene()
        proc = ImageProcessor()
        settings3 = _config(strokes)
        proc._manual_bake(img, settings3, _source_key(settings3))

        settings2 = _config(strokes[:2])
        got, _ = proc._manual_bake(img, settings2, _source_key(settings2))
        want, _ = ImageProcessor()._manual_bake(img, settings2, _source_key(settings2))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_a_new_source_key_falls_back_and_still_matches(self):
        """A different bake (flatfield toggle, new file) must not reuse another
        source's cached baseline."""
        img, strokes = _scene()
        proc = ImageProcessor()
        settings = _config(strokes[:2])
        proc._manual_bake(img, settings, _source_key(settings) + "|other")
        got, _ = proc._manual_bake(img, settings, _source_key(settings))
        want, _ = ImageProcessor()._manual_bake(img, settings, _source_key(settings))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


class TestIncrementalSkipsUnchangedWork:
    def test_appending_a_stroke_only_scores_the_new_one(self):
        """The regression this module exists to fix: adding stroke N+1 must not re-run
        detection for strokes 1..N."""
        img, strokes = _scene()
        proc = ImageProcessor()
        seen_counts: list[int] = []
        real = ip_mod.strokes_to_score

        def counting(image, s, spots):
            seen_counts.append(len(s))
            return real(image, s, spots)

        with patch.object(ip_mod, "strokes_to_score", side_effect=counting):
            for n in range(1, len(strokes) + 1):
                settings = _config(strokes[:n])
                proc._manual_bake(img, settings, _source_key(settings))

        assert seen_counts == [1, 1, 1], f"expected one new stroke scored per call, got {seen_counts}"

    def test_appending_a_stroke_only_fills_the_new_component(self):
        """apply_score_repair does the actual pixel fill per connected component; a
        previously-filled, untouched component must not be redone. repair_components
        looks the name up in its own defining module (retouch.logic), not image_processor's
        imported reference to it, so that is what has to be patched."""
        from negpy.features.retouch import logic as retouch_logic

        img, strokes = _scene()
        proc = ImageProcessor()
        call_counts: list[int] = []
        real = retouch_logic.apply_score_repair

        def counting(*a, **k):
            call_counts.append(1)
            return real(*a, **k)

        with patch.object(retouch_logic, "apply_score_repair", side_effect=counting):
            for n in range(1, len(strokes) + 1):
                call_counts.clear()
                settings = _config(strokes[:n])
                proc._manual_bake(img, settings, _source_key(settings))
                assert call_counts == [1], f"expected exactly one fill for the new component at {n} strokes, got {len(call_counts)}"
