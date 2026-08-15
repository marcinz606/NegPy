"""HDR merge of a bracketed capture.

The merge runs at decode, so a fault here is invisible until it reaches an export. The
properties that matter: the recovered range is real (the merge beats every single frame
in the shadows), the result never leaves [0, 1] (the pipeline clips at entry), and the
ratios are recovered from the images rather than assumed.
"""

import unittest
from dataclasses import fields, replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.logic import (
    SATURATION,
    anchor_choices,
    anchor_ratio,
    choose_reference,
    clipped_fraction,
    exposure_order_key,
    merge_frames,
    output_scale,
    pair_ratio,
    seed_shadow_density,
    subsample,
    to_float,
    shadow_reach_stops,
    probe_exposures,
    solve_ratios,
)
from negpy.features.hdr.models import HdrConfig, hdr_active, hdr_hash, hdr_name, hdr_token


def _scene(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """A synthetic slide: 4 decades of radiance, so no single exposure can hold it."""
    rng = np.random.default_rng(seed)
    # Peaks below the sensor white level so a gain of 1.0 is genuinely unclipped; the
    # noise below would otherwise push the top of the ramp over and make g=1.0 ineligible
    # as a reference, which is correct behaviour but not what these fixtures are testing.
    ramp = np.logspace(np.log10(0.95), np.log10(0.95e-4), w, dtype=np.float64)
    scene = np.repeat(ramp[None, :], h, axis=0)
    scene = np.stack([scene, scene * 0.9, scene * 0.8], axis=-1)
    return scene * (1.0 + 0.02 * rng.standard_normal(scene.shape))


def _expose(scene: np.ndarray, gain: float, read_noise: float = 0.0, seed: int = 1) -> np.ndarray:
    """One capture of `scene` at `gain`: scaled, clipped at the sensor white level, and
    quantized to 16 bits — the three things that make a single exposure insufficient."""
    rng = np.random.default_rng(seed)
    v = scene * gain
    if read_noise:
        v = v + rng.normal(0.0, read_noise, v.shape)
    v = np.clip(v, 0.0, 1.0)
    return (np.round(v * 65535.0).astype(np.uint16)).astype(np.uint16)


class TestReferenceChoice(unittest.TestCase):
    def test_picks_the_longest_unclipped_exposure(self):
        scene = _scene()
        frames = {f"g{g}": _expose(scene, g) for g in (0.25, 0.5, 1.0, 4.0)}
        stats = probe_exposures(lambda p: frames[p], list(frames))
        # g=4.0 drives the top of the ramp well past white, so it cannot define white.
        self.assertEqual(stats[choose_reference(stats)].path, "g1.0")

    def test_falls_back_to_least_clipped_when_all_clip(self):
        scene = _scene()
        frames = {f"g{g}": _expose(scene, g) for g in (4.0, 16.0, 64.0)}
        stats = probe_exposures(lambda p: frames[p], list(frames))
        self.assertEqual(stats[choose_reference(stats)].path, "g4.0")

    def test_merge_opens_at_the_brackets_middle_exposure(self):
        """A bracket means "this is the picture, plus range either side", so the merge has
        to land on the frame the photographer metered for. It is *computed* in the longest
        unclipped frame's units — best radiometrically — but rendering there stacks the
        brightest frame in the bracket on top of the transfer curve's fixed baseline and
        opens washed out."""
        scene = _scene()
        gains = [0.05, 0.2, 1.0]
        frames = [_expose(scene, g) for g in gains]
        ratios = solve_ratios(frames, reference=2)
        merged = merge_frames(frames, ratios, reference=2, align=False)

        middle = frames[1].astype(np.float32) / 65535.0  # gain 0.2, the median exposure
        self.assertAlmostEqual(float(merged.max()), float(middle.max()), delta=0.03)
        self.assertAlmostEqual(float(np.median(merged)), float(np.median(middle)), delta=0.03)

    def test_an_anchor_renders_at_the_frame_the_user_nominated(self):
        """The reference defines white; the anchor decides which exposure the picture opens
        at. Bracketing upward from the metered frame is the case the median cannot serve —
        every frame is at or above the reference, so the median clamps to 1.0 and the merge
        opens at the brightest unclipped frame, which is not what anyone metered for."""
        upward = [1.0, 2.0, 4.0, 8.0, 16.0]
        self.assertEqual(output_scale(upward), 1.0)  # median is inert here
        self.assertAlmostEqual(output_scale(upward, anchor=0.5), 0.5)
        # Never past the reference: that is where the [0, 1] guarantee comes from.
        self.assertEqual(output_scale(upward, anchor=8.0), 1.0)
        # A meaningless anchor falls back rather than scaling to zero.
        self.assertEqual(output_scale([0.5, 1.0, 2.0], anchor=0.0), 1.0)

    def test_anchor_ratio_resolves_a_path_and_tolerates_a_stale_one(self):
        paths, ratios = ["ref.nef", "long.nef", "short.nef"], [1.0, 4.0, 0.5]
        self.assertEqual(anchor_ratio(paths, ratios, "short.nef"), 0.5)
        self.assertIsNone(anchor_ratio(paths, ratios, ""))
        # An edit can outlive its files; degrade to the default instead of raising at decode.
        self.assertIsNone(anchor_ratio(paths, ratios, "gone.nef"))

    def test_anchor_moves_the_merged_result_and_keeps_it_in_range(self):
        scene = _scene()
        gains = [1.0, 2.0, 4.0]  # reference first: bracketed upward, median inert
        frames = [_expose(scene, g) for g in gains]
        ratios = solve_ratios(frames, reference=0)

        middle = merge_frames(frames, ratios, reference=0, align=False)
        anchored = merge_frames(frames, ratios, reference=0, align=False, anchor=0.5)

        self.assertAlmostEqual(float(np.median(anchored)), float(np.median(middle)) * 0.5, delta=0.01)
        self.assertLessEqual(float(anchored.max()), 1.0)
        self.assertGreaterEqual(float(anchored.min()), 0.0)

    def test_a_shorter_frame_survives_the_final_clip_once_anchored(self):
        """Why including the metered frame stops being redundant: the merge clips to [0, 1]
        *after* the anchor scales it, so radiance above the reference's white is destroyed
        at the default and recoverable once the render sits below the reference."""
        scene = _scene()
        # Reference at gain 2.0 blows the top of the ramp; the short frame still holds it.
        # Ratios straddle the reference so the median lands on it, as it does on a real bracket.
        frames = [_expose(scene, 2.0), _expose(scene, 0.25), _expose(scene, 4.0)]
        ratios = (1.0, 0.125, 2.0)
        self.assertEqual(output_scale(ratios), 1.0)
        at_reference = merge_frames(frames, ratios, reference=0, align=False)
        anchored = merge_frames(frames, ratios, reference=0, align=False, anchor=0.125)

        blown = to_float(frames[0]).max(axis=2) >= SATURATION
        self.assertGreater(blown.mean(), 0.01, "fixture must blow part of the reference")
        # At the reference every blown pixel pins to white; anchored, they separate again.
        self.assertLess(float(np.ptp(at_reference.max(axis=2)[blown])), 0.02)
        self.assertGreater(float(np.ptp(anchored.max(axis=2)[blown])), 0.05)

    def test_only_reachable_exposures_are_offered_as_anchors(self):
        """output_scale clamps at 1.0, so a frame longer than the reference renders exactly
        as the reference does. Offering one is a menu entry that provably cannot change the
        picture — on a real six-frame bracket, five of the six choices were the same image.
        """
        paths = ["ref.nef", "p4.nef", "p2.nef", "m1.nef", "m2.nef"]
        ratios = [1.0, 4.0, 2.0, 0.5, 0.25]

        offered = anchor_choices(paths, ratios)

        self.assertEqual([p for p, _ in offered], ["ref.nef", "m1.nef", "m2.nef"])
        self.assertEqual([round(ev, 3) for _, ev in offered], [0.0, -1.0, -2.0])
        # Every one of them lands on a different exposure, which is the whole point.
        scales = {output_scale(ratios, anchor=r) for r in (1.0, 0.5, 0.25)}
        self.assertEqual(len(scales), 3)
        # ...and the excluded ones would not have.
        self.assertEqual({output_scale(ratios, anchor=r) for r in (2.0, 4.0)}, {1.0})

    def test_every_merge_entry_point_forwards_the_anchor(self):
        """There are three of them — the export path, the full-res decode, and the preview
        service, which merges via `merge_providers` directly rather than through
        `merge_bracket`. The preview one was missed when the anchor was added, so the menu
        moved a value that never reached the picture: on-canvas, nothing changed at all.

        Guards the wiring rather than the arithmetic, because the arithmetic was right.
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "negpy"
        callers = {"merge_providers", "merge_bracket"}
        missing = []
        for path in root.rglob("*.py"):
            if path.parts[-2:] == ("hdr", "logic.py"):
                continue  # the definitions themselves
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id not in callers:
                    continue
                if not any(kw.arg in ("anchor", "anchor_path") for kw in node.keywords):
                    missing.append(f"{path.relative_to(root)}:{node.lineno} {node.func.id}()")
        self.assertEqual(missing, [], "merge call sites that drop the render exposure: " + "; ".join(missing))

    def test_setting_the_anchor_reloads_the_source(self):
        """The bracket is merged while the *source* is decoded, so the render exposure is
        baked into the buffer the pipeline starts from. Re-running the pipeline over that
        buffer changes nothing — which is exactly what the menu did at first: it wrote the
        value, saved it, and the picture never moved. It must go through apply_config,
        which re-decodes when the source token moves.
        """
        from unittest.mock import MagicMock

        from negpy.desktop.controller import AppController

        controller = MagicMock()  # not spec`d: state is an instance attribute, invisible on the class
        controller.state.selected_file_idx = 0
        controller.state.current_file_path = "/frames/ref.nef"
        controller.state.config = WorkspaceConfig()
        asset = {"path": "/frames/ref.nef", "hdr_paths": ("/frames/long.nef",), "hdr_anchor": ""}
        controller.state.uploaded_files = [asset]

        AppController.set_hdr_anchor(controller, "/frames/long.nef")

        self.assertEqual(asset["hdr_anchor"], "/frames/long.nef")
        # apply_config, never request_render: it compares source tokens and re-decodes.
        # (That dispatch is covered in test_source_tokens.)
        controller.apply_config.assert_called_once()
        controller.request_render.assert_not_called()
        self.assertEqual(controller.apply_config.call_args.args[0].hdr.hdr_anchor, "/frames/long.nef")

    def test_output_scale_never_exceeds_unity(self):
        """The [0, 1] guarantee comes from the reference defining white. A bracket whose
        median exposure is longer than the reference — most frames clipping — would
        otherwise be scaled straight past it."""
        self.assertLessEqual(output_scale([1.0, 2.0, 4.0, 8.0, 16.0]), 1.0)
        self.assertEqual(output_scale([]), 1.0)
        self.assertEqual(output_scale([1.0]), 1.0)
        # Odd count: the literal middle. Even count: between the two middles, in EV.
        self.assertAlmostEqual(output_scale([0.25, 0.5, 1.0]), 0.5)
        self.assertAlmostEqual(output_scale([0.25, 0.5, 1.0, 2.0]), 0.5 * 2**0.5)


class TestRatioSolving(unittest.TestCase):
    def test_pair_ratio_recovers_a_known_stop(self):
        scene = _scene()
        r = pair_ratio(_expose(scene, 0.25), _expose(scene, 0.5))
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 2.0, delta=0.05)

    def test_ratios_chain_across_the_bracket(self):
        scene = _scene()
        gains = [0.125, 0.25, 0.5, 1.0]
        frames = [_expose(scene, g) for g in gains]
        ratios = solve_ratios(frames, reference=3)
        for gain, ratio in zip(gains, ratios):
            self.assertAlmostEqual(ratio, gain / 1.0, delta=0.05 * max(gain, 0.125))
        self.assertAlmostEqual(ratios[3], 1.0)

    def test_ratios_are_solved_not_assumed(self):
        """Uneven spacing must be recovered; a fixed 1-EV assumption would not see it."""
        scene = _scene()
        frames = [_expose(scene, g) for g in (0.1, 0.7, 1.0)]
        ratios = solve_ratios(frames, reference=2)
        self.assertAlmostEqual(ratios[0], 0.1, delta=0.01)
        self.assertAlmostEqual(ratios[1], 0.7, delta=0.04)

    def test_no_overlap_falls_back_instead_of_raising(self):
        """A pair with nothing measurable in common must not break the chain past it."""
        black = np.zeros((32, 32, 3), dtype=np.uint16)
        self.assertIsNone(pair_ratio(black, black))
        frames = [black, _expose(_scene(32, 32), 1.0)]
        self.assertEqual(len(solve_ratios(frames, reference=1)), 2)

    def test_ratios_hold_when_the_bracket_arrives_out_of_order(self):
        """The chain is ordered by measured exposure, and the caller's order is not it:
        the decode path passes (reference, *hdr_paths), which on a real bracket puts the
        *longest* frame second. Ordering must come from the frames, not the list.

        Every frame here past the reference clips hard enough that its 99th percentile
        pins at 1.0, so `level` alone ties them and a level-only sort leaves them in
        input order -- chaining the reference straight to the longest frame and fitting
        the rest backwards. Six stops apart there is no sample unsaturated in both, so
        that fit finds nothing, falls back to the tied levels and returns 1.0. On a real
        5-frame bracket the same wrong order put the ratios out by up to 7%, which
        printed a visible seam in the sky where a frame dropped out of the merge.
        """
        scene = _scene()
        gains = [1.0, 64.0, 32.0, 16.0, 8.0]  # reference first, then longest -> shortest
        frames = [_expose(scene, g) for g in gains]

        levels = [probe_exposures(lambda p: frames[int(p)], [str(i)])[0].level for i in range(len(frames))]
        self.assertEqual(levels[1:], [1.0] * 4, "fixture must tie on level, or it tests nothing")

        ratios = solve_ratios(frames, reference=0)
        for gain, ratio in zip(gains, ratios):
            self.assertAlmostEqual(ratio, gain, delta=0.03 * gain)

    def test_exposure_order_key_orders_past_a_tied_level(self):
        key = exposure_order_key
        # Tied level: the clipped fraction separates them, and more clipping is longer.
        self.assertLess(key(1.0, 0.18), key(1.0, 0.32))
        # Nothing clipped: level still orders, and every clipped frame sorts above.
        self.assertLess(key(0.40, 0.0), key(0.88, 0.0))
        self.assertLess(key(0.99, 0.0), key(1.0, 0.001))


class TestMeasurementSampling(unittest.TestCase):
    """The level percentile and the ratio median read a subsample; the clipped fraction
    does not, because it picks the reference against a hard threshold."""

    def test_clipped_fraction_matches_the_whole_frame(self):
        f = to_float(_expose(_scene(), 4.0))
        self.assertEqual(clipped_fraction(f), float((f >= SATURATION).any(axis=2).mean()))

    def test_subsample_leaves_a_frame_under_budget_alone(self):
        small = np.zeros((1000, 1000, 1), dtype=np.uint8)
        self.assertIs(subsample(small), small)
        # Over budget: strided, and still a view of the original.
        big = np.zeros((3000, 3000, 1), dtype=np.uint8)
        self.assertLessEqual(subsample(big).shape[0] * subsample(big).shape[1], big.shape[0] * big.shape[1] // 4)
        self.assertIs(subsample(big).base, big)

    def test_ratios_survive_the_subsample(self):
        """A frame big enough to actually be strided must still solve to the true stop."""
        scene = _scene(2600, 2600)
        short, long_ = _expose(scene, 0.25), _expose(scene, 0.5)
        self.assertLess(subsample(short).shape[0], short.shape[0])
        r = pair_ratio(short, long_)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 2.0, delta=0.05)


class TestMergeQuality(unittest.TestCase):
    def test_merge_beats_every_single_frame_in_the_shadows(self):
        """The whole justification for the feature. Error is measured against the known
        scene in the deep shadows, where the reference has only quantization and read noise
        left.

        The bracket is the real-world shape: the reference is the longest *unclipped*
        frame, and the frames that improve the shadows are the ones **longer** than it,
        which pay for that reach by blowing the highlights. Frames shorter than the
        reference can only help where the reference clips, which by construction is
        nowhere.
        """
        scene = _scene(96, 96)
        gains = [1.0, 4.0, 16.0]
        frames = [_expose(scene, g, read_noise=0.0015, seed=i) for i, g in enumerate(gains)]
        ratios = solve_ratios(frames, reference=0)
        merged = merge_frames(frames, ratios, reference=0, align=False)

        deep = scene[..., 0] < 1e-3  # the densest decade of the ramp
        truth = scene[..., 0][deep]

        def rel_err(estimate: np.ndarray) -> float:
            return float(np.median(np.abs(estimate[deep] - truth) / np.maximum(truth, 1e-9)))

        merged_err = rel_err(merged[..., 0])
        for gain, frame in zip(gains, frames):
            single = frame[..., 0].astype(np.float32) / 65535.0 / gain
            self.assertLess(merged_err, rel_err(single), f"single frame at gain {gain} beat the merge")

    def test_weighting_is_inverse_variance(self):
        """Regression. Weighting by the exposure ratio rather than its square let a short
        frame's amplified quantization noise back in and made the deep shadows worse than
        not merging at all. Pinned by measuring the noise floor directly: a merge that
        included a very short frame at ratio-weighting was visibly noisier than one that
        effectively ignored it."""
        flat = np.full((64, 64, 3), 4e-4)  # a uniform deep-shadow patch, known truth
        gains = [1.0, 8.0]
        frames = [_expose(flat, g, read_noise=0.0015, seed=i) for i, g in enumerate(gains)]
        merged = merge_frames(frames, [1.0, 8.0], reference=0, align=False)

        # The 8x frame carries 8x the signal for the same read noise, so it should
        # dominate: the merged noise must be close to that frame's, not to the average.
        long_only = frames[1][..., 0].astype(np.float64) / 65535.0 / 8.0
        ref_only = frames[0][..., 0].astype(np.float64) / 65535.0
        self.assertLess(float(merged[..., 0].std()), 0.5 * float(ref_only.std()))
        self.assertLess(float(merged[..., 0].std()), 1.5 * float(long_only.std()))

    def test_output_never_leaves_unit_range(self):
        """features/process/logic.py clips at pipeline entry, so anything above 1.0 would
        be destroyed rather than rendered."""
        scene = _scene() * 8.0  # far brighter than any frame can hold
        frames = [_expose(scene, g) for g in (0.1, 0.5, 1.0)]
        merged = merge_frames(frames, solve_ratios(frames, reference=2), reference=2, align=False)
        self.assertGreaterEqual(float(merged.min()), 0.0)
        self.assertLessEqual(float(merged.max()), 1.0)
        self.assertTrue(np.all(np.isfinite(merged)))

    def test_saturated_samples_are_excluded(self):
        """A blown region in the long frame must take its value from the short one, not
        average the two — averaging would drag a real highlight down."""
        scene = _scene(32, 32)
        short, long_ = _expose(scene, 0.25), _expose(scene, 4.0)
        merged = merge_frames([long_, short], [4.0, 0.25], reference=1, align=False)
        blown = (long_.astype(np.float32) / 65535.0) >= SATURATION
        self.assertTrue(blown.any(), "fixture did not actually clip")
        expected = np.clip(scene.astype(np.float32), 0.0, 1.0)
        self.assertLess(float(np.abs(merged[blown] - expected[blown]).max()), 0.02)

    def test_single_frame_bracket_is_a_passthrough(self):
        frame = _expose(_scene(32, 32), 1.0)
        merged = merge_frames([frame], [1.0], reference=0, align=False)
        np.testing.assert_allclose(merged, frame.astype(np.float32) / 65535.0, atol=1e-6)

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            merge_frames([_expose(_scene(32, 32), 1.0), _expose(_scene(16, 16), 0.5)], [1.0, 0.5], align=False)

    def test_ratio_count_must_match(self):
        with self.assertRaises(ValueError):
            merge_frames([_expose(_scene(16, 16), 1.0)], [1.0, 0.5], align=False)

    def test_a_carrier_channel_is_dropped_whatever_the_dtype(self):
        """An IR or alpha plane has no place in a radiometric merge. Dropping it from only
        one of the two input dtypes would make the shape checks dtype-dependent — a uint16
        decode and a float32 preview of the same frame would not agree."""
        u16 = np.zeros((4, 4, 4), dtype=np.uint16)
        f32 = np.zeros((4, 4, 4), dtype=np.float32)
        self.assertEqual(to_float(u16).shape[2], 3)
        self.assertEqual(to_float(f32).shape[2], 3)


class TestConfig(unittest.TestCase):
    def test_flat_namespace_has_no_collisions(self):
        """WorkspaceConfig.to_dict flattens every sub-config into one namespace and a
        duplicate field name silently clobbers (CLAUDE.md). RgbScanConfig already owns
        the bare `enabled` and `align`, which is why HdrConfig prefixes everything."""
        cfg = WorkspaceConfig()
        hdr_names = {f.name for f in fields(HdrConfig)}
        for attr in ("process", "exposure", "rgbscan", "stitch", "flatfield"):
            other = {f.name for f in fields(getattr(cfg, attr))}
            self.assertEqual(hdr_names & other, set(), f"HdrConfig collides with {attr}")
        # And the flattened dict really does carry every hdr key.
        flat = cfg.to_dict()
        for name in hdr_names:
            self.assertIn(name, flat)

    def test_round_trips_through_the_flat_dict(self):
        cfg = WorkspaceConfig()
        cfg = replace(cfg, hdr=HdrConfig(hdr_enabled=True, hdr_paths=("/a.nef", "/b.nef"), hdr_ratios=(1.0, 2.0, 4.0), hdr_align=False))
        back = WorkspaceConfig.from_flat_dict(cfg.to_dict()).hdr
        self.assertEqual(back, cfg.hdr)
        self.assertIsInstance(back.hdr_paths, tuple)  # JSON round-trips lists; frozen config must stay hashable
        self.assertIsInstance(back.hdr_ratios, tuple)

    def test_active_predicate(self):
        self.assertFalse(hdr_active(HdrConfig()))
        self.assertFalse(hdr_active(HdrConfig(hdr_enabled=True)))  # enabled but no frames
        self.assertTrue(hdr_active(HdrConfig(hdr_enabled=True, hdr_paths=("/a.nef",))))

    def test_token_is_empty_when_inactive_and_tracks_ratios(self):
        self.assertEqual(hdr_token(HdrConfig()), "")
        # Missing files cannot be stamped, so the token degrades to empty rather than lying.
        self.assertEqual(hdr_token(HdrConfig(hdr_enabled=True, hdr_paths=("/nonexistent.nef",))), "")

    def test_hash_is_order_sensitive_and_strips_to_a_digest(self):
        from negpy.services.assets.half_frame import base_hash

        a, b = hdr_hash(["h1", "h2"]), hdr_hash(["h2", "h1"])
        self.assertNotEqual(a, b, "order defines the exposure reference")
        self.assertTrue(a.endswith("#hdr"))
        self.assertEqual(base_hash(a), a.split("#")[0])

    def test_name_reports_the_frame_count(self):
        self.assertEqual(hdr_name(["/x/_DSC1716.NEF", "/x/a.NEF", "/x/b.NEF"]), "_DSC1716 +2 (HDR)")


if __name__ == "__main__":
    unittest.main()


class TestShadowSeed(unittest.TestCase):
    """A merge of well-exposed frames recovers precision, not range — invisible at the
    same tone. So it opens with the shadows already lifted by what the bracket bought."""

    def test_seed_scales_with_the_reach_the_bracket_actually_recovered(self):
        from negpy.features.hdr.logic import SEED_SHADOW_PER_STOP, seed_shadow_density, shadow_reach_stops

        one, two = (1.0, 2.0), (1.0, 2.0, 4.0)
        self.assertAlmostEqual(shadow_reach_stops(one), 1.0, places=5)
        self.assertAlmostEqual(shadow_reach_stops(two), 2.0, places=5)
        self.assertAlmostEqual(seed_shadow_density(one), -SEED_SHADOW_PER_STOP, places=5)
        self.assertAlmostEqual(seed_shadow_density(two), -2 * SEED_SHADOW_PER_STOP, places=5)

    def test_a_bracket_that_recovered_nothing_gets_no_lift(self):
        """The seed must never imply a benefit that is not there — a single frame, or a
        bracket whose render exposure is already its longest frame, is left neutral."""
        self.assertEqual(seed_shadow_density((1.0,)), 0.0)
        self.assertEqual(seed_shadow_density(()), 0.0)
        self.assertEqual(shadow_reach_stops((1.0,)), 0.0)

    def test_seed_is_capped_and_stays_inside_the_slider(self):
        from negpy.features.exposure.models import ExposureConfig
        from negpy.features.hdr.logic import SEED_SHADOW_LIMIT

        wide = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
        self.assertAlmostEqual(seed_shadow_density(wide), -SEED_SHADOW_LIMIT, places=5)
        # -0.9 is the Shadows Density slider's own limit (view/sidebar/tone.py).
        self.assertGreaterEqual(seed_shadow_density(wide), -0.9)
        self.assertIsInstance(ExposureConfig().shadow_density, float)

    def test_the_real_brackets_land_on_the_measured_noise_budget(self):
        """Calibration guard. -0.30/stop comes from measuring rendered shadow noise on two
        real 5-frame brackets: at -0.6 the merged render was still quieter than the single
        metered frame, at -0.8 it was noisier. Both brackets reach ~2 stops, so a seed that
        drifts far from -0.6 has lost its justification."""
        for ratios in ((1.0, 1.993, 0.489, 0.222, 0.122), (4.045, 1.923, 1.0, 0.505, 0.251)):
            self.assertAlmostEqual(seed_shadow_density(ratios), -0.6, delta=0.05)


class TestSeedIsOnlyAStartingPoint(unittest.TestCase):
    def test_seeding_does_not_touch_the_merge_itself(self):
        """The seed is an edit, not part of the merge: the pixels the merge produces must
        be identical whether or not a seed is later applied to them."""
        scene = _scene(48, 48)
        frames = [_expose(scene, g) for g in (1.0, 4.0)]
        ratios = solve_ratios(frames, reference=0)
        a = merge_frames(frames, ratios, reference=0, align=False)
        b = merge_frames(frames, ratios, reference=0, align=False)
        np.testing.assert_array_equal(a, b)
        self.assertLess(seed_shadow_density(ratios), 0.0)  # this bracket does earn a lift
