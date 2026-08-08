import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import smooth_polyline
from negpy.features.local.logic import compute_local_maps
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskShape


def _center_square_mask(stops: float, feather: float = 0.0) -> LocalMask:
    """Polygon covering the central 50% of the frame."""
    return LocalMask(
        vertices=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
        stops=stops,
        feather=feather,
    )


def _ev(cfg: LocalAdjustmentsConfig, h: int = 100, w: int = 100) -> np.ndarray:
    return compute_local_maps(cfg, h, w, (h, w))[:, :, 0]


def _grades(cfg: LocalAdjustmentsConfig, h: int = 100, w: int = 100) -> np.ndarray:
    return compute_local_maps(cfg, h, w, (h, w))[:, :, 1]


class TestComputeEvMap(unittest.TestCase):
    """Plane 0 is the shared CPU/GPU primitive — per-pixel print exposure in stops,
    positive = burn."""

    def test_all_zeros_when_empty(self) -> None:
        ev = _ev(LocalAdjustmentsConfig())
        np.testing.assert_array_equal(ev, np.zeros((100, 100), dtype=np.float32))

    def test_interior_equals_the_masks_stops(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        ev = _ev(cfg)
        self.assertAlmostEqual(float(ev[50, 50]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[5, 5]), 0.0, places=5)

    def test_a_dodge_is_negative(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(-1.5),))
        ev = _ev(cfg)
        self.assertAlmostEqual(float(ev[50, 50]), -1.5, places=5)

    def test_overlapping_masks_are_additive(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(0.5), _center_square_mask(0.75)))
        ev = _ev(cfg)
        self.assertAlmostEqual(float(ev[50, 50]), 1.25, places=5)

    def test_degenerate_mask_skipped(self) -> None:
        """A mask with fewer than 3 vertices is ignored."""
        cfg = LocalAdjustmentsConfig(masks=(LocalMask(vertices=((0.4, 0.4), (0.6, 0.6)), stops=-1.0),))
        ev = _ev(cfg)
        np.testing.assert_array_equal(ev, np.zeros((100, 100), dtype=np.float32))

    def test_feathered_mask_stays_within_its_stops(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0, feather=0.05),))
        ev = _ev(cfg)
        self.assertEqual(ev.shape, (100, 100))
        self.assertGreaterEqual(float(ev.min()), 0.0)
        self.assertLessEqual(float(ev.max()), 1.0 + 1e-5)
        self.assertGreater(float(ev[50, 50]), 0.9)


class TestGradePlane(unittest.TestCase):
    """Plane 1 carries the local grade in ISO-R points — same rasterisation as the EV plane."""

    def test_zero_without_local_grade(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        np.testing.assert_array_equal(_grades(cfg), np.zeros((100, 100), dtype=np.float32))

    def test_interior_equals_delta_and_exterior_is_clean(self) -> None:
        cfg = LocalAdjustmentsConfig(masks=(LocalMask(vertices=_center_square_mask(0.0).vertices, stops=0.0, grade=-30.0),))
        grades = _grades(cfg)
        self.assertAlmostEqual(float(grades[50, 50]), -30.0, places=4)
        self.assertAlmostEqual(float(grades[5, 5]), 0.0, places=5)

    def test_overlapping_masks_are_additive(self) -> None:
        mask = _center_square_mask(0.0)
        cfg = LocalAdjustmentsConfig(
            masks=(
                LocalMask(vertices=mask.vertices, stops=0.0, grade=-10.0),
                LocalMask(vertices=mask.vertices, stops=0.0, grade=-15.0),
            )
        )
        self.assertAlmostEqual(float(_grades(cfg)[50, 50]), -25.0, places=4)

    def test_exposure_and_grade_ride_the_same_alpha(self) -> None:
        """One mask, both values: the feathered edge must weight them identically."""
        cfg = LocalAdjustmentsConfig(masks=(LocalMask(vertices=_center_square_mask(0.0).vertices, stops=1.0, feather=0.05, grade=-20.0),))
        maps = compute_local_maps(cfg, 100, 100, (100, 100))
        alpha_ev = maps[:, :, 0] / 1.0
        alpha_grade = maps[:, :, 1] / -20.0
        np.testing.assert_allclose(alpha_ev, alpha_grade, atol=1e-6)


class TestMaskShapes(unittest.TestCase):
    """Oval and card-edge masks feed the same two planes as a polygon."""

    def test_an_oval_fills_its_axes_and_not_the_bounding_corners(self) -> None:
        # Frame centre, both radii 0.25. The corner of that box is outside the oval.
        oval = LocalMask(vertices=((0.5, 0.5), (0.75, 0.5), (0.5, 0.75)), stops=1.0, feather=0.0, shape=MaskShape.OVAL)
        ev = _ev(LocalAdjustmentsConfig(masks=(oval,)))
        self.assertAlmostEqual(float(ev[50, 50]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[50, 72]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[72, 72]), 0.0, places=5)

    def test_an_ovals_axes_need_not_be_perpendicular(self) -> None:
        """The control points are an affine frame, so a tilted oval is a sheared one."""
        tilted = LocalMask(vertices=((0.5, 0.5), (0.75, 0.6), (0.4, 0.75)), stops=1.0, feather=0.0, shape=MaskShape.OVAL)
        ev = _ev(LocalAdjustmentsConfig(masks=(tilted,)))
        self.assertAlmostEqual(float(ev[50, 50]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[5, 5]), 0.0, places=5)

    def test_a_card_edge_ramps_from_full_to_nothing(self) -> None:
        grad = LocalMask(vertices=((0.25, 0.5), (0.75, 0.5)), stops=1.0, shape=MaskShape.GRADIENT)
        ev = _ev(LocalAdjustmentsConfig(masks=(grad,)))
        self.assertAlmostEqual(float(ev[50, 10]), 1.0, places=5)  # behind the full edge
        self.assertAlmostEqual(float(ev[50, 90]), 0.0, places=5)  # past the fade-out
        self.assertAlmostEqual(float(ev[50, 50]), 0.5, places=2)
        # The ramp decreases across the axis and stays constant along it.
        row = ev[50, 25:75]
        self.assertTrue(np.all(np.diff(row) <= 1e-6))
        np.testing.assert_allclose(ev[10, :], ev[90, :], atol=1e-6)

    def test_a_tilted_card_edge_can_burn_a_full_corner(self) -> None:
        """A tilted line through a point in the frame always cuts a corner off the full
        side. The start must go outside the picture to hold the full top edge."""
        axis = (0.2, 0.4)

        def top_corners(start):
            grad = LocalMask(
                vertices=(start, (start[0] + axis[0], start[1] + axis[1])),
                stops=1.0,
                shape=MaskShape.GRADIENT,
            )
            ev = _ev(LocalAdjustmentsConfig(masks=(grad,)))
            return float(ev[0, 0]), float(ev[0, 99])

        inside_left, inside_right = top_corners((0.5, 0.0))
        self.assertAlmostEqual(inside_left, 1.0, places=5)
        self.assertLess(inside_right, 1.0)  # the corner the tilt cuts off

        outside_left, outside_right = top_corners((1.1, 0.0))
        self.assertAlmostEqual(outside_left, 1.0, places=5)
        self.assertAlmostEqual(outside_right, 1.0, places=5)

    def test_a_card_edge_needs_only_two_points(self) -> None:
        grad = LocalMask(vertices=((0.25, 0.5), (0.75, 0.5)), stops=1.0, shape=MaskShape.GRADIENT)
        self.assertGreater(float(_ev(LocalAdjustmentsConfig(masks=(grad,))).max()), 0.9)

    def test_a_degenerate_card_edge_is_skipped(self) -> None:
        grad = LocalMask(vertices=((0.5, 0.5), (0.5, 0.5)), stops=1.0, shape=MaskShape.GRADIENT)
        ev = _ev(LocalAdjustmentsConfig(masks=(grad,)))
        np.testing.assert_array_equal(ev, np.zeros((100, 100), dtype=np.float32))

    def test_invert_swaps_inside_for_outside(self) -> None:
        plain = _ev(LocalAdjustmentsConfig(masks=(_center_square_mask(1.0, feather=0.05),)))
        inverted = _ev(LocalAdjustmentsConfig(masks=(replace(_center_square_mask(1.0, feather=0.05), invert=True),)))
        np.testing.assert_allclose(plain + inverted, np.ones((100, 100), dtype=np.float32), atol=1e-6)

    def test_feather_does_not_touch_a_card_edge(self) -> None:
        soft = LocalMask(vertices=((0.25, 0.5), (0.75, 0.5)), stops=1.0, feather=0.15, shape=MaskShape.GRADIENT)
        hard = replace(soft, feather=0.0)
        np.testing.assert_allclose(_ev(LocalAdjustmentsConfig(masks=(soft,))), _ev(LocalAdjustmentsConfig(masks=(hard,))), atol=0.0)


class TestSmoothPolyline(unittest.TestCase):
    """Mask outlines and heal paths are always drawn as a Catmull-Rom curve."""

    def test_short_input_returned_unchanged(self) -> None:
        self.assertEqual(smooth_polyline([(0.0, 0.0), (1.0, 1.0)]), [(0.0, 0.0), (1.0, 1.0)])

    def test_closed_interpolates_control_points(self) -> None:
        sq = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
        out = smooth_polyline(sq, closed=True, samples_per_seg=8)
        self.assertEqual(len(out), 4 * 8)  # denser than the 4 control points
        for i, p in enumerate(sq):  # each control point is the t=0 sample of its segment
            self.assertAlmostEqual(out[i * 8][0], p[0])
            self.assertAlmostEqual(out[i * 8][1], p[1])

    def test_open_keeps_endpoints(self) -> None:
        line = [(0.1, 0.1), (0.5, 0.2), (0.9, 0.1)]
        out = smooth_polyline(line, closed=False, samples_per_seg=8)
        self.assertEqual(out[0], (0.1, 0.1))
        self.assertEqual(out[-1], (0.9, 0.1))
        self.assertGreater(len(out), len(line))

    def test_smoothed_square_mask_still_fills_interior(self) -> None:
        # Smoothing is unconditional in compute_local_maps; the interior/exterior
        # invariants the pipeline relies on must survive it.
        cfg = LocalAdjustmentsConfig(masks=(_center_square_mask(1.0),))
        ev = _ev(cfg)
        self.assertAlmostEqual(float(ev[50, 50]), 1.0, places=5)
        self.assertAlmostEqual(float(ev[5, 5]), 0.0, places=5)


class TestLocalSerialization(unittest.TestCase):
    def test_roundtrip_preserves_masks(self) -> None:
        """to_dict -> from_flat_dict preserves polygon mask fields."""
        mask = LocalMask(
            vertices=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
            stops=-0.4,
            feather=0.03,
            grade=-15.0,
        )
        cfg = WorkspaceConfig(local=LocalAdjustmentsConfig(masks=(mask,)))

        restored = WorkspaceConfig.from_flat_dict(cfg.to_dict())

        self.assertEqual(len(restored.local.masks), 1)
        out = restored.local.masks[0]
        self.assertEqual(tuple(out.vertices), mask.vertices)
        self.assertAlmostEqual(out.stops, -0.4)
        self.assertAlmostEqual(out.feather, 0.03)
        self.assertAlmostEqual(out.grade, -15.0)

    def test_legacy_mask_migrates_brightness_signed_strength_to_stops(self) -> None:
        """Pre-flip saves stored `strength` (positive = dodge); the same mask must come
        back as exposure-signed stops, at the frame's grade."""
        legacy = {"local_masks": {"masks": [{"vertices": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "strength": 0.4, "feather": 0.03}]}}
        mask = WorkspaceConfig.from_flat_dict(legacy).local.masks[0]
        self.assertAlmostEqual(mask.stops, -0.4)
        self.assertEqual(mask.grade, 0.0)

    def test_roundtrip_preserves_shape_and_invert(self) -> None:
        cfg = WorkspaceConfig(
            local=LocalAdjustmentsConfig(
                masks=(
                    LocalMask(vertices=((0.2, 0.5), (0.8, 0.5)), stops=1.0, shape=MaskShape.GRADIENT),
                    LocalMask(vertices=((0.5, 0.5), (0.7, 0.5), (0.5, 0.7)), shape=MaskShape.OVAL, invert=True),
                )
            )
        )
        restored = WorkspaceConfig.from_flat_dict(cfg.to_dict()).local.masks
        self.assertEqual(restored[0].shape, MaskShape.GRADIENT)
        self.assertEqual(restored[1].shape, MaskShape.OVAL)
        self.assertTrue(restored[1].invert)
        self.assertFalse(restored[0].invert)

    def test_a_mask_saved_before_shapes_loads_as_a_polygon(self) -> None:
        legacy = {"local_masks": {"masks": [{"vertices": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "stops": 0.5}]}}
        mask = WorkspaceConfig.from_flat_dict(legacy).local.masks[0]
        self.assertEqual(mask.shape, MaskShape.POLYGON)
        self.assertFalse(mask.invert)

    def test_a_legacy_burn_migrates_to_a_positive_burn(self) -> None:
        legacy = {"local_masks": {"masks": [{"vertices": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "strength": -1.0}]}}
        self.assertAlmostEqual(WorkspaceConfig.from_flat_dict(legacy).local.masks[0].stops, 1.0)


if __name__ == "__main__":
    unittest.main()
