"""Which masks a gesture on the selected one mutes: the ones whose tint sits on the
area being judged."""

from negpy.features.local.logic import overlapping_masks
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskShape

# Left square, one straddling its corner, and one clear of it on the right.
LEFT = ((0.10, 0.10), (0.45, 0.10), (0.45, 0.45), (0.10, 0.45))
STRADDLE = ((0.35, 0.35), (0.75, 0.35), (0.75, 0.75), (0.35, 0.75))
CLEAR = ((0.75, 0.05), (0.95, 0.05), (0.95, 0.25), (0.75, 0.25))
CARD_EDGE = ((0.20, 0.50), (0.80, 0.50))


def _config(*masks: LocalMask) -> LocalAdjustmentsConfig:
    return LocalAdjustmentsConfig(masks=masks)


def test_only_the_intersecting_mask_is_named():
    conf = _config(LocalMask(vertices=LEFT), LocalMask(vertices=STRADDLE), LocalMask(vertices=CLEAR))

    assert overlapping_masks(conf, 0) == frozenset({1})
    assert overlapping_masks(conf, 1) == frozenset({0})  # symmetric
    assert overlapping_masks(conf, 2) == frozenset()  # clear of both


def test_the_tint_reaches_as_far_as_the_feather_does():
    """The tinted area is the soft edge, not the outline, so a mask just clear of a
    neighbour's shape still sits under its feather."""

    def gap(distance: float) -> frozenset:
        near = tuple((x + 0.35 + distance, y) for x, y in LEFT)
        return overlapping_masks(_config(LocalMask(vertices=LEFT), LocalMask(vertices=near)), 0)

    assert gap(0.05) == frozenset({1})
    assert gap(0.20) == frozenset()


def test_a_card_edge_covers_its_whole_side_of_the_frame():
    """A gradient has no boundary, so everything on its exposed side intersects it."""
    conf = _config(
        LocalMask(vertices=CARD_EDGE, shape=MaskShape.GRADIENT),
        LocalMask(vertices=LEFT, feather=0.0),
        LocalMask(vertices=CLEAR, feather=0.0),
    )

    assert overlapping_masks(conf, 0) == frozenset({1})


def test_an_inverted_mask_intersects_its_surround():
    conf = _config(LocalMask(vertices=LEFT, feather=0.0, invert=True), LocalMask(vertices=CLEAR, feather=0.0))

    assert overlapping_masks(conf, 0) == frozenset({1})


def test_an_unfinished_or_missing_mask_names_nothing():
    conf = _config(LocalMask(vertices=LEFT[:2]), LocalMask(vertices=LEFT))

    assert overlapping_masks(conf, 0) == frozenset()  # too few points to have an area
    assert overlapping_masks(conf, 1) == frozenset()  # ...and so is invisible to its neighbour
    assert overlapping_masks(conf, 7) == frozenset()  # no such mask
