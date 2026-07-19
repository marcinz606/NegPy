import pytest

from negpy.infrastructure.scanners.params import clamp_frame_offset_mm
from negpy.infrastructure.scanners.sane_backend import (
    _apply_frame_offset,
    _caps_from_options,
    _frame_extent_cap,
    _window_to_option_values,
)


class _Opt:
    def __init__(self, constraint):
        self.constraint = constraint


def _int_opts():
    return {
        "tl_x": _Opt((0, 1000, 1)),
        "br_x": _Opt((0, 1000, 1)),
        "tl_y": _Opt((0, 2000, 1)),
        "br_y": _Opt((0, 2000, 1)),
    }


def test_window_maps_fraction_to_integer_pixels():
    vals = _window_to_option_values(_int_opts(), (0.4, 0.1, 0.9, 0.7))
    assert vals == {"tl_x": 400, "br_x": 900, "tl_y": 200, "br_y": 1400}
    assert all(isinstance(v, int) for v in vals.values())


def test_window_full_frame_spans_extent():
    vals = _window_to_option_values(_int_opts(), (0.0, 0.0, 1.0, 1.0))
    assert vals == {"tl_x": 0, "br_x": 1000, "tl_y": 0, "br_y": 2000}


def test_window_keeps_float_for_fixed_options():
    opts = {"tl_x": _Opt((0.0, 36.0, 0.1)), "br_x": _Opt((0.0, 36.0, 0.1))}
    vals = _window_to_option_values(opts, (0.25, 0.0, 0.75, 1.0))
    assert vals["tl_x"] == 9.0 and isinstance(vals["tl_x"], float)
    assert vals["br_x"] == 27.0


def test_window_skips_absent_options():
    opts = {"br_x": _Opt((0, 1000, 1)), "tl_x": _Opt((0, 1000, 1))}
    vals = _window_to_option_values(opts, (0.2, 0.3, 0.8, 0.7))
    assert set(vals) == {"tl_x", "br_x"}


class _OffsetDev:
    def __init__(self, has_subframe=True):
        self.opt = {"subframe": _Opt((0.0, 37.83, 0.0))} if has_subframe else {}
        self.subframe = None


def test_frame_offset_sets_subframe_when_present():
    dev = _OffsetDev(has_subframe=True)
    _apply_frame_offset(dev, 3.5)
    assert dev.subframe == 3.5


def test_frame_offset_zero_is_written():
    # 0.0 must be applied, not skipped: on a held session handle a previous
    # frame's subframe would otherwise latch into the next scan.
    dev = _OffsetDev(has_subframe=True)
    _apply_frame_offset(dev, 0.0)
    assert dev.subframe == 0.0


def test_frame_offset_absent_option_skips():
    dev = _OffsetDev(has_subframe=False)
    _apply_frame_offset(dev, 3.5)
    assert dev.subframe is None


_SUBFRAME_OPTS = {"subframe": _Opt((0.0, 37.83, 0.0))}


def test_extent_cap_shortens_any_offset_scan():
    # offset + delivered ≈ one pitch on every frame; the overrun comes back black.
    assert _frame_extent_cap(_SUBFRAME_OPTS, 5.5) == pytest.approx(1.0 - 5.5 / 37.83)


def test_extent_cap_ignores_zero_offset():
    assert _frame_extent_cap(_SUBFRAME_OPTS, 0.0) is None


def test_extent_cap_needs_a_pitch_option():
    assert _frame_extent_cap({}, 5.5) is None


def test_frame_offset_set_failure_raises():
    class _FailDev:
        opt = {"subframe": _Opt((0.0, 37.83, 0.0))}

        @property
        def subframe(self):
            return 0.0

        @subframe.setter
        def subframe(self, v):
            raise ValueError("boom")

    with pytest.raises(RuntimeError):
        _apply_frame_offset(_FailDev(), 3.5)


def test_capabilities_report_the_feed_pitch():
    caps = _caps_from_options({"subframe": _Opt((0.0, 37.83, 0.0))})
    assert caps.frame_pitch_mm == pytest.approx(37.83)


def test_capabilities_report_no_pitch_without_subframe():
    assert _caps_from_options({}).frame_pitch_mm == 0.0


def test_offset_is_held_short_of_one_pitch():
    # At offset >= pitch the extent cap collapses the window to zero height and the
    # scan comes back empty; the clamp keeps a scannable sliver instead.
    assert clamp_frame_offset_mm(50.0, 37.83) == pytest.approx(36.83)
    assert _frame_extent_cap(_SUBFRAME_OPTS, clamp_frame_offset_mm(50.0, 37.83)) > 0


def test_offset_clamp_floors_at_zero_and_passes_unknown_pitch_through():
    assert clamp_frame_offset_mm(-2.0, 37.83) == 0.0
    assert clamp_frame_offset_mm(50.0, 0.0) == pytest.approx(50.0)
