from negpy.infrastructure.scanners.params import scan_window_to_area


def test_none_rect_returns_none():
    assert scan_window_to_area(None, (36.0, 25.0)) is None


def test_full_frame_maps_to_full_extent():
    assert scan_window_to_area((0.0, 0.0, 1.0, 1.0), (36.0, 25.0)) == (0.0, 0.0, 36.0, 25.0)


def test_offset_subrect_scales_by_extent():
    rect = (0.25, 0.0, 0.75, 1.0)
    assert scan_window_to_area(rect, (36.0, 24.0)) == (9.0, 0.0, 27.0, 24.0)


def test_non_default_extent():
    rect = (0.0, 0.5, 0.5, 1.0)
    assert scan_window_to_area(rect, (40.0, 30.0)) == (0.0, 15.0, 20.0, 30.0)


def test_malformed_rect_returns_none():
    assert scan_window_to_area((0.1, 0.2), (36.0, 25.0)) is None
