from negpy.desktop.view.widgets.scan_window_geometry import (
    hit_corner,
    normalize_rect,
    resize_corner,
)


def test_normalize_orders_reversed_corners():
    assert normalize_rect((0.8, 0.9, 0.2, 0.1)) == (0.2, 0.1, 0.8, 0.9)


def test_normalize_clamps_out_of_range():
    assert normalize_rect((-0.5, -0.2, 1.5, 1.2)) == (0.0, 0.0, 1.0, 1.0)


def test_normalize_grows_below_min_size():
    x1, y1, x2, y2 = normalize_rect((0.5, 0.5, 0.505, 0.505), min_size=0.02)
    assert round(x2 - x1, 6) == 0.02
    assert round(y2 - y1, 6) == 0.02


def test_hit_corner_identifies_each_corner():
    rect = (0.2, 0.3, 0.8, 0.7)
    assert hit_corner(rect, 0.2, 0.3, tol=0.05) == 0  # TL
    assert hit_corner(rect, 0.8, 0.3, tol=0.05) == 1  # TR
    assert hit_corner(rect, 0.8, 0.7, tol=0.05) == 2  # BR
    assert hit_corner(rect, 0.2, 0.7, tol=0.05) == 3  # BL


def test_hit_corner_returns_none_when_far():
    assert hit_corner((0.2, 0.3, 0.8, 0.7), 0.5, 0.5, tol=0.05) is None


def test_resize_corner_moves_only_that_corner():
    rect = (0.2, 0.3, 0.8, 0.7)
    assert resize_corner(rect, 2, 0.9, 0.95) == (0.2, 0.3, 0.9, 0.95)  # BR
    assert resize_corner(rect, 0, 0.1, 0.15) == (0.1, 0.15, 0.8, 0.7)  # TL


def test_resize_corner_clamps():
    assert resize_corner((0.2, 0.3, 0.8, 0.7), 2, 1.5, 1.5) == (0.2, 0.3, 1.0, 1.0)
