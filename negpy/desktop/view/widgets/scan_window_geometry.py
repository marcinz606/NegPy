"""Pure rect math for the scan-window picker — no Qt, unit-testable.

Rects are normalized ``(x1, y1, x2, y2)`` in 0..1 (left, top, right, bottom),
matching the crop convention (``GeometryConfig.manual_crop_rect``).
"""

Rect = tuple[float, float, float, float]

MIN_SIZE = 0.02  # smallest window edge as a fraction of the frame


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def normalize_rect(rect: Rect, min_size: float = MIN_SIZE) -> Rect:
    """Order corners, clamp to 0..1, and enforce a minimum edge length."""
    x1, y1, x2, y2 = rect
    lo_x, hi_x = sorted((_clamp01(x1), _clamp01(x2)))
    lo_y, hi_y = sorted((_clamp01(y1), _clamp01(y2)))
    if hi_x - lo_x < min_size:
        hi_x = min(1.0, lo_x + min_size)
        lo_x = max(0.0, hi_x - min_size)
    if hi_y - lo_y < min_size:
        hi_y = min(1.0, lo_y + min_size)
        lo_y = max(0.0, hi_y - min_size)
    return (lo_x, lo_y, hi_x, hi_y)


def hit_corner(rect: Rect, fx: float, fy: float, tol: float) -> int | None:
    """Index of the corner (0=TL, 1=TR, 2=BR, 3=BL) within `tol` of (fx, fy), else None."""
    x1, y1, x2, y2 = rect
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    best: int | None = None
    best_d = tol * tol
    for i, (cx, cy) in enumerate(corners):
        d = (cx - fx) ** 2 + (cy - fy) ** 2
        if d <= best_d:
            best, best_d = i, d
    return best


def resize_corner(rect: Rect, corner: int, fx: float, fy: float) -> Rect:
    """Move `corner` to (fx, fy), clamped to 0..1. Result is un-ordered — normalize on release."""
    x1, y1, x2, y2 = rect
    fx, fy = _clamp01(fx), _clamp01(fy)
    if corner == 0:
        x1, y1 = fx, fy
    elif corner == 1:
        x2, y1 = fx, fy
    elif corner == 2:
        x2, y2 = fx, fy
    elif corner == 3:
        x1, y2 = fx, fy
    return (x1, y1, x2, y2)
