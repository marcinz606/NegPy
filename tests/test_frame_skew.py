import cv2
import numpy as np
import pytest

from negpy.features.geometry.logic import apply_fine_rotation, apply_keystone, keystone_matrix
from negpy.features.geometry.skew import measure_frame_skew, trusted_frame_skew


def _strip(h: int = 600, w: int = 900, frame: tuple[float, float] = (0.62, 0.55), gap: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(3)
    img = np.ones((h, w), np.float32)
    top, bottom = int(0.2 * h), int(0.8 * h)
    img[top:bottom] = 0.78
    fw, fh = frame[0] * w, frame[1] * (bottom - top)
    fy1 = int((top + bottom - fh) / 2)
    for center in (w / 2 - fw - gap * w, w / 2, w / 2 + fw + gap * w):
        x1, x2 = int(center - fw / 2), int(center + fw / 2)
        texture = 0.18 + 0.12 * rng.random((int(fh), max(0, min(w, x2) - max(0, x1))))
        img[fy1 : fy1 + int(fh), max(0, x1) : min(w, x2)] = cv2.GaussianBlur(texture.astype(np.float32), (0, 0), 3)
    return np.repeat(img[:, :, None], 3, axis=2)


def _scan(straight: np.ndarray, angle: float = 0.0, converge_v: float = 0.0, converge_h: float = 0.0) -> np.ndarray:
    h, w = straight.shape[:2]
    img = straight
    if converge_v or converge_h:
        inverse = np.linalg.inv(keystone_matrix(converge_v, converge_h, w, h))
        img = cv2.warpPerspective(img, inverse, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return apply_fine_rotation(np.ascontiguousarray(img), -angle) if angle else img


@pytest.mark.parametrize("angle", [-3.0, -0.4, 0.0, 0.25, 1.5, 6.0])
def test_rotation_round_trip(angle: float) -> None:
    skew = measure_frame_skew(_scan(_strip(), angle))

    assert skew is not None
    assert skew.fine_rotation == pytest.approx(angle, abs=0.03)


def test_the_fitted_rotation_squares_the_frame_through_the_app_warp() -> None:
    scan = _scan(_strip(), 2.0)
    skew = measure_frame_skew(scan)
    assert skew is not None

    again = measure_frame_skew(apply_fine_rotation(scan, skew.fine_rotation))

    assert again is not None
    assert again.fine_rotation == pytest.approx(0.0, abs=0.03)


def test_a_portrait_frame_whose_strip_runs_off_the_canvas() -> None:
    straight = _strip(h=900, w=600, frame=(0.5, 0.55), gap=0.05)

    skew = measure_frame_skew(_scan(straight, 0.5))

    assert skew is not None
    assert skew.fine_rotation == pytest.approx(0.5, abs=0.03)


def _masked_strip(h: int = 600, w: int = 900) -> np.ndarray:
    """Film base to the top and left canvas edges, the next frame running off the right."""
    rng = np.random.default_rng(7)
    img = np.full((h, w), 0.78, np.float32)
    y1, y2 = int(0.1 * h), int(0.88 * h)
    for x1, x2 in ((int(0.08 * w), int(0.72 * w)), (int(0.745 * w), w)):
        texture = 0.18 + 0.12 * rng.random((y2 - y1, x2 - x1))
        img[y1:y2, x1:x2] = cv2.GaussianBlur(texture.astype(np.float32), (0, 0), 3)
    return np.repeat(img[:, :, None], 3, axis=2)


@pytest.mark.parametrize("angle", [-1.1, 0.8])
def test_a_holder_edge_beside_the_picture_is_not_a_film_edge(angle: float, monkeypatch: pytest.MonkeyPatch) -> None:
    # The holder band masks the film edges and is square to the camera, so the film detector boxes
    # the holder and the next frame; the gate refiner still boxes the picture.
    from dataclasses import replace

    import negpy.features.geometry.skew as skew_module

    scan = _scan(_masked_strip(), angle)
    h, w = scan.shape[:2]
    scan[int(0.92 * h) :] = 0.01
    real = skew_module.detect_film_bounds_with_confidence

    def clamped(image: np.ndarray):
        band = int(np.argmax(image[:, :, 0].mean(axis=1) < 0.1))
        return replace(real(image), roi=(0, band, 0, image.shape[1]))

    def picture(image: np.ndarray, _roi):
        lum = image[:, :, 0]
        inside = (lum > 0.1) & (lum < 0.5)
        inside[:, int(0.73 * w) :] = False
        ys, xs = np.nonzero(inside)
        return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1), None, None

    monkeypatch.setattr(skew_module, "detect_film_bounds_with_confidence", clamped)
    monkeypatch.setattr(skew_module, "_refine_roi_to_image", picture)

    skew = trusted_frame_skew(scan)

    assert skew is not None
    assert skew.fine_rotation == pytest.approx(angle, abs=0.05)
    rows = sorted(0.5 * (line.p1[1] + line.p2[1]) for line in skew.lines if line.horizontal)
    cols = sorted(0.5 * (line.p1[0] + line.p2[0]) for line in skew.lines if not line.horizontal)
    assert rows == pytest.approx([0.1 * h, 0.88 * h], abs=5)
    assert cols == pytest.approx([0.08 * w, 0.72 * w], abs=5)


def _slide(h: int = 600, w: int = 900) -> np.ndarray:
    """A slide in its strip: black rebate, a textured picture, and sprocket holes as a dashed
    line of open light just outside the long frame edges."""
    rng = np.random.default_rng(9)
    img = np.full((h, w), 0.01, np.float32)
    y1, y2, x1, x2 = int(0.12 * h), int(0.88 * h), int(0.1 * w), int(0.9 * w)
    img[y1:y2, x1:x2] = 0.3 + 0.5 * rng.random((y2 - y1, x2 - x1))
    for y in (int(0.04 * h), int(0.94 * h)):
        for x in range(int(0.05 * w), w, int(0.08 * w)):
            img[y : y + int(0.03 * h), x : x + int(0.04 * w)] = 1.0
    return np.repeat(img[:, :, None], 3, axis=2)


@pytest.mark.parametrize("angle", [-0.9, 0.4])
def test_a_slide_squares_on_its_picture_box_past_the_sprocket_holes(angle: float, monkeypatch: pytest.MonkeyPatch) -> None:
    # The film detector finds no film on a slide (black rebate on a dark bed), and under a
    # sprocket hole the hole's edge outweighs the frame edge at that station.
    import negpy.features.geometry.skew as skew_module

    real = skew_module.detect_film_bounds_with_confidence
    empty = real(np.full((40, 60, 3), 0.5, np.float32))
    monkeypatch.setattr(skew_module, "detect_film_bounds_with_confidence", lambda _image: empty)

    skew = trusted_frame_skew(_scan(_slide(), angle))

    assert skew is not None
    assert skew.fine_rotation == pytest.approx(angle, abs=0.05)
    assert all(line.source.startswith("gate") for line in skew.lines)


def _bowed(scan: np.ndarray, k: float = 0.06) -> np.ndarray:
    """`scan` through a radial distortion, as a scanning lens or film curl bows a frame edge."""
    h, w = scan.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)
    r2 = nx * nx + ny * ny
    return cv2.remap(scan, xs + k * nx * r2 * (w / 2), ys + k * ny * r2 * (h / 2), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def test_a_bowed_edge_is_levelled_by_its_chord() -> None:
    skew = trusted_frame_skew(_bowed(_scan(_strip(), 1.0)))

    assert skew is not None
    assert skew.fine_rotation == pytest.approx(1.0, abs=0.05)
    # The two film edges and the four picture edges; the bowed long ones are kept, untrimmed.
    assert len(skew.lines) == 6
    assert all(line.inlier_fraction == 1.0 for line in skew.lines)


def test_a_square_frame() -> None:
    skew = measure_frame_skew(_scan(_strip(h=700, w=800, frame=(0.55, 0.8)), -1.2))

    assert skew is not None
    assert skew.fine_rotation == pytest.approx(-1.2, abs=0.03)


@pytest.mark.parametrize(("converge_v", "converge_h"), [(1.5, 0.0), (0.0, -2.0), (-2.5, 1.0)])
def test_keystone_round_trip(converge_v: float, converge_h: float) -> None:
    skew = measure_frame_skew(_scan(_strip(), 0.7, converge_v, converge_h))

    assert skew is not None
    assert skew.fine_rotation == pytest.approx(0.7, abs=0.05)
    assert (skew.converge_v or 0.0) == pytest.approx(converge_v, abs=0.1)
    assert (skew.converge_h or 0.0) == pytest.approx(converge_h, abs=0.1)
    squared = apply_keystone(
        apply_fine_rotation(_scan(_strip(), 0.7, converge_v, converge_h), skew.fine_rotation),
        skew.converge_v or 0.0,
        skew.converge_h or 0.0,
    )
    again = measure_frame_skew(squared)
    assert again is not None and abs(again.fine_rotation) < 0.05


def test_keystone_below_the_noise_floor_is_left_out() -> None:
    skew = measure_frame_skew(_scan(_strip(), 1.0))

    assert skew is not None
    assert skew.converge_v is None and skew.converge_h is None


def test_rotation_only_never_reports_keystone() -> None:
    skew = measure_frame_skew(_scan(_strip(), 1.0, 2.0, 0.0), fit_keystone=False)

    assert skew is not None
    assert skew.converge_v is None and skew.converge_h is None


@pytest.mark.parametrize(
    "orient",
    [
        lambda a: np.rot90(a, 1),
        lambda a: np.rot90(a, 2),
        lambda a: np.rot90(a, 3),
        np.fliplr,
        np.flipud,
    ],
)
def test_the_fit_squares_the_frame_after_any_turn_or_flip(orient) -> None:
    turned = np.ascontiguousarray(orient(_scan(_strip(), 1.3)))
    skew = measure_frame_skew(turned)
    assert skew is not None

    again = measure_frame_skew(apply_fine_rotation(turned, skew.fine_rotation))

    assert again is not None
    assert again.fine_rotation == pytest.approx(0.0, abs=0.03)


def test_a_strip_fit_reports_its_film_box_lineage() -> None:
    skew = measure_frame_skew(_scan(_strip(), 1.0))

    assert skew is not None
    assert skew.from_film_box is True


def test_a_fit_without_a_film_box_is_not_box_led(monkeypatch: pytest.MonkeyPatch) -> None:
    # The film detector abstains, so the profile fallback carries the fit.
    import negpy.features.geometry.skew as skew_module

    real = skew_module.detect_film_bounds_with_confidence
    empty = real(np.full((40, 60, 3), 0.5, np.float32))
    monkeypatch.setattr(skew_module, "detect_film_bounds_with_confidence", lambda _image: empty)

    skew = measure_frame_skew(_scan(_strip(), 0.8))

    assert skew is not None
    assert skew.from_film_box is False
    assert skew.fine_rotation == pytest.approx(0.8, abs=0.05)


def _tight_scan(h: int = 600, w: int = 900, border: float = 0.035) -> np.ndarray:
    rng = np.random.default_rng(5)
    img = np.full((h, w), 0.72, np.float32)
    by, bx = int(border * h), int(border * w)
    texture = 0.2 + 0.15 * rng.random((h - 2 * by, w - 2 * bx))
    img[by : h - by, bx : w - bx] = cv2.GaussianBlur(texture.astype(np.float32), (0, 0), 3)
    return np.repeat(img[:, :, None], 3, axis=2)


@pytest.mark.parametrize("angle", [-1.2, 0.6])
def test_a_tight_scan_is_trusted(angle: float) -> None:
    skew = trusted_frame_skew(_scan(_tight_scan(), angle))

    assert skew is not None
    assert skew.fine_rotation == pytest.approx(angle, abs=0.05)


def test_a_tight_scan_the_film_detector_cannot_box_is_trusted_through_its_border_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real tight scans often fail the detector's surround checks.
    import negpy.features.geometry.skew as skew_module

    real = skew_module.detect_film_bounds_with_confidence
    empty = real(np.full((40, 60, 3), 0.5, np.float32))
    monkeypatch.setattr(skew_module, "detect_film_bounds_with_confidence", lambda _image: empty)
    monkeypatch.setattr(skew_module, "_refine_roi_to_image", lambda _image, roi: (roi, None, None))  # nor a picture box

    skew = trusted_frame_skew(_scan(_tight_scan(), -1.2))

    assert skew is not None
    assert skew.from_film_box is False
    assert skew.at_canvas_edges is True
    assert skew.fine_rotation == pytest.approx(-1.2, abs=0.05)


def test_an_opaque_ring_is_a_holder_not_a_film_border(monkeypatch: pytest.MonkeyPatch) -> None:
    import negpy.features.geometry.skew as skew_module

    real = skew_module.detect_film_bounds_with_confidence
    empty = real(np.full((40, 60, 3), 0.5, np.float32))
    monkeypatch.setattr(skew_module, "detect_film_bounds_with_confidence", lambda _image: empty)
    window = _tight_scan()
    h, w = window.shape[:2]
    by, bx = int(0.035 * h), int(0.035 * w)
    window[:by], window[h - by :], window[:, :bx], window[:, w - bx :] = 0.01, 0.01, 0.01, 0.01

    skew = measure_frame_skew(_scan(window, -1.2))

    assert skew is None or skew.at_canvas_edges is False


def test_interior_picture_lines_are_not_canvas_border(monkeypatch: pytest.MonkeyPatch) -> None:
    import negpy.features.geometry.skew as skew_module

    real = skew_module.detect_film_bounds_with_confidence
    empty = real(np.full((40, 60, 3), 0.5, np.float32))
    monkeypatch.setattr(skew_module, "detect_film_bounds_with_confidence", lambda _image: empty)

    skew = measure_frame_skew(_scan(_strip(), 0.8))

    assert skew is not None
    assert skew.at_canvas_edges is False
    assert trusted_frame_skew(_scan(_strip(), 0.8)) is None


def test_trust_demands_box_led_lines_and_an_opposite_pair() -> None:
    import negpy.features.geometry.skew as skew_module
    from negpy.features.geometry.skew import FrameSkew

    def fake(**fields):
        base = dict(fine_rotation=0.5, converge_v=None, converge_h=None, confidence=0.9, lines=(), residual=0.0)
        return FrameSkew(**{**base, **fields})

    cases = [
        (fake(from_film_box=True, has_opposite_pair=True, confidence=0.55), True),
        (fake(from_film_box=True, has_opposite_pair=True, confidence=0.45), False),
        (fake(from_film_box=False, has_opposite_pair=True), False),
        (fake(from_film_box=True, has_opposite_pair=False), False),
        (fake(from_film_box=False, has_opposite_pair=True, at_canvas_edges=True), True),
        (fake(from_film_box=False, has_opposite_pair=False, at_canvas_edges=True), False),
    ]
    original = skew_module.measure_frame_skew
    try:
        for skew, expected in cases:
            skew_module.measure_frame_skew = lambda _img, _fit=True, _s=skew: _s
            assert (trusted_frame_skew(np.zeros((8, 8, 3), np.float32)) is not None) is expected
    finally:
        skew_module.measure_frame_skew = original


def test_a_blank_frame_abstains() -> None:
    assert measure_frame_skew(np.full((400, 600, 3), 0.5, np.float32)) is None


def test_noise_is_not_trusted() -> None:
    noise = np.random.default_rng(1).random((400, 600, 3)).astype(np.float32)

    assert trusted_frame_skew(noise) is None


def test_a_single_long_picture_line_is_not_a_frame() -> None:
    img = np.full((500, 700, 3), 0.4, np.float32)
    cv2.line(img, (0, 200), (699, 260), (0.9, 0.9, 0.9), 6)

    assert trusted_frame_skew(img) is None
