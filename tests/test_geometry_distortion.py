import dataclasses
import math

import numpy as np
import pytest

from negpy.features.geometry.logic import (
    _radial_maps,
    apply_radial_distortion,
    compute_distortion_scale,
    map_point_radial,
)
from negpy.infrastructure.gpu.device import GPUDevice

# Slider range; the model must be well-behaved (no fold-over) across it.
_K1_RANGE = [-0.25, -0.1, -0.02, 0.02, 0.1, 0.25]


def test_zero_k1_is_identity():
    img = np.random.rand(40, 60, 3).astype(np.float32)
    out = apply_radial_distortion(img, 0.0)
    assert out is img  # no-op short-circuit
    assert compute_distortion_scale(0.0, 60, 40) == 1.0


@pytest.mark.parametrize("k1", _K1_RANGE)
def test_brightness_preserving(k1):
    """Pure geometric remap of a flat field must not change any value."""
    img = np.full((50, 80, 3), 0.37, dtype=np.float32)
    out = apply_radial_distortion(img, k1)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, 0.37, atol=1e-5)


@pytest.mark.parametrize("k1", _K1_RANGE)
def test_no_empty_borders(k1):
    """Scale-to-fill must keep every output sample inside the input frame, so the
    corrected frame has no empty/replicated edges. Verified on the source maps."""
    w, h = 90, 60
    map_x, map_y = _radial_maps(k1, w, h)
    tol = 0.5  # half-pixel slack
    # Every output pixel samples a valid interior point -> no empty/replicated border.
    assert map_x.min() >= -tol and map_x.max() <= (w - 1) + tol
    assert map_y.min() >= -tol and map_y.max() <= (h - 1) + tol
    # And the fit is tight (not over-zoomed): at least one pair of edges is reached.
    # A uniform scale can only fill the binding axis; the other may be cropped.
    min_gap = min(map_x.min(), (w - 1) - map_x.max(), map_y.min(), (h - 1) - map_y.max())
    assert min_gap <= 1.0


@pytest.mark.parametrize("k1", _K1_RANGE)
def test_no_fold_over(k1):
    """1 + k1·r² must stay positive across the frame (no mirror fold)."""
    w, h = 90, 60
    s = compute_distortion_scale(k1, w, h)
    halfdiag = 0.5 * math.hypot(w, h)
    corner = math.hypot((w - 1) / 2.0, (h - 1) / 2.0) * s
    assert 1.0 + k1 * (corner / halfdiag) ** 2 > 0.0


@pytest.mark.parametrize("k1", _K1_RANGE)
def test_point_mapper_inverts_resample(k1):
    """map_point_radial is the inverse of the resample map: forward (output→source)
    then map_point_radial (source→output) round-trips to identity."""
    w, h = 90, 60
    map_x, map_y = _radial_maps(k1, w, h)
    for oy, ox in [(10, 15), (5, 80), (55, 3), (30, 45), (0, 0)]:
        src_x, src_y = float(map_x[oy, ox]), float(map_y[oy, ox])
        back_x, back_y = map_point_radial(src_x, src_y, k1, w, h)
        assert abs(back_x - ox) < 0.75
        assert abs(back_y - oy) < 0.75


@pytest.mark.parametrize("k1", _K1_RANGE)
def test_corrects_then_recorrects_round_trip(k1):
    """Applying k1 then a re-distortion that undoes it returns near the original
    interior (sanity on the model; ignores the cropped border ring). Uses a smooth
    image — high-frequency content is lost to bilinear interpolation, not the model."""
    yy, xx = np.mgrid[0:60, 0:90].astype(np.float32)
    grad = (0.2 + 0.6 * xx / 89.0 + 0.2 * yy / 59.0).astype(np.float32)
    img = np.stack([grad, grad * 0.8 + 0.1, 1.0 - grad * 0.5], axis=-1).astype(np.float32)
    corrected = apply_radial_distortion(img, k1)
    # Build the inverse map explicitly and resample back.
    import cv2

    w, h = 90, 60
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    inv_x = np.empty_like(xs)
    inv_y = np.empty_like(ys)
    for j in range(h):
        for i in range(w):
            mx, my = map_point_radial(float(xs[j, i]), float(ys[j, i]), k1, w, h)
            inv_x[j, i] = mx
            inv_y[j, i] = my
    back = cv2.remap(corrected, inv_x, inv_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = slice(12, h - 12), slice(18, w - 18)
    assert np.mean(np.abs(back[m] - img[m])) < 0.06


def test_create_uv_grid_matches_f64_reference():
    """The float32 fast path must reproduce the original f64-meshgrid formula bitwise
    (np.linspace(..., dtype=f32) == np.linspace(...).astype(f32) for [0,1] ranges)."""
    import cv2

    from negpy.services.view.coordinate_mapping import CoordinateMapping

    def reference(rh, rw, rotation, fine_rot, flip_h, flip_v, roi):
        u, v = np.meshgrid(np.linspace(0, 1, rw), np.linspace(0, 1, rh))
        g = np.stack([u, v], axis=-1).astype(np.float32)
        if rotation != 0:
            g = np.rot90(g, k=rotation).astype(np.float32)
        if flip_h:
            g = np.fliplr(g).astype(np.float32)
        if flip_v:
            g = np.flipud(g).astype(np.float32)
        if fine_rot != 0.0:
            h_r, w_r = g.shape[:2]
            m = cv2.getRotationMatrix2D((w_r / 2.0, h_r / 2.0), fine_rot, 1.0)
            g = cv2.warpAffine(g, m, (w_r, h_r), flags=cv2.INTER_LINEAR).astype(np.float32)
        if roi:
            y1, y2, x1, x2 = roi
            g = g[y1:y2, x1:x2].astype(np.float32)
        return g

    cases = [
        (33, 47, 0, 0.0, False, False, None),
        (33, 47, 1, 0.0, True, False, None),
        (33, 47, 2, 1.5, False, True, None),
        (33, 47, 3, 0.0, True, True, (2, 20, 3, 30)),
    ]
    for rh, rw, rot, fine, fh, fv, roi in cases:
        got = CoordinateMapping.create_uv_grid(
            rh,
            rw,
            rotation=rot,
            fine_rot=fine,
            flip_h=fh,
            flip_v=fv,
            autocrop=roi is not None,
            autocrop_params={"roi": roi} if roi else None,
        )
        assert got.dtype == np.float32
        assert got.flags["C_CONTIGUOUS"]
        assert np.array_equal(got, reference(rh, rw, rot, fine, fh, fv, roi))


@pytest.mark.parametrize("k1", _K1_RANGE)
def test_uv_grid_and_point_mapper_are_consistent(k1):
    """The crop tool maps clicks->raw via the uv grid (forward) and renders features
    raw->display via map_coords_to_geometry (inverse). They must round-trip, or crop
    rects / retouch spots / masks drift from where the user placed them."""
    from negpy.features.geometry.logic import map_coords_to_geometry
    from negpy.services.view.coordinate_mapping import CoordinateMapping

    rh, rw = 80, 120
    uv = CoordinateMapping.create_uv_grid(rh, rw, rotation=0, fine_rot=0.0, distortion_k1=k1)
    for nx, ny in [(0.5, 0.5), (0.2, 0.3), (0.85, 0.7), (0.1, 0.9)]:
        rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv)
        vx, vy = map_coords_to_geometry(rx, ry, (rh, rw), distortion_k1=k1)
        assert abs(vx - nx) < 0.02 and abs(vy - ny) < 0.02


@pytest.mark.skipif(not GPUDevice.get().is_available, reason="GPU not available")
@pytest.mark.parametrize("k1", [-0.12, 0.12])
def test_cpu_gpu_distortion_parity(k1):
    """The radial model lives in two places (apply_radial_distortion / transform.wgsl);
    they must agree or GPU previews drift from CPU exports."""
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    processor = ImageProcessor()
    if processor.engine_gpu is None:
        pytest.skip("GPU engine not initialised")

    rng = np.random.default_rng(1)
    h, w = 80, 100
    grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
    img = np.repeat(grad[None, :], h, axis=0)
    img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
    img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

    base = WorkspaceConfig()
    # apply=True gates the distortion; empty reference_path makes the photometric flat a no-op.
    settings = dataclasses.replace(base, flatfield=dataclasses.replace(base.flatfield, apply=True, k1=k1))

    def render(prefer_gpu):
        result, _ = processor.run_pipeline(
            img, settings, "distort-parity", render_size_ref=float(max(h, w)), prefer_gpu=prefer_gpu, readback_metrics=False
        )
        arr = result.readback() if hasattr(result, "readback") else result
        return np.asarray(arr)[:, :, :3].astype(np.float64)

    cpu = render(False)
    gpu = render(True)
    assert cpu.shape == gpu.shape
    mad = float(np.mean(np.abs(cpu - gpu)))
    assert mad < 0.02, f"mean abs diff {mad:.4f}"
