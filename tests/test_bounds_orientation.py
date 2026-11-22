import numpy as np
import pytest

from negpy.features.exposure.normalization import analyze_log_exposure_bounds_from_log, prefilter_log_grid


def _frame(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    img = rng.uniform(0.05, 0.9, (h, w, 3)).astype(np.float32)
    img[:, 1::7] = 0.95
    img[::5, :] *= 0.2
    return np.ascontiguousarray(img)


@pytest.mark.parametrize("shape", [(1059, 1600), (1060, 1601), (1059, 1601), (1600, 1059), (2050, 3000), (2051, 3001)])
def test_bounds_do_not_depend_on_orientation(shape):
    img = _frame(*shape)
    results = []
    for k in range(4):
        for flip in (False, True):
            view = np.rot90(img, k)
            view = view[:, ::-1] if flip else view
            grid = prefilter_log_grid(np.ascontiguousarray(view), None, 0.05)
            b = analyze_log_exposure_bounds_from_log(grid, None, 0.0, percentile_clip=0.0, color_clip=1.0)
            results.append([*b.floors, *b.ceils])
    np.testing.assert_allclose(results, [results[0]] * len(results), atol=1e-6)
