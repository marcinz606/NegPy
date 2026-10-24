use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bounds::{self, ColorPass};

fn grid_of<'a>(grid: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'a [f32]> {
    if grid.ndim() != 3 || grid.shape()[2] != 3 {
        return Err(PyValueError::new_err("needs an HxWx3 array"));
    }
    grid.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

/// `np.percentile(grid[:, :, ch], qs)` per channel of an HxWx3 float32 grid, float64 interpolation.
#[pyfunction]
fn channel_percentiles(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    qs: Vec<f64>,
) -> PyResult<[Vec<f64>; 3]> {
    let g = grid_of(&grid)?;
    Ok(super::detach(py, g.len(), || {
        bounds::channel_percentiles(g, &qs)
    }))
}

/// `_same_pixel_color_floor_refs` with its constants (chroma quantile, chroma cap, first-pass
/// cap, minimum pixels, band width).
#[pyfunction]
fn same_pixel_color_floor_refs(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    luma_floors: [f64; 3],
    luma_ceils: [f64; 3],
    base: [f64; 3],
    color_clip: f64,
    constants: (f64, f64, f64, usize, f64),
) -> PyResult<Option<[f64; 3]>> {
    let g = grid_of(&grid)?;
    let (chroma_quantile, chroma_cap, first_pass_cap, min_pixels, band_width) = constants;
    let c = ColorPass {
        chroma_quantile,
        chroma_cap,
        first_pass_cap,
        min_pixels,
        band_width,
    };
    Ok(super::detach(py, g.len(), || {
        bounds::same_pixel_color_floor_refs(g, luma_floors, luma_ceils, base, color_clip, c)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(channel_percentiles, m)?)?;
    m.add_function(wrap_pyfunction!(same_pixel_color_floor_refs, m)?)
}
