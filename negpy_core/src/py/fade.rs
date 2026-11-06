use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::fade;

fn cells<'a>(grid: &'a PyReadonlyArrayDyn<'_, f32>, name: &str) -> PyResult<&'a [[f32; 3]]> {
    let shape = grid.shape();
    if shape.len() < 2 || shape[shape.len() - 1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} needs an (..., 3) array"
        )));
    }
    Ok(grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?
        .as_chunks::<3>()
        .0)
}

/// `_measured_slide_bounds`' percentiles of an (..., 3) float32 log grid: one (r, g, b) per q.
#[pyfunction]
fn slide_extremes(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    clipped_log: f64,
    qs: Vec<f64>,
) -> PyResult<Option<Vec<[f32; 3]>>> {
    let data = cells(&grid, "slide_extremes")?;
    Ok(super::detach(py, data.len(), || {
        fade::slide_extremes(data, clipped_log, &qs)
    }))
}

/// `measure_white` of an (..., 3) float32 log grid at percentile q: (r, g, b) or None.
#[pyfunction]
fn fade_white(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    clipped_log: f64,
    q: f64,
) -> PyResult<Option<[f32; 3]>> {
    let data = cells(&grid, "fade_white")?;
    Ok(super::detach(py, data.len(), || {
        fade::white(data, clipped_log, q)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(slide_extremes, m)?)?;
    m.add_function(wrap_pyfunction!(fade_white, m)?)?;
    Ok(())
}
