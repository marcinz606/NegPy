use numpy::{PyArray1, PyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::fade::{self, BandParams};

/// (inner band edges, channel weights, range percentiles, min span, neutral spread, cluster
/// radius, mean-shift steps, min weight, clipped log, min cells).
type BandConsts = (
    Vec<f64>,
    [f64; 3],
    (f64, f64),
    f64,
    f64,
    f64,
    usize,
    f64,
    f64,
    usize,
);

/// Each band's centre (NaN with no weight), its weight, and the total weight.
type BandsOut<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, f64);

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

/// `neutral_bands(grid)` of an (..., 3) float32 log grid: (centre, weight, total) or None.
#[pyfunction]
fn neutral_bands<'py>(
    py: Python<'py>,
    grid: PyReadonlyArrayDyn<'py, f32>,
    consts: BandConsts,
) -> PyResult<Option<BandsOut<'py>>> {
    let data = cells(&grid, "neutral_bands")?;
    let (
        inner_edges,
        weights,
        range_pct,
        min_span,
        neutral_spread,
        cluster_radius,
        steps,
        min_weight,
        clipped_log,
        min_cells,
    ) = consts;
    if inner_edges.len() > 254 {
        return Err(PyValueError::new_err(
            "neutral_bands takes at most 255 bands",
        ));
    }
    let p = BandParams {
        inner_edges,
        weights,
        range_pct,
        min_span,
        neutral_spread,
        cluster_radius,
        steps,
        min_weight,
        clipped_log,
        min_cells,
    };
    let Some(bands) = super::detach(py, data.len(), || fade::neutral_bands(data, &p)) else {
        return Ok(None);
    };
    let centre = PyArray2::from_vec2(
        py,
        &bands.centre.iter().map(|c| c.to_vec()).collect::<Vec<_>>(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Some((
        centre,
        PyArray1::from_vec(py, bands.weight),
        bands.total,
    )))
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

/// `range_points`' percentiles of an (..., 3) float32 density grid's cell means, or None.
#[pyfunction]
fn mean_percentiles(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    qs: Vec<f64>,
    min_cells: usize,
) -> PyResult<Option<Vec<f64>>> {
    let data = cells(&grid, "mean_percentiles")?;
    Ok(super::detach(py, data.len(), || {
        fade::mean_percentiles(data, &qs, min_cells)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neutral_bands, m)?)?;
    m.add_function(wrap_pyfunction!(slide_extremes, m)?)?;
    m.add_function(wrap_pyfunction!(fade_white, m)?)?;
    m.add_function(wrap_pyfunction!(mean_percentiles, m)?)?;
    Ok(())
}
