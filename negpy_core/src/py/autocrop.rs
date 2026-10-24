use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::autocrop::{self, Template};

/// `_top_edge_slope(lum, (y1, y2, x1, x2))` on an HxW float32 luminance plane: degrees or None.
#[pyfunction]
fn top_edge_slope(
    py: Python<'_>,
    lum: PyReadonlyArray2<'_, f32>,
    roi: (i64, i64, i64, i64),
) -> PyResult<Option<f64>> {
    let dims = (lum.shape()[0], lum.shape()[1]);
    let img = lum
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous plane"))?;
    let (y1, _, x1, x2) = roi;
    Ok(super::detach(py, img.len(), || {
        autocrop::top_edge_slope(img, dims, (y1, x1, x2))
    }))
}

/// `_rect_from_edge_profile` for a float32 profile, its contrast and the roll template's
/// (center_x, width, fallback_width, top, height, center_mad, width_mad).
#[pyfunction]
fn rect_from_edge_profile(
    py: Python<'_>,
    profile: PyReadonlyArray1<'_, f32>,
    contrast: f64,
    template: (f64, f64, f64, f64, f64, f64, f64),
) -> PyResult<Option<(f64, f64, f64, f64)>> {
    let p = profile
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous profile"))?;
    let (center_x, width, fallback_width, top, height, center_mad, width_mad) = template;
    let t = Template {
        center_x,
        width,
        fallback_width,
        top,
        height,
        center_mad,
        width_mad,
    };
    Ok(super::detach(py, p.len(), || {
        autocrop::rect_from_edge_profile(p, contrast, &t)
    }))
}

/// `_place_window_by_occupancy(start, end, target_len, occupancy, scale)` for a float32 profile.
#[pyfunction]
fn place_window_by_occupancy(
    py: Python<'_>,
    start: i64,
    end: i64,
    target_len: f64,
    occupancy: PyReadonlyArray1<'_, f32>,
    scale: f64,
) -> PyResult<i64> {
    let occ = occupancy
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous profile"))?;
    Ok(super::detach(py, occ.len(), || {
        autocrop::place_window_by_occupancy(start, end, target_len, occ, scale)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(top_edge_slope, m)?)?;
    m.add_function(wrap_pyfunction!(rect_from_edge_profile, m)?)?;
    m.add_function(wrap_pyfunction!(place_window_by_occupancy, m)?)
}
