use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{lsd, skew};

type Samples<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// `_fit_side`'s `sample(center, half, nearest)` on an HxW float32 luminance plane, read along
/// rows (`horizontal`) or columns: (stations, positions) of the profile peaks.
#[pyfunction]
fn skew_sample<'py>(
    py: Python<'py>,
    lum: PyReadonlyArray2<'py, f32>,
    horizontal: bool,
    stations: PyReadonlyArray1<'py, f64>,
    centers: PyReadonlyArray1<'py, f64>,
    half: f64,
    nearest: bool,
) -> PyResult<Samples<'py>> {
    let (h, w) = (lum.shape()[0], lum.shape()[1]);
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let img = lum.as_slice().map_err(not_contiguous)?;
    let (st, ce) = (
        stations.as_slice().map_err(not_contiguous)?,
        centers.as_slice().map_err(not_contiguous)?,
    );
    if st.len() != ce.len() {
        return Err(PyValueError::new_err("needs one center per station"));
    }
    let (xs, ys) = super::detach(py, img.len(), || {
        skew::sample(img, (h, w), horizontal, (st, ce), half, nearest)
    });
    Ok((xs.into_pyarray(py), ys.into_pyarray(py)))
}

type PyLine = ((f64, f64), (f64, f64), bool, f64, f64);

fn py_line(l: skew::Line) -> PyLine {
    (l.p1, l.p2, l.horizontal, l.inliers, l.rms)
}

fn plane<'a>(lum: &'a PyReadonlyArray2<'_, f32>) -> PyResult<(&'a [f32], (usize, usize))> {
    let dims = (lum.shape()[0], lum.shape()[1]);
    let img = lum
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous plane"))?;
    Ok((img, dims))
}

/// `_fit_side(lum, horizontal, at, start, stop, outward)` on an HxW float32 luminance plane:
/// (p1, p2, horizontal, inlier fraction, rms) or None.
#[pyfunction]
fn skew_fit_side(
    py: Python<'_>,
    lum: PyReadonlyArray2<'_, f32>,
    horizontal: bool,
    at: f64,
    start: f64,
    stop: f64,
    outward: i32,
) -> PyResult<Option<PyLine>> {
    let (img, dims) = plane(&lum)?;
    Ok(super::detach(py, img.len(), || {
        skew::fit_side(img, dims, horizontal, (at, start, stop), outward)
    })
    .map(py_line))
}

/// `_profile_lines(lum)`: the outermost long edges, each (p1, p2, horizontal, inliers, rms).
#[pyfunction]
fn skew_profile_lines(py: Python<'_>, lum: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<PyLine>> {
    let (img, dims) = plane(&lum)?;
    let lines = super::detach(py, img.len(), || skew::profile_lines(img, dims));
    Ok(lines.into_iter().map(py_line).collect())
}

/// `_at_canvas_edges` for lines given as (horizontal, position).
#[pyfunction]
fn skew_at_canvas_edges(
    py: Python<'_>,
    lum: PyReadonlyArray2<'_, f32>,
    lines: Vec<(bool, f64)>,
) -> PyResult<bool> {
    let (img, dims) = plane(&lum)?;
    Ok(super::detach(py, img.len(), || {
        skew::at_canvas_edges(img, dims, &lines)
    }))
}

/// `_coarse_angles` on an HxW float32 luminance plane: the dominant folded segment angles in
/// degrees, strongest first.
#[pyfunction]
fn coarse_angles(
    py: Python<'_>,
    lum: PyReadonlyArray2<'_, f32>,
    min_segment: f64,
    max_skew: f64,
    angle_bin: f64,
) -> PyResult<Vec<f64>> {
    let (h, w) = (lum.shape()[0], lum.shape()[1]);
    let img = lum
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    if img.is_empty() || angle_bin <= 0.0 || max_skew <= 0.0 {
        return Err(PyValueError::new_err(
            "needs a non-empty plane and positive bins",
        ));
    }
    Ok(super::detach(py, img.len(), || {
        skew::coarse_angles(img, (h, w), min_segment, max_skew, angle_bin)
    }))
}

/// `cv2.createLineSegmentDetector().detect(img)[0]` for an HxW uint8 image, as an (N, 4)
/// float32 array of (x1, y1, x2, y2).
#[pyfunction]
fn line_segments<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w) = (img.shape()[0], img.shape()[1]);
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let lines = super::detach(py, src.len(), || lsd::line_segments(src, (h, w)));
    let n = lines.len();
    let flat: Vec<f32> = lines.into_iter().flatten().collect();
    flat.into_pyarray(py).reshape([n, 4])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(skew_sample, m)?)?;
    m.add_function(wrap_pyfunction!(coarse_angles, m)?)?;
    m.add_function(wrap_pyfunction!(line_segments, m)?)?;
    m.add_function(wrap_pyfunction!(skew_fit_side, m)?)?;
    m.add_function(wrap_pyfunction!(skew_profile_lines, m)?)?;
    m.add_function(wrap_pyfunction!(skew_at_canvas_edges, m)?)
}
