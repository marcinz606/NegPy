use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::lens::{self, Frame};

type Maps<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>);
type FrameArgs = ((f64, f64), (f64, f64), (f64, f64), f64, f64);

/// `RectilinearWarp.remap`'s maps with the distortion applied, for rows (start, stop) of a
/// w-wide frame: the coefficients (k0..k3, t0, t1), then the frame's buffer origin, scale,
/// warp center, radius and fill scale.
#[pyfunction]
fn rectilinear_map<'py>(
    py: Python<'py>,
    w: usize,
    rows: (usize, usize),
    coefficients: [f64; 6],
    frame: FrameArgs,
) -> PyResult<Maps<'py>> {
    let (start, stop) = rows;
    if stop < start || w == 0 {
        return Err(PyValueError::new_err("invalid rows"));
    }
    let (origin, scale, center, radius, fill) = frame;
    let f = Frame {
        origin,
        scale,
        center,
        radius,
        fill,
    };
    let n = (stop - start) * w;
    let (mut mx, mut my) = (vec![0f32; n], vec![0f32; n]);
    super::detach(py, n, || {
        lens::rectilinear_map(w, start..stop, coefficients, f, &mut mx, &mut my)
    });
    Ok((
        mx.into_pyarray(py).reshape([stop - start, w])?,
        my.into_pyarray(py).reshape([stop - start, w])?,
    ))
}

fn maps<'py>(
    py: Python<'py>,
    (rows, w): (usize, usize),
    fill: impl FnOnce(&mut [f32], &mut [f32]) + Send,
) -> PyResult<Maps<'py>> {
    let n = rows * w;
    let (mut mx, mut my) = (vec![0f32; n], vec![0f32; n]);
    super::detach(py, n, || fill(&mut mx, &mut my));
    Ok((
        mx.into_pyarray(py).reshape([rows, w])?,
        my.into_pyarray(py).reshape([rows, w])?,
    ))
}

/// `RectilinearWarp.remap`'s maps with lateral CA only, for rows (start, stop) of a w-wide
/// frame: the channel's and the green coefficients, then the frame as for `rectilinear_map`.
#[pyfunction]
fn rectilinear_ca_map<'py>(
    py: Python<'py>,
    w: usize,
    rows: (usize, usize),
    coefficients: ([f64; 6], [f64; 6]),
    frame: FrameArgs,
) -> PyResult<Maps<'py>> {
    let (start, stop) = rows;
    if stop < start || w == 0 {
        return Err(PyValueError::new_err("invalid rows"));
    }
    let (origin, scale, center, radius, fill) = frame;
    let f = Frame {
        origin,
        scale,
        center,
        radius,
        fill,
    };
    maps(py, (stop - start, w), |mx, my| {
        lens::rectilinear_ca_map(w, start..stop, coefficients, f, mx, my)
    })
}

/// `PanasonicWarp.remap`'s maps with the distortion applied, for rows (start, stop) of an h x w
/// frame, with (scale, a, b, c) and the fill scale.
#[pyfunction]
fn panasonic_map<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    rows: (usize, usize),
    model: (f64, f64, f64, f64),
    fill: f64,
) -> PyResult<Maps<'py>> {
    let (start, stop) = rows;
    if stop < start || shape.1 == 0 {
        return Err(PyValueError::new_err("invalid rows"));
    }
    maps(py, (stop - start, shape.1), |mx, my| {
        lens::panasonic_map(shape, start..stop, model, fill, mx, my)
    })
}

/// `SonyWarp.remap`'s maps for rows (start, stop) of an h x w frame, with the combined knot
/// factors and the fill scale.
#[pyfunction]
fn sony_map<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    rows: (usize, usize),
    factors: Vec<f64>,
    fill: f64,
) -> PyResult<Maps<'py>> {
    let (start, stop) = rows;
    if stop < start || shape.1 == 0 || factors.len() < 2 {
        return Err(PyValueError::new_err("invalid rows or knots"));
    }
    maps(py, (stop - start, shape.1), |mx, my| {
        lens::sony_map(shape, start..stop, &factors, fill, mx, my)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rectilinear_map, m)?)?;
    m.add_function(wrap_pyfunction!(rectilinear_ca_map, m)?)?;
    m.add_function(wrap_pyfunction!(panasonic_map, m)?)?;
    m.add_function(wrap_pyfunction!(sony_map, m)?)
}
