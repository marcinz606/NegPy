use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::sensor::{self, Layer, Matrix};

fn data<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'a [f32]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))
}

pub(super) fn matrix(m: &PyReadonlyArray2<'_, f32>) -> PyResult<Matrix> {
    let m = m.as_array();
    if m.shape() != [3, 3] {
        return Err(PyValueError::new_err("needs a 3x3 matrix"));
    }
    Ok(std::array::from_fn(|r| std::array::from_fn(|c| m[[r, c]])))
}

fn triplet(a: &PyReadonlyArray1<'_, f32>) -> PyResult<[f32; 3]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?
        .try_into()
        .map_err(|_| PyValueError::new_err("needs a 3-vector"))
}

/// The writable data of `out`, which must have `shape`.
fn target<'a>(
    out: &'a mut PyReadwriteArrayDyn<'_, f32>,
    shape: &[usize],
) -> PyResult<&'a mut [f32]> {
    if out.shape() != shape {
        return Err(PyValueError::new_err("out does not match the input"));
    }
    out.as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous out"))
}

fn image(x: &PyReadonlyArrayDyn<'_, f32>) -> PyResult<()> {
    if x.shape().len() != 3 || x.shape()[2] != 3 {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    }
    Ok(())
}

/// `_unmix_kernel(x, m, out)`.
#[pyfunction]
fn sensor_unmix(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<'_, f32>,
    m: PyReadonlyArray2<'_, f32>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    image(&x)?;
    let (src, m) = (data(&x)?, matrix(&m)?);
    let dst = target(&mut out, x.shape())?;
    super::detach(py, dst.len(), || sensor::unmix(src, dst, m));
    Ok(())
}

/// `_log_unmix_kernel(x, c, offset, floor, out)`.
#[pyfunction]
fn sensor_log_unmix(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<'_, f32>,
    c: PyReadonlyArray2<'_, f32>,
    offset: PyReadonlyArray1<'_, f32>,
    floor: PyReadonlyArray1<'_, f32>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    image(&x)?;
    let (src, c, offset, floor) = (data(&x)?, matrix(&c)?, triplet(&offset)?, triplet(&floor)?);
    let dst = target(&mut out, x.shape())?;
    super::detach(py, dst.len(), || {
        sensor::log_unmix(src, dst, c, offset, floor)
    });
    Ok(())
}

/// `_two_scale_layer_kernel(blurred, m, own, floor, log_floor, start, full, color, weight)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sensor_two_scale_layer(
    py: Python<'_>,
    blurred: PyReadonlyArrayDyn<'_, f32>,
    m: PyReadonlyArray2<'_, f32>,
    own: PyReadonlyArray1<'_, f32>,
    floor: f64,
    log_floor: f64,
    start: f64,
    full: f64,
    mut color: PyReadwriteArrayDyn<'_, f32>,
    mut weight: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    image(&blurred)?;
    let (src, m, own) = (data(&blurred)?, matrix(&m)?, triplet(&own)?);
    let shape = blurred.shape().to_vec();
    let c = target(&mut color, &shape)?;
    let w = target(&mut weight, &shape[..2])?;
    let s = Layer {
        floor,
        log_floor,
        start,
        full,
    };
    super::detach(py, src.len(), || {
        sensor::two_scale_layer(src, m, own, s, c, w)
    });
    Ok(())
}

/// `_two_scale_kernel(x, blurred, color, weight, m, c, floor, log_defect, out)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sensor_two_scale(
    py: Python<'_>,
    x: PyReadonlyArrayDyn<'_, f32>,
    blurred: PyReadonlyArrayDyn<'_, f32>,
    color: PyReadonlyArrayDyn<'_, f32>,
    weight: PyReadonlyArrayDyn<'_, f32>,
    m: PyReadonlyArray2<'_, f32>,
    c: PyReadonlyArray2<'_, f32>,
    floor: f64,
    log_defect: f64,
    mut out: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    image(&x)?;
    let shape = x.shape().to_vec();
    if blurred.shape() != shape || color.shape() != shape || weight.shape() != &shape[..2] {
        return Err(PyValueError::new_err("the layers do not match the image"));
    }
    let (xs, b, col, w) = (data(&x)?, data(&blurred)?, data(&color)?, data(&weight)?);
    let (m, c) = (matrix(&m)?, matrix(&c)?);
    let dst = target(&mut out, &shape)?;
    super::detach(py, dst.len(), || {
        sensor::two_scale(xs, b, col, w, m, c, (floor, log_defect), dst)
    });
    Ok(())
}

/// `film_base` of its HxWx3 analysis crop with (percentile, clip plateau, grid edge, log
/// floor, share): the base color, or None.
#[pyfunction]
fn film_base(
    py: Python<'_>,
    sample: PyReadonlyArrayDyn<'_, f32>,
    constants: (f64, f64, f64, f32, f64),
) -> PyResult<Option<[f64; 3]>> {
    let [h, w, 3] = sample.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 crop"));
    };
    let s = data(&sample)?;
    let (percentile, clip_plateau, grid_edge, log_floor, share) = constants;
    let c = sensor::FilmBase {
        percentile,
        clip_plateau,
        grid_edge,
        log_floor,
        share,
    };
    Ok(super::detach(py, s.len(), || {
        sensor::film_base(s, (h, w), &c)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(film_base, m)?)?;
    m.add_function(wrap_pyfunction!(sensor_unmix, m)?)?;
    m.add_function(wrap_pyfunction!(sensor_log_unmix, m)?)?;
    m.add_function(wrap_pyfunction!(sensor_two_scale_layer, m)?)?;
    m.add_function(wrap_pyfunction!(sensor_two_scale, m)?)
}
