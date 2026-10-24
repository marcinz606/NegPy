use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::flatfield::{self, Lit};

fn image3<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], (usize, usize))> {
    let [h, w, 3] = a.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    Ok((s, (h, w)))
}

fn lit(c: (f64, f64, f64)) -> Lit {
    Lit {
        percentile: c.0,
        fraction: c.1,
        min_fraction: c.2,
    }
}

/// `_lit_mask(ref)` with (percentile, fraction, minimum lit fraction): an HxW bool mask, or
/// None when too little of the frame is lit.
#[pyfunction]
fn lit_mask<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    constants: (f64, f64, f64),
) -> PyResult<Option<Bound<'py, numpy::PyArray2<bool>>>> {
    use numpy::IntoPyArray;
    let (src, (h, w)) = image3(&img)?;
    let c = lit(constants);
    let mask = super::detach(py, src.len(), || flatfield::lit_mask(src, (h, w), &c));
    mask.map(|m| m.into_pyarray(py).reshape([h, w])).transpose()
}

/// `_is_clipped(image)`'s (fraction of lit pixels at `level`, frame too dark for a lit mask).
#[pyfunction]
fn clipped_fraction(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    constants: (f64, f64, f64),
    level: f32,
) -> PyResult<(f64, bool)> {
    let (src, dims) = image3(&img)?;
    let c = lit(constants);
    Ok(super::detach(py, src.len(), || {
        flatfield::clipped(src, dims, &c, level)
    }))
}

type Evenness<'py> = (Bound<'py, PyArrayDyn<f32>>, (f64, f64, f64, f64, bool));

/// `_evenness(image)`: (ratio, (low, high, color, clipped fraction, too dark)).
#[pyfunction]
fn evenness<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    constants: (f64, f64, f64),
    level: f32,
) -> PyResult<Evenness<'py>> {
    let (src, dims) = image3(&img)?;
    let c = lit(constants);
    let out = py
        .import("numpy")?
        .getattr("empty")?
        .call1((img.shape().to_vec(), "float32"))?
        .cast_into::<PyArrayDyn<f32>>()?;
    let e = {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, src.len(), || {
            flatfield::evenness(src, dims, &c, level, dst)
        })
        .ok_or_else(|| PyValueError::new_err("invalid evenness input"))?
    };
    Ok((out, (e.low, e.high, e.color, e.clipped, e.dark)))
}

/// `compute_gain` of the reference at its working size, with the lit-mask constants, sigma,
/// eps, the minimum lit weight and (gain min, gain max): (gain, too dark).
#[pyfunction]
fn compute_gain<'py>(
    py: Python<'py>,
    reference: PyReadonlyArrayDyn<'py, f32>,
    constants: (f64, f64, f64),
    sigma: f64,
    eps: f32,
    min_lit_weight: f32,
    range: (f32, f32),
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, bool)> {
    let (src, dims) = image3(&reference)?;
    let c = flatfield::Gain {
        lit: lit(constants),
        sigma,
        eps,
        min_lit_weight,
        range,
    };
    let out = PyArrayDyn::<f32>::zeros(py, reference.shape(), false);
    let dark = {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, src.len(), || {
            flatfield::compute_gain(src, dims, &c, dst)
        })
        .ok_or_else(|| PyValueError::new_err("invalid reference"))?
    };
    Ok((out, dark))
}

/// `_fill_from_nearest(values, valid)` of an HxWxC float32 array: a filled copy.
#[pyfunction]
fn fill_from_nearest<'py>(
    py: Python<'py>,
    values: PyReadonlyArrayDyn<'py, f32>,
    valid: PyReadonlyArrayDyn<'py, bool>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, cn] = values.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWxC array"));
    };
    if valid.shape() != [h, w] {
        return Err(PyValueError::new_err("needs an HxW mask"));
    }
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let mut v = values.as_slice().map_err(not_contiguous)?.to_vec();
    let m = valid.as_slice().map_err(not_contiguous)?;
    super::detach(py, v.len(), || {
        flatfield::fill_from_nearest(&mut v, m, (h, w), cn)
    });
    numpy::PyArray1::from_vec(py, v).reshape(vec![h, w, cn])
}

/// `evenness_view`'s map in place on a ratio from `evenness`, with twice the span.
#[pyfunction]
fn evenness_view(
    py: Python<'_>,
    mut ratio: numpy::PyReadwriteArrayDyn<'_, f32>,
    span: f32,
) -> PyResult<()> {
    let r = ratio
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous ratio"))?;
    super::detach(py, r.len(), || flatfield::evenness_view(r, span));
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evenness_view, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gain, m)?)?;
    m.add_function(wrap_pyfunction!(fill_from_nearest, m)?)?;
    m.add_function(wrap_pyfunction!(lit_mask, m)?)?;
    m.add_function(wrap_pyfunction!(clipped_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(evenness, m)?)
}
