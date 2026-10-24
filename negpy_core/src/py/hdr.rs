use numpy::{PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::hdr;

fn contiguous<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'a [f32]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))
}

/// `_count_clipped_rows(values, out)` of an HxWxC float32 frame.
#[pyfunction]
fn count_clipped_rows(
    py: Python<'_>,
    values: PyReadonlyArrayDyn<'_, f32>,
    saturation: f64,
    mut out: PyReadwrite1<'_>,
) -> PyResult<()> {
    let [h, w, cn] = values.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWxC frame"));
    };
    if out.len() != h {
        return Err(PyValueError::new_err("out needs one entry per row"));
    }
    let v = contiguous(&values)?;
    let dst = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous out"))?;
    super::detach(py, dst.len(), || {
        hdr::count_clipped_rows(v, (w, cn), saturation, dst)
    });
    Ok(())
}

type PyReadwrite1<'py> = PyReadwriteArray1<'py, i64>;

/// `_accumulate(num, den, values, ratio)`, with the module's saturation constants.
#[pyfunction]
fn hdr_accumulate(
    py: Python<'_>,
    mut num: PyReadwriteArrayDyn<'_, f32>,
    mut den: PyReadwriteArrayDyn<'_, f32>,
    values: PyReadonlyArrayDyn<'_, f32>,
    ratio: f64,
    saturation: f64,
    rolloff_start: f64,
) -> PyResult<()> {
    if num.shape() != values.shape() || den.shape() != values.shape() {
        return Err(PyValueError::new_err("the sums do not match the frame"));
    }
    let v = contiguous(&values)?;
    let n = num
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous sums"))?;
    let d = den
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous sums"))?;
    super::detach(py, v.len(), || {
        hdr::accumulate(n, d, v, ratio, (saturation, rolloff_start))
    });
    Ok(())
}

type Dims = (usize, usize, usize);

fn frame3<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], Dims)> {
    let [h, w, cn] = a.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWxC frame"));
    };
    if cn < 3 {
        return Err(PyValueError::new_err("needs three channels or more"));
    }
    Ok((contiguous(a)?, (h, w, cn)))
}

fn new_like<'py>(
    py: Python<'py>,
    shape: &[usize],
    fill: impl FnOnce(&mut [f32]) + Send,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let out = py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.to_vec(), "float32"))?
        .cast_into::<PyArrayDyn<f32>>()?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || fill(dst));
    }
    Ok(out)
}

/// `measure_exposure`'s level of a float32 frame: the q-th percentile of `frame[::step, ::step, :3]`.
#[pyfunction]
fn exposure_level(
    py: Python<'_>,
    frame: PyReadonlyArrayDyn<'_, f32>,
    step: usize,
    q: f64,
) -> PyResult<f32> {
    let (f, dims) = frame3(&frame)?;
    super::detach(py, f.len(), || hdr::exposure_level(f, dims, step, q))
        .ok_or_else(|| PyValueError::new_err("percentile of an empty frame"))
}

/// `pair_ratio` of two float32 frames of one shape, with the subsample step and (floor,
/// saturation).
#[pyfunction]
fn pair_ratio(
    py: Python<'_>,
    short: PyReadonlyArrayDyn<'_, f32>,
    long: PyReadonlyArrayDyn<'_, f32>,
    step: usize,
    limits: (f32, f32),
) -> PyResult<Option<f64>> {
    let (s, dims) = frame3(&short)?;
    if long.shape() != short.shape() {
        return Err(PyValueError::new_err("bracket frames differ in shape"));
    }
    let l = contiguous(&long)?;
    Ok(super::detach(py, s.len(), || {
        hdr::pair_ratio(s, l, dims, step, limits)
    }))
}

/// The end of `merge_providers`: num / den, the reference where den <= 0.
#[pyfunction]
fn finish_merge<'py>(
    py: Python<'py>,
    num: PyReadonlyArrayDyn<'py, f32>,
    den: PyReadonlyArrayDyn<'py, f32>,
    reference: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    if den.shape() != num.shape() || reference.shape() != num.shape() {
        return Err(PyValueError::new_err("needs arrays of one shape"));
    }
    let (n, d, r) = (
        contiguous(&num)?,
        contiguous(&den)?,
        contiguous(&reference)?,
    );
    new_like(py, num.shape(), |dst| hdr::finish_merge(n, d, r, dst))
}

/// `apply_render_exposure`: merged times scale, clipped to 0..1.
#[pyfunction]
fn render_exposure<'py>(
    py: Python<'py>,
    merged: PyReadonlyArrayDyn<'py, f32>,
    scale: f32,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let m = contiguous(&merged)?;
    new_like(py, merged.shape(), |dst| {
        hdr::render_exposure(m, scale, dst)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_clipped_rows, m)?)?;
    m.add_function(wrap_pyfunction!(exposure_level, m)?)?;
    m.add_function(wrap_pyfunction!(pair_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(finish_merge, m)?)?;
    m.add_function(wrap_pyfunction!(render_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(hdr_accumulate, m)?)
}
