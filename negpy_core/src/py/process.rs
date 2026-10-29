use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::process;

fn pixels<'a>(
    a: &'a PyReadonlyArrayDyn<'_, f32>,
    cn_min: usize,
) -> PyResult<(&'a [f32], usize, usize)> {
    let shape = a.shape();
    let (w, cn) = match shape {
        [_, w, c] if *c >= cn_min => (*w, *c),
        _ => {
            return Err(PyValueError::new_err(format!(
                "needs an HxWxC array with C >= {cn_min}"
            )));
        }
    };
    let data = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    Ok((data, w, cn))
}

/// `lightbox_level`'s float64 reference value of an HxWx3 float32 image.
#[pyfunction]
#[pyo3(signature = (img, step, matrix, pct))]
fn lightbox_white(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    step: usize,
    matrix: Option<[[f64; 3]; 3]>,
    pct: f64,
) -> PyResult<f64> {
    let (data, _, cn) = pixels(&img, 3)?;
    if cn != 3 {
        return Err(PyValueError::new_err("lightbox_white needs 3 channels"));
    }
    Ok(super::detach(py, data.len() / step.max(1), || {
        process::lightbox_white(data, step, matrix, pct)
    }))
}

/// `detect_process_mode`'s statistics of an HxWx3 float32 image: (min correlation,
/// (r mean, b mean, r p25, b p25, r p98, g p98, b p98)).
#[pyfunction]
fn process_mode_stats(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(f64, [f64; 7])> {
    let (data, _, cn) = pixels(&img, 3)?;
    if cn != 3 || data.is_empty() {
        return Err(PyValueError::new_err(
            "process_mode_stats needs a non-empty HxWx3 array",
        ));
    }
    Ok(super::detach(py, data.len(), || {
        process::process_mode_stats(data)
    }))
}

/// `detect_process_and_source`'s chroma spread of an HxWx3 float32 image, or None when too few
/// blocks are usable.
#[pyfunction]
fn chroma_spread(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    block: usize,
    lo: f32,
    hi: f32,
    min_share: f64,
) -> PyResult<Option<f64>> {
    let (data, w, cn) = pixels(&img, 3)?;
    if cn != 3 {
        return Err(PyValueError::new_err("chroma_spread needs 3 channels"));
    }
    Ok(super::detach(py, data.len(), || {
        process::chroma_spread(data, w, block, lo, hi, min_share)
    }))
}

/// `sample_patches` of an HxWxC float32 image, one (x0, x1, y0, y1) box per patch; with `gamma`,
/// of the clipped image decoded from the working encoding.
#[pyfunction]
fn chart_patches(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    boxes: Vec<(usize, usize, usize, usize)>,
    gamma: Option<f32>,
) -> PyResult<Vec<[f64; 3]>> {
    let (data, w, cn) = pixels(&img, 3)?;
    let h = img.shape()[0];
    if boxes
        .iter()
        .any(|&(x0, x1, y0, y1)| x0 >= x1 || y0 >= y1 || x1 > w || y1 > h)
    {
        return Err(PyValueError::new_err("patch box outside the image"));
    }
    let n = boxes
        .iter()
        .map(|b| (b.1 - b.0) * (b.3 - b.2))
        .sum::<usize>()
        * 3;
    Ok(super::detach(py, n, || {
        process::chart_patches(data, w, cn, &boxes, gamma)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lightbox_white, m)?)?;
    m.add_function(wrap_pyfunction!(process_mode_stats, m)?)?;
    m.add_function(wrap_pyfunction!(chroma_spread, m)?)?;
    m.add_function(wrap_pyfunction!(chart_patches, m)?)
}
