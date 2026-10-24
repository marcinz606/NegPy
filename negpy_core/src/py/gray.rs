use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::filter::Border;
use crate::gray;

fn empty_u8<'py>(py: Python<'py>, shape: &[usize]) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    Ok(py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.to_vec(), "uint8"))?
        .cast_into::<PyArrayDyn<u8>>()?)
}

fn contiguous<'a>(src: &'a PyReadonlyArrayDyn<'_, u8>) -> PyResult<&'a [u8]> {
    src.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

/// `cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)` for an HxWx3 uint8 image.
#[pyfunction]
fn bgr_to_gray<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = src.shape().to_vec();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("bgr_to_gray needs an HxWx3 array"));
    }
    let data = contiguous(&src)?;
    let out = empty_u8(py, &shape[..2])?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || gray::bgr_to_gray(data, dst));
    }
    Ok(out)
}

/// `cv2.GaussianBlur(src, ksize, sigma_x, sigmaY=sigma_y, borderType=border)` for uint8.
#[pyfunction]
#[pyo3(signature = (src, ksize, sigma_x, sigma_y=0.0, border_type=4))]
fn gaussian_blur_u8<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    ksize: (usize, usize),
    sigma_x: f64,
    sigma_y: f64,
    border_type: i32,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = src.shape().to_vec();
    let dims = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => return Err(PyValueError::new_err("needs a 2-D or 3-D array")),
    };
    let border = Border::from_cv(border_type)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported border type {border_type}")))?;
    let data = contiguous(&src)?;
    let out = empty_u8(py, &shape)?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            gray::gaussian_blur_u8(data, dst, dims, ksize, (sigma_x, sigma_y), border)
        })
        .ok_or_else(|| PyValueError::new_err("invalid kernel size"))?;
    }
    Ok(out)
}

/// `cv2.threshold(src, 0, maxval, THRESH_BINARY[_INV] + THRESH_OTSU)` for uint8: (threshold, mask).
#[pyfunction]
#[pyo3(signature = (src, maxval=255, invert=false))]
fn threshold_otsu<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    maxval: u8,
    invert: bool,
) -> PyResult<(f64, Bound<'py, PyArrayDyn<u8>>)> {
    let shape = src.shape().to_vec();
    let data = contiguous(&src)?;
    if data.is_empty() {
        return Err(PyValueError::new_err("threshold of an empty array"));
    }
    let out = empty_u8(py, &shape)?;
    let t = {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            let t = gray::otsu_threshold(data);
            gray::threshold_binary(data, dst, t, maxval, invert);
            t
        })
    };
    Ok((t, out))
}

/// `cv2.Canny(src, low, high)` for a 2-D uint8 image (aperture 3, L1 gradient).
#[pyfunction]
fn canny<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    low: f64,
    high: f64,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = src.shape().to_vec();
    let [h, w] = shape[..] else {
        return Err(PyValueError::new_err("canny needs a 2-D array"));
    };
    let data = contiguous(&src)?;
    let out = empty_u8(py, &shape)?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || gray::canny(data, dst, h, w, low, high))
            .ok_or_else(|| PyValueError::new_err("invalid canny input"))?;
    }
    Ok(out)
}

/// `cv2.medianBlur(src, ksize)` for a single-channel uint8 image.
#[pyfunction]
fn median_blur<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    ksize: usize,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = src.shape().to_vec();
    let (h, w) = match shape.as_slice() {
        [h, w] | [h, w, 1] => (*h, *w),
        _ => {
            return Err(PyValueError::new_err(
                "median_blur needs a single-channel array",
            ));
        }
    };
    let data = contiguous(&src)?;
    let out = empty_u8(py, &shape)?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            crate::median::median_blur_u8(data, dst, h, w, ksize)
        })
        .ok_or_else(|| PyValueError::new_err("ksize must be odd"))?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(median_blur, m)?)?;
    m.add_function(wrap_pyfunction!(bgr_to_gray, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_blur_u8, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_otsu, m)?)?;
    m.add_function(wrap_pyfunction!(canny, m)?)
}
