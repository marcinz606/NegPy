use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{register, sift};

fn gray<'a>(img: &'a PyReadonlyArray2<'_, u8>) -> PyResult<(&'a [u8], usize, usize)> {
    let (h, w) = (img.shape()[0], img.shape()[1]);
    let data = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous uint8 image"))?;
    Ok((data, h, w))
}

/// `register_pair(ref_proxy, mov_proxy, min_inliers)`: (2x3 matrix as 6 floats, inliers), or
/// None when the pair has no usable overlap.
#[pyfunction]
fn register_pair(
    py: Python<'_>,
    ref_proxy: PyReadonlyArray2<'_, u8>,
    mov_proxy: PyReadonlyArray2<'_, u8>,
    min_inliers: usize,
) -> PyResult<Option<([f64; 6], usize)>> {
    let (r, m) = (gray(&ref_proxy)?, gray(&mov_proxy)?);
    Ok(super::detach(py, r.0.len() + m.0.len(), || {
        register::register_pair(r, m, min_inliers)
    }))
}

type Detected<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u8>>);

/// `cv2.SIFT_create(nfeatures).detectAndCompute(gray, None)`: an Nx2 float32 array of keypoint
/// positions and the Nx128 uint8 descriptors.
#[pyfunction]
fn sift_detect<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    nfeatures: usize,
) -> PyResult<Detected<'py>> {
    let (data, h, w) = gray(&img)?;
    let (kp, desc) = super::detach(py, data.len(), || {
        sift::detect_and_compute(data, h, w, nfeatures)
    });
    let pts: Vec<Vec<f32>> = kp.iter().map(|k| vec![k.x, k.y]).collect();
    let desc: Vec<Vec<u8>> = desc.iter().map(|d| d.to_vec()).collect();
    let err = |e: numpy::FromVecError| PyValueError::new_err(e.to_string());
    let pts = if pts.is_empty() {
        PyArray2::zeros(py, [0, 2], false)
    } else {
        PyArray2::from_vec2(py, &pts).map_err(err)?
    };
    let desc = if desc.is_empty() {
        PyArray2::zeros(py, [0, sift::DESCR_LEN], false)
    } else {
        PyArray2::from_vec2(py, &desc).map_err(err)?
    };
    Ok((pts, desc))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_pair, m)?)?;
    m.add_function(wrap_pyfunction!(sift_detect, m)?)
}
