use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::color::{self, LabSpace};

fn filled<'py>(
    py: Python<'py>,
    src: &PyReadonlyArrayDyn<'py, f32>,
    fill: impl FnOnce(&[f32], &mut [f32]) + Send,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let out = PyArrayDyn::<f32>::zeros(py, src.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || fill(data, dst));
    }
    Ok(out)
}

/// `_oetf_encode_flat` of a contiguous float32 array.
#[pyfunction]
fn oetf_encode<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    inv_gamma: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    filled(py, &img, |s, d| color::oetf_encode(s, d, inv_gamma))
}

/// `_oetf_decode_flat` of a contiguous float32 array.
#[pyfunction]
fn oetf_decode<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    gamma: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    filled(py, &img, |s, d| color::oetf_decode(s, d, gamma))
}

/// `srgb_to_linear` of a float32 array, any shape.
#[pyfunction]
fn srgb_to_linear<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    filled(py, &img, color::srgb_to_linear)
}

pub(crate) fn space(
    m: &PyReadonlyArray2<'_, f32>,
    white: &PyReadonlyArray1<'_, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<LabSpace> {
    let m = m.as_array();
    let white = white.as_array();
    if m.shape() != [3, 3] || white.len() != 3 {
        return Err(PyValueError::new_err(
            "needs a 3x3 matrix and a 3-vector white",
        ));
    }
    Ok(LabSpace::new(
        std::array::from_fn(|r| std::array::from_fn(|c| m[[r, c]])),
        std::array::from_fn(|c| white[c]),
        eps,
        kappa,
    ))
}

fn triplets(img: &PyReadonlyArrayDyn<'_, f32>) -> PyResult<()> {
    if img.shape().last() != Some(&3) {
        return Err(PyValueError::new_err("needs a last axis of 3"));
    }
    Ok(())
}

/// `_rgb_to_lab_kernel` of an (..., 3) float32 array.
#[pyfunction]
fn rgb_to_lab<'py>(
    py: Python<'py>,
    px: PyReadonlyArrayDyn<'py, f32>,
    m: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    triplets(&px)?;
    let s = space(&m, &white, eps, kappa)?;
    filled(py, &px, |src, d| color::rgb_to_lab(src, d, s))
}

/// `_lab_to_rgb_kernel` of an (..., 3) float32 array.
#[pyfunction]
fn lab_to_rgb<'py>(
    py: Python<'py>,
    lab: PyReadonlyArrayDyn<'py, f32>,
    m: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    triplets(&lab)?;
    let s = space(&m, &white, eps, kappa)?;
    filled(py, &lab, |src, d| color::lab_to_rgb(src, d, s))
}

/// `_skin_chroma_rein_kernel` of an (..., 3) float32 Lab array.
#[pyfunction]
fn skin_chroma_rein<'py>(
    py: Python<'py>,
    lab: PyReadonlyArrayDyn<'py, f32>,
    strength: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    triplets(&lab)?;
    filled(py, &lab, |src, d| color::skin_chroma_rein(src, d, strength))
}

/// `_skin_weight(l, a, b)`.
#[pyfunction]
fn skin_weight(l: f32, a: f32, b: f32) -> f64 {
    color::skin_weight_of(l, a, b)
}

/// `_matmul_3x3_kernel` of an (..., 3) float32 array.
#[pyfunction]
fn matmul_3x3<'py>(
    py: Python<'py>,
    px: PyReadonlyArrayDyn<'py, f32>,
    m: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    triplets(&px)?;
    let m = m.as_array();
    if m.shape() != [3, 3] {
        return Err(PyValueError::new_err("needs a 3x3 matrix"));
    }
    let m = std::array::from_fn(|r| std::array::from_fn(|c| m[[r, c]]));
    filled(py, &px, |src, d| color::matmul_3x3(src, d, m))
}

/// `_gamut_aware_chroma_scale_kernel` of an (..., 3) float32 Lab array.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn gamut_aware_chroma_scale<'py>(
    py: Python<'py>,
    lab: PyReadonlyArrayDyn<'py, f32>,
    saturation: f32,
    m: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
    iters: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    triplets(&lab)?;
    let s = space(&m, &white, eps, kappa)?;
    filled(py, &lab, |src, d| {
        color::gamut_aware_chroma_scale(src, d, saturation, iters, s)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(oetf_encode, m)?)?;
    m.add_function(wrap_pyfunction!(oetf_decode, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_lab, m)?)?;
    m.add_function(wrap_pyfunction!(lab_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(skin_chroma_rein, m)?)?;
    m.add_function(wrap_pyfunction!(skin_weight, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_3x3, m)?)?;
    m.add_function(wrap_pyfunction!(srgb_to_linear, m)?)?;
    m.add_function(wrap_pyfunction!(gamut_aware_chroma_scale, m)?)
}
