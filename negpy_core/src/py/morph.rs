use numpy::{
    Element, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::filter::Border;
use crate::morph::{self, Op, Pixel};

fn run<'py, T: Pixel + Element>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    op: Op,
    kernel: morph::Kernel,
    iterations: usize,
    border: Border,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let src: PyReadonlyArrayDyn<'py, T> = src.extract()?;
    let shape = src.shape().to_vec();
    let dims = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => return Err(PyValueError::new_err("needs a 2-D or 3-D array")),
    };
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let out = py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.clone(), dtype))?
        .cast_into::<PyArrayDyn<T>>()?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            if kernel.mask.iter().all(|&v| v != 0) {
                let ksize = (kernel.w, kernel.h);
                morph::morphology(data, dst, dims, op, ksize, iterations, border)
            } else {
                morph::morphology_mask(data, dst, dims, op, kernel, iterations, border)
            }
        })
        .ok_or_else(|| PyValueError::new_err("invalid morphology input"))?;
    }
    Ok(out.into_any())
}

/// `cv2.morphologyEx(src, op, kernel, iterations=..., borderType=...)` for a uint8 or float32
/// image and any kernel, with OpenCV's default anchor and border value.
#[pyfunction]
#[pyo3(signature = (src, op, kernel, iterations=1, border_type=0))]
fn morphology_ex<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    op: i32,
    kernel: PyReadonlyArray2<'py, u8>,
    iterations: usize,
    border_type: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let op =
        Op::from_cv(op).ok_or_else(|| PyValueError::new_err(format!("unsupported op {op}")))?;
    let border = Border::from_cv(border_type)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported border type {border_type}")))?;
    let k = kernel.as_array();
    let cells: Vec<u8> = k.iter().copied().collect();
    let kernel = morph::Kernel {
        mask: &cells,
        w: k.ncols(),
        h: k.nrows(),
    };
    if src.extract::<PyReadonlyArrayDyn<u8>>().is_ok() {
        run::<u8>(py, src, op, kernel, iterations, border, "uint8")
    } else if src.extract::<PyReadonlyArrayDyn<f32>>().is_ok() {
        run::<f32>(py, src, op, kernel, iterations, border, "float32")
    } else {
        Err(PyTypeError::new_err(
            "morphology needs a uint8 or float32 array",
        ))
    }
}

/// `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))` as an h x w uint8 array.
#[pyfunction]
fn ellipse_kernel<'py>(
    py: Python<'py>,
    ksize: (usize, usize),
) -> PyResult<Bound<'py, numpy::PyArray2<u8>>> {
    let (w, h) = ksize;
    if w == 0 || h == 0 {
        return Err(PyValueError::new_err("kernel size must be positive"));
    }
    numpy::PyArray1::from_vec(py, morph::ellipse_kernel(w, h)).reshape([h, w])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for (name, code) in [
        ("MORPH_ERODE", 0),
        ("MORPH_DILATE", 1),
        ("MORPH_OPEN", 2),
        ("MORPH_CLOSE", 3),
        ("MORPH_GRADIENT", 4),
        ("MORPH_TOPHAT", 5),
        ("MORPH_BLACKHAT", 6),
    ] {
        m.add(name, code)?;
    }
    m.add_function(wrap_pyfunction!(morphology_ex, m)?)?;
    m.add_function(wrap_pyfunction!(ellipse_kernel, m)?)
}
