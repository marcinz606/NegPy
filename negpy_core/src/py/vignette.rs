use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::vignette;

/// `apply_luminance_vignette(img, strength, feather, roundness)` of a contiguous HxWxC float32 image.
#[pyfunction]
fn apply_luminance_vignette<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    strength: f64,
    feather: f64,
    roundness: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, cn] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWxC image"));
    };
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            vignette::apply_luminance_vignette(src, h, w, cn, strength, feather, roundness, dst)
        });
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_luminance_vignette, m)?)
}
