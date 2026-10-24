use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::loaders;

/// `normalize_linear` of a float64 image whose last axis holds the channels.
#[pyfunction]
fn normalize_linear<'py>(
    py: Python<'py>,
    lin: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let shape = lin.shape().to_vec();
    let c = match shape.last() {
        Some(&c) if c > 0 && shape.len() >= 2 => c,
        _ => {
            return Err(PyValueError::new_err(
                "normalize_linear needs an ...xC array",
            ));
        }
    };
    let data = lin
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let out = py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape, "float64"))?
        .cast_into::<PyArrayDyn<f64>>()?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || loaders::normalize_linear(data, c, dst));
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_linear, m)?)
}
