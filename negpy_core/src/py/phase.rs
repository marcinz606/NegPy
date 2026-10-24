use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `cv2.phaseCorrelate(a, b, cv2.createHanningWindow(...))` of two same-size HxW float32
/// planes: ((dx, dy), response).
#[pyfunction]
fn phase_correlate(
    py: Python<'_>,
    a: PyReadonlyArrayDyn<'_, f32>,
    b: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<((f64, f64), f64)> {
    let [h, w] = a.shape()[..] else {
        return Err(PyValueError::new_err("phase_correlate needs 2-D arrays"));
    };
    if b.shape() != a.shape() || h < 2 || w < 2 {
        return Err(PyValueError::new_err(
            "phase_correlate needs two same-size planes of at least 2x2",
        ));
    }
    let (Ok(a), Ok(b)) = (a.as_slice(), b.as_slice()) else {
        return Err(PyValueError::new_err("needs contiguous arrays"));
    };
    let (dx, dy, r) = super::detach(py, h * w * 8, || crate::phase::phase_correlate(a, b, h, w));
    Ok(((dx, dy), r))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(phase_correlate, m)?)
}
