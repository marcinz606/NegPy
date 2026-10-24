use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `_channel_means` of a float32 mosaic and its uint8 CFA color indices.
#[pyfunction]
fn channel_means(
    py: Python<'_>,
    mosaic: PyReadonlyArrayDyn<'_, f32>,
    colors: PyReadonlyArrayDyn<'_, u8>,
    black: f64,
) -> PyResult<(f64, f64, f64)> {
    if mosaic.shape() != colors.shape() {
        return Err(PyValueError::new_err("mosaic and colors differ in shape"));
    }
    let (Ok(m), Ok(c)) = (mosaic.as_slice(), colors.as_slice()) else {
        return Err(PyValueError::new_err("needs contiguous arrays"));
    };
    let r = super::detach(py, m.len(), || crate::rgbscan::channel_means(m, c, black));
    Ok((r[0], r[1], r[2]))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(channel_means, m)?)
}
