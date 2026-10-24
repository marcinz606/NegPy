use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::stats;

/// `np.percentile(values, q)` for each q, bit for bit, on a contiguous 1-D float32 array.
#[pyfunction]
fn percentiles(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f32>,
    qs: Vec<f64>,
) -> PyResult<Vec<f32>> {
    if qs.iter().any(|q| !(0.0..=100.0).contains(q)) {
        return Err(PyValueError::new_err(
            "Percentiles must be in the range [0, 100]",
        ));
    }
    let values = values
        .as_slice()
        .map_err(|_| PyValueError::new_err("percentiles needs a contiguous array"))?;
    super::detach(py, values.len(), || stats::percentiles(values, &qs))
        .ok_or_else(|| PyIndexError::new_err("percentiles of an empty array"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(percentiles, m)?)
}
