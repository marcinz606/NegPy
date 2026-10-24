use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Pillow's `save(format="JPEG", quality=..., subsampling=0, optimize=True, icc_profile=...,
/// dpi=...)` of an HxW or HxWx3 uint8 array, with restart markers every `restart_rows` MCU rows
/// (0 for none, None to split the scan across the cores).
#[pyfunction]
#[pyo3(signature = (img, quality=90, icc=None, dpi=None, restart_rows=None))]
fn encode_jpeg<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    quality: i32,
    icc: Option<&[u8]>,
    dpi: Option<(u16, u16)>,
    restart_rows: Option<usize>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (h, w, cn) = match img.shape()[..] {
        [h, w] => (h, w, 1),
        [h, w, 3] => (h, w, 3),
        _ => {
            return Err(PyValueError::new_err(
                "encode_jpeg needs an HxW or HxWx3 array",
            ));
        }
    };
    let data = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("encode_jpeg needs a contiguous array"))?;
    let options = crate::jpeg::Options {
        quality,
        icc,
        dpi,
        restart_rows,
    };
    let out = super::detach(py, data.len(), || {
        crate::jpeg::encode(data, (h, w, cn), &options)
    })
    .ok_or_else(|| PyValueError::new_err("encode_jpeg: image empty or over 65535 px"))?;
    Ok(PyBytes::new(py, &out))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_jpeg, m)?)
}
