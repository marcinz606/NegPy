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

fn dims(shape: &[usize]) -> PyResult<(usize, usize, usize)> {
    match shape[..] {
        [h, w] => Ok((h, w, 1)),
        [h, w, c] if (1..=4).contains(&c) => Ok((h, w, c)),
        _ => Err(PyValueError::new_err(
            "encode_png needs an HxW or HxWxC array, C 1 to 4",
        )),
    }
}

/// A PNG of an HxW or HxWxC (C 1 to 4) uint8 or uint16 array, zlib `level` 0 to 9, filtered
/// and deflated on every core. No ancillary chunks.
#[pyfunction]
#[pyo3(signature = (img, level=6))]
fn encode_png<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    level: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let png = if let Ok(a) = img.extract::<PyReadonlyArrayDyn<u16>>() {
        let (h, w, cn) = dims(a.shape())?;
        let src = a
            .as_slice()
            .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
        super::detach(py, src.len(), || {
            crate::png::encode(h, w, cn, 16, level, |y, out| {
                for v in &src[y * w * cn..(y + 1) * w * cn] {
                    out.extend_from_slice(&v.to_be_bytes());
                }
            })
        })
    } else {
        let a = img.extract::<PyReadonlyArrayDyn<u8>>()?;
        let (h, w, cn) = dims(a.shape())?;
        let src = a
            .as_slice()
            .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
        super::detach(py, src.len(), || {
            crate::png::encode(h, w, cn, 8, level, |y, out| {
                out.extend_from_slice(&src[y * w * cn..(y + 1) * w * cn])
            })
        })
    };
    Ok(PyBytes::new(py, &png))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_png, m)?)?;
    m.add_function(wrap_pyfunction!(encode_jpeg, m)?)
}
