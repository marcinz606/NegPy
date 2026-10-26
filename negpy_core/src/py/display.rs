use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::display::{self, Lut3};

fn rgb<'a, T: numpy::Element>(img: &'a PyReadonlyArrayDyn<'_, T>) -> PyResult<&'a [T]> {
    if img.shape().len() != 3 || img.shape()[2] != 3 {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    }
    img.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))
}

fn lut3<'a>(lut: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<Lut3<'a>> {
    let [n, n1, n2, 3] = lut.shape()[..] else {
        return Err(PyValueError::new_err("the LUT must be N x N x N x 3"));
    };
    if n != n1 || n != n2 || n < 2 {
        return Err(PyValueError::new_err("the LUT must be N x N x N x 3"));
    }
    Ok(Lut3 {
        table: lut
            .as_slice()
            .map_err(|_| PyValueError::new_err("needs a contiguous LUT"))?,
        n,
    })
}

fn filled<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    fill: impl FnOnce(&mut [T]) + Send,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let out = PyArrayDyn::<T>::zeros(py, shape, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || fill(dst));
    }
    Ok(out)
}

/// `_apply_lut_u16_jit(img, lut)`.
#[pyfunction]
fn apply_lut_u16<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u16>,
    lut: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u16>>> {
    let (src, l) = (rgb(&img)?, lut3(&lut)?);
    filled(py, img.shape(), |d| display::apply_lut_u16(src, l, d))
}

/// `_apply_lut_f32_jit(img, lut)`.
#[pyfunction]
fn apply_lut_f32<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    lut: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let (src, l) = (rgb(&img)?, lut3(&lut)?);
    filled(py, img.shape(), |d| display::apply_lut_f32(src, l, d))
}

/// `_matrix_trc_u8_jit(img, dec_lut, m, enc_lut)`.
#[pyfunction]
fn matrix_trc_u8<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    dec_lut: PyReadonlyArray2<'py, f32>,
    m: PyReadonlyArray2<'py, f32>,
    enc_lut: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let src = rgb(&img)?;
    if dec_lut.shape() != [3, 256]
        || m.shape() != [3, 3]
        || enc_lut.shape()[0] != 3
        || enc_lut.shape()[1] < 2
    {
        return Err(PyValueError::new_err(
            "needs a 3 x 256 decode, a 3x3 matrix and a 3 x K encode",
        ));
    }
    let dec = dec_lut
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous tables"))?;
    let enc = enc_lut
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous tables"))?;
    let m = m.as_array();
    let m = std::array::from_fn(|r| std::array::from_fn(|c| m[[r, c]]));
    filled(py, img.shape(), |d| {
        display::matrix_trc_u8(src, dec, m, enc, d)
    })
}

/// `matrix_trc_u16(img, dec_lut, m, enc_lut)`: the 16-bit matrix/TRC transform of an HxWx3
/// uint16 image, a 3 x 65536 decode, a 3x3 matrix and a 3 x K encode sampled at t^2.
#[pyfunction]
fn matrix_trc_u16<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u16>,
    dec_lut: PyReadonlyArray2<'py, f32>,
    m: PyReadonlyArray2<'py, f32>,
    enc_lut: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u16>>> {
    let src = rgb(&img)?;
    if dec_lut.shape() != [3, 65536]
        || m.shape() != [3, 3]
        || enc_lut.shape()[0] != 3
        || enc_lut.shape()[1] < 2
    {
        return Err(PyValueError::new_err(
            "needs a 3 x 65536 decode, a 3x3 matrix and a 3 x K encode",
        ));
    }
    let dec = dec_lut
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous tables"))?;
    let enc = enc_lut
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous tables"))?;
    let m = m.as_array();
    let m = std::array::from_fn(|r| std::array::from_fn(|c| m[[r, c]]));
    filled(py, img.shape(), |d| {
        display::matrix_trc_u16(src, dec, m, enc, d)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matrix_trc_u16, m)?)?;
    m.add_function(wrap_pyfunction!(apply_lut_u16, m)?)?;
    m.add_function(wrap_pyfunction!(apply_lut_f32, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_trc_u8, m)?)
}
