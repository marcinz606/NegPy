use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::{self, Level};

fn contiguous<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

fn empty<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    dtype: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.to_vec(), dtype))?
        .cast_into::<PyArrayDyn<T>>()?)
}

/// Fills a new array of `shape` and `dtype` from `fill(dst)` with the GIL released.
fn filled<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    dtype: &str,
    fill: impl FnOnce(&mut [T]) + Send,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let out = empty::<T>(py, shape, dtype)?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || fill(dst));
    }
    Ok(out)
}

/// `_to_uint8_jit` of a contiguous float32 array.
#[pyfunction]
fn to_uint8<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "uint8", |d| convert::to_uint(src, d))
}

/// `_to_uint16_jit` of a contiguous float32 array.
#[pyfunction]
fn to_uint16<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u16>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "uint16", |d| convert::to_uint(src, d))
}

/// `uint8_to_float32` of a contiguous uint8 array.
#[pyfunction]
fn uint8_to_float32<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "float32", |d| {
        convert::uint8_to_float32(src, d)
    })
}

/// `uint16_to_float32` of a contiguous uint16 array.
#[pyfunction]
fn uint16_to_float32<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u16>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "float32", |d| {
        convert::uint16_to_float32(src, d)
    })
}

fn luma_to<'py, T: Level + numpy::Element>(
    py: Python<'py>,
    img: &PyReadonlyArrayDyn<'py, f32>,
    dtype: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let (h, w, cn) = match img.shape()[..] {
        [h, w] => (h, w, 1),
        [h, w, c] if c >= 3 => (h, w, c),
        _ => {
            return Err(PyValueError::new_err("needs an HxW or HxWxC array, C >= 3"));
        }
    };
    let src = contiguous(img)?;
    filled(py, &[h, w], dtype, |d| convert::luma_to_uint(src, cn, d))
}

/// `float_to_uint_luma(img, bit_depth)` of a contiguous float32 HxW or HxWxC array.
#[pyfunction]
fn luma_to_uint<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    bit_depth: u8,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(if bit_depth == 16 {
        luma_to::<u16>(py, &img, "uint16")?.into_any()
    } else {
        luma_to::<u8>(py, &img, "uint8")?.into_any()
    })
}

/// `rgb_to_rgba_into(src, dst)`: the first three channels of `src` into the HxWx4 `dst`.
#[pyfunction]
fn rgb_to_rgba_into(
    py: Python<'_>,
    src: PyReadonlyArray3<'_, f32>,
    mut dst: PyReadwriteArray3<'_, f32>,
) -> PyResult<()> {
    let (s, mut d) = (src.as_array(), dst.as_array_mut());
    if s.dim().0 != d.dim().0 || s.dim().1 != d.dim().1 || s.dim().2 < 3 || d.dim().2 != 4 {
        return Err(PyValueError::new_err("needs HxWx3 into HxWx4"));
    }
    let cn = s.dim().2;
    if let (Some(src), Some(dst)) = (s.as_slice(), d.as_slice_mut()) {
        super::detach(py, dst.len(), || convert::rgb_to_rgba(src, cn, dst));
        return Ok(());
    }
    super::detach(py, 0, || {
        for (srow, mut drow) in s.outer_iter().zip(d.outer_iter_mut()) {
            for (sp, mut dp) in srow.outer_iter().zip(drow.outer_iter_mut()) {
                for c in 0..3 {
                    dp[c] = sp[c];
                }
            }
        }
    });
    Ok(())
}

/// `rgba_to_rgb_into(src, dst, oy, ox)`: the RGB of `src[oy:oy+h, ox:ox+w]` into `dst`.
#[pyfunction]
fn rgba_to_rgb_into(
    py: Python<'_>,
    src: PyReadonlyArray3<'_, f32>,
    mut dst: PyReadwriteArray3<'_, f32>,
    oy: usize,
    ox: usize,
) -> PyResult<()> {
    let (s, mut d) = (src.as_array(), dst.as_array_mut());
    let (h, w, dc) = d.dim();
    if oy + h > s.dim().0 || ox + w > s.dim().1 || s.dim().2 < 3 || dc < 3 {
        return Err(PyValueError::new_err("region outside the source"));
    }
    super::detach(py, 0, || {
        for (y, mut drow) in d.outer_iter_mut().enumerate() {
            let srow = s.index_axis(numpy::ndarray::Axis(0), oy + y);
            // A padded readback has whole RGBA rows: copy them as slices when it does.
            if let (Some(sr), Some(dr)) = (srow.as_slice(), drow.as_slice_mut()) {
                let sc = srow.dim().1;
                for (dp, sp) in dr.chunks_exact_mut(dc).zip(sr[ox * sc..].chunks_exact(sc)) {
                    dp[..3].copy_from_slice(&sp[..3]);
                }
                continue;
            }
            for (x, mut dp) in drow.outer_iter_mut().enumerate() {
                for c in 0..3 {
                    dp[c] = srow[[ox + x, c]];
                }
            }
        }
    });
    Ok(())
}

/// `a * b` of two float32 arrays of one shape (`cv2.multiply` without the copies NumPy makes).
#[pyfunction]
fn multiply<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    if a.shape() != b.shape() {
        return Err(PyValueError::new_err("needs arrays of one shape"));
    }
    let (sa, sb) = (contiguous(&a)?, contiguous(&b)?);
    filled(py, a.shape(), "float32", |d| convert::multiply(sa, sb, d))
}

/// `img.mean(axis=2)` of an HxWx3 float32 image.
#[pyfunction]
fn mean_channels<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("mean_channels needs an HxWx3 array"));
    };
    let src = contiguous(&img)?;
    filled(py, &[h, w], "float32", |d| convert::mean_channels(src, d))
}

fn pick<'py, T: numpy::Element + Copy + Send + Sync>(
    py: Python<'py>,
    src: [&Bound<'py, PyAny>; 3],
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let arrays: Vec<PyReadonlyArrayDyn<'py, T>> =
        src.iter().map(|a| a.extract()).collect::<PyResult<_>>()?;
    let shape = arrays[0].shape().to_vec();
    let [_, _, 3] = shape[..] else {
        return Err(PyValueError::new_err("pick_channels needs HxWx3 arrays"));
    };
    if arrays.iter().any(|a| a.shape() != shape.as_slice()) {
        return Err(PyValueError::new_err("needs arrays of one shape"));
    }
    let s = [
        contiguous(&arrays[0])?,
        contiguous(&arrays[1])?,
        contiguous(&arrays[2])?,
    ];
    Ok(filled(py, &shape, dtype, |d| convert::pick_channels(s, d))?.into_any())
}

/// An HxWx3 uint8, uint16 or float32 image whose channel c is channel c of the c-th source.
#[pyfunction]
fn pick_channels<'py>(
    py: Python<'py>,
    r: &Bound<'py, PyAny>,
    g: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let src = [r, g, b];
    if r.extract::<PyReadonlyArrayDyn<f32>>().is_ok() {
        pick::<f32>(py, src, "float32")
    } else if r.extract::<PyReadonlyArrayDyn<u16>>().is_ok() {
        pick::<u16>(py, src, "uint16")
    } else if r.extract::<PyReadonlyArrayDyn<u8>>().is_ok() {
        pick::<u8>(py, src, "uint8")
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "pick_channels needs uint8, uint16 or float32 arrays",
        ))
    }
}

/// Channel `c` of an HxWxC float32 image as a new HxW array.
#[pyfunction]
fn extract_channel<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    c: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, cn] = img.shape()[..] else {
        return Err(PyValueError::new_err(
            "extract_channel needs an HxWxC array",
        ));
    };
    if c >= cn {
        return Err(PyValueError::new_err("channel out of range"));
    }
    let src = contiguous(&img)?;
    filled(py, &[h, w], "float32", |d| {
        convert::extract_channel(src, cn, c, d)
    })
}

/// `np.stack([lum, lum, lum], axis=-1)` of a float32 plane.
#[pyfunction]
fn gray_to_rgb<'py>(
    py: Python<'py>,
    lum: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let mut shape = lum.shape().to_vec();
    shape.push(3);
    let src = contiguous(&lum)?;
    filled(py, &shape, "float32", |d| convert::gray_to_rgb(src, d))
}

/// `a + b` of two float32 arrays of one shape.
#[pyfunction]
fn add<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    if a.shape() != b.shape() {
        return Err(PyValueError::new_err("needs arrays of one shape"));
    }
    let (sa, sb) = (contiguous(&a)?, contiguous(&b)?);
    filled(py, a.shape(), "float32", |d| convert::add(sa, sb, d))
}

/// `img[..., c].any()` of an HxWxC float32 image.
#[pyfunction]
fn channel_any(py: Python<'_>, img: PyReadonlyArrayDyn<'_, f32>, c: usize) -> PyResult<bool> {
    let [_, _, cn] = img.shape()[..] else {
        return Err(PyValueError::new_err("channel_any needs an HxWxC array"));
    };
    if c >= cn {
        return Err(PyValueError::new_err("channel out of range"));
    }
    let src = contiguous(&img)?;
    Ok(super::detach(py, src.len(), || {
        convert::channel_any(src, cn, c)
    }))
}

/// `np.clip(img, 0, 1)` of a float32 array, as a new array.
#[pyfunction]
fn clip01<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "float32", |d| convert::clip_unit(src, d))
}

/// `np.nan_to_num(img, nan=nan, posinf=posinf, neginf=neginf)` of a float32 array.
#[pyfunction]
fn nan_to_num<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    nan: f32,
    posinf: f32,
    neginf: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "float32", |d| {
        convert::nan_to_num(src, d, (nan, posinf, neginf))
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nan_to_num, m)?)?;
    m.add_function(wrap_pyfunction!(gray_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(channel_any, m)?)?;
    m.add_function(wrap_pyfunction!(clip01, m)?)?;
    m.add_function(wrap_pyfunction!(mean_channels, m)?)?;
    m.add_function(wrap_pyfunction!(pick_channels, m)?)?;
    m.add_function(wrap_pyfunction!(extract_channel, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(to_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(to_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(uint8_to_float32, m)?)?;
    m.add_function(wrap_pyfunction!(uint16_to_float32, m)?)?;
    m.add_function(wrap_pyfunction!(luma_to_uint, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_rgba_into, m)?)?;
    m.add_function(wrap_pyfunction!(rgba_to_rgb_into, m)?)
}
