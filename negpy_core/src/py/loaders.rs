use numpy::{
    PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
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

type RgbIr<'py> = (Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<f32>>);

fn split<'py, T: numpy::Element + Copy + Into<f32> + Sync>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, T>,
    scale: f32,
) -> PyResult<RgbIr<'py>> {
    let [h, w, 4] = img.shape()[..] else {
        return Err(PyValueError::new_err("split_rgb_ir needs an HxWx4 array"));
    };
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let rgb = PyArray3::<f32>::zeros(py, [h, w, 3], false);
    let ir = PyArray2::<f32>::zeros(py, [h, w], false);
    {
        let (mut r, mut i) = (rgb.readwrite(), ir.readwrite());
        let (r, i) = (r.as_slice_mut()?, i.as_slice_mut()?);
        super::detach(py, src.len(), || loaders::split_rgb_ir(src, scale, r, i));
    }
    Ok((rgb, ir))
}

/// The RGB pixels and the fourth-channel plane of an HxWx4 uint8, uint16 or float32 image,
/// scaled to 0..1 by the integer type's maximum and clipped.
#[pyfunction]
fn split_rgb_ir<'py>(py: Python<'py>, img: &Bound<'py, PyAny>) -> PyResult<RgbIr<'py>> {
    if let Ok(a) = img.extract::<PyReadonlyArrayDyn<u16>>() {
        return split(py, a, (1.0f64 / 65535.0) as f32);
    }
    if let Ok(a) = img.extract::<PyReadonlyArrayDyn<u8>>() {
        return split(py, a, (1.0f64 / 255.0) as f32);
    }
    split(py, img.extract::<PyReadonlyArrayDyn<f32>>()?, 1.0)
}

fn levels<'py, T: numpy::Element + Copy + Into<f64> + Sync>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, T>,
    black: [f64; 3],
    denom: [f64; 3],
    lut: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("levels_to_unit needs an HxWx3 array"));
    };
    if lut.as_ref().is_some_and(|t| t.is_empty()) {
        return Err(PyValueError::new_err("needs a non-empty table"));
    }
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let out = PyArray3::<f32>::zeros(py, [h, w, 3], false);
    {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        super::detach(py, dst.len(), || {
            loaders::levels_to_unit(src, lut.as_deref(), black, denom, dst)
        });
    }
    Ok(out)
}

/// `np.clip((lut[v] - black) / denom, 0, 1)` in float64 of an HxWx3 uint8, uint16, float32
/// or float64 image, as float32; `denom` is per channel and nonzero.
#[pyfunction]
#[pyo3(signature = (img, black, denom, lut=None))]
fn levels_to_unit<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    black: [f64; 3],
    denom: [f64; 3],
    lut: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if let Ok(a) = img.extract::<PyReadonlyArrayDyn<u16>>() {
        return levels(py, a, black, denom, lut);
    }
    if let Ok(a) = img.extract::<PyReadonlyArrayDyn<u8>>() {
        return levels(py, a, black, denom, lut);
    }
    if let Ok(a) = img.extract::<PyReadonlyArrayDyn<f32>>() {
        return levels(py, a, black, denom, lut);
    }
    levels(
        py,
        img.extract::<PyReadonlyArrayDyn<f64>>()?,
        black,
        denom,
        lut,
    )
}

fn curve_rows<'a>(curves: &'a PyReadonlyArray2<'_, f32>) -> PyResult<(&'a [f32], usize)> {
    let [3, n] = curves.shape()[..] else {
        return Err(PyValueError::new_err("needs 3 x N curves"));
    };
    let rows = curves
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous curves"))?;
    Ok((rows, n))
}

fn decode_int<'py, T: numpy::Element + Copy + Into<usize> + Sync>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, T>,
    curves: &PyReadonlyArray2<'py, f32>,
    top: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (rows, n) = curve_rows(curves)?;
    let [h, w, c] = img.shape()[..] else {
        return Err(PyValueError::new_err("decode_curves needs an HxWxC array"));
    };
    if c < 3 || n <= top {
        return Err(PyValueError::new_err(
            "needs 3 channels and a curve entry per sample value",
        ));
    }
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let out = PyArray3::<f32>::zeros(py, [h, w, 3], false);
    {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        super::detach(py, dst.len(), || {
            loaders::decode_curves_int(src, c, rows, n, dst)
        });
    }
    Ok(out)
}

/// The first three channels of an HxWxC image through three tone curves (3 x N float32):
/// a lookup for uint8 or uint16 samples, `np.interp` over `grid` for float32 ones.
#[pyfunction]
#[pyo3(signature = (img, curves, grid=None))]
fn decode_curves<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    curves: PyReadonlyArray2<'py, f32>,
    grid: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if let Ok(a) = img.extract::<PyReadonlyArrayDyn<u16>>() {
        return decode_int(py, a, &curves, u16::MAX as usize);
    }
    if let Ok(a) = img.extract::<PyReadonlyArrayDyn<u8>>() {
        return decode_int(py, a, &curves, u8::MAX as usize);
    }
    let img = img.extract::<PyReadonlyArrayDyn<f32>>()?;
    let (rows, n) = curve_rows(&curves)?;
    let grid = grid.ok_or_else(|| PyValueError::new_err("float samples need a grid"))?;
    let grid = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous grid"))?;
    let [h, w, c] = img.shape()[..] else {
        return Err(PyValueError::new_err("decode_curves needs an HxWxC array"));
    };
    if c < 3 || n < 2 || grid.len() != n {
        return Err(PyValueError::new_err(
            "needs 3 channels and a grid of the curves' length",
        ));
    }
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let out = PyArray3::<f32>::zeros(py, [h, w, 3], false);
    {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        super::detach(py, dst.len(), || {
            loaders::decode_curves_f32(src, c, grid, rows, dst)
        });
    }
    Ok(out)
}

/// `_into_gamut` of an (..., 3) float32 image that is not all in 0..1, given its luminance
/// `y = rgb @ w` (one float32 per pixel).
#[pyfunction]
fn into_gamut<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArrayDyn<'py, f32>,
    y: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let (src, y) = match (rgb.as_slice(), y.as_slice()) {
        (Ok(s), Ok(y)) => (s, y),
        _ => return Err(PyValueError::new_err("needs contiguous arrays")),
    };
    if rgb.shape().last() != Some(&3) || y.len() * 3 != src.len() {
        return Err(PyValueError::new_err(
            "needs an (..., 3) image and a luminance per pixel",
        ));
    }
    let out = PyArrayDyn::<f32>::zeros(py, rgb.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        super::detach(py, dst.len(), || loaders::into_gamut(src, y, dst));
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(split_rgb_ir, m)?)?;
    m.add_function(wrap_pyfunction!(levels_to_unit, m)?)?;
    m.add_function(wrap_pyfunction!(decode_curves, m)?)?;
    m.add_function(wrap_pyfunction!(into_gamut, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_linear, m)?)
}
