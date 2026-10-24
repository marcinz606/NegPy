use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::resize;

/// `cv2.resize(src, dsize, interpolation=cv2.INTER_AREA)` of a contiguous float32 image.
#[pyfunction]
fn resize_area<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dsize: (usize, usize),
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = src.shape().to_vec();
    let (h, w, cn) = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => {
            return Err(PyValueError::new_err(
                "resize_area needs a 2-D or 3-D array",
            ));
        }
    };
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("resize_area needs a contiguous array"))?;
    let (dw, dh) = dsize;
    let out = super::detach(py, data.len(), || {
        resize::resize_area(data, h, w, cn, dh, dw)
    })
    .ok_or_else(|| {
        PyValueError::new_err("resize_area does not cover a 2x2 two-channel downscale")
    })?;
    let dims: Vec<usize> = if shape.len() == 2 {
        vec![dh, dw]
    } else {
        vec![dh, dw, cn]
    };
    let arr = numpy::ndarray::ArrayD::from_shape_vec(dims, out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// `cv2.resize(src, dsize, interpolation=cv2.INTER_LINEAR)` of a contiguous float32 image, as
/// OpenCV computes it without IPP.
#[pyfunction]
fn resize_linear<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dsize: (usize, usize),
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = src.shape().to_vec();
    let (h, w, cn) = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => {
            return Err(PyValueError::new_err(
                "resize_linear needs a 2-D or 3-D array",
            ));
        }
    };
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("resize_linear needs a contiguous array"))?;
    let (dw, dh) = dsize;
    let mut dims = vec![dh, dw];
    dims.extend_from_slice(&shape[2..]);
    // NumPy's allocator asks for huge pages, so the output faults in far fewer pages.
    let out = PyArrayDyn::<f32>::zeros(py, dims, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            resize::resize_linear(data, (h, w, cn), (dh, dw), false, dst)
        })
        .ok_or_else(|| {
            PyValueError::new_err(
                "resize_linear needs a non-empty size and covers no 2x2 two-channel halving",
            )
        })?;
    }
    Ok(out)
}

/// `cv2.resize(src, dsize, interpolation=cv2.INTER_NEAREST)` of a contiguous uint8 image.
#[pyfunction]
fn resize_nearest<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    dsize: (usize, usize),
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = src.shape().to_vec();
    let (h, w, cn) = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => {
            return Err(PyValueError::new_err(
                "resize_nearest needs a 2-D or 3-D array",
            ));
        }
    };
    let (dw, dh) = dsize;
    if h == 0 || w == 0 || cn == 0 || dh == 0 || dw == 0 {
        return Err(PyValueError::new_err(
            "resize_nearest needs a non-empty size",
        ));
    }
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("resize_nearest needs a contiguous array"))?;
    let mut dims = vec![dh, dw];
    dims.extend_from_slice(&shape[2..]);
    let out = PyArrayDyn::<u8>::zeros(py, dims, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            resize::resize_nearest(data, (h, w, cn), (dh, dw), dst)
        });
    }
    Ok(out)
}

/// `expand_mask_plane(plane, (h, w), roi)` times `scale` for a float32 plane and a clamped,
/// non-empty roi (y1, y2, x1, x2).
#[pyfunction]
fn expand_plane<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    out_shape: (usize, usize),
    roi: (usize, usize, usize, usize),
    scale: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [ph, pw] = plane.shape()[..] else {
        return Err(PyValueError::new_err("expand_plane needs a 2-D plane"));
    };
    let src = plane
        .as_slice()
        .map_err(|_| PyValueError::new_err("expand_plane needs a contiguous plane"))?;
    let (h, w) = out_shape;
    let out = PyArrayDyn::<f32>::zeros(py, vec![h, w], false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            resize::expand_plane(src, (ph, pw), (h, w), roi, scale, dst)
        })
        .ok_or_else(|| {
            PyValueError::new_err("expand_plane needs a non-empty roi inside the output")
        })?;
    }
    Ok(out)
}

/// `cv2.resize(src, dsize, interpolation=cv2.INTER_CUBIC)` (`lanczos=False`) or
/// `cv2.INTER_LANCZOS4` of a contiguous float32 image.
#[pyfunction]
#[pyo3(signature = (src, dsize, lanczos=false))]
fn resize_kernel<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dsize: (usize, usize),
    lanczos: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = src.shape().to_vec();
    let (h, w, cn) = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => {
            return Err(PyValueError::new_err(
                "resize_kernel needs a 2-D or 3-D array",
            ));
        }
    };
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("resize_kernel needs a contiguous array"))?;
    let (dw, dh) = dsize;
    let mut dims = vec![dh, dw];
    dims.extend_from_slice(&shape[2..]);
    let kernel = if lanczos {
        resize::Kernel::Lanczos4
    } else {
        resize::Kernel::Cubic
    };
    let out = PyArrayDyn::<f32>::zeros(py, dims, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            resize::resize_kernel(data, (h, w, cn), (dh, dw), kernel, dst)
        })
        .ok_or_else(|| PyValueError::new_err("resize_kernel needs a non-empty image and size"))?;
    }
    Ok(out)
}

/// `cv2.resize(src, dsize, interpolation=...)` of a contiguous uint8 image: `cubic` False is
/// INTER_AREA, True INTER_CUBIC as OpenCV computes it without IPP.
#[pyfunction]
#[pyo3(signature = (src, dsize, cubic=false))]
fn resize_u8<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, u8>,
    dsize: (usize, usize),
    cubic: bool,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = src.shape().to_vec();
    let (h, w, cn) = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => return Err(PyValueError::new_err("resize_u8 needs a 2-D or 3-D array")),
    };
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("resize_u8 needs a contiguous array"))?;
    let (dw, dh) = dsize;
    let mut dims = vec![dh, dw];
    dims.extend_from_slice(&shape[2..]);
    let out = PyArrayDyn::<u8>::zeros(py, dims, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            if cubic {
                resize::resize_cubic_u8(data, (h, w, cn), (dh, dw), dst)
            } else {
                resize::resize_area_u8(data, (h, w, cn), (dh, dw), dst)
            }
        })
        .ok_or_else(|| PyValueError::new_err("resize_u8 needs a non-empty image and size"))?;
    }
    Ok(out)
}

type RowResize<T> = fn(&[T], (usize, usize, usize), usize, &mut [T]) -> Option<()>;

/// `cv2.resize(src, (w, dh), interpolation=cv2.INTER_LINEAR)` of a contiguous uint8 or uint16
/// image whose width stays, as OpenCV computes it without IPP.
#[pyfunction]
fn resize_rows_linear<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    dh: usize,
) -> PyResult<Bound<'py, PyAny>> {
    fn run<'py, T: numpy::Element + Copy + Send + Sync + Default>(
        py: Python<'py>,
        src: &Bound<'py, PyAny>,
        dh: usize,
        f: RowResize<T>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a: PyReadonlyArrayDyn<'py, T> = src.extract()?;
        let shape = a.shape().to_vec();
        let (h, w, cn) = match shape.as_slice() {
            [h, w] => (*h, *w, 1),
            [h, w, c] => (*h, *w, *c),
            _ => {
                return Err(PyValueError::new_err(
                    "resize_rows_linear needs a 2-D or 3-D array",
                ));
            }
        };
        let data = a
            .as_slice()
            .map_err(|_| PyValueError::new_err("resize_rows_linear needs a contiguous array"))?;
        let mut dims = shape.clone();
        dims[0] = dh;
        let out = PyArrayDyn::<T>::zeros(py, dims, false);
        {
            let mut rw = out.readwrite();
            let dst = rw
                .as_slice_mut()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            super::detach(py, dst.len(), || f(data, (h, w, cn), dh, dst)).ok_or_else(|| {
                PyValueError::new_err("resize_rows_linear needs a non-empty image and size")
            })?;
        }
        Ok(out.into_any())
    }
    if src.extract::<PyReadonlyArrayDyn<u8>>().is_ok() {
        run::<u8>(py, src, dh, resize::resize_rows_linear_u8)
    } else if src.extract::<PyReadonlyArrayDyn<u16>>().is_ok() {
        run::<u16>(py, src, dh, resize::resize_rows_linear_u16)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "resize_rows_linear needs uint8 or uint16",
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_rows_linear, m)?)?;
    m.add_function(wrap_pyfunction!(resize_u8, m)?)?;
    m.add_function(wrap_pyfunction!(resize_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(expand_plane, m)?)?;
    m.add_function(wrap_pyfunction!(resize_nearest, m)?)?;
    m.add_function(wrap_pyfunction!(resize_area, m)?)?;
    m.add_function(wrap_pyfunction!(resize_linear, m)?)
}
