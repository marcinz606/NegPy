use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::filter::{self, Border};

/// A contiguous HxW or HxWxC float32 array.
struct Image<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    dims: (usize, usize, usize),
}

fn image<'a>(src: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<Image<'a>> {
    let shape = src.shape().to_vec();
    let dims = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => return Err(PyValueError::new_err("needs a 2-D or 3-D array")),
    };
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    Ok(Image { data, shape, dims })
}

fn border(code: i32) -> PyResult<Border> {
    Border::from_cv(code)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported border type {code}")))
}

/// Runs `fill` on a new `numpy.empty` array shaped like the input, with the GIL released.
/// `numpy.empty` neither zeroes the memory nor faults in 4 KiB pages one at a time.
fn filled<'py>(
    py: Python<'py>,
    shape: &[usize],
    fill: impl FnOnce(&mut [f32]) -> Option<()> + Send,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let out = py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.to_vec(), "float32"))?
        .cast_into::<PyArrayDyn<f32>>()?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || fill(dst))
            .ok_or_else(|| PyValueError::new_err("kernel shape not covered by negpy_core"))?;
    }
    Ok(out)
}

/// `cv2.GaussianBlur(src, ksize, sigma_x, sigmaY=sigma_y, borderType=border)` for float32.
#[pyfunction]
#[pyo3(signature = (src, ksize, sigma_x, sigma_y=0.0, border_type=4))]
fn gaussian_blur<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    ksize: (usize, usize),
    sigma_x: f64,
    sigma_y: f64,
    border_type: i32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    let b = border(border_type)?;
    filled(py, &img.shape, |dst| {
        filter::gaussian_blur(img.data, dst, img.dims, ksize, (sigma_x, sigma_y), b)
    })
}

/// `cv2.sepFilter2D(src, -1, kx, ky, borderType=border)` for float32.
#[pyfunction]
#[pyo3(signature = (src, kx, ky, border_type=4))]
fn sep_filter2d<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    kx: PyReadonlyArray1<'py, f32>,
    ky: PyReadonlyArray1<'py, f32>,
    border_type: i32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    let (h, w, cn) = img.dims;
    let b = border(border_type)?;
    let kx = kx
        .as_slice()
        .map_err(|_| PyValueError::new_err("kx must be contiguous"))?
        .to_vec();
    let ky = ky
        .as_slice()
        .map_err(|_| PyValueError::new_err("ky must be contiguous"))?
        .to_vec();
    filled(py, &img.shape, |dst| {
        filter::sep_filter2d(img.data, dst, (h, w, cn), (&kx, &ky), b)
    })
}

/// `cv2.Sobel(src, cv2.CV_32F, dx, dy, ksize=3, borderType=border)` for float32: sepFilter2D
/// with getDerivKernels' 3-tap kernels.
#[pyfunction]
#[pyo3(signature = (src, dx, dy, border_type=4))]
fn sobel<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dx: u8,
    dy: u8,
    border_type: i32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let kernel = |order: u8| match order {
        0 => Ok(vec![1.0f32, 2.0, 1.0]),
        1 => Ok(vec![-1.0f32, 0.0, 1.0]),
        _ => Err(PyValueError::new_err(
            "sobel covers derivative orders 0 and 1",
        )),
    };
    let (kx, ky) = (kernel(dx)?, kernel(dy)?);
    let img = image(&src)?;
    let b = border(border_type)?;
    filled(py, &img.shape, |dst| {
        filter::sep_filter2d(img.data, dst, img.dims, (&kx, &ky), b)
    })
}

/// `cv2.boxFilter(src, -1, ksize, normalize=normalize, borderType=border_type)` for float32.
#[pyfunction]
#[pyo3(signature = (src, ksize, normalize=true, border_type=4))]
fn box_filter<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    ksize: (usize, usize),
    normalize: bool,
    border_type: i32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    let b = border(border_type)?;
    filled(py, &img.shape, |dst| {
        filter::box_filter(img.data, dst, img.dims, ksize, normalize, b)
    })
}

/// `cv2.getGaussianKernel(n, sigma, cv2.CV_32F)` as a flat list.
#[pyfunction]
fn gaussian_kernel(n: usize, sigma: f64) -> Vec<f32> {
    filter::gaussian_kernel(n, sigma)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // OpenCV's border codes, so callers need no cv2 import.
    m.add("BORDER_CONSTANT", 0)?;
    m.add("BORDER_REPLICATE", 1)?;
    m.add("BORDER_REFLECT", 2)?;
    m.add("BORDER_REFLECT_101", 4)?;
    m.add_function(wrap_pyfunction!(box_filter, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(sep_filter2d, m)?)?;
    m.add_function(wrap_pyfunction!(sobel, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_kernel, m)?)
}
