use numpy::{
    PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::warp::{self, Border, Image, Interp};

fn image<'a>(src: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<Image<'a>> {
    let (h, w, cn) = match src.shape() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => return Err(PyValueError::new_err("needs an HxW or HxWxC image")),
    };
    if h == 0 || w == 0 || cn == 0 {
        return Err(PyValueError::new_err("needs a non-empty image"));
    }
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    Ok(Image { data, h, w, cn })
}

/// A new float32 output of `dh` x `dw` with the source's channel layout, filled by `f`.
fn output<'py>(
    py: Python<'py>,
    src: &PyReadonlyArrayDyn<'py, f32>,
    (dw, dh): (usize, usize),
    f: impl FnOnce(&mut [f32]) + Send,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let mut dims = vec![dh, dw];
    dims.extend_from_slice(&src.shape()[2..]);
    let out = PyArrayDyn::<f32>::zeros(py, dims, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || f(dst));
    }
    Ok(out)
}

fn matrix<const N: usize>(m: &PyReadonlyArray2<'_, f64>, rows: usize) -> PyResult<[f64; N]> {
    let a = m.as_array();
    if a.shape() != [rows, 3] {
        return Err(PyValueError::new_err(format!("needs a {rows}x3 matrix")));
    }
    Ok(std::array::from_fn(|i| a[[i / 3, i % 3]]))
}

/// The interpolation ("nearest", "linear", "cubic") and border ("replicate", "constant").
fn mode(interpolation: &str, border: &str, nearest: bool) -> PyResult<(Interp, Border)> {
    let interp = match interpolation {
        "nearest" if nearest => Interp::Nearest,
        "linear" => Interp::Linear,
        "cubic" => Interp::Cubic,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown interpolation {interpolation:?}"
            )));
        }
    };
    let border = match border {
        "replicate" => Border::Replicate,
        "constant" => Border::Constant,
        _ => return Err(PyValueError::new_err(format!("unknown border {border:?}"))),
    };
    Ok((interp, border))
}

/// `cv2.remap(src, map_x, map_y, interpolation, borderMode)` of a float32 image; a constant
/// border reads zero.
#[pyfunction]
#[pyo3(signature = (src, map_x, map_y, interpolation="linear", border="replicate"))]
fn remap<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    map_x: PyReadonlyArray2<'py, f32>,
    map_y: PyReadonlyArray2<'py, f32>,
    interpolation: &str,
    border: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    let mode = mode(interpolation, border, true)?;
    if map_x.shape() != map_y.shape() {
        return Err(PyValueError::new_err("the maps differ in size"));
    }
    let (dh, dw) = (map_x.shape()[0], map_x.shape()[1]);
    let not_contiguous = |_| PyValueError::new_err("needs contiguous maps");
    let mx = map_x.as_slice().map_err(not_contiguous)?;
    let my = map_y.as_slice().map_err(not_contiguous)?;
    output(py, &src, (dw, dh), |dst| {
        warp::remap(&img, mode, (mx, my), dw, dst)
    })
}

/// `apply_radial_distortion` of a float32 image: the remap with the scale-to-fill `s`, the
/// center and 1 / halfdiag^2, each rounded to float32.
#[pyfunction]
fn remap_radial<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    center: (f32, f32),
    s: f32,
    k1: f32,
    inv_hd2: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    output(py, &src, (img.w, img.h), |dst| {
        warp::remap_radial(&img, center, s, k1, inv_hd2, dst)
    })
}

/// `cv2.warpAffine(src, m, dsize, interpolation, borderMode)` of a float32 image, bilinear or
/// bicubic; a constant border reads zero.
#[pyfunction]
#[pyo3(signature = (src, m, dsize, inverse_map=false, interpolation="linear", border="replicate"))]
fn warp_affine<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    m: PyReadonlyArray2<'py, f64>,
    dsize: (usize, usize),
    inverse_map: bool,
    interpolation: &str,
    border: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    let mode = mode(interpolation, border, false)?;
    let m = matrix::<6>(&m, 2)?;
    output(py, &src, dsize, |dst| {
        warp::warp_affine(&img, mode, m, inverse_map, dsize.0, dst)
    })
}

/// `cv2.warpPerspective(src, m, dsize, interpolation, borderMode)` of a float32 image, bilinear
/// or bicubic; a constant border reads zero.
#[pyfunction]
#[pyo3(signature = (src, m, dsize, inverse_map=false, interpolation="linear", border="replicate"))]
fn warp_perspective<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    m: PyReadonlyArray2<'py, f64>,
    dsize: (usize, usize),
    inverse_map: bool,
    interpolation: &str,
    border: &str,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    let mode = mode(interpolation, border, false)?;
    let m = matrix::<9>(&m, 3)?;
    output(py, &src, dsize, |dst| {
        warp::warp_perspective(&img, mode, m, inverse_map, dsize, dst)
    })
}

/// `cv2.getRotationMatrix2D(center, angle, scale)` as a 2x3 float64 array.
#[pyfunction]
fn rotation_matrix<'py>(
    py: Python<'py>,
    center: (f32, f32),
    angle: f64,
    scale: f64,
) -> Bound<'py, PyArray2<f64>> {
    let m = warp::rotation_matrix(center, angle, scale);
    PyArray2::from_vec2(py, &[m[..3].to_vec(), m[3..].to_vec()]).expect("two rows of three")
}

/// `cv2.getPerspectiveTransform(src, dst)` for two 4x2 float32 point arrays, as 3x3 float64.
#[pyfunction]
fn perspective_transform<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, f32>,
    dst: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let quad = |p: &PyReadonlyArray2<'_, f32>| -> PyResult<[[f32; 2]; 4]> {
        let a = p.as_array();
        if a.shape() != [4, 2] {
            return Err(PyValueError::new_err("needs 4x2 points"));
        }
        Ok(std::array::from_fn(|i| [a[[i, 0]], a[[i, 1]]]))
    };
    let m = warp::perspective_transform(&quad(&src)?, &quad(&dst)?)
        .ok_or_else(|| PyValueError::new_err("the points do not define a perspective transform"))?;
    Ok(
        PyArray2::from_vec2(py, &[m[..3].to_vec(), m[3..6].to_vec(), m[6..].to_vec()])
            .expect("3x3"),
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rotation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(perspective_transform, m)?)?;
    m.add_function(wrap_pyfunction!(remap, m)?)?;
    m.add_function(wrap_pyfunction!(remap_radial, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine, m)?)?;
    m.add_function(wrap_pyfunction!(warp_perspective, m)?)
}
