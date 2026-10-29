use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::color::space;
use crate::lab;

/// `_usm_sharpening_block(img, amount, radius, masking)` of an HxWx3 float32 image: `gain` is
/// `np.float32(amount * 2.5)`, `k` the blur taps, then the working Lab constants.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn usm_sharpen<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    gain: f32,
    masking: f64,
    k: PyReadonlyArray1<'py, f32>,
    to_xyz: PyReadonlyArray2<'py, f32>,
    to_rgb: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let spaces = (
        space(&to_xyz, &white, eps, kappa)?,
        space(&to_rgb, &white, eps, kappa)?,
    );
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let src = img.as_slice().map_err(not_contiguous)?;
    let k = k.as_slice().map_err(not_contiguous)?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            lab::usm_sharpen(src, (h, w), (gain, masking), k, spaces, dst)
        })
        .ok_or_else(|| PyValueError::new_err("the image is too small for the blur"))?;
    }
    Ok(out)
}

/// `apply_chroma_denoise` of an HxWx3 float32 image at a known low-pass `sigma` with blur taps
/// `k`, then the working Lab constants.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn chroma_denoise<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    sigma: f64,
    k: PyReadonlyArray1<'py, f32>,
    to_xyz: PyReadonlyArray2<'py, f32>,
    to_rgb: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let spaces = (
        space(&to_xyz, &white, eps, kappa)?,
        space(&to_rgb, &white, eps, kappa)?,
    );
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let src = img.as_slice().map_err(not_contiguous)?;
    let k = k.as_slice().map_err(not_contiguous)?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            lab::chroma_denoise(src, (h, w), sigma, k, spaces, dst)
        })
        .ok_or_else(|| PyValueError::new_err("the image is too small for the blur"))?;
    }
    Ok(out)
}

type SpreadArgs = ((f32, f32, f32), Vec<(f32, f64)>);

/// `apply_glow_and_halation` of an HxWx3 float32 image. A spread is `((r, g, b) amounts,
/// [(weight, sigma)])`, glow `(spread, threshold)`, each None when off; `lum` is the luminance
/// row, `inv_gamma` the working TRC's.
#[pyfunction]
fn glow_and_halation<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    glow: Option<(SpreadArgs, f64)>,
    halation: Option<SpreadArgs>,
    lum: (f32, f32, f32),
    inv_gamma: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    let spread = |(a, psf): SpreadArgs| lab::Spread {
        amounts: a.into(),
        psf,
    };
    let glow = glow.map(|(s, threshold)| (spread(s), threshold));
    let halation = halation.map(spread);
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            lab::glow_and_halation(src, (h, w), (glow, halation), (lum.into(), inv_gamma), dst)
        })
        .ok_or_else(|| PyValueError::new_err("a blur failed"))?;
    }
    Ok(out)
}

/// `apply_clahe(img, strength)` of an HxWx3 float32 image for strength > 0, then the working
/// Lab constants.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn clahe<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    strength: f64,
    to_xyz: PyReadonlyArray2<'py, f32>,
    to_rgb: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let spaces = (
        space(&to_xyz, &white, eps, kappa)?,
        space(&to_rgb, &white, eps, kappa)?,
    );
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            lab::clahe(src, (h, w), strength, spaces, dst)
        });
    }
    Ok(out)
}

/// `apply_rl_sharpening(img, amount, radius, masking)` of an HxWx3 float32 image for amount > 0:
/// `amount` as float32, the blur taps `k`, the iteration count, the luminance row and RL_EPS.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rl_sharpen<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    amount: f32,
    masking: f64,
    k: PyReadonlyArray1<'py, f32>,
    iterations: usize,
    lum: (f32, f32, f32),
    eps: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let src = img.as_slice().map_err(not_contiguous)?;
    let k = k.as_slice().map_err(not_contiguous)?;
    let params = lab::Rl {
        amount,
        masking,
        iterations,
        lum: lum.into(),
        eps,
    };
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            lab::rl_sharpen(src, (h, w), &params, k, dst)
        })
        .ok_or_else(|| PyValueError::new_err("the image is too small for the blur"))?;
    }
    Ok(out)
}

/// `sharpen_grain` of an HxWx3 float32 linear image with the luminance row `lum`.
#[pyfunction]
fn sharpen_grain<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    lum: (f32, f32, f32),
) -> PyResult<Option<f32>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    Ok(super::detach(py, src.len(), || {
        lab::sharpen_grain(src, (h, w), lum.into())
    }))
}

/// `apply_split_toning` of an HxWx3 float32 image: each tint is (cos, sin) of its hue times 20
/// times its strength, or None when off; then the working Lab constants.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn split_toning<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    shadow: Option<(f64, f64)>,
    highlight: Option<(f64, f64)>,
    to_xyz: PyReadonlyArray2<'py, f32>,
    to_rgb: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    if img.shape().last() != Some(&3) {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    }
    let spaces = (
        space(&to_xyz, &white, eps, kappa)?,
        space(&to_rgb, &white, eps, kappa)?,
    );
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    let tints = (shadow.map(|t| [t.0, t.1]), highlight.map(|t| [t.0, t.1]));
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || lab::split_toning(src, tints, spaces, dst));
    }
    Ok(out)
}

/// `apply_hue_trim` of an HxWx3 float32 image with (cos, sin) of a nonzero angle, then the
/// working Lab constants.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn hue_trim<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    rotation: (f32, f32),
    to_xyz: PyReadonlyArray2<'py, f32>,
    to_rgb: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    if img.shape().last() != Some(&3) {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    }
    let spaces = (
        space(&to_xyz, &white, eps, kappa)?,
        space(&to_rgb, &white, eps, kappa)?,
    );
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || lab::hue_trim(src, rotation, spaces, dst));
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hue_trim, m)?)?;
    m.add_function(wrap_pyfunction!(split_toning, m)?)?;
    m.add_function(wrap_pyfunction!(rl_sharpen, m)?)?;
    m.add_function(wrap_pyfunction!(sharpen_grain, m)?)?;
    m.add_function(wrap_pyfunction!(clahe, m)?)?;
    m.add_function(wrap_pyfunction!(glow_and_halation, m)?)?;
    m.add_function(wrap_pyfunction!(usm_sharpen, m)?)?;
    m.add_function(wrap_pyfunction!(chroma_denoise, m)?)
}
