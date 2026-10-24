use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::color::space;
use crate::altprocess::{self, Cyanotype, Interp, Lith, Sabattier};
use crate::color::LabSpace;

/// The HxWx3 image, and a new output of its shape filled by `f`.
fn run<'py>(
    py: Python<'py>,
    img: &PyReadonlyArrayDyn<'py, f32>,
    f: impl FnOnce(&[f32], (usize, usize), &mut [f32]) -> Option<()> + Send,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || f(src, (h, w), dst))
            .ok_or_else(|| PyValueError::new_err("the image is too small for the blur"))?;
    }
    Ok(out)
}

/// The working Lab space for Lab to RGB: the matrix, the white, eps and kappa.
type Working<'py> = (
    PyReadonlyArray2<'py, f32>,
    PyReadonlyArray2<'py, f32>,
    PyReadonlyArray1<'py, f32>,
    f32,
    f32,
);

fn to_rgb(lab: &Working<'_>) -> PyResult<LabSpace> {
    space(&lab.1, &lab.2, lab.3, lab.4)
}

fn path(xp: [f32; 4], anchors: [(f32, f32); 4]) -> (Interp, Interp) {
    (
        Interp::new(xp, anchors.map(|a| a.0)),
        Interp::new(xp, anchors.map(|a| a.1)),
    )
}

/// `apply_sabattier` once enabled: (re, fold_width, edge_width, strength, d_max) as float32,
/// and the Mackie-line taps (empty: no line).
#[pyfunction]
fn sabattier<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    scalars: (f32, f32, f32, f32, f32),
    k: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let (re, fold_width, edge_width, strength, d_max) = scalars;
    let s = Sabattier {
        re,
        fold_width,
        edge_width,
        strength,
        d_max,
    };
    let k = k
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous taps"))?;
    run(py, &img, |src, dims, dst| {
        altprocess::sabattier(src, dims, &s, k, dst)
    })
}

/// `apply_cyanotype` once enabled. `scalars` is (d_max, over, scale, mid_compress, bleach,
/// bleach_floor, tannin, tannin_restore, tannin_dmax_gain) as float32; then the brown
/// direction, the hue path and the working Lab constants.
#[pyfunction]
fn cyanotype<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    scalars: [f32; 9],
    brown: (f32, f32),
    path_u: [f32; 4],
    anchors: [(f32, f32); 4],
    lab: Working<'py>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [
        d_max,
        over,
        scale,
        mid_compress,
        bleach,
        bleach_floor,
        tannin,
        tannin_restore,
        tannin_dmax_gain,
    ] = scalars;
    let (a_path, b_path) = path(path_u, anchors);
    let c = Cyanotype {
        d_max,
        over,
        scale,
        mid_compress,
        bleach,
        bleach_floor,
        tannin,
        tannin_restore,
        tannin_dmax_gain,
        brown: [brown.0, brown.1],
        a_path,
        b_path,
        lab_eps: lab.3,
        lab_kappa: lab.4,
    };
    let s = to_rgb(&lab)?;
    run(py, &img, |src, _, dst| {
        altprocess::cyanotype(src, &c, s, dst);
        Some(())
    })
}

/// `apply_lith` once enabled. `scalars` is (d_max, over, foot_max, foot_veil, foot_rate, knee,
/// width) as float32; then the hue path and the working Lab constants.
#[pyfunction]
fn lith<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    scalars: [f32; 7],
    path_u: [f32; 4],
    anchors: [(f32, f32); 4],
    lab: Working<'py>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [d_max, over, foot_max, foot_veil, foot_rate, knee, width] = scalars;
    let (a_path, b_path) = path(path_u, anchors);
    let c = Lith {
        d_max,
        over,
        foot_max,
        foot_veil,
        foot_rate,
        knee,
        width,
        a_path,
        b_path,
        lab_eps: lab.3,
        lab_kappa: lab.4,
    };
    let s = to_rgb(&lab)?;
    run(py, &img, |src, _, dst| {
        altprocess::lith(src, &c, s, dst);
        Some(())
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sabattier, m)?)?;
    m.add_function(wrap_pyfunction!(cyanotype, m)?)?;
    m.add_function(wrap_pyfunction!(lith, m)?)
}
