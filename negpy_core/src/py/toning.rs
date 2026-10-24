use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::toning::{self, Bath, Toning};

/// A float32 gain triplet, widened to float64 as the kernel's arithmetic reads it.
fn gain(a: &PyReadonlyArray1<'_, f32>) -> PyResult<[f64; 3]> {
    let g: [f32; 3] = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?
        .try_into()
        .map_err(|_| PyValueError::new_err("a gain needs 3 values"))?;
    Ok(g.map(f64::from))
}

/// `_apply_chemical_toning_jit` of an HxWx3 float32 image; the arguments are the kernel's.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn apply_chemical_toning<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    sel_strength: f64,
    sep_strength: f64,
    gold_strength: f64,
    sel_d_ref: f64,
    sel_power: f64,
    sel_gain: PyReadonlyArray1<'py, f32>,
    sep_d_bleach: f64,
    sep_power: f64,
    sep_gain: PyReadonlyArray1<'py, f32>,
    gold_d_ref: f64,
    gold_power: f64,
    gold_gain: PyReadonlyArray1<'py, f32>,
    gold_sepia_gain: PyReadonlyArray1<'py, f32>,
    blue_strength: f64,
    blue_d_ref: f64,
    blue_power: f64,
    blue_gain: PyReadonlyArray1<'py, f32>,
    copper_strength: f64,
    copper_d_ref: f64,
    copper_power: f64,
    copper_gain: PyReadonlyArray1<'py, f32>,
    van_strength: f64,
    van_d_ref: f64,
    van_power: f64,
    van_gain: PyReadonlyArray1<'py, f32>,
    gold_flat: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    if img.shape().last() != Some(&3) {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    }
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let bath = |strength, d_ref, power, g: &PyReadonlyArray1<'py, f32>| -> PyResult<Bath> {
        Ok(Bath {
            strength,
            d_ref,
            power,
            gain: gain(g)?,
        })
    };
    let t = Toning {
        selenium: bath(sel_strength, sel_d_ref, sel_power, &sel_gain)?,
        sepia: bath(sep_strength, sep_d_bleach, sep_power, &sep_gain)?,
        gold: bath(gold_strength, gold_d_ref, gold_power, &gold_gain)?,
        gold_sepia_gain: gain(&gold_sepia_gain)?,
        blue: bath(blue_strength, blue_d_ref, blue_power, &blue_gain)?,
        copper: bath(copper_strength, copper_d_ref, copper_power, &copper_gain)?,
        vanadium: bath(van_strength, van_d_ref, van_power, &van_gain)?,
        gold_flat: gold_flat > 0.5,
    };
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            toning::apply_chemical_toning(src, dst, &t)
        });
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_chemical_toning, m)?)
}
