use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::transfer::{self, Transfer};

fn get<'py, T: FromPyObjectOwned<'py>>(p: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    p.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing transfer parameter {key}")))?
        .extract()
        .map_err(Into::into)
}

fn f3<'py>(p: &Bound<'py, PyDict>, key: &str) -> PyResult<[f32; 3]> {
    let v: [f64; 3] = get(p, key)?;
    Ok(v.map(|x| x as f32))
}

fn b3<'py>(p: &Bound<'py, PyDict>, key: &str) -> PyResult<[bool; 3]> {
    get(p, key)
}

fn f1<'py>(p: &Bound<'py, PyDict>, key: &str) -> PyResult<f32> {
    Ok(get::<f64>(p, key)? as f32)
}

/// `apply_transfer_curve(img, params)` of a contiguous HxWx3 float32 image; `params` is the dict
/// `transparency.logic._core_params` builds, every value as NumPy would round it to float32.
#[pyfunction]
fn apply_transfer_curve<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    params: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [_, _, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let p = params;
    let t = Transfer {
        density_range: f1(p, "density_range")?,
        cast_gain: f3(p, "cast_gain")?,
        cast_offset: f3(p, "cast_offset")?,
        cast_on: b3(p, "cast_on")?,
        exposure_offset: f1(p, "exposure_offset")?,
        cmy: f3(p, "cmy")?,
        exposure_on: b3(p, "exposure_on")?,
        contrast: f1(p, "contrast")?,
        contrast_on: get(p, "contrast_on")?,
        pivot: f1(p, "pivot")?,
        shadow_cmy: f3(p, "shadow_cmy")?,
        highlight_cmy: f3(p, "highlight_cmy")?,
        regional_on: b3(p, "regional_on")?,
        wb_c: f1(p, "wb_c")?,
        wb_k: f1(p, "wb_k")?,
        shadow_density: f1(p, "shadow_density")?,
        highlight_density: f1(p, "highlight_density")?,
        zone_on: get(p, "zone_on")?,
        zone_sh: f1(p, "zone_sh")?,
        zone_hi: f1(p, "zone_hi")?,
        zone_k: f1(p, "zone_k")?,
        black_taper: f1(p, "black_taper")?,
        toe: f3(p, "toe")?,
        shoulder: f3(p, "shoulder")?,
        toe_width: f3(p, "toe_width")?,
        shoulder_width: f3(p, "shoulder_width")?,
        toe_knee: f1(p, "toe_knee")?,
        shoulder_knee: f1(p, "shoulder_knee")?,
        sep_k: f3(p, "sep_k")?,
        sep_on: get(p, "sep_on")?,
        cap_level: f1(p, "cap_level")?,
        cap_offset: f1(p, "cap_offset")?,
        cap_softness: f1(p, "cap_softness")?,
        damping: f1(p, "damping")?,
        damping_on: get(p, "damping_on")?,
        ref_spread: f1(p, "ref_spread")?,
        positive_source: get(p, "positive_source")?,
        gain: f1(p, "gain")?,
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
        super::detach(py, dst.len(), || {
            transfer::apply_transfer_curve(src, &t, dst)
        });
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_transfer_curve, m)?)
}
