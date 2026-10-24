use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::finish::{self, Carrier};

/// `apply_carrier(img, width_px, rough, flare, corner, paper, tone)` of an HxWx3 float32 image
/// for width_px > 0, with `carrier_profiles()` as `profiles`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn apply_carrier<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    width_px: f64,
    rough: f64,
    flare: f64,
    corner: f64,
    paper: (f32, f32, f32),
    tone: PyReadonlyArray2<'py, f32>,
    profiles: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    if tone.shape() != [64, 3] || profiles.shape() != [8, 2048] {
        return Err(PyValueError::new_err(
            "needs a 64x3 tone table and 8x2048 profiles",
        ));
    }
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let src = img.as_slice().map_err(not_contiguous)?;
    let tone = tone.as_slice().map_err(not_contiguous)?;
    let c = Carrier {
        width_px,
        rough,
        flare,
        corner,
        paper: paper.into(),
        tone: tone.as_chunks::<3>().0,
        profiles: profiles.as_slice().map_err(not_contiguous)?,
    };
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            finish::apply_carrier(src, (h, w), &c, dst)
        });
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_carrier, m)?)
}
