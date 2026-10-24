use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::stitch;

fn slices<'a, T: numpy::Element>(
    arrays: &'a [PyReadonlyArrayDyn<'_, T>],
    n: usize,
) -> PyResult<Vec<&'a [T]>> {
    arrays
        .iter()
        .map(|a| {
            let s = a
                .as_slice()
                .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
            if s.len() != n {
                return Err(PyValueError::new_err("array sizes disagree"));
            }
            Ok(s)
        })
        .collect()
}

/// `gain_compensate`'s gains for `moving` against `reference`, or None to leave it.
#[pyfunction]
fn compensation_gains(
    py: Python<'_>,
    reference: PyReadonlyArrayDyn<'_, f32>,
    ref_mask: PyReadonlyArrayDyn<'_, bool>,
    moving: PyReadonlyArrayDyn<'_, f32>,
    mov_mask: PyReadonlyArrayDyn<'_, bool>,
    min_overlap: usize,
) -> PyResult<Option<[f32; 3]>> {
    let [h, w, 3] = reference.shape()[..] else {
        return Err(PyValueError::new_err("needs HxWx3 parts"));
    };
    let n = h * w;
    let images = [reference, moving];
    let coverage = [ref_mask, mov_mask];
    let imgs = slices(&images, 3 * n)?;
    let masks = slices(&coverage, n)?;
    Ok(super::detach(py, n, || {
        stitch::compensation_gains(imgs[0], masks[0], imgs[1], masks[1], min_overlap)
    }))
}

/// `cv2.detail_DpSeamFinder("COLOR").find(images, [(0, 0)] * n, masks)` of HxWx3 float32 parts
/// on one canvas: the cut masks, as new uint8 arrays.
#[pyfunction]
fn dp_seams<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArrayDyn<'py, f32>>,
    masks: Vec<PyReadonlyArrayDyn<'py, u8>>,
) -> PyResult<Vec<Bound<'py, numpy::PyArrayDyn<u8>>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let Some(&[h, w, 3]) = images.first().map(|a| a.shape()) else {
        return Err(PyValueError::new_err("needs HxWx3 parts"));
    };
    if images.len() != masks.len() {
        return Err(PyValueError::new_err("needs one mask per part"));
    }
    let imgs = slices(&images, 3 * h * w)?;
    let mut cut: Vec<Vec<u8>> = slices(&masks, h * w)?
        .into_iter()
        .map(<[u8]>::to_vec)
        .collect();
    super::detach(py, h * w, || crate::seam::dp_seams(&imgs, &mut cut, (h, w)));
    cut.into_iter()
        .map(|m| m.into_pyarray(py).reshape(vec![h, w]))
        .collect()
}

type BlendInputs<'a> = (Vec<&'a [f32]>, Vec<&'a [bool]>, (usize, usize));

fn blend_inputs<'a>(
    warped: &'a [PyReadonlyArrayDyn<'_, f32>],
    masks: &'a [PyReadonlyArrayDyn<'_, bool>],
) -> PyResult<BlendInputs<'a>> {
    let Some(&[h, w, 3]) = warped.first().map(|a| a.shape()) else {
        return Err(PyValueError::new_err("needs HxWx3 parts"));
    };
    if warped.len() != masks.len() {
        return Err(PyValueError::new_err("needs one mask per part"));
    }
    Ok((slices(warped, 3 * h * w)?, slices(masks, h * w)?, (h, w)))
}

/// `feather_blend(warped, masks)` of HxWx3 float32 parts and HxW bool masks.
#[pyfunction]
fn feather_blend<'py>(
    py: Python<'py>,
    warped: Vec<PyReadonlyArrayDyn<'py, f32>>,
    masks: Vec<PyReadonlyArrayDyn<'py, bool>>,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let (imgs, m, (h, w)) = blend_inputs(&warped, &masks)?;
    let out = super::detach(py, h * w, || stitch::feather_blend(&imgs, &m, (h, w)));
    out.into_pyarray(py).reshape(vec![h, w, 3])
}

/// `seam_blend(warped, masks, seams, feather_px)` with the band already floored at 1 px.
#[pyfunction]
fn seam_blend<'py>(
    py: Python<'py>,
    warped: Vec<PyReadonlyArrayDyn<'py, f32>>,
    masks: Vec<PyReadonlyArrayDyn<'py, bool>>,
    seams: Vec<PyReadonlyArrayDyn<'py, bool>>,
    feather: f64,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let (imgs, m, (h, w)) = blend_inputs(&warped, &masks)?;
    if seams.len() != masks.len() {
        return Err(PyValueError::new_err("needs one seam per part"));
    }
    let s = slices(&seams, h * w)?;
    let out = super::detach(py, h * w, || {
        stitch::seam_blend(&imgs, &m, &s, (h, w), feather)
    });
    out.into_pyarray(py).reshape(vec![h, w, 3])
}

fn rw<'a, T: numpy::Element>(
    a: &'a mut numpy::PyReadwriteArrayDyn<'_, T>,
) -> PyResult<&'a mut [T]> {
    a.as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

/// `build_proxy`'s uint8 stretch of a float32 plane.
#[pyfunction]
fn proxy_stretch<'py>(
    py: Python<'py>,
    gray: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<u8>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let shape = gray.shape().to_vec();
    let src = gray
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let out = super::detach(py, src.len(), || {
        let mut d = vec![0u8; src.len()];
        stitch::proxy_stretch(src, &mut d);
        d
    });
    out.into_pyarray(py).reshape(shape)
}

/// One part of `blend_ir` into `acc` in place: where `sel`, the part's value, or the larger
/// value with `take_max`.
#[pyfunction]
fn ir_blend_into(
    py: Python<'_>,
    mut acc: numpy::PyReadwriteArrayDyn<'_, f32>,
    warped: PyReadonlyArrayDyn<'_, f32>,
    sel: PyReadonlyArrayDyn<'_, bool>,
    take_max: bool,
) -> PyResult<()> {
    let n = acc.len();
    let w = slices(std::slice::from_ref(&warped), n)?[0];
    let s = slices(std::slice::from_ref(&sel), n)?[0];
    let a = rw(&mut acc)?;
    super::detach(py, n, || stitch::ir_blend_into(a, w, s, take_max));
    Ok(())
}

/// `blend_ir`'s fill of uncovered canvas pixels (below 0) with 1.0, in place.
#[pyfunction]
fn ir_finish(py: Python<'_>, mut acc: numpy::PyReadwriteArrayDyn<'_, f32>) -> PyResult<()> {
    let a = rw(&mut acc)?;
    super::detach(py, a.len(), || stitch::ir_finish(a));
    Ok(())
}

/// `(a & b).sum()` of two same-shape bool masks.
#[pyfunction]
fn mask_overlap(
    py: Python<'_>,
    a: PyReadonlyArrayDyn<'_, bool>,
    b: PyReadonlyArrayDyn<'_, bool>,
) -> PyResult<usize> {
    let n = a.len();
    if b.shape() != a.shape() {
        return Err(PyValueError::new_err("masks differ in shape"));
    }
    let masks = [a, b];
    let m = slices(&masks, n)?;
    Ok(super::detach(py, n, || stitch::mask_overlap(m[0], m[1])))
}

/// `img *= gains` per channel of an HxWx3 float32 image, in place.
#[pyfunction]
fn scale_channels(
    py: Python<'_>,
    mut img: numpy::PyReadwriteArrayDyn<'_, f32>,
    gains: [f32; 3],
) -> PyResult<()> {
    let [_, _, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("scale_channels needs an HxWx3 array"));
    };
    let d = rw(&mut img)?;
    super::detach(py, d.len(), || stitch::scale_channels(d, gains));
    Ok(())
}

/// `np.clip(img, 0, 1, out=img)` of a float32 array.
#[pyfunction]
fn clip_unit(py: Python<'_>, mut img: numpy::PyReadwriteArrayDyn<'_, f32>) -> PyResult<()> {
    let d = rw(&mut img)?;
    super::detach(py, d.len(), || stitch::clip_unit(d));
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(proxy_stretch, m)?)?;
    m.add_function(wrap_pyfunction!(ir_blend_into, m)?)?;
    m.add_function(wrap_pyfunction!(ir_finish, m)?)?;
    m.add_function(wrap_pyfunction!(mask_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(scale_channels, m)?)?;
    m.add_function(wrap_pyfunction!(clip_unit, m)?)?;
    m.add_function(wrap_pyfunction!(feather_blend, m)?)?;
    m.add_function(wrap_pyfunction!(seam_blend, m)?)?;
    m.add_function(wrap_pyfunction!(dp_seams, m)?)?;
    m.add_function(wrap_pyfunction!(compensation_gains, m)?)
}
