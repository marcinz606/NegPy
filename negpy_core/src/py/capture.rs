use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::capture::{self, Stretch};
use crate::stats::{np_percentiles, np_percentiles_f64};

fn contiguous<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

fn dims(shape: &[usize]) -> PyResult<(usize, usize, usize)> {
    match shape {
        [h, w] => Ok((*h, *w, 1)),
        [h, w, c] if *c > 0 => Ok((*h, *w, *c)),
        _ => Err(PyValueError::new_err("needs an HxW or HxWxC array")),
    }
}

/// The scan preview stretch of a float32 image: `mode` "positive", "negative" (`preview_positive`)
/// or "prescan" (`_preview_u8`, always three channels out).
#[pyfunction]
fn stretch_preview<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    mode: &str,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let (h, w, cn) = dims(img.shape())?;
    let mode = match mode {
        "positive" => Stretch::Positive,
        "negative" => Stretch::Negative,
        "prescan" => Stretch::Prescan,
        _ => {
            return Err(PyValueError::new_err(
                "mode must be positive, negative or prescan",
            ));
        }
    };
    let out_cn = if mode == Stretch::Prescan { 3 } else { cn };
    let src = contiguous(&img)?;
    let out = PyArrayDyn::<u8>::zeros(py, vec![h, w, out_cn], false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, src.len(), || {
            capture::stretch_preview(src, cn, out_cn, mode, dst)
        });
    }
    Ok(out)
}

/// `sharpness` of an HxW float32 frame.
#[pyfunction]
fn sharpness(py: Python<'_>, gray: PyReadonlyArrayDyn<'_, f32>, min_mean: f64) -> PyResult<f64> {
    let [h, w] = gray.shape()[..] else {
        return Err(PyValueError::new_err("sharpness needs a 2-D array"));
    };
    if h < 3 || w < 3 {
        return Ok(0.0);
    }
    let src = contiguous(&gray)?;
    Ok(super::detach(py, src.len(), || {
        capture::sharpness(src, h, w, min_mean)
    }))
}

/// The share of pixels of an HxWxC float32 image where any of the first three channels is at least `level`.
#[pyfunction]
fn any_channel_at_least(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    level: f32,
) -> PyResult<f64> {
    let (_, _, cn) = dims(img.shape())?;
    let src = contiguous(&img)?;
    if src.is_empty() {
        return Ok(f64::NAN);
    }
    Ok(super::detach(py, src.len(), || {
        capture::any_channel_at_least(src, cn, level)
    }))
}

/// `(np.mean(patch >= level), np.percentile(patch, q))` of a uint16, float32 or float64 patch, the
/// percentile in float32 for float32 and in float64 otherwise, as NumPy computes them.
#[pyfunction]
fn patch_level_stats(
    py: Python<'_>,
    patch: &Bound<'_, PyAny>,
    level: f64,
    q: f64,
) -> PyResult<(f64, f64)> {
    fn stats(v: &[f32], level: f32, q: f64, wide: bool) -> (f64, f64) {
        let hits = v.iter().filter(|&&x| x >= level).count();
        let p = if wide {
            np_percentiles_f64(v, &[q])[0]
        } else {
            np_percentiles(v, &[q]).map_or(f64::NAN, |p| p[0] as f64)
        };
        (hits as f64 / v.len() as f64, p)
    }
    if let Ok(a) = patch.extract::<PyReadonlyArrayDyn<u16>>() {
        let src = contiguous(&a)?;
        if src.is_empty() {
            return Ok((0.0, 0.0));
        }
        Ok(super::detach(py, src.len(), || {
            let v: Vec<f32> = src.iter().map(|&x| x as f32).collect();
            stats(&v, level as f32, q, true)
        }))
    } else if let Ok(a) = patch.extract::<PyReadonlyArrayDyn<f32>>() {
        let src = contiguous(&a)?;
        if src.is_empty() {
            return Ok((0.0, 0.0));
        }
        Ok(super::detach(py, src.len(), || {
            stats(src, level as f32, q, false)
        }))
    } else if let Ok(a) = patch.extract::<PyReadonlyArrayDyn<f64>>() {
        let src = contiguous(&a)?;
        if src.is_empty() {
            return Ok((0.0, 0.0));
        }
        Ok(super::detach(py, src.len(), || {
            let hits = src.iter().filter(|&&x| x >= level).count();
            let p = if src.iter().any(|v| v.is_nan()) {
                f64::NAN
            } else {
                crate::stats::quantile_f64(&mut src.to_vec(), q / 100.0).unwrap_or(f64::NAN)
            };
            (hits as f64 / src.len() as f64, p)
        }))
    } else {
        Err(PyTypeError::new_err(
            "patch_level_stats needs uint16, float32 or float64",
        ))
    }
}

/// `_to_grey` of a uint8 or uint16 HxWxC image.
#[pyfunction]
fn to_grey<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    fn run<'py, T: numpy::Element + Copy + Send + Sync + Into<u32> + TryFrom<u32>>(
        py: Python<'py>,
        rgb: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a: PyReadonlyArrayDyn<'py, T> = rgb.extract()?;
        let (h, w, cn) = dims(a.shape())?;
        let src = contiguous(&a)?;
        let out = PyArrayDyn::<T>::zeros(py, vec![h, w], false);
        {
            let mut rw = out.readwrite();
            let dst = rw
                .as_slice_mut()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            super::detach(py, src.len(), || capture::to_grey(src, cn, dst));
        }
        Ok(out.into_any())
    }
    if rgb.extract::<PyReadonlyArrayDyn<u8>>().is_ok() {
        run::<u8>(py, rgb)
    } else if rgb.extract::<PyReadonlyArrayDyn<u16>>().is_ok() {
        run::<u16>(py, rgb)
    } else {
        Err(PyTypeError::new_err("to_grey needs uint8 or uint16"))
    }
}

/// `np.clip(img.astype(np.float32) / divisor, 0, 1)` of a uint8, uint16, float32 or float64 array.
#[pyfunction]
fn normalized<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    divisor: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    fn run<'py, T: numpy::Element + Copy + Send + Sync + Into<f64>>(
        py: Python<'py>,
        img: &Bound<'py, PyAny>,
        divisor: f32,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let a: PyReadonlyArrayDyn<'py, T> = img.extract()?;
        let src = contiguous(&a)?;
        let out = PyArrayDyn::<f32>::zeros(py, a.shape().to_vec(), false);
        {
            let mut rw = out.readwrite();
            let dst = rw
                .as_slice_mut()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            super::detach(py, src.len(), || capture::normalized(src, divisor, dst));
        }
        Ok(out)
    }
    if img.extract::<PyReadonlyArrayDyn<u16>>().is_ok() {
        run::<u16>(py, img, divisor)
    } else if img.extract::<PyReadonlyArrayDyn<f32>>().is_ok() {
        run::<f32>(py, img, divisor)
    } else if img.extract::<PyReadonlyArrayDyn<u8>>().is_ok() {
        run::<u8>(py, img, divisor)
    } else if img.extract::<PyReadonlyArrayDyn<f64>>().is_ok() {
        run::<f64>(py, img, divisor)
    } else {
        Err(PyTypeError::new_err(
            "normalized needs uint8, uint16, float32 or float64",
        ))
    }
}

/// `meter_frame`'s picture statistics of an HxWx3 float32 frame over (y1, y2, x1, x2):
/// (clipped fraction, blown, 99.9th percentile per channel of the film pixels).
#[pyfunction]
fn meter_picture(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    roi: (usize, usize, usize, usize),
    sat: f32,
    blown_share: f64,
) -> PyResult<(f64, bool, [f32; 3])> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("meter_picture needs an HxWx3 array"));
    };
    let (y1, y2, x1, x2) = roi;
    if y1 >= y2 || x1 >= x2 || y2 > h || x2 > w {
        return Err(PyValueError::new_err("roi outside the frame or empty"));
    }
    let src = contiguous(&img)?;
    Ok(super::detach(py, (y2 - y1) * (x2 - x1) * 3, || {
        capture::meter_picture(src, w, roi, sat, blown_share)
    }))
}

/// `raw_channel_clip_fraction` of a uint16 ROI and its uint8 CFA colours, keeping the colours in
/// `wanted`: (linearity fraction or 0 without a threshold, plateau fraction); None for no sites.
#[pyfunction]
#[pyo3(signature = (img, colors, wanted, threshold, guard, window, min_sites, neighbours, ratio))]
#[allow(clippy::too_many_arguments)]
fn raw_clip_fractions(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, u16>,
    colors: PyReadonlyArrayDyn<'_, u8>,
    wanted: Vec<u8>,
    threshold: Option<u16>,
    guard: u32,
    window: usize,
    min_sites: usize,
    neighbours: usize,
    ratio: f64,
) -> PyResult<Option<(f64, f64)>> {
    if img.shape() != colors.shape() {
        return Err(PyValueError::new_err("image and colours differ in shape"));
    }
    let (v, c) = (contiguous(&img)?, contiguous(&colors)?);
    Ok(super::detach(py, v.len(), || {
        let values: Vec<u16> = v
            .iter()
            .zip(c)
            .filter(|(_, k)| wanted.contains(k))
            .map(|(&x, _)| x)
            .collect();
        (!values.is_empty()).then(|| {
            capture::raw_clip_fractions(
                &values,
                threshold,
                guard,
                (window, min_sites, neighbours, ratio),
            )
        })
    }))
}

/// `_gutter_scan`'s column profile of an HxW float32 plane: every `step`th row over `peak`, averaged per column.
#[pyfunction]
fn column_means<'py>(
    py: Python<'py>,
    a: PyReadonlyArrayDyn<'py, f32>,
    step: usize,
    peak: f32,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let [_, w] = a.shape()[..] else {
        return Err(PyValueError::new_err("column_means needs a 2-D array"));
    };
    let src = contiguous(&a)?;
    let out = super::detach(py, src.len() / step.max(1), || {
        capture::column_means(src, w, step, peak)
    });
    Ok(numpy::PyArray1::from_vec(py, out))
}

/// `_looks_border_crushed`'s medians of an HxW float32 sample: (ring `m` wide, central half).
#[pyfunction]
fn ring_inner_medians(
    py: Python<'_>,
    arr: PyReadonlyArrayDyn<'_, f32>,
    m: usize,
) -> PyResult<(f32, f32)> {
    let [h, w] = arr.shape()[..] else {
        return Err(PyValueError::new_err(
            "ring_inner_medians needs a 2-D array",
        ));
    };
    if 2 * m > h || 2 * m > w || h < 2 || w < 2 {
        return Err(PyValueError::new_err("ring wider than the sample"));
    }
    let src = contiguous(&arr)?;
    Ok(super::detach(py, src.len(), || {
        capture::ring_inner_medians(src, h, w, m)
    }))
}

/// `print_dmax` of an HxWx3 uint8 thumbnail: its D-max colour, or None without pixels.
#[pyfunction]
fn print_dmax(
    py: Python<'_>,
    rgb: PyReadonlyArrayDyn<'_, u8>,
    share: f64,
) -> PyResult<Option<(i64, i64, i64)>> {
    let src = contiguous(&rgb)?;
    if src.len() % 3 != 0 {
        return Err(PyValueError::new_err("print_dmax needs RGB pixels"));
    }
    let px = src.as_chunks::<3>().0;
    Ok(super::detach(py, src.len(), || capture::print_dmax(px, share)).map(|c| (c[0], c[1], c[2])))
}

/// `_peak_map(np.max(img, axis=2))` of an HxWx3 float32 frame with blocks of f x f.
#[pyfunction]
fn peak_map<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    f: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("peak_map needs an HxWx3 array"));
    };
    let src = contiguous(&img)?;
    let (out, ph, pw) = super::detach(py, src.len(), || capture::peak_map(src, h, w, f.max(1)));
    Ok(numpy::PyArray1::from_vec(py, out)
        .reshape([ph, pw])?
        .into_any())
}

/// `preview_positive`'s log stretch of a linear HxWx3 float32 thumbnail, as uint8.
#[pyfunction]
fn log_stretch_u8<'py>(
    py: Python<'py>,
    linear: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let [h, w, 3] = linear.shape()[..] else {
        return Err(PyValueError::new_err("log_stretch_u8 needs an HxWx3 array"));
    };
    let src = contiguous(&linear)?;
    if src.is_empty() {
        return Err(PyValueError::new_err("log_stretch_u8 needs pixels"));
    }
    let out = PyArrayDyn::<u8>::zeros(py, vec![h, w, 3], false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, src.len(), || capture::log_stretch_u8(src, dst));
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(log_stretch_u8, m)?)?;
    m.add_function(wrap_pyfunction!(peak_map, m)?)?;
    m.add_function(wrap_pyfunction!(normalized, m)?)?;
    m.add_function(wrap_pyfunction!(meter_picture, m)?)?;
    m.add_function(wrap_pyfunction!(raw_clip_fractions, m)?)?;
    m.add_function(wrap_pyfunction!(column_means, m)?)?;
    m.add_function(wrap_pyfunction!(ring_inner_medians, m)?)?;
    m.add_function(wrap_pyfunction!(print_dmax, m)?)?;
    m.add_function(wrap_pyfunction!(stretch_preview, m)?)?;
    m.add_function(wrap_pyfunction!(sharpness, m)?)?;
    m.add_function(wrap_pyfunction!(any_channel_at_least, m)?)?;
    m.add_function(wrap_pyfunction!(patch_level_stats, m)?)?;
    m.add_function(wrap_pyfunction!(to_grey, m)?)
}
