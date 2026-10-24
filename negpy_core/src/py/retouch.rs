use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::retouch;

fn plane<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>, n: usize) -> PyResult<&'a [f32]> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    if s.len() != n {
        return Err(PyValueError::new_err("array sizes disagree"));
    }
    Ok(s)
}

/// `_hair_encode(crop, lo, inv_span, out_u8)` of an HxWxC float32 tile.
#[pyfunction]
fn hair_encode(
    py: Python<'_>,
    crop: numpy::PyReadonlyArrayDyn<'_, f32>,
    lo: f32,
    inv_span: f32,
    inv_gamma: f32,
    mut out: numpy::PyReadwriteArrayDyn<'_, u8>,
) -> PyResult<()> {
    use numpy::PyUntypedArrayMethods;
    let [_, _, cn] = crop.shape()[..] else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "needs an HxWxC tile",
        ));
    };
    if cn < 3 || out.shape() != crop.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "needs 3 channels and a matching out",
        ));
    }
    let src = crop
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("needs a contiguous tile"))?;
    let dst = out
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("needs a contiguous out"))?;
    super::detach(py, dst.len(), || {
        crate::retouch::hair_encode(src, cn, lo, inv_span, inv_gamma, dst)
    });
    Ok(())
}

/// `_ir_clean_base(img_det, ratio)` with (window, identity, sigma, min_support).
#[pyfunction]
fn ir_clean_base<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    ratio: PyReadonlyArrayDyn<'py, f32>,
    constants: (usize, f32, f32, f32),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    if ratio.shape() != [h, w] {
        return Err(PyValueError::new_err("needs an HxW ratio"));
    }
    let (src, ratio) = (plane(&img, 3 * h * w)?, plane(&ratio, h * w)?);
    let (win, identity, sigma, min_support) = constants;
    let c = retouch::CleanBase {
        win,
        identity,
        sigma,
        min_support,
    };
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            retouch::ir_clean_base(src, ratio, (h, w), &c, dst)
        })
        .ok_or_else(|| PyValueError::new_err("invalid clean-base input"))?;
    }
    Ok(out)
}

fn plane2<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], (usize, usize))> {
    let [h, w] = a.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxW plane"));
    };
    Ok((plane(a, h * w)?, (h, w)))
}

fn mask<'a>(a: &'a PyReadonlyArrayDyn<'_, bool>, n: usize) -> PyResult<&'a [bool]> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    if s.len() != n {
        return Err(PyValueError::new_err("array sizes disagree"));
    }
    Ok(s)
}

fn bool_array<'py>(
    py: Python<'py>,
    data: Vec<bool>,
    (h, w): (usize, usize),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<bool>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    data.into_pyarray(py).reshape(vec![h, w])
}

/// `_ir_live(plane)` with (dead floor, minimum holder area as a fraction of the frame).
#[pyfunction]
fn ir_live<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    floor: f32,
    min_area: f64,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<bool>>> {
    let (src, dims) = plane2(&plane)?;
    let live = super::detach(py, src.len(), || {
        retouch::ir_live(src, dims, floor, min_area)
    });
    bool_array(py, live, dims)
}

/// `_ir_glare(plane, live)` with (sample cap, sigmas).
#[pyfunction]
fn ir_glare<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    live: PyReadonlyArrayDyn<'py, bool>,
    cap: usize,
    sigmas: f64,
) -> PyResult<Option<Bound<'py, numpy::PyArrayDyn<bool>>>> {
    let (src, dims) = plane2(&plane)?;
    let live = mask(&live, src.len())?;
    super::detach(py, src.len(), || retouch::ir_glare(src, live, cap, sigmas))
        .map(|g| bool_array(py, g, dims))
        .transpose()
}

/// `normalize_ir(plane, glare)` for a float32 plane, with the base window in pixels.
#[pyfunction]
#[pyo3(signature = (plane, glare, win))]
fn normalize_ir<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    glare: Option<PyReadonlyArrayDyn<'py, bool>>,
    win: usize,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let (src, dims) = plane2(&plane)?;
    let glare = glare.as_ref().map(|g| mask(g, src.len())).transpose()?;
    let out = PyArrayDyn::<f32>::zeros(py, plane.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            retouch::normalize_ir(src, glare, dims, win, dst)
        })
        .ok_or_else(|| PyValueError::new_err("invalid normalize_ir input"))?;
    }
    Ok(out)
}

/// `_ir_normalize_ratio(ratio, live)` in place, with (identity, noise sigma, pivot floor,
/// reference sigma, scale cap); returns (pivot, sigma), or None when it left the ratio alone.
#[pyfunction]
fn ir_normalize_ratio(
    py: Python<'_>,
    mut ratio: PyReadwriteArrayDyn<'_, f32>,
    live: PyReadonlyArrayDyn<'_, bool>,
    constants: (f64, f64, f64, f64, f64),
) -> PyResult<Option<(f64, f64)>> {
    let r = ratio
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    let live = mask(&live, r.len())?;
    let (identity, noise_sigma, pivot_lo, ref_sigma, scale_max) = constants;
    let c = retouch::RatioNorm {
        identity,
        noise_sigma,
        pivot_lo,
        ref_sigma,
        scale_max,
    };
    Ok(super::detach(py, r.len(), || {
        retouch::ir_normalize_ratio(r, live, &c)
    }))
}

/// `_ir_decontaminate`'s division of an HxW ratio by the ghost of HxWx3 `vis_log`.
#[pyfunction]
fn ir_ghost_divide<'py>(
    py: Python<'py>,
    ratio: PyReadonlyArrayDyn<'py, f32>,
    vis_log: PyReadonlyArrayDyn<'py, f32>,
    b: [f32; 3],
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let (r, (h, w)) = plane2(&ratio)?;
    if vis_log.shape() != [h, w, 3] {
        return Err(PyValueError::new_err("needs an HxWx3 vis_log"));
    }
    let v = plane(&vis_log, 3 * h * w)?;
    let out = PyArrayDyn::<f32>::zeros(py, ratio.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || retouch::ir_ghost_divide(r, v, b, dst));
    }
    Ok(out)
}

fn image3<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], (usize, usize))> {
    let [h, w, 3] = a.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    Ok((plane(a, 3 * h * w)?, (h, w)))
}

fn new_image<'py>(
    py: Python<'py>,
    shape: &[usize],
    fill: impl FnOnce(&mut [f32]) -> Option<()> + Send,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    // numpy.empty, not zeros: NumPy's allocator asks for huge pages, and every pixel is written.
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
            .ok_or_else(|| PyValueError::new_err("invalid input"))?;
    }
    Ok(out)
}

/// `out[idx] += _borrow_clean_grain(src, clean, sigma, idx)[0]` for every pixel `clean` leaves
/// out, in place.
#[pyfunction]
fn add_clean_grain(
    py: Python<'_>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
    src: PyReadonlyArrayDyn<'_, f32>,
    clean: PyReadonlyArrayDyn<'_, bool>,
    sigma: f64,
) -> PyResult<()> {
    let (s, dims) = image3(&src)?;
    let clean = mask(&clean, dims.0 * dims.1)?;
    if out.shape() != src.shape() {
        return Err(PyValueError::new_err("needs a matching out"));
    }
    let o = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous out"))?;
    super::detach(py, s.len(), || {
        retouch::add_clean_grain(s, clean, dims, sigma, None, o)
    });
    Ok(())
}

/// `apply_score_repair` of an HxWx3 image and a score at its resolution, with the supports, the
/// floor rule, the grain sigma and (score floor, tau, write hi, write hi - write lo).
#[pyfunction]
fn apply_score_repair<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    score: PyReadonlyArrayDyn<'py, f32>,
    supports: Vec<usize>,
    floor: bool,
    sigma: f64,
    constants: (f32, f32, f32, f32),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let (src, dims) = image3(&img)?;
    let sc = plane(&score, dims.0 * dims.1)?;
    let (score_floor, tau, write_hi, write_span) = constants;
    let c = retouch::ScoreRepair {
        score_floor,
        tau,
        write_hi,
        write_span,
    };
    new_image(py, img.shape(), |dst| {
        retouch::apply_score_repair(src, sc, dims, &supports, floor, sigma, &c, dst)
    })
}

/// `repair_components` into `out`, which holds the base, with the supports, the floor rule,
/// the grain sigma, (score floor, tau, write hi, write hi - write lo), the crop pad and the
/// component cap; returns False, leaving `out`, when the score holds no defect.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (img, score, out, supports, floor, sigma, constants, pad, max_components, dirty=None))]
fn repair_components(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    score: PyReadonlyArrayDyn<'_, f32>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
    supports: Vec<usize>,
    floor: bool,
    sigma: f64,
    constants: (f32, f32, f32, f32),
    pad: usize,
    max_components: usize,
    dirty: Option<PyReadonlyArrayDyn<'_, bool>>,
) -> PyResult<bool> {
    let (src, dims) = image3(&img)?;
    let n = dims.0 * dims.1;
    let sc = plane(&score, n)?;
    let dirty = dirty.as_ref().map(|d| mask(d, n)).transpose()?;
    if out.shape() != img.shape() {
        return Err(PyValueError::new_err("needs a matching out"));
    }
    let o = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous out"))?;
    let (score_floor, tau, write_hi, write_span) = constants;
    let c = retouch::ScoreRepair {
        score_floor,
        tau,
        write_hi,
        write_span,
    };
    super::detach(py, src.len(), || {
        retouch::repair_components(
            src,
            sc,
            dims,
            &supports,
            floor,
            sigma,
            &c,
            pad,
            max_components,
            dirty,
            o,
        )
    })
    .ok_or_else(|| PyValueError::new_err("invalid repair input"))
}

/// `route_wide_defects` before its budget check, with the at-floor bar, the radius and the
/// dilate kernel: a uint8 HxW mask, or None.
#[pyfunction]
fn route_wide_defects<'py>(
    py: Python<'py>,
    score: PyReadonlyArrayDyn<'py, f32>,
    bar: f32,
    radius: usize,
    kernel: numpy::PyReadonlyArray2<'py, u8>,
) -> PyResult<Option<Bound<'py, numpy::PyArray2<u8>>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let (sc, (h, w)) = plane2(&score)?;
    let k = kernel.as_array();
    let cells: Vec<u8> = k.iter().copied().collect();
    let kernel = crate::morph::Kernel {
        mask: &cells,
        w: k.ncols(),
        h: k.nrows(),
    };
    super::detach(py, sc.len(), || {
        retouch::route_wide_defects(sc, (h, w), bar, radius, kernel)
    })
    .map(|r| r.into_pyarray(py).reshape([h, w]))
    .transpose()
}

/// `_fit_refraction_gammas(ratio, vis_log, img_det)` with (band low, band high), the flat
/// percentile, the minimum flat count, the sample cap, the fallback and (gamma low, high).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_refraction_gammas(
    py: Python<'_>,
    ratio: PyReadonlyArrayDyn<'_, f32>,
    vis_log: PyReadonlyArrayDyn<'_, f32>,
    img: PyReadonlyArrayDyn<'_, f32>,
    band: (f32, f32),
    flat_pct: f64,
    min_flat: usize,
    cap: usize,
    fallback: f64,
    range: (f32, f32),
) -> PyResult<(f64, f64, f64)> {
    let (r, dims) = plane2(&ratio)?;
    let (v, (h, w)) = image3(&vis_log)?;
    let (src, idims) = image3(&img)?;
    if dims != (h, w) || idims != dims {
        return Err(PyValueError::new_err("array shapes disagree"));
    }
    let c = retouch::GammaFit {
        band,
        flat_pct,
        min_flat,
        cap,
        fallback,
        range,
    };
    let g = super::detach(py, r.len(), || {
        retouch::fit_refraction_gammas(r, v, src, dims, &c)
    })
    .ok_or_else(|| PyValueError::new_err("invalid gamma fit input"))?;
    Ok((g[0], g[1], g[2]))
}

/// `ir_defect_score(ratio, cutoff)` with (cutoff, span, 1 - floor, floor) as float32.
#[pyfunction]
fn ir_defect_score<'py>(
    py: Python<'py>,
    ratio: PyReadonlyArrayDyn<'py, f32>,
    constants: (f32, f32, f32, f32),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let (r, dims) = plane2(&ratio)?;
    new_image(py, ratio.shape(), |dst| {
        retouch::ir_defect_score(r, dims, constants, dst)
    })
}

type Plane<'py> = Bound<'py, numpy::PyArray2<f32>>;

fn plane_array<'py>(py: Python<'py>, v: Vec<f32>, (h, w): (usize, usize)) -> PyResult<Plane<'py>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    v.into_pyarray(py).reshape([h, w])
}

/// `compute_dust_stats` from the density plane, with the median and texture windows, the
/// opening disk and (minimum spread, averaging window, MAD gain, sigma floor).
#[pyfunction]
fn dust_stats<'py>(
    py: Python<'py>,
    dens: PyReadonlyArrayDyn<'py, f32>,
    v_win: usize,
    w_win: usize,
    disk: numpy::PyReadonlyArray2<'py, u8>,
    constants: (f64, usize, f32, f32),
) -> PyResult<(Plane<'py>, Plane<'py>, Plane<'py>, Plane<'py>)> {
    let (d, dims) = plane2(&dens)?;
    let k = disk.as_array();
    let cells: Vec<u8> = k.iter().copied().collect();
    let (min_spread, avg_px, mad_gain, sigma_min) = constants;
    let c = retouch::DustStats {
        v_win,
        w_win,
        disk: crate::morph::Kernel {
            mask: &cells,
            w: k.ncols(),
            h: k.nrows(),
        },
        min_spread,
        avg_px,
        mad_gain,
        sigma_min,
    };
    let [p, b, z, t] = super::detach(py, d.len(), || retouch::dust_stats(d, dims, &c))
        .ok_or_else(|| PyValueError::new_err("invalid dust stats input"))?;
    Ok((
        plane_array(py, p, dims)?,
        plane_array(py, b, dims)?,
        plane_array(py, z, dims)?,
        plane_array(py, t, dims)?,
    ))
}

fn u8_array<'py>(
    py: Python<'py>,
    v: Vec<u8>,
    (h, w): (usize, usize),
) -> PyResult<Bound<'py, numpy::PyArray2<u8>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    v.into_pyarray(py).reshape([h, w])
}

type SplitHairs<'py> = (
    Bound<'py, numpy::PyArray2<u8>>,
    Option<Bound<'py, numpy::PyArray2<u8>>>,
);

/// `split_hairs(mask)` of a uint8 mask with (minimum hair area, minimum elongation).
#[pyfunction]
fn split_hairs<'py>(
    py: Python<'py>,
    mask: numpy::PyReadonlyArray2<'py, u8>,
    hair: (usize, f64),
) -> PyResult<SplitHairs<'py>> {
    let dims = (mask.shape()[0], mask.shape()[1]);
    let m = mask
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous mask"))?;
    let (compact, hairs) = super::detach(py, m.len(), || retouch::split_hairs(m, dims, hair));
    Ok((
        u8_array(py, compact, dims)?,
        hairs.map(|h| u8_array(py, h, dims)).transpose()?,
    ))
}

type LumaScore<'py> = (Option<Plane<'py>>, Option<Bound<'py, numpy::PyArray2<u8>>>);

/// `detect_luma_score` from (proxy, background, z, texture) with (hi, hi_hair, lo), the
/// reach disk, and the constants it reads, in the order of `retouch::LumaDetect`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn detect_luma_score<'py>(
    py: Python<'py>,
    stats: (
        PyReadonlyArrayDyn<'py, f32>,
        PyReadonlyArrayDyn<'py, f32>,
        PyReadonlyArrayDyn<'py, f32>,
        PyReadonlyArrayDyn<'py, f32>,
    ),
    bars: (f64, f64, f64),
    reach_disk: numpy::PyReadonlyArray2<'py, u8>,
    texture: (f32, f32, f32),
    hair: (usize, f64),
    hair_texture: (f64, f64, f32),
    texture_win: usize,
    pad: (f64, f64),
) -> PyResult<LumaScore<'py>> {
    let (proxy, dims) = plane2(&stats.0)?;
    let n = proxy.len();
    let (background, z, tex) = (
        plane(&stats.1, n)?,
        plane(&stats.2, n)?,
        plane(&stats.3, n)?,
    );
    let k = reach_disk.as_array();
    let cells: Vec<u8> = k.iter().copied().collect();
    let c = retouch::LumaDetect {
        hi: bars.0,
        hi_hair: bars.1,
        lo: bars.2,
        proxy_min: texture.0,
        reach_disk: crate::morph::Kernel {
            mask: &cells,
            w: k.ncols(),
            h: k.nrows(),
        },
        texture_knee: texture.1,
        texture_gain: texture.2,
        hair,
        hair_fill_gain: hair_texture.2,
        hair_texture: (hair_texture.0, hair_texture.1),
        texture_win,
        pad_px: pad.0,
        score_floor: pad.1,
    };
    let (score, hairs) = super::detach(py, n, || {
        retouch::detect_luma_score([proxy, background, z, tex], dims, &c)
    })
    .ok_or_else(|| PyValueError::new_err("invalid detection input"))?;
    Ok((
        score.map(|s| plane_array(py, s, dims)).transpose()?,
        hairs.map(|h| u8_array(py, h, dims)).transpose()?,
    ))
}

/// `_scratch_ridge` from the density plane, with (fine sigma, broad sigma, noise window).
#[pyfunction]
fn scratch_ridge<'py>(
    py: Python<'py>,
    dens: PyReadonlyArrayDyn<'py, f32>,
    constants: (f64, f64, usize),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let (d, dims) = plane2(&dens)?;
    new_image(py, dens.shape(), |dst| {
        retouch::scratch_ridge(d, dims, constants, dst)
    })
}

type SlopeFound<'py> = (f64, usize, usize, Bound<'py, numpy::PyArray1<f32>>);

/// `trace_scratch`'s slope search over the ridge band `z`: (strength, slope index, row,
/// that sheared row), or None for bad input.
#[pyfunction]
fn scratch_slope_search<'py>(
    py: Python<'py>,
    z: PyReadonlyArrayDyn<'py, f32>,
    cx: f64,
    rows: (usize, usize),
    pull: numpy::PyReadonlyArray1<'py, f64>,
    slopes: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<Option<SlopeFound<'py>>> {
    use numpy::IntoPyArray;
    let (zs, dims) = plane2(&z)?;
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let pull = pull.as_slice().map_err(not_contiguous)?;
    let slopes = slopes.as_slice().map_err(not_contiguous)?;
    let found = super::detach(py, zs.len() * slopes.len(), || {
        retouch::scratch_slope_search(zs, dims, cx, rows.0..rows.1, pull, slopes)
    });
    Ok(found.map(|(s, i, k, row)| (s, i, k, row.into_pyarray(py))))
}

/// `apply_clone_stroke` in place: the crop (x0, y0, x1, y1), the smoothed path in crop
/// pixels, the brush radius in whole pixels, then (radius, sigma), the source shift and
/// (feather, strength, match tone).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn clone_stroke(
    py: Python<'_>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
    rect: (usize, usize, usize, usize),
    path: numpy::PyReadonlyArrayDyn<'_, i32>,
    brush: i64,
    sizes: (f64, f64),
    shift: (i64, i64),
    brush_settings: (f64, f64, bool),
) -> PyResult<()> {
    use numpy::PyUntypedArrayMethods;
    let [h, w, 3] = out.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let (x0, y0, x1, y1) = rect;
    if x1 > w || y1 > h || x0 >= x1 || y0 >= y1 {
        return Err(PyValueError::new_err("crop outside the image"));
    }
    let path = path_points(&path)?;
    let (radius, sigma) = sizes;
    let (feather, strength, match_tone) = brush_settings;
    let s = retouch::CloneStroke {
        rect,
        path: &path,
        brush,
        radius,
        shift,
        sigma,
        feather,
        strength,
        match_tone,
    };
    let o = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    super::detach(py, (x1 - x0) * (y1 - y0), || {
        retouch::clone_stroke(o, (h, w), &s)
    })
    .ok_or_else(|| PyValueError::new_err("invalid clone stroke"))
}

fn path_points(path: &numpy::PyReadonlyArrayDyn<'_, i32>) -> PyResult<Vec<(i64, i64)>> {
    let pts = path
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous path"))?;
    Ok(pts
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0] as i64, p[1] as i64))
        .collect())
}

/// `drop_exclusions`' score release in place under a uint8 footprint, with the rim in pixels;
/// returns whether anything stays under 1.
#[pyfunction]
fn release_score(
    py: Python<'_>,
    mut score: PyReadwriteArrayDyn<'_, f32>,
    cover: numpy::PyReadonlyArray2<'_, u8>,
    rim: f32,
) -> PyResult<bool> {
    let [h, w] = score.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxW score"));
    };
    if cover.shape() != [h, w] {
        return Err(PyValueError::new_err(
            "needs a footprint of the score's shape",
        ));
    }
    let c = cover
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous footprint"))?;
    let s = score
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous score"))?;
    Ok(super::detach(py, s.len(), || {
        retouch::release_score(s, c, (h, w), rim)
    }))
}

/// One stroke of `strokes_to_score`: the density crop, the smoothed path in crop pixels, (cap
/// radius, line thickness), then (window, z min, z hi, z grow, grow fraction, pad, floor, rim).
#[pyfunction]
fn heal_region<'py>(
    py: Python<'py>,
    dens: PyReadonlyArrayDyn<'py, f32>,
    path: numpy::PyReadonlyArrayDyn<'py, i32>,
    brush: (i64, i64),
    constants: (usize, f64, f64, f64, f64, f64, f64, f32),
) -> PyResult<Option<Plane<'py>>> {
    let (d, dims) = plane2(&dens)?;
    let path = path_points(&path)?;
    let (win, z_min, z_hi, z_grow, z_grow_frac, pad_px, score_floor, rim) = constants;
    let c = retouch::HealSearch {
        win,
        z_min,
        z_hi,
        z_grow,
        z_grow_frac,
        pad_px,
        score_floor,
        rim,
    };
    super::detach(py, d.len(), || {
        retouch::heal_region(d, dims, &path, brush, &c)
    })
    .map(|r| plane_array(py, r, dims))
    .transpose()
}

/// `ir_ratio_and_gain`'s visible logs of an HxWx3 image, with the base window in pixels.
#[pyfunction]
fn ir_vis_log<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    win: usize,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let (src, dims) = image3(&img)?;
    new_image(py, img.shape(), |dst| {
        retouch::ir_vis_log(src, dims, win, dst)
    })
}

/// `ir_ratio_and_gain`'s gain from the ratio, the image, its clean base and the gammas, with
/// (identity, clamp).
#[pyfunction]
fn ir_gain<'py>(
    py: Python<'py>,
    ratio: PyReadonlyArrayDyn<'py, f32>,
    img: PyReadonlyArrayDyn<'py, f32>,
    clean: PyReadonlyArrayDyn<'py, f32>,
    gammas: [f64; 3],
    constants: (f32, f32),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let (src, (h, w)) = image3(&img)?;
    let r = plane(&ratio, h * w)?;
    let c = plane(&clean, 3 * h * w)?;
    new_image(py, img.shape(), |dst| {
        retouch::ir_gain(r, src, c, gammas, constants, dst);
        Some(())
    })
}

/// `cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)` of an HxWx3 uint8 image.
#[pyfunction]
fn inpaint_ns<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    radius: f64,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<u8>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    if mask.shape() != [h, w] {
        return Err(PyValueError::new_err("needs an HxW mask"));
    }
    let (src, m) = match (img.as_slice(), mask.as_slice()) {
        (Ok(s), Ok(m)) => (s, m),
        _ => return Err(PyValueError::new_err("needs contiguous arrays")),
    };
    let out = super::detach(py, src.len(), || {
        let mut out = src.to_vec();
        crate::inpaint::inpaint_ns(&mut out, m, (h, w), radius);
        out
    });
    out.into_pyarray(py).reshape(vec![h, w, 3])
}

/// `apply_hair_inpaint` of an HxWx3 image and its dilated hair mask, before the clean grain, with
/// (radius, dilate px, pad, tile, halo, range sample px, gamma).
#[pyfunction]
fn hair_fill<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    constants: (f64, usize, usize, usize, usize, usize, f64),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let (src, dims) = image3(&img)?;
    let m = mask
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous mask"))?;
    let (radius, dilate_px, pad, tile, halo, sample_px, gamma) = constants;
    if m.len() != dims.0 * dims.1 || tile == 0 || sample_px == 0 {
        return Err(PyValueError::new_err(
            "needs an HxW mask and nonzero tile and sample",
        ));
    }
    let c = crate::inpaint::HairFill {
        radius,
        dilate_px,
        pad,
        tile,
        halo,
        sample_px,
        gamma,
    };
    new_image(py, img.shape(), |dst| {
        dst.copy_from_slice(src);
        crate::inpaint::hair_fill(src, dims, m, &c, dst);
        Some(())
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inpaint_ns, m)?)?;
    m.add_function(wrap_pyfunction!(hair_fill, m)?)?;
    m.add_function(wrap_pyfunction!(ir_vis_log, m)?)?;
    m.add_function(wrap_pyfunction!(ir_gain, m)?)?;
    m.add_function(wrap_pyfunction!(release_score, m)?)?;
    m.add_function(wrap_pyfunction!(heal_region, m)?)?;
    m.add_function(wrap_pyfunction!(clone_stroke, m)?)?;
    m.add_function(wrap_pyfunction!(scratch_ridge, m)?)?;
    m.add_function(wrap_pyfunction!(scratch_slope_search, m)?)?;
    m.add_function(wrap_pyfunction!(split_hairs, m)?)?;
    m.add_function(wrap_pyfunction!(detect_luma_score, m)?)?;
    m.add_function(wrap_pyfunction!(dust_stats, m)?)?;
    m.add_function(wrap_pyfunction!(ir_defect_score, m)?)?;
    m.add_function(wrap_pyfunction!(fit_refraction_gammas, m)?)?;
    m.add_function(wrap_pyfunction!(route_wide_defects, m)?)?;
    m.add_function(wrap_pyfunction!(repair_components, m)?)?;
    m.add_function(wrap_pyfunction!(add_clean_grain, m)?)?;
    m.add_function(wrap_pyfunction!(apply_score_repair, m)?)?;
    m.add_function(wrap_pyfunction!(ir_ghost_divide, m)?)?;
    m.add_function(wrap_pyfunction!(ir_live, m)?)?;
    m.add_function(wrap_pyfunction!(ir_glare, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ir, m)?)?;
    m.add_function(wrap_pyfunction!(ir_normalize_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ir_clean_base, m)?)?;
    m.add_function(wrap_pyfunction!(hair_encode, m)?)
}
