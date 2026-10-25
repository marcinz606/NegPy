use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::analysis;

fn empty<'py, T: Element>(
    py: Python<'py>,
    shape: &[usize],
    dtype: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.to_vec(), dtype))?
        .cast_into::<PyArrayDyn<T>>()?)
}

fn slice<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'a [f32]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

/// `get_luminance` of an HxWx3 float32 image.
#[pyfunction]
fn luminance<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = rgb.shape().to_vec();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("luminance needs an HxWx3 array"));
    }
    let data = slice(&rgb)?;
    let out = empty::<f32>(py, &shape[..2], "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || analysis::luminance(data, dst));
    }
    Ok(out)
}

/// `to_log_density` of a float32 image, any shape.
#[pyfunction]
fn to_log_density<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = img.shape().to_vec();
    let data = slice(&img)?;
    let out = empty::<f32>(py, &shape, "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || analysis::to_log_density(data, dst));
    }
    Ok(out)
}

/// `_normalize_to_uint8` of a float32 image.
#[pyfunction]
fn normalize_to_uint8<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = img.shape().to_vec();
    let data = slice(&img)?;
    let out = empty::<u8>(py, &shape, "uint8")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || analysis::normalize_to_uint8(data, dst));
    }
    Ok(out)
}

/// `_detection_luma` of a float32 luminance plane.
#[pyfunction]
fn detection_luma<'py>(
    py: Python<'py>,
    lum: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = lum.shape().to_vec();
    let data = slice(&lum)?;
    let out = empty::<f32>(py, &shape, "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || analysis::detection_luma(data, dst)).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("percentiles of an empty array")
        })?;
    }
    Ok(out)
}

/// `_base_blocks(det, band)`: an Nx3 float32 array of the base blocks' means, or None.
#[pyfunction]
fn base_blocks<'py>(
    py: Python<'py>,
    det: PyReadonlyArrayDyn<'py, f32>,
    band: PyReadonlyArrayDyn<'py, bool>,
) -> PyResult<Option<Bound<'py, numpy::PyArray2<f32>>>> {
    let shape = det.shape().to_vec();
    let [h, w, 3] = shape[..] else {
        return Err(PyValueError::new_err(
            "flat_block_means needs an HxWx3 array",
        ));
    };
    if band.shape() != [h, w] {
        return Err(PyValueError::new_err("band must be HxW"));
    }
    let data = slice(&det)?;
    let mask: Vec<u8> = band
        .as_slice()
        .map_err(|_| PyValueError::new_err("band must be contiguous"))?
        .iter()
        .map(|&v| v as u8)
        .collect();
    let found = super::detach(py, data.len(), || analysis::base_blocks(data, &mask, h, w));
    found
        .map(|rows| {
            let flat: Vec<Vec<f32>> = rows.iter().map(|r| r.to_vec()).collect();
            numpy::PyArray2::from_vec2(py, &flat).map_err(|e| PyValueError::new_err(e.to_string()))
        })
        .transpose()
}

/// `_find_rebate_level(lum, film_roi, bed)` for a 2-D float32 plane: (level, spread) or None.
#[pyfunction]
#[pyo3(signature = (lum, film_roi, bed=None))]
fn find_rebate_level(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (usize, usize, usize, usize),
    bed: Option<f64>,
) -> PyResult<Option<(f64, f64)>> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err("find_rebate_level needs a 2-D array"));
    };
    let data = slice(&lum)?;
    Ok(super::detach(py, data.len(), || {
        analysis::find_rebate_level(data, h, w, film_roi, bed)
    }))
}

/// `_walk_border_depth(lines, limit, bridge, relative_step)` for an L x N float32 array.
#[pyfunction]
fn walk_border_depth<'py>(
    py: Python<'py>,
    lines: PyReadonlyArrayDyn<'py, f32>,
    limit: usize,
    bridge: i64,
    relative_step: bool,
) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
    let [rows, n] = lines.shape()[..] else {
        return Err(PyValueError::new_err("walk_border_depth needs a 2-D array"));
    };
    let data = slice(&lines)?;
    let out = super::detach(py, data.len(), || {
        analysis::walk_border_depth(data, rows, n, limit, bridge, relative_step)
    })
    .ok_or_else(|| PyValueError::new_err("walk_border_depth needs at least limit + 3 rows"))?;
    Ok(numpy::PyArray1::from_vec(py, out))
}

/// `measure_film_edges(lum, film_roi)`: (top, bottom, left, right) border fractions.
#[pyfunction]
fn measure_film_edges(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (usize, usize, usize, usize),
) -> PyResult<(f64, f64, f64, f64)> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err(
            "measure_film_edges needs a 2-D array",
        ));
    };
    let data = slice(&lum)?;
    let [t, b, l, r] = super::detach(py, data.len(), || {
        analysis::measure_film_edges(data, h, w, film_roi)
    });
    Ok((t, b, l, r))
}

/// `_trim_film_edges(lum, film_roi)`.
#[pyfunction]
fn trim_film_edges(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (i64, i64, i64, i64),
) -> PyResult<(i64, i64, i64, i64)> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err("trim_film_edges needs a 2-D array"));
    };
    let data = slice(&lum)?;
    Ok(super::detach(py, data.len(), || {
        analysis::trim_film_edges(data, h, w, film_roi)
    }))
}

/// `measure_film_border(lum, film_roi)`: (top, bottom, left, right) border fractions.
#[pyfunction]
fn measure_film_border(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (usize, usize, usize, usize),
) -> PyResult<(f64, f64, f64, f64)> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err(
            "measure_film_border needs a 2-D array",
        ));
    };
    let data = slice(&lum)?;
    let [t, b, l, r] = super::detach(py, data.len(), || {
        analysis::measure_film_border(data, h, w, film_roi)
    });
    Ok((t, b, l, r))
}

/// `_roi_from_measured_border(lum, film_roi)`.
#[pyfunction]
fn roi_from_measured_border(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (i64, i64, i64, i64),
) -> PyResult<Option<(i64, i64, i64, i64)>> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err(
            "roi_from_measured_border needs a 2-D array",
        ));
    };
    let data = slice(&lum)?;
    Ok(super::detach(py, data.len(), || {
        analysis::roi_from_measured_border(data, h, w, film_roi)
    }))
}

/// `_smooth_signal(signal, window)` for a 1-D float32 signal.
#[pyfunction]
fn smooth_signal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArrayDyn<'py, f32>,
    window: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let data = slice(&signal)?.to_vec();
    let out = super::detach(py, data.len(), || analysis::smooth_signal(&data, window)).ok_or_else(
        || {
            PyValueError::new_err(
                "smooth_signal covers a window of at most 31, no longer than the signal",
            )
        },
    )?;
    Ok(numpy::PyArray1::from_vec(py, out))
}

/// `_snap_edge_to_gradient(profile, idx, is_start, window_out, window_in, min_dominance, min_window_px)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn snap_edge_to_gradient(
    py: Python<'_>,
    profile: PyReadonlyArrayDyn<'_, f32>,
    idx: i64,
    is_start: bool,
    window_out: f64,
    window_in: f64,
    min_dominance: f64,
    min_window_px: i64,
) -> PyResult<i64> {
    let data = slice(&profile)?;
    let snap = analysis::Snap {
        window_out,
        window_in,
        min_dominance,
        min_window_px,
    };
    Ok(super::detach(py, data.len(), || {
        analysis::snap_edge_to_gradient(data, idx, is_start, snap)
    }))
}

type Refined<'py> = (
    (i64, i64, i64, i64),
    Bound<'py, numpy::PyArray1<f32>>,
    Bound<'py, numpy::PyArray1<f32>>,
);

/// `_refine_film_roi_by_tiers(lum, film_roi)`: (roi, row occupancy, column occupancy) or None.
#[pyfunction]
fn refine_film_roi_by_tiers<'py>(
    py: Python<'py>,
    lum: PyReadonlyArrayDyn<'py, f32>,
    film_roi: (usize, usize, usize, usize),
) -> PyResult<Option<Refined<'py>>> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err(
            "refine_film_roi_by_tiers needs a 2-D array",
        ));
    };
    let data = slice(&lum)?;
    let found = super::detach(py, data.len(), || {
        analysis::refine_film_roi_by_tiers(data, h, w, film_roi)
    });
    Ok(found.map(|r| {
        (
            r.roi,
            numpy::PyArray1::from_vec(py, r.rows),
            numpy::PyArray1::from_vec(py, r.cols),
        )
    }))
}

/// `_refine_frame_bounds(image)`'s (left, top, right, bottom) for an HxW or HxWx3 float32 image.
#[pyfunction]
fn refine_frame_bounds(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(i64, i64, i64, i64)> {
    let (h, w, cn) = match image.shape()[..] {
        [h, w] => (h, w, 1),
        [h, w, 3] => (h, w, 3),
        _ => {
            return Err(PyValueError::new_err(
                "refine_frame_bounds needs an HxW or HxWx3 array",
            ));
        }
    };
    let data = slice(&image)?;
    super::detach(py, data.len(), || {
        analysis::refine_frame_bounds(data, h, w, cn)
    })
    .ok_or_else(|| {
        PyValueError::new_err("refine_frame_bounds needs an image at least 31 px on each side")
    })
}

type Gated<'py> = (
    analysis::Roi,
    Option<Bound<'py, numpy::PyArray1<f32>>>,
    Option<Bound<'py, numpy::PyArray1<f32>>>,
);

/// `_refine_roi_to_image(img, film_roi)` for an HxW or HxWx3 float32 image: (roi, row
/// occupancy or None, column occupancy or None).
#[pyfunction]
fn refine_roi_to_image<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    film_roi: analysis::Roi,
) -> PyResult<Gated<'py>> {
    let (h, w, cn) = match img.shape()[..] {
        [h, w] => (h, w, 1),
        [h, w, 3] => (h, w, 3),
        _ => {
            return Err(PyValueError::new_err(
                "refine_roi_to_image needs an HxW or HxWx3 array",
            ));
        }
    };
    let data = slice(&img)?;
    let (roi, tiers) = super::detach(py, data.len(), || {
        analysis::refine_roi_to_image(data, h, w, cn, film_roi)
    })
    .ok_or_else(|| {
        PyValueError::new_err("refine_roi_to_image: image too small for the Sobel path")
    })?;
    Ok(match tiers {
        Some(t) => (
            roi,
            Some(numpy::PyArray1::from_vec(py, t.rows)),
            Some(numpy::PyArray1::from_vec(py, t.cols)),
        ),
        None => (roi, None, None),
    })
}

type DetectionFields<'py> = (
    Option<ConsensusFields>,
    Bound<'py, numpy::PyArray1<f32>>,
    f64,
);

/// `detect_film_bounds_with_confidence(img)` for an HxW or HxWx3 float32 image:
/// ((roi, angle, confidence, sides, corners, sources, geometry) or None, profile, contrast).
#[pyfunction]
fn detect_film_bounds<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<DetectionFields<'py>> {
    let (h, w, cn) = match img.shape()[..] {
        [h, w] => (h, w, 1),
        [h, w, 3] => (h, w, 3),
        _ => {
            return Err(PyValueError::new_err(
                "detect_film_bounds needs an HxW or HxWx3 array",
            ));
        }
    };
    let data = slice(&img)?;
    let d = super::detach(py, data.len(), || {
        crate::detect::detect_film_bounds(data, h, w, cn)
    });
    Ok((
        d.consensus.map(consensus_fields),
        numpy::PyArray1::from_vec(py, d.profile),
        d.contrast,
    ))
}

/// `normalize_log_image(img_log, bounds)` of an HxWx3 float32 log image.
#[pyfunction]
fn normalize_log_image<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    floors: [f32; 3],
    ceils: [f32; 3],
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = img.shape().to_vec();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err(
            "normalize_log_image needs an HxWx3 array",
        ));
    }
    let data = slice(&img)?;
    let out = empty::<f32>(py, &shape, "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            analysis::normalize_log_image(data, floors, ceils, dst)
        });
    }
    Ok(out)
}

/// `_block_median_grid` of an HxWxC float32 image in `b` x `b` blocks.
#[pyfunction]
fn block_median_grid<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    b: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, c] = img.shape()[..] else {
        return Err(PyValueError::new_err(
            "block_median_grid needs an HxWxC array",
        ));
    };
    if b == 0 {
        return Err(PyValueError::new_err("block size must be positive"));
    }
    let data = slice(&img)?;
    let grid = super::detach(py, data.len(), || {
        analysis::block_median_grid(data, (h, w, c), b)
    });
    numpy::PyArray1::from_vec(py, grid).reshape(vec![h / b, w / b, c])
}

/// `_get_threshold_autocrop_coords` of an HxW float32 luma plane: (y1, y2, x1, x2) or None.
#[pyfunction]
fn threshold_box(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    threshold: f32,
) -> PyResult<Option<(usize, usize, usize, usize)>> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err("threshold_box needs a 2-D array"));
    };
    let data = slice(&lum)?;
    Ok(super::detach(py, data.len(), || {
        analysis::threshold_box(data, h, w, threshold)
    }))
}

/// Batch autocrop's `_vertical_edge_profile` of an HxW or HxWxC (C >= 3) float32 image:
/// (profile, contrast).
#[pyfunction]
fn batch_edge_profile<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f32>>, f64)> {
    let (h, w, cn) = match img.shape()[..] {
        [h, w] | [h, w, 1] => (h, w, 1),
        [h, w, c] if c >= 3 => (h, w, c),
        _ => {
            return Err(PyValueError::new_err(
                "batch_edge_profile needs an HxW, HxWx1 or HxWxC array, C >= 3",
            ));
        }
    };
    let data = slice(&img)?;
    let (profile, contrast) = super::detach(py, data.len(), || {
        analysis::batch_edge_profile(data, h, w, cn)
    });
    Ok((numpy::PyArray1::from_vec(py, profile), contrast))
}

/// The HxW of an HxWx3 detection image.
fn det_dims(det: &PyReadonlyArrayDyn<'_, f32>) -> PyResult<(usize, usize)> {
    match det.shape()[..] {
        [h, w, 3] => Ok((h, w)),
        _ => Err(PyValueError::new_err("needs an HxWx3 array")),
    }
}

/// `film_and_picture(det)` of an HxWx3 float32 detection image: (film roi or None, gate).
#[pyfunction]
fn film_and_picture(
    py: Python<'_>,
    det: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(Option<analysis::Roi>, analysis::Roi)> {
    let (h, w) = det_dims(&det)?;
    let data = slice(&det)?;
    super::detach(py, data.len(), || analysis::film_and_picture(data, h, w)).ok_or_else(|| {
        PyValueError::new_err("film_and_picture: image too small for the Sobel path")
    })
}

/// `measure_base_log` of an HxWx3 float32 detection image: (r, g, b) log10 or None.
#[pyfunction]
fn measure_base_log(
    py: Python<'_>,
    det: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<Option<[f64; 3]>> {
    let (h, w) = det_dims(&det)?;
    let data = slice(&det)?;
    Ok(super::detach(py, data.len(), || {
        analysis::measure_base_log(data, h, w)
    }))
}

type ConsensusFields = (
    analysis::Roi,
    f64,
    f64,
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
    f64,
);

fn consensus_fields(c: crate::detect::Consensus) -> ConsensusFields {
    (
        c.roi,
        c.correction_angle,
        c.confidence,
        c.sides,
        c.corners,
        c.sources,
        c.geometry_score,
    )
}

fn known(value: &str, names: &[&'static str]) -> PyResult<&'static str> {
    names
        .iter()
        .find(|n| **n == value)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("unknown name {value:?}")))
}

fn flags(values: &[String], names: &[&'static str]) -> PyResult<[bool; 4]> {
    let mut out = [false; 4];
    for v in values {
        out[names
            .iter()
            .position(|n| n == v)
            .ok_or_else(|| PyValueError::new_err(format!("unknown name {v:?}")))?] = true;
    }
    Ok(out)
}

type CandidateFields = (
    analysis::Roi,
    f64,
    String,
    Option<usize>,
    String,
    f64,
    f64,
    Vec<String>,
    Vec<String>,
);

/// `_select_consensus_cluster` over (roi, angle, source, threshold index, polarity, boundary,
/// geometry, sides, corners) candidates in an HxW image.
#[pyfunction]
fn select_consensus(
    candidates: Vec<CandidateFields>,
    h: usize,
    w: usize,
) -> PyResult<Option<ConsensusFields>> {
    use crate::detect::{CORNER_SIDES, Candidate, EVIDENCE_SOURCE_ORDER, SIDE_NAMES};
    let corners: Vec<&'static str> = CORNER_SIDES.iter().map(|c| c.0).collect();
    let cands = candidates
        .into_iter()
        .map(
            |(roi, angle, source, index, polarity, boundary, geometry, s, c)| {
                Ok(Candidate {
                    roi,
                    correction_angle: angle,
                    source: known(&source, &EVIDENCE_SOURCE_ORDER)?,
                    threshold_index: index,
                    polarity: known(&polarity, &["dark", "bright"])?,
                    boundary_score: boundary,
                    geometry_score: geometry,
                    sides: flags(&s, &SIDE_NAMES)?,
                    corners: flags(&c, &corners)?,
                })
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    Ok(crate::detect::select_consensus(&cands, h, w).map(consensus_fields))
}

fn lum_dims<'a>(lum: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], usize, usize)> {
    match lum.shape()[..] {
        [h, w] => Ok((slice(lum)?, h, w)),
        _ => Err(PyValueError::new_err("needs an HxW array")),
    }
}

/// `_film_box_covers_enough(lum, roi)`.
#[pyfunction]
fn film_box_covers_enough(lum: PyReadonlyArrayDyn<'_, f32>, roi: analysis::Roi) -> PyResult<bool> {
    let (_, h, w) = lum_dims(&lum)?;
    Ok(crate::detect::film_box_covers_enough(h, w, &roi))
}

/// `_film_surround_is_plausible(lum, roi)`.
#[pyfunction]
fn film_surround_is_plausible(
    lum: PyReadonlyArrayDyn<'_, f32>,
    roi: analysis::Roi,
) -> PyResult<bool> {
    let (data, h, w) = lum_dims(&lum)?;
    Ok(crate::detect::film_surround_is_plausible(data, h, w, &roi))
}

fn rgb_dims(a: &PyReadonlyArrayDyn<'_, f32>, name: &str) -> PyResult<(usize, usize)> {
    match a.shape()[..] {
        [h, w, 3] => Ok((h, w)),
        _ => Err(PyValueError::new_err(format!(
            "{name} needs an HxWx3 array"
        ))),
    }
}

fn filled_f32<'py>(
    py: Python<'py>,
    shape: &[usize],
    n: usize,
    f: impl FnOnce(&mut [f32]) -> Option<()> + Send,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let out = empty::<f32>(py, shape, "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, n, || f(dst))
            .ok_or_else(|| PyValueError::new_err("blur sigma must be positive"))?;
    }
    Ok(out)
}

/// `measure_clip_fractions` of an HxWx3 float32 image over the window (y1, y2, x1, x2).
#[pyfunction]
fn clip_fractions(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    window: (usize, usize, usize, usize),
    level: f32,
) -> PyResult<(f64, f64, f64)> {
    let (h, w) = rgb_dims(&img, "clip_fractions")?;
    let (y1, y2, x1, x2) = window;
    if y1 > y2 || x1 > x2 || y2 > h || x2 > w {
        return Err(PyValueError::new_err("window outside the image"));
    }
    let data = slice(&img)?;
    let f = super::detach(py, (y2 - y1) * (x2 - x1) * 3 / 16, || {
        analysis::clip_fractions(data, w, window, level)
    });
    Ok((f[0], f[1], f[2]))
}

/// The median of a float32 grid's float64 Rec. 709 luma, any shape that ends in 3.
#[pyfunction]
fn luma_median(py: Python<'_>, grid: PyReadonlyArrayDyn<'_, f32>) -> PyResult<f64> {
    if grid.shape().last() != Some(&3) {
        return Err(PyValueError::new_err("luma_median needs an ...x3 array"));
    }
    let data = slice(&grid)?;
    Ok(super::detach(py, data.len(), || {
        analysis::luma_median(data)
    }))
}

/// Each pixel of an HxWx3 float32 plane set to its float32 Rec. 709 luma.
#[pyfunction]
fn luma_planes<'py>(
    py: Python<'py>,
    val: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let (h, w) = rgb_dims(&val, "luma_planes")?;
    let data = slice(&val)?;
    filled_f32(py, &[h, w, 3], data.len(), |dst| {
        analysis::luma_planes(data, dst);
        Some(())
    })
}

/// `contrast_mask_plane`'s blurred zero-mean luma of a normalized HxWx3 plane, and its mean.
#[pyfunction]
fn contrast_mask_tail<'py>(
    py: Python<'py>,
    val: PyReadonlyArrayDyn<'py, f32>,
    sigma: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, f32)> {
    let (h, w) = rgb_dims(&val, "contrast_mask_tail")?;
    if h * w == 0 {
        return Err(PyValueError::new_err(
            "contrast_mask_tail needs a non-empty plane",
        ));
    }
    let data = slice(&val)?;
    let (plane, centre) = super::detach(py, data.len(), || {
        analysis::contrast_mask_tail(data, h, w, sigma)
    })
    .ok_or_else(|| PyValueError::new_err("blur sigma must be positive"))?;
    Ok((
        numpy::PyArray1::from_vec(py, plane).reshape(vec![h, w])?,
        centre,
    ))
}

/// The Contrast Mask's print stops from its plane, any shape; `k` is
/// (a, b, lo, hi, g_lo, g_hi, lim) from `contrast_mask_coeffs`.
#[pyfunction]
fn contrast_mask_stops<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    k: [f32; 7],
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let data = slice(&plane)?;
    let shape = plane.shape().to_vec();
    filled_f32(py, &shape, data.len(), |dst| {
        analysis::contrast_mask_stops(data, dst, k);
        Some(())
    })
}

/// `diffusion_grid`'s blurred light of an unmixed HxWx3 log plane, as log10.
#[pyfunction]
fn diffusion_tail<'py>(
    py: Python<'py>,
    unmixed: PyReadonlyArrayDyn<'py, f32>,
    sigma: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let (h, w) = rgb_dims(&unmixed, "diffusion_tail")?;
    let data = slice(&unmixed)?;
    filled_f32(py, &[h, w, 3], data.len(), |dst| {
        analysis::diffusion_tail(data, h, w, sigma, dst)
    })
}

/// The normalization processors' open-ended log density of a float32 image, any shape.
#[pyfunction]
fn log_density_open<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = img.shape().to_vec();
    let data = slice(&img)?;
    let out = empty::<f32>(py, &shape, "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || analysis::log_density_open(data, dst));
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(luminance, m)?)?;
    m.add_function(wrap_pyfunction!(to_log_density, m)?)?;
    m.add_function(wrap_pyfunction!(log_density_open, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_to_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(detection_luma, m)?)?;
    m.add_function(wrap_pyfunction!(base_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(find_rebate_level, m)?)?;
    m.add_function(wrap_pyfunction!(walk_border_depth, m)?)?;
    m.add_function(wrap_pyfunction!(measure_film_edges, m)?)?;
    m.add_function(wrap_pyfunction!(trim_film_edges, m)?)?;
    m.add_function(wrap_pyfunction!(measure_film_border, m)?)?;
    m.add_function(wrap_pyfunction!(roi_from_measured_border, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_signal, m)?)?;
    m.add_function(wrap_pyfunction!(snap_edge_to_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(refine_film_roi_by_tiers, m)?)?;
    m.add_function(wrap_pyfunction!(refine_frame_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(refine_roi_to_image, m)?)?;
    m.add_function(wrap_pyfunction!(detect_film_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(select_consensus, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_log_image, m)?)?;
    m.add_function(wrap_pyfunction!(block_median_grid, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_box, m)?)?;
    m.add_function(wrap_pyfunction!(batch_edge_profile, m)?)?;
    m.add_function(wrap_pyfunction!(film_and_picture, m)?)?;
    m.add_function(wrap_pyfunction!(measure_base_log, m)?)?;
    m.add_function(wrap_pyfunction!(film_box_covers_enough, m)?)?;
    m.add_function(wrap_pyfunction!(film_surround_is_plausible, m)?)?;
    m.add_function(wrap_pyfunction!(clip_fractions, m)?)?;
    m.add_function(wrap_pyfunction!(luma_median, m)?)?;
    m.add_function(wrap_pyfunction!(luma_planes, m)?)?;
    m.add_function(wrap_pyfunction!(contrast_mask_tail, m)?)?;
    m.add_function(wrap_pyfunction!(contrast_mask_stops, m)?)?;
    m.add_function(wrap_pyfunction!(diffusion_tail, m)?)
}
