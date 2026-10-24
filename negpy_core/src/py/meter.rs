use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::meter::{self, Activity, ChannelNorm, NeutralConsts};

/// Per channel (floor, floor is float64, span, span is float64), as the caller's types decide.
type Norm = [(f64, bool, f64, bool); 3];

/// The data and size of an HxWx3 float32 grid.
fn grid_of<'a>(grid: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], usize, usize)> {
    let [h, w, 3] = grid.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 array"));
    };
    let data = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    Ok((data, h, w))
}

fn activity((block, gate, min_fraction): (usize, f32, f64)) -> PyResult<Activity> {
    if block == 0 {
        return Err(PyValueError::new_err("block must be positive"));
    }
    Ok(Activity {
        block,
        gate,
        min_fraction,
    })
}

/// The textured luma of a grid, normalized when `norm` is given.
fn textured((data, h, w): (&[f32], usize, usize), norm: Option<Norm>, a: Activity) -> Vec<f32> {
    let norm = norm.map(|n| {
        n.map(|(floor, floor_wide, span, span_wide)| ChannelNorm {
            floor,
            floor_wide,
            span,
            span_wide,
        })
    });
    meter::textured_cells(meter::luma(data, norm.as_ref()), h, w, a)
}

/// `np.percentile` (float32 q) of the grid's textured normalized luma: shadow reach, highlight hold.
#[pyfunction]
fn textured_percentile(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    norm: Norm,
    gate: (usize, f32, f64),
    q: f64,
) -> PyResult<f32> {
    let (g, a) = (grid_of(&grid)?, activity(gate)?);
    Ok(super::detach(py, g.0.len(), || {
        meter::percentile(&textured(g, Some(norm), a), q)
    }))
}

/// The anchor meter's measurement over the grid's textured normalized luma.
#[pyfunction]
fn textured_anchor(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    norm: Norm,
    gate: (usize, f32, f64),
    clip: f64,
) -> PyResult<f64> {
    let (g, a) = (grid_of(&grid)?, activity(gate)?);
    Ok(super::detach(py, g.0.len(), || {
        meter::anchor_measure(&textured(g, Some(norm), a), clip)
    }))
}

/// The textural range of the grid's textured raw-log luma.
#[pyfunction]
fn textured_range(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    gate: (usize, f32, f64),
    clip: f64,
) -> PyResult<f64> {
    let (g, a) = (grid_of(&grid)?, activity(gate)?);
    Ok(super::detach(py, g.0.len(), || {
        meter::textural_range(&textured(g, None, a), clip)
    }))
}

type Band = (f32, f32);
type Sector = (f32, f32, f32);
type Triplet = (f64, f64, f64);
/// (midtone, shadow, highlight or None, confidence).
type Axis = (Triplet, Triplet, Option<Triplet>, f64);
/// quantile, cap, first-pass cap, min pixels, highlight/mid/shadow bands, foliage and sky
/// sectors, n0, agreement (dead zone, scale), width (dead zone, scale).
type Consts = (
    f64,
    f64,
    f64,
    usize,
    Band,
    Band,
    Band,
    Sector,
    Sector,
    f64,
    (f64, f64),
    (f64, f64),
);

fn triplet(r: [f64; 3]) -> Triplet {
    (r[0], r[1], r[2])
}

/// `measure_neutral_axis_from_log` of a cropped block-median grid: (midtone, shadow,
/// highlight or None, confidence), or None.
#[pyfunction]
fn neutral_axis(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    floors: Triplet,
    ceils: Triplet,
    exclude_subjects: bool,
    consts: Consts,
) -> PyResult<Option<Axis>> {
    let (data, _, _) = grid_of(&grid)?;
    let (
        quantile,
        cap,
        first_pass_cap,
        min_pixels,
        highlight,
        mid,
        shadow,
        foliage,
        sky,
        n0,
        agreement,
        width,
    ) = consts;
    let c = NeutralConsts {
        quantile,
        cap,
        first_pass_cap,
        min_pixels,
        highlight,
        mid,
        shadow,
        subjects: [foliage, sky],
        n0,
        agreement,
        width,
    };
    let f = [floors.0, floors.1, floors.2];
    let e = [ceils.0, ceils.1, ceils.2];
    let axis = super::detach(py, data.len(), || {
        meter::neutral_axis(
            data,
            (f.map(|v| v as f32), e.map(|v| v as f32)),
            (f, e),
            exclude_subjects,
            &c,
        )
    });
    Ok(axis.map(|a| {
        (
            triplet(a.mid),
            triplet(a.shadow),
            a.highlight.map(triplet),
            a.confidence,
        )
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(textured_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(textured_anchor, m)?)?;
    m.add_function(wrap_pyfunction!(textured_range, m)?)?;
    m.add_function(wrap_pyfunction!(neutral_axis, m)?)
}
