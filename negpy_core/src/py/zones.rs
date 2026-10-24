use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::zones;

type Frame<'a> = (&'a [f32], (usize, usize, usize));

fn frame<'a>(rgb: &'a PyReadonlyArray3<'_, f32>) -> PyResult<Frame<'a>> {
    let s = rgb.shape();
    let dims = (s[0], s[1], s[2]);
    if dims.0 < 1 || dims.1 < 1 || dims.2 < 3 {
        return Err(PyValueError::new_err("needs an HxWx3 frame"));
    }
    let data = rgb
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous frame"))?;
    Ok((data, dims))
}

fn grid<'py, T: numpy::Element>(
    py: Python<'py>,
    v: Vec<T>,
    h: usize,
    w: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    v.into_pyarray(py).reshape([h, w])
}

/// `zone_field(rgb, max_edge)` for an HxWxC float32 display frame and the encoded mid gray.
#[pyfunction]
fn zone_field<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    max_edge: usize,
    mid: f64,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (data, dims) = frame(&rgb)?;
    let (z, h, w) = super::detach(py, data.len(), || {
        zones::zone_field(data, dims, max_edge, mid)
    })
    .ok_or_else(|| PyValueError::new_err("invalid frame"))?;
    grid(py, z, h, w)
}

/// `zone_grid(rgb, cells)` for an HxWxC float32 display frame and the encoded mid gray.
#[pyfunction]
fn zone_grid<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    cells: usize,
    mid: f64,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let (data, dims) = frame(&rgb)?;
    let step = (dims.0.min(dims.1) / (4 * cells.max(1))).max(1);
    let sampled = dims.0.div_ceil(step) * dims.1.div_ceil(step) * 3;
    let (z, h, w) = super::detach(py, sampled, || zones::zone_grid(data, dims, cells, mid))
        .ok_or_else(|| PyValueError::new_err("invalid frame"))?;
    grid(py, z, h, w)
}

/// `zone_anchors(zones, min_share)` for a uint8 zone map: [(x, y, zone, inset)].
#[pyfunction]
fn zone_anchors(
    py: Python<'_>,
    zones: PyReadonlyArray2<'_, u8>,
    min_share: f64,
) -> PyResult<Vec<(usize, usize, u8, f32)>> {
    let dims = (zones.shape()[0], zones.shape()[1]);
    let z = zones
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous map"))?;
    Ok(super::detach(py, z.len(), || {
        zones::zone_anchors(z, dims, min_share)
    }))
}

/// `zone_region_labels(zones)` for an int32 zone grid: [(col, row, zone)].
#[pyfunction]
fn zone_region_labels(
    py: Python<'_>,
    zones: PyReadonlyArray2<'_, i32>,
) -> PyResult<Vec<(usize, usize, i32)>> {
    let dims = (zones.shape()[0], zones.shape()[1]);
    let z = zones
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous grid"))?;
    Ok(super::detach(py, z.len(), || {
        zones::zone_region_labels(z, dims)
    }))
}

/// `zone_shares(zones)` for a uint8 zone map.
#[pyfunction]
fn zone_shares<'py>(
    py: Python<'py>,
    zones: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let z = zones
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous map"))?;
    Ok(super::detach(py, z.len(), || zones::zone_shares(z)).into_pyarray(py))
}

/// `zone_occupancy(l_bins)` for float64 bins and the encoded mid gray.
#[pyfunction]
fn zone_occupancy<'py>(
    py: Python<'py>,
    bins: PyReadonlyArray1<'py, f64>,
    mid: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let b = bins
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous bins"))?;
    Ok(zones::zone_occupancy(b, mid).into_pyarray(py))
}

/// `loupe_acutance(patch)` for an HxWxC float32 display patch, at least 3x3.
#[pyfunction]
fn loupe_acutance(py: Python<'_>, patch: PyReadonlyArray3<'_, f32>) -> PyResult<f64> {
    let (data, dims) = frame(&patch)?;
    Ok(super::detach(py, data.len(), || {
        zones::loupe_acutance(data, dims)
    }))
}

type Lookup<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// `_scope_lookup(size, log)`: (bin per pixel, pixels per bin, unsampled bins, their pixels).
#[pyfunction]
fn scope_lookup(py: Python<'_>, size: usize, log: bool) -> PyResult<Lookup<'_>> {
    let (idx, area, orphans, px) =
        super::detach(py, size * size, || zones::scope_lookup(size, log));
    Ok((
        grid(py, idx, size, size)?,
        area.into_pyarray(py),
        orphans.into_pyarray(py),
        px.into_pyarray(py),
    ))
}

/// `vectorscope_density(counts, size, log)` from float64 counts and a `scope_lookup`.
#[pyfunction]
fn vectorscope_density<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray1<'py, f64>,
    idx: PyReadonlyArray2<'py, i64>,
    area: PyReadonlyArray1<'py, f64>,
    orphans: PyReadonlyArray1<'py, i64>,
    px: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let bad = |_| PyValueError::new_err("needs contiguous arrays");
    let (c, i, a, o, p) = (
        counts.as_slice().map_err(bad)?,
        idx.as_slice().map_err(bad)?,
        area.as_slice().map_err(bad)?,
        orphans.as_slice().map_err(bad)?,
        px.as_slice().map_err(bad)?,
    );
    let n = i.len();
    if a.len() > c.len()
        || i.iter().any(|&v| v >= a.len() as i64)
        || o.len() != p.len()
        || o.iter().any(|&v| v < 0 || v >= c.len() as i64)
        || p.iter().any(|&v| v < 0 || v >= n as i64)
    {
        return Err(PyValueError::new_err("counts and lookup do not match"));
    }
    let out = super::detach(py, n, || zones::vectorscope_density(c, (i, a, o, p)));
    let size = idx.shape()[0];
    grid(py, out, size, idx.shape()[1])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zone_field, m)?)?;
    m.add_function(wrap_pyfunction!(zone_grid, m)?)?;
    m.add_function(wrap_pyfunction!(zone_anchors, m)?)?;
    m.add_function(wrap_pyfunction!(zone_region_labels, m)?)?;
    m.add_function(wrap_pyfunction!(zone_shares, m)?)?;
    m.add_function(wrap_pyfunction!(zone_occupancy, m)?)?;
    m.add_function(wrap_pyfunction!(loupe_acutance, m)?)?;
    m.add_function(wrap_pyfunction!(scope_lookup, m)?)?;
    m.add_function(wrap_pyfunction!(vectorscope_density, m)?)
}
