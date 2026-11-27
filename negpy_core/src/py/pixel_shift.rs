use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::pixel_shift::{self, Texture};

fn mosaic<'a>(a: &'a PyReadonlyArrayDyn<'_, u16>) -> PyResult<(&'a [u16], (usize, usize))> {
    let [h, w] = a.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxW mosaic"));
    };
    if h < 2 || w < 2 {
        return Err(PyValueError::new_err("the mosaic is too small"));
    }
    let data = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous mosaic"))?;
    Ok((data, (h, w)))
}

/// One shot of `merge_set`, written into `out` and `second_green` in place.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pixel_shift_place(
    py: Python<'_>,
    mut out: PyReadwriteArrayDyn<'_, u16>,
    mut second_green: PyReadwriteArrayDyn<'_, u16>,
    shot: PyReadonlyArrayDyn<'_, u16>,
    cfa: [u8; 4],
    scale: usize,
    origin: (usize, usize),
    half: (usize, usize),
) -> PyResult<()> {
    let (data, (h, w)) = mosaic(&shot)?;
    if !(1..=2).contains(&scale)
        || origin.0 > 1
        || origin.1 > 1
        || half.0 >= scale
        || half.1 >= scale
    {
        return Err(PyValueError::new_err(
            "the shot's position is off the output grid",
        ));
    }
    let size = (scale * (h - 1), scale * (w - 1));
    if cfa.iter().any(|&c| c > 3) {
        return Err(PyValueError::new_err("needs CFA color indices 0 to 3"));
    }
    if out.shape() != [size.0, size.1, 3] || second_green.shape() != [size.0, size.1] {
        return Err(PyValueError::new_err("the output does not match the shot"));
    }
    let sites = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous output"))?;
    let greens = second_green
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous output"))?;
    super::detach(py, data.len(), || {
        pixel_shift::place_shot(data, (h, w), cfa, scale, origin, half, sites, greens)
    });
    Ok(())
}

/// The green plane of `out` becomes the mean of the two greens, in place.
#[pyfunction]
fn pixel_shift_average_greens(
    py: Python<'_>,
    mut out: PyReadwriteArrayDyn<'_, u16>,
    second_green: PyReadonlyArrayDyn<'_, u16>,
) -> PyResult<()> {
    let [h, w, 3] = out.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 output"));
    };
    if second_green.shape() != [h, w] {
        return Err(PyValueError::new_err(
            "the second green does not match the output",
        ));
    }
    let greens = second_green
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous plane"))?;
    let sites = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous output"))?;
    super::detach(py, sites.len(), || {
        pixel_shift::average_greens(sites, greens)
    });
    Ok(())
}

/// The CFA site (row, col) the light shows brightest.
#[pyfunction]
fn pixel_shift_brightest_site(
    py: Python<'_>,
    shot: PyReadonlyArrayDyn<'_, u16>,
) -> PyResult<(usize, usize)> {
    let (data, (_, w)) = mosaic(&shot)?;
    Ok(super::detach(py, data.len(), || {
        pixel_shift::brightest_site(data, w)
    }))
}

/// `(steps, total)` of one CFA site: the sum of the steps between its neighboring samples and the
/// sum of the samples.
#[pyfunction]
fn pixel_shift_site_detail(
    py: Python<'_>,
    shot: PyReadonlyArrayDyn<'_, u16>,
    site: (usize, usize),
) -> PyResult<(u64, u64)> {
    let (data, (_, w)) = mosaic(&shot)?;
    if site.0 > 1 || site.1 > 1 {
        return Err(PyValueError::new_err("a CFA site is row and column 0 or 1"));
    }
    Ok(super::detach(py, data.len(), || {
        pixel_shift::site_detail(data, w, site)
    }))
}

/// `(top, dx, dy)` per textured row band: the move of `current` against `previous` less `expected`,
/// sensor pixels.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pixel_shift_band_moves(
    py: Python<'_>,
    previous: PyReadonlyArrayDyn<'_, u16>,
    current: PyReadonlyArrayDyn<'_, u16>,
    site: (usize, usize),
    band: (usize, usize),
    lefts: Vec<usize>,
    expected: (f64, f64),
    min_contrast: f64,
    min_response: f64,
) -> PyResult<Vec<(usize, f64, f64)>> {
    let (a, (h, w)) = mosaic(&previous)?;
    let (b, size) = mosaic(&current)?;
    if size != (h, w) {
        return Err(PyValueError::new_err("the shots differ in size"));
    }
    if site.0 > 1 || site.1 > 1 || lefts.iter().any(|&l| l + band.1 > w) {
        return Err(PyValueError::new_err("the bands do not fit the mosaic"));
    }
    // Too small to hold a band: nothing to judge.
    if band.0 < 8 || band.1 < 8 || band.0 > h {
        return Ok(Vec::new());
    }
    let texture = Texture {
        contrast: min_contrast,
        response: min_response,
    };
    Ok(super::detach(py, a.len(), || {
        pixel_shift::band_moves(a, b, (h, w), site, band, &lefts, expected, texture)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pixel_shift_place, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_shift_average_greens, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_shift_brightest_site, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_shift_site_detail, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_shift_band_moves, m)?)
}
