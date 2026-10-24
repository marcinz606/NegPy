use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::contours;

type PyRotatedRect = ((f32, f32), (f32, f32), f32);

/// `cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` for a uint8 image:
/// (contours as Nx1x2 int32 arrays, hierarchy as 1xKx4 int32), in OpenCV's order.
#[pyfunction]
fn find_external_contours<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyAny>)> {
    let arr = src.as_array();
    let (h, w) = arr.dim();
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let found = super::detach(py, data.len(), || {
        contours::find_external_contours(data, h, w)
    });
    let k = found.len();
    let arrays: Vec<Bound<'py, PyArray3<i32>>> = found
        .iter()
        .map(|c| {
            let flat: Vec<Vec<Vec<i32>>> = c.iter().map(|&(x, y)| vec![vec![x, y]]).collect();
            PyArray3::from_vec3(py, &flat)
        })
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let hierarchy: Vec<Vec<i32>> = (0..k)
        .map(|i| {
            vec![
                if i + 1 < k { i as i32 + 1 } else { -1 },
                i as i32 - 1,
                -1,
                -1,
            ]
        })
        .collect();
    let np = py.import("numpy")?;
    let hier = if k == 0 {
        py.None().into_bound(py)
    } else {
        np.getattr("array")?.call1((vec![hierarchy], "int32"))?
    };
    Ok((PyTuple::new(py, arrays)?, hier))
}

fn int_points(pts: &numpy::PyReadonlyArrayDyn<'_, i32>) -> PyResult<Vec<(i32, i32)>> {
    let flat = pts
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be contiguous"))?;
    if flat.len() % 2 != 0 {
        return Err(PyValueError::new_err("points must be pairs"));
    }
    Ok(flat
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0], p[1]))
        .collect())
}

/// `cv2.minAreaRect(points)` of int32 points: ((cx, cy), (w, h), angle).
#[pyfunction]
fn min_area_rect(pts: numpy::PyReadonlyArrayDyn<'_, i32>) -> PyResult<PyRotatedRect> {
    let r = crate::shapes::min_area_rect(&int_points(&pts)?);
    Ok((r.center, r.size, r.angle))
}

/// `cv2.boxPoints(rect)`: four (x, y) float32 corners.
#[pyfunction]
fn box_points(rect: PyRotatedRect) -> Vec<(f32, f32)> {
    let r = crate::shapes::RotatedRect {
        center: rect.0,
        size: rect.1,
        angle: rect.2,
    };
    crate::shapes::box_points(&r).to_vec()
}

/// `cv2.convexHull(points)` of int32 points, as a list of (x, y).
#[pyfunction]
fn convex_hull(pts: numpy::PyReadonlyArrayDyn<'_, i32>) -> PyResult<Vec<(i32, i32)>> {
    Ok(crate::shapes::convex_hull(&int_points(&pts)?))
}

/// `cv2.fillConvexPoly(zeros((h, w), uint8), points, 1)` as a new mask.
#[pyfunction]
fn fill_convex_poly<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    pts: numpy::PyReadonlyArrayDyn<'_, i32>,
) -> PyResult<Bound<'py, numpy::PyArray2<u8>>> {
    let (h, w) = shape;
    let v: Vec<(i64, i64)> = int_points(&pts)?
        .iter()
        .map(|&(x, y)| (x as i64, y as i64))
        .collect();
    let mut mask = vec![0u8; h * w];
    crate::shapes::fill_convex_poly(&mut mask, w, h, &v, 1);
    numpy::PyArray1::from_vec(py, mask).reshape([h, w])
}

/// `cv2.fillPoly(zeros((h, w), uint8), [points], 255)` as a new mask.
#[pyfunction]
fn fill_poly<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    pts: numpy::PyReadonlyArrayDyn<'_, i32>,
) -> PyResult<Bound<'py, numpy::PyArray2<u8>>> {
    let (h, w) = shape;
    let v: Vec<(i64, i64)> = int_points(&pts)?
        .iter()
        .map(|&(x, y)| (x as i64, y as i64))
        .collect();
    let mut mask = vec![0u8; h * w];
    crate::shapes::fill_poly(&mut mask, w, h, &v, 255);
    numpy::PyArray1::from_vec(py, mask).reshape([h, w])
}

/// One mask for `local_maps`: ("fill", outline Nx2 int32, sigma) or ("ramp", (ax, ay, kx, ky))
/// or ("empty",), then invert, plane (-1 for none), stops, grade, flash.
type MaskSpec<'py> = (
    String,
    Option<numpy::PyReadonlyArrayDyn<'py, i32>>,
    (f64, f32, f32, f32, f32),
    bool,
    i64,
    (f32, f32, f32),
);

fn local_mask(spec: MaskSpec<'_>, planes: usize) -> PyResult<crate::shapes::LocalMask> {
    use crate::shapes::{LocalMask, MaskAlpha};
    let (kind, outline, (sigma, ax, ay, kx, ky), invert, plane, (stops, grade, flash)) = spec;
    let alpha = match kind.as_str() {
        "fill" => {
            let pts = outline.ok_or_else(|| PyValueError::new_err("a fill needs its outline"))?;
            let outline = int_points(&pts)?
                .iter()
                .map(|&(x, y)| (x as i64, y as i64))
                .collect();
            MaskAlpha::Fill { outline, sigma }
        }
        "ramp" => MaskAlpha::Ramp { ax, ay, kx, ky },
        _ => MaskAlpha::Empty,
    };
    let plane = usize::try_from(plane).ok();
    if plane.is_some_and(|k| 3 + k >= planes) {
        return Err(PyValueError::new_err("plane out of range"));
    }
    Ok(LocalMask {
        alpha,
        invert,
        plane,
        stops,
        grade,
        flash,
    })
}

/// `rasterise` of one mask spec (as for `local_maps`) to an HxW float32 alpha.
#[pyfunction]
fn rasterise_mask<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    spec: MaskSpec<'py>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let (h, w) = shape;
    let m = local_mask(spec, 4)?;
    let mut alpha = vec![0f32; h * w];
    super::detach(py, alpha.len(), || {
        crate::shapes::rasterise(&m.alpha, (h, w), m.invert, &mut alpha)
    })
    .ok_or_else(|| PyValueError::new_err("invalid mask"))?;
    numpy::PyArray1::from_vec(py, alpha).reshape([h, w])
}

/// `compute_local_maps`' HxWxP float32 planes from masks already in pixel space.
#[pyfunction]
fn local_maps<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize),
    masks: Vec<MaskSpec<'py>>,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let (h, w, planes) = shape;
    let specs = masks
        .into_iter()
        .map(|m| local_mask(m, planes))
        .collect::<PyResult<Vec<_>>>()?;
    let out = PyArrayDyn::<f32>::zeros(py, vec![h, w, planes], false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            crate::shapes::local_maps(&specs, (h, w), planes, dst)
        })
        .ok_or_else(|| PyValueError::new_err("invalid mask"))?;
    }
    Ok(out)
}

/// Brush footprints drawn into one new uint8 mask: for each (points, radius, thickness),
/// `cv2.polylines(mask, [pts], False, 1, thickness)` when there are two points or more, then
/// `cv2.circle(mask, p, radius, 1, -1)` at every point.
#[pyfunction]
fn stroke_cover<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    strokes: Vec<(numpy::PyReadonlyArrayDyn<'py, i32>, i64, i64)>,
) -> PyResult<Bound<'py, numpy::PyArray2<u8>>> {
    let (h, w) = shape;
    let mut mask = vec![0u8; h * w];
    for (pts, radius, thickness) in &strokes {
        let v: Vec<(i64, i64)> = int_points(pts)?
            .iter()
            .map(|&(x, y)| (x as i64, y as i64))
            .collect();
        crate::shapes::stroke_cover(&mut mask, w, h, &v, *radius, *thickness);
    }
    numpy::PyArray1::from_vec(py, mask).reshape([h, w])
}

/// Whether two same-shape float32 alphas both exceed `threshold` at any pixel.
#[pyfunction]
fn alphas_overlap(
    a: numpy::PyReadonlyArrayDyn<'_, f32>,
    b: numpy::PyReadonlyArrayDyn<'_, f32>,
    threshold: f32,
) -> PyResult<bool> {
    use numpy::PyUntypedArrayMethods;
    if a.shape() != b.shape() {
        return Err(PyValueError::new_err("alphas differ in shape"));
    }
    let (Ok(a), Ok(b)) = (a.as_slice(), b.as_slice()) else {
        return Err(PyValueError::new_err("needs contiguous arrays"));
    };
    Ok(crate::shapes::alphas_overlap(a, b, threshold))
}

/// The overlay's zone outline: `cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)`,
/// the contours of at least `min_area` (`cv2.contourArea`), each `cv2.approxPolyDP(c, epsilon, True)`.
#[pyfunction]
fn contour_outlines(
    py: Python<'_>,
    src: PyReadonlyArray2<'_, u8>,
    min_area: f64,
    epsilon: f64,
) -> PyResult<Vec<Vec<(i32, i32)>>> {
    let (h, w) = src.as_array().dim();
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    Ok(super::detach(py, data.len(), || {
        contours::find_list_contours(data, h, w)
            .into_iter()
            .filter(|c| contours::contour_area(c) >= min_area)
            .map(|c| contours::approx_poly_dp(&c, epsilon, true))
            .collect()
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(contour_outlines, m)?)?;
    m.add_function(wrap_pyfunction!(alphas_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(stroke_cover, m)?)?;
    m.add_function(wrap_pyfunction!(local_maps, m)?)?;
    m.add_function(wrap_pyfunction!(rasterise_mask, m)?)?;
    m.add_function(wrap_pyfunction!(fill_poly, m)?)?;
    m.add_function(wrap_pyfunction!(find_external_contours, m)?)?;
    m.add_function(wrap_pyfunction!(min_area_rect, m)?)?;
    m.add_function(wrap_pyfunction!(box_points, m)?)?;
    m.add_function(wrap_pyfunction!(convex_hull, m)?)?;
    m.add_function(wrap_pyfunction!(fill_convex_poly, m)?)
}
