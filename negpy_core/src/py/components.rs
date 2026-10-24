use numpy::{PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::components;

type Labelled<'py> = (
    usize,
    Bound<'py, numpy::PyArrayDyn<i32>>,
    Bound<'py, numpy::PyArray2<i32>>,
    Bound<'py, numpy::PyArray2<f64>>,
);

/// `cv2.connectedComponentsWithStats(mask, connectivity=...)` of an HxW uint8 mask:
/// (count, labels, stats, centroids).
#[pyfunction]
#[pyo3(signature = (mask, connectivity=8))]
fn connected_components_with_stats<'py>(
    py: Python<'py>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    connectivity: u8,
) -> PyResult<Labelled<'py>> {
    let [h, w] = mask.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxW array"));
    };
    if connectivity != 4 && connectivity != 8 {
        return Err(PyValueError::new_err("connectivity must be 4 or 8"));
    }
    let data = mask
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let c = super::detach(py, data.len(), || {
        components::connected_components(data, h, w, connectivity)
    });
    let labels = numpy::PyArray1::from_vec(py, c.labels).reshape(vec![h, w])?;
    let stats =
        numpy::PyArray2::from_vec2(py, &c.stats.iter().map(|s| s.to_vec()).collect::<Vec<_>>())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let centroids = numpy::PyArray2::from_vec2(
        py,
        &c.centroids.iter().map(|s| s.to_vec()).collect::<Vec<_>>(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((c.count, labels, stats, centroids))
}

fn mask_of<'a>(mask: &'a PyReadonlyArrayDyn<'_, u8>) -> PyResult<(&'a [u8], usize, usize)> {
    let [h, w] = mask.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxW array"));
    };
    let data = mask
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    Ok((data, h, w))
}

/// `cv2.distanceTransform(mask, dist_type, mask_size)` for DIST_C (3) and DIST_L2 (3 or 5).
#[pyfunction]
fn distance_transform<'py>(
    py: Python<'py>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    dist_type: i32,
    mask_size: i32,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use crate::distance::{self, Metric};
    let metric = match (dist_type, mask_size) {
        (3, _) => Metric::Chessboard,
        (2, 3) => Metric::L2Mask3,
        (2, 5) => Metric::L2Mask5,
        _ => {
            return Err(PyValueError::new_err(
                "covers DIST_C and DIST_L2 with mask 3 or 5",
            ));
        }
    };
    let (data, h, w) = mask_of(&mask)?;
    let d = super::detach(py, data.len(), || {
        distance::distance_transform(data, h, w, metric)
    });
    numpy::PyArray1::from_vec(py, d).reshape(vec![h, w])
}

type Distances<'py> = (
    Bound<'py, numpy::PyArrayDyn<f32>>,
    Bound<'py, numpy::PyArrayDyn<i32>>,
);

/// `cv2.distanceTransformWithLabels(mask, DIST_L2, _, labelType=DIST_LABEL_PIXEL)`: (dist, labels).
#[pyfunction]
fn distance_transform_labels<'py>(
    py: Python<'py>,
    mask: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Distances<'py>> {
    let (data, h, w) = mask_of(&mask)?;
    let (d, l) = super::detach(py, data.len(), || {
        crate::distance::distance_transform_labels(data, h, w)
    });
    Ok((
        numpy::PyArray1::from_vec(py, d).reshape(vec![h, w])?,
        numpy::PyArray1::from_vec(py, l).reshape(vec![h, w])?,
    ))
}

/// Retouch's `_mask_to_score(mask, pad_px)` with its score floor.
#[pyfunction]
fn mask_to_score<'py>(
    py: Python<'py>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    pad: f64,
    floor: f64,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let (data, h, w) = mask_of(&mask)?;
    let d = super::detach(py, data.len(), || {
        crate::distance::mask_to_score(data, h, w, pad, floor)
    });
    numpy::PyArray1::from_vec(py, d).reshape(vec![h, w])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mask_to_score, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform_labels, m)?)
}
