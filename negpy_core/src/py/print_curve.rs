use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::print_curve::{self, Diffusion, Keys, Lut, Plane, PrintCurve};

fn contiguous<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))
}

fn triplet<T: numpy::Element + Copy>(a: &PyReadonlyArray1<'_, T>) -> PyResult<[T; 3]> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    s.try_into()
        .map_err(|_| PyValueError::new_err("needs a 3-vector"))
}

/// An HxW plane of the image's height, or an error.
fn plane<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>, h: usize, w: usize) -> PyResult<Plane<'a>> {
    if a.shape().len() < 2 || a.shape()[0] != h || a.shape()[1] != w {
        return Err(PyValueError::new_err(
            "a per-pixel map does not match the image",
        ));
    }
    Ok(Plane {
        data: contiguous(a)?,
        w,
    })
}

/// `_apply_print_curve_kernel` of an HxWx3 float32 image; the arguments are the kernel's.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    img, pivots, slopes, curvatures, toe, shoulder, toe_width, shoulder_width, cmy_offsets,
    shadow_cmy, highlight_cmy, d_min_rgb, d_max, a_toe_base, a_sh_base, width_ref, toe_height,
    sh_height, zone_center, shadow_density, highlight_density, shadow_grade, highlight_grade,
    zone_sh_center, zone_hi_center, zone_k, v_star, midtone_gamma, gamma_width, dye_mix,
    use_dye_mix, sep_k3, sep_damping, sep_ref, use_sep_damping, ev_map, ev_scale, use_ev,
    grade_map, use_grade, key_alpha, key_params, use_key, grade_deltas, r0, r_min, r_max, flash,
    flash_v, flash_gamma, flash_map, use_flash, diff_plane, diff_rect, diff_mix, diff_range,
    use_diff, lut, lut_e_max, lut_black, use_lut, bpc=false, black_hold=0.0
))]
fn apply_print_curve<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    pivots: PyReadonlyArray1<'py, f32>,
    slopes: PyReadonlyArray1<'py, f32>,
    curvatures: PyReadonlyArray1<'py, f32>,
    toe: PyReadonlyArray1<'py, f64>,
    shoulder: PyReadonlyArray1<'py, f64>,
    toe_width: PyReadonlyArray1<'py, f64>,
    shoulder_width: PyReadonlyArray1<'py, f64>,
    cmy_offsets: PyReadonlyArray1<'py, f32>,
    shadow_cmy: PyReadonlyArray1<'py, f32>,
    highlight_cmy: PyReadonlyArray1<'py, f32>,
    d_min_rgb: PyReadonlyArray1<'py, f64>,
    d_max: f64,
    a_toe_base: f64,
    a_sh_base: f64,
    width_ref: f64,
    toe_height: f64,
    sh_height: f64,
    zone_center: f64,
    shadow_density: f64,
    highlight_density: f64,
    shadow_grade: PyReadonlyArray1<'py, f64>,
    highlight_grade: PyReadonlyArray1<'py, f64>,
    zone_sh_center: f64,
    zone_hi_center: f64,
    zone_k: f64,
    v_star: f64,
    midtone_gamma: PyReadonlyArray1<'py, f64>,
    gamma_width: f64,
    dye_mix: PyReadonlyArray2<'py, f64>,
    use_dye_mix: bool,
    sep_k3: PyReadonlyArray1<'py, f64>,
    sep_damping: f64,
    sep_ref: f64,
    use_sep_damping: bool,
    ev_map: PyReadonlyArrayDyn<'py, f32>,
    ev_scale: PyReadonlyArray1<'py, f32>,
    use_ev: bool,
    grade_map: PyReadonlyArrayDyn<'py, f32>,
    use_grade: bool,
    key_alpha: PyReadonlyArrayDyn<'py, f32>,
    key_params: PyReadonlyArray2<'py, f64>,
    use_key: bool,
    grade_deltas: PyReadonlyArrayDyn<'py, f32>,
    r0: f64,
    r_min: f64,
    r_max: f64,
    flash: f64,
    flash_v: f64,
    flash_gamma: f64,
    flash_map: PyReadonlyArrayDyn<'py, f32>,
    use_flash: bool,
    diff_plane: PyReadonlyArrayDyn<'py, f32>,
    diff_rect: PyReadonlyArray1<'py, f64>,
    diff_mix: f64,
    diff_range: PyReadonlyArray1<'py, f64>,
    use_diff: bool,
    lut: PyReadonlyArrayDyn<'py, f32>,
    lut_e_max: PyReadonlyArray1<'py, f32>,
    lut_black: f64,
    use_lut: bool,
    bpc: bool,
    black_hold: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let src = contiguous(&img)?;
    let dye = dye_mix.as_array();
    if use_dye_mix && dye.shape() != [3, 3] {
        return Err(PyValueError::new_err("dye_mix must be 3x3"));
    }
    let keys = if use_key {
        let kp = key_params.as_array();
        let n = kp.shape()[0];
        if kp.shape()[1] != 4 || key_alpha.shape() != [h, w, n] {
            return Err(PyValueError::new_err(
                "key_alpha must be HxWxK with K x 4 key_params",
            ));
        }
        Some((
            contiguous(&key_alpha)?,
            (0..n)
                .map(|i| std::array::from_fn(|j| kp[[i, j]]))
                .collect::<Vec<[f64; 4]>>(),
            plane(&grade_deltas, h, w)?,
        ))
    } else {
        None
    };
    let diffusion = if use_diff {
        let [gh, gw, 3] = diff_plane.shape()[..] else {
            return Err(PyValueError::new_err("diff_plane must be gh x gw x 3"));
        };
        Some(Diffusion {
            plane: contiguous(&diff_plane)?,
            gh,
            gw,
            rect: triplet_n(&diff_rect)?,
            mix: diff_mix,
            range: triplet(&diff_range)?,
        })
    } else {
        None
    };
    let lut = if use_lut {
        let [n, n1, n2, 3] = lut.shape()[..] else {
            return Err(PyValueError::new_err("lut must be N x N x N x 3"));
        };
        if n != n1 || n != n2 || n < 2 {
            return Err(PyValueError::new_err("lut must be N x N x N x 3"));
        }
        Some(Lut {
            table: contiguous(&lut)?,
            n,
            e_max: triplet(&lut_e_max)?,
            black: lut_black,
        })
    } else {
        None
    };
    let p = PrintCurve {
        pivots: triplet(&pivots)?,
        slopes: triplet(&slopes)?,
        curvatures: triplet(&curvatures)?,
        toe: triplet(&toe)?,
        shoulder: triplet(&shoulder)?,
        toe_width: triplet(&toe_width)?,
        shoulder_width: triplet(&shoulder_width)?,
        cmy_offsets: triplet(&cmy_offsets)?,
        shadow_cmy: triplet(&shadow_cmy)?,
        highlight_cmy: triplet(&highlight_cmy)?,
        d_min_rgb: triplet(&d_min_rgb)?,
        d_max,
        a_toe_base,
        a_sh_base,
        width_ref,
        toe_height,
        sh_height,
        zone_center,
        shadow_density,
        highlight_density,
        shadow_grade: triplet(&shadow_grade)?,
        highlight_grade: triplet(&highlight_grade)?,
        zone_sh_center,
        zone_hi_center,
        zone_k,
        v_star,
        midtone_gamma: triplet(&midtone_gamma)?,
        gamma_width,
        dye_mix: use_dye_mix.then(|| std::array::from_fn(|r| std::array::from_fn(|c| dye[[r, c]]))),
        sep: use_sep_damping
            .then(|| triplet(&sep_k3).map(|k| (k, sep_damping, sep_ref)))
            .transpose()?,
        ev: use_ev.then(|| plane(&ev_map, h, w)).transpose()?,
        ev_scale: triplet(&ev_scale)?,
        grade: (use_grade && !use_key)
            .then(|| plane(&grade_map, h, w))
            .transpose()?,
        keys: keys.as_ref().map(|(alpha, params, deltas)| Keys {
            alpha,
            params,
            deltas: *deltas,
        }),
        r0,
        r_min,
        r_max,
        flash,
        flash_v,
        flash_gamma,
        flash_map: use_flash.then(|| plane(&flash_map, h, w)).transpose()?,
        diffusion,
        lut,
        bpc,
        black_hold,
    };
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            print_curve::apply_print_curve(src, w, &p, dst)
        });
    }
    Ok(out)
}

fn triplet_n(a: &PyReadonlyArray1<'_, f64>) -> PyResult<[f64; 4]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?
        .try_into()
        .map_err(|_| PyValueError::new_err("diff_rect must hold 4 values"))
}

/// `tetra_lookup(table, e_max, e0, e1, e2)` of an N x N x N x 3 float32 table.
#[pyfunction]
fn tetra_lookup(
    table: PyReadonlyArrayDyn<'_, f32>,
    e_max: PyReadonlyArray1<'_, f32>,
    e0: f64,
    e1: f64,
    e2: f64,
) -> PyResult<(f64, f64, f64)> {
    let [n, n1, n2, 3] = table.shape()[..] else {
        return Err(PyValueError::new_err("the table must be N x N x N x 3"));
    };
    if n != n1 || n != n2 || n < 2 {
        return Err(PyValueError::new_err("the table must be N x N x N x 3"));
    }
    let l = Lut {
        table: contiguous(&table)?,
        n,
        e_max: triplet(&e_max)?,
        black: 0.0,
    };
    let [r, g, b] = print_curve::tetra(&l, [e0, e1, e2]);
    Ok((r, g, b))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_print_curve, m)?)?;
    m.add_function(wrap_pyfunction!(tetra_lookup, m)?)
}
