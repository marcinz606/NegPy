//! The sensor unmix kernels of negpy/features/process/sensor.py, in the types Numba gives them:
//! float32 arrays, and float64 where a Python float scalar joins the arithmetic.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::{exp_f64, ln_f64};

pub type Matrix = [[f32; 3]; 3];

/// One row of the 3x3 product in float32, contracted as Numba does.
#[inline(always)]
fn row(m: [f32; 3], x: [f32; 3]) -> f32 {
    x[2].mul_add(m[2], x[1].mul_add(m[1], m[0] * x[0]))
}

/// `_unmix_kernel`: the unmix, clipped at zero.
pub fn unmix(src: &[f32], dst: &mut [f32], m: Matrix) {
    dst.par_chunks_mut(3 * 4096)
        .zip(src.par_chunks(3 * 4096))
        .for_each(|(d, s)| unmix_block(s, d, m));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn unmix_block(src: &[f32], dst: &mut [f32], m: Matrix) {
    for (d, &x) in dst
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0)
    {
        *d = m.map(|r| {
            let v = row(r, x);
            if v < 0.0 { 0.0 } else { v }
        });
    }
}

/// `_log_unmix_kernel`: exp(c ln x + offset) per pixel, each channel held at its floor or above.
pub fn log_unmix(src: &[f32], dst: &mut [f32], c: Matrix, offset: [f32; 3], floor: [f32; 3]) {
    dst.par_chunks_mut(3 * 4096)
        .zip(src.par_chunks(3 * 4096))
        .for_each(|(d, s)| log_unmix_block(s, d, c, offset, floor));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn log_unmix_block(src: &[f32], dst: &mut [f32], c: Matrix, offset: [f32; 3], floor: [f32; 3]) {
    for (d, x) in dst
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0)
    {
        let l: [f32; 3] = std::array::from_fn(|k| {
            let v = if floor[k] > x[k] { floor[k] } else { x[k] };
            ln_f64(v as f64) as f32
        });
        // Numba adds the offset first.
        *d = std::array::from_fn(|k| {
            let a = l[2].mul_add(
                c[k][2],
                l[1].mul_add(c[k][1], l[0].mul_add(c[k][0], offset[k])),
            );
            exp_f64(a as f64) as f32
        });
    }
}

/// The scalars of the two-scale layer pass, Python floats (float64) in the kernel.
#[derive(Clone, Copy)]
pub struct Layer {
    pub floor: f64,
    pub log_floor: f64,
    pub start: f64,
    pub full: f64,
}

/// `_two_scale_layer_kernel`: the soft-floored unmix of the blurred capture into `color`, and
/// the blend weight, a smoothstep on the largest unmix gain over its gain at the film base.
pub fn two_scale_layer(
    blurred: &[f32],
    m: Matrix,
    own: [f32; 3],
    s: Layer,
    color: &mut [f32],
    weight: &mut [f32],
) {
    color
        .par_chunks_mut(3 * 4096)
        .zip(weight.par_chunks_mut(4096))
        .zip(blurred.par_chunks(3 * 4096))
        .for_each(|((c, w), b)| layer_block(b, m, own, s, c, w));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn layer_block(
    blurred: &[f32],
    m: Matrix,
    own: [f32; 3],
    s: Layer,
    color: &mut [f32],
    weight: &mut [f32],
) {
    for ((c, w), x) in color
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(weight.iter_mut())
        .zip(blurred.as_chunks::<3>().0)
    {
        let mut ratio = 0.0f64;
        for k in 0..3 {
            let u = row(m[k], *x);
            let gain = (x[k] * own[k]) as f64 / (u as f64).max(s.log_floor);
            ratio = if gain > ratio { gain } else { ratio };
            let f = s.floor * m[k][k] as f64 * x[k] as f64;
            let d = u as f64 - f;
            c[k] = (0.5 * (u as f64 + f + (d * d + f * f).sqrt())).max(s.log_floor) as f32;
        }
        let t = ((ratio - s.start) / (s.full - s.start)).clamp(0.0, 1.0);
        *w = (t * t * (3.0 - 2.0 * t)) as f32;
    }
}

/// `_two_scale_kernel`: linear, blended by `weight` toward `color` carrying the detail
/// x / blurred through `c`; a channel whose detail sits `log_defect` below both others takes
/// their mean.
#[allow(clippy::too_many_arguments)]
pub fn two_scale(
    x: &[f32],
    blurred: &[f32],
    color: &[f32],
    weight: &[f32],
    m: Matrix,
    c: Matrix,
    (floor, log_defect): (f64, f64),
    out: &mut [f32],
) {
    out.par_chunks_mut(3 * 4096)
        .zip(x.par_chunks(3 * 4096))
        .zip(blurred.par_chunks(3 * 4096))
        .zip(color.par_chunks(3 * 4096))
        .zip(weight.par_chunks(4096))
        .for_each(|((((o, x), b), col), w)| {
            two_scale_block(x, b, col, w, m, c, (floor, log_defect), o)
        });
}

#[allow(clippy::too_many_arguments)]
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn two_scale_block(
    x: &[f32],
    blurred: &[f32],
    color: &[f32],
    weight: &[f32],
    m: Matrix,
    c: Matrix,
    (floor, log_defect): (f64, f64),
    out: &mut [f32],
) {
    let (x, b, col) = (
        x.as_chunks::<3>().0,
        blurred.as_chunks::<3>().0,
        color.as_chunks::<3>().0,
    );
    for (i, o) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let wt = weight[i];
        let mut detail = [0f64; 3];
        if wt > 0.0 {
            let e: [f64; 3] = std::array::from_fn(|k| {
                ln_f64((x[i][k] as f64).max(floor) / (b[i][k] as f64).max(floor))
            });
            let pick = |k: usize, p: usize, q: usize| {
                if e[k] < log_defect + e[p].min(e[q]) {
                    0.5 * (e[p] + e[q])
                } else {
                    e[k]
                }
            };
            detail = [pick(0, 1, 2), pick(1, 0, 2), pick(2, 0, 1)];
        }
        *o = std::array::from_fn(|k| {
            let v = row(m[k], x[i]);
            let mut lin = if v < 0.0 { 0.0 } else { v } as f64;
            if wt > 0.0 {
                let gain = c[k][0] as f64 * detail[0]
                    + c[k][1] as f64 * detail[1]
                    + c[k][2] as f64 * detail[2];
                let capped = col[i][k] as f64 * exp_f64(gain);
                lin += wt as f64 * (capped - lin);
            }
            lin as f32
        });
    }
}

/// The scalars of `film_base`, as Python floats.
pub struct FilmBase {
    pub percentile: f64,
    pub clip_plateau: f64,
    pub grid_edge: f64,
    pub log_floor: f32,
    pub share: f64,
}

/// `film_base` of its analysis crop (h x w x 3): the median log of the cells thinnest in all
/// three channels together, back through `exp`. `None` when the thin end is clipped. The
/// final `exp` of the three medians is libm's; NumPy's float64 exp runs SVML on AVX-512.
pub fn film_base(sample: &[f32], (h, w): (usize, usize), c: &FilmBase) -> Option<[f64; 3]> {
    if h * w == 0 {
        return None;
    }
    let clipped =
        (0..3).into_par_iter().any(|ch| {
            let vals: Vec<f32> = sample.iter().skip(ch).step_by(3).copied().collect();
            let Some(p) = crate::stats::np_percentiles(&vals, &[c.percentile]) else {
                return true;
            };
            let max = vals.iter().copied().fold(f32::NEG_INFINITY, |m, v| {
                if v > m || v.is_nan() { v } else { m }
            });
            p[0] >= c.clip_plateau as f32 * max
        });
    if clipped {
        return None;
    }
    let scale = c.grid_edge / h.max(w) as f64;
    let small;
    let (cells, (sh, sw)) = if scale < 1.0 {
        let dw = ((w as f64 * scale).round_ties_even() as usize).max(1);
        let dh = ((h as f64 * scale).round_ties_even() as usize).max(1);
        small = crate::resize::resize_area(sample, h, w, 3, dh, dw)?;
        (&small[..], (dh, dw))
    } else {
        (sample, (h, w))
    };
    let n = sh * sw;
    let floor = c.log_floor;
    let logs: Vec<f64> = cells
        .par_iter()
        .map(|&v| crate::mathf::np_logf(if v.is_nan() || v > floor { v } else { floor }) as f64)
        .collect();
    let q = c.percentile / 100.0;
    let thin_end: Vec<f64> = (0..3)
        .into_par_iter()
        .map(|ch| {
            let mut col: Vec<f64> = logs.iter().skip(ch).step_by(3).copied().collect();
            crate::stats::quantile_f64(&mut col, q).unwrap_or(f64::NAN)
        })
        .collect();
    let thin: Vec<f64> = (0..n)
        .map(|i| {
            (0..3).fold(f64::INFINITY, |m, ch| {
                let v = logs[3 * i + ch] - thin_end[ch];
                if v < m || v.is_nan() { v } else { m }
            })
        })
        .collect();
    let bar = crate::stats::quantile_f64(&mut thin.clone(), 1.0 - c.share)?;
    Some(std::array::from_fn(|ch| {
        let mut sel: Vec<f64> = (0..n)
            .filter(|&i| thin[i] >= bar)
            .map(|i| logs[3 * i + ch])
            .collect();
        libm::exp(crate::stats::median_f64(&mut sel))
    }))
}
