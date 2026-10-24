//! The alternative processes of negpy/features/{sabattier,cyanotype,lith}/logic.py, in NumPy's
//! float32 order. The transcendentals are `mathf`'s, correctly rounded or nearly; NumPy's
//! float32 ones run SVML or its own SIMD code, whose bits differ by CPU.

// The steps index several per-pixel arrays of one run in step.
#![allow(clippy::needless_range_loop)]

use multiversion::multiversion;
use rayon::prelude::*;

use crate::color::{self, LabSpace};
use crate::filter::{self, Border};
use crate::mathf::{cbrtf, exp_f64, exp10_f64, log1p_f64, log10f};

/// `np.clip(v, lo, hi)`: a NaN comes through.
#[inline(always)]
fn clip(v: f32, lo: f32, hi: f32) -> f32 {
    let m = if v.is_nan() || v > lo { v } else { lo };
    if m.is_nan() || m < hi { m } else { hi }
}

#[inline(always)]
fn expf(x: f32) -> f32 {
    exp_f64(x as f64) as f32
}

/// `np.power(np.float32(10.0), x)`.
#[inline(always)]
fn exp10f(x: f32) -> f32 {
    exp10_f64(x as f64) as f32
}

/// Pixels per pass: each step runs as a flat loop over a run, so it vectorizes.
const RUN: usize = 256;

/// The print density of each pixel of a run: `-log10` of the clipped Rec. 709 luma.
#[inline(always)]
fn density_run(px: &[[f32; 3]], out: &mut [f32]) {
    for (o, p) in out.iter_mut().zip(px) {
        let c = |v: f32| clip(v, 1e-6, 1.0);
        let luma = c(p[0]) * 0.2126 + c(p[1]) * 0.7152 + c(p[2]) * 0.0722;
        *o = -log10f(clip(luma, 1e-6, 1.0));
    }
}

/// `np.interp(x, xp, fp)` of float32 anchors, in float64: the slope of each segment, then
/// slope * (x - xp[j]) + fp[j]; the ends hold, and a NaN comes through.
#[derive(Clone, Copy)]
pub struct Interp {
    xp: [f64; 4],
    fp: [f64; 4],
    slope: [f64; 3],
}

impl Interp {
    pub fn new(xp: [f32; 4], fp: [f32; 4]) -> Self {
        let (xp, fp) = (xp.map(|v| v as f64), fp.map(|v| v as f64));
        let slope = std::array::from_fn(|j| (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]));
        Self { xp, fp, slope }
    }

    #[inline(always)]
    fn at(&self, x: f32) -> f32 {
        let (xp, fp, sl) = (self.xp, self.fp, self.slope);
        let x = x as f64;
        let (s, x0, f0) = if x < xp[1] {
            (sl[0], xp[0], fp[0])
        } else if x < xp[2] {
            (sl[1], xp[1], fp[1])
        } else {
            (sl[2], xp[2], fp[2])
        };
        let v = s * (x - x0) + f0;
        let v = if x >= xp[3] { fp[3] } else { v };
        (if x < xp[0] { fp[0] } else { v }) as f32
    }
}

/// L* of a linear grey level (D65, Yn = 1), the CIELAB constants as float32.
#[inline(always)]
fn lightness(grey: f32, eps: f32, kappa: f32) -> f32 {
    let c = cbrtf(grey);
    let lin = kappa * grey + (16.0f64 / 116.0) as f32;
    116.0 * (if grey > eps { c } else { lin }) - 16.0
}

/// Lab of each pixel from `block`, then clipped working RGB.
fn print_lab(
    src: &[f32],
    block: impl Fn(&[[f32; 3]], &mut [[f32; 3]]) + Sync,
    to_rgb: LabSpace,
    dst: &mut [f32],
) {
    let mut lab = vec![0f32; src.len()];
    lab.par_chunks_mut(3 * 4096)
        .zip(src.par_chunks(3 * 4096))
        .for_each(|(o, s)| block(s.as_chunks::<3>().0, o.as_chunks_mut::<3>().0));
    color::lab_to_rgb(&lab, dst, to_rgb);
    dst.par_iter_mut().for_each(|v| *v = clip(*v, 0.0, 1.0));
}

/// The scalars of `apply_sabattier`; float32 ones are rounded as NumPy rounds them.
pub struct Sabattier {
    /// `np.float32(reexposure * d_max)`.
    pub re: f32,
    pub fold_width: f32,
    pub edge_width: f32,
    pub strength: f32,
    pub d_max: f32,
}

/// Density and reversal of a block; with `developed`, the silver the bromide mask reads.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn sabattier_block(
    px: &[[f32; 3]],
    s: &Sabattier,
    dens: &mut [f32],
    rev: &mut [f32],
    developed: Option<&mut [f32]>,
) {
    density_run(px, dens);
    for (r, &d) in rev.iter_mut().zip(dens.iter()) {
        let x = clip((s.re - d) / s.fold_width, -30.0, 30.0);
        *r = s.strength * (s.fold_width * log1p_f64(expf(x) as f64) as f32);
    }
    if let Some(dev) = developed {
        for (o, &d) in dev.iter_mut().zip(dens.iter()) {
            *o = 1.0 / (1.0 + expf(-clip((d - s.re) / s.edge_width, -30.0, 30.0)));
        }
    }
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn sabattier_out(dens: &[f32], rev: &[f32], d_max: f32, out: &mut [[f32; 3]]) {
    for ((o, &d), &r) in out.iter_mut().zip(dens).zip(rev) {
        let v = d + r;
        let m = if v.is_nan() || v < d_max { v } else { d_max };
        *o = [clip(exp10f(-m), 0.0, 1.0); 3];
    }
}

/// `apply_sabattier` once enabled; `k` holds the Mackie-line taps (empty: no line).
pub fn sabattier(
    src: &[f32],
    (h, w): (usize, usize),
    s: &Sabattier,
    k: &[f32],
    dst: &mut [f32],
) -> Option<()> {
    let n = h * w;
    let (mut dens, mut rev) = (vec![0f32; n], vec![0f32; n]);
    let px = src.as_chunks::<3>().0;
    let line = !k.is_empty();
    let mut developed = vec![0f32; if line { n } else { 0 }];
    let blocks = dens
        .par_chunks_mut(RUN)
        .zip(rev.par_chunks_mut(RUN))
        .zip(px.par_chunks(RUN));
    if line {
        blocks
            .zip(developed.par_chunks_mut(RUN))
            .for_each(|(((d, r), p), dev)| sabattier_block(p, s, d, r, Some(dev)));
    } else {
        blocks.for_each(|((d, r), p)| sabattier_block(p, s, d, r, None));
    }
    if line {
        let mut bromide = vec![0f32; n];
        filter::sep_filter2d(
            &developed,
            &mut bromide,
            (h, w, 1),
            (k, k),
            Border::Replicate,
        )?;
        rev.par_iter_mut()
            .zip(bromide.par_iter())
            .for_each(|(r, &b)| *r *= 1.0 - b);
    }
    dst.as_chunks_mut::<3>()
        .0
        .par_chunks_mut(RUN)
        .zip(dens.par_chunks(RUN).zip(rev.par_chunks(RUN)))
        .for_each(|(o, (d, r))| sabattier_out(d, r, s.d_max, o));
    Some(())
}

/// The scalars of `apply_cyanotype`, float32 as NumPy rounds them.
pub struct Cyanotype {
    pub d_max: f32,
    /// `np.float32(0.301 * exposure)` and `np.float32(max(scale, 0.1))`.
    pub over: f32,
    pub scale: f32,
    pub mid_compress: f32,
    pub bleach: f32,
    pub bleach_floor: f32,
    pub tannin: f32,
    pub tannin_restore: f32,
    pub tannin_dmax_gain: f32,
    pub brown: [f32; 2],
    pub a_path: Interp,
    pub b_path: Interp,
    pub lab_eps: f32,
    pub lab_kappa: f32,
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn cyanotype_block(px: &[[f32; 3]], c: &Cyanotype, out: &mut [[f32; 3]]) {
    let (mut base, mut dens, mut frac) = ([0f32; RUN], [0f32; RUN], [0f32; RUN]);
    for (px, out) in px.chunks(RUN).zip(out.chunks_mut(RUN)) {
        let n = px.len();
        density_run(px, &mut base[..n]);
        for i in 0..n {
            let t = clip((base[i] + c.over) / c.scale, 0.0, 1.0);
            let m = c.mid_compress;
            let v = 2.0 * t - 1.0;
            let u0 = (1.0 - m) * t + m * 0.5 * (1.0 + v * v.abs());
            let u_b = u0 * (1.0 - c.bleach * (1.0 - c.bleach_floor * u0));
            let u = u_b + c.tannin * (u0 * c.tannin_restore - u_b);
            dens[i] = c.d_max * (1.0 + c.tannin * c.tannin_dmax_gain) * u;
            frac[i] = clip(u, 0.0, 1.0);
        }
        for i in 0..n {
            out[i][0] = lightness(exp10f(-dens[i]), c.lab_eps, c.lab_kappa);
        }
        for i in 0..n {
            let (f, a, b) = (frac[i], c.a_path.at(frac[i]), c.b_path.at(frac[i]));
            out[i][1] = a + c.tannin * (c.brown[0] * f - a);
            out[i][2] = b + c.tannin * (c.brown[1] * f - b);
        }
    }
}

/// `apply_cyanotype` once enabled.
pub fn cyanotype(src: &[f32], c: &Cyanotype, to_rgb: LabSpace, dst: &mut [f32]) {
    print_lab(src, |px, out| cyanotype_block(px, c, out), to_rgb, dst);
}

/// The scalars of `apply_lith`, float32 as NumPy rounds them.
pub struct Lith {
    pub d_max: f32,
    /// `np.float32(0.301 * exposure)`.
    pub over: f32,
    pub foot_max: f32,
    pub foot_veil: f32,
    pub foot_rate: f32,
    pub knee: f32,
    pub width: f32,
    pub a_path: Interp,
    pub b_path: Interp,
    pub lab_eps: f32,
    pub lab_kappa: f32,
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn lith_block(px: &[[f32; 3]], c: &Lith, out: &mut [[f32; 3]]) {
    let (mut base, mut dens) = ([0f32; RUN], [0f32; RUN]);
    for (px, out) in px.chunks(RUN).zip(out.chunks_mut(RUN)) {
        let n = px.len();
        density_run(px, &mut base[..n]);
        for i in 0..n {
            let d0 = base[i] + c.over;
            let d_foot = base[i] + c.over * c.foot_veil;
            let d_h = c.foot_max * (1.0 - expf(-c.foot_rate * d_foot / c.foot_max));
            let g = 1.0 / (1.0 + expf(-clip((d0 - c.knee) / c.width, -30.0, 30.0)));
            dens[i] = d_h + (c.d_max - d_h) * g;
        }
        for i in 0..n {
            out[i][0] = lightness(exp10f(-dens[i]), c.lab_eps, c.lab_kappa);
        }
        for i in 0..n {
            let frac = clip(dens[i] / c.d_max, 0.0, 1.0);
            out[i][1] = c.a_path.at(frac);
            out[i][2] = c.b_path.at(frac);
        }
    }
}

/// `apply_lith` once enabled.
pub fn lith(src: &[f32], c: &Lith, to_rgb: LabSpace, dst: &mut [f32]) {
    print_lab(src, |px, out| lith_block(px, c, out), to_rgb, dst);
}
