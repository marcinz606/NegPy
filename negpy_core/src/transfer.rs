//! `apply_transfer_curve` of negpy/features/transparency/logic.py: normalized log density to a
//! scene-linear positive, the render of every slide. Each step keeps NumPy's float32 order; the
//! transcendentals run in float64 (`mathf`) and round once, so they sit within a few ULP of
//! NumPy's SIMD float32 functions and give the same bits on every platform.

use crate::mathf::{exp_f64, exp10_f64, log1p_f64, pow_f64, tanh_f64};
use rayon::prelude::*;

/// Every scalar the curve reads, already in float32 as NumPy's NEP 50 rule rounds them.
#[derive(Clone, Copy)]
pub struct Transfer {
    pub density_range: f32,
    pub cast_gain: [f32; 3],
    pub cast_offset: [f32; 3],
    pub cast_on: [bool; 3],
    pub exposure_offset: f32,
    pub cmy: [f32; 3],
    pub exposure_on: [bool; 3],
    pub contrast: f32,
    pub contrast_on: bool,
    pub pivot: f32,
    pub shadow_cmy: [f32; 3],
    pub highlight_cmy: [f32; 3],
    pub regional_on: [bool; 3],
    pub wb_c: f32,
    pub wb_k: f32,
    pub shadow_density: f32,
    pub highlight_density: f32,
    pub zone_on: bool,
    pub zone_sh: f32,
    pub zone_hi: f32,
    pub zone_k: f32,
    pub black_taper: f32,
    pub toe: [f32; 3],
    pub shoulder: [f32; 3],
    pub toe_width: [f32; 3],
    pub shoulder_width: [f32; 3],
    pub toe_knee: f32,
    pub shoulder_knee: f32,
    pub sep_k: [f32; 3],
    pub sep_on: bool,
    pub cap_level: f32,
    pub cap_offset: f32,
    pub cap_softness: f32,
    pub damping: f32,
    pub damping_on: bool,
    pub ref_spread: f32,
    pub positive_source: bool,
    pub gain: f32,
}

/// `np.maximum(a, b)` with b not NaN: NaN in `a` propagates.
#[inline(always)]
fn np_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || a >= b { a } else { b }
}

/// `_sigmoid`: 0.5 * (1 + tanh(0.5 x)) in float32.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    0.5f32 * (1.0f32 + tanh_f64((0.5f32 * x) as f64) as f32)
}

/// `_softplus`: width * (logaddexp(0, -|t|) + max(t, 0)) with t = x / width.
#[inline(always)]
fn softplus(x: f32, width: f32) -> f32 {
    let t = x / width;
    let lae = log1p_f64(exp_f64(-(t.abs() as f64))) as f32;
    width * (lae + np_max(t, 0.0))
}

/// Pixels per staged run: each step is a flat loop over a run, so the float64 functions vectorize.
const RUN: usize = 256;

impl Transfer {
    /// `shape(d, ch)` over a run: one channel's capture density through every control above
    /// Dye Separation.
    #[inline(always)]
    fn shape(&self, d: &mut [f32], ch: usize) {
        if self.cast_on[ch] {
            let (g, o) = (self.cast_gain[ch], self.cast_offset[ch]);
            d.iter_mut().for_each(|v| *v = *v * g + o);
        }
        if self.exposure_on[ch] {
            let (e, c) = (self.exposure_offset, self.cmy[ch]);
            d.iter_mut().for_each(|v| *v = *v - e + c);
        }
        if self.contrast_on {
            let (p, k) = (self.pivot, self.contrast);
            d.iter_mut().for_each(|v| *v = p + (*v - p) * k);
        }
        if self.regional_on[ch] {
            let (k, c, s, h) = (
                self.wb_k,
                self.wb_c,
                self.shadow_cmy[ch],
                self.highlight_cmy[ch],
            );
            d.iter_mut().for_each(|v| {
                let w_sh = sigmoid(k * (*v - c));
                *v = *v + s * w_sh + h * (1.0f32 - w_sh);
            });
        }
        if self.zone_on {
            let t = *self;
            d.iter_mut().for_each(|v| {
                let x = *v;
                let b = ((t.density_range - x) / t.black_taper).clamp(0.0, 1.0);
                let w_sh = sigmoid(t.zone_k * (x - t.zone_sh)) * (b * b * (3.0f32 - 2.0f32 * b));
                let w_hi = 1.0f32 - sigmoid(t.zone_k * (x - t.zone_hi));
                *v = x + t.shadow_density * w_sh + t.highlight_density * w_hi;
            });
        }
        if self.toe[ch] != 0.0 {
            let (a, k, w) = (self.toe[ch], self.toe_knee, self.toe_width[ch]);
            d.iter_mut().for_each(|v| *v -= a * softplus(*v - k, w));
        }
        if self.shoulder[ch] != 0.0 {
            let (a, k, w) = (
                self.shoulder[ch],
                self.shoulder_knee,
                self.shoulder_width[ch],
            );
            d.iter_mut().for_each(|v| *v += a * softplus(k - *v, w));
        }
    }

    /// Dye Separation over a run: `dens` (the shaped channels) pulled from or toward the
    /// reference, the mean of the capped and shaped channels.
    #[inline(always)]
    fn separate(&self, capture: &[[f32; RUN]; 3], dens: &mut [[f32; RUN]; 3], n: usize) {
        let (soft, level, off) = (self.cap_softness, self.cap_level, self.cap_offset);
        let mut cap = [0f32; RUN];
        for i in 0..n {
            let (a, b, c) = (capture[0][i], capture[1][i], capture[2][i]);
            let lo = if a.is_nan() || b.is_nan() || c.is_nan() {
                f32::NAN
            } else {
                a.min(b).min(c)
            };
            cap[i] = level + softplus(lo + off, soft);
        }
        let mut capped = [[0f32; RUN]; 3];
        for ch in 0..3 {
            for i in 0..n {
                capped[ch][i] = cap[i] - softplus(cap[i] - capture[ch][i], soft);
            }
            self.shape(&mut capped[ch][..n], ch);
        }
        for i in 0..n {
            let d_ref = (capped[0][i] + capped[1][i] + capped[2][i]) / 3.0f32;
            let e: [f32; 3] = std::array::from_fn(|ch| dens[ch][i] - d_ref);
            if self.damping_on {
                let ce: [f32; 3] = std::array::from_fn(|ch| capped[ch][i] - d_ref);
                let (a, b, c) = (ce[0] - ce[1], ce[1] - ce[2], ce[0] - ce[2]);
                let chroma = ((a * a + b * b + c * c) / 3.0f32).sqrt();
                let h = (self.ref_spread - chroma) / (self.ref_spread + chroma);
                let expo = (1.0f32 - self.damping) + self.damping * h;
                for ch in 0..3 {
                    let k = self.sep_k[ch];
                    let kf = pow_f64(k as f64, expo as f64) as f32;
                    let kf = if kf.is_nan() || kf <= 3.0 { kf } else { 3.0 };
                    dens[ch][i] = d_ref + if k <= 0.0 { 0.0 } else { kf } * e[ch];
                }
            } else {
                for ch in 0..3 {
                    dens[ch][i] = d_ref + self.sep_k[ch] * e[ch];
                }
            }
        }
    }
}

#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn transfer_block(t: &Transfer, src: &[[f32; 3]], dst: &mut [[f32; 3]]) {
    let mut capture = [[0f32; RUN]; 3];
    let mut dens = [[0f32; RUN]; 3];
    for (o, p) in dst.chunks_mut(RUN).zip(src.chunks(RUN)) {
        let n = p.len();
        for ch in 0..3 {
            for i in 0..n {
                capture[ch][i] = p[i][ch] * t.density_range;
            }
            dens[ch][..n].copy_from_slice(&capture[ch][..n]);
            t.shape(&mut dens[ch][..n], ch);
        }
        if t.sep_on {
            t.separate(&capture, &mut dens, n);
        }
        for ch in 0..3 {
            for i in 0..n {
                let v = exp10_f64(-(dens[ch][i] as f64)) as f32;
                o[i][ch] = if t.positive_source {
                    v.clamp(0.0, 1.0)
                } else {
                    let x = np_max(v * t.gain, 0.0);
                    let num = x * (2.51f32 * x + 0.03f32);
                    let den = x * (2.43f32 * x + 0.59f32) + 0.14f32;
                    (num / np_max(den, 1e-8)).clamp(0.0, 1.0)
                };
            }
        }
    }
}

/// The transfer curve of an HxWx3 normalized image into `dst`.
pub fn apply_transfer_curve(src: &[f32], t: &Transfer, dst: &mut [f32]) {
    let t = *t;
    dst.par_chunks_mut(3 * 4096)
        .zip(src.par_chunks(3 * 4096))
        .for_each(move |(d, s)| transfer_block(&t, s.as_chunks::<3>().0, d.as_chunks_mut::<3>().0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_is_relu_far_from_the_knee() {
        assert!((softplus(10.0, 0.45) - 10.0).abs() < 1e-5);
        assert!(softplus(-10.0, 0.45) < 1e-6);
        assert!((softplus(0.0, 1.0) - std::f32::consts::LN_2).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_is_half_at_zero() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(40.0) > 0.999_99);
    }

    #[test]
    fn np_max_keeps_nan() {
        assert!(np_max(f32::NAN, 0.0).is_nan());
        assert_eq!(np_max(-1.0, 0.0), 0.0);
    }
}
