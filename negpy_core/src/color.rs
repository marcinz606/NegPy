//! The working-space colour kernels of negpy/kernel/image/logic.py: the Adobe RGB TRC and
//! CIELAB. Numba's fastmath build computes them in float64 with FMA contraction; these copy its
//! operation order, and `mathf::pow_f64` stands in for glibc's pow.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::pow_f64;

const BLOCK: usize = 1 << 14;

/// `_oetf_encode_flat`: clamped to 0..1 (NaN stays), raised to `inv_gamma`.
pub fn oetf_encode(src: &[f32], dst: &mut [f32], inv_gamma: f32) {
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| encode_block(s, d, inv_gamma as f64));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn encode_block(src: &[f32], dst: &mut [f32], y: f64) {
    for (d, &v) in dst.iter_mut().zip(src) {
        // clamp keeps NaN, as the kernel did.
        let x = (v as f64).clamp(0.0, 1.0);
        *d = pow_f64(x, y) as f32;
    }
}

/// `_oetf_decode_flat`: negatives to 0 (NaN stays), raised to `gamma`.
pub fn oetf_decode(src: &[f32], dst: &mut [f32], gamma: f32) {
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| decode_block(s, d, gamma as f64));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn decode_block(src: &[f32], dst: &mut [f32], y: f64) {
    for (d, &v) in dst.iter_mut().zip(src) {
        let x = v as f64;
        *d = pow_f64(if x < 0.0 { 0.0 } else { x }, y) as f32;
    }
}

/// `srgb_to_linear`: v / 12.92 up to 0.04045, else ((v + 0.055) / 1.055)^2.4, the power by
/// `mathf::pow_f64` rounded to float32.
pub fn srgb_to_linear(src: &[f32], dst: &mut [f32]) {
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| srgb_block(s, d));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn srgb_block(src: &[f32], dst: &mut [f32]) {
    for (d, &v) in dst.iter_mut().zip(src) {
        let curve = pow_f64(((v + 0.055) / 1.055) as f64, 2.4) as f32;
        // np.where picks the linear branch for v <= 0.04045, NaN included in neither.
        *d = if v <= 0.04045 { v / 12.92 } else { curve };
    }
}

/// The constants of the Lab kernels, as float64.
#[derive(Clone, Copy)]
pub struct LabSpace {
    pub m: [[f64; 3]; 3],
    pub white: [f64; 3],
    pub eps: f64,
    pub kappa: f64,
    pub inv_kappa: f64,
}

impl LabSpace {
    pub fn new(m: [[f32; 3]; 3], white: [f32; 3], eps: f32, kappa: f32) -> Self {
        Self {
            m: m.map(|r| r.map(|v| v as f64)),
            white: white.map(|v| v as f64),
            eps: eps as f64,
            kappa: kappa as f64,
            inv_kappa: 1.0 / kappa as f64,
        }
    }
}

/// `np.float32(16/116)` as float64.
const C: f64 = (16.0f32 / 116.0f32) as f64;

/// One row of the 3x3 product as Numba contracts it.
#[inline(always)]
fn row(m: [f64; 3], x: f64, y: f64, z: f64) -> f64 {
    z.mul_add(m[2], y.mul_add(m[1], x * m[0]))
}

/// `_rgb_to_lab_kernel` over N x 3 linear RGB.
pub fn rgb_to_lab(src: &[f32], dst: &mut [f32], s: LabSpace) {
    dst.par_chunks_mut(BLOCK * 3)
        .zip(src.par_chunks(BLOCK * 3))
        .for_each(|(d, p)| rgb_to_lab_block(p, d, s));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn rgb_to_lab_block(src: &[f32], dst: &mut [f32], s: LabSpace) {
    // Three passes over short runs: the cube root vectorizes only as a flat loop.
    const RUN: usize = 3 * 512;
    let mut t = [0f64; RUN];
    for (d, p) in dst.chunks_mut(RUN).zip(src.chunks(RUN)) {
        let t = &mut t[..p.len()];
        for (t, p) in t
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(p.as_chunks::<3>().0)
        {
            let [r, g, b] = p.map(|v| {
                let v = v as f64;
                if v < 0.0 { 0.0 } else { v }
            });
            *t = std::array::from_fn(|c| row(s.m[c], r, g, b) / s.white[c]);
        }
        for v in t.iter_mut() {
            let root = pow_f64(*v, 1.0 / 3.0);
            let lin = s.kappa.mul_add(*v, C);
            *v = if *v > s.eps { root } else { lin };
        }
        for (d, f) in d
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(t.as_chunks::<3>().0)
        {
            let [fx, fy, fz] = *f;
            d[0] = fy.mul_add(116.0, -16.0) as f32;
            d[1] = ((fx - fy) * 500.0) as f32;
            d[2] = ((fy - fz) * 200.0) as f32;
        }
    }
}

/// `_lab_to_rgb_kernel` over N x 3 Lab: Numba folds the divisions into reciprocal products
/// and `fy - c` into `L / 116 + (16/116 - c)`.
pub fn lab_to_rgb(src: &[f32], dst: &mut [f32], s: LabSpace) {
    dst.par_chunks_mut(BLOCK * 3)
        .zip(src.par_chunks(BLOCK * 3))
        .for_each(|(d, p)| lab_to_rgb_block(p, d, s));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn lab_to_rgb_block(src: &[f32], dst: &mut [f32], s: LabSpace) {
    for (d, &[l, a, b]) in dst
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0)
    {
        let [xr, yr, zr] = lab_to_xyz(l, a, b, &s);
        *d = std::array::from_fn(|c| {
            let v = row(s.m[c], xr, yr, zr);
            if v > 0.0 { v as f32 } else { 0.0 }
        });
    }
}

/// XYZ of one Lab triplet in float64, in the Numba build's order.
#[inline(always)]
fn lab_to_xyz(l: f32, a: f32, b: f32, s: &LabSpace) -> [f64; 3] {
    let inv_kappa = s.inv_kappa;
    let (l, a, b) = (l as f64, a as f64, b as f64);
    let fy_l = l * (1.0 / 116.0);
    let fy = fy_l + 16.0 / 116.0;
    let fx = a.mul_add(0.002, fy);
    let fz = (-b).mul_add(0.005, fy);
    let (fx3, fy3, fz3) = (fx * fx * fx, fy * fy * fy, fz * fz * fz);
    let xr = if fx3 > s.eps {
        fx3 * s.white[0]
    } else {
        ((fx - C) * s.white[0]) * inv_kappa
    };
    let yr = if fy3 > s.eps {
        fy3
    } else {
        (fy_l + (16.0 / 116.0 - C)) * inv_kappa
    } * s.white[1];
    let zr = if fz3 > s.eps {
        fz3
    } else {
        (fz - C) * inv_kappa
    } * s.white[2];
    [xr, yr, zr]
}

// Skin-tone mask: an axis-aligned reduction of the CIELAB skin locus. Hue stays put across the
// tonal range, chroma is bounded, and lightness is free apart from the two ends where the hue
// turns to noise. The chroma window is the measured skin locus, not a gamut bound: sunset,
// terracotta and brick sit inside the hue band, and wood, tan leather and sand at skin's own
// chroma are skin's colour. Neither is separable per pixel.
const HUE_CENTER: f32 = 52.0;
const HUE_WIDTH: f64 = 20.0;
const CHROMA_FULL: f32 = 35.0;
const CHROMA_ZERO: f32 = 60.0;
const L_LO: f32 = 15.0;
const L_HI: f32 = 95.0;
// The ceiling is CEIL_AT_FULL / strength, so it runs past any reachable chroma as strength nears
// 0 and the control fades out without a step. The knee starts at a fraction of the ceiling, so
// ordinary skin below it passes untouched.
const CEIL_AT_FULL: f32 = 22.0;
const KNEE_START: f32 = 0.6;

/// `smoothstep` of a clamped `t`, as float64: Numba returns `float(...)`.
#[inline(always)]
fn clamp01(t: f32) -> f64 {
    if t < 0.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        t as f64
    }
}

/// `_skin_chroma_rein_kernel` over N x 3 Lab. Numba's fastmath build folds the smoothstep
/// divisions into reciprocal constants and regroups the weight's products; this copies its
/// machine code.
pub fn skin_chroma_rein(src: &[f32], dst: &mut [f32], strength: f32) {
    let ceiling = CEIL_AT_FULL / strength;
    let start = ceiling * KNEE_START;
    let span = ceiling * (1.0 - KNEE_START);
    let inv_span = 1.0f32 / span;
    dst.par_chunks_mut(BLOCK * 3)
        .zip(src.par_chunks(BLOCK * 3))
        .for_each(|(d, p)| rein_block(p, d, (start, span, inv_span)));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn rein_block(src: &[f32], dst: &mut [f32], (start, span, inv_span): (f32, f32, f32)) {
    // The skin candidates of each run are gathered first, so their weights compute in one
    // vectorized loop; every other pixel keeps its values (a scale of 1 is exact).
    const RUN: usize = 512;
    let mut idx = [0usize; RUN];
    let mut cand = [[0f32; 4]; RUN];
    let mut scale = [1f64; RUN];
    dst.copy_from_slice(src);
    for (d, p) in dst.chunks_mut(3 * RUN).zip(src.chunks(3 * RUN)) {
        let mut n = 0;
        for (i, &[l, a, b]) in p.as_chunks::<3>().0.iter().enumerate() {
            let chroma = b.mul_add(b, a * a).sqrt();
            idx[n] = i;
            cand[n] = [l, a, b, chroma];
            // `chroma <= start` is unordered-or-less: a NaN chroma keeps the pixel.
            n += !(chroma <= start || chroma < 2.0) as usize;
        }
        for (s, &[l, a, b, chroma]) in scale[..n].iter_mut().zip(&cand[..n]) {
            let w = skin_weight(l, a, b, chroma);
            let gap = start - chroma;
            let e = crate::mathf::exp_f64((gap * inv_span) as f64) as f32;
            let knee_gap = (1.0 - e).mul_add(span, gap);
            let c = chroma as f64;
            let rein = w.mul_add(knee_gap as f64, c) / c;
            *s = if w > 0.0 { rein } else { 1.0 };
        }
        let out = d.as_chunks_mut::<3>().0;
        for ((&i, &[_, a, b, _]), &s) in idx[..n].iter().zip(&cand[..n]).zip(&scale[..n]) {
            out[i][1] = (s * a as f64) as f32;
            out[i][2] = (s * b as f64) as f32;
        }
    }
}

/// `_skin_weight`: 0 below a chroma of 2, where the hue angle is noise.
pub fn skin_weight_of(l: f32, a: f32, b: f32) -> f64 {
    let chroma = b.mul_add(b, a * a).sqrt();
    if chroma < 2.0 {
        0.0
    } else {
        skin_weight(l, a, b, chroma)
    }
}

/// `_skin_weight` for a chroma above 2.
#[inline(always)]
fn skin_weight(l: f32, a: f32, b: f32, chroma: f32) -> f64 {
    let deg = 180.0f32 / std::f32::consts::PI;
    let dist = crate::mathf::atan2f(b, a).mul_add(deg, -HUE_CENTER);
    let turns = (dist * (1.0f32 / 360.0)).round_ties_even();
    let wrapped = (-(turns as f64)).mul_add(360.0, dist as f64);
    let w_hue = crate::mathf::exp_f64(wrapped * wrapped * (-0.5 / (HUE_WIDTH * HUE_WIDTH)));
    let window = CHROMA_ZERO - CHROMA_FULL;
    let tc = clamp01(chroma.mul_add(1.0 / window, -CHROMA_FULL / window));
    let w_chroma = (-(tc * tc)).mul_add((-tc).mul_add(2.0, 3.0), 1.0);
    let t_lo = clamp01(l * (1.0 / L_LO));
    let t_hi = clamp01(l.mul_add(1.0 / (100.0 - L_HI), -L_HI / (100.0 - L_HI)));
    let rise = (w_hue * w_chroma) * ((t_lo * t_lo) * (-t_lo).mul_add(2.0, 3.0));
    let fall = (t_hi * t_hi) * (-t_hi).mul_add(2.0, 3.0);
    (-fall).mul_add(rise, rise)
}

/// `_matmul_3x3_kernel` over N x 3 float32 pixels, contracted as Numba does.
pub fn matmul_3x3(src: &[f32], dst: &mut [f32], m: [[f32; 3]; 3]) {
    dst.par_chunks_mut(BLOCK * 3)
        .zip(src.par_chunks(BLOCK * 3))
        .for_each(|(d, p)| matmul_block(p, d, m));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn matmul_block(src: &[f32], dst: &mut [f32], m: [[f32; 3]; 3]) {
    for (d, &[r, g, b]) in dst
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0)
    {
        *d = m.map(|row| b.mul_add(row[2], g.mul_add(row[1], row[0] * r)));
    }
}

/// `_in_gamut_lab`: (L, a, b) decodes to linear RGB within 0..1, 1e-4 either side; the bounds
/// are float32 sums, compared in float64.
#[inline(always)]
fn in_gamut(l: f32, a: f32, b: f32, s: &LabSpace) -> bool {
    let [xr, yr, zr] = lab_to_xyz(l, a, b, s);
    let (lo, hi) = (-1e-4f32 as f64, (1.0f32 + 1e-4f32) as f64);
    s.m.iter().fold(true, |ok, &m| {
        let v = row(m, xr, yr, zr);
        ok & (v >= lo) & (v <= hi)
    })
}

/// `_gamut_aware_chroma_scale_kernel` over N x 3 Lab: the flat `saturation` where its full push
/// stays in gamut, else a softplus knee toward the pixel's own headroom, found by bisection.
pub fn gamut_aware_chroma_scale(
    src: &[f32],
    dst: &mut [f32],
    saturation: f32,
    iters: usize,
    s: LabSpace,
) {
    dst.par_chunks_mut(BLOCK * 3)
        .zip(src.par_chunks(BLOCK * 3))
        .for_each(|(d, p)| gamut_block(p, d, saturation, iters, &s));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn gamut_block(src: &[f32], dst: &mut [f32], saturation: f32, iters: usize, s: &LabSpace) {
    if saturation <= 1.0 {
        // L times 1 is exact; a flat pattern of 48 lanes vectorizes where triplets do not.
        let pattern: [f32; 48] = std::array::from_fn(|i| if i % 3 == 0 { 1.0 } else { saturation });
        for (d, p) in dst.chunks_mut(48).zip(src.chunks(48)) {
            for ((d, &p), &k) in d.iter_mut().zip(p).zip(&pattern) {
                *d = p * k;
            }
        }
        return;
    }
    // Pixels whose full push clips are gathered per run and bisected in lockstep, so each step
    // vectorizes across them.
    const RUN: usize = 512;
    let mut idx = [0usize; RUN];
    let mut cand = [[0f32; 3]; RUN];
    let mut ok = [false; RUN];
    let mut full = [false; RUN];
    let (mut lo, mut hi) = ([1f32; RUN], [saturation; RUN]);
    for (d, sp) in dst.chunks_mut(3 * RUN).zip(src.chunks(3 * RUN)) {
        let (px, src_px) = (d.as_chunks_mut::<3>().0, sp.as_chunks::<3>().0);
        for (f, &[l, a, b]) in full.iter_mut().zip(src_px) {
            *f = in_gamut(l, a * saturation, b * saturation, s);
        }
        let mut n = 0;
        for (i, ((p, &q), &f)) in px.iter_mut().zip(src_px).zip(&full).enumerate() {
            idx[n] = i;
            cand[n] = q;
            n += !f as usize;
            *p = [q[0], q[1] * saturation, q[2] * saturation];
        }
        for (k, &[l, a, b]) in ok[..n].iter_mut().zip(&cand[..n]) {
            *k = in_gamut(l, a, b, s);
        }
        lo[..n].fill(1.0);
        hi[..n].fill(saturation);
        for _ in 0..iters {
            for j in 0..n {
                let [l, a, b] = cand[j];
                let mid = (lo[j] + hi[j]) / 2.0;
                let fits = ok[j] & in_gamut(l, a * mid, b * mid, s);
                lo[j] = if fits { mid } else { lo[j] };
                hi[j] = if fits { hi[j] } else { mid };
            }
        }
        for j in 0..n {
            let [l, a, b] = cand[j];
            let knee = lo[j].max(1.0f32 + 1e-4f32) - 1.0;
            let e = crate::mathf::exp_f64(((1.0 - saturation) / knee) as f64) as f32;
            let eff = (1.0 - e).mul_add(knee, 1.0);
            px[idx[j]] = [l, a * eff, b * eff];
        }
    }
}
