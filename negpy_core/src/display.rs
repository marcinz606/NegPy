//! The display transforms of negpy/infrastructure/display/icc_lut.py: a 3D LUT applied by
//! trilinear interpolation, and the matrix/TRC transform between two profiles. Numba types the
//! fractions float64 (a float32 minus an integer), so the blends run in float64.

// The blend steps index several per-pixel arrays of one run in step.
#![allow(clippy::needless_range_loop)]

use multiversion::multiversion;
use rayon::prelude::*;

/// An N x N x N x 3 float32 LUT indexed [r, g, b].
#[derive(Clone, Copy)]
pub struct Lut3<'a> {
    pub table: &'a [f32],
    pub n: usize,
}

/// x (1 - d) + y d, contracted as Numba's fastmath build contracts it.
#[inline(always)]
fn lerp(x: f64, y: f64, d: f64) -> f64 {
    x.mul_add(1.0 - d, y * d)
}

/// The lattice cell and float64 fractions of three scaled float32 coordinates (0 to n - 1).
#[inline(always)]
fn cell(pos: [f32; 3], n: usize) -> ((usize, usize, usize), (f64, f64, f64)) {
    let idx = pos.map(|p| (crate::convert::trunc_bits(p as f64) as usize).min(n - 2));
    let frac: [f64; 3] = std::array::from_fn(|k| pos[k] as f64 - idx[k] as f64);
    ((idx[0], idx[1], idx[2]), (frac[0], frac[1], frac[2]))
}

/// Pixels per pass: the cell lookup and each channel's blend run as flat loops over a run, so
/// the corner reads vectorize as gathers.
const RUN: usize = 256;

/// The LUT blend of a run of scaled coordinates (0 to n - 1), per channel.
#[inline(always)]
fn blend_run(lut: &Lut3, pos: &[[f32; 3]], out: &mut [[f64; RUN]; 3]) {
    let n = lut.n;
    let len = pos.len();
    let mut base = [0usize; RUN];
    let mut frac = [[0f64; RUN]; 3];
    for (i, p) in pos.iter().enumerate() {
        let ((r0, g0, b0), (dr, dg, db)) = cell(*p, n);
        base[i] = (r0 * n + g0) * n + b0;
        frac[0][i] = dr;
        frac[1][i] = dg;
        frac[2][i] = db;
    }
    let t = lut.table;
    let last = t.len() - 1;
    let (sg, sr) = (n, n * n);
    for (c, o) in out.iter_mut().enumerate() {
        for i in 0..len {
            let at = |off: usize| t[((base[i] + off) * 3 + c).min(last)] as f64;
            let (dr, dg, db) = (frac[0][i], frac[1][i], frac[2][i]);
            let c00 = lerp(at(0), at(1), db);
            let c01 = lerp(at(sg), at(sg + 1), db);
            let c10 = lerp(at(sr), at(sr + 1), db);
            let c11 = lerp(at(sr + sg), at(sr + sg + 1), db);
            o[i] = lerp(lerp(c00, c01, dg), lerp(c10, c11, dg), dr);
        }
    }
}

/// `_apply_lut_u16_jit`: a uint16 image through the LUT, rounded half up and clipped.
pub fn apply_lut_u16(img: &[u16], lut: Lut3, out: &mut [u16]) {
    out.par_chunks_mut(3 * 4096)
        .zip(img.par_chunks(3 * 4096))
        .for_each(|(o, i)| lut_u16_block(i, o, lut));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn lut_u16_block(img: &[u16], out: &mut [u16], lut: Lut3) {
    let scale = (lut.n - 1) as f32 / 65535.0;
    let mut pos = [[0f32; 3]; RUN];
    let mut v = [[0f64; RUN]; 3];
    for (o, p) in out.chunks_mut(3 * RUN).zip(img.chunks(3 * RUN)) {
        let (o, p) = (o.as_chunks_mut::<3>().0, p.as_chunks::<3>().0);
        let pos = &mut pos[..p.len()];
        for (q, p) in pos.iter_mut().zip(p) {
            *q = p.map(|x| x as f32 * scale);
        }
        blend_run(&lut, pos, &mut v);
        for (c, v) in v.iter().enumerate() {
            for (o, &x) in o.iter_mut().zip(v) {
                o[c] = crate::convert::trunc_bits((x * 65535.0 + 0.5).clamp(0.0, 65535.0)) as u16;
            }
        }
    }
}

/// `_apply_lut_f32_jit`: a float32 image clipped to 0..1 through the LUT.
pub fn apply_lut_f32(img: &[f32], lut: Lut3, out: &mut [f32]) {
    out.par_chunks_mut(3 * 4096)
        .zip(img.par_chunks(3 * 4096))
        .for_each(|(o, i)| lut_f32_block(i, o, lut));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn lut_f32_block(img: &[f32], out: &mut [f32], lut: Lut3) {
    let scale = (lut.n - 1) as f32;
    let mut pos = [[0f32; 3]; RUN];
    let mut v = [[0f64; RUN]; 3];
    for (o, p) in out.chunks_mut(3 * RUN).zip(img.chunks(3 * RUN)) {
        let (o, p) = (o.as_chunks_mut::<3>().0, p.as_chunks::<3>().0);
        let pos = &mut pos[..p.len()];
        for (q, p) in pos.iter_mut().zip(p) {
            *q = p.map(|x| x.clamp(0.0, 1.0) * scale);
        }
        blend_run(&lut, pos, &mut v);
        for (c, v) in v.iter().enumerate() {
            for (o, &x) in o.iter_mut().zip(v) {
                o[c] = x as f32;
            }
        }
    }
}

/// `_matrix_trc_u8_jit`: TRC decode through `dec` (3 x 256), the 3x3 `m` with a gamut clip,
/// TRC encode through `enc` (3 x K) by linear interpolation, to 8-bit rounded half up.
pub fn matrix_trc_u8(img: &[u8], dec: &[f32], m: [[f32; 3]; 3], enc: &[f32], out: &mut [u8]) {
    out.par_chunks_mut(3 * 4096)
        .zip(img.par_chunks(3 * 4096))
        .for_each(|(o, i)| matrix_trc_block(i, o, dec, m, enc));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn matrix_trc_block(img: &[u8], out: &mut [u8], dec: &[f32], m: [[f32; 3]; 3], enc: &[f32]) {
    let k = enc.len() / 3;
    let kmax = (k - 1) as f64;
    let (dlast, elast) = (dec.len() - 1, enc.len() - 1);
    let mut rgb = [[0f32; 3]; RUN];
    let (mut idx, mut frac) = ([0usize; RUN], [0f64; RUN]);
    for (o, p) in out.chunks_mut(3 * RUN).zip(img.chunks(3 * RUN)) {
        let (o, p) = (o.as_chunks_mut::<3>().0, p.as_chunks::<3>().0);
        let n = p.len();
        for (q, p) in rgb.iter_mut().zip(p) {
            *q = std::array::from_fn(|c| dec[(c * 256 + p[c] as usize).min(dlast)]);
        }
        for c in 0..3 {
            let row = m[c];
            // The index arithmetic vectorizes; the table reads stay scalar, faster than gathers.
            for i in 0..n {
                let q = rgb[i];
                let v = q[2]
                    .mul_add(row[2], q[1].mul_add(row[1], row[0] * q[0]))
                    .clamp(0.0, 1.0);
                let f = v as f64 * kmax;
                let i0 = (crate::convert::trunc_bits(f) as usize).min(k - 2);
                idx[i] = c * k + i0;
                frac[i] = f - i0 as f64;
            }
            for i in 0..n {
                let (lo, hi) = (enc[idx[i].min(elast)], enc[(idx[i] + 1).min(elast)]);
                let e = lerp(lo as f64, hi as f64, frac[i]);
                o[i][c] = crate::convert::trunc_bits((e * 255.0 + 0.5).clamp(0.0, 255.0)) as u8;
            }
        }
    }
}

/// The 16-bit matrix/TRC transform: TRC decode through `dec` (3 x 65536, one entry per code),
/// the 3x3 `m` with a gamut clip, TRC encode through `enc` (3 x K) sampled at linear values
/// t^2 for t evenly spaced in 0..1, so a steep encode near black still interpolates finely;
/// to 16-bit rounded half up.
pub fn matrix_trc_u16(img: &[u16], dec: &[f32], m: [[f32; 3]; 3], enc: &[f32], out: &mut [u16]) {
    out.par_chunks_mut(3 * 4096)
        .zip(img.par_chunks(3 * 4096))
        .for_each(|(o, i)| matrix_trc_u16_block(i, o, dec, m, enc));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn matrix_trc_u16_block(img: &[u16], out: &mut [u16], dec: &[f32], m: [[f32; 3]; 3], enc: &[f32]) {
    let k = enc.len() / 3;
    let kmax = (k - 1) as f64;
    let mut rgb = [[0f32; 3]; RUN];
    let (mut idx, mut frac) = ([0usize; RUN], [0f64; RUN]);
    for (o, p) in out.chunks_mut(3 * RUN).zip(img.chunks(3 * RUN)) {
        let (o, p) = (o.as_chunks_mut::<3>().0, p.as_chunks::<3>().0);
        let n = p.len();
        for (q, p) in rgb.iter_mut().zip(p) {
            *q = std::array::from_fn(|c| dec[c * 65536 + p[c] as usize]);
        }
        for c in 0..3 {
            let row = m[c];
            for i in 0..n {
                let q = rgb[i];
                let v = q[2]
                    .mul_add(row[2], q[1].mul_add(row[1], row[0] * q[0]))
                    .clamp(0.0, 1.0);
                let f = (v as f64).sqrt() * kmax;
                let i0 = (crate::convert::trunc_bits(f) as usize).min(k - 2);
                idx[i] = c * k + i0;
                frac[i] = f - i0 as f64;
            }
            for i in 0..n {
                let e = lerp(enc[idx[i]] as f64, enc[idx[i] + 1] as f64, frac[i]);
                o[i][c] =
                    crate::convert::trunc_bits((e * 65535.0 + 0.5).clamp(0.0, 65535.0)) as u16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_u16_round_trips_every_code() {
        let k = 4097;
        let dec: Vec<f32> = (0..3 * 65536)
            .map(|i| (i % 65536) as f32 / 65535.0)
            .collect();
        let enc: Vec<f32> = (0..3 * k)
            .map(|i| {
                let t = (i % k) as f64 / (k - 1) as f64;
                (t * t) as f32
            })
            .collect();
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let img: Vec<u16> = (0..3 * 30000).map(|i| ((i * 7) % 65536) as u16).collect();
        let mut out = vec![0u16; img.len()];
        matrix_trc_u16(&img, &dec, eye, &enc, &mut out);
        for (a, b) in img.iter().zip(&out) {
            assert!((*a as i32 - *b as i32).abs() <= 1, "{a} -> {b}");
        }
    }
}
