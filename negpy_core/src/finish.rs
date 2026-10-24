//! The filed negative carrier of negpy/features/finish/logic.py (`apply_carrier`), in the
//! float32 order NumPy evaluates it. Python floats join float32 arrays as float32, so each
//! Python-side product (width * rough, r * r, ...) is rounded once before it meets an array.

// The steps index several per-pixel arrays of one run in step.
#![allow(clippy::needless_range_loop)]

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::cbrtf;

// Fractions of the rebate width, mirrored as WGSL literals in finish.wgsl. The picture-side
// boundary is the camera's film gate, machine-cut, so it gets JITTER * INNER_ROUGH and no
// slider. The paper-side one is the hand-filed aperture.
const SAMPLES: usize = 2048;
const JITTER: f64 = 0.24;
const INNER_ROUGH: f64 = 0.2;
const OUTER_JITTER: f64 = 0.4;
const CORNER: f64 = 1.4;
const GATE_CORNER: f64 = 0.2;
// The film never sits centered in the carrier.
const OFFSET_X: f64 = -0.2;
const OFFSET_Y: f64 = -0.25;
const MARGIN: f64 = 0.7;
// The gate prints soft; filed metal is a hard stop, which is what reads as filed.
const SOFT: f64 = 0.22;
const FILED_SOFT: f64 = 0.06;
const FLARE_DEPTH: f64 = 0.25;
const FLARE_GAIN: f64 = 0.6;
const FLARE_BASE: f64 = 0.35;
// A 1-D profile is a height field, so it cannot overhang or shed a fleck. The 2-D field
// displacing the distance field is what makes the edge read as torn metal. Hash noise rather
// than a library, because WGSL has to reproduce it bit for bit.
const NOISE_SEED: u32 = 0x51ED_270B;
const NOISE_CELL: f64 = 0.35;
const NOISE_OCTAVES: usize = 4;
const NOISE_OUTER: f64 = 0.12;
const NOISE_INNER: f64 = 0.02;
const TONE_SAMPLES: usize = 64;

/// `np.clip(v, lo, hi)`: a NaN comes through.
#[inline(always)]
fn clip(v: f32, lo: f32, hi: f32) -> f32 {
    let m = if v.is_nan() || v > lo { v } else { lo };
    if m.is_nan() || m < hi { m } else { hi }
}

/// A whole float32 below 2^22 in magnitude as an integer: added to 1.5 * 2^23, it fills the low
/// mantissa bits. The saturating `as` cast keeps LLVM from vectorizing.
#[inline(always)]
fn whole(v: f32) -> i32 {
    (v + 12_582_912.0).to_bits() as i32 - 0x4B40_0000
}

/// `_hash_lattice`: a lattice value in [-1, 1) from u32 wrap-around arithmetic.
#[inline(always)]
fn hash_lattice(ix: u32, iy: u32) -> f32 {
    let mut h = ix.wrapping_mul(0x27D4_EB2D) ^ iy.wrapping_mul(0x1656_67B1) ^ NOISE_SEED;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 13;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 16;
    (h >> 8) as f32 * (2.0f64 / 16_777_216.0) as f32 - 1.0
}

/// `_value_noise` at one point.
#[inline(always)]
fn value_noise(x: f32, y: f32) -> f32 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ux, uy) = (x - fx, y - fy);
    let ux = ux * ux * (3.0 - 2.0 * ux);
    let uy = uy * uy * (3.0 - 2.0 * uy);
    // astype(int64).astype(uint32) wraps a negative index.
    let (i0, j0) = (whole(fx) as u32, whole(fy) as u32);
    let (i1, j1) = (i0.wrapping_add(1), j0.wrapping_add(1));
    let lo = hash_lattice(i0, j0);
    let lo = lo + (hash_lattice(i1, j0) - lo) * ux;
    let hi = hash_lattice(i0, j1);
    let hi = hi + (hash_lattice(i1, j1) - hi) * ux;
    lo + (hi - lo) * uy
}

/// `carrier_noise`: fBm value noise in [-1, 1], x and y in noise cells.
#[inline(always)]
fn carrier_noise(x: f32, y: f32) -> f32 {
    let (mut total, mut amp, mut freq, mut norm) = (0.0f32, 1.0f32, 1.0f32, 0.0f32);
    for _ in 0..NOISE_OCTAVES {
        total += amp * value_noise(x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    total / norm
}

/// The inputs of `apply_carrier`; `profiles` is `carrier_profiles()` (8 x 2048) and `tone` the
/// (64, 3) rebate tone table.
pub struct Carrier<'a> {
    pub width_px: f64,
    pub rough: f64,
    pub flare: f64,
    pub corner: f64,
    pub paper: [f32; 3],
    pub tone: &'a [[f32; 3]],
    pub profiles: &'a [f32],
}

/// The scalars `_carrier_block` derives from the inputs.
struct Derived {
    soft: f32,
    soft_filed: f32,
    margin: f64,
    radius: f64,
    gate_radius: f64,
    cell: f32,
    off_x: f32,
    off_y: f32,
    reach: f32,
}

/// `arc(r, at, end)`: the corner retreat of a boundary `at` px from the print edge.
#[inline(always)]
fn arc(r: f64, at: f64, end: f32) -> f32 {
    if r <= 0.0 {
        return 0.0;
    }
    let rf = r as f32;
    let x = clip(rf - (end - at as f32), 0.0, rf);
    let s = (r * r) as f32 - x * x;
    rf - (if s.is_nan() || s > 0.0 { s } else { 0.0 }).sqrt()
}

/// `carrier_tone_lookup` of one exposure fraction: np.interp on the table at
/// cbrt(t) * 63, in float64.
#[inline(always)]
fn tone_at(tone: &[[f32; 3]], t: f32) -> [f32; 3] {
    let u = cbrtf(clip(t, 0.0, 1.0)) * (TONE_SAMPLES - 1) as f32;
    let last = TONE_SAMPLES - 1;
    // u is in 0..63 (or NaN, which np.interp passes through).
    let j = (whole(u.floor()).clamp(0, last as i32 - 1)) as usize;
    let x = u as f64;
    std::array::from_fn(|c| {
        let (f0, f1) = (tone[j][c] as f64, tone[j + 1][c] as f64);
        let v = ((f1 - f0) * (x - j as f64) + f0) as f32;
        if x >= last as f64 { tone[last][c] } else { v }
    })
}

impl Carrier<'_> {
    fn derived(&self) -> Derived {
        let w = self.width_px;
        Derived {
            soft: (w * SOFT).max(1.0) as f32,
            soft_filed: (w * FILED_SOFT).max(1.0) as f32,
            margin: w * MARGIN,
            radius: w * CORNER * self.corner,
            gate_radius: w * GATE_CORNER,
            cell: (w * NOISE_CELL).max(1.0) as f32,
            off_x: (w * OFFSET_X) as f32,
            off_y: (w * OFFSET_Y) as f32,
            reach: (w * FLARE_DEPTH).max(1.0) as f32,
        }
    }

    /// The edge band past which no edge has an effect.
    fn band(&self) -> usize {
        let w = self.width_px;
        let extent = MARGIN
            + CORNER * self.corner
            + 1.0
            + JITTER
            + NOISE_OUTER
            + NOISE_INNER
            + OFFSET_X.abs().max(OFFSET_Y.abs());
        (w * extent + (w * SOFT).max(1.0)).ceil() as usize + 1
    }

    #[inline(always)]
    fn prof(&self, row: usize, s: f32) -> f32 {
        // astype(int32) truncates; s is positive.
        let i = whole((s * SAMPLES as f32).floor()).clamp(0, SAMPLES as i32 - 1);
        self.profiles[row * SAMPLES + i as usize]
    }
}

/// Pixels per pass: each step runs as a flat loop over a run, so it vectorizes.
const RUN: usize = 256;

/// `_carrier_block` over the pixels x0.. of row y (`src` and `out`) of an HxW print, for the
/// edges in `sides`, in order, composited and clipped.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn carrier_run(
    c: &Carrier,
    d: &Derived,
    (x0, y): (usize, usize),
    (h, w): (usize, usize),
    sides: &[usize],
    src: &[[f32; 3]],
    out: &mut [[f32; 3]],
) {
    let n = src.len();
    let py = y as f32;
    let (wl, hl) = ((w as f64 - 1.0) as f32, (h as f64 - 1.0) as f32);
    let sy = (py + 0.5) / h as f32;
    let end_y = py.min(hl - py);
    let qy = py - d.off_y;
    let fend_y = qy.min(hl - qy);
    let wr = (c.width_px * c.rough) as f32;
    let mut n2 = [0f32; RUN];
    for i in 0..n {
        n2[i] = carrier_noise((x0 + i) as f32 / d.cell, py / d.cell);
    }
    let (mut a_in, mut a_out) = ([1f32; RUN], [1f32; RUN]);
    let (mut peaks, mut gates) = ([[0f32; RUN]; 4], [[0f32; RUN]; 4]);
    for (k, &e) in sides.iter().enumerate() {
        for i in 0..n {
            let px = (x0 + i) as f32;
            let sx = (px + 0.5) / w as f32;
            let end_x = px.min(wl - px);
            let qx = px - d.off_x;
            let fend_x = qx.min(wl - qx);
            let (s, end, fend, dist, fdist) = match e {
                0 => (sx, end_x, fend_x, py, qy),
                1 => (sx, end_x, fend_x, hl - py, hl - qy),
                2 => (sy, end_y, fend_y, px, qx),
                _ => (sy, end_y, fend_y, wl - px, wl - qx),
            };
            let jitter = OUTER_JITTER as f32 * c.prof(e + 4, s) + NOISE_OUTER as f32 * n2[i];
            let outer = d.margin as f32 + wr * jitter + arc(d.radius, d.margin, fend);
            let wobble = (JITTER * INNER_ROUGH) as f32 * c.prof(e, s) + NOISE_INNER as f32 * n2[i];
            let inner = d.margin as f32
                + c.width_px as f32 * (1.0 + wobble)
                + arc(d.gate_radius, d.margin + c.width_px, end);
            a_in[i] *= clip((dist - inner) / d.soft + 0.5, 0.0, 1.0);
            a_out[i] *= clip((fdist - outer) / d.soft_filed + 0.5, 0.0, 1.0);
            let p = clip(1.0 - (fdist - outer).abs() / d.reach, 0.0, 1.0);
            peaks[k][i] = p * p;
            gates[k][i] = clip(1.0 + (fdist - outer) / d.reach, 0.0, 1.0);
        }
    }
    let mut lit = [0f32; RUN];
    // The bevel spans only the aperture, so the other edges gate each edge's flare.
    if c.flare > 0.0 {
        let m = sides.len();
        for e in 0..m {
            let mut g = [1f32; RUN];
            let mut first = true;
            for k in (0..m).filter(|&k| k != e) {
                for i in 0..n {
                    g[i] = if first {
                        gates[k][i]
                    } else {
                        g[i] * gates[k][i]
                    };
                }
                first = false;
            }
            for i in 0..n {
                lit[i] += peaks[e][i] * g[i];
            }
        }
        // Floored |noise|, not a one-sided gate: that left whole edges with no flare.
        let gain = (c.flare * FLARE_GAIN) as f32;
        for i in 0..n {
            lit[i] *= gain * (FLARE_BASE as f32 + (1.0 - FLARE_BASE) as f32 * n2[i].abs());
        }
    }
    for i in 0..n {
        let t = tone_at(c.tone, a_out[i] + lit[i]);
        let shown = a_in[i] * a_out[i];
        let p = src[i];
        out[i] = std::array::from_fn(|k| {
            clip(p[k] * shown + c.paper[k] * t[k] * (1.0 - shown), 0.0, 1.0)
        });
    }
}

/// `apply_carrier` for width_px > 0: the rebate composited along the print edges, then the
/// whole print clipped to 0..1.
pub fn apply_carrier(src: &[f32], (h, w): (usize, usize), c: &Carrier, dst: &mut [f32]) {
    let d = c.derived();
    let band = c.band();
    let all = 2 * band >= h || 2 * band >= w;
    dst.par_chunks_mut(w * 3).enumerate().for_each(|(y, out)| {
        let row = src[y * w * 3..][..w * 3].as_chunks::<3>().0;
        let out = out.as_chunks_mut::<3>().0;
        let vert = if y < band {
            Some(0)
        } else if y >= h - band {
            Some(1)
        } else {
            None
        };
        // Spans of one side set: the left band, the middle, the right band.
        let spans: [(usize, usize, Option<usize>); 3] = if all {
            [(0, w, None), (w, w, None), (w, w, None)]
        } else {
            [
                (0, band, Some(2)),
                (band, w - band, None),
                (w - band, w, Some(3)),
            ]
        };
        for (a, b, horiz) in spans {
            let mut sides = [0usize; 4];
            let m = if all {
                sides = [0, 1, 2, 3];
                4
            } else {
                let mut m = 0;
                for e in [vert, horiz].into_iter().flatten() {
                    sides[m] = e;
                    m += 1;
                }
                m
            };
            if m == 0 {
                for (o, p) in out[a..b].iter_mut().zip(&row[a..b]) {
                    *o = p.map(|v| clip(v, 0.0, 1.0));
                }
                continue;
            }
            let mut x = a;
            while x < b {
                let e = (x + RUN).min(b);
                carrier_run(
                    c,
                    &d,
                    (x, y),
                    (h, w),
                    &sides[..m],
                    &row[x..e],
                    &mut out[x..e],
                );
                x = e;
            }
        }
    });
}
