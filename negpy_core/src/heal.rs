//! Multi-scale PatchMatch and Image Melding in log-transmittance space for manual heal strokes.
//!
//! Deterministic across platforms and core counts via SplitMix64 and fixed-order relaxation.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::{np_expf, np_logf};
use crate::morph::{self, Kernel, Op};
use crate::stats::median;

/// Upper bound on a heal brush radius in pixels (guards arithmetic and circle rasterization).
pub const MAX_HEAL_BRUSH: i64 = 4096;

const LOG_FLOOR: f32 = 1.0e-5;
const LOG_CEIL: f32 = 64.0;
const LUMA_R: f32 = 0.2974;
const LUMA_G: f32 = 0.6273;
const LUMA_B: f32 = 0.0753;

const R_PATCH: isize = 2; // 5x5 patches
const SCREEN_LAMBDA: f32 = 0.08;
const N_XFORM: usize = 25;

#[derive(Clone, Copy)]
struct XformSpec {
    a_f: f32,
    b_f: f32,
    a_q8: i32,
    b_q8: i32,
    pen: f32,
}

/// Deterministic 2D similarity table (`a = s cos(theta), b = s sin(theta)`):
/// `[0]` exact integer identity (`0°, 1.00x`), `[1..=14]` small-angle rotations (`±6°..±45°` at `1.00x`),
/// and `[15..=24]` tight perspective foreshortening scales (`0.88x` and `1.14x` at `0°, ±8°, ±18°`).
/// Stored as exact Q8 fixed-point and IEEE-754 f32 literals so bits never depend on libm/CPU trig.
#[allow(clippy::excessive_precision)]
const XFORM_TABLE: [XformSpec; N_XFORM] = [
    XformSpec {
        a_f: 1.00000000,
        b_f: 0.00000000,
        a_q8: 256,
        b_q8: 0,
        pen: 0.00000000,
    }, // [ 0]  +0 deg, 1.00x
    XformSpec {
        a_f: 0.99452190,
        b_f: 0.10452846,
        a_q8: 255,
        b_q8: 27,
        pen: 0.02631595,
    }, // [ 1]  +6 deg, 1.00x
    XformSpec {
        a_f: 0.99452190,
        b_f: -0.10452846,
        a_q8: 255,
        b_q8: -27,
        pen: 0.02631595,
    }, // [ 2]  -6 deg, 1.00x
    XformSpec {
        a_f: 0.97814760,
        b_f: 0.20791169,
        a_q8: 250,
        b_q8: 53,
        pen: 0.03026379,
    }, // [ 3] +12 deg, 1.00x
    XformSpec {
        a_f: 0.97814760,
        b_f: -0.20791169,
        a_q8: 250,
        b_q8: -53,
        pen: 0.03026379,
    }, // [ 4] -12 deg, 1.00x
    XformSpec {
        a_f: 0.95105652,
        b_f: 0.30901699,
        a_q8: 243,
        b_q8: 79,
        pen: 0.03684353,
    }, // [ 5] +18 deg, 1.00x
    XformSpec {
        a_f: 0.95105652,
        b_f: -0.30901699,
        a_q8: 243,
        b_q8: -79,
        pen: 0.03684353,
    }, // [ 6] -18 deg, 1.00x
    XformSpec {
        a_f: 0.91354546,
        b_f: 0.40673664,
        a_q8: 234,
        b_q8: 104,
        pen: 0.04605516,
    }, // [ 7] +24 deg, 1.00x
    XformSpec {
        a_f: 0.91354546,
        b_f: -0.40673664,
        a_q8: 234,
        b_q8: -104,
        pen: 0.04605516,
    }, // [ 8] -24 deg, 1.00x
    XformSpec {
        a_f: 0.86602540,
        b_f: 0.50000000,
        a_q8: 222,
        b_q8: 128,
        pen: 0.05789868,
    }, // [ 9] +30 deg, 1.00x
    XformSpec {
        a_f: 0.86602540,
        b_f: -0.50000000,
        a_q8: 222,
        b_q8: -128,
        pen: 0.05789868,
    }, // [10] -30 deg, 1.00x
    XformSpec {
        a_f: 0.78801075,
        b_f: 0.61566148,
        a_q8: 202,
        b_q8: 158,
        pen: 0.07778411,
    }, // [11] +38 deg, 1.00x
    XformSpec {
        a_f: 0.78801075,
        b_f: -0.61566148,
        a_q8: 202,
        b_q8: -158,
        pen: 0.07778411,
    }, // [12] -38 deg, 1.00x
    XformSpec {
        a_f: core::f32::consts::FRAC_1_SQRT_2,
        b_f: core::f32::consts::FRAC_1_SQRT_2,
        a_q8: 181,
        b_q8: 181,
        pen: 0.09902203,
    }, // [13] +45 deg, 1.00x
    XformSpec {
        a_f: core::f32::consts::FRAC_1_SQRT_2,
        b_f: -core::f32::consts::FRAC_1_SQRT_2,
        a_q8: 181,
        b_q8: -181,
        pen: 0.09902203,
    }, // [14] -45 deg, 1.00x
    XformSpec {
        a_f: 0.88000000,
        b_f: 0.00000000,
        a_q8: 225,
        b_q8: 0,
        pen: 0.03317069,
    }, // [15]  +0 deg, 0.88x
    XformSpec {
        a_f: 0.87143590,
        b_f: 0.12247233,
        a_q8: 223,
        b_q8: 31,
        pen: 0.03551015,
    }, // [16]  +8 deg, 0.88x
    XformSpec {
        a_f: 0.87143590,
        b_f: -0.12247233,
        a_q8: 223,
        b_q8: -31,
        pen: 0.03551015,
    }, // [17]  -8 deg, 0.88x
    XformSpec {
        a_f: 0.83692973,
        b_f: 0.27193496,
        a_q8: 214,
        b_q8: 70,
        pen: 0.04501421,
    }, // [18] +18 deg, 0.88x
    XformSpec {
        a_f: 0.83692973,
        b_f: -0.27193496,
        a_q8: 214,
        b_q8: -70,
        pen: 0.04501421,
    }, // [19] -18 deg, 0.88x
    XformSpec {
        a_f: 1.14000000,
        b_f: 0.00000000,
        a_q8: 292,
        b_q8: 0,
        pen: 0.03358420,
    }, // [20]  +0 deg, 1.14x
    XformSpec {
        a_f: 1.12890560,
        b_f: 0.15865734,
        a_q8: 289,
        b_q8: 41,
        pen: 0.03592366,
    }, // [21]  +8 deg, 1.14x
    XformSpec {
        a_f: 1.12890560,
        b_f: -0.15865734,
        a_q8: 289,
        b_q8: -41,
        pen: 0.03592366,
    }, // [22]  -8 deg, 1.14x
    XformSpec {
        a_f: 1.08420443,
        b_f: 0.35227937,
        a_q8: 278,
        b_q8: 90,
        pen: 0.04542773,
    }, // [23] +18 deg, 1.14x
    XformSpec {
        a_f: 1.08420443,
        b_f: -0.35227937,
        a_q8: 278,
        b_q8: -90,
        pen: 0.04542773,
    }, // [24] -18 deg, 1.14x
];

const XFORM_NEIGHBORS: [(u8, u8); N_XFORM] = [
    (1, 2),
    (0, 3),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 8),
    (5, 9),
    (6, 10),
    (7, 11),
    (8, 12),
    (9, 13),
    (10, 14),
    (11, 15),
    (12, 20),
    (0, 16),
    (15, 18),
    (15, 19),
    (16, 5),
    (17, 6),
    (0, 21),
    (20, 23),
    (20, 24),
    (21, 5),
    (22, 6),
];

#[derive(Clone, Copy, PartialEq, Eq)]
struct NnfEntry {
    qx_q8: i32,
    qy_q8: i32,
    rot: u8,
}

impl NnfEntry {
    #[inline]
    fn int(qx: i32, qy: i32) -> Self {
        Self {
            qx_q8: qx * 256,
            qy_q8: qy * 256,
            rot: 0,
        }
    }

    #[inline]
    fn with_xform(qx: i32, qy: i32, rot: u8) -> Self {
        Self {
            qx_q8: qx * 256,
            qy_q8: qy * 256,
            rot,
        }
    }

    #[inline]
    fn nearest_px(self) -> (i32, i32) {
        ((self.qx_q8 + 128) >> 8, (self.qy_q8 + 128) >> 8)
    }

    #[inline]
    fn step_target(self, sx: i32, sy: i32) -> Self {
        if self.rot == 0 {
            Self {
                qx_q8: self.qx_q8 + sx * 256,
                qy_q8: self.qy_q8 + sy * 256,
                rot: 0,
            }
        } else {
            let spec = &XFORM_TABLE[self.rot as usize];
            Self {
                qx_q8: self.qx_q8 + sx * spec.a_q8 - sy * spec.b_q8,
                qy_q8: self.qy_q8 + sx * spec.b_q8 + sy * spec.a_q8,
                rot: self.rot,
            }
        }
    }
}

const CELL_NEUMANN: u8 = 0;
const CELL_INTERIOR: u8 = 1;
const CELL_DIRICHLET: u8 = 2;

const HARD_SKIRT: u8 = 1;
const HARD_DEAD: u8 = 2;
const HARD_AVOID: u8 = 4;
const HARD_EXCL: u8 = HARD_DEAD | HARD_AVOID;

#[inline]
fn valid_rgb_buf(buf: &[f32], (h, w): (usize, usize)) -> bool {
    h.checked_mul(w)
        .and_then(|n| n.checked_mul(3))
        .is_some_and(|len| len == buf.len())
}

#[inline]
fn context_pad(brush: i64, min_span: f64) -> isize {
    let bf = brush as f64;
    let extra = (0.4 * min_span + 2.0 * bf).min(120.0);
    (bf + (7.6 * bf).clamp(20.0, 384.0).max(extra)) as isize
}

/// Liang-Barsky segment clip of `(ax, ay) -> (bx, by)` against `[xmin, xmax] x [ymin, ymax]`.
fn clip_segment_f64(
    (ax, ay): (f64, f64),
    (bx, by): (f64, f64),
    (xmin, ymin, xmax, ymax): (f64, f64, f64, f64),
) -> Option<((f64, f64), (f64, f64))> {
    if !ax.is_finite() || !ay.is_finite() || !bx.is_finite() || !by.is_finite() {
        return None;
    }
    let dx = bx - ax;
    let dy = by - ay;
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    for (p, q) in [
        (-dx, ax - xmin),
        (dx, xmax - ax),
        (-dy, ay - ymin),
        (dy, ymax - ay),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                if r > t1 {
                    return None;
                }
                if r > t0 {
                    t0 = r;
                }
            } else {
                if r < t0 {
                    return None;
                }
                if r < t1 {
                    t1 = r;
                }
            }
        }
    }
    Some(((ax + t0 * dx, ay + t0 * dy), (ax + t1 * dx, ay + t1 * dy)))
}

/// Rasterize `pts` into `mask`, clipping extreme off-crop segments before `shapes::stroke_cover`.
fn rasterize_stroke_clipped(
    mask: &mut [u8],
    cw: usize,
    ch: usize,
    pts: &[(i64, i64)],
    radius: i64,
) {
    if pts.is_empty() || cw == 0 || ch == 0 || radius < 1 {
        return;
    }
    let pad = 2 * radius + 4;
    let x_lo = -pad;
    let y_lo = -pad;
    let x_hi = cw as i64 + pad;
    let y_hi = ch as i64 + pad;
    if pts
        .iter()
        .all(|&(px, py)| px >= x_lo && px <= x_hi && py >= y_lo && py <= y_hi)
    {
        crate::shapes::stroke_cover(mask, cw, ch, pts, radius, 2 * radius);
        return;
    }
    if pts.len() == 1 {
        let (px, py) = pts[0];
        if px >= -radius && px < cw as i64 + radius && py >= -radius && py < ch as i64 + radius {
            crate::shapes::circle(mask, cw, ch, (px, py), radius, 1, true);
        }
        return;
    }
    let bounds = (x_lo as f64, y_lo as f64, x_hi as f64, y_hi as f64);
    for pair in pts.windows(2) {
        let a = (pair[0].0 as f64, pair[0].1 as f64);
        let b = (pair[1].0 as f64, pair[1].1 as f64);
        if let Some((ca, cb)) = clip_segment_f64(a, b, bounds) {
            let seg = [
                (libm::rint(ca.0) as i64, libm::rint(ca.1) as i64),
                (libm::rint(cb.0) as i64, libm::rint(cb.1) as i64),
            ];
            crate::shapes::stroke_cover(mask, cw, ch, &seg, radius, 2 * radius);
        }
    }
}

/// One manual heal stroke in buffer pixel coordinates.
pub struct HealStroke<'a> {
    /// Working crop `(x0, y0, x1, y1)` in the full buffer.
    pub rect: (usize, usize, usize, usize),
    /// Smoothed polyline in crop coordinates.
    pub path: &'a [(i64, i64)],
    /// Brush radius in whole pixels (`>= 1`).
    pub brush: i64,
    /// Optional `ch * cw` mask of pixels to exclude from the donor pool (other strokes).
    pub avoid: Option<&'a [u8]>,
}

#[inline]
fn reflect101(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let mut m = idx % period;
    if m < 0 {
        m += period;
    }
    if m >= len as isize {
        m = period - m;
    }
    m as usize
}

#[inline]
fn sanitize_trans(v: f32) -> f32 {
    if !v.is_finite() {
        if v.is_nan() || v <= 0.0 {
            LOG_FLOOR
        } else {
            LOG_CEIL
        }
    } else {
        v.clamp(LOG_FLOOR, LOG_CEIL)
    }
}

#[inline]
fn luma_log(v: [f32; 3]) -> f32 {
    (v[0] * LUMA_R + v[1] * LUMA_G) + v[2] * LUMA_B
}

/// Deterministic SplitMix64 PRNG so PatchMatch random search produces identical bits on every OS.
struct Rng64(u64);

impl Rng64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 32) as u32
    }

    #[inline]
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        if hi <= lo {
            return lo;
        }
        let span = (hi - lo + 1) as u32;
        lo + (self.next_u32() % span) as i32
    }
}

/// Separable `3x3` `[1, 2, 1] / 4` structure filter (`sigma ≈ 0.85 px`) used ONLY inside fine-level
/// `patch_distance` so PatchMatch compares underlying structure rather than penalizing authentic
/// donor film grain against the smooth Screened-Poisson hole interior.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn smooth3_rgb(img: &[[f32; 3]], (h, w): (usize, usize)) -> Vec<[f32; 3]> {
    let n = h * w;
    let mut tmp = vec![[0.0f32; 3]; n];
    if w >= 2 {
        for y in 0..h {
            let row = y * w;
            let a0 = img[row];
            let c0 = img[row + 1];
            tmp[row] = [
                0.25 * (3.0 * a0[0] + c0[0]),
                0.25 * (3.0 * a0[1] + c0[1]),
                0.25 * (3.0 * a0[2] + c0[2]),
            ];
            for x in 1..w - 1 {
                let a = img[row + x - 1];
                let b = img[row + x];
                let c = img[row + x + 1];
                tmp[row + x] = [
                    0.25 * (a[0] + 2.0 * b[0] + c[0]),
                    0.25 * (a[1] + 2.0 * b[1] + c[1]),
                    0.25 * (a[2] + 2.0 * b[2] + c[2]),
                ];
            }
            let al = img[row + w - 2];
            let bl = img[row + w - 1];
            tmp[row + w - 1] = [
                0.25 * (al[0] + 3.0 * bl[0]),
                0.25 * (al[1] + 3.0 * bl[1]),
                0.25 * (al[2] + 3.0 * bl[2]),
            ];
        }
    } else {
        tmp.copy_from_slice(img);
    }
    let mut out = vec![[0.0f32; 3]; n];
    for y in 0..h {
        let ym = y.saturating_sub(1) * w;
        let yr = y * w;
        let yp = (y + 1).min(h - 1) * w;
        let sa = &tmp[ym..ym + w];
        let sb = &tmp[yr..yr + w];
        let sc = &tmp[yp..yp + w];
        let dst = &mut out[yr..yr + w];
        for (((d, &a), &b), &c) in dst.iter_mut().zip(sa).zip(sb).zip(sc) {
            *d = [
                0.25 * (a[0] + 2.0 * b[0] + c[0]),
                0.25 * (a[1] + 2.0 * b[1] + c[1]),
                0.25 * (a[2] + 2.0 * b[2] + c[2]),
            ];
        }
    }
    out
}

#[inline]
fn bilerp_rgb(buf: &[[f32; 3]], (h, w): (usize, usize), sx_q8: i32, sy_q8: i32) -> [f32; 3] {
    let x0 = reflect101((sx_q8 >> 8) as isize, w);
    let x1 = reflect101(((sx_q8 >> 8) + 1) as isize, w);
    let y0 = reflect101((sy_q8 >> 8) as isize, h);
    let y1 = reflect101(((sy_q8 >> 8) + 1) as isize, h);
    let fx = (sx_q8 & 255) as f32 * (1.0 / 256.0);
    let fy = (sy_q8 & 255) as f32 * (1.0 / 256.0);
    let wx0 = 1.0 - fx;
    let wy0 = 1.0 - fy;
    let w00 = wy0 * wx0;
    let w01 = wy0 * fx;
    let w10 = fy * wx0;
    let w11 = fy * fx;
    let p00 = buf[y0 * w + x0];
    let p01 = buf[y0 * w + x1];
    let p10 = buf[y1 * w + x0];
    let p11 = buf[y1 * w + x1];
    [
        (w00 * p00[0] + w01 * p01[0]) + (w10 * p10[0] + w11 * p11[0]),
        (w00 * p00[1] + w01 * p01[1]) + (w10 * p10[1] + w11 * p11[1]),
        (w00 * p00[2] + w01 * p01[2]) + (w10 * p10[2] + w11 * p11[2]),
    ]
}

#[inline]
fn bilerp_f32(buf: &[f32], (h, w): (usize, usize), sx_q8: i32, sy_q8: i32) -> f32 {
    let x0 = reflect101((sx_q8 >> 8) as isize, w);
    let x1 = reflect101(((sx_q8 >> 8) + 1) as isize, w);
    let y0 = reflect101((sy_q8 >> 8) as isize, h);
    let y1 = reflect101(((sy_q8 >> 8) + 1) as isize, h);
    let fx = (sx_q8 & 255) as f32 * (1.0 / 256.0);
    let fy = (sy_q8 & 255) as f32 * (1.0 / 256.0);
    let wx0 = 1.0 - fx;
    let wy0 = 1.0 - fy;
    (wy0 * wx0 * buf[y0 * w + x0] + wy0 * fx * buf[y0 * w + x1])
        + (fy * wx0 * buf[y1 * w + x0] + fy * fx * buf[y1 * w + x1])
}

#[inline]
fn catmull_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ]
}

/// Monotonic `4x4` Catmull-Rom bicubic sampler clamped to the `2x2` cell bounds so rotated curved
/// edges stay razor-sharp without ringing halos while retaining `~95%+` of 4K dye-cloud grain.
#[inline]
fn catmull_rom_rgb(buf: &[[f32; 3]], (h, w): (usize, usize), sx_q8: i32, sy_q8: i32) -> [f32; 3] {
    let ix = (sx_q8 >> 8) as isize;
    let iy = (sy_q8 >> 8) as isize;
    let fx = (sx_q8 & 255) as f32 * (1.0 / 256.0);
    let fy = (sy_q8 & 255) as f32 * (1.0 / 256.0);
    let wx = catmull_weights(fx);
    let wy = catmull_weights(fy);

    let xs = [
        reflect101(ix - 1, w),
        reflect101(ix, w),
        reflect101(ix + 1, w),
        reflect101(ix + 2, w),
    ];
    let mut acc = [0.0f32; 3];
    for (ky, &wy_k) in wy.iter().enumerate() {
        let ry = reflect101(iy + ky as isize - 1, h) * w;
        let p0 = buf[ry + xs[0]];
        let p1 = buf[ry + xs[1]];
        let p2 = buf[ry + xs[2]];
        let p3 = buf[ry + xs[3]];
        for c in 0..3 {
            let row_v = (wx[0] * p0[c] + wx[1] * p1[c]) + (wx[2] * p2[c] + wx[3] * p3[c]);
            acc[c] += wy_k * row_v;
        }
    }
    let y1 = reflect101(iy, h) * w;
    let y2 = reflect101(iy + 1, h) * w;
    let c00 = buf[y1 + xs[1]];
    let c01 = buf[y1 + xs[2]];
    let c10 = buf[y2 + xs[1]];
    let c11 = buf[y2 + xs[2]];
    for c in 0..3 {
        let lo = c00[c].min(c01[c]).min(c10[c].min(c11[c]));
        let hi = c00[c].max(c01[c]).max(c10[c].max(c11[c]));
        acc[c] = acc[c].clamp(lo, hi);
    }
    acc
}

/// Separable `(2*rad + 1) x (2*rad + 1)` sliding-window box filter in `O(N)` time, independent of `rad`.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn smooth_box_rgb(img: &[[f32; 3]], rad: isize, (h, w): (usize, usize)) -> Vec<[f32; 3]> {
    let r = rad.max(1);
    let inv = 1.0 / (2 * r + 1) as f32;
    let n = h * w;
    let mut tmp = vec![[0.0f32; 3]; n];
    for y in 0..h {
        let row = y * w;
        let mut acc = [0.0f32; 3];
        for k in -r..=r {
            let v = img[row + reflect101(k, w)];
            acc[0] += v[0];
            acc[1] += v[1];
            acc[2] += v[2];
        }
        tmp[row] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        for x in 1..w {
            let v_out = img[row + reflect101(x as isize - r - 1, w)];
            let v_in = img[row + reflect101(x as isize + r, w)];
            acc[0] += v_in[0] - v_out[0];
            acc[1] += v_in[1] - v_out[1];
            acc[2] += v_in[2] - v_out[2];
            tmp[row + x] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        }
    }
    let mut out = vec![[0.0f32; 3]; n];
    for x in 0..w {
        let mut acc = [0.0f32; 3];
        for k in -r..=r {
            let v = tmp[reflect101(k, h) * w + x];
            acc[0] += v[0];
            acc[1] += v[1];
            acc[2] += v[2];
        }
        out[x] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        for y in 1..h {
            let v_out = tmp[reflect101(y as isize - r - 1, h) * w + x];
            let v_in = tmp[reflect101(y as isize + r, h) * w + x];
            acc[0] += v_in[0] - v_out[0];
            acc[1] += v_in[1] - v_out[1];
            acc[2] += v_in[2] - v_out[2];
            out[y * w + x] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        }
    }
    out
}

#[inline]
fn entries_coherent(e_from: NnfEntry, sx: i32, sy: i32, e_to: NnfEntry) -> bool {
    let pred = e_from.step_target(sx, sy);
    if e_to.rot == 0 {
        pred == e_to
    } else {
        pred.rot == e_to.rot
            && (pred.qx_q8 - e_to.qx_q8).abs() <= 192
            && (pred.qy_q8 - e_to.qy_q8).abs() <= 192
    }
}

#[inline]
fn sample_donor_step_l0(
    l0: &[[f32; 3]],
    clean_src: &[bool],
    (h, w): (usize, usize),
    de: NnfEntry,
    dx: i32,
    dy: i32,
) -> Option<[f32; 3]> {
    if de.rot == 0 {
        let sx = (de.qx_q8 >> 8) + dx;
        let sy = (de.qy_q8 >> 8) + dy;
        if sx < 0 || sx >= w as i32 || sy < 0 || sy >= h as i32 {
            return None;
        }
        let sp = sy as usize * w + sx as usize;
        if !clean_src[sp] {
            return None;
        }
        Some(l0[sp])
    } else {
        let spec = &XFORM_TABLE[de.rot as usize];
        let sx_q8 = de.qx_q8 + dx * spec.a_q8 - dy * spec.b_q8;
        let sy_q8 = de.qy_q8 + dx * spec.b_q8 + dy * spec.a_q8;
        let sx = (sx_q8 + 128) >> 8;
        let sy = (sy_q8 + 128) >> 8;
        if sx < 0 || sx >= w as i32 || sy < 0 || sy >= h as i32 {
            return None;
        }
        if !clean_src[sy as usize * w + sx as usize] {
            return None;
        }
        Some(catmull_rom_rgb(l0, (h, w), sx_q8, sy_q8))
    }
}

#[inline]
fn sample_donor_step_smooth(
    sm: &[[f32; 3]],
    (h, w): (usize, usize),
    de: NnfEntry,
    dx: i32,
    dy: i32,
) -> [f32; 3] {
    if de.rot == 0 {
        let sx = reflect101(((de.qx_q8 >> 8) + dx) as isize, w);
        let sy = reflect101(((de.qy_q8 >> 8) + dy) as isize, h);
        sm[sy * w + sx]
    } else {
        let spec = &XFORM_TABLE[de.rot as usize];
        let sx_q8 = de.qx_q8 + dx * spec.a_q8 - dy * spec.b_q8;
        let sy_q8 = de.qy_q8 + dx * spec.b_q8 + dy * spec.a_q8;
        bilerp_rgb(sm, (h, w), sx_q8, sy_q8)
    }
}

/// Exterior-only paired wide tent collar shift at Dirichlet rim pixel `np = (nx, ny)` relative to
/// stepped donor `de_ext = de_p.step_target(dx, dy)`: compares ONLY clean exterior film on both
/// sides (`clean_src == true`, `cover == 0`) over `±rad_ext` so:
/// 1. Poisson's interior hole blur never attenuates thin curves, wires, or sharp step edges;
/// 2. Sub-pixel flank offsets across narrow `2-5 px` curves cancel out symmetrically (`9x9` `ref_d`);
/// 3. `3-12 px` dye-cloud clumps and bokeh along the collar are not subtracted out of `copied`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn exterior_collar_shift(
    l0: &[[f32; 3]],
    s5_l0: &[[f32; 3]],
    clean_src: &[bool],
    (h, w): (usize, usize),
    (nx, ny): (usize, usize),
    de_ext: NnfEntry,
    rad_ext: isize,
    grain_var: f32,
) -> [f32; 3] {
    let np = ny * w + nx;
    let sv_sm0 = sample_donor_step_smooth(s5_l0, (h, w), de_ext, 0, 0);
    let seed_d = [
        s5_l0[np][0] - sv_sm0[0],
        s5_l0[np][1] - sv_sm0[1],
        s5_l0[np][2] - sv_sm0[2],
    ];
    let seed_scale = 1.5 / (1.0 + 3.5 * grain_var);
    let collar_scale = 4.5 / (1.0 + 6.5 * grain_var);

    let mut ref_acc = [0.0f32; 3];
    let mut ref_w = 0.0f32;
    for oy in -4isize..=4 {
        let ty = ny as isize + oy;
        if ty < 0 || ty >= h as isize {
            continue;
        }
        let wy = (5 - oy.abs()) as f32;
        for ox in -4isize..=4 {
            let tx = nx as isize + ox;
            if tx < 0 || tx >= w as isize {
                continue;
            }
            let tp = ty as usize * w + tx as usize;
            if !clean_src[tp] {
                continue;
            }
            let Some(sv) =
                sample_donor_step_l0(l0, clean_src, (h, w), de_ext, ox as i32, oy as i32)
            else {
                continue;
            };
            let d0 = l0[tp][0] - sv[0];
            let d1 = l0[tp][1] - sv[1];
            let d2 = l0[tp][2] - sv[2];
            let s0 = d0 - seed_d[0];
            let s1 = d1 - seed_d[1];
            let s2 = d2 - seed_d[2];
            let u = (s0 * s0 + s1 * s1 + s2 * s2) * seed_scale;
            let wt = wy * (5 - ox.abs()) as f32 / (1.0 + u * u);
            ref_acc[0] += wt * d0;
            ref_acc[1] += wt * d1;
            ref_acc[2] += wt * d2;
            ref_w += wt;
        }
    }
    let ref_d = if ref_w > 1.0e-6 {
        let inv = 1.0 / ref_w;
        [ref_acc[0] * inv, ref_acc[1] * inv, ref_acc[2] * inv]
    } else {
        seed_d
    };

    let r_lim = rad_ext.max(6);
    let r_top = (r_lim + 1) as f32;
    let mut acc = [0.0f32; 3];
    let mut w_sum = 0.0f32;
    for oy in -r_lim..=r_lim {
        let ty = ny as isize + oy;
        if ty < 0 || ty >= h as isize {
            continue;
        }
        let wy = r_top - oy.abs() as f32;
        for ox in -r_lim..=r_lim {
            let tx = nx as isize + ox;
            if tx < 0 || tx >= w as isize {
                continue;
            }
            let tp = ty as usize * w + tx as usize;
            if !clean_src[tp] {
                continue;
            }
            let Some(sv) =
                sample_donor_step_l0(l0, clean_src, (h, w), de_ext, ox as i32, oy as i32)
            else {
                continue;
            };
            let d0 = l0[tp][0] - sv[0];
            let d1 = l0[tp][1] - sv[1];
            let d2 = l0[tp][2] - sv[2];
            let r0 = d0 - ref_d[0];
            let r1 = d1 - ref_d[1];
            let r2 = d2 - ref_d[2];
            let u = (r0 * r0 + r1 * r1 + r2 * r2) * collar_scale;
            let wt = wy * (r_top - ox.abs() as f32) / (1.0 + u * u);
            acc[0] += wt * d0;
            acc[1] += wt * d1;
            acc[2] += wt * d2;
            w_sum += wt;
        }
    }
    if w_sum > 1.0e-6 {
        let inv = 1.0 / w_sum;
        [acc[0] * inv, acc[1] * inv, acc[2] * inv]
    } else {
        ref_d
    }
}

/// Scale-invariant Screened-Poisson shading solver for `phi = u - copied` on `CELL_INTERIOR`:
/// 1. Inside every coherent NNF island (`entries_coherent == true`), `Delta phi = 0`, preserving
///    `100%` of native grain, dye-cloud clumping, and crisp donor edges/curves across any format;
/// 2. Across incoherent internal NNF seams, `Delta phi` cancels `copied[n] - copied[p]` while
///    retaining macro scene transitions (`finest_img[n] - finest_img[p]`) and donor high-pass steps;
/// 3. Screening `lambda_shade` and `macro_rad` scale dimensionlessly with `hole_depth`, while
///    `exterior_collar_shift` anchors rims purely on clean exterior film without interior blur bias.
#[allow(clippy::too_many_arguments)]
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn solve_donor_shading(
    finest_img: &[[f32; 3]],
    l0: &[[f32; 3]],
    s5_l0: &[[f32; 3]],
    hole_depth: f32,
    grain_floor: (f32, f32, f32, f32),
    bstate: &[u8],
    clean_src: &[bool],
    has_donor: &[bool],
    donors: &[NnfEntry],
    (h, w): (usize, usize),
) -> Vec<[f32; 3]> {
    let n = h * w;
    if !has_donor.iter().any(|&b| b) {
        return vec![[0.0f32; 3]; n];
    }
    let depth_px = hole_depth.clamp(0.0, 4096.0) as isize;
    let lambda_shade = (0.25 / (hole_depth * hole_depth + 16.0)).clamp(0.001, 0.008);
    let macro_rad = (depth_px + 4).clamp(8, 18);
    let rad_ext = (depth_px + 5).clamp(9, 15);
    let grain_var = grain_floor.0.max(grain_floor.3) + grain_floor.2;
    let macro_mix = (2.8 * grain_floor.2).clamp(0.0, 0.55);
    let fine_mix = 1.0 - macro_mix;

    let shade_state_owned: Option<Vec<u8>> =
        if (0..n).any(|p| bstate[p] == CELL_INTERIOR && !has_donor[p]) {
            Some(
                bstate
                    .iter()
                    .enumerate()
                    .map(|(p, &st)| {
                        if st == CELL_INTERIOR && !has_donor[p] {
                            CELL_NEUMANN
                        } else {
                            st
                        }
                    })
                    .collect(),
            )
        } else {
            None
        };
    let eff_bstate = shade_state_owned.as_deref().unwrap_or(bstate);

    let cells = collect_interior_cells(eff_bstate, lambda_shade, (h, w));
    if cells.is_empty() {
        return vec![[0.0f32; 3]; n];
    }

    let s_macro_l0 = smooth_box_rgb(s5_l0, macro_rad, (h, w));
    let mut phi = vec![[0.0f32; 3]; n];
    {
        let s5_fin = smooth3_rgb(&smooth3_rgb(finest_img, (h, w)), (h, w));
        let s_macro_fin = smooth_box_rgb(&s5_fin, macro_rad, (h, w));
        drop(s5_fin);
        for &c_info in &cells {
            let p = c_info.p;
            let mq = sample_donor_step_smooth(&s_macro_l0, (h, w), donors[p], 0, 0);
            phi[p] = [
                s_macro_fin[p][0] - mq[0],
                s_macro_fin[p][1] - mq[1],
                s_macro_fin[p][2] - mq[2],
            ];
        }
    }
    let seam_l0: Vec<[f32; 3]> = s5_l0
        .iter()
        .zip(&s_macro_l0)
        .map(|(&f, &m)| {
            [
                fine_mix * f[0] + macro_mix * m[0],
                fine_mix * f[1] + macro_mix * m[1],
                fine_mix * f[2] + macro_mix * m[2],
            ]
        })
        .collect();
    drop(s_macro_l0);

    let mut rhs = Vec::with_capacity(cells.len());
    for &c_info in &cells {
        let p = c_info.p;
        let mut b = [
            lambda_shade * phi[p][0],
            lambda_shade * phi[p][1],
            lambda_shade * phi[p][2],
        ];
        let de_p = donors[p];
        let s5_p0 = sample_donor_step_smooth(&seam_l0, (h, w), de_p, 0, 0);

        let faces: [(bool, usize, i32, i32); 4] = [
            ((c_info.mask & 1) != 0, p.wrapping_sub(w), 0, -1),
            ((c_info.mask & 2) != 0, p + w, 0, 1),
            ((c_info.mask & 4) != 0, p.wrapping_sub(1), -1, 0),
            ((c_info.mask & 8) != 0, p + 1, 1, 0),
        ];
        for (enabled, np, dx, dy) in faces {
            if !enabled {
                continue;
            }
            if eff_bstate[np] == CELL_DIRICHLET {
                let ext_sh = exterior_collar_shift(
                    l0,
                    s5_l0,
                    clean_src,
                    (h, w),
                    (np % w, np / w),
                    de_p.step_target(dx, dy),
                    rad_ext,
                    grain_var,
                );
                for c in 0..3 {
                    b[c] += ext_sh[c];
                }
            } else if eff_bstate[np] == CELL_INTERIOR {
                let de_n = donors[np];
                if !entries_coherent(de_p, dx, dy, de_n) {
                    let s5_pn = sample_donor_step_smooth(&seam_l0, (h, w), de_p, dx, dy);
                    let s5_n0 = sample_donor_step_smooth(&seam_l0, (h, w), de_n, 0, 0);
                    let s5_np = sample_donor_step_smooth(&seam_l0, (h, w), de_n, -dx, -dy);
                    let j0 = 0.5 * (s5_n0[0] + s5_np[0]) - 0.5 * (s5_p0[0] + s5_pn[0]);
                    let j1 = 0.5 * (s5_n0[1] + s5_np[1]) - 0.5 * (s5_p0[1] + s5_pn[1]);
                    let j2 = 0.5 * (s5_n0[2] + s5_np[2]) - 0.5 * (s5_p0[2] + s5_pn[2]);
                    let u = (j0 * j0 + j1 * j1 + j2 * j2) * 2.5;
                    let w_seam = 1.0 / (1.0 + u * u);
                    b[0] += j0 * w_seam;
                    b[1] += j1 * w_seam;
                    b[2] += j2 * w_seam;
                }
            }
        }
        rhs.push(b);
    }
    drop(seam_l0);

    let (iters_sor, omega_sor) = if cells.len() <= 256 {
        (36, 1.65)
    } else if hole_depth <= 28.0 {
        (56, 1.65)
    } else {
        (((2.2 * hole_depth) as usize).clamp(64, 220), 1.82)
    };
    relax_poisson_sparse(&mut phi, &cells, Some(&rhs), w, iters_sor, omega_sor);
    relax_poisson_sparse(&mut phi, &cells, Some(&rhs), w, 6, 1.0);
    phi
}

#[derive(Clone, Copy)]
struct InteriorCell {
    p: usize,
    mask: u8,
    inv_denom: f32,
}

fn collect_interior_cells(state: &[u8], lambda: f32, (h, w): (usize, usize)) -> Vec<InteriorCell> {
    let mut cells = Vec::new();
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let p = row + x;
            if state[p] != CELL_INTERIOR {
                continue;
            }
            let mut mask = 0u8;
            let mut deg = 0u32;
            if y > 0 && state[p - w] != CELL_NEUMANN {
                mask |= 1;
                deg += 1;
            }
            if y + 1 < h && state[p + w] != CELL_NEUMANN {
                mask |= 2;
                deg += 1;
            }
            if x > 0 && state[p - 1] != CELL_NEUMANN {
                mask |= 4;
                deg += 1;
            }
            if x + 1 < w && state[p + 1] != CELL_NEUMANN {
                mask |= 8;
                deg += 1;
            }
            let denom = deg as f32 + lambda;
            let inv_denom = if denom > 1.0e-6 { 1.0 / denom } else { 0.0 };
            cells.push(InteriorCell { p, mask, inv_denom });
        }
    }
    cells
}

/// Multi-grid Gauss-Seidel / SOR solver for screened Poisson:
/// `(deg(p) + lambda) * u(p) - sum_{n ~ p} u(n) = rhs(p)` on `CELL_INTERIOR`,
/// with `u = data` fixed on `CELL_DIRICHLET` and zero normal flux on `CELL_NEUMANN`.
fn solve_screened_poisson(
    state: &[u8],
    data: &[[f32; 3]],
    gx: &[[f32; 3]],
    gy: &[[f32; 3]],
    lambda: f32,
    (h, w): (usize, usize),
) -> Vec<[f32; 3]> {
    let cells = collect_interior_cells(state, lambda, (h, w));
    let mut rhs = Vec::with_capacity(cells.len());
    for &c_info in &cells {
        let p = c_info.p;
        let m = c_info.mask;
        let mut b = [
            lambda * data[p][0],
            lambda * data[p][1],
            lambda * data[p][2],
        ];
        if (m & 1) != 0 {
            for c in 0..3 {
                b[c] += 0.5 * (gy[p][c] + gy[p - w][c]);
            }
        }
        if (m & 2) != 0 {
            for c in 0..3 {
                b[c] -= 0.5 * (gy[p][c] + gy[p + w][c]);
            }
        }
        if (m & 4) != 0 {
            for c in 0..3 {
                b[c] += 0.5 * (gx[p][c] + gx[p - 1][c]);
            }
        }
        if (m & 8) != 0 {
            for c in 0..3 {
                b[c] -= 0.5 * (gx[p][c] + gx[p + 1][c]);
            }
        }
        rhs.push(b);
    }

    let int_count = cells.len();
    let mut u = solve_harmonic_init(state, data, (h, w));
    let iters_sor = if int_count <= 160 {
        16
    } else if int_count <= 800 {
        24
    } else {
        32
    };
    relax_poisson_sparse(&mut u, &cells, Some(&rhs), w, iters_sor, 1.55);
    relax_poisson_sparse(&mut u, &cells, Some(&rhs), w, 4, 1.0);
    u
}

fn solve_harmonic_init(state: &[u8], bc: &[[f32; 3]], (h, w): (usize, usize)) -> Vec<[f32; 3]> {
    let n = h * w;
    let mut u = vec![[0.0f32; 3]; n];
    let mut dir_sum = [0.0f64; 3];
    let mut dir_count = 0usize;
    let mut int_count = 0usize;
    for p in 0..n {
        match state[p] {
            CELL_DIRICHLET => {
                u[p] = bc[p];
                dir_sum[0] += bc[p][0] as f64;
                dir_sum[1] += bc[p][1] as f64;
                dir_sum[2] += bc[p][2] as f64;
                dir_count += 1;
            }
            CELL_INTERIOR => int_count += 1,
            _ => {}
        }
    }
    if int_count == 0 {
        return u;
    }
    let mean_bc = if dir_count > 0 {
        [
            (dir_sum[0] / dir_count as f64) as f32,
            (dir_sum[1] / dir_count as f64) as f32,
            (dir_sum[2] / dir_count as f64) as f32,
        ]
    } else {
        [0.0; 3]
    };
    let cells = collect_interior_cells(state, 0.0, (h, w));

    if h.min(w) > 16 && int_count > 64 && dir_count > 0 {
        let (ch, cw) = (h.div_ceil(2), w.div_ceil(2));
        let mut c_state = vec![CELL_NEUMANN; ch * cw];
        let mut c_bc = vec![[0.0f32; 3]; ch * cw];
        for cy in 0..ch {
            for cx in 0..cw {
                let mut d_acc = [0.0f32; 3];
                let mut d_n = 0u32;
                let mut i_acc = [0.0f32; 3];
                let mut i_n = 0u32;
                for dy in 0..2 {
                    let fy = 2 * cy + dy;
                    if fy >= h {
                        continue;
                    }
                    for dx in 0..2 {
                        let fx = 2 * cx + dx;
                        if fx >= w {
                            continue;
                        }
                        let fp = fy * w + fx;
                        match state[fp] {
                            CELL_DIRICHLET => {
                                d_acc[0] += bc[fp][0];
                                d_acc[1] += bc[fp][1];
                                d_acc[2] += bc[fp][2];
                                d_n += 1;
                            }
                            CELL_INTERIOR => {
                                i_acc[0] += bc[fp][0];
                                i_acc[1] += bc[fp][1];
                                i_acc[2] += bc[fp][2];
                                i_n += 1;
                            }
                            _ => {}
                        }
                    }
                }
                let cp = cy * cw + cx;
                if d_n > 0 {
                    let inv = 1.0 / d_n as f32;
                    c_state[cp] = CELL_DIRICHLET;
                    c_bc[cp] = [d_acc[0] * inv, d_acc[1] * inv, d_acc[2] * inv];
                } else if i_n > 0 {
                    let inv = 1.0 / i_n as f32;
                    c_state[cp] = CELL_INTERIOR;
                    c_bc[cp] = [i_acc[0] * inv, i_acc[1] * inv, i_acc[2] * inv];
                }
            }
        }
        let coarse = solve_harmonic_init(&c_state, &c_bc, (ch, cw));
        for &c_info in &cells {
            let p = c_info.p;
            let cp = ((p / w) / 2) * cw + ((p % w) / 2);
            u[p] = if c_state[cp] != CELL_NEUMANN {
                coarse[cp]
            } else {
                mean_bc
            };
        }
        relax_poisson_sparse(&mut u, &cells, None, w, 16, 1.5);
    } else {
        for &c_info in &cells {
            let p = c_info.p;
            u[p] = if dir_count > 0 { mean_bc } else { bc[p] };
        }
        if dir_count > 0 {
            relax_poisson_sparse(&mut u, &cells, None, w, 36, 1.5);
        }
    }
    u
}

#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn relax_poisson_sparse(
    u: &mut [[f32; 3]],
    cells: &[InteriorCell],
    rhs: Option<&[[f32; 3]]>,
    w: usize,
    iters: usize,
    omega: f32,
) {
    let keep = 1.0 - omega;
    for _ in 0..iters {
        for (idx, &c_info) in cells.iter().enumerate() {
            if c_info.inv_denom == 0.0 {
                continue;
            }
            let p = c_info.p;
            let m = c_info.mask;
            let mut sum = match rhs {
                Some(r) => r[idx],
                None => [0.0f32; 3],
            };
            if (m & 1) != 0 {
                let v = u[p - w];
                sum[0] += v[0];
                sum[1] += v[1];
                sum[2] += v[2];
            }
            if (m & 2) != 0 {
                let v = u[p + w];
                sum[0] += v[0];
                sum[1] += v[1];
                sum[2] += v[2];
            }
            if (m & 4) != 0 {
                let v = u[p - 1];
                sum[0] += v[0];
                sum[1] += v[1];
                sum[2] += v[2];
            }
            if (m & 8) != 0 {
                let v = u[p + 1];
                sum[0] += v[0];
                sum[1] += v[1];
                sum[2] += v[2];
            }
            let inv = c_info.inv_denom;
            let cur = u[p];
            u[p] = [
                cur[0] * keep + (sum[0] * inv) * omega,
                cur[1] * keep + (sum[1] * inv) * omega,
                cur[2] * keep + (sum[2] * inv) * omega,
            ];
        }
    }
}

struct Level {
    h: usize,
    w: usize,
    img: Vec<[f32; 3]>,
    hole: Vec<u8>,
    hard_bad: Vec<u8>,
    speck_bad: Vec<u8>,
    grain_floor: (f32, f32, f32, f32),
}

/// Exterior 4-neighbours of `hole` form the Dirichlet boundary; interior hole pixels are solved.
/// A minority extreme collar outlier (`> 6.5 sigma` and `> 0.55` log jump, < 25% of collar) is
/// released to `NEUMANN` ONLY if no clean film `5..14 px` outside `hole` shares its tone — which
/// drops a clipped dust crescent sticking `1-4 px` outside the brush while keeping every real 1D
/// line, curve, wire or highlight that continues into the surrounding scene anchored on both sides.
fn build_boundary_state(
    img: &[[f32; 3]],
    hole: &[u8],
    hard_bad: &[u8],
    speck_bad: &[u8],
    (h, w): (usize, usize),
) -> Vec<u8> {
    let n = h * w;
    let mut raw_collar = vec![false; n];
    let mut collar = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            if hole[p] != 0 {
                continue;
            }
            let adj = (y > 0 && hole[p - w] != 0)
                || (y + 1 < h && hole[p + w] != 0)
                || (x > 0 && hole[p - 1] != 0)
                || (x + 1 < w && hole[p + 1] != 0);
            raw_collar[p] = adj && (hard_bad[p] & HARD_AVOID) == 0;
            collar[p] = adj && (hard_bad[p] & HARD_EXCL) == 0;
        }
    }
    if !collar.iter().any(|&c| c) {
        collar = raw_collar;
    }

    let lumas: Vec<f32> = (0..n)
        .filter(|&p| collar[p])
        .map(|p| luma_log(img[p]))
        .collect();
    let mut rim_outlier = vec![false; n];
    if h.min(w) >= 32 && lumas.len() >= 12 {
        let med = median(&lumas);
        let dev: Vec<f32> = lumas.iter().map(|&v| (v - med).abs()).collect();
        let sigma = (1.4826 * median(&dev)).max(0.085);
        let bar = (6.5 * sigma).max(0.55);
        // Compute `ext_dist` lazily when an extreme collar outlier exists.
        let mut ext_dist_opt: Option<Vec<f32>> = None;
        let mut out_n = 0usize;
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                if !collar[p] {
                    continue;
                }
                let lp = luma_log(img[p]);
                if speck_bad[p] == 0 && (lp - med).abs() <= bar {
                    continue;
                }
                let ext_dist = ext_dist_opt.get_or_insert_with(|| {
                    let inv_hole: Vec<u8> = hole.iter().map(|&v| (v == 0) as u8).collect();
                    crate::distance::distance_transform(
                        &inv_hole,
                        h,
                        w,
                        crate::distance::Metric::Chessboard,
                    )
                });
                // Check whether clean exterior film `5..14 px` from `hole` carries this tone.
                let ya = y.saturating_sub(14);
                let yb = (y + 15).min(h);
                let xa = x.saturating_sub(14);
                let xb = (x + 15).min(w);
                let mut clean_cand = 0usize;
                let mut support = 0usize;
                for sy in ya..yb {
                    let row = sy * w;
                    for sx in xa..xb {
                        let sp = row + sx;
                        if ext_dist[sp] >= 5.0
                            && (hard_bad[sp] & HARD_EXCL) == 0
                            && speck_bad[sp] == 0
                        {
                            clean_cand += 1;
                            if (luma_log(img[sp]) - lp).abs() <= 0.28 {
                                support += 1;
                            }
                        }
                    }
                }
                if clean_cand >= 16 && support < 3 {
                    rim_outlier[p] = true;
                    out_n += 1;
                }
            }
        }
        if out_n * 4 > lumas.len() || lumas.len().saturating_sub(out_n) < 6 {
            rim_outlier.fill(false);
        }
    }

    let mut state = vec![CELL_NEUMANN; n];
    for p in 0..n {
        if hole[p] != 0 {
            state[p] = CELL_INTERIOR;
        } else if collar[p] && !rim_outlier[p] {
            state[p] = CELL_DIRICHLET;
        }
    }
    state
}

fn max_hole_depth(hole: &[u8], (h, w): (usize, usize)) -> f32 {
    if !hole.iter().any(|&v| v != 0) {
        return 0.0;
    }
    if hole.iter().all(|&v| v != 0) {
        return (h.min(w) as f32 * 0.5).max(1.0);
    }
    let d = crate::distance::distance_transform(hole, h, w, crate::distance::Metric::Chessboard);
    d.into_iter().fold(0.0f32, f32::max).min(h.max(w) as f32)
}

/// Detect isolated 0D point outliers that jump across all 8 orientations (`±3 px` ring).
/// Lines, curves and edges stay aligned with at least one tangent and pass with `dir_dev ≈ 0`.
fn detect_point_specks(
    log_y: &[f32],
    hole: &[u8],
    hard_bad: &[u8],
    (h, w): (usize, usize),
    sigma_mult: f32,
    min_bar: f32,
) -> Vec<u8> {
    let n = h * w;
    if h < 8 || w < 8 {
        return vec![0u8; n];
    }
    let mut dir_devs = vec![0.0f32; n];
    for y in 0..h {
        let ym3 = reflect101(y as isize - 3, h);
        let yp3 = reflect101(y as isize + 3, h);
        let ym2 = reflect101(y as isize - 2, h);
        let yp2 = reflect101(y as isize + 2, h);
        let ym1 = reflect101(y as isize - 1, h);
        let yp1 = reflect101(y as isize + 1, h);
        let row = y * w;
        for x in 0..w {
            let xm3 = reflect101(x as isize - 3, w);
            let xp3 = reflect101(x as isize + 3, w);
            let xm2 = reflect101(x as isize - 2, w);
            let xp2 = reflect101(x as isize + 2, w);
            let xm1 = reflect101(x as isize - 1, w);
            let xp1 = reflect101(x as isize + 1, w);
            let ly = log_y[row + x];
            let d_h = (ly - 0.5 * (log_y[row + xm3] + log_y[row + xp3])).abs();
            let d_v = (ly - 0.5 * (log_y[ym3 * w + x] + log_y[yp3 * w + x])).abs();
            let d_d1 = (ly - 0.5 * (log_y[ym2 * w + xm2] + log_y[yp2 * w + xp2])).abs();
            let d_d2 = (ly - 0.5 * (log_y[yp2 * w + xm2] + log_y[ym2 * w + xp2])).abs();
            let d_k1 = (ly - 0.5 * (log_y[ym1 * w + xm2] + log_y[yp1 * w + xp2])).abs();
            let d_k2 = (ly - 0.5 * (log_y[yp1 * w + xm2] + log_y[ym1 * w + xp2])).abs();
            let d_k3 = (ly - 0.5 * (log_y[ym2 * w + xm1] + log_y[yp2 * w + xp1])).abs();
            let d_k4 = (ly - 0.5 * (log_y[yp2 * w + xm1] + log_y[ym2 * w + xp1])).abs();
            let m4 = d_h.min(d_v).min(d_d1.min(d_d2));
            let mk = d_k1.min(d_k2).min(d_k3.min(d_k4));
            dir_devs[row + x] = m4.min(mk);
        }
    }
    let ext_devs: Vec<f32> = (0..n)
        .filter(|&p| hole[p] == 0 && hard_bad[p] == 0)
        .map(|p| dir_devs[p])
        .collect();
    let dev_bar = if ext_devs.len() >= 16 {
        (sigma_mult * (1.4826 * median(&ext_devs)).max(0.05)).max(min_bar)
    } else {
        min_bar.max(0.55)
    };
    (0..n)
        .map(|p| (hole[p] == 0 && (hard_bad[p] & HARD_AVOID) == 0 && dir_devs[p] > dev_bar) as u8)
        .collect()
}

/// Measure clean exterior grain power at the current pyramid resolution across three bands:
/// `(s_floor, g_floor, tone_floor)` = `(||img - s5||^2, ||grad - grad_s5||^2, ||s5 - s15||^2)`.
/// Measuring directly on each downsampled level (rather than scaling by `0.25`) tracks spatially
/// correlated `4-9 px` film dye clouds accurately and dead-zones `5x5` patch-mean fluctuations
/// inside smooth Poisson holes so PatchMatch copies intact multi-pixel dye-cloud sheets.
fn measure_grain_floor(
    img: &[[f32; 3]],
    hard_bad: &[u8],
    speck_bad: &[u8],
    (h, w): (usize, usize),
) -> (f32, f32, f32, f32) {
    let n = h * w;
    if n < 16 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let s5 = smooth3_rgb(&smooth3_rgb(img, (h, w)), (h, w));
    let box_rad = (h.min(w) as isize / 8).clamp(3, 5);
    let s_bg = smooth_box_rgb(&s5, box_rad, (h, w));
    let (gx_raw, gy_raw) = compute_luma_gradients(img, (h, w));
    let (gx_sm, gy_sm) = compute_luma_gradients(&s5, (h, w));

    let mut hp2_samples = Vec::new();
    let mut gp2_samples = Vec::new();
    let mut tp2_samples = Vec::new();
    let stride = (n / 4096).max(1);
    for p in (0..n).step_by(stride) {
        if hard_bad[p] == 0 && speck_bad[p] == 0 {
            let d0 = img[p][0] - s5[p][0];
            let d1 = img[p][1] - s5[p][1];
            let d2 = img[p][2] - s5[p][2];
            hp2_samples.push(d0 * d0 + d1 * d1 + d2 * d2);
            let dg0 = gx_raw[p] - gx_sm[p];
            let dg1 = gy_raw[p] - gy_sm[p];
            gp2_samples.push(dg0 * dg0 + dg1 * dg1);
            let t0 = s5[p][0] - s_bg[p][0];
            let t1 = s5[p][1] - s_bg[p][1];
            let t2 = s5[p][2] - s_bg[p][2];
            tp2_samples.push(t0 * t0 + t1 * t1 + t2 * t2);
        }
    }
    if hp2_samples.len() >= 8 {
        hp2_samples.sort_by(f32::total_cmp);
        gp2_samples.sort_by(f32::total_cmp);
        tp2_samples.sort_by(f32::total_cmp);
        let med_hp2 = hp2_samples[hp2_samples.len() / 2];
        let med_gp2 = gp2_samples[gp2_samples.len() / 2];
        let q25_hp2 = hp2_samples[hp2_samples.len() / 4];
        let q25_gp2 = gp2_samples[gp2_samples.len() / 4];
        let q25_tp2 = tp2_samples[tp2_samples.len() / 4];
        // Background `q25` grain shrinkage penalty for `rot != 0`: unaffected by foreground arcs
        // on small coarse crops while scaling unclamped on high-ISO grainy scans.
        let xform_grain_pen = (4.5 * q25_hp2 + 4.0 * q25_gp2).min(1.5);
        (
            (4.0 * med_hp2).min(0.20),
            (4.0 * med_gp2).min(0.08),
            (5.5 * q25_tp2).min(0.14),
            xform_grain_pen,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    }
}

fn downsample_level(prev: &Level) -> Level {
    let (ch, cw) = (prev.h.div_ceil(2), prev.w.div_ceil(2));
    let cn = ch * cw;
    let mut img = vec![[0.0f32; 3]; cn];
    let mut hole = vec![0u8; cn];
    let mut hard_bad = vec![0u8; cn];
    let mut speck_bad = vec![0u8; cn];

    for cy in 0..ch {
        for cx in 0..cw {
            let mut clean_acc = [0.0f32; 3];
            let mut clean_n = 0u32;
            let mut all_acc = [0.0f32; 3];
            let mut all_n = 0u32;
            let mut any_hard = 0u8;
            let mut speck_n = 0u32;
            for dy in 0..2 {
                let fy = 2 * cy + dy;
                if fy >= prev.h {
                    continue;
                }
                for dx in 0..2 {
                    let fx = 2 * cx + dx;
                    if fx >= prev.w {
                        continue;
                    }
                    let fp = fy * prev.w + fx;
                    let v = prev.img[fp];
                    all_acc[0] += v[0];
                    all_acc[1] += v[1];
                    all_acc[2] += v[2];
                    all_n += 1;
                    if prev.hole[fp] == 0 {
                        clean_acc[0] += v[0];
                        clean_acc[1] += v[1];
                        clean_acc[2] += v[2];
                        clean_n += 1;
                    }
                    if prev.hard_bad[fp] != 0 {
                        any_hard |= prev.hard_bad[fp];
                    }
                    if prev.speck_bad[fp] != 0 {
                        speck_n += 1;
                    }
                }
            }
            let cp = cy * cw + cx;
            if clean_n > 0 {
                let inv = 1.0 / clean_n as f32;
                img[cp] = [clean_acc[0] * inv, clean_acc[1] * inv, clean_acc[2] * inv];
                hole[cp] = 0;
            } else {
                let inv = 1.0 / all_n.max(1) as f32;
                img[cp] = [all_acc[0] * inv, all_acc[1] * inv, all_acc[2] * inv];
                hole[cp] = 1;
            }
            hard_bad[cp] = any_hard;
            speck_bad[cp] = (speck_n >= 2) as u8;
        }
    }
    if ch.min(cw) >= 32 {
        let coarse_log_y: Vec<f32> = img.iter().map(|&v| luma_log(v)).collect();
        let multi_speck = detect_point_specks(&coarse_log_y, &hole, &hard_bad, (ch, cw), 7.5, 0.55);
        for (sb, &ms) in speck_bad.iter_mut().zip(&multi_speck) {
            *sb |= ms;
        }
    }
    let m = measure_grain_floor(&img, &hard_bad, &speck_bad, (ch, cw));
    let grain_floor = (
        m.0.min(prev.grain_floor.0 * 0.85),
        m.1.min(prev.grain_floor.1 * 0.85),
        m.2.min(prev.grain_floor.2 * 0.85),
        m.3.min(prev.grain_floor.3 * 0.85),
    );
    Level {
        h: ch,
        w: cw,
        img,
        hole,
        hard_bad,
        speck_bad,
        grain_floor,
    }
}

#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn compute_luma_gradients(img: &[[f32; 3]], (h, w): (usize, usize)) -> (Vec<f32>, Vec<f32>) {
    let n = h * w;
    let luma: Vec<f32> = img.iter().map(|&v| luma_log(v)).collect();
    let mut gx = vec![0.0f32; n];
    let mut gy = vec![0.0f32; n];
    for y in 0..h {
        let ym = y.saturating_sub(1) * w;
        let yr = y * w;
        let yp = (y + 1).min(h - 1) * w;
        if w >= 2 {
            gx[yr] = 0.5 * (luma[yr + 1] - luma[yr]);
            for x in 1..w - 1 {
                gx[yr + x] = 0.5 * (luma[yr + x + 1] - luma[yr + x - 1]);
            }
            gx[yr + w - 1] = 0.5 * (luma[yr + w - 1] - luma[yr + w - 2]);
        }
        let su = &luma[ym..ym + w];
        let sd = &luma[yp..yp + w];
        let dst_y = &mut gy[yr..yr + w];
        for ((dy_out, &d), &u) in dst_y.iter_mut().zip(sd).zip(su) {
            *dy_out = 0.5 * (d - u);
        }
    }
    (gx, gy)
}

#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn compute_rgb_gradients(
    img: &[[f32; 3]],
    (h, w): (usize, usize),
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let n = h * w;
    let mut gx = vec![[0.0f32; 3]; n];
    let mut gy = vec![[0.0f32; 3]; n];
    for y in 0..h {
        let ym = y.saturating_sub(1) * w;
        let yr = y * w;
        let yp = (y + 1).min(h - 1) * w;
        if w >= 2 {
            let l0 = img[yr];
            let r0 = img[yr + 1];
            gx[yr] = [
                0.5 * (r0[0] - l0[0]),
                0.5 * (r0[1] - l0[1]),
                0.5 * (r0[2] - l0[2]),
            ];
            for x in 1..w - 1 {
                let l = img[yr + x - 1];
                let r = img[yr + x + 1];
                gx[yr + x] = [
                    0.5 * (r[0] - l[0]),
                    0.5 * (r[1] - l[1]),
                    0.5 * (r[2] - l[2]),
                ];
            }
            let ll = img[yr + w - 2];
            let rl = img[yr + w - 1];
            gx[yr + w - 1] = [
                0.5 * (rl[0] - ll[0]),
                0.5 * (rl[1] - ll[1]),
                0.5 * (rl[2] - ll[2]),
            ];
        }
        let su = &img[ym..ym + w];
        let sd = &img[yp..yp + w];
        let dst_y = &mut gy[yr..yr + w];
        for ((dy_out, &d), &u) in dst_y.iter_mut().zip(sd).zip(su) {
            *dy_out = [
                0.5 * (d[0] - u[0]),
                0.5 * (d[1] - u[1]),
                0.5 * (d[2] - u[2]),
            ];
        }
    }
    (gx, gy)
}

/// Valid donor centres where the 5x5 patch avoids holes, `avoid` masks and unclicked dust.
fn collect_valid_donors(lvl: &Level) -> (Vec<bool>, Vec<(i32, i32)>) {
    let (h, w) = (lvl.h, lvl.w);
    let n = h * w;
    let scan_pixels = |allow_skirt: bool, allow_dead: bool| -> (Vec<bool>, Vec<(i32, i32)>) {
        let mut ok = vec![false; n];
        let mut list = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                if lvl.hole[p] != 0 || (lvl.hard_bad[p] & HARD_AVOID) != 0 {
                    continue;
                }
                if !allow_dead && (lvl.hard_bad[p] & HARD_DEAD) != 0 {
                    continue;
                }
                if !allow_skirt && (lvl.hard_bad[p] & HARD_SKIRT) != 0 {
                    continue;
                }
                ok[p] = true;
                list.push((x as i32, y as i32));
            }
        }
        (ok, list)
    };

    if h <= (2 * R_PATCH) as usize || w <= (2 * R_PATCH) as usize {
        for (allow_skirt, allow_dead) in [(false, false), (true, false), (true, true)] {
            let (ok, list) = scan_pixels(allow_skirt, allow_dead);
            if !list.is_empty() {
                return (ok, list);
            }
        }
        return (vec![false; n], Vec::new());
    }

    let rp = R_PATCH as usize;
    let try_mask = |src_bad: &[u8]| -> (Vec<bool>, Vec<(i32, i32)>) {
        // Separable O(N) 5x5 dilation of `src_bad`.
        let mut h_dil = vec![0u8; n];
        for y in 0..h {
            let row = y * w;
            for x in rp..w - rp {
                let s = &src_bad[row + x - rp..=row + x + rp];
                h_dil[row + x] = (s[0] | s[1] | s[2] | s[3] | s[4] != 0) as u8;
            }
        }
        let mut ok = vec![false; n];
        let mut list = Vec::new();
        for y in rp..h - rp {
            for x in rp..w - rp {
                let mut any = 0u8;
                for dy in -R_PATCH..=R_PATCH {
                    any |= h_dil[(y as isize + dy) as usize * w + x];
                }
                if any == 0 {
                    ok[y * w + x] = true;
                    list.push((x as i32, y as i32));
                }
            }
        }
        (ok, list)
    };

    let combined: Vec<u8> = lvl
        .hard_bad
        .iter()
        .zip(&lvl.speck_bad)
        .map(|(&hb, &sb)| hb | sb)
        .collect();
    let (ok, list) = try_mask(&combined);
    if list.len() >= 16 {
        return (ok, list);
    }
    // Relax `speck_bad` first while keeping skirt, dead margin and `avoid` excluded.
    let (ok_hard, list_hard) = try_mask(&lvl.hard_bad);
    if !list_hard.is_empty() {
        return (ok_hard, list_hard);
    }
    // Relax `HARD_SKIRT` next, never relaxing `HARD_AVOID`.
    let hole_or_excl: Vec<u8> = lvl
        .hole
        .iter()
        .zip(&lvl.hard_bad)
        .map(|(&hv, &hb)| hv | (hb & HARD_EXCL))
        .collect();
    let (ok_no_skirt, list_no_skirt) = try_mask(&hole_or_excl);
    if !list_no_skirt.is_empty() {
        return (ok_no_skirt, list_no_skirt);
    }
    for (allow_skirt, allow_dead) in [(false, false), (true, false), (true, true)] {
        let (ok_px, list_px) = scan_pixels(allow_skirt, allow_dead);
        if !list_px.is_empty() {
            return (ok_px, list_px);
        }
    }
    (vec![false; n], Vec::new())
}

/// Active target patch centres: every pixel within `R_PATCH` of `hole != 0` via separable 5x5 dilation.
fn collect_active_targets(hole: &[u8], (h, w): (usize, usize)) -> (Vec<bool>, Vec<(i32, i32)>) {
    let n = h * w;
    let rp = R_PATCH as usize;
    let mut h_dil = vec![false; n];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let xa = x.saturating_sub(rp);
            let xb = (x + rp + 1).min(w);
            let mut any = 0u8;
            for &v in &hole[row + xa..row + xb] {
                any |= v;
            }
            h_dil[row + x] = any != 0;
        }
    }
    let mut active = vec![false; n];
    let mut list = Vec::new();
    for y in 0..h {
        let ya = y.saturating_sub(rp);
        let yb = (y + rp + 1).min(h);
        for x in 0..w {
            let mut touches = false;
            for ny in ya..yb {
                if h_dil[ny * w + x] {
                    touches = true;
                    break;
                }
            }
            if touches {
                active[y * w + x] = true;
                list.push((x as i32, y as i32));
            }
        }
    }
    (active, list)
}

/// 3x3 erosion of `donor_ok` (`±3 px` clean around `(qx, qy)`, `±4..5 px` from `cover` incl. `skirt`)
/// so rotated/scaled `5x5` patches (`max corner = 2.83 px`) stay clear of holes and dust while keeping
/// coarse pyramid levels (`h >= 7`) open for curved shoulder matching.
fn build_donor_ok_rot(donor_ok: &[bool], (h, w): (usize, usize)) -> Vec<bool> {
    let n = h * w;
    if h <= 6 || w <= 6 {
        return vec![false; n];
    }
    let mut row_ok = vec![false; n];
    for y in 0..h {
        let r = y * w;
        for x in 1..w - 1 {
            row_ok[r + x] = donor_ok[r + x - 1] && donor_ok[r + x] && donor_ok[r + x + 1];
        }
    }
    let mut out = vec![false; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            out[y * w + x] =
                row_ok[(y - 1) * w + x] && row_ok[y * w + x] && row_ok[(y + 1) * w + x];
        }
    }
    out
}

#[inline]
fn is_donor_valid(
    e: NnfEntry,
    donor_ok: &[bool],
    donor_ok_rot: &[bool],
    (h, w): (usize, usize),
) -> bool {
    let (qx, qy) = e.nearest_px();
    if qx < 0 || qx >= w as i32 || qy < 0 || qy >= h as i32 {
        return false;
    }
    let idx = qy as usize * w + qx as usize;
    if e.rot == 0 {
        donor_ok[idx]
    } else {
        donor_ok_rot[idx]
    }
}

struct TargetPatch {
    vp_c: [[f32; 3]; 25],
    tgx: [f32; 25],
    tgy: [f32; 25],
    mu_p: [f32; 3],
    inv_w: f32,
    tone_deadzone: f32,
    collar_w_frac: f32,
    collar_mask: u32,
    skip_mask: u32,
    has_edge: bool,
}

#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn build_target_patches(
    lvl: &Level,
    s5_img: &[[f32; 3]],
    gx: &[f32],
    gy: &[f32],
    gx_s5: &[f32],
    gy_s5: &[f32],
    active_list: &[(i32, i32)],
) -> Vec<TargetPatch> {
    let (h, w) = (lvl.h, lvl.w);
    let edge_span_bar = 3.5 * lvl.grain_floor.0.max(lvl.grain_floor.3) + 0.06;
    let mut out = Vec::with_capacity(active_list.len());

    for &(px, py) in active_list {
        let mut vp_raw = [[0.0f32; 3]; 25];
        let mut tgx = [0.0f32; 25];
        let mut tgy = [0.0f32; 25];
        let mut mu_p = [0.0f32; 3];
        let mut w_sum = 0.0f32;
        let mut collar_mask = 0u32;
        let mut skip_mask = 0u32;
        let mut s5_lo = [f32::INFINITY; 3];
        let mut s5_hi = [f32::NEG_INFINITY; 3];
        let mut k = 0usize;

        for dy in -R_PATCH..=R_PATCH {
            let ty = reflect101(py as isize + dy, h);
            let trow = ty * w;
            for dx in -R_PATCH..=R_PATCH {
                let tx = reflect101(px as isize + dx, w);
                let tp = trow + tx;
                let excl = lvl.hole[tp] == 0 && (lvl.hard_bad[tp] & HARD_EXCL) != 0;
                if excl {
                    skip_mask |= 1u32 << k;
                    k += 1;
                    continue;
                }
                let sv = s5_img[tp];
                for c in 0..3 {
                    s5_lo[c] = s5_lo[c].min(sv[c]);
                    s5_hi[c] = s5_hi[c].max(sv[c]);
                }
                let wt = if lvl.hole[tp] == 0 {
                    collar_mask |= 1u32 << k;
                    vp_raw[k] = lvl.img[tp];
                    tgx[k] = gx[tp];
                    tgy[k] = gy[tp];
                    4.0
                } else {
                    vp_raw[k] = sv;
                    tgx[k] = gx_s5[tp];
                    tgy[k] = gy_s5[tp];
                    1.0
                };
                mu_p[0] += wt * vp_raw[k][0];
                mu_p[1] += wt * vp_raw[k][1];
                mu_p[2] += wt * vp_raw[k][2];
                w_sum += wt;
                k += 1;
            }
        }
        let inv_w = 1.0 / w_sum.max(1.0e-6);
        mu_p[0] *= inv_w;
        mu_p[1] *= inv_w;
        mu_p[2] *= inv_w;
        let mut vp_c = [[0.0f32; 3]; 25];
        for i in 0..25 {
            if ((skip_mask >> i) & 1) == 0 {
                vp_c[i] = [
                    vp_raw[i][0] - mu_p[0],
                    vp_raw[i][1] - mu_p[1],
                    vp_raw[i][2] - mu_p[2],
                ];
            }
        }
        let d0 = (s5_hi[0] - s5_lo[0]).max(0.0);
        let d1 = (s5_hi[1] - s5_lo[1]).max(0.0);
        let d2 = (s5_hi[2] - s5_lo[2]).max(0.0);
        let has_edge = w_sum > 0.0 && (d0 * d0 + d1 * d1 + d2 * d2) > edge_span_bar;
        let valid_n = (25 - skip_mask.count_ones()).max(1);
        let int_n = valid_n.saturating_sub(collar_mask.count_ones());
        let int_frac = int_n as f32 / valid_n as f32;
        let tone_deadzone = int_frac * lvl.grain_floor.2;
        let collar_w_frac = ((4.0 * collar_mask.count_ones() as f32) * inv_w).clamp(0.0, 1.0);

        out.push(TargetPatch {
            vp_c,
            tgx,
            tgy,
            mu_p,
            inv_w,
            tone_deadzone,
            collar_w_frac,
            collar_mask,
            skip_mask,
            has_edge,
        });
    }
    out
}

#[allow(clippy::too_many_arguments)]
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn patch_distance_tp(
    tp: &TargetPatch,
    lvl: &Level,
    s5_img: &[[f32; 3]],
    gx: &[f32],
    gy: &[f32],
    gx_s5: &[f32],
    gy_s5: &[f32],
    (px, py): (i32, i32),
    e: NnfEntry,
    diag_inv: f32,
) -> (f32, [f32; 3]) {
    let (h, w) = (lvl.h, lvl.w);
    let (s_floor, g_floor, _, xform_grain_pen) = lvl.grain_floor;
    let inv_w = tp.inv_w;

    if e.rot == 0 {
        let qx = e.qx_q8 >> 8;
        let qy = e.qy_q8 >> 8;
        let mut mu_q = [0.0f32; 3];
        let mut struct_ssd = 0.0f32;
        let mut grad_ssd = 0.0f32;

        if tp.skip_mask == 0 && qx >= 2 && qx + 2 < w as i32 && qy >= 2 && qy + 2 < h as i32 {
            let base_x = (qx - 2) as usize;
            let base_y = (qy - 2) as usize;
            if tp.collar_mask == 0 {
                // Fast path A: all 25 taps are interior (`hole != 0`), branchless contiguous 5x5 read.
                for ry in 0..5 {
                    let srow = (base_y + ry) * w + base_x;
                    for rx in 0..5 {
                        let vq = s5_img[srow + rx];
                        mu_q[0] += vq[0];
                        mu_q[1] += vq[1];
                        mu_q[2] += vq[2];
                    }
                }
                mu_q[0] *= inv_w;
                mu_q[1] *= inv_w;
                mu_q[2] *= inv_w;
                let mut k = 0usize;
                for ry in 0..5 {
                    let srow = (base_y + ry) * w + base_x;
                    for rx in 0..5 {
                        let sp = srow + rx;
                        let vq = s5_img[sp];
                        let vc = tp.vp_c[k];
                        let d0 = vc[0] - (vq[0] - mu_q[0]);
                        let d1 = vc[1] - (vq[1] - mu_q[1]);
                        let d2 = vc[2] - (vq[2] - mu_q[2]);
                        let raw_s = d0 * d0 + d1 * d1 + d2 * d2;
                        let dgx = tp.tgx[k] - gx_s5[sp];
                        let dgy = tp.tgy[k] - gy_s5[sp];
                        let raw_g = dgx * dgx + dgy * dgy;
                        struct_ssd += (raw_s - s_floor).max(0.0);
                        grad_ssd += (raw_g - g_floor).max(0.0);
                        k += 1;
                    }
                }
            } else {
                // Fast path B: mixed collar/hole patch, contiguous 5x5 read without `reflect101`.
                let mut k = 0usize;
                let mask = tp.collar_mask;
                for ry in 0..5 {
                    let srow = (base_y + ry) * w + base_x;
                    for rx in 0..5 {
                        let sp = srow + rx;
                        if ((mask >> k) & 1) != 0 {
                            let vq = lvl.img[sp];
                            mu_q[0] += 4.0 * vq[0];
                            mu_q[1] += 4.0 * vq[1];
                            mu_q[2] += 4.0 * vq[2];
                        } else {
                            let vq = s5_img[sp];
                            mu_q[0] += vq[0];
                            mu_q[1] += vq[1];
                            mu_q[2] += vq[2];
                        }
                        k += 1;
                    }
                }
                mu_q[0] *= inv_w;
                mu_q[1] *= inv_w;
                mu_q[2] *= inv_w;
                k = 0;
                for ry in 0..5 {
                    let srow = (base_y + ry) * w + base_x;
                    for rx in 0..5 {
                        let sp = srow + rx;
                        let vc = tp.vp_c[k];
                        if ((mask >> k) & 1) != 0 {
                            let vq = lvl.img[sp];
                            let d0 = vc[0] - (vq[0] - mu_q[0]);
                            let d1 = vc[1] - (vq[1] - mu_q[1]);
                            let d2 = vc[2] - (vq[2] - mu_q[2]);
                            struct_ssd += 4.0 * (d0 * d0 + d1 * d1 + d2 * d2);
                            let dgx = tp.tgx[k] - gx[sp];
                            let dgy = tp.tgy[k] - gy[sp];
                            grad_ssd += 4.0 * (dgx * dgx + dgy * dgy);
                        } else {
                            let vq = s5_img[sp];
                            let d0 = vc[0] - (vq[0] - mu_q[0]);
                            let d1 = vc[1] - (vq[1] - mu_q[1]);
                            let d2 = vc[2] - (vq[2] - mu_q[2]);
                            let raw_s = d0 * d0 + d1 * d1 + d2 * d2;
                            let dgx = tp.tgx[k] - gx_s5[sp];
                            let dgy = tp.tgy[k] - gy_s5[sp];
                            let raw_g = dgx * dgx + dgy * dgy;
                            struct_ssd += (raw_s - s_floor).max(0.0);
                            grad_ssd += (raw_g - g_floor).max(0.0);
                        }
                        k += 1;
                    }
                }
            }
        } else {
            let mut k = 0usize;
            let mask = tp.collar_mask;
            let skip = tp.skip_mask;
            for dy in -R_PATCH..=R_PATCH {
                let sy = reflect101(qy as isize + dy, h);
                for dx in -R_PATCH..=R_PATCH {
                    if ((skip >> k) & 1) != 0 {
                        k += 1;
                        continue;
                    }
                    let sx = reflect101(qx as isize + dx, w);
                    let sp = sy * w + sx;
                    if ((mask >> k) & 1) != 0 {
                        let vq = lvl.img[sp];
                        mu_q[0] += 4.0 * vq[0];
                        mu_q[1] += 4.0 * vq[1];
                        mu_q[2] += 4.0 * vq[2];
                    } else {
                        let vq = s5_img[sp];
                        mu_q[0] += vq[0];
                        mu_q[1] += vq[1];
                        mu_q[2] += vq[2];
                    }
                    k += 1;
                }
            }
            mu_q[0] *= inv_w;
            mu_q[1] *= inv_w;
            mu_q[2] *= inv_w;
            k = 0;
            for dy in -R_PATCH..=R_PATCH {
                let sy = reflect101(qy as isize + dy, h);
                for dx in -R_PATCH..=R_PATCH {
                    if ((skip >> k) & 1) != 0 {
                        k += 1;
                        continue;
                    }
                    let sx = reflect101(qx as isize + dx, w);
                    let sp = sy * w + sx;
                    let vc = tp.vp_c[k];
                    if ((mask >> k) & 1) != 0 {
                        let vq = lvl.img[sp];
                        let d0 = vc[0] - (vq[0] - mu_q[0]);
                        let d1 = vc[1] - (vq[1] - mu_q[1]);
                        let d2 = vc[2] - (vq[2] - mu_q[2]);
                        struct_ssd += 4.0 * (d0 * d0 + d1 * d1 + d2 * d2);
                        let dgx = tp.tgx[k] - gx[sp];
                        let dgy = tp.tgy[k] - gy[sp];
                        grad_ssd += 4.0 * (dgx * dgx + dgy * dgy);
                    } else {
                        let vq = s5_img[sp];
                        let d0 = vc[0] - (vq[0] - mu_q[0]);
                        let d1 = vc[1] - (vq[1] - mu_q[1]);
                        let d2 = vc[2] - (vq[2] - mu_q[2]);
                        let raw_s = d0 * d0 + d1 * d1 + d2 * d2;
                        let dgx = tp.tgx[k] - gx_s5[sp];
                        let dgy = tp.tgy[k] - gy_s5[sp];
                        let raw_g = dgx * dgx + dgy * dgy;
                        struct_ssd += (raw_s - s_floor).max(0.0);
                        grad_ssd += (raw_g - g_floor).max(0.0);
                    }
                    k += 1;
                }
            }
        }

        let shift = [
            tp.mu_p[0] - mu_q[0],
            tp.mu_p[1] - mu_q[1],
            tp.mu_p[2] - mu_q[2],
        ];
        let raw_tone = shift[0] * shift[0] + shift[1] * shift[1] + shift[2] * shift[2];
        let tone_pen = (raw_tone - tp.tone_deadzone).max(0.0);
        let ddx = (px - qx) as f32;
        let ddy = (py - qy) as f32;
        let dist_pen = (ddx * ddx + ddy * ddy) * diag_inv;

        let cost = (struct_ssd + 0.90 * grad_ssd) * inv_w + 0.50 * tone_pen + 0.025 * dist_pen;
        return (cost, shift);
    }

    let spec = &XFORM_TABLE[e.rot as usize];
    let mask = tp.collar_mask;
    let skip = tp.skip_mask;
    let mut mu_q = [0.0f32; 3];
    let mut vq_cache = [[0.0f32; 3]; 25];
    let mut k = 0usize;

    for dy in -R_PATCH..=R_PATCH {
        for dx in -R_PATCH..=R_PATCH {
            if ((skip >> k) & 1) != 0 {
                k += 1;
                continue;
            }
            let sx_q8 = e.qx_q8 + (dx as i32) * spec.a_q8 - (dy as i32) * spec.b_q8;
            let sy_q8 = e.qy_q8 + (dx as i32) * spec.b_q8 + (dy as i32) * spec.a_q8;
            let (wt, vq) = if ((mask >> k) & 1) != 0 {
                (4.0, bilerp_rgb(&lvl.img, (h, w), sx_q8, sy_q8))
            } else {
                (1.0, bilerp_rgb(s5_img, (h, w), sx_q8, sy_q8))
            };
            vq_cache[k] = vq;
            k += 1;
            mu_q[0] += wt * vq[0];
            mu_q[1] += wt * vq[1];
            mu_q[2] += wt * vq[2];
        }
    }
    mu_q[0] *= inv_w;
    mu_q[1] *= inv_w;
    mu_q[2] *= inv_w;

    let mut struct_ssd = 0.0f32;
    let mut grad_ssd = 0.0f32;
    k = 0;
    for dy in -R_PATCH..=R_PATCH {
        for dx in -R_PATCH..=R_PATCH {
            if ((skip >> k) & 1) != 0 {
                k += 1;
                continue;
            }
            let sx_q8 = e.qx_q8 + (dx as i32) * spec.a_q8 - (dy as i32) * spec.b_q8;
            let sy_q8 = e.qy_q8 + (dx as i32) * spec.b_q8 + (dy as i32) * spec.a_q8;
            let vq = vq_cache[k];
            let vc = tp.vp_c[k];
            let d0 = vc[0] - (vq[0] - mu_q[0]);
            let d1 = vc[1] - (vq[1] - mu_q[1]);
            let d2 = vc[2] - (vq[2] - mu_q[2]);
            if ((mask >> k) & 1) != 0 {
                struct_ssd += 4.0 * (d0 * d0 + d1 * d1 + d2 * d2);
                let sgx = bilerp_f32(gx, (h, w), sx_q8, sy_q8);
                let sgy = bilerp_f32(gy, (h, w), sx_q8, sy_q8);
                let r_gx = spec.a_f * sgx + spec.b_f * sgy;
                let r_gy = -spec.b_f * sgx + spec.a_f * sgy;
                let dgx = tp.tgx[k] - r_gx;
                let dgy = tp.tgy[k] - r_gy;
                grad_ssd += 4.0 * (dgx * dgx + dgy * dgy);
            } else {
                let raw_s = d0 * d0 + d1 * d1 + d2 * d2;
                let sgx = bilerp_f32(gx_s5, (h, w), sx_q8, sy_q8);
                let sgy = bilerp_f32(gy_s5, (h, w), sx_q8, sy_q8);
                let r_gx = spec.a_f * sgx + spec.b_f * sgy;
                let r_gy = -spec.b_f * sgx + spec.a_f * sgy;
                let dgx = tp.tgx[k] - r_gx;
                let dgy = tp.tgy[k] - r_gy;
                let raw_g = dgx * dgx + dgy * dgy;
                struct_ssd += (raw_s - s_floor).max(0.0);
                grad_ssd += (raw_g - g_floor).max(0.0);
            }
            k += 1;
        }
    }
    let shift = [
        tp.mu_p[0] - mu_q[0],
        tp.mu_p[1] - mu_q[1],
        tp.mu_p[2] - mu_q[2],
    ];
    let raw_tone = shift[0] * shift[0] + shift[1] * shift[1] + shift[2] * shift[2];
    let tone_pen = (raw_tone - tp.tone_deadzone).max(0.0);
    let qx_f = e.qx_q8 as f32 * (1.0 / 256.0);
    let qy_f = e.qy_q8 as f32 * (1.0 / 256.0);
    let ddx = px as f32 - qx_f;
    let ddy = py as f32 - qy_f;
    let dist_pen = (ddx * ddx + ddy * ddy) * diag_inv;
    let xform_pen =
        (0.55 * s_floor).max((0.55 + 0.65 * tp.collar_w_frac) * xform_grain_pen) + spec.pen;

    let cost =
        (struct_ssd + 0.90 * grad_ssd) * inv_w + 0.50 * tone_pen + 0.025 * dist_pen + xform_pen;
    (cost, shift)
}

#[inline]
fn coherent_degree(
    (px, py): (i32, i32),
    e: NnfEntry,
    nnf: &[NnfEntry],
    active: &[bool],
    (h, w): (usize, usize),
) -> u8 {
    let mut deg = 0u8;
    for (nx, ny, sx, sy) in [
        (px - 1, py, 1, 0),
        (px + 1, py, -1, 0),
        (px, py - 1, 0, 1),
        (px, py + 1, 0, -1),
    ] {
        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
            let np = ny as usize * w + nx as usize;
            if active[np] && entries_coherent(nnf[np], sx, sy, e) {
                deg += 1;
            }
        }
    }
    deg
}

#[inline]
fn coherence_eff(cost: f32, deg: u8, coh_gain: f32, coh_floor: f32) -> f32 {
    match deg {
        0 => cost + coh_floor,
        1 => cost * coh_gain + 0.15 * coh_floor,
        _ => cost * coh_gain,
    }
}

type VoteMaps = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<f32>);

#[allow(clippy::too_many_arguments)]
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn vote_m_step(
    lvl: &Level,
    active_list: &[(i32, i32)],
    nnf: &[NnfEntry],
    shifts: &[[f32; 3]],
    costs: &[f32],
    sigma_cost: f32,
    rgb_gx: &[[f32; 3]],
    rgb_gy: &[[f32; 3]],
) -> VoteMaps {
    let (h, w) = (lvl.h, lvl.w);
    let n = h * w;
    let mut c_vote = vec![[0.0f32; 3]; n];
    let mut gx_vote = vec![[0.0f32; 3]; n];
    let mut gy_vote = vec![[0.0f32; 3]; n];
    let mut w_vote = vec![0.0f32; n];

    for &(px, py) in active_list {
        let p = py as usize * w + px as usize;
        let e = nnf[p];
        let sh = shifts[p];
        let wp = np_expf(-(costs[p] / sigma_cost).clamp(0.0, 12.0)).max(1.0e-5);

        if e.rot == 0 {
            let qx = e.qx_q8 >> 8;
            let qy = e.qy_q8 >> 8;
            for dy in -R_PATCH..=R_PATCH {
                let ty = py as isize + dy;
                if ty < 0 || ty >= h as isize {
                    continue;
                }
                let wy = (3 - dy.abs()) as f32;
                let sy = reflect101(qy as isize + dy, h);
                for dx in -R_PATCH..=R_PATCH {
                    let tx = px as isize + dx;
                    if tx < 0 || tx >= w as isize {
                        continue;
                    }
                    let tp = ty as usize * w + tx as usize;
                    if lvl.hole[tp] == 0 {
                        continue;
                    }
                    let w_tap = wp * wy * (3 - dx.abs()) as f32;
                    let sp = sy * w + reflect101(qx as isize + dx, w);
                    let sv = lvl.img[sp];
                    let sgx = rgb_gx[sp];
                    let sgy = rgb_gy[sp];
                    for c in 0..3 {
                        c_vote[tp][c] += w_tap * (sv[c] + sh[c]);
                        gx_vote[tp][c] += w_tap * sgx[c];
                        gy_vote[tp][c] += w_tap * sgy[c];
                    }
                    w_vote[tp] += w_tap;
                }
            }
        } else {
            let spec = &XFORM_TABLE[e.rot as usize];
            for dy in -R_PATCH..=R_PATCH {
                let ty = py as isize + dy;
                if ty < 0 || ty >= h as isize {
                    continue;
                }
                let wy = (3 - dy.abs()) as f32;
                for dx in -R_PATCH..=R_PATCH {
                    let tx = px as isize + dx;
                    if tx < 0 || tx >= w as isize {
                        continue;
                    }
                    let tp = ty as usize * w + tx as usize;
                    if lvl.hole[tp] == 0 {
                        continue;
                    }
                    let w_tap = wp * wy * (3 - dx.abs()) as f32;
                    let sx_q8 = e.qx_q8 + (dx as i32) * spec.a_q8 - (dy as i32) * spec.b_q8;
                    let sy_q8 = e.qy_q8 + (dx as i32) * spec.b_q8 + (dy as i32) * spec.a_q8;
                    let sv = bilerp_rgb(&lvl.img, (h, w), sx_q8, sy_q8);
                    let sgx = bilerp_rgb(rgb_gx, (h, w), sx_q8, sy_q8);
                    let sgy = bilerp_rgb(rgb_gy, (h, w), sx_q8, sy_q8);
                    for c in 0..3 {
                        c_vote[tp][c] += w_tap * (sv[c] + sh[c]);
                        gx_vote[tp][c] += w_tap * (spec.a_f * sgx[c] + spec.b_f * sgy[c]);
                        gy_vote[tp][c] += w_tap * (-spec.b_f * sgx[c] + spec.a_f * sgy[c]);
                    }
                    w_vote[tp] += w_tap;
                }
            }
        }
    }
    (c_vote, gx_vote, gy_vote, w_vote)
}

type CoarseNnf<'a> = (&'a [NnfEntry], (usize, usize));
type OwnedNnf = (Vec<NnfEntry>, (usize, usize));

/// Run EM PatchMatch + screened Poisson melding at level `lvl`, updating `lvl.img` inside `lvl.hole`
/// and returning the final `(nnf, shifts)` on `(lvl.h, lvl.w)`.
fn solve_level(
    lvl: &mut Level,
    coarse_nnf: Option<CoarseNnf<'_>>,
    level_idx: usize,
) -> (Vec<NnfEntry>, Vec<[f32; 3]>) {
    let (h, w) = (lvl.h, lvl.w);
    let n = h * w;
    let mut nnf = vec![NnfEntry::int(0, 0); n];
    for y in 0..h {
        for x in 0..w {
            nnf[y * w + x] = NnfEntry::int(x as i32, y as i32);
        }
    }
    let mut final_shifts = vec![[0.0f32; 3]; n];
    if !lvl.hole.iter().any(|&v| v != 0) {
        return (nnf, final_shifts);
    }

    let (donor_ok, donor_list) = collect_valid_donors(lvl);
    if donor_list.is_empty() {
        // Fallback: pure harmonic Laplace fill.
        let bstate =
            build_boundary_state(&lvl.img, &lvl.hole, &lvl.hard_bad, &lvl.speck_bad, (h, w));
        let filled = solve_harmonic_init(&bstate, &lvl.img, (h, w));
        for ((dst, &hv), &fv) in lvl.img.iter_mut().zip(&lvl.hole).zip(&filled) {
            if hv != 0 {
                *dst = fv;
            }
        }
        return (nnf, final_shifts);
    }
    let donor_ok_rot = build_donor_ok_rot(&donor_ok, (h, w));

    let (active, active_list) = collect_active_targets(&lvl.hole, (h, w));
    let diag_inv = 1.0 / ((h * h + w * w).max(1) as f32);

    // Initialize NNF on active targets from upsampled coarse NNF or deterministic random donors.
    let s5_up = if coarse_nnf.is_some() && level_idx == 0 && lvl.grain_floor.3 > 0.05 {
        Some(smooth3_rgb(&smooth3_rgb(&lvl.img, (h, w)), (h, w)))
    } else {
        None
    };
    let mut rng = Rng64::new(0xA5A5_5A5A_0123_4567 ^ ((level_idx as u64) << 32) ^ (n as u64));
    for &(px, py) in &active_list {
        let p = py as usize * w + px as usize;
        let mut picked = None;
        if let Some((c_nnf, (ch, cw))) = coarse_nnf {
            let cx = ((px as usize) / 2).min(cw - 1);
            let cy = ((py as usize) / 2).min(ch - 1);
            let ce = c_nnf[cy * cw + cx];
            if ce.rot != 0 {
                let cs = &XFORM_TABLE[ce.rot as usize];
                let ox = if (px & 1) == 0 { -1 } else { 1 };
                let oy = if (py & 1) == 0 { -1 } else { 1 };
                // On grainy scans at full resolution, keep `rot != 0` ONLY where a strong
                // local contrast edge exceeds the grain floor; otherwise snap to the exact `0°`
                // integer lattice so Catmull-Rom resampling never creates subpixel moiré swirls.
                let keep_rot = if let Some(ref s5u) = s5_up {
                    let (cqx, cqy) = ce.nearest_px();
                    let dqx = (2 * cqx).clamp(0, w as i32 - 1);
                    let dqy = (2 * cqy).clamp(0, h as i32 - 1);
                    let span2_at = |cx0: i32, cy0: i32| -> f32 {
                        let ya = (cy0 - 2).max(0) as usize;
                        let yb = (cy0 + 2).min(h as i32 - 1) as usize;
                        let xa = (cx0 - 2).max(0) as usize;
                        let xb = (cx0 + 2).min(w as i32 - 1) as usize;
                        let mut lo = [f32::INFINITY; 3];
                        let mut hi = [f32::NEG_INFINITY; 3];
                        for sy in ya..=yb {
                            for sx in xa..=xb {
                                let v = s5u[sy * w + sx];
                                for c in 0..3 {
                                    lo[c] = lo[c].min(v[c]);
                                    hi[c] = hi[c].max(v[c]);
                                }
                            }
                        }
                        let d0 = hi[0] - lo[0];
                        let d1 = hi[1] - lo[1];
                        let d2 = hi[2] - lo[2];
                        d0 * d0 + d1 * d1 + d2 * d2
                    };
                    span2_at(px, py).max(span2_at(dqx, dqy)) > 5.0 * lvl.grain_floor.3 + 0.12
                } else {
                    true
                };
                if keep_rot {
                    let cand = NnfEntry {
                        qx_q8: 2 * ce.qx_q8 + (ox * cs.a_q8 - oy * cs.b_q8) / 2,
                        qy_q8: 2 * ce.qy_q8 + (ox * cs.b_q8 + oy * cs.a_q8) / 2,
                        rot: ce.rot,
                    };
                    if is_donor_valid(cand, &donor_ok, &donor_ok_rot, (h, w)) {
                        picked = Some(cand);
                    }
                }
            }
            if picked.is_none() {
                let (cqx, cqy) = ce.nearest_px();
                let ux = (2 * cqx + (px & 1)).clamp(0, w as i32 - 1);
                let uy = (2 * cqy + (py & 1)).clamp(0, h as i32 - 1);
                if donor_ok[uy as usize * w + ux as usize] {
                    picked = Some(NnfEntry::int(ux, uy));
                } else {
                    'jitter: for dy in -2..=2 {
                        for dx in -2..=2 {
                            let jx = (ux + dx).clamp(0, w as i32 - 1);
                            let jy = (uy + dy).clamp(0, h as i32 - 1);
                            if donor_ok[jy as usize * w + jx as usize] {
                                picked = Some(NnfEntry::int(jx, jy));
                                break 'jitter;
                            }
                        }
                    }
                }
            }
        }
        nnf[p] = picked.unwrap_or_else(|| {
            let idx = rng.next_u32() as usize % donor_list.len();
            let (dx, dy) = donor_list[idx];
            NnfEntry::int(dx, dy)
        });
    }

    let hole_count = lvl.hole.iter().filter(|&&v| v != 0).count();
    let em_steps = if coarse_nnf.is_none() {
        if hole_count <= 200 { 2 } else { 3 }
    } else if hole_count > 12_000 {
        1
    } else {
        2
    };
    let pm_iters = if coarse_nnf.is_some() && hole_count > 32_768 {
        0
    } else if coarse_nnf.is_some() && (level_idx == 0 || hole_count > 6_000) {
        2
    } else {
        3
    };
    let max_search_rad = if coarse_nnf.is_some() {
        14i32
    } else {
        (h.max(w) as i32 / 2).max(4)
    };
    let coh_gain = if level_idx == 0 {
        0.72f32
    } else if coarse_nnf.is_none() {
        0.85f32
    } else {
        0.88f32
    };
    let coh_floor = 0.50 * (lvl.grain_floor.0 + lvl.grain_floor.2);
    let xform_gate = 1.5 * lvl.grain_floor.0.max(lvl.grain_floor.3) + 0.025;

    let bstate = build_boundary_state(&lvl.img, &lvl.hole, &lvl.hard_bad, &lvl.speck_bad, (h, w));

    // Synthesize fine-level sharpness from upsampled `coarse_nnf` BEFORE running fine PatchMatch,
    // so interior hole patches carry crisp lines/curves rather than blurry 2x-upsampled pixels.
    if coarse_nnf.is_some() && hole_count <= 32_768 {
        let (gx0, gy0) = compute_luma_gradients(&lvl.img, (h, w));
        let s5_0 = smooth3_rgb(&smooth3_rgb(&lvl.img, (h, w)), (h, w));
        let (gx_s5_0, gy_s5_0) = compute_luma_gradients(&s5_0, (h, w));
        let tps_0 = build_target_patches(lvl, &s5_0, &gx0, &gy0, &gx_s5_0, &gy_s5_0, &active_list);
        let mut init_costs = vec![f32::INFINITY; n];
        let mut init_shifts = vec![[0.0f32; 3]; n];
        for (idx, &(px, py)) in active_list.iter().enumerate() {
            let p = py as usize * w + px as usize;
            let (c, sh) = patch_distance_tp(
                &tps_0[idx],
                lvl,
                &s5_0,
                &gx0,
                &gy0,
                &gx_s5_0,
                &gy_s5_0,
                (px, py),
                nnf[p],
                diag_inv,
            );
            init_costs[p] = c;
            init_shifts[p] = sh;
        }
        let mut ac: Vec<f32> = active_list
            .iter()
            .map(|&(px, py)| init_costs[py as usize * w + px as usize])
            .filter(|c| c.is_finite())
            .collect();
        let sc = if ac.is_empty() {
            0.01
        } else {
            (crate::stats::np_median(&mut ac) * 1.5).max(1.0e-3)
        };
        let (rgb_gx, rgb_gy) = compute_rgb_gradients(&lvl.img, (h, w));
        let (cv, gxv_raw, gyv_raw, wv) = vote_m_step(
            lvl,
            &active_list,
            &nnf,
            &init_shifts,
            &init_costs,
            sc,
            &rgb_gx,
            &rgb_gy,
        );
        let mut data = lvl.img.clone();
        let mut gxv = rgb_gx;
        let mut gyv = rgb_gy;
        for p in 0..n {
            if lvl.hole[p] != 0 && wv[p] > 0.0 {
                let inv = 1.0 / wv[p];
                for c in 0..3 {
                    data[p][c] = cv[p][c] * inv;
                    gxv[p][c] = gxv_raw[p][c] * inv;
                    gyv[p][c] = gyv_raw[p][c] * inv;
                }
            }
        }
        let sharp_init = solve_screened_poisson(&bstate, &data, &gxv, &gyv, SCREEN_LAMBDA, (h, w));
        for ((dst, &hv), &sv) in lvl.img.iter_mut().zip(&lvl.hole).zip(&sharp_init) {
            if hv != 0 {
                *dst = sv;
            }
        }
    }

    for em in 0..em_steps {
        let (gx, gy) = compute_luma_gradients(&lvl.img, (h, w));
        let s5_buf;
        let gx_s5_buf;
        let gy_s5_buf;
        let (s5_img, gx_s5, gy_s5): (&[[f32; 3]], &[f32], &[f32]) =
            if level_idx == 0 || lvl.grain_floor.0 > 0.015 {
                s5_buf = smooth3_rgb(&smooth3_rgb(&lvl.img, (h, w)), (h, w));
                let (gxs, gys) = compute_luma_gradients(&s5_buf, (h, w));
                gx_s5_buf = gxs;
                gy_s5_buf = gys;
                (&s5_buf, &gx_s5_buf, &gy_s5_buf)
            } else {
                (&lvl.img, &gx, &gy)
            };
        let tps = build_target_patches(lvl, s5_img, &gx, &gy, gx_s5, gy_s5, &active_list);
        let mut costs = vec![f32::INFINITY; n];
        let mut shifts = vec![[0.0f32; 3]; n];

        // On the coarsest level's initial pass, find a default background ring offset AND allow
        // each pixel to pick the best of the 36 directional ring offsets (plus similarity sweep on
        // structural mismatches) so curves and wide multi-scratch bands lock onto clean context.
        if coarse_nnf.is_none() && em == 0 {
            let step_probe = active_list.len().div_ceil(24).max(1);
            let probe_indices: Vec<usize> = (0..active_list.len()).step_by(step_probe).collect();
            let r1 = (max_hole_depth(&lvl.hole, (h, w)) as i32 + 2 * R_PATCH as i32 + 1).max(4);
            let r2 = (r1 * 7) / 4 + 2;
            const DIRS: [(i32, i32); 12] = [
                (4, 0),
                (-4, 0),
                (0, 4),
                (0, -4),
                (3, 3),
                (-3, 3),
                (3, -3),
                (-3, -3),
                (4, 2),
                (-4, 2),
                (2, 4),
                (2, -4),
            ];
            let mut offsets = Vec::with_capacity(24);
            for rad_mul in [r1, r2] {
                for &(ux, uy) in &DIRS {
                    let odx = (ux * rad_mul) / 4;
                    let ody = (uy * rad_mul) / 4;
                    if (odx != 0 || ody != 0) && !offsets.contains(&(odx, ody)) {
                        offsets.push((odx, ody));
                    }
                }
            }
            let mut best_global = None;
            let mut best_score = f32::INFINITY;
            for &(odx, ody) in &offsets {
                let mut sum = 0.0f32;
                let mut ok_n = 0usize;
                for &p_i in &probe_indices {
                    let (px, py) = active_list[p_i];
                    let tx = px + odx;
                    let ty = py + ody;
                    if tx >= 0
                        && tx < w as i32
                        && ty >= 0
                        && ty < h as i32
                        && donor_ok[ty as usize * w + tx as usize]
                    {
                        sum += patch_distance_tp(
                            &tps[p_i],
                            lvl,
                            s5_img,
                            &gx,
                            &gy,
                            gx_s5,
                            gy_s5,
                            (px, py),
                            NnfEntry::int(tx, ty),
                            diag_inv,
                        )
                        .0;
                        ok_n += 1;
                    }
                }
                if ok_n * 4 >= probe_indices.len() * 3 {
                    let avg = sum / ok_n as f32;
                    if avg < best_score {
                        best_score = avg;
                        best_global = Some((odx, ody));
                    }
                }
            }
            for (idx, &(px, py)) in active_list.iter().enumerate() {
                let tp = &tps[idx];
                let p = py as usize * w + px as usize;
                let mut best_e = nnf[p];
                let mut best_eff = patch_distance_tp(
                    tp,
                    lvl,
                    s5_img,
                    &gx,
                    &gy,
                    gx_s5,
                    gy_s5,
                    (px, py),
                    best_e,
                    diag_inv,
                )
                .0;
                for &(odx, ody) in &offsets {
                    let tx = px + odx;
                    let ty = py + ody;
                    if tx >= 0
                        && tx < w as i32
                        && ty >= 0
                        && ty < h as i32
                        && donor_ok[ty as usize * w + tx as usize]
                    {
                        let cand = NnfEntry::int(tx, ty);
                        let c = patch_distance_tp(
                            tp,
                            lvl,
                            s5_img,
                            &gx,
                            &gy,
                            gx_s5,
                            gy_s5,
                            (px, py),
                            cand,
                            diag_inv,
                        )
                        .0;
                        let eff = if best_global == Some((odx, ody)) {
                            c * 0.90
                        } else {
                            c
                        };
                        if eff < best_eff {
                            best_eff = eff;
                            best_e = cand;
                        }
                    }
                }
                if tp.has_edge && best_eff > xform_gate {
                    const COARSE_XFORMS: [u8; 10] = [3, 4, 7, 8, 11, 12, 13, 14, 15, 20];
                    for &(odx, ody) in &offsets {
                        let tx = px + odx;
                        let ty = py + ody;
                        if tx >= 0
                            && tx < w as i32
                            && ty >= 0
                            && ty < h as i32
                            && donor_ok_rot[ty as usize * w + tx as usize]
                        {
                            for xf in COARSE_XFORMS {
                                let cand = NnfEntry::with_xform(tx, ty, xf);
                                let c = patch_distance_tp(
                                    tp,
                                    lvl,
                                    s5_img,
                                    &gx,
                                    &gy,
                                    gx_s5,
                                    gy_s5,
                                    (px, py),
                                    cand,
                                    diag_inv,
                                )
                                .0;
                                if c < best_eff {
                                    best_eff = c;
                                    best_e = cand;
                                }
                            }
                        }
                    }
                }
                nnf[p] = best_e;
            }
        }

        for (idx, &(px, py)) in active_list.iter().enumerate() {
            let p = py as usize * w + px as usize;
            let (c, sh) = patch_distance_tp(
                &tps[idx],
                lvl,
                s5_img,
                &gx,
                &gy,
                gx_s5,
                gy_s5,
                (px, py),
                nnf[p],
                diag_inv,
            );
            costs[p] = c;
            shifts[p] = sh;
        }

        for iter in 0..pm_iters {
            let rev = iter % 2 == 1;
            let len = active_list.len();
            for k in 0..len {
                let idx = if rev { len - 1 - k } else { k };
                let tp = &tps[idx];
                let (px, py) = active_list[idx];
                let p = py as usize * w + px as usize;
                let mut best_e = nnf[p];
                let mut best_cost = costs[p];
                let mut best_shift = shifts[p];
                let mut best_deg = coherent_degree((px, py), best_e, &nnf, &active, (h, w));
                let mut best_eff = coherence_eff(best_cost, best_deg, coh_gain, coh_floor);

                // 1. Coherent spatial propagation under 2D sheet Lyapunov energy.
                let nbrs = if !rev {
                    [(px - 1, py, 1, 0), (px, py - 1, 0, 1)]
                } else {
                    [(px + 1, py, -1, 0), (px, py + 1, 0, -1)]
                };
                for (nx, ny, sx, sy) in nbrs {
                    if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                        continue;
                    }
                    let np = ny as usize * w + nx as usize;
                    if !active[np] {
                        continue;
                    }
                    let cand = nnf[np].step_target(sx, sy);
                    if cand == best_e {
                        let deg = coherent_degree((px, py), best_e, &nnf, &active, (h, w));
                        if deg > best_deg {
                            best_deg = deg;
                            best_eff = coherence_eff(best_cost, best_deg, coh_gain, coh_floor);
                        }
                        continue;
                    }
                    if !is_donor_valid(cand, &donor_ok, &donor_ok_rot, (h, w)) {
                        continue;
                    }
                    let (c, sh) = patch_distance_tp(
                        tp,
                        lvl,
                        s5_img,
                        &gx,
                        &gy,
                        gx_s5,
                        gy_s5,
                        (px, py),
                        cand,
                        diag_inv,
                    );
                    let deg = coherent_degree((px, py), cand, &nnf, &active, (h, w)).max(1);
                    let eff = coherence_eff(c, deg, coh_gain, coh_floor);
                    if eff < best_eff {
                        best_eff = eff;
                        best_cost = c;
                        best_e = cand;
                        best_shift = sh;
                        best_deg = deg;
                    }
                }

                // 2. Exponential random search around `best_e` + edge-gated similarity refinement.
                let mut rad = max_search_rad;
                while rad >= 1 {
                    let (bqx, bqy) = best_e.nearest_px();
                    let rx = (bqx + rng.range_i32(-rad, rad)).clamp(0, w as i32 - 1);
                    let ry = (bqy + rng.range_i32(-rad, rad)).clamp(0, h as i32 - 1);
                    let cand0 = NnfEntry::int(rx, ry);
                    if cand0 != best_e && donor_ok[ry as usize * w + rx as usize] {
                        let (c, sh) = patch_distance_tp(
                            tp,
                            lvl,
                            s5_img,
                            &gx,
                            &gy,
                            gx_s5,
                            gy_s5,
                            (px, py),
                            cand0,
                            diag_inv,
                        );
                        let deg = coherent_degree((px, py), cand0, &nnf, &active, (h, w));
                        let eff = coherence_eff(c, deg, coh_gain, coh_floor);
                        if eff < best_eff {
                            best_eff = eff;
                            best_cost = c;
                            best_e = cand0;
                            best_shift = sh;
                        }
                    }
                    if best_e.rot != 0 {
                        let cand_r = NnfEntry::with_xform(rx, ry, best_e.rot);
                        if cand_r != best_e && donor_ok_rot[ry as usize * w + rx as usize] {
                            let (c, sh) = patch_distance_tp(
                                tp,
                                lvl,
                                s5_img,
                                &gx,
                                &gy,
                                gx_s5,
                                gy_s5,
                                (px, py),
                                cand_r,
                                diag_inv,
                            );
                            let deg = coherent_degree((px, py), cand_r, &nnf, &active, (h, w));
                            let eff = coherence_eff(c, deg, coh_gain, coh_floor);
                            if eff < best_eff {
                                best_eff = eff;
                                best_cost = c;
                                best_e = cand_r;
                                best_shift = sh;
                            }
                        }
                    }
                    if tp.has_edge
                        && best_cost > xform_gate
                        && donor_ok_rot[ry as usize * w + rx as usize]
                    {
                        for _ in 0..2 {
                            let r_xf = 1 + (rng.next_u32() as usize % (N_XFORM - 1)) as u8;
                            let cand_x = NnfEntry::with_xform(rx, ry, r_xf);
                            if cand_x != best_e {
                                let (c, sh) = patch_distance_tp(
                                    tp,
                                    lvl,
                                    s5_img,
                                    &gx,
                                    &gy,
                                    gx_s5,
                                    gy_s5,
                                    (px, py),
                                    cand_x,
                                    diag_inv,
                                );
                                let deg = coherent_degree((px, py), cand_x, &nnf, &active, (h, w));
                                let eff = coherence_eff(c, deg, coh_gain, coh_floor);
                                if eff < best_eff {
                                    best_eff = eff;
                                    best_cost = c;
                                    best_e = cand_x;
                                    best_shift = sh;
                                }
                            }
                        }
                    }
                    rad /= 2;
                }

                // 3. Pivot refinement around `best_e`'s center on structural curves/edges.
                if tp.has_edge && best_cost > xform_gate {
                    let (bqx, bqy) = best_e.nearest_px();
                    if bqx >= 0
                        && bqx < w as i32
                        && bqy >= 0
                        && bqy < h as i32
                        && donor_ok_rot[bqy as usize * w + bqx as usize]
                    {
                        let (n1, n2) = XFORM_NEIGHBORS[best_e.rot as usize];
                        let r_xf = 1 + (rng.next_u32() as usize % (N_XFORM - 1)) as u8;
                        for xf in [n1, n2, r_xf] {
                            let cand_p = if xf == 0 {
                                NnfEntry::int(bqx, bqy)
                            } else {
                                NnfEntry {
                                    qx_q8: best_e.qx_q8,
                                    qy_q8: best_e.qy_q8,
                                    rot: xf,
                                }
                            };
                            if cand_p == best_e
                                || !is_donor_valid(cand_p, &donor_ok, &donor_ok_rot, (h, w))
                            {
                                continue;
                            }
                            let (c, sh) = patch_distance_tp(
                                tp,
                                lvl,
                                s5_img,
                                &gx,
                                &gy,
                                gx_s5,
                                gy_s5,
                                (px, py),
                                cand_p,
                                diag_inv,
                            );
                            let deg = coherent_degree((px, py), cand_p, &nnf, &active, (h, w));
                            let eff = coherence_eff(c, deg, coh_gain, coh_floor);
                            if eff < best_eff {
                                best_eff = eff;
                                best_cost = c;
                                best_e = cand_p;
                                best_shift = sh;
                            }
                        }
                    }
                }

                nnf[p] = best_e;
                costs[p] = best_cost;
                shifts[p] = best_shift;
            }
        }

        // M-step: coherence-weighted color + gradient patch voting, then screened Poisson solve.
        let mut active_costs: Vec<f32> = active_list
            .iter()
            .map(|&(px, py)| costs[py as usize * w + px as usize])
            .filter(|c| c.is_finite())
            .collect();
        let sigma_cost = if active_costs.is_empty() {
            0.01
        } else {
            let med = crate::stats::np_median(&mut active_costs);
            (med * 1.5).max(1.0e-3)
        };

        let (rgb_gx, rgb_gy) = compute_rgb_gradients(&lvl.img, (h, w));
        let (c_vote, gx_vote, gy_vote, w_vote) = vote_m_step(
            lvl,
            &active_list,
            &nnf,
            &shifts,
            &costs,
            sigma_cost,
            &rgb_gx,
            &rgb_gy,
        );

        let mut data = lvl.img.clone();
        let mut gxv = rgb_gx;
        let mut gyv = rgb_gy;
        for p in 0..n {
            if lvl.hole[p] != 0 && w_vote[p] > 0.0 {
                let inv = 1.0 / w_vote[p];
                for c in 0..3 {
                    data[p][c] = c_vote[p][c] * inv;
                    gxv[p][c] = gx_vote[p][c] * inv;
                    gyv[p][c] = gy_vote[p][c] * inv;
                }
            }
        }

        let melded = solve_screened_poisson(&bstate, &data, &gxv, &gyv, SCREEN_LAMBDA, (h, w));
        for ((dst, &hv), &mv) in lvl.img.iter_mut().zip(&lvl.hole).zip(&melded) {
            if hv != 0 {
                *dst = mv;
            }
        }
        final_shifts = shifts;
    }

    (nnf, final_shifts)
}

type HealWrites = Vec<(usize, [f32; 3])>;
type CropLog = (Vec<[f32; 3]>, Vec<f32>, Vec<f32>, Vec<bool>);

#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn load_crop_log(
    img: &[f32],
    w: usize,
    (x0, y0): (usize, usize),
    (ch, cw): (usize, usize),
) -> CropLog {
    let n = ch * cw;
    let mut l0 = vec![[0.0f32; 3]; n];
    let mut log_y = vec![0.0f32; n];
    let mut lin_y = vec![0.0f32; n];
    let mut non_finite = vec![false; n];
    for y in 0..ch {
        let row_src = (y0 + y) * w + x0;
        for x in 0..cw {
            let p = y * cw + x;
            let idx = 3 * (row_src + x);
            let px = &img[idx..idx + 3];
            non_finite[p] = !px[0].is_finite() || !px[1].is_finite() || !px[2].is_finite();
            let c0 = sanitize_trans(px[0]);
            let c1 = sanitize_trans(px[1]);
            let c2 = sanitize_trans(px[2]);
            lin_y[p] = (c0 * LUMA_R + c1 * LUMA_G) + c2 * LUMA_B;
            let lr = [np_logf(c0), np_logf(c1), np_logf(c2)];
            l0[p] = lr;
            log_y[p] = luma_log(lr);
        }
    }
    (l0, log_y, lin_y, non_finite)
}

#[allow(clippy::too_many_arguments)]
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn encode_crop_writes(
    cover: &[u8],
    has_donor: &[bool],
    copied: &[[f32; 3]],
    shading: &[[f32; 3]],
    finest_img: &[[f32; 3]],
    (x0, y0): (usize, usize),
    (ch, cw): (usize, usize),
    w: usize,
) -> HealWrites {
    let mut writes = Vec::new();
    for y in 0..ch {
        for x in 0..cw {
            let p = y * cw + x;
            if cover[p] == 0 {
                continue;
            }
            let mut px = [0.0f32; 3];
            for c in 0..3 {
                let val = if has_donor[p] {
                    copied[p][c] + shading[p][c]
                } else {
                    finest_img[p][c]
                };
                px[c] = np_expf(val.clamp(-16.0, 8.0)).max(0.0);
            }
            writes.push(((y0 + y) * w + x0 + x, px));
        }
    }
    writes
}

fn compute_heal_patch(img: &[f32], (h, w): (usize, usize), s: &HealStroke) -> Option<HealWrites> {
    if !valid_rgb_buf(img, (h, w)) || s.brush < 1 || s.brush > MAX_HEAL_BRUSH {
        return None;
    }
    let (mut x0, mut y0, mut x1, mut y1) = s.rect;
    if x0 >= x1 || y0 >= y1 || x1 > w || y1 > h {
        return None;
    }
    let (mut ch, mut cw) = (y1 - y0, x1 - x0);
    let mut n = ch.checked_mul(cw)?;
    if let Some(av) = s.avoid
        && av.len() != n
    {
        return None;
    }
    if s.path.is_empty() {
        return Some(Vec::new());
    }

    let brush = s.brush;
    let mut shifted_path: Option<Vec<(i64, i64)>> = None;
    if s.avoid.is_none() && n > 262_144 {
        let r_clip = (2 * brush + 4) as f64;
        let bounds = (-r_clip, -r_clip, cw as f64 + r_clip, ch as f64 + r_clip);
        let mut px_min = f64::INFINITY;
        let mut px_max = f64::NEG_INFINITY;
        let mut py_min = f64::INFINITY;
        let mut py_max = f64::NEG_INFINITY;
        let mut any_hit = false;
        if s.path.len() == 1 {
            let p = (s.path[0].0 as f64, s.path[0].1 as f64);
            if let Some((ca, _)) = clip_segment_f64(p, p, bounds) {
                px_min = ca.0;
                px_max = ca.0;
                py_min = ca.1;
                py_max = ca.1;
                any_hit = true;
            }
        } else {
            for pair in s.path.windows(2) {
                let a = (pair[0].0 as f64, pair[0].1 as f64);
                let b = (pair[1].0 as f64, pair[1].1 as f64);
                if let Some((ca, cb)) = clip_segment_f64(a, b, bounds) {
                    px_min = px_min.min(ca.0.min(cb.0));
                    px_max = px_max.max(ca.0.max(cb.0));
                    py_min = py_min.min(ca.1.min(cb.1));
                    py_max = py_max.max(ca.1.max(cb.1));
                    any_hit = true;
                }
            }
        }
        if !any_hit {
            return Some(Vec::new());
        }
        let cx_min = px_min.clamp(0.0, cw as f64);
        let cx_max = px_max.clamp(0.0, cw as f64);
        let cy_min = py_min.clamp(0.0, ch as f64);
        let cy_max = py_max.clamp(0.0, ch as f64);
        let min_span = (cx_max - cx_min).min(cy_max - cy_min);
        let tight_pad = context_pad(brush, min_span) + 16;
        let tx0 = (libm::floor(cx_min) as isize)
            .saturating_sub(tight_pad)
            .clamp(0, cw as isize) as usize;
        let ty0 = (libm::floor(cy_min) as isize)
            .saturating_sub(tight_pad)
            .clamp(0, ch as isize) as usize;
        let tx1 = (libm::ceil(cx_max) as isize)
            .saturating_add(tight_pad + 1)
            .clamp(0, cw as isize) as usize;
        let ty1 = (libm::ceil(cy_max) as isize)
            .saturating_add(tight_pad + 1)
            .clamp(0, ch as isize) as usize;
        if tx0 < tx1 && ty0 < ty1 && (tx1 - tx0) * (ty1 - ty0) < (n / 4) * 3 {
            shifted_path = Some(
                s.path
                    .iter()
                    .map(|&(px, py)| (px.saturating_sub(tx0 as i64), py.saturating_sub(ty0 as i64)))
                    .collect(),
            );
            x1 = x0 + tx1;
            y1 = y0 + ty1;
            x0 += tx0;
            y0 += ty0;
            cw = x1 - x0;
            ch = y1 - y0;
            n = ch * cw;
        }
    }
    let path_ref: &[(i64, i64)] = shifted_path.as_deref().unwrap_or(s.path);

    let mut cover = vec![0u8; n];
    rasterize_stroke_clipped(&mut cover, cw, ch, path_ref, brush);
    if !cover.iter().any(|&c| c != 0) || cover.iter().all(|&c| c != 0) {
        return Some(Vec::new());
    }

    let (l0, log_y, lin_y, non_finite) = load_crop_log(img, w, (x0, y0), (ch, cw));

    // Dilate `cover` by a PSF skirt (`collect_valid_donors` adds another `R_PATCH = 2` px)
    // so donor patches clear soft defect rims while staying close enough to follow curves.
    let margin = (brush as usize / 8).clamp(1, 6);
    let m_side = 2 * margin + 1;
    let m_cells = morph::ellipse_kernel(m_side, m_side);
    let mut skirt = vec![0u8; n];
    morph::morphology_mask(
        &cover,
        &mut skirt,
        (ch, cw, 1),
        Op::Dilate,
        Kernel {
            mask: &m_cells,
            w: m_side,
            h: m_side,
        },
        1,
        crate::filter::Border::Constant,
    )?;

    let mut skirt_or_nf = vec![0u8; n];
    for p in 0..n {
        if skirt[p] != 0 {
            skirt_or_nf[p] |= HARD_SKIRT;
        }
        if non_finite[p] {
            skirt_or_nf[p] |= HARD_AVOID;
        }
    }
    let speck_0 = detect_point_specks(&log_y, &cover, &skirt_or_nf, (ch, cw), 6.5, 0.45);
    drop(skirt_or_nf);
    let mut severe_speck: Vec<bool> = speck_0.iter().map(|&v| v != 0).collect();

    let dst_lumas: Vec<f32> = (0..n)
        .filter(|&p| cover[p] == 0 && !severe_speck[p] && !non_finite[p])
        .map(|p| lin_y[p])
        .collect();
    let dst_live = !dst_lumas.is_empty() && median(&dst_lumas) > 5.0e-3;
    drop(dst_lumas);

    let mut hard_bad = vec![0u8; n];
    for p in 0..n {
        let av = s.avoid.is_some_and(|m| m[p] != 0);
        let dead = dst_live && lin_y[p] < 2.0e-4;
        let mut hb = 0u8;
        if skirt[p] != 0 {
            hb |= HARD_SKIRT;
        }
        if dead {
            hb |= HARD_DEAD;
        }
        if av || non_finite[p] {
            hb |= HARD_AVOID;
        }
        hard_bad[p] = hb;
    }
    drop(skirt);
    drop(lin_y);

    let ext_ly0: Vec<f32> = (0..n)
        .filter(|&p| cover[p] == 0 && (hard_bad[p] & HARD_EXCL) == 0 && !severe_speck[p])
        .map(|p| log_y[p])
        .collect();
    let med_ly0 = if !ext_ly0.is_empty() {
        Some(median(&ext_ly0))
    } else {
        None
    };
    drop(ext_ly0);

    let s5_l0 = smooth3_rgb(&smooth3_rgb(&l0, (ch, cw)), (ch, cw));
    let hole_depth_0 = max_hole_depth(&cover, (ch, cw));
    let grain_floor = measure_grain_floor(&l0, &hard_bad, &speck_0, (ch, cw));

    // Build coarse-to-fine pyramid until the hole's inner depth drops to <= 2 px.
    let mut pyramid = vec![Level {
        h: ch,
        w: cw,
        img: l0.clone(),
        hole: cover.clone(),
        hard_bad,
        speck_bad: speck_0,
        grain_floor,
    }];
    while pyramid.len() < 9 {
        let last = pyramid.last().unwrap();
        if last.h.min(last.w) < 16 || max_hole_depth(&last.hole, (last.h, last.w)) <= 2.0 {
            break;
        }
        let next = downsample_level(last);
        if !next.hole.iter().any(|&v| v != 0) {
            break;
        }
        pyramid.push(next);
    }

    // Back-project multi-scale point specks onto finer levels so level-0 random search
    // also excludes wide `8K`/`12K` dust blobs whose interiors look flat in a `±3 px` ring.
    if pyramid.len() > 1
        && let Some(med_y) = med_ly0
    {
        for lvl_idx in (1..pyramid.len()).rev() {
            let (ch_c, cw_c) = (pyramid[lvl_idx].h, pyramid[lvl_idx].w);
            let coarse_sb = pyramid[lvl_idx].speck_bad.clone();
            let fine = &mut pyramid[lvl_idx - 1];
            for fy in 0..fine.h {
                let cy = (fy / 2).min(ch_c - 1);
                for fx in 0..fine.w {
                    let cx = (fx / 2).min(cw_c - 1);
                    if coarse_sb[cy * cw_c + cx] != 0 {
                        let fp = fy * fine.w + fx;
                        if fine.hole[fp] == 0
                            && (fine.hard_bad[fp] & HARD_AVOID) == 0
                            && (luma_log(fine.img[fp]) - med_y).abs() > 0.35
                        {
                            fine.speck_bad[fp] = 1;
                        }
                    }
                }
            }
        }
        for (sv, &sb) in severe_speck.iter_mut().zip(&pyramid[0].speck_bad) {
            if sb != 0 {
                *sv = true;
            }
        }
    }
    drop(log_y);

    // Seed the coarsest level's hole with harmonic Laplace interpolation.
    {
        let top = pyramid.last_mut().unwrap();
        let bstate = build_boundary_state(
            &top.img,
            &top.hole,
            &top.hard_bad,
            &top.speck_bad,
            (top.h, top.w),
        );
        if bstate.contains(&CELL_DIRICHLET) {
            let init = solve_harmonic_init(&bstate, &top.img, (top.h, top.w));
            for ((dst, &hv), &iv) in top.img.iter_mut().zip(&top.hole).zip(&init) {
                if hv != 0 {
                    *dst = iv;
                }
            }
        } else {
            let mut sum = [0.0f64; 3];
            let mut cnt = 0usize;
            for p in 0..top.h * top.w {
                if top.hole[p] == 0 && (top.hard_bad[p] & HARD_EXCL) == 0 {
                    sum[0] += top.img[p][0] as f64;
                    sum[1] += top.img[p][1] as f64;
                    sum[2] += top.img[p][2] as f64;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                let mean = [
                    (sum[0] / cnt as f64) as f32,
                    (sum[1] / cnt as f64) as f32,
                    (sum[2] / cnt as f64) as f32,
                ];
                for (dst, &hv) in top.img.iter_mut().zip(&top.hole) {
                    if hv != 0 {
                        *dst = mean;
                    }
                }
            }
        }
    }

    // Solve coarse-to-fine.
    let num_levels = pyramid.len();
    let mut prev_nnf: Option<OwnedNnf> = None;
    for lvl_idx in (0..num_levels).rev() {
        let coarse_ref = prev_nnf.as_ref().map(|(v, dims)| (v.as_slice(), *dims));
        let (nnf, _) = solve_level(&mut pyramid[lvl_idx], coarse_ref, lvl_idx);
        let dims = (pyramid[lvl_idx].h, pyramid[lvl_idx].w);
        if lvl_idx > 0 {
            let coarse_img = pyramid[lvl_idx].img.clone();
            let fine = &mut pyramid[lvl_idx - 1];
            for y in 0..fine.h {
                let cy = (y / 2).min(dims.0 - 1);
                for x in 0..fine.w {
                    let fp = y * fine.w + x;
                    if fine.hole[fp] != 0 {
                        let cx = (x / 2).min(dims.1 - 1);
                        fine.img[fp] = coarse_img[cy * dims.1 + cx];
                    }
                }
            }
        }
        prev_nnf = Some((nnf, dims));
    }

    let finest = pyramid.swap_remove(0);
    drop(pyramid);
    let (nnf_0, _) = prev_nnf.unwrap();
    let (donor_ok_0, donor_list_0) = collect_valid_donors(&finest);
    let donor_ok_rot_0 = build_donor_ok_rot(&donor_ok_0, (ch, cw));

    let clean_src: Vec<bool> = (0..n)
        .map(|p| {
            cover[p] == 0
                && !severe_speck[p]
                && !non_finite[p]
                && (finest.hard_bad[p] & HARD_EXCL) == 0
        })
        .collect();
    let clean_donors_0: Vec<(i32, i32)> = donor_list_0
        .into_iter()
        .filter(|&(dx, dy)| clean_src[dy as usize * cw + dx as usize])
        .collect();

    let mut has_donor = vec![false; n];
    let mut donors = vec![NnfEntry::int(0, 0); n];
    for y in 0..ch {
        for x in 0..cw {
            let p = y * cw + x;
            if cover[p] == 0 {
                continue;
            }
            let mut de = nnf_0[p];
            let (mut qx, mut qy) = de.nearest_px();
            qx = qx.clamp(0, cw as i32 - 1);
            qy = qy.clamp(0, ch as i32 - 1);
            let mut q = qy as usize * cw + qx as usize;
            if de.rot != 0 && !donor_ok_rot_0[q] {
                de = NnfEntry::int(qx, qy);
            }
            if (!donor_ok_0[q] || !clean_src[q]) && !clean_donors_0.is_empty() {
                let mut found = None;
                // Prefer coherent continuation from resolved left/top neighbors first.
                for (has_nbr, np, sx_step, sy_step) in [
                    (x > 0 && has_donor[p - 1], p.wrapping_sub(1), 1i32, 0i32),
                    (y > 0 && has_donor[p - cw], p.wrapping_sub(cw), 0, 1),
                ] {
                    if has_nbr {
                        let (nx, ny) = donors[np].step_target(sx_step, sy_step).nearest_px();
                        if nx >= 0 && nx < cw as i32 && ny >= 0 && ny < ch as i32 {
                            let nq = ny as usize * cw + nx as usize;
                            if donor_ok_0[nq] && clean_src[nq] {
                                found = Some((nx, ny));
                                break;
                            }
                        }
                    }
                }
                if found.is_none() {
                    'ring: for rad in 1..=12i32 {
                        for dy in -rad..=rad {
                            for dx in -rad..=rad {
                                if dx.abs().max(dy.abs()) != rad {
                                    continue;
                                }
                                let sx = qx + dx;
                                let sy = qy + dy;
                                if sx >= 0 && sx < cw as i32 && sy >= 0 && sy < ch as i32 {
                                    let sq = sy as usize * cw + sx as usize;
                                    if donor_ok_0[sq] && clean_src[sq] {
                                        found = Some((sx, sy));
                                        break 'ring;
                                    }
                                }
                            }
                        }
                    }
                }
                let (nx, ny) = found.unwrap_or_else(|| {
                    let mut best = clean_donors_0[0];
                    let mut best_d2 = i64::MAX;
                    let stride = (clean_donors_0.len() / 256).max(1);
                    for &(dx, dy) in clean_donors_0.iter().step_by(stride) {
                        let d2 = (dx as i64 - x as i64).pow(2) + (dy as i64 - y as i64).pow(2);
                        if d2 < best_d2 {
                            best_d2 = d2;
                            best = (dx, dy);
                        }
                    }
                    best
                });
                qx = nx;
                qy = ny;
                q = qy as usize * cw + qx as usize;
                de = NnfEntry::int(qx, qy);
            }
            if clean_src[q] {
                donors[p] = de;
                has_donor[p] = true;
            }
        }
    }

    // Remove isolated 1x1 confetti donor islands inside `cover` strictly within the scan's own
    // measured grain variance (`despeckle_bar`), preserving grainless 1-px curve pivots untouched.
    let despeckle_bar = 1.5 * (grain_floor.0 + grain_floor.2);
    if despeckle_bar > 2.0e-3 {
        for _ in 0..2 {
            let prev_donors = donors.clone();
            for y in 0..ch {
                for x in 0..cw {
                    let p = y * cw + x;
                    if !has_donor[p] {
                        continue;
                    }
                    let de = prev_donors[p];
                    let nbrs = [
                        (x as i32 - 1, y as i32, 1i32, 0i32),
                        (x as i32 + 1, y as i32, -1, 0),
                        (x as i32, y as i32 - 1, 0, 1),
                        (x as i32, y as i32 + 1, 0, -1),
                    ];
                    let mut coh_self = 0u8;
                    for &(nx, ny, sx, sy) in &nbrs {
                        if nx >= 0 && nx < cw as i32 && ny >= 0 && ny < ch as i32 {
                            let np = ny as usize * cw + nx as usize;
                            if has_donor[np] && entries_coherent(prev_donors[np], sx, sy, de) {
                                coh_self += 1;
                            }
                        }
                    }
                    if coh_self > 0 {
                        continue;
                    }
                    let mut best_cand = None;
                    let mut best_votes = 0u8;
                    let mut best_err = f32::INFINITY;
                    for &(nx, ny, sx, sy) in &nbrs {
                        if nx < 0 || nx >= cw as i32 || ny < 0 || ny >= ch as i32 {
                            continue;
                        }
                        let np = ny as usize * cw + nx as usize;
                        if !has_donor[np] {
                            continue;
                        }
                        let cand = prev_donors[np].step_target(sx, sy);
                        if !is_donor_valid(cand, &donor_ok_0, &donor_ok_rot_0, (ch, cw)) {
                            continue;
                        }
                        let (cqx, cqy) = cand.nearest_px();
                        let cq = cqy as usize * cw + cqx as usize;
                        if !clean_src[cq] {
                            continue;
                        }
                        let mut votes = 0u8;
                        for &(mx, my, msx, msy) in &nbrs {
                            if mx >= 0 && mx < cw as i32 && my >= 0 && my < ch as i32 {
                                let mp = my as usize * cw + mx as usize;
                                if has_donor[mp]
                                    && entries_coherent(prev_donors[mp], msx, msy, cand)
                                {
                                    votes += 1;
                                }
                            }
                        }
                        let sm_c = sample_donor_step_smooth(&s5_l0, (ch, cw), cand, 0, 0);
                        let d0 = sm_c[0] - finest.img[p][0];
                        let d1 = sm_c[1] - finest.img[p][1];
                        let d2 = sm_c[2] - finest.img[p][2];
                        let err = d0 * d0 + d1 * d1 + d2 * d2;
                        if err <= despeckle_bar
                            && (votes > best_votes || (votes == best_votes && err < best_err))
                        {
                            best_votes = votes;
                            best_err = err;
                            best_cand = Some(cand);
                        }
                    }
                    if let Some(cand) = best_cand {
                        donors[p] = cand;
                    }
                }
            }
        }
    }

    let mut copied = finest.img.clone();
    for p in 0..n {
        if has_donor[p] {
            let de = donors[p];
            copied[p] = if de.rot == 0 {
                let (qx, qy) = de.nearest_px();
                l0[qy as usize * cw + qx as usize]
            } else {
                catmull_rom_rgb(&l0, (ch, cw), de.qx_q8, de.qy_q8)
            };
        }
    }

    let bstate_0 = build_boundary_state(
        &finest.img,
        &finest.hole,
        &finest.hard_bad,
        &finest.speck_bad,
        (ch, cw),
    );
    let shading = solve_donor_shading(
        &finest.img,
        &l0,
        &s5_l0,
        hole_depth_0,
        grain_floor,
        &bstate_0,
        &clean_src,
        &has_donor,
        &donors,
        (ch, cw),
    );
    Some(encode_crop_writes(
        &cover,
        &has_donor,
        &copied,
        &shading,
        &finest.img,
        (x0, y0),
        (ch, cw),
        w,
    ))
}

struct SubStroke {
    rect: (usize, usize, usize, usize),
    path: Vec<(i64, i64)>,
    brush: i64,
    avoid: Option<Vec<u8>>,
}

/// Partition sparse, long polylines into overlapping arc-length crops so memory stays O(L * brush).
/// Each sub-stroke masks out unhealed future continuations while fusing onto healed prefixes.
fn subdivide_long_stroke(s: &HealStroke, (h, w): (usize, usize)) -> Option<Vec<SubStroke>> {
    if s.avoid.is_some() || s.path.len() < 2 {
        return None;
    }
    let (x0, y0, x1, y1) = s.rect;
    if x0 >= x1 || y0 >= y1 || x1 > w || y1 > h || s.brush < 1 || s.brush > MAX_HEAL_BRUSH {
        return None;
    }
    let bbox_area = (x1 - x0).checked_mul(y1 - y0)?;
    if bbox_area <= 131_072 {
        return None;
    }
    let seg_limit = (8 * s.brush).clamp(128, 384) as f64;
    let overlap = (2 * s.brush).clamp(8, 96) as f64;

    let r_clip = (2 * s.brush + 4) as f64;
    let clip_bounds = (
        x0 as f64 - r_clip,
        y0 as f64 - r_clip,
        x1 as f64 + r_clip,
        y1 as f64 + r_clip,
    );
    let mut clipped_segs: Vec<((f64, f64), (f64, f64))> = Vec::with_capacity(s.path.len());
    let mut c_min_x = f64::INFINITY;
    let mut c_max_x = f64::NEG_INFINITY;
    let mut c_min_y = f64::INFINITY;
    let mut c_max_y = f64::NEG_INFINITY;
    let mut raw_clipped_arc = 0.0f64;
    for pair in s.path.windows(2) {
        let a = (x0 as f64 + pair[0].0 as f64, y0 as f64 + pair[0].1 as f64);
        let b = (x0 as f64 + pair[1].0 as f64, y0 as f64 + pair[1].1 as f64);
        let Some((ca, cb)) = clip_segment_f64(a, b, clip_bounds) else {
            continue;
        };
        let dist = libm::hypot(cb.0 - ca.0, cb.1 - ca.1);
        if dist < 1.0e-6 {
            continue;
        }
        c_min_x = c_min_x.min(ca.0.min(cb.0));
        c_max_x = c_max_x.max(ca.0.max(cb.0));
        c_min_y = c_min_y.min(ca.1.min(cb.1));
        c_max_y = c_max_y.max(ca.1.max(cb.1));
        raw_clipped_arc += dist;
        clipped_segs.push((ca, cb));
    }
    if clipped_segs.is_empty() {
        return None;
    }
    let span_x = (c_max_x.min(x1 as f64) - c_min_x.max(x0 as f64)).max(0.0);
    let span_y = (c_max_y.min(y1 as f64) - c_min_y.max(y0 as f64)).max(0.0);
    let span = span_x.max(span_y);
    if span <= seg_limit * 1.35 || raw_clipped_arc <= seg_limit * 1.35 {
        return None;
    }
    // Dense 2D back-and-forth scribbles heal faster and cleaner as one unified mask.
    if raw_clipped_arc > 6.0 * (span_x + span_y + 2.0 * s.brush as f64) {
        return None;
    }

    type DenseRun = (Vec<(f64, f64)>, Vec<f64>);
    let step_pt = (2 * s.brush).clamp(16, 64) as f64;
    let mut runs: Vec<DenseRun> = Vec::new();
    for (ca, cb) in clipped_segs {
        let dx = cb.0 - ca.0;
        let dy = cb.1 - ca.1;
        let dist = libm::hypot(dx, dy);
        let steps = (libm::ceil(dist / step_pt) as usize).max(1);
        let cont = runs.last().is_some_and(|(d, _)| {
            let last = *d.last().unwrap();
            (last.0 - ca.0).abs() <= 1.0e-3 && (last.1 - ca.1).abs() <= 1.0e-3
        });
        if !cont {
            runs.push((vec![ca], vec![0.0f64]));
        }
        let (dense, cum) = runs.last_mut().unwrap();
        let base_arc = *cum.last().unwrap();
        for k in 1..=steps {
            let t = k as f64 / steps as f64;
            dense.push((ca.0 + dx * t, ca.1 + dy * t));
            cum.push(base_arc + dist * t);
        }
    }

    let stride = (seg_limit - overlap).max(32.0);
    let mut windows: Vec<(usize, usize, usize)> = Vec::new();
    for (run_idx, (dense, cum)) in runs.iter().enumerate() {
        if dense.len() < 2 {
            continue;
        }
        let run_arc = *cum.last().unwrap();
        if run_arc <= 1.0e-6 {
            continue;
        }
        let mut start_arc = 0.0f64;
        let mut i0 = 0usize;
        while start_arc < run_arc {
            let end_arc = (start_arc + seg_limit).min(run_arc);
            while i0 + 1 < cum.len() && cum[i0 + 1] <= start_arc {
                i0 += 1;
            }
            let mut i1 = i0;
            while i1 + 1 < cum.len() && cum[i1] < end_arc {
                i1 += 1;
            }
            if i1 > i0 {
                windows.push((run_idx, i0, i1));
                if windows.len() > 256 {
                    return None;
                }
            }
            if i1 + 1 >= cum.len() {
                break;
            }
            start_arc += stride;
        }
    }
    if windows.len() < 2 {
        return None;
    }

    let r_av = s.brush + (s.brush / 8).clamp(1, 6);
    let mut subs = Vec::with_capacity(windows.len());
    for &(run_idx, i0, i1) in &windows {
        let slice = &runs[run_idx].0[i0..=i1];
        let mut wx_min = f64::INFINITY;
        let mut wx_max = f64::NEG_INFINITY;
        let mut wy_min = f64::INFINITY;
        let mut wy_max = f64::NEG_INFINITY;
        for &(fx, fy) in slice {
            wx_min = wx_min.min(fx);
            wx_max = wx_max.max(fx);
            wy_min = wy_min.min(fy);
            wy_max = wy_max.max(fy);
        }
        let cx_min = wx_min.clamp(x0 as f64, x1 as f64);
        let cx_max = wx_max.clamp(x0 as f64, x1 as f64);
        let cy_min = wy_min.clamp(y0 as f64, y1 as f64);
        let cy_max = wy_max.clamp(y0 as f64, y1 as f64);
        let min_span = (cx_max - cx_min).min(cy_max - cy_min);
        let pad = context_pad(s.brush, min_span);
        let sx0 = (libm::floor(cx_min) as isize)
            .saturating_sub(pad)
            .clamp(x0 as isize, x1 as isize) as usize;
        let sy0 = (libm::floor(cy_min) as isize)
            .saturating_sub(pad)
            .clamp(y0 as isize, y1 as isize) as usize;
        let sx1 = (libm::ceil(cx_max) as isize)
            .saturating_add(pad + 1)
            .clamp(x0 as isize, x1 as isize) as usize;
        let sy1 = (libm::ceil(cy_max) as isize)
            .saturating_add(pad + 1)
            .clamp(y0 as isize, y1 as isize) as usize;
        if sx0 >= sx1 || sy0 >= sy1 {
            continue;
        }
        let scw = sx1 - sx0;
        let sch = sy1 - sy0;
        let local_path: Vec<(i64, i64)> = slice
            .iter()
            .map(|&(fx, fy)| {
                (
                    libm::rint(fx) as i64 - sx0 as i64,
                    libm::rint(fy) as i64 - sy0 as i64,
                )
            })
            .collect();

        let has_future = i1 + 1 < runs[run_idx].0.len() || run_idx + 1 < runs.len();
        let avoid = if has_future {
            let bx0 = sx0 as f64 - (r_av + 2) as f64;
            let by0 = sy0 as f64 - (r_av + 2) as f64;
            let bx1 = sx1 as f64 + (r_av + 2) as f64;
            let by1 = sy1 as f64 + (r_av + 2) as f64;
            let mut tail_mask = vec![0u8; scw * sch];
            let mut touched = false;
            let cur_tail = runs[run_idx].0[i1..].windows(2);
            let later_tails = runs[run_idx + 1..].iter().flat_map(|(d, _)| d.windows(2));
            for pair in cur_tail.chain(later_tails) {
                let (ax, ay) = pair[0];
                let (bx, by) = pair[1];
                if ax.max(bx) < bx0 || ax.min(bx) > bx1 || ay.max(by) < by0 || ay.min(by) > by1 {
                    continue;
                }
                let seg_pts = [
                    (
                        libm::rint(ax) as i64 - sx0 as i64,
                        libm::rint(ay) as i64 - sy0 as i64,
                    ),
                    (
                        libm::rint(bx) as i64 - sx0 as i64,
                        libm::rint(by) as i64 - sy0 as i64,
                    ),
                ];
                rasterize_stroke_clipped(&mut tail_mask, scw, sch, &seg_pts, r_av);
                touched = true;
            }
            if touched {
                let mut self_cover = vec![0u8; scw * sch];
                rasterize_stroke_clipped(&mut self_cover, scw, sch, &local_path, s.brush);
                let mut any_left = false;
                for (av, &sc) in tail_mask.iter_mut().zip(&self_cover) {
                    if sc != 0 {
                        *av = 0;
                    } else if *av != 0 {
                        any_left = true;
                    }
                }
                if any_left { Some(tail_mask) } else { None }
            } else {
                None
            }
        } else {
            None
        };
        subs.push(SubStroke {
            rect: (sx0, sy0, sx1, sy1),
            path: local_path,
            brush: s.brush,
            avoid,
        });
    }

    if subs.len() >= 2 { Some(subs) } else { None }
}

/// Apply one multi-scale PatchMatch + Image Melding stroke in place on an `HxWx3` linear image.
pub fn heal_stroke(out: &mut [f32], (h, w): (usize, usize), s: &HealStroke) -> Option<()> {
    if !valid_rgb_buf(out, (h, w)) || s.brush < 1 || s.brush > MAX_HEAL_BRUSH {
        return None;
    }
    if let Some(subs) = subdivide_long_stroke(s, (h, w)) {
        let refs: Vec<HealStroke<'_>> = subs
            .iter()
            .map(|sub| HealStroke {
                rect: sub.rect,
                path: &sub.path,
                brush: sub.brush,
                avoid: sub.avoid.as_deref(),
            })
            .collect();
        return heal_strokes_flat(out, (h, w), &refs);
    }
    let writes = compute_heal_patch(out, (h, w), s)?;
    for (p, px) in writes {
        out[3 * p..3 * p + 3].copy_from_slice(&px);
    }
    Some(())
}

#[inline]
fn rects_overlap(a: (usize, usize, usize, usize), b: (usize, usize, usize, usize)) -> bool {
    a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
}

/// Tight bounding box of pixels that `s` can write (`path_bbox ± (brush + 1)` clamped to `s.rect`).
fn stroke_write_rect(s: &HealStroke) -> (usize, usize, usize, usize) {
    let (x0, y0, x1, y1) = s.rect;
    if s.path.is_empty() {
        return (x0, y0, x0, y0);
    }
    let mut min_px = i64::MAX;
    let mut max_px = i64::MIN;
    let mut min_py = i64::MAX;
    let mut max_py = i64::MIN;
    for &(px, py) in s.path {
        min_px = min_px.min(px);
        max_px = max_px.max(px);
        min_py = min_py.min(py);
        max_py = max_py.max(py);
    }
    let r = s.brush.saturating_add(1);
    let x0_i = x0 as i64;
    let y0_i = y0 as i64;
    let x1_i = x1 as i64;
    let y1_i = y1 as i64;
    let wx0 = x0_i
        .saturating_add(min_px)
        .saturating_sub(r)
        .clamp(x0_i, x1_i) as usize;
    let wy0 = y0_i
        .saturating_add(min_py)
        .saturating_sub(r)
        .clamp(y0_i, y1_i) as usize;
    let wx1 = x0_i
        .saturating_add(max_px)
        .saturating_add(r.saturating_add(1))
        .clamp(x0_i, x1_i) as usize;
    let wy1 = y0_i
        .saturating_add(max_py)
        .saturating_add(r.saturating_add(1))
        .clamp(y0_i, y1_i) as usize;
    (wx0, wy0, wx1, wy1)
}

fn heal_strokes_flat(
    out: &mut [f32],
    (h, w): (usize, usize),
    strokes: &[HealStroke],
) -> Option<()> {
    const WAVE_AREA_BUDGET: usize = 16_777_216;
    if strokes.is_empty() {
        return Some(());
    }
    if strokes.len() == 1 {
        let writes = compute_heal_patch(out, (h, w), &strokes[0])?;
        for (p, px) in writes {
            out[3 * p..3 * p + 3].copy_from_slice(&px);
        }
        return Some(());
    }
    let m = strokes.len();
    let write_rects: Vec<(usize, usize, usize, usize)> =
        strokes.iter().map(stroke_write_rect).collect();
    let mut wave_of = vec![0usize; m];
    let mut max_wave = 0usize;
    for k in 0..m {
        let rk = strokes[k].rect;
        let wk = write_rects[k];
        let mut w_idx = 0usize;
        for j in 0..k {
            // RAW / WAW (`write[j]` touches `read[k]`): `k` runs in a strictly later wave.
            if rects_overlap(write_rects[j], rk) {
                w_idx = w_idx.max(wave_of[j] + 1);
            } else if rects_overlap(strokes[j].rect, wk) {
                // WAR (`write[k]` touches `read[j]`): `k` runs in the same or later wave.
                w_idx = w_idx.max(wave_of[j]);
            }
        }
        wave_of[k] = w_idx;
        max_wave = max_wave.max(w_idx);
    }
    let mut waves: Vec<Vec<usize>> = vec![Vec::new(); max_wave + 1];
    for (k, &wv) in wave_of.iter().enumerate() {
        waves[wv].push(k);
    }
    for wave in waves {
        if wave.len() == 1 {
            let writes = compute_heal_patch(out, (h, w), &strokes[wave[0]])?;
            for (p, px) in writes {
                out[3 * p..3 * p + 3].copy_from_slice(&px);
            }
            continue;
        }
        let mut patches_all: Vec<HealWrites> = Vec::with_capacity(wave.len());
        let img_ref: &[f32] = out;
        let mut chunk_start = 0usize;
        while chunk_start < wave.len() {
            let mut chunk_end = chunk_start;
            let mut acc_area = 0usize;
            while chunk_end < wave.len() {
                let r = strokes[wave[chunk_end]].rect;
                let area = (r.2 - r.0).saturating_mul(r.3 - r.1);
                if chunk_end > chunk_start && acc_area.saturating_add(area) > WAVE_AREA_BUDGET {
                    break;
                }
                acc_area = acc_area.saturating_add(area);
                chunk_end += 1;
            }
            let sub = &wave[chunk_start..chunk_end];
            if sub.len() == 1 {
                patches_all.push(compute_heal_patch(img_ref, (h, w), &strokes[sub[0]])?);
            } else {
                let mut sub_patches = sub
                    .par_iter()
                    .map(|&idx| compute_heal_patch(img_ref, (h, w), &strokes[idx]))
                    .collect::<Option<Vec<_>>>()?;
                patches_all.append(&mut sub_patches);
            }
            chunk_start = chunk_end;
        }
        for writes in patches_all {
            for (p, px) in writes {
                out[3 * p..3 * p + 3].copy_from_slice(&px);
            }
        }
    }
    Some(())
}

/// Apply an ordered slice of manual heal strokes in place, parallelizing non-overlapping crops
/// across rayon waves so the output is bit-for-bit identical to sequential stroke application.
pub fn heal_strokes(out: &mut [f32], (h, w): (usize, usize), strokes: &[HealStroke]) -> Option<()> {
    if !valid_rgb_buf(out, (h, w)) {
        return None;
    }
    if strokes.is_empty() {
        return Some(());
    }
    for s in strokes {
        if s.brush < 1 || s.brush > MAX_HEAL_BRUSH {
            return None;
        }
        let (x0, y0, x1, y1) = s.rect;
        if x0 >= x1 || y0 >= y1 || x1 > w || y1 > h {
            return None;
        }
    }

    let mut sub_storage: Vec<Option<Vec<SubStroke>>> = Vec::with_capacity(strokes.len());
    let mut any_sub = false;
    for s in strokes {
        let sub = subdivide_long_stroke(s, (h, w));
        if sub.is_some() {
            any_sub = true;
        }
        sub_storage.push(sub);
    }
    if !any_sub {
        return heal_strokes_flat(out, (h, w), strokes);
    }

    let mut expanded: Vec<HealStroke<'_>> = Vec::new();
    for (s, maybe_subs) in strokes.iter().zip(&sub_storage) {
        if let Some(subs) = maybe_subs {
            for sub in subs {
                expanded.push(HealStroke {
                    rect: sub.rect,
                    path: &sub.path,
                    brush: sub.brush,
                    avoid: sub.avoid.as_deref(),
                });
            }
        } else {
            expanded.push(HealStroke {
                rect: s.rect,
                path: s.path,
                brush: s.brush,
                avoid: s.avoid,
            });
        }
    }
    heal_strokes_flat(out, (h, w), &expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_grain(h: usize, w: usize, base: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; 3 * h * w];
        let mut s = 0x1234_5678u32;
        for v in &mut out {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
            *v = (base + 0.03 * n).clamp(0.01, 0.99);
        }
        out
    }

    #[test]
    fn heals_speck_dodges_neighbour_dust_and_preserves_film_grain() {
        let (h, w) = (120, 120);
        let clean = synthetic_grain(h, w, 0.45);
        let mut img = clean.clone();
        // Primary speck at (60, 60) and unclicked neighbour dust specks nearby.
        for (cy, cx) in [(60usize, 60usize), (60, 76), (46, 60)] {
            for y in cy - 3..=cy + 3 {
                for x in cx - 3..=cx + 3 {
                    let i = 3 * (y * w + x);
                    img[i..i + 3].fill(0.95);
                }
            }
        }
        let orig = img.clone();
        heal_stroke(
            &mut img,
            (h, w),
            &HealStroke {
                rect: (20, 20, 100, 100),
                path: &[(40, 40)],
                brush: 8,
                avoid: None,
            },
        )
        .unwrap();

        // Mean level restored and real grain variance preserved inside healed disc!
        let mut vals = Vec::new();
        for y in 57..=63 {
            for x in 57..=63 {
                vals.push(img[3 * (y * w + x)]);
            }
        }
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        let var: f32 =
            vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
        assert!((mean - 0.45).abs() < 0.02, "mean={mean}");
        assert!(
            var.sqrt() > 0.004,
            "grain was blurred away: std={}",
            var.sqrt()
        );

        // Outside the brush radius (8 px) must stay byte-identical.
        assert_eq!(img[3 * (60 * w + 76)], orig[3 * (60 * w + 76)]);
    }

    #[test]
    fn reconstructs_horizon_and_gradient_across_curved_hair() {
        let (h, w) = (140, 180);
        let mut clean = vec![0.0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let ramp = 0.25 + 0.35 * (x as f32 / w as f32);
                let step = if y < 70 { 1.0 } else { 0.40 };
                let i = 3 * (y * w + x);
                clean[i..i + 3].fill(ramp * step);
            }
        }
        let mut img = clean.clone();
        // Curved hair crossing sky -> horizon -> ground!
        let mut path = Vec::new();
        for y in 30..110 {
            let x = 90 + ((y as i64 - 70) * (y as i64 - 70)) / 80;
            path.push((x - 40, y as i64 - 10));
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let yy = (y as i64 + dy) as usize;
                    let xx = (x + dx) as usize;
                    let i = 3 * (yy * w + xx);
                    img[i..i + 3].fill(0.96);
                }
            }
        }

        heal_stroke(
            &mut img,
            (h, w),
            &HealStroke {
                rect: (40, 10, 150, 130),
                path: &path,
                brush: 5,
                avoid: None,
            },
        )
        .unwrap();

        for y in 35..105 {
            let x = (90 + ((y as i64 - 70) * (y as i64 - 70)) / 80) as usize;
            let got = img[3 * (y * w + x)];
            let want = clean[3 * (y * w + x)];
            assert!(
                (got - want).abs() < 0.03,
                "row {y} col {x}: got {got:.4}, want {want:.4}"
            );
        }
    }

    #[test]
    fn reconstructs_tight_circular_arc_via_rotated_patches() {
        // Upper semicircle only (`rect` stops above circle center `(60, 58)`), with the horizontal
        // top apex `(60, 32)` wiped out by a vertical scratch: NO unrotated horizontal arc segment
        // exists anywhere in the crop, so PatchMatch must rotate shoulder patches (`±18°..±38°`).
        let (h, w) = (80, 120);
        let mut clean = synthetic_grain(h, w, 0.22);
        let (cx, cy, rad) = (60.0f32, 58.0f32, 26.0f32);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - cx).hypot(y as f32 - cy) - rad).abs();
                if d <= 2.2 {
                    let w_ring = ((2.2 - d) / 1.4).clamp(0.0, 1.0);
                    let i = 3 * (y * w + x);
                    for c in 0..3 {
                        clean[i + c] = clean[i + c] * (1.0 - w_ring) + 0.82 * w_ring;
                    }
                }
            }
        }
        let mut img = clean.clone();
        let mut path = Vec::new();
        for y in 22..=44 {
            path.push((60 - 12, y - 10));
            for dx in -3..=3 {
                let i = 3 * (y as usize * w + (60 + dx) as usize);
                img[i..i + 3].fill(0.04);
            }
        }
        heal_stroke(
            &mut img,
            (h, w),
            &HealStroke {
                rect: (12, 10, 108, 56),
                path: &path,
                brush: 5,
                avoid: None,
            },
        )
        .unwrap();

        // Apex `(60, 32)` must be reconstructed as a bright ring peak (`> 0.62`), while inside `(60, 40)`
        // and outside `(60, 24)` stay dark background (`< 0.34`).
        let apex = img[3 * (32 * w + 60)];
        let outside = img[3 * (24 * w + 60)];
        let inside = img[3 * (40 * w + 60)];
        assert!(apex > 0.60, "circle apex broken: apex={apex:.4}");
        assert!(
            outside < 0.34 && inside < 0.34,
            "ring smeared: out={outside:.4}, in={inside:.4}"
        );
    }

    #[test]
    fn reconstructs_tapered_perspective_stripe_via_scaled_patches() {
        let (h, w) = (120, 120);
        let mut clean = synthetic_grain(h, w, 0.20);
        for y in 0..h {
            let hw = 1.6 + 0.045 * y as f32;
            for x in 0..w {
                let d = (x as f32 - 60.0).abs();
                if d <= hw + 1.0 {
                    let t = ((hw + 1.0 - d) / 1.2).clamp(0.0, 1.0);
                    let i = 3 * (y * w + x);
                    for c in 0..3 {
                        clean[i + c] = clean[i + c] * (1.0 - t) + 0.80 * t;
                    }
                }
            }
        }
        let mut img = clean.clone();
        let mut path = Vec::new();
        for x in 40..=80 {
            path.push((x - 20, 56 - 20));
            for dy in -2..=2 {
                let i = 3 * ((56 + dy) as usize * w + x as usize);
                img[i..i + 3].fill(0.04);
            }
        }
        heal_stroke(
            &mut img,
            (h, w),
            &HealStroke {
                rect: (20, 20, 100, 96),
                path: &path,
                brush: 4,
                avoid: None,
            },
        )
        .unwrap();

        let center = img[3 * (56 * w + 60)];
        let flank = img[3 * (56 * w + 52)];
        assert!(
            center > 0.68 && flank < 0.30,
            "center={center:.4}, flank={flank:.4}"
        );
    }

    #[test]
    fn preserves_correlated_coarse_dye_cloud_clumps_without_confetti() {
        let (h, w) = (160, 160);
        // Synthesize spatially correlated ~6 px dye-cloud clumps (not 1x1 white noise).
        let raw = synthetic_grain(h, w, 0.32);
        let mut rgb = vec![[0.0f32; 3]; h * w];
        for p in 0..h * w {
            rgb[p] = [
                (raw[3 * p] - 0.32) * 5.5,
                (raw[3 * p + 1] - 0.32) * 5.5,
                (raw[3 * p + 2] - 0.32) * 5.5,
            ];
        }
        let clouds = smooth_box_rgb(&smooth3_rgb(&rgb, (h, w)), 2, (h, w));
        let fine = synthetic_grain(h, w, 0.0);
        let mut clean = vec![0.0f32; 3 * h * w];
        for p in 0..h * w {
            for c in 0..3 {
                let fn_v = (fine[3 * p + c] - 0.025) * 0.35;
                clean[3 * p + c] = (0.32 + clouds[p][c] + fn_v).clamp(0.05, 0.92);
            }
        }
        let mut img = clean.clone();
        let mut path = Vec::new();
        for y in 40..=120 {
            path.push((80 - 20, y - 20));
            for dx in -5..=5 {
                let i = 3 * (y as usize * w + (80 + dx) as usize);
                img[i..i + 3].fill(0.95);
            }
        }
        heal_stroke(
            &mut img,
            (h, w),
            &HealStroke {
                rect: (20, 10, 140, 150),
                path: &path,
                brush: 8,
                avoid: None,
            },
        )
        .unwrap();

        // Measure lag-4 RMS (multi-pixel clump amplitude) and lag-1 / lag-4 ratio inside healed strip vs clean.
        let band_stats = |buf: &[f32]| -> (f32, f32) {
            let mut s1 = 0.0f64;
            let mut n1 = 0usize;
            let mut s4 = 0.0f64;
            let mut n4 = 0usize;
            for y in 44..116 {
                for x in 74..85 {
                    for c in 0..3 {
                        let d1 = buf[3 * (y * w + x + 1) + c] - buf[3 * (y * w + x) + c];
                        s1 += (d1 * d1) as f64;
                        n1 += 1;
                    }
                }
                for x in 74..82 {
                    for c in 0..3 {
                        let d4 = buf[3 * (y * w + x + 4) + c] - buf[3 * (y * w + x) + c];
                        s4 += (d4 * d4) as f64;
                        n4 += 1;
                    }
                }
            }
            (
                (s1 / n1 as f64).sqrt() as f32,
                (s4 / n4 as f64).sqrt() as f32,
            )
        };
        let (c_l1, c_l4) = band_stats(&clean);
        let (h_l1, h_l4) = band_stats(&img);
        assert!(
            h_l4 > 0.75 * c_l4,
            "coarse dye-cloud clumps collapsed: healed_lag4={h_l4:.4}, clean_lag4={c_l4:.4}"
        );
        assert!(
            h_l1 / h_l4 < 1.25 * (c_l1 / c_l4),
            "clumpy grain turned into 1x1 confetti: healed_ratio={:.3}, clean_ratio={:.3}",
            h_l1 / h_l4,
            c_l1 / c_l4
        );
    }
}
