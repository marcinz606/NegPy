//! The print curve of `apply_characteristic_curve` (negpy/features/exposure/logic.py): a straight
//! line of slope `slope` through the exposure pivot, bounded above by the toe (shadows to paper
//! black d_max) and below by the shoulder (highlights to paper white), two independent softplus
//! bounds. Output is linear reflectance, 10^-D; the working-space OETF comes at the engine output.
//! The float64 order follows the Numba kernel it replaced, which fastmath regrouped.

// The steps index several per-pixel arrays of one run in step.
#![allow(clippy::needless_range_loop)]

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::{exp_f64, exp10_f64, log1p_f64, log10_f64, pow_f64, tanh_f64};

const LUMA_R: f64 = 0.2126;
const LUMA_G: f64 = 0.7152;
const LUMA_B: f64 = 0.0722;

/// An HxW (or HxWxK) float32 plane.
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [f32],
    pub w: usize,
}

/// Tone-limited masks: K shape alphas per pixel, (stops, ISO-R delta, e0, e1) per mask, and the
/// unlimited masks' ISO-R plane.
#[derive(Clone, Copy)]
pub struct Keys<'a> {
    pub alpha: &'a [f32],
    pub params: &'a [[f64; 4]],
    pub deltas: Plane<'a>,
}

#[derive(Clone, Copy)]
pub struct Diffusion<'a> {
    pub plane: &'a [f32],
    pub gh: usize,
    pub gw: usize,
    pub rect: [f64; 4],
    pub mix: f64,
    pub range: [f64; 3],
}

#[derive(Clone, Copy)]
pub struct Lut<'a> {
    pub table: &'a [f32],
    pub n: usize,
    pub e_max: [f32; 3],
    pub black: f64,
}

/// Every input but the image. Toe and shoulder are per channel (value plus layer trim), scaled
/// by toe_shoulder_strength. `d_min_rgb` is paper white per channel; `dye_mix` couples the dye
/// densities above it. `ev` is dodge and burn in print stops times `ev_scale` (positive burns),
/// `grade` a slope factor per pixel. `keys` are tone-limited masks whose ISO-R grade adds to
/// their deltas about `r0` and replaces `grade`. `flash` is preflash as a fraction of the
/// threshold exposure. `bpc` maps paper black (or the LUT's `black`) to display black. `lut`
/// replaces 10^-D on the density above base.
#[derive(Clone, Copy)]
pub struct PrintCurve<'a> {
    pub pivots: [f32; 3],
    pub slopes: [f32; 3],
    pub curvatures: [f32; 3],
    pub toe: [f64; 3],
    pub shoulder: [f64; 3],
    pub toe_width: [f64; 3],
    pub shoulder_width: [f64; 3],
    pub cmy_offsets: [f32; 3],
    pub shadow_cmy: [f32; 3],
    pub highlight_cmy: [f32; 3],
    pub d_min_rgb: [f64; 3],
    pub d_max: f64,
    pub a_toe_base: f64,
    pub a_sh_base: f64,
    pub width_ref: f64,
    pub toe_height: f64,
    pub sh_height: f64,
    pub zone_center: f64,
    pub shadow_density: f64,
    pub highlight_density: f64,
    pub shadow_grade: [f64; 3],
    pub highlight_grade: [f64; 3],
    pub zone_sh_center: f64,
    pub zone_hi_center: f64,
    pub zone_k: f64,
    pub v_star: f64,
    pub midtone_gamma: [f64; 3],
    pub gamma_width: f64,
    pub dye_mix: Option<[[f64; 3]; 3]>,
    /// Per-channel k, damping, reference spread.
    pub sep: Option<([f64; 3], f64, f64)>,
    pub ev: Option<Plane<'a>>,
    pub ev_scale: [f32; 3],
    pub grade: Option<Plane<'a>>,
    pub keys: Option<Keys<'a>>,
    pub r0: f64,
    pub r_min: f64,
    pub r_max: f64,
    pub flash: f64,
    pub flash_v: f64,
    pub flash_gamma: f64,
    pub flash_map: Option<Plane<'a>>,
    pub diffusion: Option<Diffusion<'a>>,
    pub lut: Option<Lut<'a>>,
    pub bpc: bool,
    pub black_hold: f64,
}

/// The per-channel bounds the kernel's prologue derives.
#[derive(Clone, Copy)]
struct Bounds {
    a_hl: [f64; 3],
    a_sh: [f64; 3],
    d_min_eff: [f64; 3],
    d_max_eff: [f64; 3],
    bpc_black: [f64; 3],
    use_split: bool,
}

fn bounds(p: &PrintCurve) -> Bounds {
    let eps = 1e-6;
    let mut b = Bounds {
        a_hl: [0.0; 3],
        a_sh: [0.0; 3],
        d_min_eff: [0.0; 3],
        d_max_eff: [0.0; 3],
        bpc_black: [0.0; 3],
        use_split: p
            .shadow_grade
            .iter()
            .chain(&p.highlight_grade)
            .any(|&g| g != 0.0),
    };
    for ch in 0..3 {
        b.a_hl[ch] = p.a_sh_base * p.width_ref / p.shoulder_width[ch].max(eps);
        let a_sh_w = p.a_toe_base * p.width_ref / p.toe_width[ch].max(eps);
        let t = p.toe[ch];
        // A negative toe tightens the shadow roll-off instead of moving d_max_eff past paper black.
        let d_max_base = if t >= 0.0 {
            b.a_sh[ch] = a_sh_w;
            p.d_max - t * p.toe_height
        } else {
            b.a_sh[ch] = a_sh_w * (1.0 - t * 4.0);
            p.d_max
        };
        let dmn = (p.d_min_rgb[ch] + p.shoulder[ch] * p.sh_height).max(0.0);
        b.d_min_eff[ch] = dmn;
        b.d_max_eff[ch] = if d_max_base < dmn + 0.1 {
            dmn + 0.1
        } else {
            d_max_base
        };
        // BPC references the physical d_max, so toe lifts survive; a negative toe raises the clip.
        let db = if t < 0.0 {
            p.d_max + t * p.toe_height
        } else {
            p.d_max
        };
        b.bpc_black[ch] = exp10_f64(-db);
    }
    b
}

/// `_fast_sigmoid`: 1 / (1 + e^-x) above 0, e^x / (1 + e^x) below, both from e^-|x| so the
/// loop has no branch.
#[inline(always)]
fn sigmoid(x: f64) -> f64 {
    let z = exp_f64(-x.abs());
    let d = 1.0 + z;
    if x >= 0.0 { 1.0 / d } else { z / d }
}

/// `_softplus`: x + log1p(e^-x) above 0, log1p(e^x) below: max(x, 0) + log1p(e^-|x|).
#[inline(always)]
fn softplus(x: f64) -> f64 {
    let l = log1p_f64(exp_f64(-x.abs()));
    if x > 0.0 { x + l } else { l }
}

#[inline(always)]
fn smoothstep01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `quadratic_core`: k (u - pivot) + curv u^2, held at its vertex where it would fold back.
#[inline(always)]
fn quadratic_core(k: f64, pivot: f64, curv: f64, mut u: f64) -> f64 {
    if curv != 0.0 {
        let vertex = -k / (2.0 * curv);
        if (curv > 0.0 && u < vertex) || (curv < 0.0 && u > vertex) {
            u = vertex;
        }
    }
    k * (u - pivot) + curv * u * u
}

/// `separation_damping_gain`.
#[inline(always)]
fn separation_gain(k: f64, damping: f64, chroma: f64, reference: f64) -> f64 {
    if k <= 0.0 {
        return 0.0;
    }
    let h = (reference - chroma) / (reference + chroma);
    pow_f64(k, (1.0 - damping) + damping * h).min(3.0)
}

/// `plane_taps`: OpenCV's half-pixel bilinear taps on a gh x gw plane covering `rect`.
#[inline(always)]
fn plane_taps(
    gh: usize,
    gw: usize,
    x: usize,
    y: usize,
    r: [f64; 4],
) -> (usize, usize, usize, usize, f64, f64) {
    let px = (x as f64 + 0.5 - r[0]) * gw as f64 / r[2] - 0.5;
    let py = (y as f64 + 0.5 - r[1]) * gh as f64 / r[3] - 0.5;
    let lx = px.floor().clamp(0.0, gw as f64 - 1.0);
    let ly = py.floor().clamp(0.0, gh as f64 - 1.0);
    let hx = (lx + 1.0).min(gw as f64 - 1.0);
    let hy = (ly + 1.0).min(gh as f64 - 1.0);
    let fx = (px - lx).clamp(0.0, 1.0);
    let fy = (py - ly).clamp(0.0, 1.0);
    (lx as usize, hx as usize, ly as usize, hy as usize, fx, fy)
}

/// `tetra_lookup`: tetrahedral interpolation clamped to the table; corner differences in float32
/// as Numba computes them on the float32 table.
#[inline(always)]
pub fn tetra(l: &Lut, e: [f64; 3]) -> [f64; 3] {
    let n = l.n - 1;
    let pos: [f64; 3] =
        std::array::from_fn(|c| (e[c] / l.e_max[c] as f64).clamp(0.0, 1.0) * n as f64);
    let idx: [usize; 3] = std::array::from_fn(|c| (pos[c] as usize).min(n - 1));
    let [fx, fy, fz]: [f64; 3] = std::array::from_fn(|c| pos[c] - idx[c] as f64);
    let at = |di: usize, dj: usize, dk: usize, c: usize| {
        l.table[(((idx[0] + di) * l.n + idx[1] + dj) * l.n + idx[2] + dk) * 3 + c]
    };
    std::array::from_fn(|c| {
        let c000 = at(0, 0, 0, c);
        let c111 = at(1, 1, 1, c);
        let d = |a: f32, b: f32| (b - a) as f64;
        if fx >= fy {
            if fy >= fz {
                let (c100, c110) = (at(1, 0, 0, c), at(1, 1, 0, c));
                c000 as f64 + fx * d(c000, c100) + fy * d(c100, c110) + fz * d(c110, c111)
            } else if fx >= fz {
                let (c100, c101) = (at(1, 0, 0, c), at(1, 0, 1, c));
                c000 as f64 + fx * d(c000, c100) + fz * d(c100, c101) + fy * d(c101, c111)
            } else {
                let (c001, c101) = (at(0, 0, 1, c), at(1, 0, 1, c));
                c000 as f64 + fz * d(c000, c001) + fx * d(c001, c101) + fy * d(c101, c111)
            }
        } else if fz >= fy {
            let (c001, c011) = (at(0, 0, 1, c), at(0, 1, 1, c));
            c000 as f64 + fz * d(c000, c001) + fy * d(c001, c011) + fx * d(c011, c111)
        } else if fz >= fx {
            let (c010, c011) = (at(0, 1, 0, c), at(0, 1, 1, c));
            c000 as f64 + fy * d(c000, c010) + fz * d(c010, c011) + fx * d(c011, c111)
        } else {
            let (c010, c110) = (at(0, 1, 0, c), at(1, 1, 0, c));
            c000 as f64 + fy * d(c000, c010) + fx * d(c010, c110) + fz * d(c110, c111)
        }
    })
}

/// The print curve over an HxWx3 float32 image.
pub fn apply_print_curve(img: &[f32], w: usize, p: &PrintCurve, dst: &mut [f32]) {
    let b = bounds(p);
    dst.par_chunks_mut(w * 3)
        .zip(img.par_chunks(w * 3))
        .enumerate()
        .for_each(|(y, (d, s))| {
            for (i, (d, s)) in d.chunks_mut(3 * RUN).zip(s.chunks(3 * RUN)).enumerate() {
                run(s, d, (y, i * RUN, w), p, &b);
            }
        });
}

/// Pixels per pass: each step of the curve runs as a flat loop over a run, so it vectorizes.
const RUN: usize = 256;

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn run(
    src: &[f32],
    dst: &mut [f32],
    (y, x0, w): (usize, usize, usize),
    p: &PrintCurve,
    b: &Bounds,
) {
    let px = src.as_chunks::<3>().0;
    let n = px.len();
    let at = |plane: &Plane, i: usize| plane.data[y * plane.w + x0 + i];
    let mut gfac = [1f64; RUN];
    let mut ev_px = [0f64; RUN];
    if let Some(k) = &p.keys {
        let m = k.params.len();
        for i in 0..n {
            if let Some(ev) = &p.ev {
                ev_px[i] = at(ev, i) as f64;
            }
            // The unburned tone: no mask can move the pixels it selects.
            let q = px[i];
            let lum = LUMA_R * q[0] as f64 + LUMA_G * q[1] as f64 + LUMA_B * q[2] as f64;
            let mut dr = at(&k.deltas, i) as f64;
            for (j, kp) in k.params.iter().enumerate() {
                let a = k.alpha[(y * w + x0 + i) * m + j] as f64
                    * smoothstep01((lum - kp[2]) / (kp[3] - kp[2]));
                ev_px[i] += kp[0] * a;
                dr += kp[1] * a;
            }
            gfac[i] = p.r0 / (p.r0 + dr).clamp(p.r_min, p.r_max);
        }
    } else if let Some(g) = &p.grade {
        for i in 0..n {
            gfac[i] = at(g, i) as f64;
        }
    }
    let mut fl = [p.flash; RUN];
    if let Some(m) = &p.flash_map {
        for i in 0..n {
            fl[i] += at(m, i) as f64;
        }
    }
    let any_flash = fl[..n].iter().any(|&f| f > 0.0);
    let mut dens = [[0f64; RUN]; 3];
    for ch in 0..3 {
        let v = &mut dens[ch];
        for i in 0..n {
            v[i] = px[i][ch] as f64;
        }
        if let Some(d) = &p.diffusion {
            // Diffusion mixes light, not density: (1-d) I + d blur(I), I = 10^(val r).
            let r = d.range[ch];
            for i in 0..n {
                let (i0x, i1x, i0y, i1y, fx, fy) = plane_taps(d.gh, d.gw, x0 + i, y, d.rect);
                let tap = |yy: usize, xx: usize| d.plane[(yy * d.gw + xx) * 3 + ch] as f64;
                let top = tap(i0y, i0x) * (1.0 - fx) + tap(i0y, i1x) * fx;
                let bot = tap(i1y, i0x) * (1.0 - fx) + tap(i1y, i1x) * fx;
                let blur = top * (1.0 - fy) + bot * fy;
                v[i] = log10_f64((1.0 - d.mix) * exp10_f64(v[i] * r) + d.mix * exp10_f64(blur * r))
                    / r;
            }
        }
        let offset = p.cmy_offsets[ch] as f64;
        let scale = p.ev_scale[ch];
        if p.keys.is_some() {
            for i in 0..n {
                v[i] = v[i] + offset + (ev_px[i] * scale as f64) as f32 as f64;
            }
        } else if let Some(ev) = &p.ev {
            for i in 0..n {
                v[i] = v[i] + offset + (at(ev, i) * scale) as f64;
            }
        } else {
            for x in v[..n].iter_mut() {
                *x += offset;
            }
        }
        let (slope, pivot, curv) = (
            p.slopes[ch] as f64,
            p.pivots[ch] as f64,
            p.curvatures[ch] as f64,
        );
        for i in 0..n {
            v[i] = quadratic_core(slope * gfac[i], pivot, curv, v[i]);
        }
        if any_flash {
            for i in 0..n {
                let lifted = p.flash_v
                    + p.flash_gamma
                        * log10_f64(exp10_f64((v[i] - p.flash_v) / p.flash_gamma) + fl[i]);
                v[i] = if fl[i] > 0.0 { lifted } else { v[i] };
            }
        }
        let mg = p.midtone_gamma[ch];
        if mg != 0.0 {
            for x in v[..n].iter_mut() {
                *x += mg * p.gamma_width * tanh_f64((*x - p.v_star) / p.gamma_width);
            }
        }
        let (s_cmy, h_cmy) = (p.shadow_cmy[ch] as f64, p.highlight_cmy[ch] as f64);
        for x in v[..n].iter_mut() {
            let w_sh = sigmoid(3.0 * (*x - p.zone_center));
            *x += s_cmy * w_sh + h_cmy * (1.0 - w_sh);
        }
        if b.use_split {
            let (sg, hg) = (p.shadow_grade[ch], p.highlight_grade[ch]);
            for x in v[..n].iter_mut() {
                let w_gsh = sigmoid(p.zone_k * (*x - p.zone_sh_center));
                let w_ghi = 1.0 - sigmoid(p.zone_k * (*x - p.zone_hi_center));
                *x = *x
                    + sg * w_gsh * (*x - p.zone_sh_center)
                    + hg * w_ghi * (*x - p.zone_hi_center);
            }
        }
        if p.shadow_density != 0.0 || p.highlight_density != 0.0 {
            let hold = p.black_hold > 0.0 && p.shadow_density < 0.0;
            let dmx = b.d_max_eff[ch];
            for x in v[..n].iter_mut() {
                let mut w_zsh = sigmoid(p.zone_k * (*x - p.zone_sh_center));
                if hold {
                    w_zsh *= smoothstep01((dmx - *x) / p.black_hold);
                }
                let w_zhi = 1.0 - sigmoid(p.zone_k * (*x - p.zone_hi_center));
                *x += p.shadow_density * w_zsh + p.highlight_density * w_zhi;
            }
        }
        let (dmn, dmx, a_hl, a_sh) = (b.d_min_eff[ch], b.d_max_eff[ch], b.a_hl[ch], b.a_sh[ch]);
        for x in v[..n].iter_mut() {
            let v1 = dmn + softplus(a_hl * (*x - dmn)) / a_hl;
            *x = dmx - softplus(a_sh * (dmx - v1)) / a_sh;
        }
    }
    let dmin = p.d_min_rgb;
    if let Some(m) = &p.dye_mix {
        for i in 0..n {
            let e = [
                dens[0][i] - dmin[0],
                dens[1][i] - dmin[1],
                dens[2][i] - dmin[2],
            ];
            for c in 0..3 {
                dens[c][i] = dmin[c] + m[c][0] * e[0] + m[c][1] * e[1] + m[c][2] * e[2];
            }
        }
    }
    if let Some((k3, damping, reference)) = &p.sep {
        for i in 0..n {
            let s = [
                dens[0][i] - dmin[0],
                dens[1][i] - dmin[1],
                dens[2][i] - dmin[2],
            ];
            let mean = (s[0] + s[1] + s[2]) / 3.0;
            let chroma = (((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[0] - s[2]).powi(2))
                / 3.0)
                .sqrt();
            for c in 0..3 {
                dens[c][i] = dmin[c]
                    + mean
                    + separation_gain(k3[c], *damping, chroma, *reference) * (s[c] - mean);
            }
        }
    }
    let out = dst.as_chunks_mut::<3>().0;
    if let Some(l) = &p.lut {
        let base: [f64; 3] = std::array::from_fn(|c| exp10_f64(-dmin[c]));
        for i in 0..n {
            let rgb = tetra(
                l,
                [
                    dens[0][i] - dmin[0],
                    dens[1][i] - dmin[1],
                    dens[2][i] - dmin[2],
                ],
            );
            out[i] = std::array::from_fn(|c| {
                let mut t = base[c] * rgb[c];
                if p.bpc {
                    t = (t - l.black) / (1.0 - l.black);
                }
                t.clamp(0.0, 1.0) as f32
            });
        }
    } else {
        for c in 0..3 {
            let (black, bpc) = (b.bpc_black[c], p.bpc);
            for i in 0..n {
                let t = exp10_f64(-dens[c][i]);
                let t = if bpc { (t - black) / (1.0 - black) } else { t };
                out[i][c] = t.clamp(0.0, 1.0) as f32;
            }
        }
    }
}
