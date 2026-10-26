//! The Lab-stage kernels of negpy/features/lab/logic.py, in the float32 order NumPy evaluates
//! them, around the bit-exact OpenCV filters of `filter` and `morph`.

use rayon::prelude::*;

use crate::color::{self, LabSpace};
use crate::filter::{self, Border};

// Sharpen constants; shaders/lab.wgsl mirrors them. The gate separates grain from detail in
// L*, sized against |L - blur| at a 1 px radius, which tops out near 1.0.
const GATE_LO: f32 = 0.25;
const GATE_HI: f64 = 0.33;
// L*-domain USM exaggerates light halos, so overshoot above the local max is clamped tighter
// than undershoot below the local min.
const OVERSHOOT_LIGHT: f32 = 1.0;
const OVERSHOOT_DARK: f32 = 2.0;
const MASK_T_HI: f64 = 10.0;
const SHADOW_FLOOR: f64 = 1.0 / 3.0;
const SHADOW_L_HI: f32 = 35.0;

/// `_smoothstep(e0, e1, x)`: the edges are Python floats, so e1 - e0 is a float64 difference
/// rounded to float32 once.
#[inline(always)]
fn smoothstep(e0: f64, e1: f64, x: f32) -> f32 {
    let t = clip((x - e0 as f32) / (e1 - e0) as f32, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `sharpen_shadow_gain(l)`.
#[inline(always)]
fn shadow_gain(l: f32) -> f32 {
    let f = SHADOW_FLOOR as f32;
    f + (1.0 - f) * smoothstep(0.0, SHADOW_L_HI as f64, l)
}

/// The edge mask's gradient: central differences on the edge-padded plane, `np.hypot` (glibc's
/// float64 formula), then a 3x3 replicate-border box mean.
fn box_gradient(l: &[f32], (h, w): (usize, usize)) -> Option<Vec<f32>> {
    let mut hyp = vec![0f32; h * w];
    hyp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let (up, down) = (y.saturating_sub(1), (y + 1).min(h - 1));
        let row = &l[y * w..][..w];
        for (x, o) in out.iter_mut().enumerate() {
            let gx = (row[(x + 1).min(w - 1)] - row[x.saturating_sub(1)]) * 0.5;
            let gy = (l[down * w + x] - l[up * w + x]) * 0.5;
            let (gx, gy) = (gx as f64, gy as f64);
            *o = (gx * gx + gy * gy).sqrt() as f32;
        }
    });
    let mut grad = vec![0f32; h * w];
    filter::box_filter(&hyp, &mut grad, (h, w, 1), (3, 3), true, Border::Replicate)?;
    Some(grad)
}

/// `_edge_mask(l, masking)`: the smoothstep of `box_gradient` over 0.5t..t.
fn edge_mask(l: &[f32], (h, w): (usize, usize), masking: f64) -> Option<Vec<f32>> {
    let mut grad = box_gradient(l, (h, w))?;
    let t = MASK_T_HI * masking;
    grad.par_iter_mut()
        .for_each(|g| *g = smoothstep(0.5 * t, t, *g));
    Some(grad)
}

// Auto Masking meter; shaders/sharpen_grain_hist.wgsl and sharpen_grain_pick.wgsl mirror it.
// Bins are 1/GRAIN_BIN_SCALE L* per pixel wide, so the range tops out at 4 L*/px.
pub const GRAIN_BINS: usize = 1024;
const GRAIN_BIN_SCALE: f32 = 256.0;
const GRAIN_L_LO: f32 = 30.0;
const GRAIN_L_HI: f32 = 85.0;

/// The grain the edge mask has to clear: the 30th percentile of `box_gradient` over the midtone
/// pixels (L* of linear luminance `lum`), as its bin centre. Flat midtones hold mostly grain, and
/// the percentile stays below their edges. None without midtones or below the filter's size.
pub fn sharpen_grain(src: &[f32], (h, w): (usize, usize), lum: [f32; 3]) -> Option<f32> {
    let l: Vec<f32> = src
        .par_chunks(3)
        .map(|q| lab_l_from_y(q[0] * lum[0] + q[1] * lum[1] + q[2] * lum[2]))
        .collect();
    let grad = box_gradient(&l, (h, w))?;
    let hist = grad
        .par_chunks(1 << 16)
        .zip(l.par_chunks(1 << 16))
        .map(|(g, l)| {
            let mut bins = [0u32; GRAIN_BINS];
            for (&g, &l) in g.iter().zip(l) {
                if l > GRAIN_L_LO && l < GRAIN_L_HI {
                    bins[((g * GRAIN_BIN_SCALE) as usize).min(GRAIN_BINS - 1)] += 1;
                }
            }
            bins
        })
        .reduce(
            || [0u32; GRAIN_BINS],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                a
            },
        );
    let total: u64 = hist.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return None;
    }
    let rank = total * 3 / 10;
    let mut cum = 0u64;
    let bin = hist.iter().position(|&c| {
        cum += c as u64;
        cum > rank
    })?;
    Some((bin as f32 + 0.5) / GRAIN_BIN_SCALE)
}

/// `_usm_sharpening_block` of an HxWx3 linear image: the L* channel's soft-gated unsharp mask
/// with `k` as the blur taps, clamped to the local 3x3 range, then clipped to 0..1 in RGB.
/// `gain` is `np.float32(amount * 2.5)`. None where a filter does not cover the shape.
pub fn usm_sharpen(
    src: &[f32],
    (h, w): (usize, usize),
    (gain, masking): (f32, f64),
    k: &[f32],
    (to_lab, to_rgb): (LabSpace, LabSpace),
    dst: &mut [f32],
) -> Option<()> {
    let n = h * w;
    let mut lab = vec![0f32; n * 3];
    color::rgb_to_lab(src, &mut lab, to_lab);
    let l: Vec<f32> = lab.par_chunks(3).map(|p| p[0]).collect();
    let mut blur = vec![0f32; n];
    filter::sep_filter2d(&l, &mut blur, (h, w, 1), (k, k), Border::Reflect101)?;
    let edge = if masking > 0.0 {
        Some(edge_mask(&l, (h, w), masking)?)
    } else {
        None
    };
    // The 3x3 erode and dilate (replicated edges) run inline, a row at a time.
    lab.par_chunks_mut(w * 3).enumerate().for_each(|(y, out)| {
        let rows = [y.saturating_sub(1), y, (y + 1).min(h - 1)].map(|r| &l[r * w..][..w]);
        let col = |x: usize, f: fn(f32, f32) -> f32| f(f(rows[0][x], rows[1][x]), rows[2][x]);
        let (mut lo_col, mut hi_col) = (vec![0f32; w], vec![0f32; w]);
        for x in 0..w {
            lo_col[x] = col(x, f32_lo);
            hi_col[x] = col(x, f32_hi);
        }
        for (x, p) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let lo = f32_lo(f32_lo(lo_col[xl], lo_col[x]), lo_col[xr]);
            let hi = f32_hi(f32_hi(hi_col[xl], hi_col[x]), hi_col[xr]);
            let i = y * w + x;
            let (l0, d) = (l[i], l[i] - blur[i]);
            let mut g = gain * smoothstep(GATE_LO as f64, GATE_HI, d.abs()) * shadow_gain(l0);
            if let Some(e) = &edge {
                g *= e[i];
            }
            let v = clip(l0 + d * g, lo - OVERSHOOT_DARK, hi + OVERSHOOT_LIGHT);
            p[0] = clip(v, 0.0, 100.0);
        }
    });
    color::lab_to_rgb(&lab, dst, to_rgb);
    dst.par_iter_mut().for_each(|v| *v = clip(*v, 0.0, 1.0));
    Some(())
}

/// `cv2.erode`'s float32 minimum.
fn f32_lo(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

/// `cv2.dilate`'s float32 maximum.
fn f32_hi(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// `np.clip(v, lo, hi)`, NumPy's `min(max(v, lo), hi)` with `max(a, b) = a > b ? a : b`
/// unless a is NaN: a NaN in any argument comes through, and clip(-0.0, 0.0, ..) is +0.0.
#[inline(always)]
fn clip(v: f32, lo: f32, hi: f32) -> f32 {
    let m = if v.is_nan() || v > lo { v } else { lo };
    if m.is_nan() || m < hi { m } else { hi }
}

// Chroma Denoise edge test: the low band's gradient times sigma, in CIELAB a*/b* units. Chroma
// above the cutoff is kept from LO, fully from HI. lab.wgsl mirrors them as literals.
const CHROMA_EDGE_LO: f64 = 1.0;
const CHROMA_EDGE_HI: f64 = 2.5;

/// `apply_chroma_denoise` once sigma is known: the a*/b* band above the Gaussian low pass `k`
/// (reflect-101 edges) is kept only where the low band has a color edge (`chroma_edge_keep`).
pub fn chroma_denoise(
    src: &[f32],
    (h, w): (usize, usize),
    sigma: f64,
    k: &[f32],
    (to_lab, to_rgb): (LabSpace, LabSpace),
    dst: &mut [f32],
) -> Option<()> {
    let n = h * w;
    let mut lab = vec![0f32; n * 3];
    color::rgb_to_lab(src, &mut lab, to_lab);
    let ab: Vec<f32> = lab.par_chunks(3).flat_map_iter(|p| [p[1], p[2]]).collect();
    let mut low = vec![0f32; n * 2];
    filter::sep_filter2d(&ab, &mut low, (h, w, 2), (k, k), Border::Reflect101)?;
    let span = CHROMA_EDGE_HI - CHROMA_EDGE_LO;
    let (scale, shift) = ((sigma / span) as f32, (CHROMA_EDGE_LO / span) as f32);
    lab.par_chunks_mut(w * 3).enumerate().for_each(|(y, out)| {
        let (up, down) = (y.saturating_sub(1), (y + 1).min(h - 1));
        let at = |r: usize, x: usize| &low[(r * w + x) * 2..][..2];
        for (x, p) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (left, right) = (at(y, x.saturating_sub(1)), at(y, (x + 1).min(w - 1)));
            let (top, bottom) = (at(up, x), at(down, x));
            // cv2.filter2D drops the zero tap: -0.5 a + 0.5 c rounds once, as (c - a) * 0.5 does.
            let sq: [f32; 2] = std::array::from_fn(|c| {
                let gx = (right[c] - left[c]) * 0.5;
                let gy = (bottom[c] - top[c]) * 0.5;
                let s = gx * gx;
                s + gy * gy
            });
            let mut g = (sq[0] + sq[1]).sqrt();
            g *= scale;
            g -= shift;
            let t = clip(g, 0.0, 1.0);
            let keep = t * t * (3.0 - 2.0 * t);
            let i = y * w + x;
            for c in 0..2 {
                let lo = low[i * 2 + c];
                let mut v = ab[i * 2 + c] - lo;
                v *= keep;
                p[1 + c] = v + lo;
            }
        }
    });
    color::lab_to_rgb(&lab, dst, to_rgb);
    dst.par_iter_mut().for_each(|v| *v = clip(*v, 0.0, 1.0));
    Some(())
}

/// One Gaussian bloom of `apply_glow_and_halation`: the highlights above `threshold` (a Python
/// float) of luma, blurred by (ksize, sigma) and added at `amount`.
pub struct Bloom {
    pub amount: f32,
    pub ksize: usize,
    pub sigma: f64,
    pub threshold: f64,
}

/// `apply_glow_and_halation`: glow masks highlights on the display-encoded luma (`inv_gamma`
/// TRC), halation the red channel on linear luma, each blurred and added, then clipped. `lum`
/// is the luminance row.
pub fn glow_and_halation(
    src: &[f32],
    (h, w): (usize, usize),
    (glow, halation): (Option<Bloom>, Option<Bloom>),
    (lum, inv_gamma): ([f32; 3], f32),
    dst: &mut [f32],
) -> Option<()> {
    let n = h * w;
    dst.copy_from_slice(src);
    let luma = |p: &[f32]| p[0] * lum[0] + p[1] * lum[1] + p[2] * lum[2];
    let ramp = |v: f32, t: f64| clip((v - t as f32) / (1.0 - t) as f32, 0.0, 1.0);
    let mut light = vec![0f32; n * 3];
    let mut blur = vec![0f32; n * 3];
    let mut add = |light: &[f32], b: &Bloom, dst: &mut [f32]| -> Option<()> {
        filter::gaussian_blur(
            light,
            &mut blur,
            (h, w, 3),
            (b.ksize, b.ksize),
            (b.sigma, 0.0),
            Border::Reflect101,
        )?;
        dst.par_iter_mut()
            .zip(blur.par_iter())
            .for_each(|(d, &v)| *d += v * b.amount);
        Some(())
    };
    if let Some(b) = &glow {
        let mut enc = vec![0f32; n * 3];
        color::oetf_encode(src, &mut enc, inv_gamma);
        light
            .par_chunks_mut(3)
            .zip(src.par_chunks(3))
            .zip(enc.par_chunks(3))
            .for_each(|((o, p), e)| {
                let m = ramp(luma(e), b.threshold);
                o.copy_from_slice(&[p[0] * m, p[1] * m, p[2] * m]);
            });
        add(&light, b, dst)?;
    }
    if let Some(b) = &halation {
        light
            .par_chunks_mut(3)
            .zip(src.par_chunks(3))
            .for_each(|(o, p)| {
                let r = p[0] * ramp(luma(p), b.threshold);
                o.copy_from_slice(&[r, r * 0.3, r * 0.05]);
            });
        add(&light, b, dst)?;
    }
    dst.par_iter_mut().for_each(|v| *v = clip(*v, 0.0, 1.0));
    Some(())
}

const CLAHE_GRID: usize = 8;
const CLAHE_BINS: usize = 256;

/// `_clahe_cdfs`: per tile of the 8x8 grid, the clipped CDF of the bins. The limit is the
/// float32 product truncated, at least 1; the excess spreads evenly, the remainder to the
/// first bins.
fn clahe_cdfs(bins: &[u8], (h, w): (usize, usize), clip_limit: f32) -> Vec<[f32; CLAHE_BINS]> {
    let (tsy, tsx) = (h.div_ceil(CLAHE_GRID), w.div_ceil(CLAHE_GRID));
    let tiles = CLAHE_GRID * CLAHE_GRID;
    // One chunk of rows per thread: rayon's fold would make a histogram per split job.
    let chunk = h.div_ceil(rayon::current_num_threads()).max(1);
    let hist = bins
        .par_chunks(chunk * w)
        .enumerate()
        .map(|(ci, block)| {
            let mut acc = vec![0i64; tiles * CLAHE_BINS];
            for (dy, row) in block.chunks(w).enumerate() {
                let base = ((ci * chunk + dy) / tsy) * CLAHE_GRID;
                for (x, &b) in row.iter().enumerate() {
                    acc[(base + x / tsx) * CLAHE_BINS + b as usize] += 1;
                }
            }
            acc
        })
        .reduce(
            || vec![0i64; tiles * CLAHE_BINS],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                a
            },
        );
    hist.as_chunks::<CLAHE_BINS>()
        .0
        .iter()
        .map(|t| {
            let total: i64 = t.iter().sum();
            let limit = ((clip_limit * total as f32 / CLAHE_BINS as f32) as i64).max(1);
            let excess: i64 = t.iter().map(|&c| c - c.min(limit)).sum();
            let (inc, rem) = (excess / CLAHE_BINS as i64, excess % CLAHE_BINS as i64);
            let denom = total.max(1) as f32;
            let mut run = 0i64;
            std::array::from_fn(|b| {
                run += t[b].min(limit) + inc + ((b as i64) < rem) as i64;
                run as f32 / denom
            })
        })
        .collect()
}

/// `_clahe_axis(n)`: per position the two tile indices and the smoothstep fraction, from
/// tile_pos = pos / n * 8 - 0.5 in float32.
fn clahe_axis(n: usize) -> Vec<(usize, usize, f32)> {
    (0..n)
        .map(|i| {
            let tp = i as f32 / n as f32 * CLAHE_GRID as f32 - 0.5;
            let tf = tp.floor();
            let rf = tp - tf;
            let t = tf as i32;
            let (lo, hi) = (
                t.max(0) as usize,
                (t + 1).min(CLAHE_GRID as i32 - 1) as usize,
            );
            (lo, hi, rf * rf * (3.0 - 2.0 * rf))
        })
        .collect()
}

/// `apply_clahe(img, strength)` for strength > 0: the L* channel equalized per tile of a
/// fixed 8x8 grid, blended between tiles, and mixed back by `strength`; clahe_*.wgsl mirror it.
pub fn clahe(
    src: &[f32],
    (h, w): (usize, usize),
    strength: f64,
    (to_lab, to_rgb): (LabSpace, LabSpace),
    dst: &mut [f32],
) {
    let n = h * w;
    let mut lab = vec![0f32; n * 3];
    color::rgb_to_lab(src, &mut lab, to_lab);
    // np.clip(...).astype(int32) truncates; a NaN lands in bin 0 here.
    let bins: Vec<u8> = lab
        .par_chunks(3)
        .map(|p| clip(p[0] / 100.0 * 255.0, 0.0, 255.0) as u8)
        .collect();
    let cdfs = clahe_cdfs(&bins, (h, w), (strength * 2.5) as f32);
    let (ys, xs) = (clahe_axis(h), clahe_axis(w));
    let s = strength as f32;
    lab.par_chunks_mut(w * 3).enumerate().for_each(|(y, out)| {
        let (y0, y1, fy) = ys[y];
        for (x, p) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (x0, x1, fx) = xs[x];
            let b = bins[y * w + x] as usize;
            let v = |ty: usize, tx: usize| cdfs[ty * CLAHE_GRID + tx][b];
            let (v00, v10, v01, v11) = (v(y0, x0), v(y0, x1), v(y1, x0), v(y1, x1));
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            let cdf = top + (bot - top) * fy;
            let l = p[0];
            p[0] = l + (cdf * 100.0 - l) * s;
        }
    });
    color::lab_to_rgb(&lab, dst, to_rgb);
    dst.par_iter_mut().for_each(|v| *v = clip(*v, 0.0, 1.0));
}

/// The scalars of `apply_rl_sharpening`: `amount` as float32, the edge `masking`, the
/// iteration count, the luminance row and the noise floor `eps`.
pub struct Rl {
    pub amount: f32,
    pub masking: f64,
    pub iterations: usize,
    pub lum: [f32; 3],
    pub eps: f32,
}

/// `_lab_l_from_y`: CIELAB L* of linear luminance (D65, Yn = 1), the cube root correctly
/// rounded where NumPy runs SVML's.
#[inline(always)]
fn lab_l_from_y(y: f32) -> f32 {
    let y = if y.is_nan() || y > 0.0 { y } else { 0.0 };
    let f = if y > 0.008856 {
        crate::mathf::cbrtf(y)
    } else {
        7.787 * y + (16.0f64 / 116.0) as f32
    };
    116.0 * f - 16.0
}

/// `apply_rl_sharpening` for amount > 0: Richardson-Lucy deconvolution of linear Y with the
/// Gaussian taps `k` (reflect-101 edges), applied to RGB as the ratio est / obs, gated by the
/// shadow gain and the optional edge mask, then clipped.
pub fn rl_sharpen(
    src: &[f32],
    (h, w): (usize, usize),
    p: &Rl,
    k: &[f32],
    dst: &mut [f32],
) -> Option<()> {
    let n = h * w;
    let lum = p.lum;
    let pos = |v: f32| if v.is_nan() || v > 0.0 { v } else { 0.0 };
    let obs: Vec<f32> = src
        .par_chunks(3)
        .map(|q| pos(q[0]) * lum[0] + pos(q[1]) * lum[1] + pos(q[2]) * lum[2])
        .collect();
    let floor = |v: f32| if v.is_nan() || v > p.eps { v } else { p.eps };
    let mut est = obs.clone();
    let (mut blurred, mut ratio, mut corr) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let blur = |s: &[f32], d: &mut [f32]| {
        filter::sep_filter2d(s, d, (h, w, 1), (k, k), Border::Reflect101)
    };
    for _ in 0..p.iterations {
        blur(&est, &mut blurred)?;
        ratio
            .par_iter_mut()
            .zip(obs.par_iter().zip(blurred.par_iter()))
            .for_each(|(r, (&o, &b))| *r = o / floor(b));
        blur(&ratio, &mut corr)?;
        est.par_iter_mut()
            .zip(corr.par_iter())
            .for_each(|(e, &c)| *e *= c);
    }
    let l_obs: Vec<f32> = obs.par_iter().map(|&y| lab_l_from_y(y)).collect();
    let edge = if p.masking > 0.0 {
        Some(edge_mask(&l_obs, (h, w), p.masking)?)
    } else {
        None
    };
    dst.par_chunks_mut(3)
        .zip(src.par_chunks(3))
        .enumerate()
        .for_each(|(i, (d, q))| {
            let r = est[i] / floor(obs[i]);
            let mut g = p.amount * shadow_gain(l_obs[i]);
            if let Some(e) = &edge {
                g *= e[i];
            }
            let f = 1.0 + (r - 1.0) * g;
            let f = if f.is_nan() || f > 0.0 { f } else { 0.0 };
            for c in 0..3 {
                d[c] = clip(q[c] * f, 0.0, 1.0);
            }
        });
    Some(())
}

/// `apply_split_toning`: each tint is (cos, sin) of its hue times 20 times its strength, as
/// float64; NumPy adds tint * mask to a* and b* in float64 and rounds once. The masks are the
/// float32 smoothsteps of L* below and above 50.
pub fn split_toning(
    src: &[f32],
    (shadow, highlight): (Option<[f64; 2]>, Option<[f64; 2]>),
    (to_lab, to_rgb): (LabSpace, LabSpace),
    dst: &mut [f32],
) {
    let mut lab = vec![0f32; src.len()];
    color::rgb_to_lab(src, &mut lab, to_lab);
    let step = |t: f32| {
        let t = clip(t, 0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    lab.par_chunks_mut(3).for_each(|p| {
        let l = p[0];
        for (tint, mask) in [
            (shadow, step(1.0 - l / 50.0)),
            (highlight, step((l - 50.0) / 50.0)),
        ] {
            if let Some(c) = tint {
                p[1] = (p[1] as f64 + c[0] * mask as f64) as f32;
                p[2] = (p[2] as f64 + c[1] * mask as f64) as f32;
            }
        }
    });
    color::lab_to_rgb(&lab, dst, to_rgb);
    dst.par_iter_mut().for_each(|v| *v = clip(*v, 0.0, 1.0));
}

/// `apply_hue_trim` of an HxWx3 image for a nonzero angle given as its (cos, sin): Lab, a*b*
/// rotated in float32, back to working RGB clipped to 0..1.
pub fn hue_trim(
    src: &[f32],
    (cos, sin): (f32, f32),
    spaces: (LabSpace, LabSpace),
    dst: &mut [f32],
) {
    let mut lab = vec![0f32; src.len()];
    color::rgb_to_lab(src, &mut lab, spaces.0);
    lab.par_chunks_mut(3 * 4096).for_each(move |run| {
        for p in run.as_chunks_mut::<3>().0 {
            let (a, b) = (p[1], p[2]);
            p[1] = a * cos - b * sin;
            p[2] = a * sin + b * cos;
        }
    });
    color::lab_to_rgb(&lab, dst, spaces.1);
    dst.par_chunks_mut(4096).for_each(|run| {
        for v in run {
            let c = if v.is_nan() || *v > 0.0 { *v } else { 0.0 };
            *v = if c.is_nan() || c < 1.0 { c } else { 1.0 };
        }
    });
}
