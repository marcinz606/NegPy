//! The per-pixel passes of the retouch score fill (negpy/features/retouch/logic.py), in its
//! float32 operation order.

use multiversion::multiversion;
use rayon::prelude::*;

const BLOCK: usize = 1 << 13;

/// Python's `max(a, b)`, which Numba keeps: `a` unless `b` is greater.
fn py_max(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// Python's `min(a, b)`: `a` unless `b` is smaller.
fn py_min(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn fill_rung_block(
    num: &[f32],
    den: &[f32],
    fill: &mut [f32],
    first: bool,
    reject: bool,
    (floor, tau): (f32, f32),
) {
    let inv_span = 1.0 - floor;
    for ((f, n), &dv) in fill
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(num.as_chunks::<3>().0)
        .zip(den)
    {
        let d = py_max(dv, 1e-6);
        if first {
            for c in 0..3 {
                f[c] = n[c] / d;
            }
        } else {
            let mass = if reject { (dv - floor) / inv_span } else { dv };
            let conf = py_min(py_max(mass / tau, 0.0), 1.0);
            let keep = 1.0 - conf;
            for c in 0..3 {
                f[c] = f[c] * keep + (n[c] / d) * conf;
            }
        }
    }
}

/// `_fill_rung`: one rung of the fill ladder in place over HxW pixels of three channels.
pub fn fill_rung(
    num: &[f32],
    den: &[f32],
    fill: &mut [f32],
    first: bool,
    reject_floor_mass: bool,
    constants: (f32, f32),
) {
    fill.par_chunks_mut(3 * BLOCK)
        .zip(num.par_chunks(3 * BLOCK).zip(den.par_chunks(BLOCK)))
        .for_each(|(f, (n, d))| fill_rung_block(n, d, f, first, reject_floor_mass, constants));
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn blend_block(
    src: &[f32],
    score: &[f32],
    fill: &mut [f32],
    a_out: &mut [f32],
    (hi, span): (f32, f32),
) {
    for (((f, s), &sc), ao) in fill
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0)
        .zip(score)
        .zip(a_out.iter_mut())
    {
        let a = py_min(py_max((hi - sc) / span, 0.0), 1.0);
        let a = a * a * (3.0 - 2.0 * a);
        *ao = a;
        let keep = 1.0 - a;
        for c in 0..3 {
            f[c] = s[c] * keep + f[c] * a;
        }
    }
}

/// `_blend_fill`: `fill` becomes `src * (1 - a) + fill * a` under the smoothstep write ramp of
/// the score, the ramp kept in `a_out`.
pub fn blend_fill(
    src: &[f32],
    score: &[f32],
    fill: &mut [f32],
    a_out: &mut [f32],
    ramp: (f32, f32),
) {
    fill.par_chunks_mut(3 * BLOCK)
        .zip(a_out.par_chunks_mut(BLOCK))
        .zip(src.par_chunks(3 * BLOCK).zip(score.par_chunks(BLOCK)))
        .for_each(|((f, ao), (s, sc))| blend_block(s, sc, f, ao, ramp));
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn deficit_block(out: &mut [f32], a: &[f32], blur_src: &[f32], blur_out: &[f32]) {
    for (((o, &av), bs), bo) in out
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(a)
        .zip(blur_src.as_chunks::<3>().0)
        .zip(blur_out.as_chunks::<3>().0)
    {
        for c in 0..3 {
            let v = o[c] + av * py_max(bs[c] - bo[c], 0.0);
            o[c] = py_max(v, 0.0);
        }
    }
}

/// `_add_floor_deficit`: `out += a * max(blur_src - blur_out, 0)`, then clamped at zero.
pub fn add_floor_deficit(out: &mut [f32], a: &[f32], blur_src: &[f32], blur_out: &[f32]) {
    out.par_chunks_mut(3 * BLOCK)
        .zip(a.par_chunks(BLOCK))
        .zip(
            blur_src
                .par_chunks(3 * BLOCK)
                .zip(blur_out.par_chunks(3 * BLOCK)),
        )
        .for_each(|((o, av), (bs, bo))| deficit_block(o, av, bs, bo));
}

/// `_hair_encode`: `clip((crop - lo) * inv_span, 0, 1) ** inv_gamma` to 8-bit for the first
/// three channels, in float32 (no fastmath), truncated as `np.uint8` truncates.
pub fn hair_encode(
    crop: &[f32],
    cn: usize,
    lo: f32,
    inv_span: f32,
    inv_gamma: f32,
    out: &mut [u8],
) {
    out.par_chunks_mut(cn * 1024)
        .zip(crop.par_chunks(cn * 1024))
        .for_each(|(o, c)| hair_encode_block(c, o, cn, (lo, inv_span, inv_gamma)));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn hair_encode_block(
    crop: &[f32],
    out: &mut [u8],
    cn: usize,
    (lo, inv_span, inv_gamma): (f32, f32, f32),
) {
    let encode = |v: f32| {
        let v = ((v - lo) * inv_span).clamp(0.0, 1.0);
        let p = crate::mathf::pow_f64(v as f64, inv_gamma as f64) as f32;
        (p * 255.0 + 0.5) as u8
    };
    if cn == 3 {
        for (o, &v) in out.iter_mut().zip(crop) {
            *o = encode(v);
        }
        return;
    }
    for (o, c) in out.chunks_mut(cn).zip(crop.chunks(cn)) {
        for k in 0..3 {
            o[k] = encode(c[k]);
        }
    }
}

/// The scalars of `_ir_clean_base`, float32 as NumPy rounds them.
pub struct CleanBase {
    pub win: usize,
    pub identity: f32,
    pub sigma: f32,
    pub min_support: f32,
}

/// `_ir_clean_base(img_det, ratio)` of an HxWx3 image: the clean-film mean minus `sigma` of its
/// spread where enough of the window is clean, else the box mean of the window maximum.
pub fn ir_clean_base(
    img: &[f32],
    ratio: &[f32],
    (h, w): (usize, usize),
    c: &CleanBase,
    dst: &mut [f32],
) -> Option<()> {
    use crate::filter::{Border, box_filter};
    use crate::morph::{self, Op};
    let n = h * w;
    if ratio.len() != n || img.len() != 3 * n || dst.len() != img.len() {
        return None;
    }
    let win = (c.win, c.win);
    let identity = c.identity;
    let clean: Vec<f32> = ratio
        .par_iter()
        .map(move |&r| if r >= identity { 1.0 } else { 0.0 })
        .collect();
    let (mut p1, mut p2) = (vec![0f32; 3 * n], vec![0f32; 3 * n]);
    p1.par_chunks_mut(3 * BLOCK)
        .zip(p2.par_chunks_mut(3 * BLOCK))
        .zip(img.par_chunks(3 * BLOCK).zip(clean.par_chunks(BLOCK)))
        .for_each(|((p1, p2), (px, wc))| {
            for (i, &k) in wc.iter().enumerate() {
                for j in 3 * i..3 * i + 3 {
                    let v = px[j];
                    p1[j] = v * k;
                    p2[j] = v * v * k;
                }
            }
        });
    let border = Border::Reflect101;
    let mut den = vec![0f32; n];
    box_filter(&clean, &mut den, (h, w, 1), win, true, border)?;
    let (mut b1, mut b2) = (vec![0f32; 3 * n], vec![0f32; 3 * n]);
    box_filter(&p1, &mut b1, (h, w, 3), win, true, border)?;
    box_filter(&p2, &mut b2, (h, w, 3), win, true, border)?;
    // p1 holds the dilated image, then its box mean.
    morph::morphology(
        img,
        &mut p1,
        (h, w, 3),
        Op::Dilate,
        win,
        1,
        Border::Constant,
    )?;
    box_filter(&p1, &mut p2, (h, w, 3), win, true, border)?;
    let (sigma, support) = (c.sigma, c.min_support);
    dst.par_chunks_mut(3 * BLOCK)
        .zip(den.par_chunks(BLOCK))
        .zip(b1.par_chunks(3 * BLOCK).zip(b2.par_chunks(3 * BLOCK)))
        .zip(p2.par_chunks(3 * BLOCK))
        .for_each(|(((o, d), (b1, b2)), dil)| clean_base_block(o, d, b1, b2, dil, sigma, support));
    Some(())
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn clean_base_block(
    out: &mut [f32],
    den: &[f32],
    b1: &[f32],
    b2: &[f32],
    dil: &[f32],
    sigma: f32,
    support: f32,
) {
    for (i, &d) in den.iter().enumerate() {
        let d = py_max(d, 1e-6);
        for j in 3 * i..3 * i + 3 {
            let mean = b1[j] / d;
            let var = b2[j] / d - mean * mean;
            let var = if var.is_nan() || var > 0.0 { var } else { 0.0 };
            let base = mean - sigma * var.sqrt();
            out[j] = if d > support { base } else { dil[j] };
        }
    }
}

/// `_ir_live(plane)`: 1 where film lies under the head, 0 in the below-`floor` regions of at
/// least `min_area` of the frame.
pub fn ir_live(plane: &[f32], (h, w): (usize, usize), floor: f32, min_area: f64) -> Vec<bool> {
    let dead: Vec<u8> = plane.par_iter().map(move |&v| (v < floor) as u8).collect();
    if !dead.par_iter().any(|&d| d != 0) {
        return vec![true; h * w];
    }
    let cc = crate::components::connected_components(&dead, h, w, 8);
    let bar = min_area * (h * w) as f64;
    let holder: Vec<bool> = (0..cc.count)
        .map(|l| l > 0 && cc.stats[l][4] as f64 >= bar)
        .collect();
    cc.labels.par_iter().map(|&l| !holder[l as usize]).collect()
}

/// Strided flat indices of `mask`, at most `cap` of them, as `_fit_sample` takes them: every
/// `step`-th set pixel in raster order, counted per block and then picked in parallel.
fn fit_indices(mask: &[bool], cap: usize) -> Vec<usize> {
    const CHUNK: usize = 1 << 16;
    let counts: Vec<usize> = mask
        .par_chunks(CHUNK)
        .map(|c| c.iter().filter(|&&m| m).count())
        .collect();
    let step = counts.iter().sum::<usize>().div_ceil(cap).max(1);
    let starts: Vec<usize> = counts
        .iter()
        .scan(0, |acc, &c| {
            let s = *acc;
            *acc += c;
            Some(s)
        })
        .collect();
    mask.par_chunks(CHUNK)
        .zip(starts)
        .enumerate()
        .flat_map_iter(|(b, (c, start))| {
            let mut rank = start;
            c.iter().enumerate().filter_map(move |(i, &m)| {
                if !m {
                    return None;
                }
                let take = rank % step == 0;
                rank += 1;
                take.then_some(b * CHUNK + i)
            })
        })
        .collect()
}

/// `_fit_sample(mask)`: the values under `mask`, every `step`-th one so at most `cap` remain.
fn fit_sample(values: &[f32], mask: &[bool], cap: usize) -> Vec<f32> {
    fit_indices(mask, cap)
        .into_iter()
        .map(|i| values[i])
        .collect()
}

/// The median of `sample` and 1.4826 times its median absolute deviation, as Python floats.
fn median_and_sigma(sample: &[f32]) -> (f64, f64) {
    let med = crate::stats::median(sample);
    let dev: Vec<f32> = sample.par_iter().map(move |&v| (v - med).abs()).collect();
    (med as f64, 1.4826 * crate::stats::median(&dev) as f64)
}

/// `_ir_glare(plane, live)`: 1 more than `sigmas` deviations over the live median, `None` when
/// fewer than 500 samples or no pixel qualifies.
pub fn ir_glare(plane: &[f32], live: &[bool], cap: usize, sigmas: f64) -> Option<Vec<bool>> {
    let sample = fit_sample(plane, live, cap);
    if sample.len() < 500 {
        return None;
    }
    let (med, sigma) = median_and_sigma(&sample);
    let bar = (med + sigmas * sigma) as f32;
    let glare: Vec<bool> = plane.par_iter().map(move |&v| v > bar).collect();
    glare.par_iter().any(|&g| g).then_some(glare)
}

/// `normalize_ir(plane, glare)`: the plane over the box mean of its `win` x `win` maximum,
/// glare pixels zeroed for the base.
pub fn normalize_ir(
    plane: &[f32],
    glare: Option<&[bool]>,
    (h, w): (usize, usize),
    win: usize,
    dst: &mut [f32],
) -> Option<()> {
    use crate::filter::{Border, box_filter};
    use crate::morph::{self, Op};
    let n = h * w;
    if plane.len() != n || dst.len() != n || glare.is_some_and(|g| g.len() != n) {
        return None;
    }
    let src: std::borrow::Cow<[f32]> = match glare {
        Some(g) => plane
            .par_iter()
            .zip(g)
            .map(|(&v, &g)| if g { 0.0 } else { v })
            .collect::<Vec<f32>>()
            .into(),
        None => plane.into(),
    };
    let mut dil = vec![0f32; n];
    morph::morphology(
        &src,
        &mut dil,
        (h, w, 1),
        Op::Dilate,
        (win, win),
        1,
        Border::Constant,
    )?;
    box_filter(&dil, dst, (h, w, 1), (win, win), true, Border::Reflect101)?;
    crate::par_zip(dst, plane, move |d, &p| *d = p / py_max(*d, 1e-4));
    Some(())
}

/// The scalars of `_ir_normalize_ratio`, as Python floats.
pub struct RatioNorm {
    pub identity: f64,
    pub noise_sigma: f64,
    pub pivot_lo: f64,
    pub ref_sigma: f64,
    pub scale_max: f64,
}

/// `np.clip(v, lo, hi)` of Python floats: a NaN comes through.
fn clip_f64(v: f64, lo: f64, hi: f64) -> f64 {
    let m = if v.is_nan() || v > lo { v } else { lo };
    if m.is_nan() || m < hi { m } else { hi }
}

/// `_ir_normalize_ratio(ratio, live)` in place; returns the (pivot, sigma) it measured, or
/// `None` when fewer than 500 live pixels leave the ratio alone.
pub fn ir_normalize_ratio(ratio: &mut [f32], live: &[bool], c: &RatioNorm) -> Option<(f64, f64)> {
    let sample: Vec<f32> = ratio
        .par_iter()
        .zip(live)
        .filter(|&(_, &l)| l)
        .map(|(&v, _)| v)
        .collect();
    if sample.len() < 500 {
        return None;
    }
    let (med, sigma) = median_and_sigma(&sample);
    let pivot = clip_f64(med - c.noise_sigma * sigma, c.pivot_lo, c.identity);
    let scale = c.identity / pivot;
    let spread = sigma * scale;
    let k = clip_f64(
        c.ref_sigma / if 1e-6 > spread { 1e-6 } else { spread },
        1.0,
        c.scale_max,
    );
    let (s32, k32, id32) = (scale as f32, k as f32, c.identity as f32);
    let stretch = k > 1.0;
    ratio.par_chunks_mut(BLOCK).for_each(move |run| {
        for v in run {
            let r = *v * s32;
            *v = if stretch && r < id32 {
                id32 - (id32 - r) * k32
            } else {
                r
            };
        }
    });
    Some((pivot, sigma))
}

/// `np.clip(ratio / np.exp((vis_log * b).sum(-1)), 0.0, 1.5)`: `_ir_decontaminate`'s division,
/// in float32 (lstsq keeps float32).
pub fn ir_ghost_divide(ratio: &[f32], vis_log: &[f32], b: [f32; 3], dst: &mut [f32]) {
    dst.par_chunks_mut(BLOCK)
        .zip(ratio.par_chunks(BLOCK).zip(vis_log.par_chunks(3 * BLOCK)))
        .for_each(move |(o, (r, v))| ghost_divide_block(o, r, v, b));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn ghost_divide_block(out: &mut [f32], ratio: &[f32], vis_log: &[f32], b: [f32; 3]) {
    for ((o, &r), v) in out.iter_mut().zip(ratio).zip(vis_log.as_chunks::<3>().0) {
        let s = ((-0.0 + v[0] * b[0]) + v[1] * b[1]) + v[2] * b[2];
        let q = r / crate::mathf::np_expf(s);
        let m = if q.is_nan() || q > 0.0 { q } else { 0.0 };
        *o = if m.is_nan() || m < 1.5 { m } else { 1.5 };
    }
}

/// `score_weighted_fill(img, score, scales, reject_floor_mass)` of an HxWx3 image: box means
/// of the score-weighted image over the score at every support but the last, which is a
/// Gaussian, each rung blended over the coarser ones by `fill_rung`. `reject_floor_mass` counts
/// a rung's confidence above the score floor: without the original-floor rule, a support that
/// sees only defect would otherwise win with the defect as its candidate.
pub fn score_weighted_fill(
    img: &[f32],
    score: &[f32],
    (h, w): (usize, usize),
    scales: &[usize],
    reject_floor_mass: bool,
    constants: (f32, f32),
    fill: &mut [f32],
) -> Option<()> {
    use crate::filter::{Border, box_filter, gaussian_blur};
    let n = h * w;
    if score.len() != n || img.len() != 3 * n || fill.len() != img.len() {
        return None;
    }
    let mut weighted = vec![0f32; 3 * n];
    weighted
        .par_chunks_mut(3 * BLOCK)
        .zip(img.par_chunks(3 * BLOCK).zip(score.par_chunks(BLOCK)))
        .for_each(|(o, (px, s))| {
            for (i, &k) in s.iter().enumerate() {
                for j in 3 * i..3 * i + 3 {
                    o[j] = px[j] * k;
                }
            }
        });
    let (mut num, mut den) = (vec![0f32; 3 * n], vec![0f32; n]);
    let border = Border::Reflect101;
    for (i, &k) in scales.iter().enumerate() {
        if i == scales.len() - 1 {
            gaussian_blur(&weighted, &mut num, (h, w, 3), (k, k), (0.0, 0.0), border)?;
            gaussian_blur(score, &mut den, (h, w, 1), (k, k), (0.0, 0.0), border)?;
        } else {
            box_filter(&weighted, &mut num, (h, w, 3), (k, k), true, border)?;
            box_filter(score, &mut den, (h, w, 1), (k, k), true, border)?;
        }
        fill_rung(&num, &den, fill, i == 0, reject_floor_mass, constants);
    }
    Some(())
}

/// `_borrow_clean_grain` added into `out` at every pixel `clean` leaves out: the detail of the
/// pixel mirrored through the nearest clean one (that clean pixel itself when the mirror is
/// not clean), high-passed at `sigma`, times `a` when given. Returns the blur of `src`, or
/// `None` without touching `out` when no pixel or every pixel is clean.
pub fn add_clean_grain(
    src: &[f32],
    clean: &[bool],
    (h, w): (usize, usize),
    sigma: f64,
    a: Option<&[f32]>,
    out: &mut [f32],
) -> Option<Vec<f32>> {
    if !clean.par_iter().any(|&c| c) || clean.par_iter().all(|&c| c) {
        return None;
    }
    let holes: Vec<u8> = clean.par_iter().map(|&c| !c as u8).collect();
    let (_, labels) = crate::distance::distance_transform_labels(&holes, h, w);
    // The labels number the clean pixels from 1 in raster order.
    let donors: Vec<usize> = (0..h * w).filter(|&p| clean[p]).collect();
    let mut blur = vec![0f32; 3 * h * w];
    crate::filter::gaussian_blur(
        src,
        &mut blur,
        (h, w, 3),
        (0, 0),
        (sigma, sigma),
        crate::filter::Border::Reflect101,
    )?;
    let (hi, wi) = (h as isize - 1, w as isize - 1);
    out.par_chunks_mut(3 * w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let p = y * w + x;
            if clean[p] {
                continue;
            }
            let nearest = donors[labels[p] as usize - 1];
            let (ny, nx) = ((nearest / w) as isize, (nearest % w) as isize);
            let my = (2 * ny - y as isize).clamp(0, hi) as usize;
            let mx = (2 * nx - x as isize).clamp(0, wi) as usize;
            let mirrored = my * w + mx;
            let take = if clean[mirrored] { mirrored } else { nearest };
            for c in 0..3 {
                let g = src[3 * take + c] - blur[3 * take + c];
                row[3 * x + c] += match a {
                    Some(a) => a[p] * g,
                    None => g,
                };
            }
        }
    });
    Some(blur)
}

/// The scalars of `apply_score_repair`, float32 as NumPy rounds them.
pub struct ScoreRepair {
    pub score_floor: f32,
    pub tau: f32,
    pub write_hi: f32,
    /// `np.float32(write_hi - write_lo)`, subtracted as Python floats.
    pub write_span: f32,
}

/// `apply_score_repair` of an HxWx3 image and a score at its resolution, with the fill
/// supports and grain `sigma` already chosen.
#[allow(clippy::too_many_arguments)]
pub fn apply_score_repair(
    src: &[f32],
    score: &[f32],
    (h, w): (usize, usize),
    supports: &[usize],
    floor: bool,
    sigma: f64,
    c: &ScoreRepair,
    out: &mut [f32],
) -> Option<()> {
    let n = h * w;
    score_weighted_fill(
        src,
        score,
        (h, w),
        supports,
        !floor,
        (c.score_floor, c.tau),
        out,
    )?;
    let mut a = vec![0f32; n];
    blend_fill(src, score, out, &mut a, (c.write_hi, c.write_span));
    let hi = c.write_hi;
    let clean: Vec<bool> = score.par_iter().map(move |&s| s >= hi).collect();
    let blur_src = add_clean_grain(src, &clean, (h, w), sigma, Some(&a), out);
    if floor {
        let gauss = |img: &[f32]| {
            let mut b = vec![0f32; 3 * n];
            crate::filter::gaussian_blur(
                img,
                &mut b,
                (h, w, 3),
                (0, 0),
                (sigma, sigma),
                crate::filter::Border::Reflect101,
            )
            .map(|_| b)
        };
        let blur_src = match blur_src {
            Some(b) => b,
            None => gauss(src)?,
        };
        let blur_out = gauss(out)?;
        add_floor_deficit(out, &a, &blur_src, &blur_out);
    } else {
        out.par_chunks_mut(BLOCK).for_each(|run| {
            for v in run {
                *v = if v.is_nan() || *v > 0.0 { *v } else { 0.0 };
            }
        });
    }
    Some(())
}

/// Repaired pixels of one defect: flat index and value.
type Patch = Vec<(usize, [f32; 3])>;

/// `repair_components`: `apply_score_repair` per 8-connected defect (score < 1), each in its
/// own crop padded by `pad`, written back on that defect's pixels only. `out` holds the base
/// to write into. With `dirty`, a defect holding no dirty pixel keeps `out` as it is. Past
/// `max_components` the whole frame is repaired at once and `dirty` is ignored. Returns false,
/// with `out` untouched, when the score holds no defect.
#[allow(clippy::too_many_arguments)]
pub fn repair_components(
    src: &[f32],
    score: &[f32],
    (h, w): (usize, usize),
    supports: &[usize],
    floor: bool,
    sigma: f64,
    c: &ScoreRepair,
    pad: usize,
    max_components: usize,
    dirty: Option<&[bool]>,
    out: &mut [f32],
) -> Option<bool> {
    let m: Vec<u8> = score.par_iter().map(|&s| (s < 1.0) as u8).collect();
    if !m.par_iter().any(|&v| v != 0) {
        return Some(false);
    }
    let cc = crate::components::connected_components(&m, h, w, 8);
    if cc.count - 1 > max_components {
        apply_score_repair(src, score, (h, w), supports, floor, sigma, c, out)?;
        return Some(true);
    }
    let labels = &cc.labels;
    let fixes: Vec<Patch> = (1..cc.count)
        .into_par_iter()
        .map(|i| {
            let [bx, by, bw, bh, _] = cc.stats[i].map(|v| v as usize);
            let own = |y: usize, x: usize| labels[y * w + x] == i as i32;
            if let Some(d) = dirty {
                let touched =
                    (by..by + bh).any(|y| (bx..bx + bw).any(|x| own(y, x) && d[y * w + x]));
                if !touched {
                    return Some(Vec::new());
                }
            }
            let (x0, y0) = (bx.saturating_sub(pad), by.saturating_sub(pad));
            let (x1, y1) = ((bx + bw + pad).min(w), (by + bh + pad).min(h));
            let (cw, ch) = (x1 - x0, y1 - y0);
            let mut crop = Vec::with_capacity(3 * cw * ch);
            let mut crop_score = Vec::with_capacity(cw * ch);
            for y in y0..y1 {
                crop.extend_from_slice(&src[3 * (y * w + x0)..3 * (y * w + x1)]);
                crop_score.extend_from_slice(&score[y * w + x0..y * w + x1]);
            }
            let mut sub = vec![0f32; crop.len()];
            apply_score_repair(
                &crop,
                &crop_score,
                (ch, cw),
                supports,
                floor,
                sigma,
                c,
                &mut sub,
            )?;
            let mut px = Vec::new();
            for y in y0..y1 {
                for x in x0..x1 {
                    if own(y, x) {
                        let k = 3 * ((y - y0) * cw + x - x0);
                        px.push((y * w + x, [sub[k], sub[k + 1], sub[k + 2]]));
                    }
                }
            }
            Some(px)
        })
        .collect::<Option<Vec<_>>>()?;
    for px in fixes {
        for (p, v) in px {
            out[3 * p..3 * p + 3].copy_from_slice(&v);
        }
    }
    Some(true)
}

/// `route_wide_defects` up to its budget check: 1 on the at-floor (score <= `bar`) components
/// that hold a chessboard distance of `radius` from their own edge, dilated by `kernel`;
/// `None` when no component qualifies.
pub fn route_wide_defects(
    score: &[f32],
    (h, w): (usize, usize),
    bar: f32,
    radius: usize,
    kernel: crate::morph::Kernel,
) -> Option<Vec<u8>> {
    use crate::distance::{Metric, distance_transform};
    let at_floor: Vec<u8> = score.par_iter().map(move |&s| (s <= bar) as u8).collect();
    if !at_floor.par_iter().any(|&v| v != 0) {
        return None;
    }
    let cc = crate::components::connected_components(&at_floor, h, w, 8);
    let side = (2 * radius).saturating_sub(1);
    let labels = &cc.labels;
    let wide: Vec<usize> = (1..cc.count)
        .into_par_iter()
        .filter(|&i| {
            let [bx, by, bw, bh, _] = cc.stats[i].map(|v| v as usize);
            if bw.min(bh) < side {
                return false;
            }
            // The component alone in its box, with a zero border.
            let (pw, ph) = (bw + 2, bh + 2);
            let mut own = vec![0u8; pw * ph];
            for y in 0..bh {
                for x in 0..bw {
                    own[(y + 1) * pw + x + 1] = (labels[(by + y) * w + bx + x] == i as i32) as u8;
                }
            }
            let d = distance_transform(&own, ph, pw, Metric::Chessboard);
            d.iter().fold(0f32, |m, &v| m.max(v)) >= radius as f32
        })
        .collect();
    if wide.is_empty() {
        return None;
    }
    let mut routed = vec![0u8; h * w];
    for &i in &wide {
        let [bx, by, bw, bh, _] = cc.stats[i].map(|v| v as usize);
        for y in by..by + bh {
            for x in bx..bx + bw {
                if labels[y * w + x] == i as i32 {
                    routed[y * w + x] = 1;
                }
            }
        }
    }
    let mut out = vec![0u8; h * w];
    crate::morph::morphology_mask(
        &routed,
        &mut out,
        (h, w, 1),
        crate::morph::Op::Dilate,
        kernel,
        1,
        crate::filter::Border::Constant,
    )?;
    Some(out)
}

/// The scalars of `_fit_refraction_gammas`.
pub struct GammaFit {
    pub band: (f32, f32),
    pub flat_pct: f64,
    pub min_flat: usize,
    pub cap: usize,
    pub fallback: f64,
    pub range: (f32, f32),
}

/// `_fit_refraction_gammas(ratio, vis_log, img_det)`: per channel, the median slope of the
/// visible log over the IR log across the shallow-dust band, on its locally flat part. The
/// flatness is cv2's 5x5 Laplacian of the green channel without IPP.
pub fn fit_refraction_gammas(
    ratio: &[f32],
    vis_log: &[f32],
    img: &[f32],
    (h, w): (usize, usize),
    c: &GammaFit,
) -> Option<[f64; 3]> {
    use crate::filter::{Border, sep_filter2d};
    let (lo, hi) = c.band;
    let band: Vec<bool> = ratio.par_iter().map(move |&r| r > lo && r < hi).collect();
    if band.par_iter().filter(|&&b| b).count() < 500 {
        return Some([c.fallback; 3]);
    }
    let mut green = vec![0f32; h * w];
    green
        .par_chunks_mut(BLOCK)
        .zip(img.par_chunks(3 * BLOCK))
        .for_each(|(g, px)| {
            for (g, p) in g.iter_mut().zip(px.as_chunks::<3>().0) {
                *g = p[1];
            }
        });
    let d2 = [1.0f32, 0.0, -2.0, 0.0, 1.0];
    let sm = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let (mut dx, mut dy) = (vec![0f32; h * w], vec![0f32; h * w]);
    sep_filter2d(&green, &mut dx, (h, w, 1), (&d2, &sm), Border::Reflect101)?;
    sep_filter2d(&green, &mut dy, (h, w, 1), (&sm, &d2), Border::Reflect101)?;
    crate::par_zip(&mut dx, &dy, |a, &b| *a = (*a + b).abs());
    let in_band: Vec<f32> = dx
        .par_chunks(BLOCK)
        .zip(band.par_chunks(BLOCK))
        .flat_map_iter(|(e, b)| {
            e.iter()
                .zip(b)
                .filter(|&(_, &b)| b)
                .map(|(&e, _)| e)
                .collect::<Vec<_>>()
        })
        .collect();
    let bar = crate::stats::np_percentiles(&in_band, &[c.flat_pct])?[0];
    let flat: Vec<bool> = band
        .par_iter()
        .zip(&dx)
        .map(move |(&b, &e)| b && e < bar)
        .collect();
    let enough = flat.par_iter().filter(|&&f| f).count() >= c.min_flat;
    let fit = fit_indices(if enough { &flat } else { &band }, c.cap);
    let xb: Vec<f32> = fit
        .iter()
        .map(|&i| crate::mathf::np_logf(ratio[i]))
        .collect();
    let (glo, ghi) = c.range;
    Some(std::array::from_fn(|ch| {
        let slopes: Vec<f32> = fit
            .iter()
            .zip(&xb)
            .map(|(&i, &x)| vis_log[3 * i + ch] / x)
            .collect();
        let m = crate::stats::median(&slopes);
        let m = if m.is_nan() || m > glo { m } else { glo };
        (if m.is_nan() || m < ghi { m } else { ghi }) as f64
    }))
}

/// `ir_defect_score`: the ratio mapped from `cutoff` (score `floor`) to `cutoff + span`
/// (score 1), clipped, then a 3x3 erode, in float32.
pub fn ir_defect_score(
    ratio: &[f32],
    (h, w): (usize, usize),
    (cutoff, span, keep, floor): (f32, f32, f32, f32),
    dst: &mut [f32],
) -> Option<()> {
    let mut score = vec![0f32; h * w];
    crate::par_zip(&mut score, ratio, move |s, &r| {
        let v = ((r - cutoff) / span) * keep + floor;
        let v = if v.is_nan() || v > floor { v } else { floor };
        *s = if v.is_nan() || v < 1.0 { v } else { 1.0 };
    });
    crate::morph::morphology(
        &score,
        dst,
        (h, w, 1),
        crate::morph::Op::Erode,
        (3, 3),
        1,
        crate::filter::Border::Constant,
    )
}

/// `_box_mean_std(plane, win)`: the box mean and the standard deviation over `win` x `win`,
/// `sqrt(max(blur(p * p) - mean * mean, 0))` in float32.
pub fn box_mean_std(
    plane: &[f32],
    (h, w): (usize, usize),
    win: usize,
    mean: &mut [f32],
    std: &mut [f32],
) -> Option<()> {
    use crate::filter::{Border, box_filter};
    let mut sq = vec![0f32; h * w];
    crate::par_zip(&mut sq, plane, |s, &p| *s = p * p);
    box_filter(plane, mean, (h, w, 1), (win, win), true, Border::Reflect101)?;
    box_filter(&sq, std, (h, w, 1), (win, win), true, Border::Reflect101)?;
    crate::par_zip(std, mean, |s, &m| {
        let v = *s - m * m;
        *s = (if v.is_nan() || v > 0.0 { v } else { 0.0 }).sqrt();
    });
    Some(())
}

/// `_u8(plane * gain)`: clipped to 0..1, scaled to 8 bits and rounded half up, in float32.
fn to_u8(plane: &[f32], gain: f32) -> Vec<u8> {
    plane
        .par_iter()
        .map(move |&p| {
            let v = p * gain;
            let v = if v.is_nan() || v > 0.0 { v } else { 0.0 };
            let v = if v.is_nan() || v < 1.0 { v } else { 1.0 };
            (v * 255.0 + 0.5) as u8
        })
        .collect()
}

/// The scalars of `compute_dust_stats`.
pub struct DustStats<'a> {
    /// Median and texture windows.
    pub v_win: usize,
    pub w_win: usize,
    pub disk: crate::morph::Kernel<'a>,
    pub min_spread: f64,
    pub avg_px: usize,
    pub mad_gain: f32,
    pub sigma_min: f32,
}

/// `compute_dust_stats` from the density plane: (proxy, background, z, texture).
pub fn dust_stats(dens: &[f32], (h, w): (usize, usize), c: &DustStats) -> Option<[Vec<f32>; 4]> {
    use crate::filter::{Border, box_filter};
    let n = h * w;
    let q = crate::stats::np_percentiles_f64(dens, &[0.5, 99.5]);
    let spread = q[1] - q[0];
    let spread = if c.min_spread > spread {
        c.min_spread
    } else {
        spread
    };
    let (lo, spread) = (q[0] as f32, spread as f32);
    let mut proxy = vec![0f32; n];
    crate::par_zip(&mut proxy, dens, move |p, &d| {
        let v = (d - lo) / spread;
        let v = if v.is_nan() || v > 0.0 { v } else { 0.0 };
        *p = if v.is_nan() || v < 1.0 { v } else { 1.0 };
    });
    let mut med = vec![0u8; n];
    crate::median::median_blur_u8(&to_u8(&proxy, 1.0), &mut med, h, w, c.v_win)?;
    let mut background = vec![0f32; n];
    crate::morph::morphology_mask(
        &proxy,
        &mut background,
        (h, w, 1),
        crate::morph::Op::Open,
        crate::morph::Kernel { ..c.disk },
        1,
        Border::Constant,
    )?;
    background
        .par_chunks_mut(BLOCK)
        .zip(med.par_chunks(BLOCK))
        .for_each(|(b, m)| {
            for (b, &m) in b.iter_mut().zip(m) {
                let a = m as f32 / 255.0;
                *b = if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else if a > *b {
                    a
                } else {
                    *b
                };
            }
        });
    let mut diff = vec![0f32; n];
    diff.par_chunks_mut(BLOCK)
        .zip(proxy.par_chunks(BLOCK).zip(background.par_chunks(BLOCK)))
        .for_each(|(d, (p, b))| {
            for ((d, &p), &b) in d.iter_mut().zip(p).zip(b) {
                *d = p - b;
            }
        });
    let mut z = vec![0f32; n];
    box_filter(
        &diff,
        &mut z,
        (h, w, 1),
        (c.avg_px, c.avg_px),
        true,
        Border::Reflect101,
    )?;
    let excess_u8: Vec<u8> = {
        let abs: Vec<f32> = z.par_iter().map(|v| v.abs()).collect();
        to_u8(&abs, c.mad_gain)
    };
    crate::median::median_blur_u8(&excess_u8, &mut med, h, w, c.w_win)?;
    let (div, floor) = (255.0 * c.mad_gain, c.sigma_min);
    z.par_chunks_mut(BLOCK)
        .zip(med.par_chunks(BLOCK))
        .for_each(|(z, m)| {
            for (z, &m) in z.iter_mut().zip(m) {
                let sigma = (m as f32 / div) / 0.6745;
                let sigma = if sigma.is_nan() || sigma > floor {
                    sigma
                } else {
                    floor
                };
                *z /= sigma;
            }
        });
    let (mut mean, mut texture) = (vec![0f32; n], vec![0f32; n]);
    box_mean_std(&proxy, (h, w), c.w_win, &mut mean, &mut texture)?;
    Some([proxy, background, z, texture])
}

/// `_is_hair`: thin rather than a speck. Twice the largest L2 (5x5) distance from the edge of
/// the padded component is its thickness; area over thickness squared reads as length over
/// thickness for a ribbon, bent or straight.
fn is_hair(
    own: &[u8],
    (bh, bw): (usize, usize),
    area: usize,
    (min_area, min_elong): (usize, f64),
) -> bool {
    use crate::distance::{Metric, distance_transform};
    if area < min_area {
        return false;
    }
    let (ph, pw) = (bh + 2, bw + 2);
    let mut padded = vec![0u8; ph * pw];
    for y in 0..bh {
        padded[(y + 1) * pw + 1..(y + 1) * pw + 1 + bw].copy_from_slice(&own[y * bw..(y + 1) * bw]);
    }
    let d = distance_transform(&padded, ph, pw, Metric::L2Mask5);
    let thickness = 2.0 * d.iter().fold(0f32, |m, &v| m.max(v)) as f64;
    let sq = thickness * thickness;
    area as f64 / if 1e-6 > sq { 1e-6 } else { sq } >= min_elong
}

/// Component `i` of `labels` in its bounding box, 1 on its pixels.
fn component_crop(
    labels: &[i32],
    w: usize,
    i: usize,
    s: [i32; 5],
) -> (Vec<u8>, (usize, usize, usize, usize)) {
    let [bx, by, bw, bh, _] = s.map(|v| v as usize);
    let mut own = vec![0u8; bw * bh];
    for y in 0..bh {
        for x in 0..bw {
            own[y * bw + x] = (labels[(by + y) * w + bx + x] == i as i32) as u8;
        }
    }
    (own, (bx, by, bw, bh))
}

/// `split_hairs(mask)`: (compact, hairs), hairs `None` when no component is thin enough.
pub fn split_hairs(
    mask: &[u8],
    (h, w): (usize, usize),
    hair: (usize, f64),
) -> (Vec<u8>, Option<Vec<u8>>) {
    let cc = crate::components::connected_components(mask, h, w, 8);
    let classes: Vec<bool> = (1..cc.count)
        .into_par_iter()
        .map(|i| {
            let (own, (_, _, bw, bh)) = component_crop(&cc.labels, w, i, cc.stats[i]);
            is_hair(&own, (bh, bw), cc.stats[i][4] as usize, hair)
        })
        .collect();
    let (mut compact, mut hairs) = (vec![0u8; h * w], vec![0u8; h * w]);
    let mut any_hair = false;
    for (p, &l) in cc.labels.iter().enumerate() {
        if l > 0 {
            if classes[l as usize - 1] {
                hairs[p] = 1;
                any_hair = true;
            } else {
                compact[p] = 1;
            }
        }
    }
    (compact, any_hair.then_some(hairs))
}

/// The scalars of `detect_luma_score`, as Python floats unless noted.
pub struct LumaDetect<'a> {
    pub hi: f64,
    pub hi_hair: f64,
    pub lo: f64,
    pub proxy_min: f32,
    pub reach_disk: crate::morph::Kernel<'a>,
    pub texture_knee: f32,
    pub texture_gain: f32,
    pub hair: (usize, f64),
    pub hair_fill_gain: f32,
    pub hair_texture: (f64, f64),
    pub texture_win: usize,
    pub pad_px: f64,
    pub score_floor: f64,
}

/// A fill score for specks and an inpaint mask for hairs, each `None` when empty.
pub type Detection = (Option<Vec<f32>>, Option<Vec<u8>>);

/// `detect_luma_score` from its stats planes: (score, hair mask).
pub fn detect_luma_score(
    stats: [&[f32]; 4],
    (h, w): (usize, usize),
    c: &LumaDetect,
) -> Option<Detection> {
    let [proxy, background, z, texture] = stats;
    let n = h * w;
    let seed_bar = c.hi.min(c.hi_hair) as f32;
    let (pmin, lo) = (c.proxy_min, c.lo as f32);
    let seeds: Vec<u8> = z
        .par_iter()
        .zip(proxy)
        .map(move |(&z, &p)| (z >= seed_bar && p > pmin) as u8)
        .collect();
    if !seeds.par_iter().any(|&s| s != 0) {
        return Some((None, None));
    }
    let mut near = vec![0u8; n];
    crate::morph::morphology_mask(
        &seeds,
        &mut near,
        (h, w, 1),
        crate::morph::Op::Dilate,
        crate::morph::Kernel { ..c.reach_disk },
        1,
        crate::filter::Border::Constant,
    )?;
    let grow: Vec<u8> = near
        .par_iter()
        .zip(z)
        .map(move |(&nv, &z)| (nv != 0 && z >= lo) as u8)
        .collect();
    let cc = crate::components::connected_components(&grow, h, w, 8);
    let lab = &cc.labels;
    let (hi32, knee, gain) = (c.hi as f32, c.texture_knee, c.texture_gain);
    let mut strong = vec![false; cc.count];
    let mut peak = vec![0f32; cc.count];
    for p in 0..n {
        if seeds[p] == 0 {
            continue;
        }
        let l = lab[p] as usize;
        let t = texture[p] - knee;
        let bar = hi32 * (1.0 + gain * if t.is_nan() || t > 0.0 { t } else { 0.0 });
        if z[p] >= bar {
            strong[l] = true;
        }
        if z[p] > peak[l] || z[p].is_nan() {
            peak[l] = if peak[l].is_nan() { peak[l] } else { z[p] };
        }
    }
    let candidates: Vec<usize> = (1..cc.count).filter(|&i| peak[i] > 0.0).collect();
    let hairy: Vec<bool> = candidates
        .par_iter()
        .map(|&i| {
            let (own, (_, _, bw, bh)) = component_crop(lab, w, i, cc.stats[i]);
            is_hair(&own, (bh, bw), cc.stats[i][4] as usize, c.hair)
        })
        .collect();
    let mut keep = vec![false; cc.count];
    let mut is_hair_label = vec![false; cc.count];
    for (&i, &hr) in candidates.iter().zip(&hairy) {
        if hr {
            is_hair_label[i] = true;
        } else {
            keep[i] = strong[i];
        }
    }
    if is_hair_label.iter().any(|&b| b) {
        let filled_src: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|p| {
                if is_hair_label[lab[p] as usize] {
                    background[p]
                } else {
                    proxy[p]
                }
            })
            .collect();
        let (mut mean, mut filled) = (vec![0f32; n], vec![0f32; n]);
        box_mean_std(&filled_src, (h, w), c.texture_win, &mut mean, &mut filled)?;
        let fill_gain = c.hair_fill_gain;
        let hair_tex: Vec<f32> = texture
            .par_iter()
            .zip(&filled)
            .map(move |(&t, &f)| {
                let g = fill_gain * f;
                if t.is_nan() || g.is_nan() {
                    f32::NAN
                } else if g < t {
                    g
                } else {
                    t
                }
            })
            .collect();
        let (hknee, hgain) = c.hair_texture;
        for (&i, _) in candidates.iter().zip(&hairy).filter(|(_, h)| **h) {
            let [bx, by, bw, bh, _] = cc.stats[i].map(|v| v as usize);
            let vals: Vec<f32> = (by..by + bh)
                .flat_map(|y| (bx..bx + bw).map(move |x| y * w + x))
                .filter(|&p| lab[p] == i as i32)
                .map(|p| hair_tex[p])
                .collect();
            let tex = crate::stats::median(&vals) as f64;
            let over = tex - hknee;
            // Python's max keeps a NaN first argument.
            let bar = c.hi_hair * (1.0 + hgain * if 0.0 > over { 0.0 } else { over });
            keep[i] = peak[i] >= bar as f32;
        }
    }
    if !keep.iter().any(|&k| k) {
        return Some((None, None));
    }
    let hit: Vec<u8> = lab.par_iter().map(|&l| keep[l as usize] as u8).collect();
    let (compact, hairs) = split_hairs(&hit, (h, w), c.hair);
    let score = compact
        .par_iter()
        .any(|&v| v != 0)
        .then(|| crate::distance::mask_to_score(&compact, h, w, c.pad_px, c.score_floor));
    Some((score, hairs))
}

/// `_scratch_ridge` from the density plane: the vertical band-pass (Gaussian of `fine` minus
/// Gaussian of `broad` sigma, down the columns only) over the box mean of its magnitude across
/// `win`, which is 0.8 sigma for Gaussian noise.
pub fn scratch_ridge(
    dens: &[f32],
    (h, w): (usize, usize),
    (fine, broad, win): (f64, f64, usize),
    dst: &mut [f32],
) -> Option<()> {
    use crate::filter::{Border, box_filter, gaussian_blur};
    let n = h * w;
    let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
    gaussian_blur(
        dens,
        &mut a,
        (h, w, 1),
        (1, 0),
        (0.0, fine),
        Border::Reflect101,
    )?;
    gaussian_blur(
        dens,
        &mut b,
        (h, w, 1),
        (1, 0),
        (0.0, broad),
        Border::Reflect101,
    )?;
    crate::par_zip(&mut a, &b, |r, &b| *r -= b);
    let abs: Vec<f32> = a.par_iter().map(|v| v.abs()).collect();
    box_filter(
        &abs,
        &mut b,
        (h, w, 1),
        (win, win),
        true,
        Border::Reflect101,
    )?;
    dst.par_chunks_mut(BLOCK)
        .zip(a.par_chunks(BLOCK).zip(b.par_chunks(BLOCK)))
        .for_each(|(o, (r, s))| {
            for ((o, &r), &s) in o.iter_mut().zip(r).zip(s) {
                let s = s / 0.8;
                *o = r / if s.is_nan() || s > 1e-6 { s } else { 1e-6 };
            }
        });
    Some(())
}

/// The strongest sheared row of `trace_scratch`'s slope search: (strength, slope index, row,
/// that row). For each slope the ridge band `z` is sheared about `cx` as `_shear_rows` does,
/// rows `rows` of it averaged and weighted by `pull`. The first slope wins a tie.
pub fn scratch_slope_search(
    z: &[f32],
    (h, w): (usize, usize),
    cx: f64,
    rows: std::ops::Range<usize>,
    pull: &[f64],
    slopes: &[f64],
) -> Option<(f64, usize, usize, Vec<f32>)> {
    use crate::warp::{Border, Image, Interp, warp_affine};
    if rows.end > h || pull.len() != rows.len() || z.len() != h * w {
        return None;
    }
    let img = Image {
        data: z,
        h,
        w,
        cn: 1,
    };
    let found: Vec<(f64, usize, Vec<f32>)> = slopes
        .par_iter()
        .map(|&slope| {
            // The matrix goes through float32, as `np.float32([...])` builds it.
            let m = [
                1.0,
                0.0,
                0.0,
                -slope as f32 as f64,
                1.0,
                (slope * cx) as f32 as f64,
            ];
            let mut sheared = vec![0f32; h * w];
            warp_affine(
                &img,
                (Interp::Linear, Border::Replicate),
                m,
                false,
                w,
                &mut sheared,
            );
            let mut best = (f64::NEG_INFINITY, 0usize);
            for (i, (y, &p)) in rows.clone().zip(pull).enumerate() {
                let row = &sheared[y * w..(y + 1) * w];
                let mean = crate::stats::pairwise_sum_f32(row) / w as f32;
                let s = mean.abs() as f64 * p;
                if s > best.0 || (s.is_nan() && !best.0.is_nan()) {
                    best = (s, i);
                }
            }
            let y = rows.start + best.1;
            (best.0, best.1, sheared[y * w..(y + 1) * w].to_vec())
        })
        .collect();
    let mut best: Option<(f64, usize, usize, Vec<f32>)> = None;
    for (i, (s, k, row)) in found.into_iter().enumerate() {
        if best.as_ref().is_none_or(|b| s > b.0) {
            best = Some((s, i, k, row));
        }
    }
    best
}

/// One clone stroke's geometry, as `apply_clone_stroke` derives it.
pub struct CloneStroke<'a> {
    /// The crop it works in: x0, y0, x1, y1.
    pub rect: (usize, usize, usize, usize),
    /// Its smoothed path, in crop pixels.
    pub path: &'a [(i64, i64)],
    pub brush: i64,
    pub radius: f64,
    pub shift: (i64, i64),
    pub sigma: f64,
    pub feather: f64,
    pub strength: f64,
    pub match_tone: bool,
}

/// `_local_mean(plane, weight, sigma)` of a 3-channel crop: a Gaussian mean over the pixels the
/// weight admits.
fn local_mean(
    plane: &[f32],
    weight: &[f32],
    (h, w): (usize, usize),
    sigma: f64,
) -> Option<Vec<f32>> {
    use crate::filter::{Border, gaussian_blur};
    let weighted: Vec<f32> = plane
        .as_chunks::<3>()
        .0
        .iter()
        .zip(weight)
        .map(|(p, &k)| [p[0] * k, p[1] * k, p[2] * k])
        .collect::<Vec<_>>()
        .into_flattened();
    let (mut num, mut den) = (vec![0f32; 3 * h * w], vec![0f32; h * w]);
    gaussian_blur(
        &weighted,
        &mut num,
        (h, w, 3),
        (0, 0),
        (sigma, sigma),
        Border::Reflect101,
    )?;
    gaussian_blur(
        weight,
        &mut den,
        (h, w, 1),
        (0, 0),
        (sigma, sigma),
        Border::Reflect101,
    )?;
    for (n, &d) in num.as_chunks_mut::<3>().0.iter_mut().zip(&den) {
        let d = if d.is_nan() || d > 1e-6 { d } else { 1e-6 };
        for v in n {
            *v /= d;
        }
    }
    Some(num)
}

/// `apply_clone_stroke` in place on an HxWx3 image: the source patch, tone-matched to the
/// destination's surround when asked, blended over the brush footprint.
pub fn clone_stroke(out: &mut [f32], (h, w): (usize, usize), s: &CloneStroke) -> Option<()> {
    use crate::distance::{Metric, distance_transform};
    let (x0, y0, x1, y1) = s.rect;
    let (ch, cw) = (y1 - y0, x1 - x0);
    let mut cover = vec![0u8; ch * cw];
    crate::shapes::stroke_cover(&mut cover, cw, ch, s.path, s.brush, 2 * s.brush);
    if !cover.iter().any(|&c| c != 0) {
        return Some(());
    }
    let (sx, sy) = s.shift;
    let (hi, wi) = (h as i64, w as i64);
    let mut valid = vec![false; ch * cw];
    let (mut src, mut dst) = (vec![0f32; 3 * ch * cw], vec![0f32; 3 * ch * cw]);
    for y in 0..ch {
        let yy = (y0 + y) as i64 + sy;
        let ys = yy.clamp(0, hi - 1) as usize;
        for x in 0..cw {
            let xx = (x0 + x) as i64 + sx;
            let xs = xx.clamp(0, wi - 1) as usize;
            let p = y * cw + x;
            valid[p] = (0..hi).contains(&yy) && (0..wi).contains(&xx);
            src[3 * p..3 * p + 3].copy_from_slice(&out[3 * (ys * w + xs)..3 * (ys * w + xs) + 3]);
            let d = 3 * ((y0 + y) * w + x0 + x);
            dst[3 * p..3 * p + 3].copy_from_slice(&out[d..d + 3]);
        }
    }
    let mut alpha: Vec<f32> = if s.feather > 0.0 {
        let depth = distance_transform(&cover, ch, cw, Metric::L2Mask3);
        let deepest = depth.iter().fold(0f32, |m, &v| m.max(v)) as f64;
        // Python's min and max: the first argument unless the second is strictly past it.
        let reach = if deepest < s.radius {
            deepest
        } else {
            s.radius
        };
        let denom = s.feather * reach;
        let denom = (if 1e-6 > denom { 1e-6 } else { denom }) as f32;
        depth
            .iter()
            .map(|&d| {
                let v = d / denom;
                let v = if v.is_nan() || v > 0.0 { v } else { 0.0 };
                if v.is_nan() || v < 1.0 { v } else { 1.0 }
            })
            .collect()
    } else {
        cover.iter().map(|&c| c as f32).collect()
    };
    // `alpha *= strength * valid` multiplies by a float64 array.
    for (a, &v) in alpha.iter_mut().zip(&valid) {
        *a = (*a as f64 * (s.strength * v as u8 as f64)) as f32;
    }
    let mut patch = src.clone();
    if s.match_tone {
        let ring: Vec<f32> = cover
            .iter()
            .zip(&valid)
            .map(|(&c, &v)| (c == 0 && v) as u8 as f32)
            .collect();
        if ring.iter().any(|&r| r != 0.0) {
            let near = local_mean(&dst, &ring, (ch, cw), s.sigma)?;
            let far = local_mean(&src, &ring, (ch, cw), s.sigma)?;
            for ((p, &n), &f) in patch.iter_mut().zip(&near).zip(&far) {
                let gain = n / if f.is_nan() || f > 1e-6 { f } else { 1e-6 };
                let gain = if gain.is_nan() || gain > 1.0 / 64.0 {
                    gain
                } else {
                    1.0 / 64.0
                };
                let gain = if gain.is_nan() || gain < 64.0 {
                    gain
                } else {
                    64.0
                };
                *p *= gain;
            }
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            let p = y * cw + x;
            let a = alpha[p];
            let d = 3 * ((y0 + y) * w + x0 + x);
            for c in 0..3 {
                out[d + c] = dst[3 * p + c] * (1.0 - a) + patch[3 * p + c] * a;
            }
        }
    }
    Some(())
}

/// `drop_exclusions`' release of a score under the excluded footprint: the score lifted to 1
/// over `rim` pixels of distance inside the footprint. Returns whether anything stays under 1.
pub fn release_score(score: &mut [f32], cover: &[u8], (h, w): (usize, usize), rim: f32) -> bool {
    // The distance only matters inside the footprint, and a one-pixel ring of zeros around its
    // box holds every pixel's nearest zero, so the transform runs on that box alone.
    let rows: Vec<usize> = (0..h)
        .filter(|&y| cover[y * w..(y + 1) * w].iter().any(|&c| c != 0))
        .collect();
    if let (Some(&top), Some(&bottom)) = (rows.first(), rows.last()) {
        let cols: Vec<usize> = (0..w)
            .filter(|&x| (top..=bottom).any(|y| cover[y * w + x] != 0))
            .collect();
        let (y0, y1) = (top.saturating_sub(1), (bottom + 2).min(h));
        let (x0, x1) = (cols[0].saturating_sub(1), (cols[cols.len() - 1] + 2).min(w));
        let (ch, cw) = (y1 - y0, x1 - x0);
        let crop: Vec<u8> = (y0..y1)
            .flat_map(|y| &cover[y * w + x0..y * w + x1])
            .copied()
            .collect();
        let d =
            crate::distance::distance_transform(&crop, ch, cw, crate::distance::Metric::L2Mask3);
        for y in 0..ch {
            for x in 0..cw {
                let a = d[y * cw + x] / rim;
                let a = if a.is_nan() || a > 0.0 { a } else { 0.0 };
                let a = if a.is_nan() || a < 1.0 { a } else { 1.0 };
                let s = &mut score[(y0 + y) * w + x0 + x];
                *s += a * (1.0 - *s);
            }
        }
    }
    score.par_iter().any(|&s| s < 1.0)
}

/// The scalars of `strokes_to_score`, as Python floats.
pub struct HealSearch {
    pub win: usize,
    pub z_min: f64,
    pub z_hi: f64,
    pub z_grow: f64,
    pub z_grow_frac: f64,
    pub pad_px: f64,
    pub score_floor: f64,
    pub rim: f32,
}

/// One stroke of `strokes_to_score` on its crop: the score of what stands out from the film
/// inside the brush footprint (a two-sided local z-score on density, grown by hysteresis),
/// softened at the footprint's rim. `None` when the brush found nothing.
pub fn heal_region(
    dens: &[f32],
    (h, w): (usize, usize),
    path: &[(i64, i64)],
    (radius, thickness): (i64, i64),
    c: &HealSearch,
) -> Option<Vec<f32>> {
    use crate::filter::{Border, box_filter};
    let n = h * w;
    let mut cover = vec![0u8; n];
    crate::shapes::stroke_cover(&mut cover, w, h, path, radius, thickness);
    if !cover.iter().any(|&v| v != 0) {
        return None;
    }
    let mut local = vec![0f32; n];
    box_filter(
        dens,
        &mut local,
        (h, w, 1),
        (c.win, c.win),
        true,
        Border::Reflect101,
    )?;
    let diff: Vec<f32> = dens.iter().zip(&local).map(|(&d, &l)| d - l).collect();
    let mut detail = vec![0f32; n];
    box_filter(
        &diff,
        &mut detail,
        (h, w, 1),
        (3, 3),
        true,
        Border::Reflect101,
    )?;
    let abs: Vec<f32> = detail.iter().map(|v| v.abs()).collect();
    let sigma = crate::stats::median(&abs) as f64 / 0.6745;
    let sigma = (if 1e-9 > sigma { 1e-9 } else { sigma }) as f32;
    let z: Vec<f32> = abs.iter().map(|&a| a / sigma).collect();
    let peak =
        z.iter()
            .zip(&cover)
            .filter(|&(_, &c)| c != 0)
            .fold(
                f32::NEG_INFINITY,
                |m, (&z, _)| if z > m || z.is_nan() { z } else { m },
            ) as f64;
    if peak < c.z_min {
        return None;
    }
    let hi = if peak >= c.z_hi { c.z_hi } else { peak * 0.9 };
    let lo = if hi * c.z_grow_frac > c.z_grow {
        hi * c.z_grow_frac
    } else {
        c.z_grow
    };
    let (hi32, lo32) = (hi as f32, lo as f32);
    let strong: Vec<bool> = z
        .iter()
        .zip(&cover)
        .map(|(&z, &c)| c != 0 && z >= hi32)
        .collect();
    if !strong.iter().any(|&s| s) {
        return None;
    }
    let grow: Vec<u8> = z
        .iter()
        .zip(&cover)
        .map(|(&z, &c)| (c != 0 && z >= lo32) as u8)
        .collect();
    let cc = crate::components::connected_components(&grow, h, w, 8);
    let mut seeded = vec![false; cc.count];
    for (p, &s) in strong.iter().enumerate() {
        if s {
            seeded[cc.labels[p] as usize] = true;
        }
    }
    seeded[0] = false;
    let keep: Vec<u8> = cc
        .labels
        .iter()
        .map(|&l| seeded[l as usize] as u8)
        .collect();
    let region = crate::distance::mask_to_score(&keep, h, w, c.pad_px, c.score_floor);
    let d = crate::distance::distance_transform(&cover, h, w, crate::distance::Metric::L2Mask3);
    Some(
        region
            .iter()
            .zip(&d)
            .map(|(&r, &d)| {
                let a = d / c.rim;
                let a = if a.is_nan() || a > 0.0 { a } else { 0.0 };
                let a = if a.is_nan() || a < 1.0 { a } else { 1.0 };
                1.0 - a * (1.0 - r)
            })
            .collect(),
    )
}

/// `np.log(np.clip(normalize_ir(img[:, :, c]), 1e-4, 1.0))` stacked over the three channels,
/// NumPy's float32 log included.
pub fn ir_vis_log(img: &[f32], (h, w): (usize, usize), win: usize, dst: &mut [f32]) -> Option<()> {
    let n = h * w;
    let mut plane = vec![0f32; n];
    let mut norm = vec![0f32; n];
    for c in 0..3 {
        plane
            .par_iter_mut()
            .enumerate()
            .for_each(|(p, v)| *v = img[3 * p + c]);
        normalize_ir(&plane, None, (h, w), win, &mut norm)?;
        dst.par_chunks_mut(3 * BLOCK)
            .zip(norm.par_chunks(BLOCK))
            .for_each(|(o, r)| {
                for (o, &v) in o.as_chunks_mut::<3>().0.iter_mut().zip(r) {
                    let v = if v.is_nan() || v > 1e-4 { v } else { 1e-4 };
                    let v = if v.is_nan() || v < 1.0 { v } else { 1.0 };
                    o[c] = crate::mathf::np_logf(v);
                }
            });
    }
    Some(())
}

/// `ir_ratio_and_gain`'s gain: `min(clamp, (clip(ratio / identity, 1e-4, 1)) ** -gamma)` per
/// channel, then held at the local clean base over the pixel, floored at 1. The power is
/// `mathf`'s; NumPy's float32 power runs SVML on an AVX-512 CPU.
pub fn ir_gain(
    ratio: &[f32],
    img: &[f32],
    clean: &[f32],
    gammas: [f64; 3],
    (identity, clamp): (f32, f32),
    dst: &mut [f32],
) {
    let neg = gammas.map(|g| -g as f32);
    dst.par_chunks_mut(3 * BLOCK)
        .zip(ratio.par_chunks(BLOCK))
        .zip(img.par_chunks(3 * BLOCK).zip(clean.par_chunks(3 * BLOCK)))
        .for_each(|((o, r), (px, cl))| {
            for (i, &r) in r.iter().enumerate() {
                let b = r / identity;
                let b = if b.is_nan() || b > 1e-4 { b } else { 1e-4 };
                let b = if b.is_nan() || b < 1.0 { b } else { 1.0 };
                for (c, &e) in neg.iter().enumerate() {
                    let j = 3 * i + c;
                    let p = crate::mathf::pow_f64(b as f64, e as f64) as f32;
                    let g = if clamp.is_nan() || p.is_nan() {
                        f32::NAN
                    } else if p < clamp {
                        p
                    } else {
                        clamp
                    };
                    let d = px[j];
                    let d = if d.is_nan() || d > 1e-5 { d } else { 1e-5 };
                    let cap = cl[j] / d;
                    let cap = if cap.is_nan() || cap > 1.0 { cap } else { 1.0 };
                    o[j] = if g.is_nan() || cap.is_nan() {
                        f32::NAN
                    } else if cap < g {
                        cap
                    } else {
                        g
                    };
                }
            }
        });
}
