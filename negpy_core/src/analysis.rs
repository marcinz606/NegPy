//! Ports of the frame analysis in negpy/features/geometry/logic.py and exposure/normalization.py.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::filter::Border;
use crate::stats::{np_median, np_percentiles, pairwise_sum_f32, percentiles};

const LUMA_R: f64 = 0.2126;
const LUMA_G: f64 = 0.7152;
const LUMA_B: f64 = 0.0722;

/// `kernel.image.logic.get_luminance` of an HxWx3 float32 image: Numba's fastmath kernel,
/// f64 with fused multiply-adds, then float32.
pub fn luminance(rgb: &[f32], dst: &mut [f32]) {
    dst.par_chunks_mut(4096)
        .zip(rgb.par_chunks(4096 * 3))
        .for_each(|(d, p)| luminance_chunk(p, d));
}

/// `mul_add` is one instruction only where the CPU has FMA, so this compiles twice.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
pub(crate) fn luminance_chunk(rgb: &[f32], dst: &mut [f32]) {
    for (d, p) in dst.iter_mut().zip(rgb.as_chunks::<3>().0) {
        let t = LUMA_R * p[0] as f64;
        let t = LUMA_G.mul_add(p[1] as f64, t);
        *d = LUMA_B.mul_add(p[2] as f64, t) as f32;
    }
}

/// An HxW plane as HxWx3, or an HxWx3 image as it is.
pub fn rgb(img: &[f32], cn: usize) -> std::borrow::Cow<'_, [f32]> {
    if cn == 1 {
        img.iter()
            .flat_map(|&v| [v, v, v])
            .collect::<Vec<_>>()
            .into()
    } else {
        img.into()
    }
}

/// `_detection_luma` of the luminance of an HxW or HxWx3 image whose `memo::key` is `key`;
/// None for an empty one.
pub fn detection_plane(
    img: &[f32],
    h: usize,
    w: usize,
    cn: usize,
    key: u128,
) -> Option<std::sync::Arc<Vec<f32>>> {
    static LAST: crate::memo::Memo<Option<std::sync::Arc<Vec<f32>>>, 4> = crate::memo::Memo::new();
    LAST.get_or(key, || {
        let mut lum = vec![0f32; h * w];
        luminance(&rgb(img, cn), &mut lum);
        let mut det = vec![0f32; h * w];
        detection_luma(&lum, &mut det)?;
        Some(det.into())
    })
}

/// `_normalize_to_uint8` for a float32 image: the 1st and 99th percentiles of its finite
/// values stretched to 0..255, NumPy's float32 arithmetic (NEP 50) and truncating cast.
pub fn normalize_to_uint8(img: &[f32], dst: &mut [u8]) {
    let finite = img.iter().fold(true, |all, v| all & v.is_finite());
    let valid: std::borrow::Cow<[f32]> = if finite {
        img.into()
    } else {
        img.iter().copied().filter(|v| v.is_finite()).collect()
    };
    let Some(p) = percentiles(&valid, &[1.0, 99.0]) else {
        dst.fill(0);
        return;
    };
    let low = p[0] as f64;
    let high = if p[1] as f64 <= low {
        low + 1.0
    } else {
        p[1] as f64
    };
    let (low32, scale32) = (low as f32, (255.0 / (high - low)) as f32);
    crate::par_zip(dst, img, move |d, &v| {
        let s = ((v - low32) * scale32).clamp(0.0, 255.0);
        // NumPy's float32 to uint8 cast truncates, NaN to 0 on x86; Rust's `as` does the same.
        *d = s as u8;
    });
}

/// `_detection_luma` of a float32 luminance plane: divided by its 99.5th percentile and
/// clipped to 0..2, in float32.
pub fn detection_luma(lum: &[f32], dst: &mut [f32]) -> Option<()> {
    // Python's max(nan, 1e-6) keeps the NaN.
    let p = np_percentiles(lum, &[99.5])?[0] as f64;
    let anchor = if p.is_nan() {
        f32::NAN
    } else {
        p.max(1e-6) as f32
    };
    crate::par_zip(dst, lum, move |d, &v| {
        *d = (v / anchor).clamp(0.0, 2.0);
    });
    Some(())
}

/// `to_log_density`: log10 of the value clamped to 1e-6..1, NaN to 1e-6.
pub fn to_log_density(img: &[f32], dst: &mut [f32]) {
    dst.par_chunks_mut(1 << 14)
        .zip(img.par_chunks(1 << 14))
        .for_each(|(d, s)| log_density_block(s, d));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
#[allow(clippy::manual_clamp)]
fn log_density_block(src: &[f32], dst: &mut [f32]) {
    for (d, &v) in dst.iter_mut().zip(src) {
        // max drops NaN, as np.fmax does; clamp would keep it.
        *d = crate::mathf::log10f(v.max(1e-6).min(1.0));
    }
}

/// The normalization processors' log density: log10 of the value clamped below at 1e-6, with
/// NaN and -inf at 1e-6 and +inf at 1 (`np.nan_to_num`), no upper clamp.
pub fn log_density_open(img: &[f32], dst: &mut [f32]) {
    dst.par_chunks_mut(1 << 14)
        .zip(img.par_chunks(1 << 14))
        .for_each(|(d, s)| log_density_open_block(s, d));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn log_density_open_block(src: &[f32], dst: &mut [f32]) {
    for (d, &v) in dst.iter_mut().zip(src) {
        let v = if v == f32::INFINITY { 1.0 } else { v };
        // max drops NaN, as nan_to_num then np.clip do.
        *d = crate::mathf::log10f(v.max(1e-6));
    }
}

/// `normalize_log_image`: `(v - floor) / (ceil - floor)` per channel of an HxWx3 log image,
/// unclamped. A span under 1e-6 divides by +-1e-6, by the span's sign.
pub fn normalize_log_image(img: &[f32], floors: [f32; 3], ceils: [f32; 3], dst: &mut [f32]) {
    let denom = std::array::from_fn::<f32, 3, _>(|c| {
        let d = ceils[c] - floors[c];
        // Numba types the span float64 here; float32 division of the difference rounds the same.
        if (d.abs() as f64) < 1e-6 {
            if d >= 0.0 { 1e-6 } else { -1e-6 }
        } else {
            d
        }
    });
    crate::par_zip(
        dst.as_chunks_mut::<3>().0,
        img.as_chunks::<3>().0,
        move |d, p| {
            for c in 0..3 {
                d[c] = ((p[c] - floors[c]) as f64 / denom[c] as f64) as f32;
            }
        },
    );
}

/// `_block_median_grid` for block size `b` over the whole `b` x `b` blocks of an HxWxC image:
/// the median of each block per channel, (H/b)x(W/b)xC. A block of 2x2 is (sum - min - max) / 2,
/// the sum in float64.
pub fn block_median_grid(img: &[f32], (h, w, c): (usize, usize, usize), b: usize) -> Vec<f32> {
    let (gh, gw) = (h / b, w / b);
    let mut out = vec![0f32; gh * gw * c];
    if gh == 0 || gw == 0 {
        return out;
    }
    let at = |y: usize, x: usize, ch: usize| img[(y * w + x) * c + ch];
    out.par_chunks_mut(gw * c)
        .enumerate()
        .for_each(|(gy, line)| {
            let mut block = Vec::with_capacity(b * b);
            for gx in 0..gw {
                for ch in 0..c {
                    let (y, x) = (gy * b, gx * b);
                    line[gx * c + ch] = if b == 2 {
                        let (p00, p01, p10, p11) = (
                            at(y, x, ch),
                            at(y, x + 1, ch),
                            at(y + 1, x, ch),
                            at(y + 1, x + 1, ch),
                        );
                        let s = p00 as f64 + p01 as f64 + p10 as f64 + p11 as f64;
                        let mn = np_min(np_min(p00, p01), np_min(p10, p11));
                        let mx = np_max(np_max(p00, p01), np_max(p10, p11));
                        ((s - mn as f64 - mx as f64) * 0.5) as f32
                    } else {
                        block.clear();
                        for yy in y..y + b {
                            block.extend((x..x + b).map(|xx| at(yy, xx, ch)));
                        }
                        np_median(&mut block)
                    };
                }
            }
        });
    out
}

/// np.minimum with NaN propagation.
fn np_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

/// `_get_threshold_autocrop_coords` on an HxW luma plane: the first and last rows and columns
/// whose float32 mean is under `threshold` (row means pairwise, column means summed down in
/// order, as NumPy sums them), or None with fewer than ten of either.
pub fn threshold_box(
    lum: &[f32],
    h: usize,
    w: usize,
    threshold: f32,
) -> Option<(usize, usize, usize, usize)> {
    let rows: Vec<usize> = (0..h)
        .filter(|&y| pairwise_sum_f32(&lum[y * w..(y + 1) * w]) / (w as f32) < threshold)
        .collect();
    let mut sums = vec![0f32; w];
    for row in lum.chunks_exact(w) {
        sums.iter_mut().zip(row).for_each(|(s, &v)| *s += v);
    }
    let cols: Vec<usize> = (0..w)
        .filter(|&x| sums[x] / (h as f32) < threshold)
        .collect();
    if rows.len() < 10 || cols.len() < 10 {
        return None;
    }
    Some((rows[0], rows[rows.len() - 1], cols[0], cols[cols.len() - 1]))
}

/// `_vertical_edge_profile` of batch autocrop for an HxW or HxWxC float32 image: the 80th
/// percentile of each column's |Sobel x| on the 2..98 percentile stretch of the channel mean,
/// blurred along the row and scaled to its peak, and its 99.5 - 50 percentile contrast.
pub fn batch_edge_profile(img: &[f32], h: usize, w: usize, cn: usize) -> (Vec<f32>, f64) {
    let gray: Vec<f32> = if cn == 1 {
        img.to_vec()
    } else {
        img.chunks_exact(cn)
            .map(|p| (p[0] + p[1] + p[2]) / 3.0)
            .collect()
    };
    let finite: Vec<f32> = gray.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || w < 2 {
        return (vec![0.0; w], 0.0);
    }
    let p = crate::stats::np_percentiles_f64(&finite, &[2.0, 98.0]);
    let (low, span) = (p[0], (p[1] - p[0]).max(1e-6));
    let normalized: Vec<f32> = gray
        .iter()
        .map(|&v| ((v as f64 - low) / span).clamp(0.0, 1.0) as f32)
        .collect();
    let mut grad = vec![0f32; h * w];
    let _ = crate::filter::sep_filter2d(
        &normalized,
        &mut grad,
        (h, w, 1),
        (&[-1.0, 0.0, 1.0], &[1.0, 2.0, 1.0]),
        Border::Reflect101,
    );
    let columns: Vec<f32> = (0..w)
        .into_par_iter()
        .map(|x| {
            let col: Vec<f32> = (0..h).map(|y| grad[y * w + x].abs()).collect();
            np_percentiles(&col, &[80.0]).map_or(f32::NAN, |p| p[0])
        })
        .collect();
    let window = {
        let k = 5.max((w as f64 * 0.012).round_ties_even() as usize);
        if k % 2 == 0 { k + 1 } else { k }
    };
    let mut profile = vec![0f32; w];
    let _ = crate::filter::gaussian_blur(
        &columns,
        &mut profile,
        (1, w, 1),
        (window, 1),
        (0.0, 0.0),
        Border::Reflect101,
    );
    let peak = profile.iter().copied().fold(f32::NEG_INFINITY, np_max);
    let pct = |q: f64| np_percentiles(&profile, &[q]).map_or(f32::NAN, |p| p[0]) as f64;
    let contrast = 0f64.max(pct(99.5) - pct(50.0));
    if peak > 0.0 {
        profile.iter_mut().for_each(|v| *v /= peak);
    }
    (profile, contrast)
}

/// The end of `measure_base_log`: per-channel log10 of the base blocks' median, or None when
/// the base is not brighter than the picture inside `gate` (median of the float32 pixel means,
/// summed left to right).
pub fn base_log_from_blocks(
    det: &[f32],
    w: usize,
    (gy1, gy2, gx1, gx2): (usize, usize, usize, usize),
    base: &[[f32; 3]],
) -> Option<[f64; 3]> {
    let mean = |p: &[f32]| (p[0] + p[1] + p[2]) / 3.0;
    let mut picture = Vec::with_capacity((gy2 - gy1) * (gx2 - gx1));
    for y in gy1..gy2 {
        picture.extend(
            det[(y * w + gx1) * 3..(y * w + gx2) * 3]
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| mean(p)),
        );
    }
    let mut base_means: Vec<f32> = base.iter().map(|p| mean(p)).collect();
    // `<=` is false for NaN, so a NaN median keeps the base.
    if np_median(&mut base_means) <= np_median(&mut picture) {
        return None;
    }
    Some(std::array::from_fn(|c| {
        let mut col: Vec<f32> = base.iter().map(|p| p[c]).collect();
        libm::log10((np_median(&mut col) as f64).max(1e-6))
    }))
}

/// `film_and_picture(det)` for an HxWx3 detection image: the film box (None when the scan
/// shows no film edge) and the picture gate inside it, or over the whole frame without a box; a
/// degenerate gate falls back to the whole frame. None when the frame is too small to refine.
pub fn film_and_picture(det: &[f32], h: usize, w: usize) -> Option<(Option<Roi>, Roi)> {
    let film = crate::detect::detect_film_bounds(det, h, w, 3)
        .consensus
        .map(|c| c.roi);
    let whole = (0, h as i64, 0, w as i64);
    let (gate, _) = refine_roi_to_image(det, h, w, 3, film.unwrap_or(whole))?;
    let gate = if gate.1 <= gate.0 || gate.3 <= gate.2 {
        whole
    } else {
        gate
    };
    Some((film, gate))
}

// The band inside the film box, per side, that holds rebate, perforations and picture edge.
const BASE_MARGIN: f64 = 0.15;

/// `measure_base_log(image)` on its HxWx3 detection image: per-channel log10 transmittance of
/// the clear base, the median of the flat blocks in the film box's margin band. None without a
/// film box, with too few flat blocks, or when the base is darker than the picture, because a
/// negative's base is its thinnest film.
pub fn measure_base_log(det: &[f32], h: usize, w: usize) -> Option<[f64; 3]> {
    let (film, gate) = film_and_picture(det, h, w)?;
    let (fy1, fy2, fx1, fx2) = film?;
    let (my, mx) = (
        ((fy2 - fy1) as f64 * BASE_MARGIN) as i64,
        ((fx2 - fx1) as f64 * BASE_MARGIN) as i64,
    );
    let clamp = |v: i64, n: usize| v.clamp(0, n as i64) as usize;
    let mut band = vec![0u8; h * w];
    for y in clamp(fy1, h)..clamp(fy2, h) {
        band[y * w + clamp(fx1, w)..y * w + clamp(fx2, w).max(clamp(fx1, w))].fill(1);
    }
    let (cy1, cy2, cx1, cx2) = (
        clamp(fy1 + my, h),
        clamp(fy2 - my, h),
        clamp(fx1 + mx, w),
        clamp(fx2 - mx, w),
    );
    for y in cy1..cy2.max(cy1) {
        band[y * w + cx1..y * w + cx2.max(cx1)].fill(0);
    }
    let base = base_blocks(det, &band, h, w)?;
    let gate = (
        clamp(gate.0, h),
        clamp(gate.1, h),
        clamp(gate.2, w),
        clamp(gate.3, w),
    );
    base_log_from_blocks(det, w, gate, &base)
}

const BASE_BLOCK: usize = 8;
const CLIP_LEVEL: f32 = 0.98;
const FLAT_CV: f32 = 0.03;
const MIN_BASE_BLOCKS: usize = 5;

/// The first half of `_base_blocks`: per-channel means of the 8x8 blocks of `det` (HxWx3) that
/// lie in `band`, hold no clipped pixel and are flat, in block order; None with fewer than
/// five. NumPy's order: each block sums row by row, the 8 rows outer, from +0.
fn flat_block_means(det: &[f32], band: &[u8], h: usize, w: usize) -> Option<Vec<[f32; 3]>> {
    let b = BASE_BLOCK;
    let (hb, wb) = (h / b, w / b);
    let rows: Vec<Vec<[f32; 3]>> = (0..hb)
        .into_par_iter()
        .map(|by| {
            let mut out = Vec::new();
            for bx in 0..wb {
                let px = |i1: usize, i3: usize| {
                    let o = ((by * b + i1) * w + bx * b + i3) * 3;
                    [det[o], det[o + 1], det[o + 2]]
                };
                let mut sum = [0f32; 3];
                let mut top = f32::NEG_INFINITY;
                let mut count = 0u32;
                for i1 in 0..b {
                    for i3 in 0..b {
                        let p = px(i1, i3);
                        for c in 0..3 {
                            sum[c] += p[c];
                            // np.max propagates NaN, so a block with one is never unclipped.
                            top = if p[c].is_nan() || top.is_nan() {
                                f32::NAN
                            } else {
                                top.max(p[c])
                            };
                        }
                        count += (band[(by * b + i1) * w + bx * b + i3] != 0) as u32;
                    }
                }
                let n = (b * b) as f32;
                let mean = sum.map(|v| v / n);
                let mut sq = [0f32; 3];
                for i1 in 0..b {
                    for i3 in 0..b {
                        let p = px(i1, i3);
                        for c in 0..3 {
                            let d = p[c] - mean[c];
                            sq[c] += d * d;
                        }
                    }
                }
                let std = sq.map(|v| (v / n).sqrt());
                let lum = ((mean[0] + mean[1]) + mean[2]) / 3.0;
                let cv = ((std[0] + std[1]) + std[2]) / 3.0 / lum.max(1e-6);
                let inside = count as f64 / (b * b) as f64 >= 0.9;
                if inside && top < CLIP_LEVEL && lum > 1e-4 && cv < FLAT_CV {
                    out.push(mean);
                }
            }
            out
        })
        .collect();
    let flat: Vec<[f32; 3]> = rows.into_iter().flatten().collect();
    (flat.len() >= MIN_BASE_BLOCKS).then_some(flat)
}

const GROUP_GAP: f32 = 0.04;

/// `_base_blocks`: the flat blocks grouped by log brightness, brightest first, split where two
/// neighbours differ by more than `GROUP_GAP` decades; the largest group of at least
/// max(5, 2%) blocks. log10 is libm's; ties sort by block index.
pub fn base_blocks(det: &[f32], band: &[u8], h: usize, w: usize) -> Option<Vec<[f32; 3]>> {
    let cand = flat_block_means(det, band, h, w)?;
    let log_lum: Vec<f32> = cand
        .iter()
        .map(|m| libm::log10f((((m[0] + m[1]) + m[2]) / 3.0).max(1e-6)))
        .collect();
    let mut order: Vec<usize> = (0..cand.len()).collect();
    order.sort_by(|&a, &b| log_lum[a].total_cmp(&log_lum[b]));
    order.reverse();
    let mut groups: Vec<&[usize]> = Vec::new();
    let mut start = 0;
    for i in 1..=order.len() {
        if i == order.len() || log_lum[order[i - 1]] - log_lum[order[i]] > GROUP_GAP {
            groups.push(&order[start..i]);
            start = i;
        }
    }
    let min_n = MIN_BASE_BLOCKS.max((0.02 * order.len() as f64) as usize);
    let mut best: Option<&[usize]> = None;
    for g in groups.into_iter().filter(|g| g.len() >= min_n) {
        if best.is_none_or(|b| g.len() > b.len()) {
            best = Some(g);
        }
    }
    best.map(|g| g.iter().map(|&i| cand[i]).collect())
}

/// A copy of `img[y1:y2, x1:x2]` of an HxW plane, row by row.
pub(crate) fn crop(
    img: &[f32],
    w: usize,
    (y1, y2, x1, x2): (usize, usize, usize, usize),
) -> Vec<f32> {
    let mut out = Vec::with_capacity(y2.saturating_sub(y1) * x2.saturating_sub(x1));
    for y in y1..y2 {
        out.extend_from_slice(&img[y * w + x1..y * w + x2]);
    }
    out
}

/// Python's round() of a float: half to even.
fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `_find_rebate_level(lum, film_roi, bed)`: the brightest uniform rebate plateau on a side of
/// the film box, (level, spread), when two opposite sides show one.
pub fn find_rebate_level(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (usize, usize, usize, usize),
    bed: Option<f64>,
) -> Option<(f64, f64)> {
    let (y1, y2, x1, x2) = (roi.0.min(h), roi.1.min(h), roi.2.min(w), roi.3.min(w));
    let (bh, bw) = (y2.saturating_sub(y1), x2.saturating_sub(x1));
    if bh < 16 || bw < 16 {
        return None;
    }
    let bed = match bed {
        Some(b) => b,
        None => np_percentiles(lum, &[99.0])?[0] as f64,
    };
    let boxed = crop(lum, w, (y1, y2, x1, x2));
    let box_median = np_percentiles(&boxed, &[50.0])?[0] as f64;
    let ring = py_round(0.04 * bh.min(bw) as f64).max(3) as usize;
    let ring_h = ring.min(bh);
    let ring_w = ring.min(bw);
    // top, bottom, left, right: NumPy slices of the box, in its row-major order.
    let sides = [
        crop(&boxed, bw, (0, ring_h, 0, bw)),
        crop(&boxed, bw, (bh - ring_h, bh, 0, bw)),
        crop(&boxed, bw, (0, bh, 0, ring_w)),
        crop(&boxed, bw, (0, bh, bw - ring_w, bw)),
    ];
    // NEP 50: the Python float threshold joins the float32 comparison as float32.
    let cut = (bed - 0.05) as f32;
    let mut qualifying: [Option<(f64, f64)>; 4] = [None; 4];
    for (slot, strip) in qualifying.iter_mut().zip(&sides) {
        let vals: Vec<f32> = strip.iter().copied().filter(|&v| v < cut).collect();
        if (vals.len() as f64) < 0.25 * strip.len() as f64 {
            continue;
        }
        let Some(p) = percentiles(&vals, &[80.0, 20.0, 60.0]) else {
            continue;
        };
        let spread = (p[0] - p[1]) as f64;
        if spread > 0.10 {
            continue;
        }
        let p60 = p[2] as f64;
        if p60 < box_median + 0.10 {
            continue;
        }
        *slot = Some((p60, spread));
    }
    let [top, bottom, left, right] = qualifying;
    if !((top.is_some() && bottom.is_some()) || (left.is_some() && right.is_some())) {
        return None;
    }
    // max() keeps the first of equal levels, in the order the sides were checked.
    qualifying
        .into_iter()
        .flatten()
        .fold(None, |best: Option<(f64, f64)>, q| match best {
            Some(b) if q.0 <= b.0 => Some(b),
            _ => Some(q),
        })
}

// Camera-scan holder mask against a bed normalized to 1.0. Dense negative bottoms out near
// 0.05, so the cut sits below that. The mask is not always this clean, which is why the walk
// bridges instead of classifying every pixel.
const EDGE_MASK_LEVEL: f32 = 0.03;
// Clear film base / bare bed. _detection_luma anchors P99.5 at 1.0, so the bed is a stable
// landmark; picture only reaches here when the frame carries no base at all.
const EDGE_BASE_LEVEL: f32 = 0.95;
// Borders that no absolute level finds, read against the picture on the same line. A rebate
// on a thin negative sits above the picture but below the bed, because the frame's own
// highlights set the P99.5 anchor. A holder edge sits below the picture but above the mask
// cut, shading from near-black to mid-grey along the side. Both must end in a step, which is
// what separates them from a bright subject or a deep shadow touching the frame edge.
const EDGE_REBATE_MARGIN: f32 = 0.10;
// The dark side runs a wider margin: dense negative crowds the bottom of the range, so a
// shadow sits nearer the picture median than a highlight does.
const EDGE_SHADE_MARGIN: f32 = 0.16;
const EDGE_REBATE_STEP: f32 = 0.18;
const EDGE_REBATE_REF: usize = 60;
// The step carries most of the drop from the outer level to the picture. A boundary does that
// and a subject shading toward the edge does not. Replaces the fixed EDGE_REBATE_STEP at
// depth, where a dense frame compresses every level.
const EDGE_STEP_SHARE: f32 = 0.45;
const EDGE_STEP_FLOOR: f32 = 0.06;

/// np.max with NaN propagation.
fn np_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// `_walk_border_depth(lines, limit, bridge, relative_step)`: per line, the depth of the
/// non-image border walking inward from the frame edge. `lines` is L x N (steps inward x
/// lines). A camera scan puts two tiers outside the picture, either of which may be absent: a
/// holder mask at the very edge, then a film-base sliver. Neither is separable by level alone,
/// but the tiers are adjacent, so the walk carries across a short gap and stops at the last
/// border evidence. A run that ended on a relative test takes its boundary from the strongest
/// step inside it, and that step must clear a gate (fixed, or a share of the run's own drop).
pub fn walk_border_depth(
    lines: &[f32],
    rows: usize,
    n: usize,
    limit: usize,
    bridge: i64,
    relative_step: bool,
) -> Option<Vec<i64>> {
    if rows < limit + 3 || lines.len() != rows * n {
        return None;
    }
    let at = |r: usize, c: usize| lines[r * n + c];
    let ref_end = (limit + EDGE_REBATE_REF).min(rows);
    let picture: Vec<f32> = (0..n)
        .map(|c| np_median(&mut (limit..ref_end).map(|r| at(r, c)).collect::<Vec<_>>()))
        .collect();

    let mut depth = vec![0i64; n];
    let mut last = vec![-1i64; n];
    let mut soft = vec![false; n];
    let mut alive = vec![true; n];
    let mut firm_depth = vec![0i64; n];
    let mut firm_last = vec![-1i64; n];
    let mut firm_alive = vec![true; n];
    for i in 0..=limit {
        let ii = i as i64;
        for c in 0..n {
            alive[c] &= ii - last[c] <= bridge;
            firm_alive[c] &= ii - firm_last[c] <= bridge;
        }
        if !alive.iter().any(|&a| a) && !firm_alive.iter().any(|&a| a) {
            break;
        }
        for c in 0..n {
            let v = at(i, c);
            let firm = v < EDGE_MASK_LEVEL || v > EDGE_BASE_LEVEL;
            let border =
                firm || v > picture[c] + EDGE_REBATE_MARGIN || v < picture[c] - EDGE_SHADE_MARGIN;
            if alive[c] && border {
                last[c] = ii;
                depth[c] = ii + 1;
                soft[c] = !firm;
            }
            if firm_alive[c] && firm {
                firm_last[c] = ii;
                firm_depth[c] = ii + 1;
            }
        }
    }

    let out = (0..n)
        .map(|c| {
            // span[j] = |lines[j + 3] - lines[j]|, masked to -1 past the walk's depth.
            let masked: Vec<f32> = (0..limit)
                .map(|j| {
                    if (j as i64) < depth[c] {
                        (at(j + 3, c) - at(j, c)).abs()
                    } else {
                        -1.0
                    }
                })
                .collect();
            let peak = masked.iter().copied().fold(masked[0], np_max);
            let near = 0.9f32 * peak;
            let peak_at = masked.iter().position(|&m| m >= near).unwrap_or(0) as i64;
            let resolved = (peak_at + 3).clamp(0, limit as i64);
            let gate = if relative_step {
                let outer = np_median(&mut [at(0, c), at(1, c), at(2, c)]);
                let g = EDGE_STEP_SHARE * (outer - picture[c]).abs();
                if g.is_nan() {
                    g
                } else {
                    g.max(EDGE_STEP_FLOOR)
                }
            } else {
                EDGE_REBATE_STEP
            };
            let d = if soft[c] {
                if peak >= gate { resolved } else { 0 }
            } else {
                depth[c]
            };
            d.max(firm_depth[c])
        })
        .collect();
    Some(out)
}

// Gap the walk carries across, covering the mid-gray ramp between mask and base.
const EDGE_BRIDGE: i64 = 8;
const EDGE_TRIM_CAP: f64 = 0.03;
const EDGE_CORE: f64 = 0.70;
const EDGE_QUANTILE: f64 = 80.0;
const EDGE_MIN_SUPPORT: f64 = 0.45;
// Second, deeper walk. EDGE_TRIM_CAP is the depth a border may reach on trust alone. Past it
// a border must prove itself, because that much of a side is worth real picture. A wide
// holder margin, a sprocket rail or a neighbouring frame all run several times deeper than
// the cap, and clamping there leaves a band the eye reads as an uncropped edge.
const EDGE_DEEP_CAP: f64 = 0.10;
// Wider than EDGE_BRIDGE: a border this thick is layered (mask, ramp, base, sometimes a second
// mask segment) and the mid-gray runs between the tiers are longer at this depth.
const EDGE_DEEP_BRIDGE: i64 = 12;
// Proof one: the step ends a straight edge. A film boundary is a cut line, so the per-line
// depths fit a sloped line to within a fraction of the allowance. Picture content resolved by
// the same rule scatters several times wider.
const EDGE_DEEP_STRAIGHT: f64 = 0.025;
// Proof two: the walk stopped on its own. Reaching the allowance means no boundary was found
// inside it, and the depth is then an artefact of where the search was cut off.
const EDGE_DEEP_FULL: f64 = 0.92;
// Proof three is EDGE_STEP_SHARE. Proof four: the outermost pixels are one material. Mask,
// bed and film base each hold their level along the whole side, so the outer level barely
// moves from line to line, while picture reaching the frame edge moves with the subject.
const EDGE_DEEP_OUTER_SPREAD: f64 = 0.15;

/// `np.percentile` of float64 data (or integers, which NumPy interpolates in float64).
fn np_percentile_f64(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    if v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let n = v.len();
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let lo = (virtual_index.floor().max(0.0) as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let t = virtual_index - lo as f64;
    v.sort_by(f64::total_cmp);
    let (a, b) = (v[lo], v[hi]);
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// `np.median` of float64 data.
fn np_median_f64(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `_pool_border_depth`: the 80th percentile of the line depths, capped at `limit`, or 0 when
/// fewer than 45% of the lines show a border.
fn pool_border_depth(depth: &[i64], limit: i64) -> i64 {
    let live = depth.iter().filter(|&&d| d > 0).count();
    if (live as f64 / depth.len() as f64) < EDGE_MIN_SUPPORT {
        return 0;
    }
    let as_f: Vec<f64> = depth.iter().map(|&d| d as f64).collect();
    (limit as f64).min(np_percentile_f64(&as_f, EDGE_QUANTILE)) as i64
}

/// `_border_edge_scatter`: the median residual of the live depths around a Theil-Sen line.
fn border_edge_scatter(depth: &[i64]) -> f64 {
    let index: Vec<f64> = (0..depth.len())
        .filter(|&i| depth[i] > 0)
        .map(|i| i as f64)
        .collect();
    if index.len() < 8 {
        return f64::INFINITY;
    }
    let value: Vec<f64> = depth
        .iter()
        .filter(|&&d| d > 0)
        .map(|&d| d as f64)
        .collect();
    let step = (index.len() / 48).max(1);
    let a: Vec<f64> = index.iter().step_by(step).copied().collect();
    let b: Vec<f64> = value.iter().step_by(step).copied().collect();
    let mut slopes = Vec::new();
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            let da = a[i] - a[j];
            if da != 0.0 {
                slopes.push((b[i] - b[j]) / da);
            }
        }
    }
    let slope = if slopes.is_empty() {
        0.0
    } else {
        np_median_f64(&slopes)
    };
    let offsets: Vec<f64> = value
        .iter()
        .zip(&index)
        .map(|(v, i)| v - slope * i)
        .collect();
    let intercept = np_median_f64(&offsets);
    let residuals: Vec<f64> = value
        .iter()
        .zip(&index)
        .map(|(v, i)| (v - (slope * i + intercept)).abs())
        .collect();
    np_median_f64(&residuals)
}

#[derive(Clone, Copy)]
enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

/// A film box inside an HxW plane: the plane's row stride, and the box's top, left, height
/// and width.
#[derive(Clone, Copy)]
struct BoxView<'a> {
    lum: &'a [f32],
    w: usize,
    y1: usize,
    x1: usize,
    bh: usize,
    bw: usize,
}

/// `_EDGE_ORIENT[side](box)`, cropped to the core columns and to its first `keep` lines (rows),
/// each one position along the edge (columns); and the uncropped line count.
fn edge_lines(b: BoxView, side: Side, keep: usize) -> (Vec<f32>, usize, usize, usize) {
    let (rows, n) = match side {
        Side::Top | Side::Bottom => (b.bh, b.bw),
        Side::Left | Side::Right => (b.bw, b.bh),
    };
    let margin = py_round((1.0 - EDGE_CORE) * 0.5 * n as f64).max(0) as usize;
    let (c0, c1) = if n >= 2 * margin + 8 {
        (margin, n - margin)
    } else {
        (0, n)
    };
    let (cols, kept) = (c1 - c0, keep.min(rows));
    let row = |y: usize| &b.lum[(b.y1 + y) * b.w + b.x1..][..b.bw];
    let mut out = vec![0f32; kept * cols];
    match side {
        Side::Top | Side::Bottom => {
            for (r, line) in out.chunks_exact_mut(cols).enumerate() {
                let y = if matches!(side, Side::Top) {
                    r
                } else {
                    b.bh - 1 - r
                };
                line.copy_from_slice(&row(y)[c0..c1]);
            }
        }
        // Read along the source rows; the box's columns are these lines.
        Side::Left | Side::Right => {
            for c in c0..c1 {
                let src = row(c);
                for r in 0..kept {
                    let x = if matches!(side, Side::Left) {
                        r
                    } else {
                        b.bw - 1 - r
                    };
                    out[r * cols + c - c0] = src[x];
                }
            }
        }
    }
    (out, rows, kept, cols)
}

/// `_edge_border_depth(box, side, limit, deep_limit)`: the border depth on one side, per line
/// (these frames are slightly tilted) and pooled. A first walk to `limit` is believed; a second
/// to `deep_limit` is adopted only where it goes deeper and passes all four proofs.
fn edge_border_depth(b: BoxView, side: Side, limit: i64, deep_limit: i64) -> i64 {
    // The walks read no line past their limit plus the picture reference.
    let keep = limit.max(deep_limit).max(0) as usize + EDGE_REBATE_REF;
    let (lines, rows, kept, cols) = edge_lines(b, side, keep);
    if limit <= 0 || (rows as i64) < limit + 8 || cols < 8 {
        return 0;
    }
    let Some(walk) = walk_border_depth(&lines, kept, cols, limit as usize, EDGE_BRIDGE, false)
    else {
        return 0;
    };
    let near = pool_border_depth(&walk, limit);
    if deep_limit <= limit || (rows as i64) < deep_limit + 8 {
        return near;
    }
    let Some(deep) = walk_border_depth(
        &lines,
        kept,
        cols,
        deep_limit as usize,
        EDGE_DEEP_BRIDGE,
        true,
    ) else {
        return near;
    };
    let far = pool_border_depth(&deep, deep_limit);
    if far <= near || far as f64 >= EDGE_DEEP_FULL * deep_limit as f64 {
        return near;
    }
    if border_edge_scatter(&deep) > EDGE_DEEP_STRAIGHT * deep_limit as f64 {
        return near;
    }
    let mut outer: Vec<f32> = (0..cols)
        .filter(|&c| deep[c] > 0)
        .map(|c| np_median(&mut [lines[c], lines[cols + c], lines[2 * cols + c]]))
        .collect();
    // np.median of nothing is NaN, and a NaN spread passes no comparison.
    if outer.is_empty() {
        return far;
    }
    let mid = np_median(&mut outer.clone());
    let mut spread: Vec<f32> = outer.iter_mut().map(|v| (*v - mid).abs()).collect();
    if np_median(&mut spread) as f64 > EDGE_DEEP_OUTER_SPREAD {
        return near;
    }
    far
}

/// `measure_film_edges(lum, film_roi)`: per side (top, bottom, left, right) the border depth as
/// a fraction of that side.
pub fn measure_film_edges(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (usize, usize, usize, usize),
) -> [f64; 4] {
    let (y1, y2, x1, x2) = (roi.0.min(h), roi.1.min(h), roi.2.min(w), roi.3.min(w));
    let (bh, bw) = (y2.saturating_sub(y1), x2.saturating_sub(x1));
    if bh < 16 || bw < 16 {
        return [0.0; 4];
    }
    let view = BoxView {
        lum,
        w,
        y1,
        x1,
        bh,
        bw,
    };
    let ly = py_round(EDGE_TRIM_CAP * bh as f64);
    let lx = py_round(EDGE_TRIM_CAP * bw as f64);
    let dy = py_round(EDGE_DEEP_CAP * bh as f64);
    let dx = py_round(EDGE_DEEP_CAP * bw as f64);
    let sides = [
        (Side::Top, ly, dy, bh),
        (Side::Bottom, ly, dy, bh),
        (Side::Left, lx, dx, bw),
        (Side::Right, lx, dx, bw),
    ];
    let mut out = [0.0; 4];
    out.par_iter_mut()
        .zip(sides)
        .for_each(|(o, (side, limit, deep, extent))| {
            *o = edge_border_depth(view, side, limit, deep) as f64 / extent as f64;
        });
    out
}

/// `_trim_film_edges(lum, film_roi)`: the box inset past the border on each side, or the box
/// itself when that would take half a side.
pub fn trim_film_edges(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (i64, i64, i64, i64),
) -> (i64, i64, i64, i64) {
    let (y1, y2, x1, x2) = roi;
    let (height, width) = (y2 - y1, x2 - x1);
    let clamp = |v: i64| v.max(0) as usize;
    let m = measure_film_edges(lum, h, w, (clamp(y1), clamp(y2), clamp(x1), clamp(x2)));
    let out = (
        y1 + py_round(m[0] * height as f64),
        y2 - py_round(m[1] * height as f64),
        x1 + py_round(m[2] * width as f64),
        x2 - py_round(m[3] * width as f64),
    );
    if ((out.1 - out.0) as f64) < 0.5 * height as f64
        || ((out.3 - out.2) as f64) < 0.5 * width as f64
    {
        return roi;
    }
    out
}

// Walk cap per side, as a fraction of that side. Real rebate plus bed slop stays well under
// it, so reaching the cap means the profile never came back down and bright picture content
// touches that edge. Those sides report NaN and pooling ignores them.
const BORDER_WALK_CAP: f64 = 0.06;
const BORDER_MIN_SEPARATION: f64 = 0.15;

/// `measure_film_border(lum, film_roi)`: per side (top, bottom, left, right) the bright
/// border's thickness as a fraction of that side; 0.0 for none, NaN for no clean transition.
pub fn measure_film_border(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (usize, usize, usize, usize),
) -> [f64; 4] {
    let (y1, y2, x1, x2) = (roi.0.min(h), roi.1.min(h), roi.2.min(w), roi.3.min(w));
    let (bh, bw) = (y2.saturating_sub(y1), x2.saturating_sub(x1));
    if bh < 16 || bw < 16 {
        return [f64::NAN; 4];
    }
    let boxed = crop(lum, w, (y1, y2, x1, x2));
    let Some(level) = np_percentiles(&boxed, &[40.0]) else {
        return [f64::NAN; 4];
    };
    let image_level = level[0] as f64;
    // mean(axis=1): a pairwise sum per row; mean(axis=0): sequential down each column.
    let rows: Vec<f32> = boxed
        .chunks_exact(bw)
        .map(|r| pairwise_sum_f32(r) / bw as f32)
        .collect();
    let mut cols = vec![0f32; bw];
    for r in boxed.chunks_exact(bw) {
        for (c, &v) in cols.iter_mut().zip(r) {
            *c += v;
        }
    }
    cols.iter_mut().for_each(|c| *c /= bh as f32);

    let side = |profile: &[f32], reversed: bool| -> f64 {
        let length = profile.len();
        let walk: Vec<f32> = if reversed {
            profile.iter().rev().copied().collect()
        } else {
            profile.to_vec()
        };
        let outer_n = py_round(0.03 * length as f64).max(2) as usize;
        let Some(b) = np_percentiles(&walk[..outer_n.min(length)], &[90.0]) else {
            return f64::NAN;
        };
        let base = b[0] as f64;
        if base - image_level < BORDER_MIN_SEPARATION {
            return 0.0;
        }
        // NEP 50: the Python float threshold joins float32 comparisons as float32.
        let threshold = (0.5 * (base + image_level)) as f32;
        let cap = ((BORDER_WALK_CAP * length as f64) as usize)
            .max(3)
            .min(length);
        let head = &walk[..cap];
        // np.argmax: the first maximum, or the first NaN.
        let peak = head.iter().position(|v| v.is_nan()).unwrap_or_else(|| {
            head.iter()
                .enumerate()
                .fold(0, |best, (i, &v)| if v > head[best] { i } else { best })
        });
        if head[peak] <= threshold {
            return 0.0;
        }
        match head[peak..].iter().position(|&v| v <= threshold) {
            Some(k) => (peak + k) as f64 / length as f64,
            None => f64::NAN,
        }
    };
    [
        side(&rows, false),
        side(&rows, true),
        side(&cols, false),
        side(&cols, true),
    ]
}

/// `_roi_from_measured_border(lum, film_roi)`: the box inset by its measured border, when an
/// opposite pair agrees.
pub fn roi_from_measured_border(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (i64, i64, i64, i64),
) -> Option<(i64, i64, i64, i64)> {
    let (y1, y2, x1, x2) = roi;
    let clamp = |v: i64| v.max(0) as usize;
    let m = measure_film_border(lum, h, w, (clamp(y1), clamp(y2), clamp(x1), clamp(x2)));
    let (height, width) = ((y2 - y1) as f64, (x2 - x1) as f64);
    let opposite = [1, 0, 3, 2];
    let mut insets = [0f64; 4];
    for i in 0..4 {
        let (mut own, other) = (m[i], m[opposite[i]]);
        if !own.is_finite() {
            own = other;
        }
        if !own.is_finite() {
            continue;
        }
        if own > 0.0 && other.is_finite() && other > 0.0 {
            own = own.min(2.0 * other);
        }
        insets[i] = own;
    }
    if !((insets[0] > 0.0 && insets[1] > 0.0) || (insets[2] > 0.0 && insets[3] > 0.0)) {
        return None;
    }
    let out = (
        y1 + py_round(insets[0] * height),
        y2 - py_round(insets[1] * height),
        x1 + py_round(insets[2] * width),
        x2 - py_round(insets[3] * width),
    );
    (out.1 > out.0 && out.3 > out.2).then_some(out)
}

// Smallest fraction of a film-box side the tier refinement may keep. A rebate never takes half
// a side, so anything below this is a broken occupancy run, not a frame.
const TIER_MIN_KEEP: f64 = 0.5;

/// `_smooth_signal(signal, window)`: `np.convolve(signal, ones(window) / window, "same")` in
/// NumPy's float32 arithmetic. A full-overlap output of a window up to 11 goes through NumPy's
/// small_correlate (float32 sums); every other output through BLAS sdot (float32 products, a
/// float64 sum). None for a window over 31 (BLAS's vector path) or longer than the signal
/// (np.convolve swaps the arrays): no caller uses either.
pub fn smooth_signal(signal: &[f32], window: usize) -> Option<Vec<f32>> {
    let n = signal.len();
    if window <= 1 || n == 0 {
        return Some(signal.to_vec());
    }
    if window > 31 || window > n {
        return None;
    }
    let k = 1.0f32 / window as f32;
    let left = window / 2;
    (0..n)
        .map(|o| {
            let lo = left.saturating_sub(o);
            let hi = window.min(n + left - o);
            let terms = (lo..hi).map(|j| signal[o + j - left] * k);
            if lo == 0 && hi == window && window <= 11 {
                terms.fold(0f32, |s, p| s + p)
            } else {
                terms.fold(0f64, |s, p| s + p as f64) as f32
            }
        })
        .collect::<Vec<f32>>()
        .into()
}

/// `_snap_edge_to_gradient`'s windows: outward and inward fractions of the profile, the
/// dominance over the window median, and the smallest window in pixels.
#[derive(Clone, Copy)]
pub struct Snap {
    pub window_out: f64,
    pub window_in: f64,
    pub min_dominance: f64,
    pub min_window_px: i64,
}

impl Default for Snap {
    fn default() -> Self {
        Snap {
            window_out: 0.06,
            window_in: 0.02,
            min_dominance: 2.0,
            min_window_px: 3,
        }
    }
}

/// `_snap_edge_to_gradient(profile, idx, is_start, ...)`: the edge moved to the strongest step
/// of the smoothed profile in a window biased outward, when that step dominates the window.
pub fn snap_edge_to_gradient(profile: &[f32], idx: i64, is_start: bool, snap: Snap) -> i64 {
    let n = profile.len();
    if n < 8 {
        return idx;
    }
    let Some(sm) = smooth_signal(profile, py_round(0.01 * n as f64).max(3) as usize) else {
        return idx;
    };
    let grad: Vec<f32> = sm.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let out_px = py_round(snap.window_out * n as f64).max(snap.min_window_px);
    let in_px = py_round(snap.window_in * n as f64).max(snap.min_window_px);
    let (lo, hi) = if is_start {
        ((idx - out_px).max(0), (idx + in_px).min(n as i64 - 1))
    } else {
        ((idx - in_px).max(0), (idx + out_px).min(n as i64 - 1))
    };
    if hi <= lo {
        return idx;
    }
    let window = &grad[lo as usize..hi as usize];
    let peak = window.iter().copied().fold(window[0], np_max);
    let median = np_median(&mut window.to_vec());
    if (peak as f64) < snap.min_dominance * median as f64 + 1e-6 || peak.is_nan() || median.is_nan()
    {
        return idx;
    }
    // np.isclose(window, peak, rtol=1e-5, atol=1e-7): the difference in float32, the tolerance
    // a Python float joining the comparison as float32 (NEP 50).
    let tol = (1e-7 + 1e-5 * (peak as f64).abs()) as f32;
    let mut best: Option<(i64, i64)> = None;
    for (i, &v) in window.iter().enumerate() {
        if (v - peak).abs() <= tol || v == peak {
            let t = lo + i as i64 + 1;
            let d = (t - idx).abs();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((t, d));
            }
        }
    }
    best.map_or(idx, |(t, _)| t)
}

/// `_longest_run_above(profile, threshold)`: the longest half-open run with profile >= threshold,
/// the first of equal length.
fn longest_run_above(profile: &[f64], threshold: f64) -> Option<(i64, i64)> {
    let idx: Vec<i64> = (0..profile.len())
        .filter(|&i| profile[i] >= threshold)
        .map(|i| i as i64)
        .collect();
    if idx.is_empty() {
        return None;
    }
    let mut best = (idx[0], idx[0]);
    let mut start = idx[0];
    for w in 0..idx.len() {
        let end_here = w + 1 == idx.len() || idx[w + 1] - idx[w] > 1;
        if end_here {
            if idx[w] - start > best.1 - best.0 {
                best = (start, idx[w]);
            }
            if w + 1 < idx.len() {
                start = idx[w + 1];
            }
        }
    }
    Some((best.0, best.1 + 1))
}

/// `_estimate_tier_levels(lum, film_roi)`: (bed, rebate, image, ring spread), or None.
fn estimate_tier_levels(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (usize, usize, usize, usize),
) -> Option<(f64, f64, f64, f64)> {
    let bed = np_percentiles(lum, &[99.0])?[0] as f64;
    let (rebate, ring_spread) = find_rebate_level(lum, h, w, roi, Some(bed))?;
    let (y1, y2, x1, x2) = (roi.0.min(h), roi.1.min(h), roi.2.min(w), roi.3.min(w));
    let boxed = crop(lum, w, (y1, y2, x1, x2));
    let cut = (rebate - 0.02) as f32;
    let dark: Vec<f32> = boxed.iter().copied().filter(|&v| v < cut).collect();
    if (dark.len() as f64) < 0.05 * boxed.len() as f64 {
        return None;
    }
    let image = np_percentiles(&dark, &[30.0])?[0] as f64;
    if rebate - image < 0.04f64.max(3.0 * ring_spread) {
        return None;
    }
    Some((bed, rebate, image, ring_spread))
}

#[derive(Clone)]
pub struct TierRefinement {
    pub roi: (i64, i64, i64, i64),
    pub rows: Vec<f32>,
    pub cols: Vec<f32>,
}

/// `_refine_film_roi_by_tiers(lum, film_roi)`: the picture box from rebate/image occupancy runs
/// snapped to the strongest gradients, with the full-length occupancy profiles.
pub fn refine_film_roi_by_tiers(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (usize, usize, usize, usize),
) -> Option<TierRefinement> {
    let (_, rebate, image, ring_spread) = estimate_tier_levels(lum, h, w, roi)?;
    let (y1, y2, x1, x2) = (roi.0.min(h), roi.1.min(h), roi.2.min(w), roi.3.min(w));
    let (bh, bw) = (y2 - y1, x2 - x1);
    let boxed = crop(lum, w, (y1, y2, x1, x2));
    let threshold = (0.5 * (rebate + image)).max(rebate - 0.04f64.max(3.0 * ring_spread)) as f32;
    let mask: Vec<bool> = boxed.iter().map(|&v| v < threshold).collect();
    let count = |r0: usize, r1: usize, c0: usize, c1: usize, by_row: bool| -> Vec<f64> {
        if by_row {
            (r0..r1)
                .map(|r| {
                    mask[r * bw + c0..r * bw + c1]
                        .iter()
                        .filter(|&&m| m)
                        .count() as f64
                        / (c1 - c0) as f64
                })
                .collect()
        } else {
            (c0..c1)
                .map(|c| (r0..r1).filter(|&r| mask[r * bw + c]).count() as f64 / (r1 - r0) as f64)
                .collect()
        }
    };
    let row_occ = count(0, bh, 0, bw, true);
    let (vt, vb) = longest_run_above(&row_occ, 0.55)?;
    let col_occ = count(vt as usize, vb as usize, 0, bw, false);
    let (hl, hr) = longest_run_above(&col_occ, 0.55)?;
    if ((vb - vt) as f64) < TIER_MIN_KEEP * bh as f64
        || ((hr - hl) as f64) < TIER_MIN_KEEP * bw as f64
    {
        return None;
    }
    let area = ((vb - vt) * (hr - hl)) as f64 / (bh * bw) as f64;
    if !(0.25..=0.95).contains(&area) {
        return None;
    }
    // box[vt:vb, :].mean(axis=0) sums down the columns; box[:, hl:hr].mean(axis=1) is a pairwise
    // sum per row.
    let (vtu, vbu, hlu, hru) = (vt as usize, vb as usize, hl as usize, hr as usize);
    let mut col_profile = vec![0f32; bw];
    for r in vtu..vbu {
        for (c, p) in col_profile.iter_mut().enumerate() {
            *p += boxed[r * bw + c];
        }
    }
    col_profile
        .iter_mut()
        .for_each(|p| *p /= (vbu - vtu) as f32);
    let row_profile: Vec<f32> = (0..bh)
        .map(|r| pairwise_sum_f32(&boxed[r * bw + hlu..r * bw + hru]) / (hru - hlu) as f32)
        .collect();
    let hl = snap_edge_to_gradient(&col_profile, hl, true, Snap::default());
    let hr = snap_edge_to_gradient(&col_profile, hr, false, Snap::default());
    let vt = snap_edge_to_gradient(&row_profile, vt, true, Snap::default());
    let vb = snap_edge_to_gradient(&row_profile, vb, false, Snap::default());
    let pad_y = py_round(0.004 * bh as f64).max(2);
    let pad_x = py_round(0.004 * bw as f64).max(2);
    let (vt, vb) = ((vt - pad_y).max(0), (vb + pad_y).min(bh as i64));
    let (hl, hr) = ((hl - pad_x).max(0), (hr + pad_x).min(bw as i64));
    if vb - vt <= 0 || hr - hl <= 0 {
        return None;
    }
    let mut rows = vec![0f32; h];
    rows[y1..y2]
        .iter_mut()
        .zip(&row_occ)
        .for_each(|(d, &v)| *d = v as f32);
    let mut cols = vec![0f32; w];
    cols[x1..x2]
        .iter_mut()
        .zip(&col_occ)
        .for_each(|(d, &v)| *d = v as f32);
    let (y1, x1) = (y1 as i64, x1 as i64);
    Some(TierRefinement {
        roi: (y1 + vt, y1 + vb, x1 + hl, x1 + hr),
        rows,
        cols,
    })
}

/// The uint8 preview and its gray plane, as `_refine_frame_bounds` and `_dark_region_bounds`
/// build them from an HxW or HxWx3 float32 image.
fn preview_gray(img: &[f32], h: usize, w: usize, cn: usize) -> Vec<u8> {
    let mut preview = vec![0u8; h * w * 3];
    normalize_to_uint8(&rgb(img, cn), &mut preview);
    let mut gray = vec![0u8; h * w];
    crate::gray::bgr_to_gray(&preview, &mut gray);
    gray
}

/// `_boundary_candidates(signal, from_start)`: (edge index, value, inner index, value).
fn boundary_candidates(signal: &[f32], from_start: bool) -> (i64, f64, i64, f64) {
    let length = signal.len() as i64;
    if length == 0 {
        return (0, 0.0, 0, 0.0);
    }
    let edge_window = (py_round(length as f64 * 0.08))
        .max(32)
        .min((length - 1).max(1));
    let search_end = py_round(length as f64 * 0.45)
        .max(edge_window + 1)
        .min(length);
    let search_start = py_round(length as f64 * 0.55).min((length - edge_window - 1).max(0));
    // np.argmax: the first maximum, or the first NaN.
    let argmax = |s: &[f32]| -> usize {
        s.iter().position(|v| v.is_nan()).unwrap_or_else(|| {
            s.iter()
                .enumerate()
                .fold(0, |b, (i, &v)| if v > s[b] { i } else { b })
        })
    };
    let at = |a: i64, b: i64| {
        &signal[a.clamp(0, length) as usize..b.clamp(0, length).max(a.clamp(0, length)) as usize]
    };
    let (edge, edge_base, search, search_base) = if from_start {
        (
            at(0, edge_window),
            0,
            at(edge_window, search_end),
            edge_window,
        )
    } else {
        (
            at(length - edge_window, length),
            length - edge_window,
            at(search_start, length - edge_window),
            search_start,
        )
    };
    let (edge_idx, edge_value) = if edge.is_empty() {
        (edge_base, 0.0)
    } else {
        let k = argmax(edge);
        (edge_base + k as i64, edge[k] as f64)
    };
    if search.is_empty() {
        return (edge_idx, edge_value, edge_idx, edge_value);
    }
    let k = argmax(search);
    (
        edge_idx,
        edge_value,
        search_base + k as i64,
        search[k] as f64,
    )
}

/// `_dark_region_bounds(image)`: (left, top, right, bottom) of the largest dark region, or None.
fn dark_region_bounds(gray: &[u8], h: usize, w: usize) -> Option<(i64, i64, i64, i64)> {
    let as_f: Vec<f64> = gray.iter().map(|&v| v as f64).collect();
    let threshold = np_percentile_f64(&as_f, 55.0);
    let mask: Vec<u8> = gray
        .iter()
        .map(|&v| if (v as f64) <= threshold { 255 } else { 0 })
        .collect();
    let mut opened = vec![0u8; h * w];
    crate::morph::morphology(
        &mask,
        &mut opened,
        (h, w, 1),
        crate::morph::Op::Open,
        (9, 9),
        1,
        Border::Constant,
    )?;
    let mut closed = vec![0u8; h * w];
    crate::morph::morphology(
        &opened,
        &mut closed,
        (h, w, 1),
        crate::morph::Op::Close,
        (31, 31),
        2,
        Border::Constant,
    )?;
    let found = crate::contours::find_external_contours(&closed, h, w);
    // max(contours, key=cv2.contourArea): the first of the largest.
    let contour = found
        .iter()
        .fold(None::<&Vec<(i32, i32)>>, |best, c| match best {
            Some(b) if crate::contours::contour_area(c) <= crate::contours::contour_area(b) => {
                Some(b)
            }
            _ => Some(c),
        })?;
    let (x, y, bw, bh) = crate::contours::bounding_rect(contour);
    let (x, y, bw, bh) = (x as i64, y as i64, bw as i64, bh as i64);
    let area_ratio = crate::contours::contour_area(contour) / ((h * w) as f64).max(1.0);
    if !(0.15..=0.85).contains(&area_ratio) {
        return None;
    }
    let (hf, wf) = (h as f64, w as f64);
    if bw < py_round(wf * 0.25) || bh < py_round(hf * 0.25) {
        return None;
    }
    let pad_x = py_round(wf * 0.004).max(4);
    let pad_y = py_round(hf * 0.004).max(4);
    let (left, top) = ((x - pad_x).max(0), (y - pad_y).max(0));
    let (right, bottom) = (
        (x + bw + pad_x).min(w as i64),
        (y + bh + pad_y).min(h as i64),
    );
    let (inset_x, inset_y) = (py_round(wf * 0.03), py_round(hf * 0.03));
    if left < inset_x
        || top < inset_y
        || (w as i64 - right) < inset_x
        || (h as i64 - bottom) < inset_y
    {
        return None;
    }
    Some((left, top, right, bottom))
}

/// `_refine_frame_bounds(image)`'s box (left, top, right, bottom) for an HxW or HxWx3 float32
/// image: Sobel profiles of the gray preview, edge and inner candidates, and the dark-region
/// fallback. None for an image narrower or shorter than the 31-tap smoothing.
pub fn refine_frame_bounds(
    img: &[f32],
    h: usize,
    w: usize,
    cn: usize,
) -> Option<(i64, i64, i64, i64)> {
    let gray = preview_gray(img, h, w, cn);
    let gray_f: Vec<f32> = gray.iter().map(|&v| v as f32).collect();
    let grad = |dx: bool| -> Option<Vec<f64>> {
        let (kx, ky): (&[f32], &[f32]) = if dx {
            (&[-1.0, 0.0, 1.0], &[1.0, 2.0, 1.0])
        } else {
            (&[1.0, 2.0, 1.0], &[-1.0, 0.0, 1.0])
        };
        let mut out = vec![0f32; h * w];
        crate::filter::sep_filter2d(&gray_f, &mut out, (h, w, 1), (kx, ky), Border::Reflect101)?;
        Some(
            out.iter()
                .map(|&v| v.abs().min(255.0) as u8 as f64)
                .collect(),
        )
    };
    let (gx, gy) = (grad(true)?, grad(false)?);
    let col: Vec<f32> = (0..w)
        .map(|c| np_percentile_f64(&(0..h).map(|r| gx[r * w + c]).collect::<Vec<_>>(), 95.0) as f32)
        .collect();
    let row: Vec<f32> = (0..h)
        .map(|r| np_percentile_f64(&gy[r * w..(r + 1) * w], 95.0) as f32)
        .collect();
    let col_signal = smooth_signal(&col, 31)?;
    let row_signal = smooth_signal(&row, 31)?;

    let (left_edge, _, left_inner, left_inner_value) = boundary_candidates(&col_signal, true);
    let (right_edge, _, right_inner, right_inner_value) = boundary_candidates(&col_signal, false);
    let (top_edge, top_edge_value, top_inner, top_inner_value) =
        boundary_candidates(&row_signal, true);
    let (bottom_edge, bottom_edge_value, bottom_inner, bottom_inner_value) =
        boundary_candidates(&row_signal, false);
    let col_floor = np_percentiles(&col_signal, &[75.0])?[0] as f64;
    let row_floor = np_percentiles(&row_signal, &[75.0])?[0] as f64;
    let (wf, hf) = (w as f64, h as f64);

    let (mut left, mut right, mut top, mut bottom) =
        (left_edge, right_edge + 1, top_edge, bottom_edge + 1);
    let pair_x = left_inner >= py_round(wf * 0.12)
        && right_inner <= py_round(wf * 0.88)
        && left_inner_value >= col_floor * 4.0
        && right_inner_value >= col_floor * 4.0
        && (right_inner - left_inner) >= py_round(wf * 0.5);
    if pair_x {
        left = left_inner;
        right = right_inner + 1;
    } else {
        if left_inner >= py_round(wf * 0.12) && left_inner_value >= col_floor * 5.0 {
            left = left_inner;
        }
        if right_inner <= py_round(wf * 0.88) && right_inner_value >= col_floor * 5.0 {
            right = right_inner + 1;
        }
    }
    let pair_y = top_inner > top_edge + 20
        && bottom_inner < bottom_edge - 20
        && top_inner_value > (top_edge_value * 1.2).max(row_floor + 25.0)
        && bottom_inner_value > (bottom_edge_value * 1.2).max(row_floor + 25.0)
        && (bottom_inner - top_inner) >= py_round(hf * 0.5);
    if pair_y {
        top = top_inner;
        bottom = bottom_inner + 1;
    } else {
        if top_inner > top_edge + 20
            && top_inner_value > (top_edge_value * 1.45).max(row_floor + 35.0)
        {
            top = top_inner;
        }
        if bottom_inner < bottom_edge - 20
            && bottom_inner_value > (bottom_edge_value * 1.45).max(row_floor + 35.0)
        {
            bottom = bottom_inner + 1;
        }
    }
    let pad_x = py_round(wf * 0.004).max(4);
    let pad_y = py_round(hf * 0.004).max(4);
    left = (left - pad_x).max(0);
    right = (right + pad_x).min(w as i64);
    top = (top - pad_y).max(0);
    bottom = (bottom + pad_y).min(h as i64);
    if right - left < py_round(wf * 0.5).max(1) {
        (left, right) = (0, w as i64);
    }
    if bottom - top < py_round(hf * 0.5).max(1) {
        (top, bottom) = (0, h as i64);
    }
    let area = ((right - left) * (bottom - top)) as f64 / (hf * wf).max(1.0);
    if area > 0.8
        && let Some((dl, dt, dr, db)) = dark_region_bounds(&gray, h, w)
    {
        let dark_area = ((dr - dl) * (db - dt)) as f64 / (hf * wf).max(1.0);
        if (0.15..=0.85).contains(&dark_area) {
            return Some((dl, dt, dr, db));
        }
    }
    Some((left, top, right, bottom))
}

pub type Roi = (i64, i64, i64, i64);

/// `_refine_roi_to_image(img, film_roi)`: the exposed picture inside the film box, by tiers,
/// then the edge walk, then the border measurement, then Sobel; with the occupancy profiles
/// when the tiers found it.
pub fn refine_roi_to_image(
    img: &[f32],
    h: usize,
    w: usize,
    cn: usize,
    roi: Roi,
) -> Option<(Roi, Option<TierRefinement>)> {
    type Refined = Option<(Roi, Option<TierRefinement>)>;
    static LAST: crate::memo::Memo<Refined, 8> = crate::memo::Memo::new();
    let key = crate::memo::key(img, &[h, w, cn]);
    let refined = || {
        let det = detection_plane(img, h, w, cn, key)?;
        refine_roi(img, &det, h, w, cn, roi)
    };
    LAST.get_or(
        crate::memo::extend(key, &[roi.0, roi.1, roi.2, roi.3]),
        refined,
    )
}

fn refine_roi(
    img: &[f32],
    det: &[f32],
    h: usize,
    w: usize,
    cn: usize,
    roi: Roi,
) -> Option<(Roi, Option<TierRefinement>)> {
    let clamp = |v: i64| v.max(0) as usize;
    let uroi = (clamp(roi.0), clamp(roi.1), clamp(roi.2), clamp(roi.3));
    if let Some(r) = refine_film_roi_by_tiers(det, h, w, uroi) {
        return Some((r.roi, Some(r)));
    }
    let walked = trim_film_edges(det, h, w, roi);
    if walked != roi {
        return Some((walked, None));
    }
    if let Some(m) = roi_from_measured_border(det, h, w, roi) {
        return Some((m, None));
    }
    if find_rebate_level(det, h, w, uroi, None).is_none() {
        return Some((roi, None));
    }
    let (y1, y2, x1, x2) = (uroi.0.min(h), uroi.1.min(h), uroi.2.min(w), uroi.3.min(w));
    let crop_w = x2.saturating_sub(x1);
    let color = rgb(img, cn);
    let crop: Vec<f32> = (y1..y2)
        .flat_map(|y| color[(y * w + x1) * 3..(y * w + x2) * 3].iter().copied())
        .collect();
    let (left, top, right, bottom) = refine_frame_bounds(&crop, y2.saturating_sub(y1), crop_w, 3)?;
    let out = (roi.0 + top, roi.0 + bottom, roi.2 + left, roi.2 + right);
    // A Sobel cut removing more than a quarter of the box latched onto picture content.
    let film_area = ((roi.1 - roi.0) * (roi.3 - roi.2)).max(1) as f64;
    if (((out.1 - out.0) * (out.3 - out.2)) as f64) < 0.75 * film_area {
        return Some((roi, None));
    }
    Some((out, None))
}

/// `measure_clip_fractions`: per channel, the share of every 4th pixel of rows y1..y2 and
/// columns x1..x2 of an HxWx3 image at or above `level`. NaN for an empty window.
pub fn clip_fractions(
    img: &[f32],
    w: usize,
    (y1, y2, x1, x2): (usize, usize, usize, usize),
    level: f32,
) -> [f64; 3] {
    let rows: Vec<usize> = (y1..y2).step_by(4).collect();
    let cols = (x1..x2).step_by(4).len();
    let counts = rows
        .par_iter()
        .map(|&y| {
            let mut c = [0u64; 3];
            let row = &img[(y * w + x1) * 3..(y * w + x2) * 3];
            for p in row.as_chunks::<3>().0.iter().step_by(4) {
                for ch in 0..3 {
                    c[ch] += (p[ch] >= level) as u64;
                }
            }
            c
        })
        .reduce(|| [0; 3], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
    let n = (rows.len() * cols) as f64;
    counts.map(|c| c as f64 / n)
}

/// `grid @ w` in float64 for Rec. 709 luma, in the order OpenBLAS's dgemv sums the three terms.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn luma_f64_chunk(grid: &[f32], dst: &mut [f64]) {
    for (d, p) in dst.iter_mut().zip(grid.as_chunks::<3>().0) {
        let t = LUMA_R.mul_add(p[0] as f64, LUMA_G * p[1] as f64);
        *d = LUMA_B.mul_add(p[2] as f64, t);
    }
}

/// `density_above_base`'s metered median: `np.median` of an Nx3 grid's float64 Rec. 709 luma.
pub fn luma_median(grid: &[f32]) -> f64 {
    let mut lum = vec![0f64; grid.len() / 3];
    lum.par_chunks_mut(4096)
        .zip(grid.par_chunks(4096 * 3))
        .for_each(|(d, p)| luma_f64_chunk(p, d));
    crate::stats::median_f64(&mut lum)
}

/// Rec. 709 luma in float32, NumPy's order for `LUMA_R * r + LUMA_G * g + LUMA_B * b` (NEP 50).
#[inline(always)]
fn luma_f32(p: &[f32; 3]) -> f32 {
    LUMA_R as f32 * p[0] + LUMA_G as f32 * p[1] + LUMA_B as f32 * p[2]
}

/// `diffusion_plane`'s panchromatic collapse: each pixel of an HxWx3 plane set to its float32 luma.
pub fn luma_planes(val: &[f32], dst: &mut [f32]) {
    crate::par_zip(
        dst.as_chunks_mut::<3>().0,
        val.as_chunks::<3>().0,
        move |d, p| *d = [luma_f32(p); 3],
    );
}

/// `contrast_mask_plane` after normalization: the float32 luma of an HxWx3 plane, blurred by
/// `sigma` with a replicated border, minus its float32 mean. Returns the plane and the mean.
pub fn contrast_mask_tail(val: &[f32], h: usize, w: usize, sigma: f64) -> Option<(Vec<f32>, f32)> {
    let mut lum = vec![0f32; h * w];
    crate::par_zip(&mut lum, val.as_chunks::<3>().0, move |d, p| {
        *d = luma_f32(p)
    });
    let mut blurred = vec![0f32; h * w];
    crate::filter::gaussian_blur(
        &lum,
        &mut blurred,
        (h, w, 1),
        (0, 0),
        (sigma, 0.0),
        Border::Replicate,
    )?;
    let centre = pairwise_sum_f32(&blurred) / blurred.len() as f32;
    blurred.par_chunks_mut(1 << 14).for_each(|d| {
        for v in d {
            *v -= centre;
        }
    });
    Some((blurred, centre))
}

/// `diffusion_grid` after the unmix: 10^x of an HxWx3 log plane, blurred by `sigma` with a
/// replicated border, back to log10.
pub fn diffusion_tail(
    unmixed: &[f32],
    h: usize,
    w: usize,
    sigma: f64,
    dst: &mut [f32],
) -> Option<()> {
    let mut light = vec![0f32; unmixed.len()];
    crate::par_zip(&mut light, unmixed, move |d, &v| {
        *d = crate::mathf::exp10_f64(v as f64) as f32;
    });
    crate::filter::gaussian_blur(
        &light,
        dst,
        (h, w, 3),
        (0, 0),
        (sigma, 0.0),
        Border::Replicate,
    )?;
    dst.par_chunks_mut(1 << 14).for_each(|d| {
        for v in d {
            *v = crate::mathf::log10f(*v);
        }
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_spans_0_to_255() {
        let img: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let mut out = vec![0u8; img.len()];
        normalize_to_uint8(&img, &mut out);
        assert_eq!(out[0], 0);
        assert_eq!(out[999], 255);
    }

    #[test]
    fn all_nan_normalizes_to_zero() {
        let img = vec![f32::NAN; 10];
        let mut out = vec![7u8; 10];
        normalize_to_uint8(&img, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn clip_fractions_count_every_fourth_pixel() {
        let (h, w) = (8, 8);
        let mut img = vec![0f32; h * w * 3];
        img[0] = 1.0;
        img[(4 * w + 5) * 3 + 1] = 1.0;
        img[(4 * w + 4) * 3 + 2] = 0.99;
        let f = clip_fractions(&img, w, (0, h, 0, w), 0.99);
        assert_eq!(f, [0.25, 0.0, 0.25]);
    }
}
