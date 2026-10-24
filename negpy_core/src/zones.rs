//! The display-buffer reads of negpy/features/exposure/analysis.py: print zones, their regions
//! and label anchors, the loupe's acutance and the vectorscope's display map.

use rayon::prelude::*;

use crate::components::connected_components;
use crate::distance::{Metric, distance_transform};
use crate::filter::{Border, gaussian_blur};

pub const ZONE_COUNT: u8 = 10;

fn py_round(v: f64) -> usize {
    v.round_ties_even() as usize
}

/// `zone_of_encoded` of a float32 lightness against the encoded mid gray `mid` (a Python
/// float, so it joins the arithmetic as float32).
#[inline]
fn zone_f32(enc: f32, mid: f32, upper: f32) -> f32 {
    let enc = if enc.is_nan() {
        enc
    } else {
        enc.clamp(0.0, 1.0)
    };
    if enc <= mid {
        5.0 * enc / mid
    } else {
        5.0 + 5.0 * (enc - mid) / upper
    }
}

/// `np.rint(np.nan_to_num(z))` clipped to the zone range.
#[inline]
fn zone_index(z: f32) -> f32 {
    let z = if z.is_nan() { 0.0 } else { z };
    z.round_ties_even().clamp(0.0, ZONE_COUNT as f32)
}

/// The first three channels of every `step`-th row and column, then INTER_AREA to `scale` of
/// that, as luminance.
fn sampled_luma(
    rgb: &[f32],
    (h, w, cn): (usize, usize, usize),
    step: usize,
    scale: impl Fn(usize, usize) -> f64,
) -> Option<(Vec<f32>, usize, usize)> {
    let (sh, sw) = (h.div_ceil(step), w.div_ceil(step));
    let copy;
    let small: &[f32] = if step == 1 && cn == 3 {
        rgb
    } else {
        copy = (0..sh)
            .into_par_iter()
            .flat_map_iter(|r| {
                let row = &rgb[r * step * w * cn..];
                (0..sw).flat_map(move |c| row[c * step * cn..c * step * cn + 3].iter().copied())
            })
            .collect::<Vec<f32>>();
        &copy
    };
    let s = scale(sh, sw);
    let (dw, dh) = (
        py_round(sw as f64 * s).max(1),
        py_round(sh as f64 * s).max(1),
    );
    let resized = crate::resize::resize_area(small, sh, sw, 3, dh, dw)?;
    let mut lum = vec![0f32; dh * dw];
    crate::analysis::luminance(&resized, &mut lum);
    Some((lum, dh, dw))
}

/// `zone_field(rgb, max_edge)`: the zone (0..10) per pixel of an HxWxC display frame, at most
/// `max_edge` on the long side, smoothed before and median-filtered after the rounding.
pub fn zone_field(
    rgb: &[f32],
    dims: (usize, usize, usize),
    max_edge: usize,
    mid: f64,
) -> Option<(Vec<u8>, usize, usize)> {
    let (h, w, _) = dims;
    let step = (h.max(w) / (2 * max_edge)).max(1);
    let (lum, dh, dw) = sampled_luma(rgb, dims, step, |sh, sw| {
        1f64.min(max_edge as f64 / sh.max(sw) as f64)
    })?;
    let mut blur = vec![0f32; lum.len()];
    gaussian_blur(
        &lum,
        &mut blur,
        (dh, dw, 1),
        (0, 0),
        (2.0, 0.0),
        Border::Reflect101,
    )?;
    let (m, upper) = (mid as f32, (1.0 - mid) as f32);
    let zones: Vec<u8> = blur
        .par_iter()
        .map(move |&v| zone_index(zone_f32(v, m, upper)) as u8)
        .collect();
    let mut out = vec![0u8; zones.len()];
    crate::median::median_blur_u8(&zones, &mut out, dh, dw, 5)?;
    Some((out, dh, dw))
}

/// `zone_grid(rgb, cells)`: the zone per cell of a grid `cells` long on the frame's long side.
pub fn zone_grid(
    rgb: &[f32],
    dims: (usize, usize, usize),
    cells: usize,
    mid: f64,
) -> Option<(Vec<i32>, usize, usize)> {
    let (h, w, _) = dims;
    let step = (h.min(w) / (4 * cells)).max(1);
    let (lum, dh, dw) = sampled_luma(rgb, dims, step, |sh, sw| cells as f64 / sh.max(sw) as f64)?;
    let (m, upper) = (mid as f32, (1.0 - mid) as f32);
    Some((
        lum.par_iter()
            .map(move |&v| zone_index(zone_f32(v, m, upper)) as i32)
            .collect(),
        dh,
        dw,
    ))
}

/// The distinct values of a zone map, ascending.
fn unique<T: Copy + Ord>(zones: &[T]) -> Vec<T> {
    let mut u = zones.to_vec();
    u.sort_unstable();
    u.dedup();
    u
}

/// `zone_anchors(zones, min_share)`: (x, y, zone, inset) per 4-connected region of at least
/// `min_share` of the frame, at its most interior pixel; the frame edge counts as an edge.
pub fn zone_anchors(
    zones: &[u8],
    (h, w): (usize, usize),
    min_share: f64,
) -> Vec<(usize, usize, u8, f32)> {
    let min_area = min_share * zones.len() as f64;
    let mut out = Vec::new();
    for zone in unique(zones) {
        let mask: Vec<u8> = zones.iter().map(|&z| (z == zone) as u8).collect();
        let cc = connected_components(&mask, h, w, 4);
        for i in 1..cc.count {
            let [left, top, cw, ch, area] = cc.stats[i];
            if (area as f64) < min_area {
                continue;
            }
            // Zeros ring the region's box, so its distances need nothing outside it.
            let (y0, x0) = (
                (top as usize).saturating_sub(1),
                (left as usize).saturating_sub(1),
            );
            let (y1, x1) = ((top + ch) as usize + 1, (left + cw) as usize + 1);
            let (y1, x1) = (y1.min(h), x1.min(w));
            let (bh, bw) = (y1 - y0, x1 - x0);
            let mut sub = vec![0u8; bh * bw];
            for y in y0.max(1)..y1.min(h - 1) {
                for x in x0.max(1)..x1.min(w - 1) {
                    sub[(y - y0) * bw + x - x0] = (cc.labels[y * w + x] == i as i32) as u8;
                }
            }
            let dist = distance_transform(&sub, bh, bw, Metric::L2Mask3);
            let best = dist
                .iter()
                .enumerate()
                .fold(0, |b, (j, &d)| if d > dist[b] { j } else { b });
            if dist[best] > 0.0 {
                out.push((x0 + best % bw, y0 + best / bw, zone, dist[best]));
            } else {
                out.push((0, 0, zone, 0.0));
            }
        }
    }
    out
}

/// `zone_region_labels(zones)`: (col, row, zone) per 4-connected region, at the region's cell
/// nearest its centroid.
pub fn zone_region_labels(zones: &[i32], (h, w): (usize, usize)) -> Vec<(usize, usize, i32)> {
    let mut out = Vec::new();
    for zone in unique(zones) {
        let mask: Vec<u8> = zones.iter().map(|&z| (z == zone) as u8).collect();
        let cc = connected_components(&mask, h, w, 4);
        let mut best = vec![(f64::INFINITY, 0usize); cc.count];
        for (p, &label) in cc.labels.iter().enumerate() {
            if label <= 0 {
                continue;
            }
            let [cx, cy] = cc.centroids[label as usize];
            let (dx, dy) = ((p % w) as f64 - cx, (p / w) as f64 - cy);
            let d = dx * dx + dy * dy;
            if d < best[label as usize].0 {
                best[label as usize] = (d, p);
            }
        }
        out.extend(best[1..].iter().map(|&(_, p)| (p % w, p / w, zone)));
    }
    out
}

/// `zone_shares(zones)`: the fraction of the frame in each zone 0..10.
pub fn zone_shares(zones: &[u8]) -> Vec<f64> {
    let mut count = [0usize; ZONE_COUNT as usize + 1];
    for &z in zones {
        if (z as usize) < count.len() {
            count[z as usize] += 1;
        }
    }
    let n = zones.len().max(1) as f64;
    count.iter().map(|&c| c as f64 / n).collect()
}

/// `zone_occupancy(l_bins)`: display-luma histogram bins folded into the share of each zone,
/// each bin at the zone its center rounds to.
pub fn zone_occupancy(bins: &[f64], mid: f64) -> Vec<f64> {
    let mut out = vec![0f64; ZONE_COUNT as usize + 1];
    let total: f64 = bins.iter().sum();
    if total <= 0.0 {
        return out;
    }
    let n = bins.len() as f64;
    for (i, &b) in bins.iter().enumerate() {
        let enc = ((i as f64 + 0.5) / n).clamp(0.0, 1.0);
        let z = if enc <= mid {
            5.0 * enc / mid
        } else {
            5.0 + 5.0 * (enc - mid) / (1.0 - mid)
        };
        out[z.round_ties_even().clamp(0.0, ZONE_COUNT as f64) as usize] += b;
    }
    out.iter().map(|&v| v / total).collect()
}

/// `loupe_acutance(patch)`: 100 x the RMS of the luminance's 3x3 Laplacian (cv2.Laplacian,
/// ksize 1, reflect-101 border) of an HxWxC display patch.
pub fn loupe_acutance(rgb: &[f32], (h, w, cn): (usize, usize, usize)) -> f64 {
    let mut lum = vec![0f32; h * w];
    if cn == 3 {
        crate::analysis::luminance_chunk(rgb, &mut lum);
    } else {
        let three: Vec<f32> = rgb
            .chunks_exact(cn)
            .map(|p| [p[0], p[1], p[2]])
            .collect::<Vec<_>>()
            .into_flattened();
        crate::analysis::luminance_chunk(&three, &mut lum);
    }
    // Reflect-101 padding, one pixel wide, so every tap reads inside the buffer.
    let reflect = |i: isize, n: usize| -> usize {
        if n == 1 {
            0
        } else if i < 0 {
            (-i) as usize
        } else if i as usize >= n {
            2 * (n - 1) - i as usize
        } else {
            i as usize
        }
    };
    let pw = w + 2;
    let mut pad = vec![0f32; (h + 2) * pw];
    for (y, out) in pad.chunks_exact_mut(pw).enumerate() {
        let src = &lum[reflect(y as isize - 1, h) * w..][..w];
        out[1..=w].copy_from_slice(src);
        out[0] = src[reflect(-1, w)];
        out[w + 1] = src[reflect(w as isize, w)];
    }
    let mut sq = vec![0f32; h * w];
    for (y, out) in sq.chunks_exact_mut(w).enumerate() {
        let (up, row, down) = (
            &pad[y * pw + 1..][..w],
            &pad[(y + 1) * pw..][..w + 2],
            &pad[(y + 2) * pw + 1..][..w],
        );
        for (x, o) in out.iter_mut().enumerate() {
            // OpenCV's float filter sums the kernel's nonzero taps in row-major order from 0.
            let v = ((((0.0 + up[x]) + row[x]) + -4.0 * row[x + 1]) + row[x + 2]) + down[x];
            *o = v * v;
        }
    }
    let mean = crate::stats::pairwise_sum_f32(&sq) / sq.len() as f32;
    // np.sqrt of the float32 mean, times a Python float, stays float32.
    (mean.sqrt() * 100.0) as f64
}

const SCOPE_HUE_BINS: usize = crate::histogram::SCOPE_HUE_BINS;
const SCOPE_CHROMA_BINS: usize = crate::histogram::SCOPE_CHROMA_BINS;
const SCOPE_MAX_CHROMA: f64 = 60.0;
const SCOPE_LOG_KNEE: f64 = 4.0;

fn scope_radius(c: f64, log: bool) -> f64 {
    if log {
        libm::log1p(c / SCOPE_LOG_KNEE) / libm::log1p(SCOPE_MAX_CHROMA / SCOPE_LOG_KNEE)
    } else {
        c / SCOPE_MAX_CHROMA
    }
}

fn scope_chroma(r: f64, log: bool) -> f64 {
    if log {
        libm::expm1(r * libm::log1p(SCOPE_MAX_CHROMA / SCOPE_LOG_KNEE)) * SCOPE_LOG_KNEE
    } else {
        r * SCOPE_MAX_CHROMA
    }
}

/// `_scope_bins` of one a*b* point.
fn scope_bin(a: f64, b: f64) -> i64 {
    use std::f64::consts::PI;
    let hue = (((libm::atan2(b, a) + PI) / (2.0 * PI) * SCOPE_HUE_BINS as f64).floor() as i64)
        .rem_euclid(SCOPE_HUE_BINS as i64);
    let radius = scope_radius(libm::hypot(a, b), true);
    let chroma = (radius * SCOPE_CHROMA_BINS as f64)
        .floor()
        .clamp(0.0, (SCOPE_CHROMA_BINS - 1) as f64) as i64;
    hue * SCOPE_CHROMA_BINS as i64 + chroma
}

/// `_scope_lookup(size, log)`: the polar bin of each display pixel (-1 outside the disc), each
/// bin's pixel count, the bins no pixel samples, and the pixel each of those folds into.
pub fn scope_lookup(size: usize, log: bool) -> (Vec<i64>, Vec<f64>, Vec<i64>, Vec<i64>) {
    let u: Vec<f64> = (0..size)
        .map(|i| (i as f64 + 0.5) / size as f64 * 2.0 - 1.0)
        .collect();
    let idx: Vec<i64> = (0..size)
        .into_par_iter()
        .flat_map_iter(|r| {
            let y = u[size - 1 - r];
            u.iter().map(move |&x| {
                let radius = libm::hypot(x, y);
                let chroma = scope_chroma(radius.min(1.0), log);
                let hue = libm::atan2(y, x);
                let bin = scope_bin(chroma * libm::cos(hue), chroma * libm::sin(hue));
                if radius > 1.0 { -1 } else { bin }
            })
        })
        .collect();
    let mut area = vec![0f64; SCOPE_HUE_BINS * SCOPE_CHROMA_BINS];
    for &i in idx.iter().filter(|&&i| i >= 0) {
        area[i as usize] += 1.0;
    }
    let orphans: Vec<i64> = (0..area.len() as i64)
        .filter(|&i| area[i as usize] == 0.0)
        .collect();
    let px = orphans
        .iter()
        .map(|&o| {
            use std::f64::consts::PI;
            let centre_hue =
                ((o / SCOPE_CHROMA_BINS as i64) as f64 + 0.5) / SCOPE_HUE_BINS as f64 * 2.0 * PI
                    - PI;
            let centre_c = scope_chroma(
                ((o % SCOPE_CHROMA_BINS as i64) as f64 + 0.5) / SCOPE_CHROMA_BINS as f64,
                true,
            );
            let r = scope_radius(centre_c, log).min(1.0);
            let last = (size - 1) as i64;
            let col =
                (((r * libm::cos(centre_hue) + 1.0) / 2.0 * size as f64) as i64).clamp(0, last);
            let row =
                (((1.0 - r * libm::sin(centre_hue)) / 2.0 * size as f64) as i64).clamp(0, last);
            row * size as i64 + col
        })
        .collect();
    (idx, area, orphans, px)
}

/// `vectorscope_density`: each polar bin's count spread over the display pixels it covers, and
/// the bins finer than a pixel added to the pixel they fold into.
pub fn vectorscope_density(
    counts: &[f64],
    (idx, area, orphans, px): (&[i64], &[f64], &[i64], &[i64]),
) -> Vec<f64> {
    let mut out: Vec<f64> = idx
        .iter()
        .map(|&i| {
            if i >= 0 {
                counts[i as usize] / area[i as usize].max(1.0)
            } else {
                0.0
            }
        })
        .collect();
    for (&o, &p) in orphans.iter().zip(px) {
        out[p as usize] += counts[o as usize];
    }
    out
}
