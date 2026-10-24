//! `cv2.inpaint(..., cv2.INPAINT_NS)` of an 8-bit 3-channel image (OpenCV 4.13
//! `modules/photo/src/inpaint.cpp`, Apache-2.0), and the hair fill of
//! negpy/features/retouch/logic.py that runs it tile by tile.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;

const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;

/// OpenCV's `FastMarching_solve`, in float64.
fn solve(f: &[u8], t: &[f32], a: usize, b: usize) -> f32 {
    let (a11, a22) = (t[a] as f64, t[b] as f64);
    let m12 = a11.min(a22);
    let sol = match (f[a] != INSIDE, f[b] != INSIDE) {
        (true, true) if (a11 - a22).abs() >= 1.0 => 1.0 + m12,
        (true, true) => (a11 + a22 + (2.0 - (a11 - a22) * (a11 - a22)).sqrt()) * 0.5,
        (true, false) => 1.0 + a11,
        (false, true) => 1.0 + a22,
        (false, false) => 1.0 + m12,
    };
    sol as f32
}

/// `cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)` of an HxWx3 uint8 image, in place: the
/// fast-marching order and the float32 weights of OpenCV, ties popped in insertion order.
pub fn inpaint_ns(img: &mut [u8], mask: &[u8], (h, w): (usize, usize), radius: f64) {
    let range = (radius.round_ties_even() as i64).clamp(1, 100) as isize;
    let (rows, cols) = (h + 2, w + 2);
    let mut f = vec![KNOWN; rows * cols];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] != 0 {
                f[(y + 1) * cols + x + 1] = INSIDE;
            }
        }
    }
    let mut t = vec![1.0e6f32; rows * cols];
    let mut heap = BinaryHeap::new();
    let mut order = 0u64;
    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            let p = y * cols + x;
            let cross = [p - cols, p - 1, p + 1, p + cols];
            if f[p] == KNOWN && cross.iter().any(|&q| f[q] == INSIDE) {
                heap.push(Reverse((0u32, order, p)));
                order += 1;
            }
        }
    }
    for &Reverse((_, _, p)) in heap.iter() {
        f[p] = BAND;
        t[p] = 0.0;
    }
    let (ri, ci) = (rows as isize, cols as isize);
    while let Some(Reverse((_, _, p))) = heap.pop() {
        f[p] = KNOWN;
        let (ii, jj) = ((p / cols) as isize, (p % cols) as isize);
        for (i, j) in [(ii - 1, jj), (ii, jj - 1), (ii + 1, jj), (ii, jj + 1)] {
            if i <= 0 || j <= 0 || i > ri - 1 || j > ci - 1 {
                continue;
            }
            let q = (i * ci + j) as usize;
            if f[q] != INSIDE {
                continue;
            }
            let (up, dn, lf, rt) = (q - cols, q + cols, q - 1, q + 1);
            let dist = solve(&f, &t, up, lf)
                .min(solve(&f, &t, dn, lf))
                .min(solve(&f, &t, up, rt).min(solve(&f, &t, dn, rt)));
            t[q] = dist;
            let mut ia = [0f32; 3];
            let mut s = [1.0e-20f32; 3];
            for k in i - range..=i + range {
                if k <= 0 || k >= ri - 1 {
                    continue;
                }
                let km = k - 1 + (k == 1) as isize;
                let kp = k - 1 - (k == ri - 2) as isize;
                for l in j - range..=j + range {
                    if l <= 0 || l >= ci - 1 {
                        continue;
                    }
                    let kl = (k * ci + l) as usize;
                    if f[kl] == INSIDE || (l - j) * (l - j) + (k - i) * (k - i) > range * range {
                        continue;
                    }
                    let lm = l - 1 + (l == 1) as isize;
                    let lp = l - 1 - (l == ci - 2) as isize;
                    let (ry, rx) = ((k - i) as f32, (l - j) as f32);
                    let rl = rx * rx + ry * ry;
                    let dst = 1.0 / (rl * rl + 1.0);
                    let (dn_k, up_k) = (f[kl + cols] != INSIDE, f[kl - cols] != INSIDE);
                    let (rt_l, lf_l) = (f[kl + 1] != INSIDE, f[kl - 1] != INSIDE);
                    // Clamped: on a one-pixel-wide image OpenCV reads out of bounds here.
                    let at = |y: isize, x: isize| {
                        3 * (y.clamp(0, h as isize - 1) as usize * w
                            + x.clamp(0, w as isize - 1) as usize)
                    };
                    let (below, mid, above) = (at(kp + 1, lm), at(kp, lm), at(km - 1, lm));
                    let (right, centre, left) = (at(km, lp + 1), at(km, lm), at(km, lm - 1));
                    let here = at(k - 1, l - 1);
                    for c in 0..3 {
                        let v = |o: usize| img[o + c] as i32;
                        let gx = match (dn_k, up_k) {
                            (true, true) => {
                                ((v(below) - v(mid)).abs() + (v(mid) - v(above)).abs()) as f32
                            }
                            (true, false) => (v(below) - v(mid)).abs() as f32 * 2.0,
                            (false, true) => (v(mid) - v(above)).abs() as f32 * 2.0,
                            (false, false) => 0.0,
                        };
                        let gy = match (rt_l, lf_l) {
                            (true, true) => {
                                ((v(right) - v(centre)).abs() + (v(centre) - v(left)).abs()) as f32
                            }
                            (true, false) => (v(right) - v(centre)).abs() as f32 * 2.0,
                            (false, true) => (v(centre) - v(left)).abs() as f32 * 2.0,
                            (false, false) => 0.0,
                        };
                        let gx = -gx;
                        let dot = rx * gx + ry * gy;
                        let dir = if (dot as f64).abs() <= 0.01 {
                            0.000001f32
                        } else {
                            (dot / (rl * (gx * gx + gy * gy)).sqrt()).abs()
                        };
                        let wgt = dst * dir;
                        ia[c] += wgt * v(here) as f32;
                        s[c] += wgt;
                    }
                }
            }
            let o = ((i - 1) as usize * w + (j - 1) as usize) * 3;
            for c in 0..3 {
                img[o + c] = (ia[c] as f64 / s[c] as f64)
                    .round_ties_even()
                    .clamp(0.0, 255.0) as u8;
            }
            f[q] = BAND;
            heap.push(Reverse((dist.to_bits(), order, q)));
            order += 1;
        }
    }
}

/// The constants of `apply_hair_inpaint`.
pub struct HairFill {
    pub radius: f64,
    pub dilate_px: usize,
    pub pad: usize,
    pub tile: usize,
    pub halo: usize,
    pub sample_px: usize,
    pub gamma: f64,
}

/// One tile of one component: the frame rectangle the inpaint sees and the core it writes.
struct Job {
    label: i32,
    lo: f32,
    inv_span: f32,
    lut: [f32; 256],
    around: [usize; 4],
    core: [usize; 4],
}

/// `apply_hair_inpaint` of an HxWx3 image and its dilated hair mask, up to the clean grain:
/// each 8-connected component is filled in its padded box, encoded against the box's clean
/// range, in tiles with a halo, and feathered over the dilate band. Writes only the filled
/// pixels of `out`; tiles run in parallel, as their written pixels never overlap.
pub fn hair_fill(src: &[f32], (h, w): (usize, usize), m: &[u8], c: &HairFill, out: &mut [f32]) {
    let comps = crate::components::connected_components(m, h, w, 8);
    let labels = &comps.labels;
    let jobs: Vec<Job> = (1..comps.count)
        .into_par_iter()
        .flat_map_iter(|i| component_jobs(src, (h, w), m, labels, i as i32, comps.stats[i], c))
        .collect();
    let writes: Vec<Vec<(usize, [f32; 3])>> = jobs
        .par_iter()
        .map(|j| fill_tile(src, (h, w), m, labels, j, c))
        .collect();
    for (p, v) in writes.into_iter().flatten() {
        out[3 * p..3 * p + 3].copy_from_slice(&v);
    }
}

fn component_jobs(
    src: &[f32],
    (h, w): (usize, usize),
    m: &[u8],
    labels: &[i32],
    label: i32,
    [bx, by, bw, bh, _]: [i32; 5],
    c: &HairFill,
) -> Vec<Job> {
    let (bx, by, bw, bh) = (bx as usize, by as usize, bw as usize, bh as usize);
    let (x0, y0) = (bx.saturating_sub(c.pad), by.saturating_sub(c.pad));
    let (x1, y1) = (w.min(bx + bw + c.pad), h.min(by + bh + c.pad));
    let (ch, cw) = (y1 - y0, x1 - x0);
    // A hair can join enough specks to span the frame: the range comes off a strided sample.
    let step = ((ch * cw / c.sample_px).max(1)).isqrt().max(1);
    let mut ctx = Vec::new();
    for y in (y0..y1).step_by(step) {
        for x in (x0..x1).step_by(step) {
            if m[y * w + x] == 0 {
                ctx.extend_from_slice(&src[3 * (y * w + x)..3 * (y * w + x) + 3]);
            }
        }
    }
    let (lo, hi) = match crate::stats::percentiles(&ctx, &[0.5, 99.5]) {
        Some(p) => (p[0] as f64, p[1] as f64),
        None => (0.0, 1.0),
    };
    let span = (hi - lo).max(1e-4);
    let lut = std::array::from_fn(|v| {
        let e = crate::mathf::pow_f64((v as f32 / 255.0) as f64, c.gamma as f32 as f64) as f32;
        e * span as f32 + lo as f32
    });
    let (nty, ntx) = (ch.div_ceil(c.tile), cw.div_ceil(c.tile));
    let mut used = vec![false; nty * ntx];
    for y in by..by + bh {
        for x in bx..bx + bw {
            if labels[y * w + x] == label {
                used[(y - y0) / c.tile * ntx + (x - x0) / c.tile] = true;
            }
        }
    }
    let mut jobs = Vec::new();
    for (t, _) in used.iter().enumerate().filter(|(_, u)| **u) {
        let (ty0, tx0) = (y0 + t / ntx * c.tile, x0 + t % ntx * c.tile);
        let (ty1, tx1) = (y1.min(ty0 + c.tile), x1.min(tx0 + c.tile));
        jobs.push(Job {
            label,
            lo: lo as f32,
            inv_span: (1.0 / span) as f32,
            lut,
            around: [
                y0.max(ty0.saturating_sub(c.halo)),
                y1.min(ty1 + c.halo),
                x0.max(tx0.saturating_sub(c.halo)),
                x1.min(tx1 + c.halo),
            ],
            core: [ty0, ty1, tx0, tx1],
        });
    }
    jobs
}

fn fill_tile(
    src: &[f32],
    (_, w): (usize, usize),
    m: &[u8],
    labels: &[i32],
    j: &Job,
    c: &HairFill,
) -> Vec<(usize, [f32; 3])> {
    let [ay0, ay1, ax0, ax1] = j.around;
    let (th, tw) = (ay1 - ay0, ax1 - ax0);
    let mut tile = Vec::with_capacity(3 * th * tw);
    let mut sub = Vec::with_capacity(th * tw);
    let mut mine = Vec::with_capacity(th * tw);
    for y in ay0..ay1 {
        tile.extend_from_slice(&src[3 * (y * w + ax0)..3 * (y * w + ax1)]);
        sub.extend_from_slice(&m[y * w + ax0..y * w + ax1]);
        mine.extend(
            labels[y * w + ax0..y * w + ax1]
                .iter()
                .map(|&l| (l == j.label) as u8),
        );
    }
    let mut enc = vec![0u8; 3 * th * tw];
    crate::retouch::hair_encode(&tile, 3, j.lo, j.inv_span, (1.0 / c.gamma) as f32, &mut enc);
    inpaint_ns(&mut enc, &sub, (th, tw), c.radius);
    let d = crate::distance::distance_transform(&mine, th, tw, crate::distance::Metric::Chessboard);
    let feather = (c.dilate_px + 1) as f32;
    let [ty0, ty1, tx0, tx1] = j.core;
    let mut writes = Vec::new();
    for y in ty0..ty1 {
        for x in tx0..tx1 {
            let p = (y - ay0) * tw + x - ax0;
            if mine[p] == 0 {
                continue;
            }
            let a = (d[p] / feather).min(1.0);
            let v = std::array::from_fn(|k| {
                tile[3 * p + k] * (1.0 - a) + j.lut[enc[3 * p + k] as usize] * a
            });
            writes.push((y * w + x, v));
        }
    }
    writes
}
