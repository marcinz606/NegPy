//! `register_pair` of negpy/features/stitch/logic.py: SIFT features, a brute-force L2 ratio
//! test and OpenCV 4.13's `estimateAffinePartial2D` (`ptsetreg.cpp`, `levmarq.cpp`,
//! Apache-2.0): RANSAC over 2-point similarity models, then Levenberg-Marquardt on the inliers.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::sift::{self, DESCR_LEN};

const NFEATURES: usize = 6000;
const RATIO: f64 = 0.75;
const REPROJ_THRESHOLD: f64 = 3.0;
const CONFIDENCE: f64 = 0.99;
const MAX_ITERS: usize = 2000;
const REFINE_ITERS: usize = 10;

/// The squared L2 distance of two descriptors; exact, as OpenCV's float32 sum of integers under
/// 2^24 is.
#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn dist2(a: &[u8; DESCR_LEN], b: &[u8; DESCR_LEN]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as i16 - y as i16;
            d as i32 * d as i32
        })
        .sum()
}

/// `BFMatcher().knnMatch(query, train, k=2)` and Lowe's ratio test: (query, train) index pairs
/// whose nearest train descriptor is nearer than RATIO times the second nearest. A tie keeps the
/// lower train index first, as OpenCV does.
pub fn ratio_matches(query: &[[u8; DESCR_LEN]], train: &[[u8; DESCR_LEN]]) -> Vec<(usize, usize)> {
    if train.len() < 2 {
        return Vec::new();
    }
    query
        .par_iter()
        .enumerate()
        .filter_map(|(qi, q)| {
            let (mut best, mut second) = ((i32::MAX, 0usize), i32::MAX);
            for (ti, t) in train.iter().enumerate() {
                let d = dist2(q, t);
                if d < best.0 {
                    second = best.0;
                    best = (d, ti);
                } else if d < second {
                    second = d;
                }
            }
            let (d1, d2) = ((best.0 as f32).sqrt() as f64, (second as f32).sqrt() as f64);
            (d1 < RATIO * d2).then_some((qi, best.1))
        })
        .collect()
}

/// OpenCV's `cv::RNG`, a multiply-with-carry generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = (self.0 as u32 as u64)
            .wrapping_mul(4_164_903_690)
            .wrapping_add(self.0 >> 32);
        self.0 as u32
    }

    fn uniform(&mut self, n: usize) -> usize {
        (self.next() % n as u32) as usize
    }
}

/// `RANSACUpdateNumIters`.
fn update_num_iters(p: f64, ep: f64, model_points: i32, max_iters: usize) -> usize {
    let (p, ep) = (p.clamp(0.0, 1.0), ep.clamp(0.0, 1.0));
    let num = (1.0 - p).max(f64::MIN_POSITIVE);
    let denom = 1.0 - libm::pow(1.0 - ep, model_points as f64);
    if denom < f64::MIN_POSITIVE {
        return 0;
    }
    let (num, denom) = (libm::log(num), libm::log(denom));
    if denom >= 0.0 || -num >= max_iters as f64 * -denom {
        max_iters
    } else {
        (num / denom).round_ties_even() as usize
    }
}

/// The 2-point similarity of `AffinePartial2DEstimatorCallback::runKernel`, as (a, b, tx, ty)
/// for the matrix [a -b tx; b a ty].
fn similarity(from: [[f32; 2]; 2], to: [[f32; 2]; 2]) -> [f64; 4] {
    let [[x1, y1], [x2, y2]] = from.map(|p| p.map(f64::from));
    let [[bx1, by1], [bx2, by2]] = to.map(|p| p.map(f64::from));
    let d = 1.0 / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    let s0 = d * ((bx1 - bx2) * (x1 - x2) + (by1 - by2) * (y1 - y2));
    let s1 = d * ((by1 - by2) * (x1 - x2) - (bx1 - bx2) * (y1 - y2));
    let s2 = d
        * ((by1 - by2) * (x1 * y2 - x2 * y1)
            - (bx1 * y2 - bx2 * y1) * (y1 - y2)
            - (bx1 * x2 - bx2 * x1) * (x1 - x2));
    let s3 = d
        * (-(bx1 - bx2) * (x1 * y2 - x2 * y1)
            - (by1 * x2 - by2 * x1) * (x1 - x2)
            - (by1 * y2 - by2 * y1) * (y1 - y2));
    [s0, s1, s2, s3]
}

/// `findInliers`: the float32 squared reprojection error against the threshold.
fn inliers(from: &[[f32; 2]], to: &[[f32; 2]], h: [f64; 4], mask: &mut [bool]) -> usize {
    let (a, b, tx, ty) = (h[0] as f32, h[1] as f32, h[2] as f32, h[3] as f32);
    let t = (REPROJ_THRESHOLD * REPROJ_THRESHOLD) as f32;
    let mut n = 0;
    for ((f, p), m) in from.iter().zip(to).zip(mask.iter_mut()) {
        let ex = a * f[0] - b * f[1] + tx - p[0];
        let ey = b * f[0] + a * f[1] + ty - p[1];
        *m = ex * ex + ey * ey <= t;
        n += *m as usize;
    }
    n
}

/// Gauss-Jordan solve of a 4x4 system with partial pivoting; None when singular.
fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for i in 0..4 {
        let p = (i..4).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs()))?;
        if a[p][i].abs() < f64::EPSILON * 1e-6 {
            return None;
        }
        a.swap(i, p);
        b.swap(i, p);
        let (pivot, pb) = (a[i], b[i]);
        for (j, (row, bj)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
            if j != i {
                let f = row[i] / pivot[i];
                for (x, &p) in row.iter_mut().zip(&pivot).skip(i) {
                    *x -= f * p;
                }
                *bj -= f * pb;
            }
        }
    }
    Some(std::array::from_fn(|i| b[i] / a[i][i]))
}

/// `LMSolver` (`levmarq.cpp`) over (a, b, tx, ty) with `AffinePartial2DRefineCallback`.
fn refine(from: &[[f32; 2]], to: &[[f32; 2]], x0: [f64; 4]) -> [f64; 4] {
    let residuals = |h: &[f64; 4]| -> Vec<f64> {
        from.iter()
            .zip(to)
            .flat_map(|(f, p)| {
                let (mx, my) = (f[0] as f64, f[1] as f64);
                [
                    h[0] * mx - h[1] * my + h[2] - p[0] as f64,
                    h[1] * mx + h[0] * my + h[3] - p[1] as f64,
                ]
            })
            .collect()
    };
    let jac_rows = |i: usize| -> [f64; 4] {
        let (mx, my) = (from[i / 2][0] as f64, from[i / 2][1] as f64);
        if i.is_multiple_of(2) {
            [mx, -my, 1.0, 0.0]
        } else {
            [my, mx, 0.0, 1.0]
        }
    };
    let n = 2 * from.len();
    let mut a = [[0f64; 4]; 4];
    for i in 0..n {
        let j = jac_rows(i);
        for r in 0..4 {
            for c in 0..4 {
                a[r][c] += j[r] * j[c];
            }
        }
    }
    let jt_r = |r: &[f64]| -> [f64; 4] {
        let mut v = [0f64; 4];
        for (i, &e) in r.iter().enumerate() {
            let j = jac_rows(i);
            for k in 0..4 {
                v[k] += j[k] * e;
            }
        }
        v
    };
    let norm2 = |r: &[f64]| r.iter().map(|e| e * e).sum::<f64>();
    let inf = |r: &[f64]| r.iter().fold(0f64, |m, e| m.max(e.abs()));
    let eps = f32::EPSILON as f64;

    let mut x = x0;
    let mut r = residuals(&x);
    let mut s = norm2(&r);
    let mut v = jt_r(&r);
    let diag: [f64; 4] = std::array::from_fn(|i| a[i][i]);
    let (rlo, rhi) = (0.25, 0.75);
    let (mut lambda, mut lc) = (1.0f64, 0.75f64);
    let mut iter = 0;
    loop {
        let mut ap = a;
        for i in 0..4 {
            ap[i][i] += lambda * diag[i];
        }
        let Some(d) = solve4(ap, v) else { break };
        let xd: [f64; 4] = std::array::from_fn(|i| x[i] - d[i]);
        let rd = residuals(&xd);
        let sd = norm2(&rd);
        let ad: [f64; 4] = std::array::from_fn(|i| (0..4).map(|k| a[i][k] * d[k]).sum::<f64>());
        let ds: f64 = (0..4).map(|i| d[i] * (2.0 * v[i] - ad[i])).sum();
        let ratio = (s - sd) / if ds.abs() > f64::EPSILON { ds } else { 1.0 };
        if ratio > rhi {
            lambda *= 0.5;
            if lambda < lc {
                lambda = 0.0;
            }
        } else if ratio < rlo {
            let t: f64 = (0..4).map(|i| d[i] * v[i]).sum();
            let mut nu = (sd - s) / if t.abs() > f64::EPSILON { t } else { 1.0 } + 2.0;
            nu = nu.clamp(2.0, 10.0);
            if lambda == 0.0 {
                let maxval = (0..4)
                    .filter_map(|i| {
                        let mut e = [0f64; 4];
                        e[i] = 1.0;
                        solve4(a, e).map(|col| col[i].abs())
                    })
                    .fold(f64::EPSILON, f64::max);
                lambda = 1.0 / maxval;
                lc = lambda;
                nu *= 0.5;
            }
            lambda *= nu;
        }
        if sd < s {
            s = sd;
            x = xd;
            r = rd;
            v = jt_r(&r);
        }
        iter += 1;
        if !(iter < REFINE_ITERS && inf(&d) >= eps && inf(&r) >= eps) {
            break;
        }
    }
    x
}

/// `cv2.estimateAffinePartial2D(from, to, method=RANSAC, ransacReprojThreshold=3)`: the matrix
/// as (a, b, tx, ty) and the inlier mask, or None when no model is found.
pub fn estimate_affine_partial(
    from: &[[f32; 2]],
    to: &[[f32; 2]],
) -> Option<([f64; 4], Vec<bool>)> {
    let count = from.len();
    if count < 2 || to.len() != count {
        return None;
    }
    if count == 2 {
        return Some((
            similarity([from[0], from[1]], [to[0], to[1]]),
            vec![true; 2],
        ));
    }
    let mut rng = Rng(u64::MAX);
    let (mut best, mut best_mask) = (None, vec![false; count]);
    let mut mask = vec![false; count];
    let (mut niters, mut max_good) = (MAX_ITERS, 0usize);
    let mut iter = 0;
    while iter < niters {
        let i0 = rng.uniform(count);
        let mut i1 = rng.uniform(count);
        while i1 == i0 {
            i1 = rng.uniform(count);
        }
        let h = similarity([from[i0], from[i1]], [to[i0], to[i1]]);
        let good = inliers(from, to, h, &mut mask);
        if good > max_good.max(1) {
            std::mem::swap(&mut mask, &mut best_mask);
            best = Some(h);
            max_good = good;
            niters = update_num_iters(CONFIDENCE, (count - good) as f64 / count as f64, 2, niters);
        }
        iter += 1;
    }
    let h = best?;
    let (src, dst): (Vec<[f32; 2]>, Vec<[f32; 2]>) = from
        .iter()
        .zip(to)
        .zip(&best_mask)
        .filter(|(_, m)| **m)
        .map(|((f, t), _)| (*f, *t))
        .unzip();
    Some((refine(&src, &dst, h), best_mask))
}

/// `register_pair`: the 2x3 similarity mapping `mov` coordinates to `ref_` coordinates and its
/// RANSAC inlier count, or None when the pair has fewer than `min_inliers` good matches or
/// inliers.
pub fn register_pair(
    ref_: (&[u8], usize, usize),
    mov: (&[u8], usize, usize),
    min_inliers: usize,
) -> Option<([f64; 6], usize)> {
    let ((ref_kp, ref_desc), (mov_kp, mov_desc)) = rayon::join(
        || sift::detect_and_compute(ref_.0, ref_.1, ref_.2, NFEATURES),
        || sift::detect_and_compute(mov.0, mov.1, mov.2, NFEATURES),
    );
    if ref_kp.len() < 2 || mov_kp.len() < 2 {
        return None;
    }
    let good = ratio_matches(&mov_desc, &ref_desc);
    if good.len() < min_inliers {
        return None;
    }
    let mov_pts: Vec<[f32; 2]> = good
        .iter()
        .map(|&(q, _)| [mov_kp[q].x, mov_kp[q].y])
        .collect();
    let ref_pts: Vec<[f32; 2]> = good
        .iter()
        .map(|&(_, t)| [ref_kp[t].x, ref_kp[t].y])
        .collect();
    let (h, mask) = estimate_affine_partial(&mov_pts, &ref_pts)?;
    let n = mask.iter().filter(|&&m| m).count();
    if n < min_inliers {
        return None;
    }
    Some(([h[0], -h[1], h[2], h[1], h[0], h[3]], n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_opencv() {
        // cv::RNG((uint64)-1): the first draws of rng.next().
        let mut rng = Rng(u64::MAX);
        let first = rng.next();
        assert_eq!(
            first,
            (u32::MAX as u64 * 4_164_903_690 + u32::MAX as u64) as u32
        );
    }

    #[test]
    fn ransac_recovers_a_similarity_among_outliers() {
        let (a, b, tx, ty) = (0.99f64, 0.03f64, 12.5f64, -7.25f64);
        let mut from = Vec::new();
        let mut to = Vec::new();
        for i in 0..200 {
            let (x, y) = ((i * 37 % 400) as f64, (i * 91 % 300) as f64);
            from.push([x as f32, y as f32]);
            let (u, v) = if i % 4 == 0 {
                ((i * 53 % 400) as f64, (i * 17 % 300) as f64)
            } else {
                (a * x - b * y + tx, b * x + a * y + ty)
            };
            to.push([u as f32, v as f32]);
        }
        let (h, mask) = estimate_affine_partial(&from, &to).unwrap();
        assert_eq!(mask.iter().filter(|&&m| m).count(), 150);
        for (got, want) in h.iter().zip([a, b, tx, ty]) {
            assert!((got - want).abs() < 1e-3, "{h:?}");
        }
    }

    #[test]
    fn iteration_count_follows_opencv() {
        assert_eq!(update_num_iters(0.99, 0.0, 2, 2000), 0);
        assert_eq!(update_num_iters(0.99, 0.5, 2, 2000), 16);
        assert_eq!(update_num_iters(0.99, 1.0, 2, 2000), 2000);
    }
}
