//! The embedded lens warps of negpy/features/lens/warps.py: inverse maps, in NumPy's operation
//! order and dtypes.

use rayon::prelude::*;

/// The frame geometry of `_coordinates`, its scalars as Python computes them.
#[derive(Clone, Copy)]
pub struct Frame {
    /// Buffer-area origin and the buffer-to-image scale per axis.
    pub origin: (f64, f64),
    pub scale: (f64, f64),
    /// The warp center and radius in buffer pixels.
    pub center: (f64, f64),
    pub radius: f64,
    pub fill: f64,
}

/// A normalized coordinate of `_coordinates`: float32 up to the division by the float64 radius.
#[inline(always)]
fn normalized(i: usize, origin: f64, scale: f64, center: f64, radius: f64, fill: f64) -> f64 {
    let v = ((origin as f32 + (i as f32 + 0.5) * scale as f32) - 0.5) - center as f32;
    v as f64 / radius * fill
}

/// `RectilinearWarp.remap` with the distortion applied: the map of rows `rows` of a w-wide
/// frame through `coefficients` (k0..k3, t0, t1), into `mx` and `my`.
pub fn rectilinear_map(
    w: usize,
    rows: std::ops::Range<usize>,
    coefficients: [f64; 6],
    f: Frame,
    mx: &mut [f32],
    my: &mut [f32],
) {
    let [k0, k1, k2, k3, t0, t1] = coefficients;
    let xs: Vec<f64> = (0..w)
        .map(|i| normalized(i, f.origin.0, f.scale.0, f.center.0, f.radius, f.fill))
        .collect();
    let start = rows.start;
    mx.par_chunks_mut(w)
        .zip(my.par_chunks_mut(w))
        .enumerate()
        .for_each(|(r, (ox, oy))| {
            let y = normalized(
                start + r,
                f.origin.1,
                f.scale.1,
                f.center.1,
                f.radius,
                f.fill,
            );
            for ((ox, oy), &x) in ox.iter_mut().zip(oy.iter_mut()).zip(&xs) {
                let r2 = x * x + y * y;
                let factor = k0 + r2 * (k1 + r2 * (k2 + r2 * k3));
                let wx = x * factor + 2.0 * t0 * x * y + t1 * (r2 + 2.0 * x * x);
                let wy = y * factor + 2.0 * t1 * x * y + t0 * (r2 + 2.0 * y * y);
                *ox =
                    ((((f.center.0 + f.radius * wx) - f.origin.0) + 0.5) / f.scale.0 - 0.5) as f32;
                *oy =
                    ((((f.center.1 + f.radius * wy) - f.origin.1) + 0.5) / f.scale.1 - 0.5) as f32;
            }
        });
}

/// `np.hypot` of float32, which is correctly rounded.
#[inline(always)]
fn hypotf(x: f32, y: f32) -> f32 {
    let (x, y) = (x as f64, y as f64);
    (x * x + y * y).sqrt() as f32
}

/// The float32 centered coordinates of the Panasonic and Sony models.
fn centered(n: usize, start: usize, len: usize, fill: f32) -> Vec<f32> {
    let half = (n as f64 * 0.5) as f32;
    (start..start + len)
        .map(|i| (i as f32 - half) * fill)
        .collect()
}

/// `PanasonicWarp.remap` with the distortion applied, for rows `rows` of an h x w frame:
/// Ru -> Rd by fixed-point iteration over the whole block, stopping when no pixel moves 1e-7.
pub fn panasonic_map(
    (h, w): (usize, usize),
    rows: std::ops::Range<usize>,
    (scale, a, b, c): (f64, f64, f64, f64),
    fill: f64,
    mx: &mut [f32],
    my: &mut [f32],
) {
    let xs = centered(w, 0, w, fill as f32);
    let ys = centered(h, rows.start, rows.len(), fill as f32);
    let half_diag = (w as f64 * 0.5).hypot(h as f64 * 0.5);
    let ru: Vec<f64> = ys
        .iter()
        .flat_map(|&y| xs.iter().map(move |&x| hypotf(x, y) as f64 / half_diag))
        .collect();
    let mut rd = ru.clone();
    for _ in 0..8 {
        let new: Vec<f64> = rd
            .par_iter()
            .zip(&ru)
            .map(|(&rd, &ru)| {
                let r2 = rd * rd;
                let f = 1.0 + scale * (a * r2 + b * r2 * r2 + c * r2 * r2 * r2);
                ru / if f > 0.0 { f } else { 1.0 }
            })
            .collect();
        let moved = new
            .par_iter()
            .zip(&rd)
            .map(|(&n, &r)| (n - r).abs())
            .reduce(
                || f64::NEG_INFINITY,
                |p, q| if q > p || q.is_nan() { q } else { p },
            );
        if moved < 1e-7 {
            break;
        }
        rd = new;
    }
    let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
    mx.par_chunks_mut(w)
        .zip(my.par_chunks_mut(w))
        .enumerate()
        .for_each(|(r, (ox, oy))| {
            let y = ys[r] as f64;
            for (i, (ox, oy)) in ox.iter_mut().zip(oy.iter_mut()).enumerate() {
                let (ru, rd) = (ru[r * w + i], rd[r * w + i]);
                let dr = if ru > 0.0 { rd / ru } else { 1.0 };
                *ox = (xs[i] as f64 * dr + cx) as f32;
                *oy = (y * dr + cy) as f32;
            }
        });
}

/// `np.interp(x, xp, fp)` of float64, with NumPy's rules at the anchors and its slope order.
fn interp(x: f64, xp: &[f64], fp: &[f64], slopes: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let v = slopes[j] * (x - xp[j]) + fp[j];
    if !v.is_nan() {
        return v;
    }
    let v = slopes[j] * (x - xp[j + 1]) + fp[j + 1];
    if v.is_nan() && fp[j] == fp[j + 1] {
        fp[j]
    } else {
        v
    }
}

/// `SonyWarp.remap` for rows `rows` of an h x w frame, its knot factors already combined.
pub fn sony_map(
    (h, w): (usize, usize),
    rows: std::ops::Range<usize>,
    factors: &[f64],
    fill: f64,
    mx: &mut [f32],
    my: &mut [f32],
) {
    let n = factors.len();
    let knots: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / (n - 1) as f64).collect();
    let slopes: Vec<f64> = (0..n.saturating_sub(1))
        .map(|j| (factors[j + 1] - factors[j]) / (knots[j + 1] - knots[j]))
        .collect();
    let xs = centered(w, 0, w, fill as f32);
    let ys = centered(h, rows.start, rows.len(), fill as f32);
    let half_diag = (w as f64 * 0.5).hypot(h as f64 * 0.5);
    let (cx, cy) = ((w as f64 * 0.5) as f32, (h as f64 * 0.5) as f32);
    mx.par_chunks_mut(w)
        .zip(my.par_chunks_mut(w))
        .enumerate()
        .for_each(|(r, (ox, oy))| {
            let y = ys[r];
            for ((ox, oy), &x) in ox.iter_mut().zip(oy.iter_mut()).zip(&xs) {
                let radius = hypotf(x, y) as f64 / half_diag;
                let factor = interp(radius, &knots, factors, &slopes) as f32;
                *ox = x * factor + cx;
                *oy = y * factor + cy;
            }
        });
}

/// `_rectilinear_xy` for one float64 point.
#[inline(always)]
fn rectilinear_xy(x: f64, y: f64, [k0, k1, k2, k3, t0, t1]: [f64; 6]) -> (f64, f64) {
    let r2 = x * x + y * y;
    let factor = k0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    (
        x * factor + 2.0 * t0 * x * y + t1 * (r2 + 2.0 * x * x),
        y * factor + 2.0 * t1 * x * y + t0 * (r2 + 2.0 * y * y),
    )
}

/// `cv2.undistortPointsIter` of one point with camera matrix diag(f, f, 1), distortion
/// (k1, k2, p1, p2, k3), no R or P, and (COUNT | EPS, `max_count`, `eps`): OpenCV's damped
/// fixed-point iteration, its operation order kept.
fn undistort_point(u: f64, v: f64, f: f64, k: [f64; 5], max_count: usize, eps: f64) -> (f64, f64) {
    let [k1, k2, p1, p2, k3] = k;
    let ifx = 1.0 / f;
    let (x0, y0) = ((u - 0.0) * ifx, (v - 0.0) * ify(f));
    let (mut x, mut y) = (x0, y0);
    let (mut error, mut prev_error) = (f64::MAX, f64::MAX);
    let mut alpha = 1.0;
    let mut j = 0;
    loop {
        if j >= max_count || error < eps {
            break;
        }
        let r2 = x * x + y * y;
        let icdist =
            (1.0 + ((0.0 * r2 + 0.0) * r2 + 0.0) * r2) / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2);
        if icdist < 0.0 {
            return ((u - 0.0) * ifx, (v - 0.0) * ify(f));
        }
        let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + 0.0 * r2 + 0.0 * r2 * r2;
        let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + 0.0 * r2 + 0.0 * r2 * r2;
        let nx = (1.0 - alpha) * x + alpha * (x0 - dx) * icdist;
        let ny = (1.0 - alpha) * y + alpha * (y0 - dy) * icdist;
        let r2 = nx * nx + ny * ny;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let a1 = 2.0 * nx * ny;
        let a2 = r2 + 2.0 * nx * nx;
        let a3 = r2 + 2.0 * ny * ny;
        let cdist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let icdist2 = 1.0 / (1.0 + 0.0 * r2 + 0.0 * r4 + 0.0 * r6);
        let xd0 = nx * cdist * icdist2 + p1 * a1 + p2 * a2 + 0.0 * r2 + 0.0 * r4;
        let yd0 = ny * cdist * icdist2 + p1 * a3 + p2 * a1 + 0.0 * r2 + 0.0 * r4;
        // The identity tilt: 1 * v + 0 * w + 0 * 1, then over 0 * . + 0 * . + 1 * 1.
        let xd = 1.0 * (1.0 * xd0 + 0.0 * yd0 + 0.0);
        let yd = 1.0 * (0.0 * xd0 + 1.0 * yd0 + 0.0);
        let (xp, yp) = (xd * f + 0.0, yd * f + 0.0);
        error = ((xp - u) * (xp - u) + (yp - v) * (yp - v)).sqrt();
        if error > prev_error {
            alpha *= 0.5;
        } else {
            x = nx;
            y = ny;
        }
        prev_error = error;
        j += 1;
    }
    (x, y)
}

#[inline(always)]
fn ify(f: f64) -> f64 {
    1.0 / f
}

/// `RectilinearWarp.remap` with lateral CA only: each point taken back through the green
/// (`common`) model, clamped to the calibrated radius, through the channel's model, and the
/// channel's offset from green scaled onto the point's own radius.
pub fn rectilinear_ca_map(
    w: usize,
    rows: std::ops::Range<usize>,
    (selected, common): ([f64; 6], [f64; 6]),
    f: Frame,
    mx: &mut [f32],
    my: &mut [f32],
) {
    let [g0, g1, g2, g3, t0, t1] = common;
    let k = [g1 / g0, g2 / g0, t0 / g0, t1 / g0, g3 / g0];
    let xs: Vec<f64> = (0..w)
        .map(|i| normalized(i, f.origin.0, f.scale.0, f.center.0, f.radius, f.fill))
        .collect();
    let start = rows.start;
    mx.par_chunks_mut(w)
        .zip(my.par_chunks_mut(w))
        .enumerate()
        .for_each(|(r, (ox, oy))| {
            let ty = normalized(
                start + r,
                f.origin.1,
                f.scale.1,
                f.center.1,
                f.radius,
                f.fill,
            );
            for ((ox, oy), &tx) in ox.iter_mut().zip(oy.iter_mut()).zip(&xs) {
                let (x, y) = undistort_point(tx, ty, g0, k, 50, 1e-8);
                let h = x.hypot(y);
                let limit = if h.is_nan() || h > 1.0 { h } else { 1.0 };
                let (x, y) = (x / limit, y / limit);
                let (wx, wy) = rectilinear_xy(x, y, selected);
                let (gx, gy) = rectilinear_xy(x, y, common);
                let gh = gx.hypot(gy);
                let scale = tx.hypot(ty) / if gh.is_nan() || gh > 1e-8 { gh } else { 1e-8 };
                let (wx, wy) = (tx + (wx - gx) * scale, ty + (wy - gy) * scale);
                *ox =
                    ((((f.center.0 + f.radius * wx) - f.origin.0) + 0.5) / f.scale.0 - 0.5) as f32;
                *oy =
                    ((((f.center.1 + f.radius * wy) - f.origin.1) + 0.5) / f.scale.1 - 0.5) as f32;
            }
        });
}
