//! The profile reads of negpy/features/geometry/batch_autocrop.py and the occupancy window of
//! geometry/logic.py.

use crate::filter::{Border, gaussian_blur, sep_filter2d};
use crate::stats::{median_f64, np_median, np_percentiles};

/// `round` of a float64 as Python rounds it: ties to even.
fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// Least-squares line through the points, as `skew::polyfit1`.
fn polyfit1(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
    }
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

/// `_top_edge_slope(lum, roi)`: the residual tilt of the film box's top edge in degrees, from
/// the strongest vertical gradient per column near it; None when unreadable.
pub fn top_edge_slope(
    lum: &[f32],
    (h, w): (usize, usize),
    (y1, x1, x2): (i64, i64, i64),
) -> Option<f64> {
    let (left, right) = (x1 + 20, x2 - 20);
    let columns: Vec<i64> = (left..right).step_by(3).collect();
    let min_points = 20i64.max(py_round((right - left) as f64 * 0.08)) as usize;
    if columns.len() < min_points {
        return None;
    }
    let radius = 12i64.max(py_round(h as f64 * 0.035));
    let (low, high) = (
        (y1 - radius).max(0) as usize,
        (y1 + radius).min(h as i64) as usize,
    );
    if high < low + 4 {
        return None;
    }
    // The blur and the 3-tap derivative reach 7 rows; a slab 8 rows wider reads like the frame.
    let (s0, s1) = (low.saturating_sub(8), (high + 8).min(h));
    let slab = &lum[s0 * w..s1 * w];
    let rows = s1 - s0;
    let mut blur = vec![0f32; slab.len()];
    gaussian_blur(
        slab,
        &mut blur,
        (rows, w, 1),
        (0, 0),
        (1.5, 0.0),
        Border::Reflect101,
    )?;
    let mut grad = vec![0f32; slab.len()];
    sep_filter2d(
        &blur,
        &mut grad,
        (rows, w, 1),
        (&[1.0, 2.0, 1.0], &[-1.0, 0.0, 1.0]),
        Border::Reflect101,
    )?;
    let at = |y: usize, x: i64| grad[(y - s0) * w + x as usize].abs();
    let mut band = Vec::with_capacity((high - low) * columns.len());
    let mut peaks = Vec::with_capacity(columns.len());
    let mut strength = Vec::with_capacity(columns.len());
    for y in low..high {
        band.extend(columns.iter().map(|&x| at(y, x)));
    }
    for (j, _) in columns.iter().enumerate() {
        let (mut best, mut row) = (f32::NEG_INFINITY, 0usize);
        for (i, y) in (low..high).enumerate() {
            let v = band[i * columns.len() + j];
            if v > best {
                best = v;
                row = y;
            }
        }
        peaks.push(row as f64);
        strength.push(best);
    }
    let med_strength = np_median(&mut strength.clone()) as f64;
    if med_strength < 1e-6f64.max(3.0 * np_median(&mut band) as f64) {
        return None;
    }
    let bar = np_percentiles(&strength, &[60.0])?[0];
    let (mut xs, mut ys): (Vec<f64>, Vec<f64>) = columns
        .iter()
        .zip(&peaks)
        .zip(&strength)
        .filter(|&(_, &s)| s >= bar)
        .map(|((&x, &y), _)| (x as f64, y))
        .unzip();
    if xs.len() < min_points {
        return None;
    }
    let mut residuals = Vec::new();
    let mut slope = 0.0;
    for _ in 0..5 {
        let intercept;
        (slope, intercept) = polyfit1(&xs, &ys);
        residuals = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| (y - (slope * x + intercept)).abs())
            .collect();
        let bar = 2f64.max(median_f64(&mut residuals.clone()) * 2.5);
        let keep: Vec<bool> = residuals.iter().map(|&r| r <= bar).collect();
        let kept = keep.iter().filter(|&&k| k).count();
        if kept == keep.len() || kept < min_points {
            break;
        }
        (xs, ys) = (0..xs.len())
            .filter(|&i| keep[i])
            .map(|i| (xs[i], ys[i]))
            .unzip();
    }
    if median_f64(&mut residuals) > 2.0 {
        return None;
    }
    Some(slope.atan().to_degrees())
}

/// The roll template fields `_rect_from_edge_profile` reads.
pub struct Template {
    pub center_x: f64,
    pub width: f64,
    pub fallback_width: f64,
    pub top: f64,
    pub height: f64,
    pub center_mad: f64,
    pub width_mad: f64,
}

const MIN_PROFILE_CONTRAST: f64 = 0.06;
/// How near the roll's film width a pair of edge-profile peaks must span before their
/// agreement replaces raw peak strength.
const EDGE_PAIR_WIDTH_TOL: f64 = 0.02;

fn clamp_rect((x1, y1, x2, y2): (f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
    let c = |v: f64| v.clamp(0.0, 1.0);
    let (x1, x2) = (c(x1).min(c(x2)), c(x1).max(c(x2)));
    let (y1, y2) = (c(y1).min(c(y2)), c(y1).max(c(y2)));
    (x1, y1, x2, y2)
}

/// `_peak_near`: the strongest profile sample within `radius` of `expected`, as (position, value).
fn peak_near(profile: &[f32], expected: f64, radius: f64) -> (f64, f64) {
    let n = profile.len();
    if n == 0 {
        return (expected, 0.0);
    }
    let center = py_round(expected * (n - 1) as f64);
    let half = 2i64.max(py_round(radius * n as f64));
    let (lo, hi) = ((center - half).max(0), (center + half + 1).min(n as i64));
    if hi <= lo {
        return (expected, 0.0);
    }
    let local = &profile[lo as usize..hi as usize];
    // np.argmax: the first NaN, else the first maximum.
    let offset = local.iter().position(|v| v.is_nan()).unwrap_or_else(|| {
        local
            .iter()
            .enumerate()
            .fold(0, |b, (i, &v)| if v > local[b] { i } else { b })
    });
    let idx = lo as usize + offset;
    (idx as f64 / 1usize.max(n - 1) as f64, local[offset] as f64)
}

/// `_rect_from_edge_profile`: a film box from the frame's vertical-edge profile at the edges the
/// roll expects, or None when the profile cannot place one.
pub fn rect_from_edge_profile(
    profile: &[f32],
    contrast: f64,
    t: &Template,
) -> Option<(f64, f64, f64, f64)> {
    if contrast < MIN_PROFILE_CONTRAST
        || profile.len() < 8
        || !profile.iter().any(|v| v.is_finite())
    {
        return None;
    }
    let q = np_percentiles(profile, &[50.0, 99.5, 70.0, 90.0])?;
    let (baseline_level, peak_level) = (q[0] as f64, q[1] as f64);
    if peak_level - baseline_level < 0.20 || peak_level < 1.35 * baseline_level.max(0.05) {
        return None;
    }
    let radius = 0.045f64.max(4.0 * t.center_mad + 2.0 * t.width_mad);
    let (left, left_strength) = peak_near(profile, t.center_x - t.width * 0.5, radius);
    let (right, right_strength) = peak_near(profile, t.center_x + t.width * 0.5, radius);
    let baseline = q[2] as f64;
    let spread = (q[3] - q[0]) as f64;
    let threshold = 0.24f64.max(baseline + 0.55 * spread);
    let (mut left_ok, mut right_ok) = (left_strength >= threshold, right_strength >= threshold);
    if !(left_ok && right_ok) && ((right - left) - t.width).abs() <= EDGE_PAIR_WIDTH_TOL {
        (left_ok, right_ok) = (true, true);
    }
    let (x1, x2) = if left_ok && right_ok {
        if right - left < 0.65 * t.width || right - left > 1.35 * t.fallback_width {
            return None;
        }
        (left, right)
    } else if left_ok && left_strength >= threshold + 0.12 {
        (left, left + t.fallback_width)
    } else if right_ok && right_strength >= threshold + 0.12 {
        (right - t.fallback_width, right)
    } else {
        return None;
    };
    Some(clamp_rect((x1, t.top, x2, t.top + t.height)))
}

/// `_place_window_by_occupancy`: the start, in input pixels, of the `target_len` window within
/// [start, end) that holds the most occupancy (detection pixels, `scale` = detection/input).
/// Ties go to the centered position.
pub fn place_window_by_occupancy(
    start: i64,
    end: i64,
    target_len: f64,
    occupancy: &[f32],
    scale: f64,
) -> i64 {
    let d_start = py_round(start as f64 * scale).max(0);
    let d_end = py_round(end as f64 * scale).min(occupancy.len() as i64);
    let d_len = py_round(target_len * scale).max(1);
    if d_len >= d_end - d_start {
        return start;
    }
    let mut cs = vec![0f64];
    let mut sum = 0f64;
    for &v in &occupancy[d_start as usize..d_end as usize] {
        sum += v as f64;
        cs.push(sum);
    }
    let n_pos = ((d_end - d_start) - d_len + 1) as usize;
    let d_len = d_len as usize;
    let scores: Vec<f64> = (0..n_pos).map(|i| cs[i + d_len] - cs[i]).collect();
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max) - 1e-9;
    let centered = ((d_end - d_start) as f64 - d_len as f64) / 2.0;
    let k = (0..n_pos)
        .filter(|&i| scores[i] >= top)
        .fold(None::<usize>, |b, i| match b {
            Some(j) if (j as f64 - centered).abs() <= (i as f64 - centered).abs() => Some(j),
            _ => Some(i),
        })
        .unwrap_or(0);
    let new_start = py_round((d_start as f64 + k as f64) / scale);
    (start.max(new_start) as f64).min(end as f64 - target_len) as i64
}
