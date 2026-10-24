//! The edge sampling of negpy/features/geometry/skew.py: `_fit_side`'s per-station profiles
//! and `_profile_peaks`, in NumPy's float32 and float64 order.

use rayon::prelude::*;

use crate::filter::{self, Border};

/// `_profile_peaks`: subpixel positions of the strong gradient steps along one profile, after a
/// sigma-1 Gaussian (reflect-101) and `np.gradient`.
pub fn profile_peaks(profile: &[f32]) -> Vec<f64> {
    let n = profile.len();
    if n < 5 {
        return Vec::new();
    }
    let mut blur = vec![0f32; n];
    if filter::gaussian_blur(
        profile,
        &mut blur,
        (n, 1, 1),
        (1, 0),
        (0.0, 1.0),
        Border::Reflect101,
    )
    .is_none()
    {
        return Vec::new();
    }
    let mut grad: Vec<f32> = (0..n)
        .map(|i| match i {
            0 => blur[1] - blur[0],
            _ if i == n - 1 => blur[n - 1] - blur[n - 2],
            _ => (blur[i + 1] - blur[i - 1]) / 2.0,
        })
        .map(f32::abs)
        .collect();
    let peak = grad.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sorted = grad.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    if peak <= 3.0 * median as f64 + 1e-6 {
        return Vec::new();
    }
    // A Python float meets the float32 gradient as float32.
    let (bar, tiny) = ((0.4 * peak) as f32, 1e-9f64 as f32);
    let g = std::mem::take(&mut grad);
    (1..n - 1)
        .filter(|&i| g[i] >= g[i - 1] && g[i] > g[i + 1] && g[i] >= bar)
        .map(|i| {
            let (a, b, c) = (g[i - 1], g[i], g[i + 1]);
            let denom = a - 2.0 * b + c;
            let frac = if denom.abs() > tiny {
                0.5 * (a - c) / denom
            } else {
                0.0
            };
            i as f64 + frac as f64
        })
        .collect()
}

/// Python's `round` of a float64: ties to even.
fn round_half_even(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `_fit_side`'s `sample(center, half, nearest)` on an HxW luminance plane read along rows
/// (`horizontal`) or columns: per station, the peaks of the 3-column mean profile within
/// `half` of its center, or only the one nearest the center. Returns (stations, positions).
pub fn sample(
    img: &[f32],
    (h, w): (usize, usize),
    horizontal: bool,
    (stations, centers): (&[f64], &[f64]),
    half: f64,
    nearest: bool,
) -> (Vec<f64>, Vec<f64>) {
    let (vh, vw) = if horizontal { (h, w) } else { (w, h) };
    let at = |r: usize, c: usize| {
        if horizontal {
            img[r * w + c]
        } else {
            img[c * w + r]
        }
    };
    let found: Vec<(f64, Vec<f64>)> = stations
        .par_iter()
        .zip(centers.par_iter())
        .map(|(&x, &c)| {
            let x0 = round_half_even(x);
            let lo = ((c - half).floor() as i64).max(0);
            let hi = ((c + half).ceil() as i64 + 1).min(vh as i64);
            if hi - lo < 5 || x0 < 1 || x0 > vw as i64 - 2 {
                return (x, Vec::new());
            }
            let (lo, x0) = (lo as usize, x0 as usize);
            let profile: Vec<f32> = (lo..hi as usize)
                .map(|r| (at(r, x0 - 1) + at(r, x0) + at(r, x0 + 1)) / 3.0)
                .collect();
            let mut peaks = profile_peaks(&profile);
            if nearest && !peaks.is_empty() {
                let want = c - lo as f64;
                let best = peaks
                    .iter()
                    .enumerate()
                    .fold((0, f64::INFINITY), |(bi, bd), (i, &p)| {
                        let d = (p - want).abs();
                        if d < bd { (i, d) } else { (bi, bd) }
                    })
                    .0;
                peaks = vec![peaks[best]];
            }
            (x, peaks.into_iter().map(|p| lo as f64 + p).collect())
        })
        .collect();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, peaks) in found {
        xs.extend(std::iter::repeat_n(x, peaks.len()));
        ys.extend(peaks);
    }
    (xs, ys)
}

/// NumPy's PCG64 as `np.random.default_rng(0)` seeds it, and `integers(0, n)` for n < 2^32
/// (Lemire's bounded draw on the 32-bit halves of each 64-bit output).
struct Pcg64 {
    state: u128,
    inc: u128,
    half: Option<u32>,
}

impl Pcg64 {
    const MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

    fn seed0() -> Self {
        Pcg64 {
            state: (0x1aa1b5345996452du128 << 64) | 0x09585eb7a69561e3,
            inc: (0x418ddadb3af71a82u128 << 64) | 0x588133bc447873a9,
            half: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        let s = self.state;
        (((s >> 64) as u64) ^ (s as u64)).rotate_right((s >> 122) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.half.take() {
            return v;
        }
        let v = self.next_u64();
        self.half = Some((v >> 32) as u32);
        v as u32
    }

    /// One draw of `integers(0, n)`, n >= 1.
    fn below(&mut self, n: u32) -> u32 {
        let rng = n - 1;
        if rng == 0 {
            return 0;
        }
        if rng == u32::MAX {
            return self.next_u32();
        }
        let excl = rng + 1;
        let mut m = self.next_u32() as u64 * excl as u64;
        let mut left = m as u32;
        if left < excl {
            let threshold = (u32::MAX - rng) % excl;
            while left < threshold {
                m = self.next_u32() as u64 * excl as u64;
                left = m as u32;
            }
        }
        (m >> 32) as u32
    }
}

/// Least-squares line `y = slope * x + intercept` through the points (`np.polyfit(x, y, 1)` up
/// to rounding: centered sums, not LAPACK's SVD).
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

/// `_ransac_line`: the straight run of peaks with the most distinct stations, preferring one
/// near `expected`.
fn ransac_line(xs: &[f64], ys: &[f64], expected: f64, half: f64, span: f64) -> Option<(f64, f64)> {
    let n = xs.len();
    if n < 12 {
        return None;
    }
    let mut rng = Pcg64::seed0();
    let draws: Vec<usize> = (0..600).map(|_| rng.below(n as u32) as usize).collect();
    let (a, b) = draws.split_at(300);
    let max_slope = 4.0f64.to_radians().tan();
    let (lo, hi) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| {
            (l.min(x), h.max(x))
        });
    let mid_x = 0.5 * (lo + hi);
    let mut best: Option<(f64, Vec<bool>)> = None;
    let mut best_stations = 0usize;
    for (&a, &b) in a.iter().zip(b) {
        if (xs[a] - xs[b]).abs() < 0.3 * span {
            continue;
        }
        let slope = (ys[b] - ys[a]) / (xs[b] - xs[a]);
        if slope.abs() > max_slope {
            continue;
        }
        let intercept = ys[a] - slope * xs[a];
        let inlier: Vec<bool> = xs
            .iter()
            .zip(ys)
            .map(|(&x, &y)| (y - (slope * x + intercept)).abs() <= 1.5)
            .collect();
        let mut seen: Vec<f64> = (0..n).filter(|&i| inlier[i]).map(|i| xs[i]).collect();
        seen.sort_unstable_by(f64::total_cmp);
        seen.dedup();
        let stations = seen.len();
        let mid = slope * mid_x + intercept;
        let score = stations as f64 * (1.0 - 0.5 * ((mid - expected).abs() / half).min(1.0));
        if best.as_ref().is_none_or(|(s, _)| score > *s) {
            best = Some((score, inlier));
            best_stations = stations;
        }
    }
    let (_, inlier) = best?;
    if best_stations < 12 {
        return None;
    }
    let (px, py): (Vec<f64>, Vec<f64>) = (0..n)
        .filter(|&i| inlier[i])
        .map(|i| (xs[i], ys[i]))
        .unzip();
    Some(polyfit1(&px, &py))
}

/// A fitted frame edge in pixels of the detection buffer.
pub struct Line {
    pub p1: (f64, f64),
    pub p2: (f64, f64),
    pub horizontal: bool,
    pub inliers: f64,
    pub rms: f64,
}

/// A box bounds its edge to within the film detector's snap window (`snap_film_bounds`).
const BOX_TOLERANCE: f64 = 0.02;
const BOX_TOLERANCE_PX: f64 = 16.0;
const MIN_LINE_SPAN: f64 = 0.25;
/// Admits the bow a scanning lens or film curl puts on a frame edge; its chord stays level.
const MAX_LINE_RMS: f64 = 2.0;
const MIN_INLIER_FRACTION: f64 = 0.5;

/// `_fit_side`: the edge running along one axis near `at`, between `start` and `stop`, on an
/// HxW luminance plane. `outward` is the sign of the box's outside across the axis, or 0 when
/// `at` is the edge itself.
pub fn fit_side(
    lum: &[f32],
    dims: (usize, usize),
    horizontal: bool,
    (at, start, stop): (f64, f64, f64),
    outward: i32,
) -> Option<Line> {
    let (h, w) = if horizontal { dims } else { (dims.1, dims.0) };
    let length = stop - start;
    if length < MIN_LINE_SPAN * w as f64 || !(at >= 2.0 && at <= h as f64 - 3.0) {
        return None;
    }
    let num = (240f64.min(24f64.max(length / 4.0))) as usize;
    let (first, last) = (start + 0.1 * length, stop - 0.1 * length);
    let step = (last - first) / (num - 1) as f64;
    let mut stations: Vec<f64> = (0..num).map(|i| i as f64 * step + first).collect();
    stations[num - 1] = last;

    let inside = 10f64.max(0.035 * h.max(w) as f64);
    let outside = if outward == 0 {
        inside
    } else {
        BOX_TOLERANCE_PX.max(BOX_TOLERANCE * h as f64)
    };
    let center = at + 0.5 * outward as f64 * (outside - inside);
    let half = 0.5 * (inside + outside);
    let (xs, ys) = sample(
        lum,
        dims,
        horizontal,
        (&stations, &vec![center; num]),
        half,
        false,
    );
    let (mut slope, mut intercept) = ransac_line(&xs, &ys, at, half, length)?;
    let centers: Vec<f64> = stations.iter().map(|&x| slope * x + intercept).collect();
    let (xs, ys) = sample(
        lum,
        dims,
        horizontal,
        (&stations, &centers),
        6f64.max(0.3 * inside),
        true,
    );
    if xs.len() < 12 {
        return None;
    }
    let mut keep = vec![true; xs.len()];
    for _ in 0..8 {
        let (kx, ky): (Vec<f64>, Vec<f64>) = (0..xs.len())
            .filter(|&i| keep[i])
            .map(|i| (xs[i], ys[i]))
            .unzip();
        (slope, intercept) = polyfit1(&kx, &ky);
        let res: Vec<f64> = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| y - (slope * x + intercept))
            .collect();
        let mut kept: Vec<f64> = (0..xs.len()).filter(|&i| keep[i]).map(|i| res[i]).collect();
        let med = crate::stats::median_f64(&mut kept.clone());
        for v in &mut kept {
            *v = (*v - med).abs();
        }
        let mad = 1.4826 * crate::stats::median_f64(&mut kept);
        let bar = 1f64.max(2.5 * mad);
        let new: Vec<bool> = res.iter().map(|r| r.abs() <= bar).collect();
        if new.iter().filter(|&&v| v).count() < 12 || new == keep {
            break;
        }
        keep = new;
    }
    let kept: Vec<usize> = (0..xs.len()).filter(|&i| keep[i]).collect();
    let inliers = kept.len() as f64 / num as f64;
    let sq: f64 = kept
        .iter()
        .map(|&i| {
            let e = ys[i] - (slope * xs[i] + intercept);
            e * e
        })
        .sum();
    let rms = (sq / kept.len() as f64).sqrt();
    if inliers < MIN_INLIER_FRACTION || rms > MAX_LINE_RMS {
        return None;
    }
    let (x1, x2) = kept
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &i| {
            (l.min(xs[i]), h.max(xs[i]))
        });
    if x2 - x1 < MIN_LINE_SPAN * w as f64 {
        return None;
    }
    let (y1, y2) = (slope * x1 + intercept, slope * x2 + intercept);
    let (p1, p2) = if horizontal {
        ((x1, y1), (x2, y2))
    } else {
        ((y1, x1), (y2, x2))
    };
    Some(Line {
        p1,
        p2,
        horizontal,
        inliers,
        rms,
    })
}

/// `_outer_peaks`: first and last local maxima of `coverage` at or above `floor`, off the
/// canvas border, at least `_MIN_PAIR_SEPARATION` of the extent apart.
fn outer_peaks(coverage: &[f64], floor: f64) -> Option<(usize, usize)> {
    let n = coverage.len();
    let peaks: Vec<usize> = (3..n.saturating_sub(3))
        .filter(|&i| {
            coverage[i] >= floor && coverage[i] >= coverage[i - 1] && coverage[i] >= coverage[i + 1]
        })
        .collect();
    let (&a, &b) = (peaks.first()?, peaks.last()?);
    (peaks.len() >= 2 && (b - a) as f64 >= 0.3 * n as f64).then_some((a, b))
}

/// Pixels of `|sobel(lum)|` at a quarter of its 99.5th percentile or more, dilated along the
/// edge's normal by `spread`.
fn strong_edges(lum: &[f32], (h, w): (usize, usize), dx: bool, spread: usize) -> Option<Vec<u8>> {
    use crate::filter::{Border, sep_filter2d};
    let (smooth, deriv) = ([1.0f32, 2.0, 1.0], [-1.0f32, 0.0, 1.0]);
    let (kx, ky) = if dx { (deriv, smooth) } else { (smooth, deriv) };
    let mut grad = vec![0f32; h * w];
    sep_filter2d(lum, &mut grad, (h, w, 1), (&kx, &ky), Border::Reflect101)?;
    grad.iter_mut().for_each(|v| *v = v.abs());
    let bar = (0.25 * crate::stats::np_percentiles(&grad, &[99.5])?[0] as f64) as f32;
    let strong: Vec<u8> = grad.iter().map(|&g| (g >= bar) as u8).collect();
    let mut out = vec![0u8; h * w];
    let k = 2 * spread + 1;
    let ksize = if dx { (k, 1) } else { (1, k) };
    crate::morph::morphology(
        &strong,
        &mut out,
        (h, w, 1),
        crate::morph::Op::Dilate,
        ksize,
        1,
        crate::filter::Border::Constant,
    )?;
    Some(out)
}

/// `_profile_lines`: the outermost long edges by row and column coverage, for frames the film
/// detector cannot box. Lines come in opposite pairs or not at all, per axis.
pub fn profile_lines(lum: &[f32], (h, w): (usize, usize)) -> Vec<Line> {
    let spread = 3usize.max((0.02 * h.max(w) as f64) as usize);
    let mut lines = Vec::new();
    let Some(gy) = strong_edges(lum, (h, w), false, spread) else {
        return lines;
    };
    let (c0, c1) = ((0.05 * w as f64) as usize, (0.95 * w as f64) as usize);
    let rows: Vec<f64> = (0..h)
        .map(|y| {
            gy[y * w + c0..y * w + c1]
                .iter()
                .map(|&v| v as u64)
                .sum::<u64>() as f64
                / (c1 - c0) as f64
        })
        .collect();
    let found_rows = outer_peaks(&rows, 0.35);
    let (top, bottom) = found_rows.unwrap_or((0, h));
    if let Some((a, b)) = found_rows {
        for y in [a, b] {
            lines.extend(fit_side(lum, (h, w), true, (y as f64, 0.0, w as f64), 0));
        }
    }
    if let Some(gx) = strong_edges(lum, (h, w), true, spread) {
        let inset = (0.05 * (bottom - top) as f64) as usize;
        let (lo, hi) = (top + inset, bottom - inset);
        if hi > lo + 10 {
            let cols: Vec<f64> = (0..w)
                .map(|x| {
                    (lo..hi).map(|y| gx[y * w + x] as u64).sum::<u64>() as f64 / (hi - lo) as f64
                })
                .collect();
            if let Some((a, b)) = outer_peaks(&cols, 0.35) {
                for x in [a, b] {
                    lines.extend(fit_side(
                        lum,
                        (h, w),
                        false,
                        (x as f64, top as f64, bottom as f64),
                        0,
                    ));
                }
            }
        }
    }
    let count = |horizontal: bool, lines: &[Line]| {
        lines.iter().filter(|l| l.horizontal == horizontal).count()
    };
    let (nh, nv) = (count(true, &lines), count(false, &lines));
    lines.retain(|l| if l.horizontal { nh >= 2 } else { nv >= 2 });
    lines
}

/// `_at_canvas_edges`: whether every line (horizontal, position) hugs the canvas edge, paired on
/// both axes, with no opaque holder outside it.
pub fn at_canvas_edges(lum: &[f32], (h, w): (usize, usize), lines: &[(bool, f64)]) -> bool {
    // A tight scan's border lines sit within this fraction of the cross extent from the edge.
    const CANVAS_EDGE_BAND: f64 = 0.12;
    // Film base or bed outside a border never reads this dark; an opaque holder does.
    const OPAQUE_RING: f64 = 0.04;
    const MIN_PAIR_SEPARATION: f64 = 0.3;
    let extent = |horizontal: bool| if horizontal { h } else { w };
    if lines.is_empty()
        || lines.iter().any(|&(horizontal, position)| {
            let e = extent(horizontal) as f64;
            position.min(e - position) > CANVAS_EDGE_BAND * e
        })
    {
        return false;
    }
    for &(horizontal, position) in lines {
        let extent = extent(horizontal);
        let index = position as usize;
        let (lo, hi) = if position < extent as f64 / 2.0 {
            (0, index.saturating_sub(2))
        } else {
            ((index + 3).min(extent), extent)
        };
        let mut outside: Vec<f32> = if horizontal {
            lum[lo * w..hi * w].to_vec()
        } else {
            lum.chunks_exact(w)
                .flat_map(|row| row[lo..hi].iter().copied())
                .collect()
        };
        if !outside.is_empty() && (crate::stats::np_median(&mut outside) as f64) < OPAQUE_RING {
            return false;
        }
    }
    let spread = |horizontal: bool, extent: usize| {
        let pos: Vec<f64> = lines
            .iter()
            .filter(|l| l.0 == horizontal)
            .map(|l| l.1)
            .collect();
        if pos.len() < 2 {
            return 0.0;
        }
        let (lo, hi) = pos
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &p| {
                (l.min(p), h.max(p))
            });
        (hi - lo) / extent as f64
    };
    spread(true, h) >= MIN_PAIR_SEPARATION && spread(false, w) >= MIN_PAIR_SEPARATION
}

/// `_coarse_angles`: the dominant folded angles (degrees) of long line segments on an HxW
/// luminance plane, strongest first, with a runner-up at a quarter of its weight or more.
pub fn coarse_angles(
    lum: &[f32],
    (h, w): (usize, usize),
    min_segment: f64,
    max_skew: f64,
    angle_bin: f64,
) -> Vec<f64> {
    // Squaring the frame and the crop's own edge fit both measure the same plane.
    static LAST: crate::memo::Memo<Vec<f64>, 4> = crate::memo::Memo::new();
    let params = [min_segment, max_skew, angle_bin].map(|v| v.to_bits() as i64);
    let key = crate::memo::extend(crate::memo::key(lum, &[h, w]), &params);
    LAST.get_or(key, || {
        segment_angles(lum, (h, w), min_segment, max_skew, angle_bin)
    })
}

fn segment_angles(
    lum: &[f32],
    (h, w): (usize, usize),
    min_segment: f64,
    max_skew: f64,
    angle_bin: f64,
) -> Vec<f64> {
    let q = crate::stats::np_percentiles_f64(lum, &[1.0, 99.0]);
    let (lo, span) = (q[0], (q[1] - q[0]).max(1e-6));
    let mut u8s = vec![0u8; lum.len()];
    crate::par_zip(&mut u8s, lum, move |o, &v| {
        *o = (((v as f64 - lo) / span).clamp(0.0, 1.0) * 255.0) as u8;
    });
    let min_len = min_segment * h.max(w) as f64;
    let found: Vec<(f64, f64)> = crate::lsd::line_segments(&u8s, (h, w))
        .iter()
        .map(|s| {
            let (dx, dy) = ((s[2] - s[0]) as f64, (s[3] - s[1]) as f64);
            let angle = libm::atan2(dy, dx).to_degrees();
            ((angle + 45.0).rem_euclid(90.0) - 45.0, dx.hypot(dy))
        })
        .filter(|&(a, l)| l >= min_len && a.abs() <= max_skew)
        .collect();
    if found.is_empty() {
        return Vec::new();
    }
    // np.arange(-max_skew, max_skew + angle_bin, angle_bin), filled as NumPy fills it.
    let start = -max_skew;
    let delta = (start + angle_bin) - start;
    let n = (((max_skew + angle_bin) - start) / angle_bin).ceil() as usize;
    let edges: Vec<f64> = (0..n).map(|i| start + i as f64 * delta).collect();
    let mut hist = vec![0f64; n - 1];
    for &(a, l) in &found {
        if a < edges[0] || a > edges[n - 1] {
            continue;
        }
        let bin = (edges.partition_point(|&e| e <= a) - 1).min(n - 2);
        hist[bin] += l * l;
    }
    let hist32: Vec<f32> = hist.iter().map(|&v| v as f32).collect();
    let mut smooth = vec![0f32; n - 1];
    if filter::gaussian_blur(
        &hist32,
        &mut smooth,
        (1, n - 1, 1),
        (0, 0),
        (3.0, 0.0),
        Border::Reflect101,
    )
    .is_none()
    {
        return Vec::new();
    }
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
    };
    let first = argmax(&smooth);
    let mut modes = vec![edges[first] + angle_bin / 2.0];
    let mut rest = smooth.clone();
    rest[first.saturating_sub(10)..(first + 11).min(n - 1)].fill(0.0);
    let second = argmax(&rest);
    if rest[second] >= 0.25 * smooth[first] {
        modes.push(edges[second] + angle_bin / 2.0);
    }
    modes
}

#[cfg(test)]
mod tests {
    use super::Pcg64;

    fn draws(n: u32, count: usize) -> Vec<u32> {
        let mut rng = Pcg64::seed0();
        (0..count).map(|_| rng.below(n)).collect()
    }

    #[test]
    fn default_rng_0_integers_match_numpy() {
        assert_eq!(draws(17, 12), [14, 10, 8, 4, 5, 0, 1, 0, 2, 13, 11, 15]);
        assert_eq!(
            draws(1_000_003, 6),
            [850626, 636963, 511138, 269787, 307830, 40973]
        );
        assert_eq!(draws(2, 8), [1, 1, 1, 0, 0, 0, 0, 0]);
    }
}
