//! The textured-cell meters of negpy/features/exposure/normalization.py: anchor, textural
//! range, shadow reach and highlight hold, read off the block-median grid.

use crate::stats::{np_median, np_percentiles, np_percentiles_f64, np_quantile, pairwise_sum_f32};
use rayon::prelude::*;

const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

/// One channel of `(grid - floor) / span`. A float64 operand (a NumPy float64 scalar) makes
/// NumPy compute that step in float64; a Python float joins the float32 math (NEP 50).
#[derive(Clone, Copy)]
pub struct ChannelNorm {
    pub floor: f64,
    pub floor_wide: bool,
    pub span: f64,
    pub span_wide: bool,
}

impl ChannelNorm {
    fn apply(&self, g: f32) -> f32 {
        if self.floor_wide || self.span_wide {
            let d = if self.floor_wide {
                g as f64 - self.floor
            } else {
                (g - self.floor as f32) as f64
            };
            (d / self.span) as f32
        } else {
            (g - self.floor as f32) / self.span as f32
        }
    }
}

/// The float32 luma of an HxWx3 grid, each channel normalized first when `norm` is given.
pub fn luma(grid: &[f32], norm: Option<&[ChannelNorm; 3]>) -> Vec<f32> {
    grid.as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            let [r, g, b] = match norm {
                Some(n) => [n[0].apply(p[0]), n[1].apply(p[1]), n[2].apply(p[2])],
                None => *p,
            };
            LUMA_R * r + LUMA_G * g + LUMA_B * b
        })
        .collect()
}

/// The activity gate: sectors of 2x2 blocks of `block` x `block` cells vote when their four
/// block means span more than `gate`.
#[derive(Clone, Copy)]
pub struct Activity {
    pub block: usize,
    pub gate: f32,
    pub min_fraction: f64,
}

/// `_textured_cells(lum)` of an HxW luma plane: the cells of the voting sectors in row order,
/// or every cell when fewer than `min_fraction` of the sectors vote.
pub fn textured_cells(lum: Vec<f32>, h: usize, w: usize, a: Activity) -> Vec<f32> {
    let b = a.block;
    let (hs, ws) = ((h / (2 * b)) * 2 * b, (w / (2 * b)) * 2 * b);
    if hs == 0 || ws == 0 {
        return lum;
    }
    let (bh, bw) = (hs / b, ws / b);
    // NumPy's block mean: each row of the block summed pairwise, the rows added in order.
    let mut means = vec![0f32; bh * bw];
    for (i, m) in means.iter_mut().enumerate() {
        let (by, bx) = (i / bw, i % bw);
        let mut sum = 0f32;
        for y in by * b..(by + 1) * b {
            sum += pairwise_sum_f32(&lum[y * w + bx * b..y * w + (bx + 1) * b]);
        }
        *m = sum / (b * b) as f32;
    }
    let (sh, sw) = (bh / 2, bw / 2);
    let active: Vec<bool> = (0..sh * sw)
        .map(|i| {
            let (sy, sx) = (i / sw, i % sw);
            let quad = [
                means[2 * sy * bw + 2 * sx],
                means[2 * sy * bw + 2 * sx + 1],
                means[(2 * sy + 1) * bw + 2 * sx],
                means[(2 * sy + 1) * bw + 2 * sx + 1],
            ];
            let (lo, hi) = quad
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                    (np_min(l, v), np_max(h, v))
                });
            hi - lo > a.gate
        })
        .collect();
    let voting = active.iter().filter(|&&v| v).count();
    if (voting as f64 / active.len() as f64) < a.min_fraction {
        return lum;
    }
    let mut out = Vec::with_capacity(voting * 4 * b * b);
    for y in 0..hs {
        let sy = y / (2 * b);
        for x in 0..ws {
            if active[sy * sw + x / (2 * b)] {
                out.push(lum[y * w + x]);
            }
        }
    }
    out
}

/// np.min / np.max with NaN propagation.
fn np_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

fn np_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// `np.percentile(lum, q)` for a Python float q: float32 interpolation.
pub fn percentile(lum: &[f32], q: f64) -> f32 {
    np_percentiles(lum, &[q]).map_or(f32::NAN, |p| p[0])
}

/// The anchor meter's measurement: the mean of the cells inside the `clip` percentile trim
/// (float32) averaged with the trim's midpoint.
pub fn anchor_measure(lum: &[f32], clip: f64) -> f64 {
    let p = np_percentiles_f64(lum, &[clip, 100.0 - clip]);
    let (lo, hi) = (p[0], p[1]);
    let inner: Vec<f32> = lum
        .iter()
        .copied()
        .filter(|&v| v as f64 >= lo && v as f64 <= hi)
        .collect();
    let mean = pairwise_sum_f32(&inner) / inner.len() as f32;
    0.5 * (mean as f64 + 0.5 * (lo + hi))
}

/// The textural range: the spread between the `clip` and `100 - clip` percentiles.
pub fn textural_range(lum: &[f32], clip: f64) -> f64 {
    let p = np_percentiles_f64(lum, &[clip, 100.0 - clip]);
    (p[1] - p[0]).abs()
}

/// The constants of `measure_neutral_axis_from_log`. Bands and subject sectors compare in
/// float32, as NumPy compares a float32 array with a Python float.
pub struct NeutralConsts {
    pub quantile: f64,
    pub cap: f64,
    pub first_pass_cap: f64,
    pub min_pixels: usize,
    pub highlight: (f32, f32),
    pub mid: (f32, f32),
    pub shadow: (f32, f32),
    /// Foliage and sky: (chroma floor, hue low, hue high) in degrees.
    pub subjects: [(f32, f32, f32); 2],
    pub n0: f64,
    /// (dead zone, scale) of the mid/shadow agreement and of the offset width.
    pub agreement: (f64, f64),
    pub width: (f64, f64),
}

pub type Refs = [f64; 3];

pub struct NeutralAxis {
    pub mid: Refs,
    pub shadow: Refs,
    pub highlight: Option<Refs>,
    pub confidence: f64,
}

struct BandRefs {
    refs: Refs,
    chroma: f64,
    size: usize,
    width: f64,
}

/// `_rms_chroma` of one normalized triplet, in float32.
fn rms_chroma(n: [f32; 3]) -> f32 {
    let [r, g, b] = n;
    ((((r - g) * (r - g) + (g - b) * (g - b)) + (r - b) * (r - b)) / 3.0).sqrt()
}

/// `_subject_cells` for one normalized triplet: foliage or sky by hue and chroma.
fn is_subject(n: [f32; 3], subjects: &[(f32, f32, f32); 2]) -> bool {
    let (dr, db) = (n[0] - n[1], n[2] - n[1]);
    let (t, s) = ((dr + db) / 2.0, (dr - db) / 2.0);
    // np.hypot of float32 rounds the exact hypotenuse once.
    let chroma = ((t as f64) * (t as f64) + (s as f64) * (s as f64)).sqrt() as f32;
    if !subjects.iter().any(|&(c, _, _)| chroma > c) {
        return false;
    }
    let hue = crate::mathf::atan2f(-s, t) * (180.0f32 / std::f32::consts::PI);
    subjects
        .iter()
        .any(|&(c, lo, hi)| chroma > c && hue > lo && hue < hi)
}

/// `measure_neutral_axis_from_log` over a block-median grid (HxWx3 raw log), after its crop.
/// `floors32`/`ceils32` are the bounds as `normalize_log_image` reads them, `floors`/`ceils`
/// as Python floats.
pub fn neutral_axis(
    grid: &[f32],
    (floors32, ceils32): ([f32; 3], [f32; 3]),
    (floors, ceils): (Refs, Refs),
    exclude_subjects: bool,
    c: &NeutralConsts,
) -> Option<NeutralAxis> {
    let mut norm_all = vec![0f32; grid.len()];
    crate::analysis::normalize_log_image(grid, floors32, ceils32, &mut norm_all);
    let (norm_all, log_all) = (norm_all.as_chunks::<3>().0, grid.as_chunks::<3>().0);
    let keep: Vec<usize> = (0..norm_all.len())
        .into_par_iter()
        .filter(|&i| !exclude_subjects || !is_subject(norm_all[i], &c.subjects))
        .collect();
    let norm: Vec<[f32; 3]> = keep.par_iter().map(|&i| norm_all[i]).collect();
    let log: Vec<[f32; 3]> = keep.par_iter().map(|&i| log_all[i]).collect();
    let lum: Vec<f32> = norm
        .par_iter()
        .map(|n| LUMA_R * n[0] + LUMA_G * n[1] + LUMA_B * n[2])
        .collect();

    let band_refs = |(lo, hi): (f32, f32), chroma: &[f32], cap: f64| -> Option<BandRefs> {
        let band: Vec<usize> = (0..lum.len())
            .into_par_iter()
            .filter(|&i| lum[i] >= lo && lum[i] <= hi)
            .collect();
        if band.len() < c.min_pixels {
            return None;
        }
        let band_chroma: Vec<f32> = band.iter().map(|&i| chroma[i]).collect();
        let thr = np_quantile(&band_chroma, c.quantile)?;
        let idx: Vec<usize> = band.into_iter().filter(|&i| chroma[i] <= thr).collect();
        let near_neutral = if idx.is_empty() {
            cap
        } else {
            np_median(&mut idx.iter().map(|&i| chroma[i]).collect::<Vec<_>>()) as f64
        };
        if idx.len() < c.min_pixels || near_neutral > cap {
            return None;
        }
        let median_of = |f: &dyn Fn(usize) -> f32| {
            np_median(&mut idx.iter().map(|&i| f(i)).collect::<Vec<_>>()) as f64
        };
        let green = median_of(&|i| log[i][1]);
        let refs = [
            green + median_of(&|i| log[i][0] - log[i][1]),
            green,
            green + median_of(&|i| log[i][2] - log[i][1]),
        ];
        let width_of = |ch: usize| {
            let off: Vec<f32> = idx.iter().map(|&i| norm[i][ch] - norm[i][1]).collect();
            let p = np_percentiles_f64(&off, &[10.0, 90.0]);
            p[1] - p[0]
        };
        let (w0, w2) = (width_of(0), width_of(2));
        Some(BandRefs {
            refs,
            chroma: near_neutral,
            size: idx.len(),
            // np.max propagates NaN.
            width: if w0.is_nan() || w2.is_nan() {
                f64::NAN
            } else {
                w0.max(w2)
            },
        })
    };
    let norm_ref = |refs: Refs| -> Refs {
        std::array::from_fn(|ch| {
            let mut denom = ceils[ch] - floors[ch];
            if denom.abs() < 1e-6 {
                denom = if denom >= 0.0 { 1e-6 } else { -1e-6 };
            }
            (refs[ch] - floors[ch]) / denom
        })
    };

    let chroma1: Vec<f32> = norm.par_iter().map(|&n| rms_chroma(n)).collect();
    let mid1 = band_refs(c.mid, &chroma1, c.first_pass_cap)?;
    let sh1 = band_refs(c.shadow, &chroma1, c.first_pass_cap)?;
    let (nm, ns) = (norm_ref(mid1.refs), norm_ref(sh1.refs));
    let mut gain = [(1.0f32, 0.0f32); 3];
    for ch in [0, 2] {
        let du = nm[ch] - ns[ch];
        let (a, b) = if du.abs() < 1e-6 {
            (1.0, nm[1] - nm[ch])
        } else {
            let a = (nm[1] - ns[1]) / du;
            (a, nm[1] - a * nm[ch])
        };
        gain[ch] = (a as f32, b as f32);
    }
    let chroma2: Vec<f32> = norm
        .par_iter()
        .map(|n| {
            rms_chroma([
                gain[0].0 * n[0] + gain[0].1,
                n[1],
                gain[2].0 * n[2] + gain[2].1,
            ])
        })
        .collect();
    let mid = band_refs(c.mid, &chroma2, c.cap)?;
    let shadow = band_refs(c.shadow, &chroma2, c.cap)?;
    let highlight = band_refs(c.highlight, &chroma2, c.cap);

    let tight = (1.0 - mid.chroma.max(shadow.chroma) / c.cap).clamp(0.0, 1.0);
    let size_term = mid.size as f64 / (mid.size as f64 + c.n0);
    let (dm, ds) = (norm_ref(mid.refs), norm_ref(shadow.refs));
    let spread = [0, 2]
        .map(|ch| ((dm[ch] - dm[1]) - (ds[ch] - ds[1])).abs())
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let agree = 1.0 - ((spread - c.agreement.0).max(0.0) / c.agreement.1).min(1.0);
    let width = mid.width.max(shadow.width);
    let single = 1.0 - ((width - c.width.0).max(0.0) / c.width.1).min(1.0);
    Some(NeutralAxis {
        mid: mid.refs,
        shadow: shadow.refs,
        highlight: highlight.map(|h| h.refs),
        confidence: (tight * size_term * agree * single).clamp(0.0, 1.0),
    })
}
