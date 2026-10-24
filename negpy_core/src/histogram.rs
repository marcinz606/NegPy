//! The scope counts of negpy/features/exposure/analysis.py: integer counts, so only the bin each
//! sample lands in must match NumPy.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::analysis::luminance_chunk;
use crate::color::{self, LabSpace};

pub const DENSITY_HIST_BINS: usize = 120;
const DENSITY_HIST_RANGE: (f64, f64) = (-0.1, 1.1);
pub const WAVEFORM_BINS: usize = 128;
pub const WAVEFORM_COLS: usize = 256;
pub const OUTPUT_HIST_BINS: usize = 256;
pub const COLOR_HIST_BINS: usize = 32;
const WAVEFORM_COLOR_SCALE: f32 = 1023.0;
const SCAN_DENSITY_RANGE: (f64, f64) = (0.0, 3.0);

/// A stride-subsampled crop of an HxWxC float32 image: rows `y1..y2` and columns `x1..x2`, every
/// `step`-th from the first, as `img[y1:y2:step, x1:x2:step]` reads them.
#[derive(Clone, Copy)]
pub struct Samples<'a, T = f32> {
    pub img: &'a [T],
    pub w: usize,
    pub cn: usize,
    pub rows: (usize, usize),
    pub cols: (usize, usize),
    pub step: usize,
}

impl<T: Copy> Samples<'_, T> {
    fn row_starts(&self) -> Vec<usize> {
        (self.rows.0..self.rows.1).step_by(self.step).collect()
    }

    /// The first three channels of row `y`'s samples, packed: a slice of the image when they
    /// already are, else a copy in `buf`.
    fn rgb_row<'b>(&'b self, y: usize, buf: &'b mut Vec<T>) -> &'b [T] {
        if self.step == 1 && self.cn == 3 {
            return &self.img[(y * self.w + self.cols.0) * 3..(y * self.w + self.cols.1) * 3];
        }
        buf.clear();
        for x in (self.cols.0..self.cols.1).step_by(self.step) {
            let p = (y * self.w + x) * self.cn;
            buf.extend_from_slice(&self.img[p..p + 3]);
        }
        buf
    }
}

/// `np.histogram(np.clip(values, lo, hi), bins, range=(lo, hi))` for float32 values with Python
/// float edges: NumPy bins in float32 (NEP 50), then corrects each index against the float32
/// `np.linspace` edges, `f32(i * step + lo)`. NaN is dropped.
#[derive(Clone, Copy)]
struct Binner {
    lo: f64,
    hi: f64,
    bins: usize,
}

const NO_BIN: u8 = u8::MAX;

impl Binner {
    /// Each value's bin, or NO_BIN for NaN; branch-free, so it vectorizes.
    fn bin_into(self, vals: &[f32], out: &mut [u8]) {
        bin_into(self, vals, out)
    }
}

#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn bin_into(b: Binner, vals: &[f32], out: &mut [u8]) {
    let (lo, hi, n) = (b.lo as f32, b.hi as f32, b.bins as i32);
    let (denom, step) = ((b.hi - b.lo) as f32, (b.hi - b.lo) / b.bins as f64);
    let edge = |i: i32| (i as f64 * step + b.lo) as f32;
    for (o, &v) in out.iter_mut().zip(vals) {
        let c = v.max(lo).min(hi);
        let i = ((((c - lo) / denom) * n as f32) as i32).min(n - 1);
        let down = c < edge(i);
        let up = !down & (i != n - 1) & (c >= edge(i + 1));
        let i = i - down as i32 + up as i32;
        *o = if v.is_nan() { NO_BIN } else { i as u8 };
    }
}

/// The parade bin of each value: np.nan_to_num and the negation in float32, the scan density
/// axis in float64, floored and clipped.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn parade_bins(vals: &[f32], out: &mut [u8]) {
    let (lo, hi) = SCAN_DENSITY_RANGE;
    for (o, &v) in out.iter_mut().zip(vals) {
        let v = if v.is_nan() {
            0.0
        } else {
            v.clamp(f32::MIN, f32::MAX)
        };
        let pos = (hi - (-v) as f64) / (hi - lo);
        *o = (pos * WAVEFORM_BINS as f64)
            .floor()
            .clamp(0.0, (WAVEFORM_BINS - 1) as f64) as u8;
    }
}

/// Neighbouring samples often share a bin, and one counter serializes on its own increments,
/// so each sample counts into lane `i % LANES` (u32: a cell holds one column's samples at most).
const LANES: usize = 4;

fn fold_lanes(lanes: &[u32]) -> Vec<u64> {
    let n = lanes.len() / LANES;
    let mut out = vec![0u64; n];
    for lane in lanes.chunks_exact(n) {
        out.iter_mut().zip(lane).for_each(|(o, &c)| *o += c as u64);
    }
    out
}

/// `density_histogram`: (4, DENSITY_HIST_BINS) counts of R, G, B and luminance.
pub fn density_histogram(s: Samples) -> Vec<f64> {
    let binner = Binner {
        lo: DENSITY_HIST_RANGE.0,
        hi: DENSITY_HIST_RANGE.1,
        bins: DENSITY_HIST_BINS,
    };
    count_rows(s, 4 * DENSITY_HIST_BINS, |rgb, counts| {
        let px = rgb.len() / 3;
        let mut lum = vec![0f32; px];
        luminance_chunk(rgb, &mut lum);
        let (mut bins, mut lbins) = (vec![0u8; 3 * px], vec![0u8; px]);
        binner.bin_into(rgb, &mut bins);
        binner.bin_into(&lum, &mut lbins);
        for (i, (p, &l)) in bins.as_chunks::<3>().0.iter().zip(&lbins).enumerate() {
            let lane = (i % LANES) * 4 * DENSITY_HIST_BINS;
            for (c, &b) in p.iter().chain([&l]).enumerate() {
                if b != NO_BIN {
                    counts[lane + c * DENSITY_HIST_BINS + b as usize] += 1;
                }
            }
        }
    })
}

/// `density_parade`: (3, WAVEFORM_BINS, WAVEFORM_COLS) counts of each channel's scan density
/// (the negated log) per parade column, the column of the sample's x in the crop.
pub fn density_parade(s: Samples) -> Vec<f64> {
    let full_w = s.cols.1 - s.cols.0;
    let cols: Vec<usize> = (0..full_w)
        .step_by(s.step)
        .map(|x| (x * WAVEFORM_COLS / full_w).min(WAVEFORM_COLS - 1))
        .collect();
    let plane = WAVEFORM_BINS * WAVEFORM_COLS;
    count_rows(s, 3 * plane, |rgb, counts| {
        let mut bins = vec![0u8; rgb.len()];
        parade_bins(rgb, &mut bins);
        for (i, (p, &col)) in bins.as_chunks::<3>().0.iter().zip(&cols).enumerate() {
            let lane = (i % LANES) * 3 * plane;
            for (c, &b) in p.iter().enumerate() {
                counts[lane + c * plane + b as usize * WAVEFORM_COLS + col] += 1;
            }
        }
    })
}

/// NumPy's float32 to int32 cast on x86: truncation, and INT_MIN for NaN or out of range.
fn np_i32(f: f32) -> i32 {
    if (-2147483648.0..2147483648.0).contains(&f) {
        f as i32
    } else {
        i32::MIN
    }
}

/// np.nan_to_num for float32.
fn nan_to_num(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(f32::MIN, f32::MAX)
    }
}

/// `clip((v * bins).astype(int32), 0, bins - 1)` in float32.
fn quantize(v: f32, bins: usize) -> usize {
    np_i32(v * bins as f32).clamp(0, bins as i32 - 1) as usize
}

/// Counts over the rows of `s` in parallel: `count(row, counts)` adds one row into `n` lanes'
/// worth of u32 counters, which fold into one u64 count per cell.
fn count_rows<T: Copy + Send + Sync>(
    s: Samples<'_, T>,
    n: usize,
    count: impl Fn(&[T], &mut [u32]) + Sync,
) -> Vec<f64> {
    // One chunk of rows per thread: rayon's fold makes an accumulator per split job, and each
    // holds lanes of counters for every cell.
    let rows = s.row_starts();
    let chunk = rows.len().div_ceil(rayon::current_num_threads()).max(1);
    rows.par_chunks(chunk)
        .map(|ys| {
            let (mut counts, mut buf) = (vec![0u32; LANES * n], Vec::new());
            for &y in ys {
                count(s.rgb_row(y, &mut buf), &mut counts);
            }
            fold_lanes(&counts)
        })
        .reduce(
            || vec![0u64; n],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        )
        .iter()
        .map(|&c| c as f64)
        .collect()
}

/// `output_histogram` of float32 samples: (4, 256) counts of R, G, B and luminance on 0..1.
/// `np.histogram(range=(0, 1))` divides by a uint64 span, so it bins in float64, where the
/// edges i / 256 are exact; values outside 0..1 and NaN are dropped.
pub fn output_histogram(s: Samples) -> Vec<f64> {
    let bin = |v: f32| {
        (0.0..=1.0)
            .contains(&v)
            .then(|| ((v as f64 * OUTPUT_HIST_BINS as f64) as usize).min(OUTPUT_HIST_BINS - 1))
    };
    count_rows(s, 4 * OUTPUT_HIST_BINS, |rgb, counts| {
        let mut lum = vec![0f32; rgb.len() / 3];
        luminance_chunk(rgb, &mut lum);
        for (i, (p, &l)) in rgb.as_chunks::<3>().0.iter().zip(&lum).enumerate() {
            let lane = (i % LANES) * 4 * OUTPUT_HIST_BINS;
            for (c, &v) in p.iter().chain([&l]).enumerate() {
                if let Some(b) = bin(v) {
                    counts[lane + c * OUTPUT_HIST_BINS + b] += 1;
                }
            }
        }
    })
}

/// `output_histogram` of uint8 samples: each channel by value, luminance of value / 255.
pub fn output_histogram_u8(s: Samples<'_, u8>) -> Vec<f64> {
    count_rows(s, 4 * OUTPUT_HIST_BINS, |rgb, counts| {
        let scaled: Vec<f32> = rgb.iter().map(|&v| v as f32 / 255.0).collect();
        let mut lum = vec![0f32; rgb.len() / 3];
        luminance_chunk(&scaled, &mut lum);
        for (i, (p, &l)) in rgb.as_chunks::<3>().0.iter().zip(&lum).enumerate() {
            let lane = (i % LANES) * 4 * OUTPUT_HIST_BINS;
            for (c, &v) in p.iter().enumerate() {
                counts[lane + c * OUTPUT_HIST_BINS + v as usize] += 1;
            }
            let b = ((l * OUTPUT_HIST_BINS as f32) as i64).min(OUTPUT_HIST_BINS as i64 - 1);
            counts[lane + 3 * OUTPUT_HIST_BINS + b as usize] += 1;
        }
    })
}

/// `color_histogram`: (32, 32, 32) joint RGB counts.
pub fn color_histogram(s: Samples) -> Vec<f64> {
    let n = COLOR_HIST_BINS;
    count_rows(s, n * n * n, |rgb, counts| {
        for (i, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
            let lane = (i % LANES) * n * n * n;
            let [r, g, b] = p.map(|v| quantize(v, n));
            counts[lane + (r * n + g) * n + b] += 1;
        }
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Waveform {
    /// (BINS, COLS) luma counts.
    Luma,
    /// (3, BINS, COLS) sums of R, G, B rounded to 0..1023, in each luma cell.
    Color,
    /// (3, BINS, COLS) counts of each channel.
    Parade,
}

/// `waveform`, `waveform_color` and `waveform_parade`: the column of a sample is its x in the
/// sampled frame, its luma cell from the float32 luminance.
pub fn waveform(s: Samples, kind: Waveform) -> Vec<f64> {
    let w = (s.cols.1 - s.cols.0).div_ceil(s.step);
    let cols: Vec<usize> = (0..w)
        .map(|x| (x * WAVEFORM_COLS / w).min(WAVEFORM_COLS - 1))
        .collect();
    let plane = WAVEFORM_BINS * WAVEFORM_COLS;
    let n = if kind == Waveform::Luma {
        plane
    } else {
        3 * plane
    };
    count_rows(s, n, |rgb, counts| {
        let mut lum = vec![0f32; rgb.len() / 3];
        if kind != Waveform::Parade {
            luminance_chunk(rgb, &mut lum);
        }
        for (i, (p, &col)) in rgb.as_chunks::<3>().0.iter().zip(&cols).enumerate() {
            let lane = (i % LANES) * n;
            match kind {
                Waveform::Luma => {
                    let b = quantize(nan_to_num(lum[i]), WAVEFORM_BINS);
                    counts[lane + b * WAVEFORM_COLS + col] += 1;
                }
                Waveform::Color => {
                    let b = quantize(nan_to_num(lum[i]), WAVEFORM_BINS);
                    for (c, &v) in p.iter().enumerate() {
                        let fixed = (nan_to_num(v).clamp(0.0, 1.0) * WAVEFORM_COLOR_SCALE)
                            .round_ties_even();
                        counts[lane + c * plane + b * WAVEFORM_COLS + col] += fixed as u32;
                    }
                }
                Waveform::Parade => {
                    for (c, &v) in p.iter().enumerate() {
                        let b = quantize(nan_to_num(v), WAVEFORM_BINS);
                        counts[lane + c * plane + b * WAVEFORM_COLS + col] += 1;
                    }
                }
            }
        }
    })
}

/// `scan_full_scale_columns`: (3, WAVEFORM_COLS) counts of samples at or above `level`, per
/// parade column of the crop.
pub fn scan_full_scale_columns(s: Samples, level: f32) -> Vec<f64> {
    let full_w = s.cols.1 - s.cols.0;
    let cols: Vec<usize> = (0..full_w)
        .step_by(s.step)
        .map(|x| (x * WAVEFORM_COLS / full_w).min(WAVEFORM_COLS - 1))
        .collect();
    count_rows(s, 3 * WAVEFORM_COLS, |rgb, counts| {
        for (i, (p, &col)) in rgb.as_chunks::<3>().0.iter().zip(&cols).enumerate() {
            let lane = (i % LANES) * 3 * WAVEFORM_COLS;
            for (c, &v) in p.iter().enumerate() {
                counts[lane + c * WAVEFORM_COLS + col] += (v >= level) as u32;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_and_extremes_land_in_the_edge_bins() {
        let b = Binner {
            lo: DENSITY_HIST_RANGE.0,
            hi: DENSITY_HIST_RANGE.1,
            bins: DENSITY_HIST_BINS,
        };
        let step = (b.hi - b.lo) / b.bins as f64;
        let mut vals = vec![-5.0, 5.0, 1.1, f32::NAN];
        vals.extend((0..DENSITY_HIST_BINS).map(|i| (i as f64 * step + b.lo) as f32));
        let mut out = vec![0u8; vals.len()];
        b.bin_into(&vals, &mut out);
        let last = (DENSITY_HIST_BINS - 1) as u8;
        assert_eq!(out[..4], [0, last, last, NO_BIN]);
        for (i, &o) in out[4..].iter().enumerate() {
            assert_eq!(o as usize, i, "edge {i}");
        }
    }
}

pub const SCOPE_HUE_BINS: usize = 180;
pub const SCOPE_CHROMA_BINS: usize = 64;
const SCOPE_MAX_CHROMA: f64 = 60.0;
const SCOPE_LOG_KNEE: f64 = 4.0;

/// `vectorscope_histogram`: a*b* counts of the sampled pixels, (hue, chroma) bins in float64
/// as `_scope_bins` computes them. The encoded values are clipped to 0..1 (NaN to 0), decoded
/// with `gamma` and converted to Lab by the working-space kernels.
pub fn vectorscope_histogram(s: Samples, space: LabSpace, gamma: f32) -> Vec<f64> {
    let mut rgb = Vec::new();
    let mut buf = Vec::new();
    for y in s.row_starts() {
        rgb.extend_from_slice(s.rgb_row(y, &mut buf));
    }
    for v in rgb.iter_mut() {
        *v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    }
    let mut lin = vec![0f32; rgb.len()];
    color::oetf_decode(&rgb, &mut lin, gamma);
    color::rgb_to_lab(&lin, &mut rgb, space);
    let denom = libm::log1p(SCOPE_MAX_CHROMA / SCOPE_LOG_KNEE);
    let bins = SCOPE_HUE_BINS * SCOPE_CHROMA_BINS;
    rgb.par_chunks(3 * 4096)
        .fold(
            || vec![0u32; bins],
            |mut acc, chunk| {
                let mut idx = [0u16; 4096];
                scope_bins(chunk, &mut idx, denom);
                for &i in &idx[..chunk.len() / 3] {
                    acc[i as usize] += 1;
                }
                acc
            },
        )
        .reduce(
            || vec![0u32; bins],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                a
            },
        )
        .into_iter()
        .map(f64::from)
        .collect()
}

/// `_scope_bins` of Lab triplets: the hue bin of atan2(b, a) and the chroma bin of the log
/// radius, flat as hue * SCOPE_CHROMA_BINS + chroma.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn scope_bins(lab: &[f32], idx: &mut [u16], denom: f64) {
    use std::f64::consts::PI;
    for (i, p) in idx.iter_mut().zip(lab.as_chunks::<3>().0) {
        let (a, b) = (p[1] as f64, p[2] as f64);
        let turn = (crate::mathf::atan2_f64(b, a) + PI) / (2.0 * PI) * SCOPE_HUE_BINS as f64;
        let hue = crate::convert::trunc_bits(turn.floor()) as usize % SCOPE_HUE_BINS;
        let chroma = (a * a + b * b).sqrt();
        let radius = crate::mathf::ln_f64(1.0 + chroma / SCOPE_LOG_KNEE) / denom;
        let c = (radius * SCOPE_CHROMA_BINS as f64)
            .floor()
            .clamp(0.0, (SCOPE_CHROMA_BINS - 1) as f64);
        *i = (hue * SCOPE_CHROMA_BINS + crate::convert::trunc_bits(c) as usize) as u16;
    }
}
