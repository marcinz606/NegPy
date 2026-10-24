//! The bracket-merge kernels of negpy/features/hdr/logic.py.

use multiversion::multiversion;
use rayon::prelude::*;

const BLOCK: usize = 1 << 14;

/// `_count_clipped_rows`: per row, the pixels with any channel at or above `saturation`,
/// compared in float64 as Numba compares a float32 with a Python float.
pub fn count_clipped_rows(
    values: &[f32],
    (w, cn): (usize, usize),
    saturation: f64,
    out: &mut [i64],
) {
    // The smallest float32 at or above the threshold: the float32 compare is then the float64 one.
    let mut limit = saturation as f32;
    if (limit as f64) < saturation {
        limit = f32::from_bits(limit.to_bits() + 1);
    }
    out.par_iter_mut()
        .zip(values.par_chunks(w * cn))
        .for_each(|(n, row)| *n = count_row(row, cn, limit));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn count_row(row: &[f32], cn: usize, limit: f32) -> i64 {
    if cn == 3 {
        return row
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| ((p[0] >= limit) | (p[1] >= limit) | (p[2] >= limit)) as i64)
            .sum();
    }
    row.chunks(cn)
        .map(|p| p.iter().fold(false, |hit, &v| hit | (v >= limit)) as i64)
        .sum()
}

/// `_accumulate`: one frame added to the weighted sums in reference units. Weight is ratio^2,
/// rolled off per channel toward saturation; the order is Numba's vector loop.
pub fn accumulate(
    num: &mut [f32],
    den: &mut [f32],
    values: &[f32],
    ratio: f64,
    (saturation, rolloff_start): (f64, f64),
) {
    // The span is a Python float difference, rounded to float32 once.
    let inv_span = 1.0 / ((saturation - rolloff_start) as f32);
    let saturation = saturation as f32;
    let weight = (ratio as f32) * (ratio as f32);
    let inv_scale = 1.0 / (ratio.max(1e-9) as f32);
    num.par_chunks_mut(1 << 14)
        .zip(den.par_chunks_mut(1 << 14))
        .zip(values.par_chunks(1 << 14))
        .for_each(|((n, d), v)| {
            accumulate_block(n, d, v, (saturation, inv_span, weight, inv_scale))
        });
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn accumulate_block(
    num: &mut [f32],
    den: &mut [f32],
    values: &[f32],
    (sat, inv_span, weight, inv_scale): (f32, f32, f32, f32),
) {
    for ((n, d), &v) in num.iter_mut().zip(den.iter_mut()).zip(values) {
        let r = (sat - v) * inv_span;
        let rolloff = r.clamp(0.0, 1.0);
        let w = rolloff * weight;
        *n = (v * inv_scale).mul_add(w, *n);
        *d += w;
    }
}

/// The first three channels of every `step`-th pixel of every `step`-th row, as
/// `frame[::step, ::step, :3]` reads them, flattened; the frame itself when that is all of it.
fn subsample(
    frame: &[f32],
    (h, w, cn): (usize, usize, usize),
    step: usize,
) -> std::borrow::Cow<'_, [f32]> {
    let step = step.max(1);
    if step == 1 && cn == 3 {
        return frame.into();
    }
    let rows: Vec<usize> = (0..h).step_by(step).collect();
    rows.par_iter()
        .flat_map_iter(|&y| {
            (0..w)
                .step_by(step)
                .flat_map(move |x| (0..3).map(move |c| frame[(y * w + x) * cn + c]))
        })
        .collect::<Vec<f32>>()
        .into()
}

/// `measure_exposure`'s level: the 99th percentile of the subsampled frame.
pub fn exposure_level(
    frame: &[f32],
    dims: (usize, usize, usize),
    step: usize,
    q: f64,
) -> Option<f32> {
    let s = subsample(frame, dims, step);
    crate::stats::np_percentiles(&s, &[q]).map(|v| v[0])
}

/// `pair_ratio`: the median of long / short over the subsampled samples that clear `floor`
/// in the short frame and stay under `saturation` in the long one; `None` under 1000 such
/// samples or for a ratio that is not finite and positive.
pub fn pair_ratio(
    short: &[f32],
    long: &[f32],
    dims: (usize, usize, usize),
    step: usize,
    (floor, saturation): (f32, f32),
) -> Option<f64> {
    let (s, l) = (subsample(short, dims, step), subsample(long, dims, step));
    let ratios: Vec<f32> = s
        .par_chunks(BLOCK)
        .zip(l.par_chunks(BLOCK))
        .flat_map_iter(|(s, l)| {
            s.iter()
                .zip(l)
                .filter(move |&(&s, &l)| s >= floor && l < saturation)
                .map(|(&s, &l)| l / s)
        })
        .collect();
    if ratios.len() < 1000 {
        return None;
    }
    let r = crate::stats::median(&ratios) as f64;
    (r.is_finite() && r > 0.0).then_some(r)
}

/// The end of `merge_providers`: `num / den`, the reference's own value where no frame left a
/// measurement (`den <= 0`).
pub fn finish_merge(num: &[f32], den: &[f32], reference: &[f32], dst: &mut [f32]) {
    dst.par_chunks_mut(BLOCK)
        .zip(num.par_chunks(BLOCK))
        .zip(den.par_chunks(BLOCK).zip(reference.par_chunks(BLOCK)))
        .for_each(|((o, n), (d, r))| {
            for (((o, &n), &d), &r) in o.iter_mut().zip(n).zip(d).zip(r) {
                *o = if d <= 0.0 { r / 1.0 } else { n / d };
            }
        });
}

/// `apply_render_exposure`: the merge times `scale`, clipped to 0..1.
pub fn render_exposure(merged: &[f32], scale: f32, dst: &mut [f32]) {
    dst.par_chunks_mut(BLOCK)
        .zip(merged.par_chunks(BLOCK))
        .for_each(move |(o, m)| {
            for (o, &m) in o.iter_mut().zip(m) {
                let v = m * scale;
                let v = if v.is_nan() || v > 0.0 { v } else { 0.0 };
                *o = if v.is_nan() || v < 1.0 { v } else { 1.0 };
            }
        });
}
