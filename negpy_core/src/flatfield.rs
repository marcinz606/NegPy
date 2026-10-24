//! The flat-field reference checks of negpy/features/flatfield/logic.py, in NumPy's float32
//! operation order.

use rayon::prelude::*;

use crate::stats::{np_percentiles, np_percentiles_f64};

const BLOCK: usize = 1 << 13;

/// `img.mean(axis=2)` of a 3-channel pixel: the three summed in order, then over 3.
#[inline(always)]
fn mean3(p: &[f32]) -> f32 {
    ((p[0] + p[1]) + p[2]) / 3.0
}

/// The scalars of `_lit_mask`, as Python floats.
pub struct Lit {
    pub percentile: f64,
    pub fraction: f64,
    pub min_fraction: f64,
}

/// `_lit_mask(ref)`: true where the mean of the channels clears `fraction` of its
/// `percentile`, eroded 5x5 with the outside dark. `None` when under `min_fraction` of the
/// frame is lit, where the caller takes the whole frame.
pub fn lit_mask(img: &[f32], (h, w): (usize, usize), c: &Lit) -> Option<Vec<bool>> {
    let lum: Vec<f32> = img.par_chunks(3).map(mean3).collect();
    let bar = (c.fraction as f32) * np_percentiles(&lum, &[c.percentile])?[0];
    let above: Vec<u8> = lum.par_iter().map(move |&l| (l > bar) as u8).collect();
    let mut lit = vec![0u8; h * w];
    crate::morph::morphology(
        &above,
        &mut lit,
        (h, w, 1),
        crate::morph::Op::Erode,
        (5, 5),
        1,
        crate::filter::Border::Constant,
    )?;
    // A dark outside clears every pixel whose window leaves the frame.
    for (y, row) in lit.chunks_exact_mut(w).enumerate() {
        if y < 2 || y + 2 >= h {
            row.fill(0);
        } else {
            row[..2.min(w)].fill(0);
            row[w.saturating_sub(2)..].fill(0);
        }
    }
    let count = lit.par_iter().filter(|&&v| v != 0).count();
    (count as f64 >= c.min_fraction * (h * w) as f64).then(|| lit.iter().map(|&v| v != 0).collect())
}

/// Lit pixels with a channel at `level` or over, as a fraction of the lit pixels (float64).
fn clipped_fraction(img: &[f32], lit: &[bool], level: f32) -> f64 {
    let (n, hit) = img
        .par_chunks(3 * BLOCK)
        .zip(lit.par_chunks(BLOCK))
        .map(|(px, l)| {
            let mut n = 0usize;
            let mut hit = 0usize;
            for (p, &l) in px.as_chunks::<3>().0.iter().zip(l) {
                if l {
                    n += 1;
                    hit += (p[0] >= level || p[1] >= level || p[2] >= level) as usize;
                }
            }
            (n, hit)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    hit as f64 / n as f64
}

/// `_is_clipped(image)`'s fraction of lit pixels at the clip level, and whether the frame was
/// too dark for a lit mask.
pub fn clipped(img: &[f32], dims: (usize, usize), c: &Lit, level: f32) -> (f64, bool) {
    let lit = lit_mask(img, dims, c);
    let dark = lit.is_none();
    let lit = lit.unwrap_or_else(|| vec![true; dims.0 * dims.1]);
    (clipped_fraction(img, &lit, level), dark)
}

/// What `_evenness` measures: (low, high, color, clipped fraction).
pub struct Evenness {
    pub low: f64,
    pub high: f64,
    pub color: f64,
    pub clipped: f64,
    /// Too dark for a lit mask: the whole frame was measured.
    pub dark: bool,
}

/// `_evenness(image)`: each channel over its lit-area median (0 off the lit area) into
/// `ratio`, and the spread of the lit pixels about it.
pub fn evenness(
    img: &[f32],
    dims: (usize, usize),
    c: &Lit,
    level: f32,
    ratio: &mut [f32],
) -> Option<Evenness> {
    let n = dims.0 * dims.1;
    let lit = lit_mask(img, dims, c);
    let dark = lit.is_none();
    let lit = lit.unwrap_or_else(|| vec![true; n]);
    let lit_px: Vec<usize> = (0..n).filter(|&p| lit[p]).collect();
    let median: [f32; 3] = std::array::from_fn(|ch| {
        let vals: Vec<f32> = lit_px.par_iter().map(|&p| img[3 * p + ch]).collect();
        let m = crate::stats::median(&vals);
        if m.is_nan() || m > 1e-6 { m } else { 1e-6 }
    });
    ratio
        .par_chunks_mut(3 * BLOCK)
        .zip(img.par_chunks(3 * BLOCK).zip(lit.par_chunks(BLOCK)))
        .for_each(|(r, (px, l))| {
            for ((r, p), &l) in r
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(px.as_chunks::<3>().0.iter())
                .zip(l)
            {
                for ch in 0..3 {
                    r[ch] = if l { p[ch] / median[ch] } else { 0.0 };
                }
            }
        });
    let lum: Vec<f32> = lit_px
        .par_iter()
        .map(|&p| mean3(&ratio[3 * p..3 * p + 3]))
        .collect();
    let dev: Vec<f32> = lum.par_iter().map(|&l| l - 1.0).collect();
    let q = np_percentiles_f64(&dev, &[1.0, 99.0]);
    let color = (0..3)
        .map(|ch| {
            let chroma: Vec<f32> = lit_px
                .par_iter()
                .zip(&lum)
                .map(|(&p, &l)| {
                    ratio[3 * p + ch] / if l.is_nan() || l > 1e-6 { l } else { 1e-6 } - 1.0
                })
                .collect();
            np_percentiles_f64(&chroma, &[1.0, 99.0])
        })
        .flat_map(|q| q.into_iter().map(f64::abs))
        .fold(
            f64::NEG_INFINITY,
            |m, v| if v > m || v.is_nan() { v } else { m },
        );
    Some(Evenness {
        low: q[0],
        high: q[1],
        color,
        clipped: clipped_fraction(img, &lit, level),
        dark,
    })
}

/// `_fill_from_nearest(values, valid)` in place: each invalid pixel takes the values of its
/// nearest valid one (OpenCV's L2 5x5 labels).
pub fn fill_from_nearest(values: &mut [f32], valid: &[bool], (h, w): (usize, usize), cn: usize) {
    if valid.iter().all(|&v| v) || !valid.iter().any(|&v| v) {
        return;
    }
    let holes: Vec<u8> = valid.iter().map(|&v| !v as u8).collect();
    let (_, labels) = crate::distance::distance_transform_labels(&holes, h, w);
    // The labels number the valid pixels from 1 in raster order.
    let donors: Vec<usize> = (0..h * w).filter(|&p| valid[p]).collect();
    for p in 0..h * w {
        if !valid[p] {
            let d = donors[labels[p] as usize - 1];
            for c in 0..cn {
                values[cn * p + c] = values[cn * d + c];
            }
        }
    }
}

/// The scalars of `compute_gain`.
pub struct Gain {
    pub lit: Lit,
    pub sigma: f64,
    pub eps: f32,
    pub min_lit_weight: f32,
    pub range: (f32, f32),
}

/// `compute_gain` of the reference at its working size: the lit-area mean of a normalized
/// Gaussian blur over the blur, per channel. Returns whether the reference was too dark for a
/// lit mask.
pub fn compute_gain(
    reference: &[f32],
    (h, w): (usize, usize),
    c: &Gain,
    gain: &mut [f32],
) -> Option<bool> {
    use crate::filter::{Border, gaussian_blur};
    let n = h * w;
    let lit_mask = lit_mask(reference, (h, w), &c.lit);
    let dark = lit_mask.is_none();
    let lit: Vec<f32> = match lit_mask {
        Some(m) => m.iter().map(|&v| v as u8 as f32).collect(),
        None => vec![1.0; n],
    };
    let weighted: Vec<f32> = reference
        .as_chunks::<3>()
        .0
        .iter()
        .zip(&lit)
        .map(|(p, &l)| [p[0] * l, p[1] * l, p[2] * l])
        .collect::<Vec<_>>()
        .into_flattened();
    let (mut num, mut den) = (vec![0f32; 3 * n], vec![0f32; n]);
    let s = (c.sigma, c.sigma);
    gaussian_blur(
        &weighted,
        &mut num,
        (h, w, 3),
        (0, 0),
        s,
        Border::Reflect101,
    )?;
    gaussian_blur(&lit, &mut den, (h, w, 1), (0, 0), s, Border::Reflect101)?;
    let eps = c.eps;
    let clip_lo = |v: f32, lo: f32| if v.is_nan() || v > lo { v } else { lo };
    for (b, &d) in num.as_chunks_mut::<3>().0.iter_mut().zip(&den) {
        let d = clip_lo(d, eps);
        for v in b {
            *v = clip_lo(*v / d, eps);
        }
    }
    let valid: Vec<bool> = den.iter().map(|&d| d >= c.min_lit_weight).collect();
    fill_from_nearest(&mut num, &valid, (h, w), 3);
    // (blur * lit).sum(axis=(0, 1)): one running sum per channel in raster order.
    let mut sums = [0f32; 3];
    for (b, &l) in num.as_chunks::<3>().0.iter().zip(&lit) {
        for ch in 0..3 {
            sums[ch] += b[ch] * l;
        }
    }
    let total = crate::stats::pairwise_sum_f32(&lit);
    let means = sums.map(|s| s / total);
    let (lo, hi) = c.range;
    for (g, b) in gain
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(num.as_chunks::<3>().0.iter())
    {
        for ch in 0..3 {
            let v = means[ch] / b[ch];
            let v = if v.is_nan() || v > lo { v } else { lo };
            g[ch] = if v.is_nan() || v < hi { v } else { hi };
        }
    }
    Some(dark)
}

/// `evenness_view`'s map, in place on the ratio: ±`span` about 1 onto 0..1.
pub fn evenness_view(ratio: &mut [f32], span: f32) {
    ratio.par_chunks_mut(BLOCK).for_each(move |run| {
        for v in run {
            let m = 0.5 + (*v - 1.0) / span;
            let m = if m.is_nan() || m > 0.0 { m } else { 0.0 };
            *v = if m.is_nan() || m < 1.0 { m } else { 1.0 };
        }
    });
}
