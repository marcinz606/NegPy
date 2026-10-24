//! Ports of the scan statistics in negpy/features/process: process-mode detection, the
//! lightbox level and the chart patch reads.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::pow_f64;
use crate::stats::{np_median, np_percentiles, pairwise_sum_f32, quantile_f64};

/// `x @ m.T` for one pixel, in the order OpenBLAS's dgemm sums it.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn matmul_rows(px: &[f64], m: &[[f64; 3]; 3], dst: &mut [f64]) {
    for (d, x) in dst
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(px.as_chunks::<3>().0)
    {
        for c in 0..3 {
            d[c] = x[2].mul_add(m[c][2], x[1].mul_add(m[c][1], x[0] * m[c][0]));
        }
    }
}

/// `lightbox_level`'s reference: the `pct` percentile of every value of every `step`th pixel
/// of an Nx3 image, through `matrix` in float64 when given. NaN when any value is NaN.
pub fn lightbox_white(img: &[f32], step: usize, matrix: Option<[[f64; 3]; 3]>, pct: f64) -> f64 {
    let mut sample: Vec<f64> = img
        .as_chunks::<3>()
        .0
        .iter()
        .step_by(step.max(1))
        .flat_map(|p| p.map(f64::from))
        .collect();
    if let Some(m) = matrix {
        let src = sample.clone();
        sample
            .par_chunks_mut(3 * 4096)
            .zip(src.par_chunks(3 * 4096))
            .for_each(|(d, s)| matmul_rows(s, &m, d));
    }
    if sample.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    quantile_f64(&mut sample, pct / 100.0).unwrap_or(f64::NAN)
}

/// `_corr`: Pearson correlation of two float32 channels in NumPy's float32 arithmetic.
fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f32;
    let ma = pairwise_sum_f32(a) / n;
    let mb = pairwise_sum_f32(b) / n;
    let da: Vec<f32> = a.iter().map(|v| v - ma).collect();
    let db: Vec<f32> = b.iter().map(|v| v - mb).collect();
    let sq = |x: &[f32], y: &[f32]| {
        pairwise_sum_f32(&x.iter().zip(y).map(|(p, q)| p * q).collect::<Vec<_>>())
    };
    let denom = ((sq(&da, &da) * sq(&db, &db)).sqrt() as f64 + 1e-12) as f32;
    (sq(&da, &db) / denom) as f64
}

/// The statistics `detect_process_mode` reads off an HxWx3 float32 image: the lowest channel
/// correlation, the red and blue means, the red and blue 25th and the RGB 98th percentiles.
pub fn process_mode_stats(img: &[f32]) -> (f64, [f64; 7]) {
    let ch: Vec<Vec<f32>> = (0..3)
        .map(|c| img.iter().skip(c).step_by(3).copied().collect())
        .collect();
    let (r, g, b) = (&ch[0], &ch[1], &ch[2]);
    // Each q interpolates in float32, as a separate np.percentile call with a Python number does.
    let (min_corr, p) = rayon::join(
        || corr(r, g).min(corr(g, b)).min(corr(r, b)),
        || {
            ch.par_iter()
                .map(|v| np_percentiles(v, &[25.0, 98.0]).unwrap_or(vec![f32::NAN; 2]))
                .collect::<Vec<_>>()
        },
    );
    let n = r.len() as f32;
    let (mr, mb) = (
        (pairwise_sum_f32(r) / n) as f64,
        (pairwise_sum_f32(b) / n) as f64,
    );
    let p = |c: usize, i: usize| p[c][i] as f64;
    (
        min_corr,
        [mr, mb, p(0, 0), p(2, 0), p(0, 1), p(1, 1), p(2, 1)],
    )
}

/// `sample_patches` of an HxWxC float32 image: the float32 median per channel of the first three
/// channels in each (x0, x1, y0, y1) box. With `gamma`, of `working_oetf_decode(np.clip(img, 0, 1))`.
pub fn chart_patches(
    img: &[f32],
    w: usize,
    cn: usize,
    boxes: &[(usize, usize, usize, usize)],
    gamma: Option<f32>,
) -> Vec<[f64; 3]> {
    boxes
        .par_iter()
        .map(|&(x0, x1, y0, y1)| {
            std::array::from_fn(|c| {
                let mut v: Vec<f32> = (y0..y1)
                    .flat_map(|yy| (x0..x1).map(move |xx| img[(yy * w + xx) * cn + c]))
                    .map(|v| {
                        let Some(g) = gamma else { return v };
                        // np.clip keeps NaN; the decode maps negatives to 0.
                        let x = v.clamp(0.0, 1.0) as f64;
                        pow_f64(if x < 0.0 { 0.0 } else { x }, g as f64) as f32
                    })
                    .collect();
                np_median(&mut v) as f64
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_channels_correlate_fully() {
        let img: Vec<f32> = (0..300).flat_map(|i| [i as f32 / 300.0; 3]).collect();
        let (c, s) = process_mode_stats(&img);
        assert!((c - 1.0).abs() < 1e-6);
        assert_eq!(s[0], s[1]);
    }

    #[test]
    fn lightbox_white_reads_the_top_percentile() {
        let img: Vec<f32> = (0..1000).flat_map(|i| [i as f32; 3]).collect();
        assert_eq!(lightbox_white(&img, 1, None, 100.0), 999.0);
        assert!(lightbox_white(&[f32::NAN, 0.0, 0.0], 1, None, 50.0).is_nan());
    }
}
