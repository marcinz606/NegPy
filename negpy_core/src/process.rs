//! Ports of the scan statistics in negpy/features/process: process-mode detection, the
//! lightbox level and the chart patch reads.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::{log10_f64, pow_f64};
use crate::stats::{median_f64, np_median, np_percentiles, pairwise_sum_f32, quantile_f64};

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

/// Robust spread of an HxWx3 image's color around its own neutral axis, in log10 units: block
/// means of `block`-pixel squares, each pixel inside (`lo`, `hi`) in every channel, then the
/// residuals of log R − log G and log B − log G after a straight-line fit on log G. Silver
/// leaves only noise there; dyes leave the scene's hues. None under `min_share` usable blocks.
pub fn chroma_spread(
    img: &[f32],
    w: usize,
    block: usize,
    lo: f32,
    hi: f32,
    min_share: f64,
) -> Option<f64> {
    let block = block.max(1);
    let h = img.len() / (3 * w.max(1));
    let (bh, bw) = (h / block, w / block);
    let mut d = Vec::with_capacity(bh * bw);
    let mut rg = Vec::with_capacity(bh * bw);
    let mut bg = Vec::with_capacity(bh * bw);
    for by in 0..bh {
        'blocks: for bx in 0..bw {
            let mut sum = [0.0f64; 3];
            for y in by * block..(by + 1) * block {
                let row = &img[(y * w + bx * block) * 3..(y * w + (bx + 1) * block) * 3];
                for px in row.as_chunks::<3>().0 {
                    if px.iter().any(|&v| !(v > lo && v < hi)) {
                        continue 'blocks;
                    }
                    for c in 0..3 {
                        sum[c] += f64::from(px[c]);
                    }
                }
            }
            let [r, g, b] = sum.map(log10_f64);
            d.push(g);
            rg.push(r - g);
            bg.push(b - g);
        }
    }
    if d.is_empty() || (d.len() as f64) < min_share * (bh * bw) as f64 {
        return None;
    }
    let n = d.len() as f64;
    let md = d.iter().sum::<f64>() / n;
    let vd: f64 = d.iter().map(|x| (x - md) * (x - md)).sum();
    let spread = |y: &[f64]| {
        let my = y.iter().sum::<f64>() / n;
        let cov: f64 = d.iter().zip(y).map(|(x, v)| (x - md) * (v - my)).sum();
        let slope = if vd > 0.0 { cov / vd } else { 0.0 };
        let mut r: Vec<f64> = d
            .iter()
            .zip(y)
            .map(|(x, v)| v - my - slope * (x - md))
            .collect();
        let mid = median_f64(&mut r);
        let mut dev: Vec<f64> = r.iter().map(|v| (v - mid).abs()).collect();
        1.4826 * median_f64(&mut dev)
    };
    let (a, b) = (spread(&rg), spread(&bg));
    Some(a.hypot(b))
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
    fn a_tinted_gray_has_no_chroma_spread() {
        let img: Vec<f32> = (0..64 * 64)
            .flat_map(|i| {
                let v = 0.05 + 0.9 * i as f32 / 4096.0;
                [v * 0.5, v, v.powf(1.1) * 0.7]
            })
            .collect();
        assert!(chroma_spread(&img, 64, 4, 0.003, 0.97, 0.1).unwrap() < 1e-3);
    }

    #[test]
    fn independent_channels_spread_and_clipped_blocks_drop_out() {
        let mut s = 1u32;
        let mut rnd = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            0.05 + 0.9 * (s >> 8) as f32 / (1 << 24) as f32
        };
        let img: Vec<f32> = (0..64 * 64 * 3).map(|_| rnd()).collect();
        assert!(chroma_spread(&img, 64, 4, 0.003, 0.97, 0.1).unwrap() > 0.05);
        assert_eq!(
            chroma_spread(&vec![1.0; 64 * 64 * 3], 64, 4, 0.003, 0.97, 0.1),
            None
        );
    }

    #[test]
    fn lightbox_white_reads_the_top_percentile() {
        let img: Vec<f32> = (0..1000).flat_map(|i| [i as f32; 3]).collect();
        assert_eq!(lightbox_white(&img, 1, None, 100.0), 999.0);
        assert!(lightbox_white(&[f32::NAN, 0.0, 0.0], 1, None, 50.0).is_nan());
    }
}
