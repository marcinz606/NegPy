//! Ports of the pixel work in negpy/infrastructure/loaders.

use rayon::prelude::*;

use crate::stats::quantile_f64;

/// `normalize_linear`: each channel of a float64 image with `c` channels stretched from its
/// 0.2nd to its 99.8th percentile onto 0..1 and clipped. A flat channel divides by 1.
pub fn normalize_linear(lin: &[f64], c: usize, dst: &mut [f64]) {
    let bounds: Vec<(f64, f64)> = (0..c)
        .into_par_iter()
        .map(|ch| {
            let mut v: Vec<f64> = lin.iter().skip(ch).step_by(c).copied().collect();
            let lo = quantile_f64(&mut v, 0.2 / 100.0).unwrap_or(f64::NAN);
            let hi = quantile_f64(&mut v, 99.8 / 100.0).unwrap_or(f64::NAN);
            let denom = hi - lo;
            (lo, if denom > 0.0 { denom } else { 1.0 })
        })
        .collect();
    dst.par_chunks_mut(c * 4096)
        .zip(lin.par_chunks(c * 4096))
        .for_each(|(d, s)| {
            for (d, s) in d.chunks_mut(c).zip(s.chunks(c)) {
                for ch in 0..c {
                    let (lo, denom) = bounds[ch];
                    d[ch] = ((s[ch] - lo) / denom).clamp(0.0, 1.0);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_channel_maps_to_zero() {
        let lin: Vec<f64> = (0..3000)
            .map(|i| if i % 3 == 1 { 5.0 } else { (i / 3) as f64 })
            .collect();
        let mut out = vec![0f64; lin.len()];
        normalize_linear(&lin, 3, &mut out);
        assert!(out.iter().skip(1).step_by(3).all(|&v| v == 0.0));
        assert_eq!(out[0], 0.0);
        assert_eq!(out[2997], 1.0);
    }
}
