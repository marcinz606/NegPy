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

/// `np.clip(px.astype(np.float32) * scale, 0, 1)` of 4-channel pixels, split into the RGB
/// pixels and a plane of the fourth channel.
pub fn split_rgb_ir<T: Copy + Into<f32> + Sync>(
    src: &[T],
    scale: f32,
    rgb: &mut [f32],
    ir: &mut [f32],
) {
    const BLOCK: usize = 1 << 14;
    rgb.par_chunks_mut(3 * BLOCK)
        .zip(ir.par_chunks_mut(BLOCK))
        .zip(src.par_chunks(4 * BLOCK))
        .for_each(move |((r, i), s)| {
            let unit = |v: T| (v.into() * scale).clamp(0.0, 1.0);
            for ((r, i), p) in r
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(i)
                .zip(s.as_chunks::<4>().0)
            {
                *r = [unit(p[0]), unit(p[1]), unit(p[2])];
                *i = unit(p[3]);
            }
        });
}

/// `np.clip((v - black) / denom, 0, 1)` of RGB pixels in float64, rounded to float32 once.
/// With `lut`, each value is first `lut[int(clip(v, 0, len(lut) - 1))]`.
pub fn levels_to_unit<T: Copy + Into<f64> + Sync>(
    src: &[T],
    lut: Option<&[f64]>,
    black: [f64; 3],
    denom: [f64; 3],
    dst: &mut [f32],
) {
    let unit = move |v: f64, c: usize| ((v - black[c]) / denom[c]).clamp(0.0, 1.0) as f32;
    let (d, s) = (dst.as_chunks_mut::<3>().0, src.as_chunks::<3>().0);
    match lut {
        Some(lut) => {
            let top = (lut.len() - 1) as f64;
            crate::par_zip(d, s, move |d, p| {
                for c in 0..3 {
                    d[c] = unit(lut[p[c].into().clamp(0.0, top) as usize], c);
                }
            });
        }
        None => crate::par_zip(d, s, move |d, p| {
            for c in 0..3 {
                d[c] = unit(p[c].into(), c);
            }
        }),
    }
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

    #[test]
    fn levels_index_the_table_and_clip() {
        let src: [u16; 6] = [0, 5, 9, 1, 2, 3];
        let lut = [0.0, 10.0, 20.0, 30.0];
        let mut out = [0f32; 6];
        levels_to_unit(&src, Some(&lut), [10.0; 3], [20.0; 3], &mut out);
        assert_eq!(out, [0.0, 1.0, 1.0, 0.0, 0.5, 1.0]);
    }
}
