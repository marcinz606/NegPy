//! Ports of the raw probes in negpy/features/rgbscan/logic.py.

use rayon::prelude::*;

use crate::stats::pairwise_sum_f32;

/// `_channel_means`: per CFA group (R = 0, G = 1 or 3, B = 2) the float32 mean of the mosaic
/// values in row-major order, NumPy's pairwise sum, minus `black` in float64. 0 for an empty group.
pub fn channel_means(mosaic: &[f32], colors: &[u8], black: f64) -> [f64; 3] {
    let groups: [&[u8]; 3] = [&[0], &[1, 3], &[2]];
    let means: Vec<f64> = groups
        .par_iter()
        .map(|g| {
            let picked: Vec<f32> = mosaic
                .iter()
                .zip(colors)
                .filter(|(_, c)| g.contains(c))
                .map(|(&v, _)| v)
                .collect();
            if picked.is_empty() {
                return 0.0;
            }
            (pairwise_sum_f32(&picked) / picked.len() as f32) as f64 - black
        })
        .collect();
    [means[0], means[1], means[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_greens_and_skips_empty() {
        let mosaic = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(channel_means(&mosaic, &[0, 1, 3, 1], 2.0), [8.0, 28.0, 0.0]);
    }
}
