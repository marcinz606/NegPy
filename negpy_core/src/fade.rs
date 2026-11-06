//! Fade Restoration's measurements on the Estimate grid (`negpy/features/process/fade.py`).

use rayon::prelude::*;

/// Whether a cell holds a reading: no decoded zero in any channel and not full scale in all three.
pub fn is_reading(p: &[f32; 3], clipped_log: f64) -> bool {
    p.iter().all(|&v| f64::from(v) > clipped_log) && p.iter().any(|&v| v < 0.0)
}

/// Each q's percentile of every channel of `cells`, as `np.percentile(cells[:, ch], q)` gives it.
fn channel_percentiles(cells: &[[f32; 3]], qs: &[f64]) -> Option<Vec<[f32; 3]>> {
    let per: Vec<Vec<f32>> = (0..3)
        .map(|ch| {
            let column: Vec<f32> = cells.par_iter().map(|c| c[ch]).collect();
            crate::stats::percentiles(&column, qs)
        })
        .collect::<Option<_>>()?;
    Some(
        (0..qs.len())
            .map(|i| [per[0][i], per[1][i], per[2][i]])
            .collect(),
    )
}

/// `_measured_slide_bounds`' percentiles: each q's percentile of each channel over the cells
/// with no decoded zero, or over every cell when all hold one. None for an empty grid.
pub fn slide_extremes(grid: &[[f32; 3]], clipped_log: f64, qs: &[f64]) -> Option<Vec<[f32; 3]>> {
    let unclipped: Vec<[f32; 3]> = grid
        .par_iter()
        .filter(|c| c.iter().all(|&v| f64::from(v) > clipped_log))
        .copied()
        .collect();
    channel_percentiles(
        if unclipped.is_empty() {
            grid
        } else {
            &unclipped
        },
        qs,
    )
}

/// `measure_white`: each channel's q percentile over the readings, every channel capped at
/// full scale first. None with no reading.
pub fn white(grid: &[[f32; 3]], clipped_log: f64, q: f64) -> Option<[f32; 3]> {
    let cells: Vec<[f32; 3]> = grid
        .par_iter()
        .map(|c| c.map(|v| v.min(0.0)))
        .filter(|c| is_reading(c, clipped_log))
        .collect();
    Some(channel_percentiles(&cells, &[q])?[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, cast: [f32; 3]) -> Vec<[f32; 3]> {
        (0..n)
            .map(|i| {
                let v = -2.5 + 2.4 * i as f32 / n as f32;
                [v + cast[0], v + cast[1], v + cast[2]]
            })
            .collect()
    }

    #[test]
    fn the_white_skips_zeros_and_light_past_the_film() {
        let mut cells = ramp(1000, [0.0; 3]);
        cells.extend([[0.2f32, 0.1, 0.3], [-9.0, -1.0, -1.0]]);
        let w = white(&cells, -5.0, 100.0).unwrap();
        assert_eq!(w, cells[999]);
        assert!(white(&[[0.0f32; 3]; 4], -5.0, 50.0).is_none());
    }

    #[test]
    fn slide_extremes_fall_back_to_every_cell() {
        let zeros = vec![[-9.0f32, -1.0, -2.0]; 5];
        assert_eq!(
            slide_extremes(&zeros, -5.0, &[50.0]).unwrap(),
            vec![[-9.0, -1.0, -2.0]]
        );
    }
}
