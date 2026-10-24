use rayon::prelude::*;

const BUCKETS: usize = 1 << 16;
const CHUNK: usize = 1 << 16;
/// Below this size one `select_nth_unstable` beats the histogram passes.
const DIRECT_MAX: usize = 1 << 17;

/// Total-order key of an f32. Every NaN maps to the top, where `np.partition` puts it.
fn order_key(v: f32) -> u32 {
    if v.is_nan() {
        return u32::MAX;
    }
    let bits = v.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn from_key(key: u32) -> f32 {
    if key & 0x8000_0000 != 0 {
        f32::from_bits(key & 0x7fff_ffff)
    } else {
        f32::from_bits(!key)
    }
}

/// The lower rank of `q` and its interpolation weight, in numpy's float64 arithmetic.
fn virtual_rank(n: usize, q: f64) -> (usize, f64) {
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let lo = (virtual_index.floor().max(0.0) as usize).min(n - 1);
    (lo, virtual_index - lo as f64)
}

/// numpy's interpolation for a Python-number q: the weight joins float32 math (NEP 50),
/// and the far half counts back from the upper sample.
fn interpolate(a: f32, b: f32, t: f64) -> f32 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * ((1.0 - t) as f32)
    } else {
        a + diff * (t as f32)
    }
}

/// The value at each rank in `ranks` (sorted, unique) if `values` were sorted.
fn select_ranks(values: &[f32], ranks: &[usize]) -> Vec<f32> {
    if values.len() <= DIRECT_MAX {
        let mut keys: Vec<u32> = values.iter().map(|&v| order_key(v)).collect();
        return ranks
            .iter()
            .map(|&r| from_key(*keys.select_nth_unstable(r).1))
            .collect();
    }

    // One histogram per thread-sized block, and at least a million values per block: each costs a
    // megabyte to clear and merge. Four interleaved u32 lanes, because on a smooth image
    // neighbours share a bucket and one counter would serialize on its own increments.
    let block = values
        .len()
        .div_ceil(rayon::current_num_threads())
        .max(1 << 20);
    let hist = values
        .par_chunks(block)
        .map(|chunk| {
            let mut h = vec![0u32; 4 * BUCKETS];
            let (quads, rest) = chunk.as_chunks::<4>();
            for q in quads {
                for (j, &v) in q.iter().enumerate() {
                    h[j * BUCKETS + (order_key(v) >> 16) as usize] += 1;
                }
            }
            for &v in rest {
                h[(order_key(v) >> 16) as usize] += 1;
            }
            let mut sum = vec![0usize; BUCKETS];
            for lane in h.as_chunks::<BUCKETS>().0 {
                sum.iter_mut()
                    .zip(lane)
                    .for_each(|(s, &c)| *s += c as usize);
            }
            sum
        })
        .reduce(
            || vec![0usize; BUCKETS],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );

    // Each rank becomes (bucket, rank inside the bucket); ranks are sorted, so one walk finds all.
    let mut located = Vec::with_capacity(ranks.len());
    let mut below = 0usize;
    let mut bucket = 0usize;
    for &r in ranks {
        while below + hist[bucket] <= r {
            below += hist[bucket];
            bucket += 1;
        }
        located.push((bucket, r - below));
    }

    let mut wanted: Vec<u32> = located.iter().map(|&(b, _)| b as u32).collect();
    wanted.dedup();
    let slot = |key: u32| wanted.iter().position(|&b| b == key >> 16);

    // Matches are rare: a vectorized test per block skips the blocks with none.
    let gathered = values
        .par_chunks(CHUNK)
        .fold(
            || vec![Vec::new(); wanted.len()],
            |mut g: Vec<Vec<u32>>, chunk| {
                for block in chunk.chunks(64) {
                    // One pass per wanted bucket keeps the test a plain vector compare.
                    let hit = wanted.iter().fold(false, |any, &b| {
                        any | block
                            .iter()
                            .fold(false, |a, &v| a | (order_key(v) >> 16 == b))
                    });
                    if hit {
                        for &v in block {
                            let key = order_key(v);
                            if let Some(i) = slot(key) {
                                g[i].push(key);
                            }
                        }
                    }
                }
                g
            },
        )
        .reduce(
            || vec![Vec::new(); wanted.len()],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| x.extend(y));
                a
            },
        );

    let mut gathered = gathered;
    located
        .iter()
        .map(|&(b, r)| {
            let keys = &mut gathered[wanted.iter().position(|&w| w as usize == b).unwrap_or(0)];
            from_key(*keys.select_nth_unstable(r).1)
        })
        .collect()
}

/// The two samples around each q's virtual rank and the weight between them.
fn rank_pairs(values: &[f32], qs: &[f64]) -> Vec<(f32, f32, f64)> {
    let n = values.len();
    let spans: Vec<(usize, usize, f64)> = qs
        .iter()
        .map(|&q| {
            let (lo, t) = virtual_rank(n, q);
            (lo, (lo + 1).min(n - 1), t)
        })
        .collect();
    let mut ranks: Vec<usize> = spans.iter().flat_map(|&(lo, hi, _)| [lo, hi]).collect();
    ranks.sort_unstable();
    ranks.dedup();
    let picked = select_ranks(values, &ranks);
    let at = |r: usize| picked[ranks.binary_search(&r).unwrap_or_default()];
    spans
        .iter()
        .map(|&(lo, hi, t)| (at(lo), at(hi), t))
        .collect()
}

/// `np.percentile(values, q)` (method "linear") for each q, bit for bit, except that -0.0 and
/// 0.0 may swap: `np.partition` ranks them equal and returns either. `None` for an empty input.
pub fn percentiles(values: &[f32], qs: &[f64]) -> Option<Vec<f32>> {
    if values.is_empty() {
        return None;
    }
    Some(
        rank_pairs(values, qs)
            .into_iter()
            .map(|(a, b, t)| interpolate(a, b, t))
            .collect(),
    )
}

/// `np.percentile(values, (a, b, ...))`: a sequence of q interpolates in float64. NaN for every
/// q when any value is NaN or the input is empty.
pub fn np_percentiles_f64(values: &[f32], qs: &[f64]) -> Vec<f64> {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; qs.len()];
    }
    rank_pairs(values, qs)
        .into_iter()
        .map(|(a, b, t)| {
            let (a, b) = (a as f64, b as f64);
            if t >= 0.5 {
                b - (b - a) * (1.0 - t)
            } else {
                a + (b - a) * t
            }
        })
        .collect()
}

/// `np.median` of float32 values: the middle one, or the float32 mean of the two middle ones;
/// NaN when any value is NaN or there are none. May reorder `values`.
pub fn np_median(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n > DIRECT_MAX {
        return median(values);
    }
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return f32::NAN;
    }
    let (_, &mut hi, _) = values.select_nth_unstable_by(n / 2, f32::total_cmp);
    if n % 2 == 1 {
        return hi;
    }
    let lo = values[..n / 2]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    (lo + hi) / 2.0
}

/// `np.median` of float32 values without reordering them, by the histogram selection.
pub fn median(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 || values.par_iter().any(|v| v.is_nan()) {
        return f32::NAN;
    }
    if n % 2 == 1 {
        return select_ranks(values, &[n / 2])[0];
    }
    let m = select_ranks(values, &[n / 2 - 1, n / 2]);
    (m[0] + m[1]) / 2.0
}

/// NumPy's float32 sum of a contiguous run: one pairwise sum (blocks of 128, eight
/// accumulators), the order `np.sum` and `mean(axis=-1)` use.
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(-0.0f32, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[i..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
}

/// `np.quantile(values, q)` for a Python-number q in 0..1: the virtual rank is `(n - 1) * q`,
/// not `percentile`'s `(n - 1) * (q / 100)`. NaN when any value is NaN; `None` when empty.
pub fn np_quantile(values: &[f32], q: f64) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    if values.iter().any(|v| v.is_nan()) {
        return Some(f32::NAN);
    }
    let n = values.len();
    let virtual_index = (n - 1) as f64 * q;
    let lo = (virtual_index.floor().max(0.0) as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let mut ranks = vec![lo, hi];
    ranks.dedup();
    let picked = select_ranks(values, &ranks);
    Some(interpolate(
        picked[0],
        picked[ranks.len() - 1],
        virtual_index - lo as f64,
    ))
}

/// `np.percentile` itself, which is NaN for every q when any value is NaN; `percentiles`
/// matches NegPy's partition helper, which callers feed only finite values.
pub fn np_percentiles(values: &[f32], qs: &[f64]) -> Option<Vec<f32>> {
    let out = percentiles(values, qs)?;
    if values.iter().any(|v| v.is_nan()) {
        return Some(vec![f32::NAN; qs.len()]);
    }
    Some(out)
}

/// NumPy's linear interpolation between two float64 samples for a float64 weight: the far
/// half counts back from the upper sample.
fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// `np.quantile(values, q)` of float64 values for q in 0..1 (`np.percentile` passes q / 100).
/// Reorders `values`; `None` when empty.
pub fn quantile_f64(values: &mut [f64], q: f64) -> Option<f64> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let virtual_index = (n - 1) as f64 * q;
    let lo = (virtual_index.floor().max(0.0) as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let a = *values.select_nth_unstable_by(lo, f64::total_cmp).1;
    let b = if hi == lo {
        a
    } else {
        values[lo + 1..]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    };
    Some(lerp_f64(a, b, virtual_index - lo as f64))
}

/// `np.median` of float64 values: the middle one, or the mean of the two middle ones.
/// Reorders `values`; NaN when empty.
pub fn median_f64(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    let hi = *values.select_nth_unstable_by(n / 2, f64::total_cmp).1;
    if n % 2 == 1 {
        return hi;
    }
    let lo = values[..n / 2]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_percentile(values: &[f32], q: f64) -> f32 {
        let mut s = values.to_vec();
        s.sort_by(f32::total_cmp);
        let (lo, t) = virtual_rank(s.len(), q);
        interpolate(s[lo], s[(lo + 1).min(s.len() - 1)], t)
    }

    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn quantile_ranks_at_n_minus_one_times_q() {
        for n in [1, 2, 7, 64, 1000, 70_000] {
            let v = lcg(n, n as u64);
            let mut s = v.clone();
            s.sort_by(f32::total_cmp);
            let vi = (n - 1) as f64 * 0.3;
            let lo = vi.floor() as usize;
            let want = interpolate(s[lo], s[(lo + 1).min(n - 1)], vi - lo as f64);
            assert_eq!(np_quantile(&v, 0.3), Some(want), "{n}");
        }
        assert_eq!(np_quantile(&[], 0.3), None);
    }

    #[test]
    fn pairwise_sum_regions() {
        let v: Vec<f32> = (0..1000).map(|i| (i as f32).sqrt()).collect();
        for n in [0, 1, 7, 8, 9, 128, 129, 1000] {
            let exact: f64 = v[..n].iter().map(|&x| x as f64).sum();
            assert!(
                (pairwise_sum_f32(&v[..n]) as f64 - exact).abs() < 1e-2,
                "n={n}"
            );
        }
    }

    #[test]
    fn empty_is_none() {
        assert!(percentiles(&[], &[50.0]).is_none());
    }

    #[test]
    fn one_element() {
        assert_eq!(
            percentiles(&[3.5], &[0.0, 50.0, 100.0]).unwrap(),
            vec![3.5; 3]
        );
    }

    #[test]
    fn two_elements_interpolate() {
        assert_eq!(
            percentiles(&[1.0, 3.0], &[50.0, 25.0, 75.0]).unwrap(),
            vec![2.0, 1.5, 2.5]
        );
    }

    #[test]
    fn all_equal() {
        let v = vec![0.25f32; 100_000];
        assert_eq!(percentiles(&v, &[1.0, 50.0, 99.0]).unwrap(), vec![0.25; 3]);
    }

    #[test]
    fn infinities_and_signed_zero() {
        let v = [f32::INFINITY, -0.0, 0.0, f32::NEG_INFINITY, 1.0];
        // numpy's lerp gives NaN at an infinite end (inf * 0), and so does the port.
        let got = percentiles(&v, &[0.0, 50.0, 100.0]).unwrap();
        assert!(got[0].is_nan() && got[2].is_nan());
        assert_eq!(got[1], 0.0);
    }

    #[test]
    fn nan_sorts_last() {
        let v = [f32::NAN, 2.0, 1.0, -f32::NAN, 3.0];
        assert_eq!(percentiles(&v, &[0.0, 25.0]).unwrap(), vec![1.0, 2.0]);
        assert!(percentiles(&v, &[50.0]).unwrap()[0].is_nan());
    }

    #[test]
    fn order_key_round_trips_and_orders() {
        let vals = [
            f32::NEG_INFINITY,
            -2.5,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            1e-30,
            7.0,
            f32::INFINITY,
        ];
        for w in vals.windows(2) {
            assert!(order_key(w[0]) < order_key(w[1]));
        }
        for v in vals {
            assert_eq!(from_key(order_key(v)).to_bits(), v.to_bits());
        }
    }

    #[test]
    fn histogram_path_matches_sort() {
        let qs = [0.0, 0.5, 1.0, 20.0, 50.0, 60.0, 80.0, 99.0, 99.5, 100.0];
        for (n, seed) in [(DIRECT_MAX + 1, 1), (1_000_003, 2)] {
            let v = lcg(n, seed);
            let got = percentiles(&v, &qs).unwrap();
            for (g, &q) in got.iter().zip(&qs) {
                assert_eq!(
                    g.to_bits(),
                    sorted_percentile(&v, q).to_bits(),
                    "n={n} q={q}"
                );
            }
        }
    }

    #[test]
    fn histogram_path_with_more_buckets_than_a_byte() {
        let v = lcg(100_000, 4);
        let qs: Vec<f64> = (0..300).map(|i| i as f64 / 3.0).collect();
        let got = percentiles(&v, &qs).unwrap();
        for (g, &q) in got.iter().zip(&qs) {
            assert_eq!(g.to_bits(), sorted_percentile(&v, q).to_bits(), "q={q}");
        }
    }

    #[test]
    fn histogram_path_with_heavy_repeats() {
        let v: Vec<f32> = lcg(200_000, 3)
            .iter()
            .map(|x| (x * 4.0).round() / 4.0)
            .collect();
        let qs = [1.0, 50.0, 99.0];
        let got = percentiles(&v, &qs).unwrap();
        for (g, &q) in got.iter().zip(&qs) {
            assert_eq!(g.to_bits(), sorted_percentile(&v, q).to_bits());
        }
    }
}
