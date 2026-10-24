//! `cv2.connectedComponents(WithStats)` for a uint8 mask, with OpenCV's labels: its Spaghetti
//! labelling numbers 4-connected components in raster order of their first pixel, and
//! 8-connected ones in raster order of their first 2x2 block. The numbering does not depend on
//! OpenCV's thread count.

pub struct Components {
    pub count: usize,
    pub labels: Vec<i32>,
    /// Per label: left, top, width, height, area.
    pub stats: Vec<[i32; 5]>,
    pub centroids: Vec<[f64; 2]>,
}

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let p = parent[i as usize];
        parent[i as usize] = parent[p as usize];
        i = p;
    }
    i
}

fn union(parent: &mut [u32], a: u32, b: u32) -> u32 {
    let (ra, rb) = (find(parent, a), find(parent, b));
    let (lo, hi) = (ra.min(rb), ra.max(rb));
    parent[hi as usize] = lo;
    lo
}

/// The `[x0, x1)` runs of nonzero bytes in a row; zero words of eight are skipped whole.
fn row_runs(row: &[u8]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut x = 0;
    let w = row.len();
    while x < w {
        if x + 8 <= w && row[x..x + 8].iter().all(|&v| v == 0) {
            x += 8;
            continue;
        }
        if row[x] == 0 {
            x += 1;
            continue;
        }
        let start = x;
        while x < w && row[x] != 0 {
            x += 1;
        }
        runs.push((start, x));
    }
    runs
}

/// Components of the nonzero pixels of an `h` x `w` mask, `connectivity` 4 or 8, labelled
/// over runs of each row so that only foreground pixels cost work.
pub fn connected_components(mask: &[u8], h: usize, w: usize, connectivity: u8) -> Components {
    use rayon::prelude::*;
    let rows: Vec<Vec<(usize, usize)>> = mask.par_chunks(w.max(1)).map(row_runs).collect();
    // An 8-connected run touches runs above that reach one column past either end.
    let reach = usize::from(connectivity == 8);
    let mut parent: Vec<u32> = vec![0];
    let mut run_label: Vec<Vec<u32>> = Vec::with_capacity(h);
    for y in 0..h {
        let mut labels = Vec::with_capacity(rows[y].len());
        let above = if y > 0 { &rows[y - 1][..] } else { &[] };
        let above_labels = if y > 0 { &run_label[y - 1][..] } else { &[] };
        let mut j = 0;
        for &(x0, x1) in &rows[y] {
            let lo = x0.saturating_sub(reach);
            let hi = x1 + reach;
            while j < above.len() && above[j].1 <= lo {
                j += 1;
            }
            let mut l = 0u32;
            let mut k = j;
            while k < above.len() && above[k].0 < hi {
                let a = above_labels[k];
                l = if l == 0 {
                    find(&mut parent, a)
                } else {
                    union(&mut parent, l, a)
                };
                k += 1;
            }
            if l == 0 {
                parent.push(parent.len() as u32);
                l = (parent.len() - 1) as u32;
            }
            labels.push(l);
        }
        run_label.push(labels);
    }

    // Number the roots by the first pixel (4-connected) or 2x2 block (8-connected) of their
    // component in raster order; a run's first pixel is its smallest key.
    let blocks_w = w.div_ceil(2);
    let mut first = vec![usize::MAX; parent.len()];
    for y in 0..h {
        for (&(x0, _), &l) in rows[y].iter().zip(&run_label[y]) {
            let root = find(&mut parent, l) as usize;
            let key = if connectivity == 8 {
                (y / 2) * blocks_w + x0 / 2
            } else {
                y * w + x0
            };
            first[root] = first[root].min(key);
        }
    }
    let mut roots: Vec<usize> = (1..parent.len())
        .filter(|&r| first[r] != usize::MAX)
        .collect();
    roots.sort_by_key(|&r| first[r]);
    let mut final_label = vec![0i32; parent.len()];
    for (i, &r) in roots.iter().enumerate() {
        final_label[r] = i as i32 + 1;
    }

    let count = roots.len() + 1;
    let mut labels = vec![0i32; h * w];
    let mut lo = vec![[i32::MAX, i32::MAX, i32::MIN, i32::MIN, 0i32]; count];
    let mut integrals = vec![[0u64; 2]; count];
    let sum_to = |n: usize| (n as u64) * (n as u64).saturating_sub(1) / 2;
    for y in 0..h {
        // The background between this row's runs.
        let mut gaps = Vec::with_capacity(rows[y].len() + 1);
        let mut x = 0;
        for &(x0, x1) in &rows[y] {
            if x0 > x {
                gaps.push((x, x0));
            }
            x = x1;
        }
        if x < w {
            gaps.push((x, w));
        }
        let fg = rows[y].iter().zip(&run_label[y]).map(|(&r, &l)| {
            let root = find(&mut parent, l) as usize;
            (r, final_label[root])
        });
        let spans: Vec<((usize, usize), i32)> =
            fg.chain(gaps.into_iter().map(|g| (g, 0))).collect();
        for ((x0, x1), l) in spans {
            if l != 0 {
                labels[y * w + x0..y * w + x1].fill(l);
            }
            let s = &mut lo[l as usize];
            let yi = y as i32;
            s[0] = s[0].min(x0 as i32);
            s[1] = s[1].min(yi);
            s[2] = s[2].max(x1 as i32 - 1);
            s[3] = s[3].max(yi);
            s[4] += (x1 - x0) as i32;
            integrals[l as usize][0] += sum_to(x1) - sum_to(x0);
            integrals[l as usize][1] += (y * (x1 - x0)) as u64;
        }
    }
    let mut stats = Vec::with_capacity(count);
    let mut centroids = Vec::with_capacity(count);
    for (s, i) in lo.iter().zip(&integrals) {
        let area = s[4] as f64;
        if area > 0.0 {
            stats.push([s[0], s[1], s[2] - s[0] + 1, s[3] - s[1] + 1, s[4]]);
            centroids.push([i[0] as f64 / area, i[1] as f64 / area]);
        } else {
            stats.push([-1, s[1], 0, 0, 0]);
            centroids.push([f64::NAN, f64::NAN]);
        }
    }
    Components {
        count,
        labels,
        stats,
        centroids,
    }
}
