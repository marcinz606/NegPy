//! `cv2.medianBlur` for single-channel uint8: the median of a `k` x `k` window, edges
//! replicated as OpenCV does. A median of integers is exact, so only the border rule matters.

// The steps index several per-column arrays of one row in step.
#![allow(clippy::needless_range_loop)]

use multiversion::multiversion;
use rayon::prelude::*;

pub fn median_blur_u8(src: &[u8], dst: &mut [u8], h: usize, w: usize, k: usize) -> Option<()> {
    if h == 0
        || w == 0
        || k == 0
        || k.is_multiple_of(2)
        || src.len() != h * w
        || dst.len() != src.len()
    {
        return None;
    }
    match k {
        1 => dst.copy_from_slice(src),
        3 => median3(src, dst, h, w),
        5 => small(src, dst, h, w, k),
        _ => histogram(src, dst, h, w, k),
    }
    Some(())
}

/// 3 x 3: each column's three values sorted, then the median of the nine as the middle of
/// (max of the lows, median of the middles, min of the highs) of three neighboring columns.
fn median3(src: &[u8], dst: &mut [u8], h: usize, w: usize) {
    dst.par_chunks_mut(w).enumerate().for_each_init(
        || vec![0u8; 3 * (w + 2)],
        |rows, (y, out)| {
            for (dy, padded) in rows.chunks_exact_mut(w + 2).enumerate() {
                let row = &src[(y + dy).saturating_sub(1).min(h - 1) * w..][..w];
                padded[1..=w].copy_from_slice(row);
                padded[0] = row[0];
                padded[w + 1] = row[w - 1];
            }
            median3_row(rows, out);
        },
    );
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn median3_row(rows: &[u8], out: &mut [u8]) {
    let w = out.len();
    let pw = w + 2;
    let (r0, rest) = rows.split_at(pw);
    let (r1, r2) = rest.split_at(pw);
    let (mut lo, mut mid, mut hi) = (vec![0u8; pw], vec![0u8; pw], vec![0u8; pw]);
    for x in 0..pw {
        let (a, b, c) = (r0[x], r1[x], r2[x]);
        let (a, b) = (a.min(b), a.max(b));
        let (b, c) = (b.min(c), b.max(c));
        let (a, b) = (a.min(b), a.max(b));
        lo[x] = a;
        mid[x] = b;
        hi[x] = c;
    }
    let med3 = |a: u8, b: u8, c: u8| a.min(b).max(a.max(b).min(c));
    for x in 0..w {
        let l = lo[x].max(lo[x + 1]).max(lo[x + 2]);
        let m = med3(mid[x], mid[x + 1], mid[x + 2]);
        let u = hi[x].min(hi[x + 1]).min(hi[x + 2]);
        out[x] = med3(l, m, u);
    }
}

/// Windows up to 5 x 5, a row at a time: the median is the largest t that at least k^2/2 + 1
/// of the window's values reach, found bit by bit, so every step is a flat loop over the row.
fn small(src: &[u8], dst: &mut [u8], h: usize, w: usize, k: usize) {
    let r = k / 2;
    dst.par_chunks_mut(w).enumerate().for_each_init(
        || vec![0u8; k * (w + 2 * r)],
        |rows, (y, out)| {
            // The k source rows, each padded by r replicated pixels on both sides.
            let pw = w + 2 * r;
            for (dy, padded) in rows.chunks_exact_mut(pw).enumerate() {
                let sy = (y + dy).saturating_sub(r).min(h - 1);
                let row = &src[sy * w..][..w];
                padded[r..r + w].copy_from_slice(row);
                padded[..r].fill(row[0]);
                padded[r + w..].fill(row[w - 1]);
            }
            small_row(rows, pw, k, out);
        },
    );
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn small_row(rows: &[u8], pw: usize, k: usize, out: &mut [u8]) {
    let w = out.len();
    let need = (k * k / 2 + 1) as u8;
    out.fill(0);
    let mut count = vec![0u8; w];
    for bit in (0..8).rev() {
        count.fill(0);
        for dy in 0..k {
            let row = &rows[dy * pw..][..pw];
            for dx in 0..k {
                let s = &row[dx..dx + w];
                for x in 0..w {
                    count[x] += (s[x] >= (out[x] | 1 << bit)) as u8;
                }
            }
        }
        for x in 0..w {
            if count[x] >= need {
                out[x] |= 1 << bit;
            }
        }
    }
}

/// Columns per stripe: a stripe's column histograms stay in cache.
const STRIPE: usize = 512;
/// Output rows per task.
const BAND: usize = 128;

/// Perreault and Hebert's constant-time median: column histograms slid down the rows, a 16-bin
/// coarse window histogram slid along each row, and each 16-bin fine segment brought up to date
/// only when the search enters it. Counts fit u16 for any window OpenCV allows.
fn histogram(src: &[u8], dst: &mut [u8], h: usize, w: usize, k: usize) {
    let tasks: Vec<(usize, usize)> = (0..h.div_ceil(BAND))
        .flat_map(|b| (0..w.div_ceil(STRIPE)).map(move |s| (b, s)))
        .collect();
    let parts: Vec<(usize, usize, Vec<u8>)> = tasks
        .into_par_iter()
        .map(|(b, s)| {
            let (y0, x0) = (b * BAND, s * STRIPE);
            let (rows, cols) = (BAND.min(h - y0), STRIPE.min(w - x0));
            let mut out = vec![0u8; rows * cols];
            histogram_tile(src, (h, w), k, (y0, x0), cols, &mut out);
            (y0, x0, out)
        })
        .collect();
    for (y0, x0, out) in parts {
        let cols = STRIPE.min(w - x0);
        for (dy, row) in out.chunks_exact(cols).enumerate() {
            dst[(y0 + dy) * w + x0..][..cols].copy_from_slice(row);
        }
    }
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn histogram_tile(
    src: &[u8],
    (h, w): (usize, usize),
    k: usize,
    (y0, x0): (usize, usize),
    cols: usize,
    out: &mut [u8],
) {
    let r = k / 2;
    let rank = (k * k / 2) as u16;
    let cy = |v: isize| v.clamp(0, h as isize - 1) as usize;
    // Column histograms for the stripe and r columns either side, replicated at the edges.
    // Fine ones by coarse bin, then column: the window slides along one contiguous run.
    let span = cols + 2 * r;
    let mut fine = vec![[0u16; 16]; 16 * span];
    let mut coarse = vec![[0u16; 16]; span];
    let mut padded = vec![0u8; span];
    let fill = |y: usize, padded: &mut [u8]| {
        let row = &src[y * w..][..w];
        for (i, p) in padded.iter_mut().enumerate() {
            *p = row[(x0 + i).saturating_sub(r).min(w - 1)];
        }
    };
    let add = |fine: &mut [[u16; 16]], coarse: &mut [[u16; 16]], padded: &[u8], up: bool| {
        for (i, (&v, c)) in padded.iter().zip(coarse.iter_mut()).enumerate() {
            let (hi, lo) = ((v >> 4) as usize, (v & 15) as usize);
            let f = &mut fine[hi * span + i][lo];
            if up {
                *f += 1;
                c[hi] += 1;
            } else {
                *f -= 1;
                c[hi] -= 1;
            }
        }
    };
    for dy in -(r as isize)..=r as isize {
        fill(cy(y0 as isize + dy), &mut padded);
        add(&mut fine, &mut coarse, &padded, true);
    }
    for (dy, out) in out.chunks_exact_mut(cols).enumerate() {
        let y = (y0 + dy) as isize;
        if dy > 0 {
            fill(cy(y - r as isize - 1), &mut padded);
            add(&mut fine, &mut coarse, &padded, false);
            fill(cy(y + r as isize), &mut padded);
            add(&mut fine, &mut coarse, &padded, true);
        }
        // Output column x reads histogram columns x ..= x + 2r.
        let mut win = [0u16; 16];
        for c in &coarse[..k] {
            for b in 0..16 {
                win[b] += c[b];
            }
        }
        let mut seg = [[0u16; 16]; 16];
        // The output column each fine segment was last summed for.
        let mut synced = [usize::MAX; 16];
        for (x, o) in out.iter_mut().enumerate() {
            let mut seen = 0u16;
            let mut b = 0;
            for (i, &c) in win.iter().enumerate() {
                b = i;
                if seen + c > rank {
                    break;
                }
                seen += c;
            }
            let run = &fine[b * span..][..span];
            let s = &mut seg[b];
            if synced[b] == usize::MAX || synced[b] + k <= x {
                *s = [0; 16];
                for col in &run[x..x + k] {
                    for i in 0..16 {
                        s[i] += col[i];
                    }
                }
            } else {
                for j in synced[b] + 1..=x {
                    let (gone, new) = (&run[j - 1], &run[j + 2 * r]);
                    for i in 0..16 {
                        s[i] = s[i].wrapping_add(new[i]).wrapping_sub(gone[i]);
                    }
                }
            }
            synced[b] = x;
            let mut v = 0;
            for (i, &c) in s.iter().enumerate() {
                v = i;
                if seen + c > rank {
                    break;
                }
                seen += c;
            }
            *o = ((b << 4) + v) as u8;
            if x + 1 < cols {
                let (gone, new) = (&coarse[x], &coarse[x + k]);
                for i in 0..16 {
                    win[i] = win[i].wrapping_add(new[i]).wrapping_sub(gone[i]);
                }
            }
        }
    }
}
