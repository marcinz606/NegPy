//! `cv2.distanceTransform` for a uint8 mask: chamfer distances to the nearest zero pixel.
//! Without labels, opencv-python runs IPP's float32 two-pass chamfer; with labels, OpenCV's own
//! 5x5 pass in 16-bit fixed point. IPP's 3x3 pass picks a path one float32 unit longer at rare
//! pixels than this two-pass order does.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// DIST_C, 3x3: steps of 1.
    Chessboard,
    /// DIST_L2, 3x3: 0.955 and 1.3693.
    L2Mask3,
    /// DIST_L2, 5x5: 1, 1.4 and 2.1969.
    L2Mask5,
}

/// (row offset, column offset, step index) of the forward pass; the backward pass mirrors it.
fn forward(metric: Metric) -> &'static [(isize, isize, usize)] {
    match metric {
        Metric::Chessboard | Metric::L2Mask3 => &[(-1, -1, 1), (-1, 0, 0), (-1, 1, 1), (0, -1, 0)],
        Metric::L2Mask5 => &[
            (-2, -1, 2),
            (-2, 1, 2),
            (-1, -2, 2),
            (-1, -1, 1),
            (-1, 0, 0),
            (-1, 1, 1),
            (-1, 2, 2),
            (0, -1, 0),
        ],
    }
}

fn steps(metric: Metric) -> [f32; 3] {
    match metric {
        Metric::Chessboard => [1.0, 1.0, 0.0],
        Metric::L2Mask3 => [0.955, 1.3693, 0.0],
        Metric::L2Mask5 => [1.0, 1.4, 2.1969],
    }
}

/// The smaller of two finite distances: a plain compare, with no NaN handling in the chain.
fn lo(x: f32, y: f32) -> f32 {
    if y < x { y } else { x }
}

/// Distances in float32: a forward and a backward pass over a frame two pixels wide. Each row
/// first takes the minimum over the rows above (below) in a vector loop, then carries the
/// left (right) neighbour in a short sequential loop: the same sums and an exact minimum, so
/// the same bits as one combined pass.
pub fn distance_transform(src: &[u8], h: usize, w: usize, metric: Metric) -> Vec<f32> {
    const B: usize = 2;
    let pw = w + 2 * B;
    let far = f32::MAX;
    let mut t = vec![far; pw * (h + 2 * B)];
    let [a, b, c] = steps(metric);
    let long = metric == Metric::L2Mask5;
    let mut cand = vec![far; w];
    for y in B..h + B {
        let (above, rest) = t.split_at_mut(y * pw);
        let (up2, up1) = above[(y - 2) * pw..].split_at(pw);
        let row = &mut rest[..pw];
        let s = &src[(y - B) * w..(y - B + 1) * w];
        vertical(&mut cand, up1, up2, (a, b, c), long);
        for x in B..w + B {
            row[x] = if s[x - B] == 0 {
                0.0
            } else {
                lo(cand[x - B], row[x - 1] + a)
            };
        }
    }
    let mut out = vec![0f32; h * w];
    for y in (B..h + B).rev() {
        let (head, below) = t.split_at_mut((y + 1) * pw);
        let row = &mut head[y * pw..];
        let (dn1, dn2) = below.split_at(pw);
        vertical_back(&mut cand, dn1, dn2, (a, b, c), long);
        let o = &mut out[(y - B) * w..(y - B + 1) * w];
        for x in (B..w + B).rev() {
            let v0 = row[x];
            if v0 > a {
                row[x] = lo(lo(v0, cand[x - B]), row[x + 1] + a);
            }
            o[x - B] = row[x];
        }
    }
    out
}

/// The forward candidates from the two rows above, for the row's `w` pixels.
fn vertical(cand: &mut [f32], up1: &[f32], up2: &[f32], (a, b, c): (f32, f32, f32), long: bool) {
    for (x, v) in cand.iter_mut().enumerate() {
        let x = x + 2;
        *v = lo(lo(up1[x - 1] + b, up1[x] + a), up1[x + 1] + b);
    }
    if long {
        for (x, v) in cand.iter_mut().enumerate() {
            let x = x + 2;
            *v = lo(
                lo(lo(lo(*v, up2[x - 1] + c), up2[x + 1] + c), up1[x - 2] + c),
                up1[x + 2] + c,
            );
        }
    }
}

/// The backward candidates from the two rows below.
fn vertical_back(
    cand: &mut [f32],
    dn1: &[f32],
    dn2: &[f32],
    (a, b, c): (f32, f32, f32),
    long: bool,
) {
    for (x, v) in cand.iter_mut().enumerate() {
        let x = x + 2;
        *v = lo(lo(dn1[x + 1] + b, dn1[x] + a), dn1[x - 1] + b);
    }
    if long {
        for (x, v) in cand.iter_mut().enumerate() {
            let x = x + 2;
            *v = lo(
                lo(lo(lo(*v, dn2[x + 1] + c), dn2[x - 1] + c), dn1[x + 2] + c),
                dn1[x - 2] + c,
            );
        }
    }
}

/// `cv2.distanceTransformWithLabels(src, DIST_L2, ..., labelType=DIST_LABEL_PIXEL)`: OpenCV's
/// 5x5 fixed-point pass, each pixel labelled with its nearest zero pixel, zeros numbered from 1
/// in raster order.
pub fn distance_transform_labels(src: &[u8], h: usize, w: usize) -> (Vec<f32>, Vec<i32>) {
    const B: usize = 2;
    const SHIFT: i32 = 16;
    let fix = |v: f32| ((v as f64) * (1 << SHIFT) as f64).round_ties_even() as u32;
    let [hv, diag, long] = steps(Metric::L2Mask5).map(fix);
    let cost = [hv, diag, long];
    let far = u32::MAX - long;
    let scale = 1.0f32 / (1 << SHIFT) as f32;
    let (pw, ph) = (w + 2 * B, h + 2 * B);
    let mut t = vec![far; pw * ph];
    let mut lab = vec![0i32; pw * ph];
    let mut next = 1;
    for y in 0..h {
        for x in 0..w {
            if src[y * w + x] == 0 {
                lab[(y + B) * pw + x + B] = next;
                next += 1;
            }
        }
    }
    let fwd = forward(Metric::L2Mask5);
    let at = |y: usize, x: usize, dy: isize, dx: isize| {
        ((y as isize + dy) as usize) * pw + (x as isize + dx) as usize
    };
    for y in B..h + B {
        for x in B..w + B {
            if src[(y - B) * w + x - B] == 0 {
                t[y * pw + x] = 0;
                continue;
            }
            let (mut v, mut l) = (far, 0);
            for &(dy, dx, s) in fwd {
                let i = at(y, x, dy, dx);
                let c = t[i].wrapping_add(cost[s]);
                if v > c {
                    v = c;
                    l = lab[i];
                }
            }
            t[y * pw + x] = v;
            lab[y * pw + x] = l;
        }
    }
    let mut dist = vec![0f32; h * w];
    let mut labels = vec![0i32; h * w];
    for y in (B..h + B).rev() {
        for x in (B..w + B).rev() {
            let (mut v, mut l) = (t[y * pw + x], lab[y * pw + x]);
            if v > hv {
                for &(dy, dx, s) in fwd {
                    let i = at(y, x, -dy, -dx);
                    let c = t[i].wrapping_add(cost[s]);
                    if v > c {
                        v = c;
                        l = lab[i];
                    }
                }
                t[y * pw + x] = v;
                lab[y * pw + x] = l;
            }
            dist[(y - B) * w + x - B] = v as f32 * scale;
            labels[(y - B) * w + x - B] = l;
        }
    }
    (dist, labels)
}

/// `_mask_to_score(mask, pad_px)` of retouch: the floor on the defect (nonzero), rising to 1 over
/// `pad` pixels of 3x3 chamfer distance along a smoothstep, in float32.
pub fn mask_to_score(mask: &[u8], h: usize, w: usize, pad: f64, floor: f64) -> Vec<f32> {
    use rayon::prelude::*;
    let clean: Vec<u8> = mask.par_iter().map(|&m| (m == 0) as u8).collect();
    let (pad, floor, rise) = (pad.max(1e-3) as f32, floor as f32, (1.0 - floor) as f32);
    // The score is flat past `pad`, and a chamfer path no longer than `pad` moves at most
    // pad / 0.955 rows, so bands with that much margin give every unclamped distance exactly.
    let halo = (pad / steps(Metric::L2Mask3)[0]).ceil() as usize + 2;
    let band = 64.max(h.div_ceil(4 * rayon::current_num_threads()));
    let mut out = vec![0f32; h * w];
    out.par_chunks_mut(band * w.max(1))
        .enumerate()
        .for_each(|(i, rows)| {
            let (y0, y1) = (i * band, i * band + rows.len() / w.max(1));
            let (lo, hi) = (y0.saturating_sub(halo), (y1 + halo).min(h));
            let d = distance_transform(&clean[lo * w..hi * w], hi - lo, w, Metric::L2Mask3);
            for (o, &v) in rows.iter_mut().zip(&d[(y0 - lo) * w..]) {
                let t = (v / pad).clamp(0.0, 1.0);
                *o = floor + rise * (t * t * (3.0 - 2.0 * t));
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banded_score_matches_one_pass() {
        let (h, w) = (700, 97);
        let mut seed = 7u64;
        let mut mask = vec![0u8; h * w];
        for m in mask.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *m = (seed >> 33).is_multiple_of(400) as u8;
        }
        for pad in [0.5, 2.5, 9.0, 40.0] {
            let clean: Vec<u8> = mask.iter().map(|&m| (m == 0) as u8).collect();
            let (p, floor) = (pad as f32, 0.02f32);
            let want: Vec<f32> = distance_transform(&clean, h, w, Metric::L2Mask3)
                .iter()
                .map(|&v| {
                    let t = (v / p).clamp(0.0, 1.0);
                    floor + (1.0 - floor) * (t * t * (3.0 - 2.0 * t))
                })
                .collect();
            let got = mask_to_score(&mask, h, w, pad, floor as f64);
            assert!(
                want.iter()
                    .zip(&got)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "pad {pad}"
            );
        }
    }
}
