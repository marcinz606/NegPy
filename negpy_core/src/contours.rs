//! `cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`, OpenCV 4.13's
//! Suzuki–Abe scanner (contours_new.cpp) with its marks, so contours and points come out in
//! OpenCV's order.

const DELTAS: [(isize, isize); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
const RIGHT: i8 = -128; // 0x80
const NEW: i8 = 2;
const FLAGS: i8 = -2; // 0xFE

/// Outer contours of the nonzero pixels of an `h` x `w` uint8 image, each a list of (x, y)
/// corner points, in OpenCV's output order (last found first).
pub fn find_external_contours(src: &[u8], h: usize, w: usize) -> Vec<Vec<(i32, i32)>> {
    // One pixel of zero border, values thresholded to 0/1, as findContours prepares them.
    let (pw, ph) = (w + 2, h + 2);
    let mut img = vec![0i8; pw * ph];
    for (line, row) in img.chunks_exact_mut(pw).skip(1).zip(src.chunks_exact(w)) {
        for (d, &v) in line[1..=w].iter_mut().zip(row) {
            *d = (v > 0) as i8;
        }
    }
    let delta = |s: usize| DELTAS[s % 8].0 + DELTAS[s % 8].1 * pw as isize;
    let mut found: Vec<Vec<(i32, i32)>> = Vec::new();

    let (width, height) = (pw - 1, ph - 1);
    let mut last_pos = (0usize, 1usize);
    for y in 1..height {
        let mut prev: i8 = 0;
        let mut x = 1;
        while x < width {
            let line = &img[y * pw..y * pw + width];
            match first_other(&line[x..], prev) {
                Some(skip) => x += skip,
                None => break,
            }
            let p = img[y * pw + x];
            // contourScan for RETR_EXTERNAL: start only at an outer border whose left
            // neighbour is not inside an already traced contour.
            let mut start = false;
            if prev == 0 && p == 1 {
                start = img[last_pos.1 * pw + last_pos.0] <= 0;
            } else if p == 0 && prev >= 1 && prev & FLAGS != 0 {
                last_pos.0 = x - 1;
            }
            if start {
                last_pos.0 = x;
                found.push(trace(
                    &mut img,
                    y * pw + x,
                    (x as i32 - 1, y as i32 - 1),
                    4,
                    &delta,
                ));
            } else {
                prev = p;
                if prev & FLAGS != 0 {
                    last_pos.0 = x;
                }
            }
            x += 1;
            if start {
                prev = img[y * pw + x - 1];
            }
        }
        last_pos = (0, y + 1);
    }
    found.reverse();
    found
}

/// `cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)`: every outer and hole border
/// of the nonzero pixels, in OpenCV's output order.
pub fn find_list_contours(src: &[u8], h: usize, w: usize) -> Vec<Vec<(i32, i32)>> {
    let (pw, ph) = (w + 2, h + 2);
    let mut img = vec![0i8; pw * ph];
    for (line, row) in img.chunks_exact_mut(pw).skip(1).zip(src.chunks_exact(w)) {
        for (d, &v) in line[1..=w].iter_mut().zip(row) {
            *d = (v > 0) as i8;
        }
    }
    let delta = |s: usize| DELTAS[s % 8].0 + DELTAS[s % 8].1 * pw as isize;
    let mut found: Vec<Vec<(i32, i32)>> = Vec::new();
    let (width, height) = (pw - 1, ph - 1);
    for y in 1..height {
        let mut prev: i8 = 0;
        let mut x = 1;
        while x < width {
            match first_other(&img[y * pw + x..y * pw + width], prev) {
                Some(skip) => x += skip,
                None => break,
            }
            let p = img[y * pw + x];
            // contourScan: an outer border starts at 0 -> 1, a hole at an inside pixel -> 0.
            let hole = if prev == 0 && p == 1 {
                Some(false)
            } else if p == 0 && prev >= 1 {
                Some(true)
            } else {
                None
            };
            match hole {
                Some(hole) => {
                    let sx = x - hole as usize;
                    let s_end = if hole { 0 } else { 4 };
                    found.push(trace(
                        &mut img,
                        y * pw + sx,
                        (sx as i32 - 1, y as i32 - 1),
                        s_end,
                        &delta,
                    ));
                    x += 1;
                    prev = img[y * pw + x - 1];
                }
                None => {
                    prev = p;
                    x += 1;
                }
            }
        }
    }
    found.reverse();
    found
}

/// `cv2.approxPolyDP(curve, epsilon, closed)` of integer points, OpenCV 4.13's
/// Douglas-Peucker with its segment-distance test and final straight-line clean-up.
pub fn approx_poly_dp(src: &[(i32, i32)], eps: f64, closed: bool) -> Vec<(i32, i32)> {
    let mut count = src.len();
    if count == 0 {
        return Vec::new();
    }
    let eps = eps * eps;
    let mut dst: Vec<(i32, i32)> = Vec::with_capacity(count);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut is_closed = closed;
    let mut init_iters = 3;
    let mut right = (0usize, 0usize);
    let mut start_pt = (-1_000_000i32, -1_000_000i32);
    let mut pos = 0usize;
    let mut le_eps = false;
    if !is_closed {
        right.0 = count;
        let (end_pt, first) = (src[0], src[count - 1]);
        start_pt = first;
        if start_pt != end_pt {
            stack.push((0, count - 1));
        } else {
            is_closed = true;
            init_iters = 1;
        }
    }
    if is_closed {
        right.0 = 0;
        for _ in 0..init_iters {
            let mut max_dist = 0f64;
            pos = (pos + right.0) % count;
            start_pt = src[pos];
            pos = (pos + 1) % count;
            for j in 1..count {
                let pt = src[pos];
                pos = (pos + 1) % count;
                let (dx, dy) = ((pt.0 - start_pt.0) as f64, (pt.1 - start_pt.1) as f64);
                let dist = dx * dx + dy * dy;
                if dist > max_dist {
                    max_dist = dist;
                    right.0 = j;
                }
            }
            le_eps = max_dist <= eps;
        }
        if !le_eps {
            let start = pos % count;
            right.1 = start;
            right.0 = (right.0 + start) % count;
            stack.push(right);
            stack.push((start, right.0));
        } else {
            dst.push(start_pt);
        }
    }
    while let Some(slice) = stack.pop() {
        let end_pt = src[slice.1];
        start_pt = src[slice.0];
        pos = (slice.0 + 1) % count;
        if pos != slice.1 {
            let (dx, dy) = (
                (end_pt.0 - start_pt.0) as f64,
                (end_pt.1 - start_pt.1) as f64,
            );
            let seg = dx * dx + dy * dy;
            let mut max_d = 0f64;
            while pos != slice.1 {
                let pt = src[pos];
                pos = (pos + 1) % count;
                let (px, py) = (pt.0 - start_pt.0, pt.1 - start_pt.1);
                let projection = px as f64 * dx + py as f64 * dy;
                let d = if projection < 0.0 {
                    (px * px + py * py) as f64 * seg
                } else if projection > seg {
                    let (ex, ey) = (pt.0 - end_pt.0, pt.1 - end_pt.1);
                    (ex * ex + ey * ey) as f64 * seg
                } else {
                    let dist = py as f64 * dx - px as f64 * dy;
                    dist * dist
                };
                if d > max_d {
                    max_d = d;
                    right.0 = (pos + count - 1) % count;
                }
            }
            le_eps = max_d <= eps * seg;
        } else {
            le_eps = true;
        }
        if le_eps {
            dst.push(start_pt);
        } else {
            stack.push((right.0, slice.1));
            stack.push((slice.0, right.0));
        }
    }
    if !is_closed {
        dst.push(src[count - 1]);
    }
    // Clean-up: drop points on nearly straight runs.
    let is_closed = closed;
    count = dst.len();
    let mut new_count = count;
    let mut pos = if is_closed { count - 1 } else { 0 };
    let read = |pos: &mut usize, d: &[(i32, i32)]| {
        let p = d[*pos];
        *pos += 1;
        if *pos >= count {
            *pos = 0;
        }
        p
    };
    let mut start_pt = read(&mut pos, &dst);
    let mut wpos = pos;
    let mut pt = read(&mut pos, &dst);
    let open = !is_closed as usize;
    let mut i = open;
    while i < count - open && new_count > 2 {
        let end_pt = read(&mut pos, &dst);
        let (dx, dy) = (
            (end_pt.0 - start_pt.0) as f64,
            (end_pt.1 - start_pt.1) as f64,
        );
        let dist = ((pt.0 - start_pt.0) as f64 * dy - (pt.1 - start_pt.1) as f64 * dx).abs();
        let inner = ((pt.0 - start_pt.0) * (end_pt.0 - pt.0)
            + (pt.1 - start_pt.1) * (end_pt.1 - pt.1)) as f64;
        if dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0.0 && dy != 0.0 && inner >= 0.0
        {
            new_count -= 1;
            start_pt = end_pt;
            dst[wpos] = end_pt;
            wpos += 1;
            if wpos >= count {
                wpos = 0;
            }
            pt = read(&mut pos, &dst);
            i += 2;
            continue;
        }
        start_pt = pt;
        dst[wpos] = pt;
        wpos += 1;
        if wpos >= count {
            wpos = 0;
        }
        pt = end_pt;
        i += 1;
    }
    if !is_closed {
        dst[wpos] = pt;
    }
    dst.truncate(new_count);
    dst
}

/// The index of the first byte that is not `v`, eight bytes at a time over the long runs.
fn first_other(line: &[i8], v: i8) -> Option<usize> {
    let run = u64::from_ne_bytes([v as u8; 8]);
    let (words, _) = line.as_chunks::<8>();
    let mut i = 0;
    for word in words {
        if u64::from_ne_bytes(word.map(|b| b as u8)) != run {
            break;
        }
        i += 8;
    }
    line[i..].iter().position(|&b| b != v).map(|p| i + p)
}

/// `icvFetchContourEx<schar>` for a border from `start` (`s_end_outer` 4 for an outer border, 0 for a hole) (an index into the padded image),
/// with CHAIN_APPROX_SIMPLE: a point wherever the chain direction changes.
fn trace(
    img: &mut [i8],
    start: usize,
    origin: (i32, i32),
    s_end_outer: usize,
    delta: &dyn Fn(usize) -> isize,
) -> Vec<(i32, i32)> {
    let at = |i: usize, d: isize| (i as isize + d) as usize;
    let mut pts = Vec::new();
    let mut pt = origin;
    let i0 = start;
    let mut s = s_end_outer;
    let mut i1;
    loop {
        s = (s + 7) & 7;
        i1 = at(i0, delta(s));
        if img[i1] != 0 || s == s_end_outer {
            break;
        }
    }
    if s == s_end_outer {
        img[i0] = NEW | RIGHT;
        pts.push(pt);
        return pts;
    }
    let mut i3 = i0;
    let mut prev_s = s ^ 4;
    loop {
        let s_end = s;
        let mut i4 = i3;
        s = s.min(15);
        while s < 15 {
            s += 1;
            i4 = at(i3, delta(s));
            if img[i4] != 0 {
                break;
            }
        }
        s &= 7;
        if s.wrapping_sub(1) < s_end {
            img[i3] = NEW | RIGHT;
        } else if img[i3] == 1 {
            img[i3] = NEW;
        }
        if s != prev_s {
            pts.push(pt);
        }
        prev_s = s;
        pt = (pt.0 + DELTAS[s].0 as i32, pt.1 + DELTAS[s].1 as i32);
        if i4 == i0 && i3 == i1 {
            break;
        }
        i3 = i4;
        s = (s + 4) & 7;
    }
    pts
}

/// `cv2.contourArea` of integer points: the shoelace sum in double, unsigned.
pub fn contour_area(pts: &[(i32, i32)]) -> f64 {
    let Some(&last) = pts.last() else { return 0.0 };
    let mut prev = (last.0 as f32, last.1 as f32);
    let mut a = 0f64;
    for &(x, y) in pts {
        let p = (x as f32, y as f32);
        a += prev.0 as f64 * p.1 as f64 - prev.1 as f64 * p.0 as f64;
        prev = p;
    }
    (a * 0.5).abs()
}

/// `cv2.boundingRect` of integer points: (x, y, width, height).
pub fn bounding_rect(pts: &[(i32, i32)]) -> (i32, i32, i32, i32) {
    if pts.is_empty() {
        return (0, 0, 0, 0);
    }
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for &(x, y) in pts {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_gives_its_four_corners() {
        let (h, w) = (6, 7);
        let mut img = vec![0u8; h * w];
        for y in 1..4 {
            for x in 2..6 {
                img[y * w + x] = 255;
            }
        }
        let c = find_external_contours(&img, h, w);
        assert_eq!(c, vec![vec![(2, 1), (2, 3), (5, 3), (5, 1)]]);
    }

    #[test]
    fn single_pixel_and_nested_blob() {
        let (h, w) = (9, 9);
        let mut img = vec![0u8; h * w];
        img[w + 1] = 1;
        for y in 3..8 {
            for x in 3..8 {
                img[y * w + x] = 1;
            }
        }
        img[5 * w + 5] = 0; // a hole: not an external contour
        let c = find_external_contours(&img, h, w);
        assert_eq!(c.len(), 2);
        assert_eq!(c[1], vec![(1, 1)]);
    }
}
