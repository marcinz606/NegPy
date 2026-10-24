//! `cv2.detail_DpSeamFinder("COLOR").find(...)` of float32 RGB parts that share one canvas
//! (corners all at the origin), after OpenCV 4.13 `modules/stitching/src/seam_finders.cpp`
//! (Apache-2.0).

use std::collections::BTreeSet;

use rayon::prelude::*;

const FIRST: u8 = 1;
const SECOND: u8 = 2;
const INTERS: u8 = 4;

/// (x, y) as OpenCV's `Point`.
type Pt = (i32, i32);

/// The state of one pair: labels over the canvas and, per component, its state, bounding box
/// and contour.
struct Pair<'a> {
    w: i32,
    h: i32,
    img1: &'a [f32],
    img2: &'a [f32],
    labels: Vec<i32>,
    contour1: Vec<bool>,
    contour2: Vec<bool>,
    states: Vec<u8>,
    tls: Vec<Pt>,
    brs: Vec<Pt>,
    contours: Vec<Vec<Pt>>,
    edges: BTreeSet<(i32, i32)>,
}

/// `floodFill(img, seed, new_val)` with 4-connectivity and zero tolerance, a row span at a time.
fn flood_fill(img: &mut [i32], (w, h): (i32, i32), seed: Pt, new_val: i32) {
    let old = img[(seed.1 * w + seed.0) as usize];
    if old == new_val {
        return;
    }
    let mut stack = vec![seed];
    while let Some((x, y)) = stack.pop() {
        let row = &mut img[(y * w) as usize..((y + 1) * w) as usize];
        if row[x as usize] != old {
            continue;
        }
        let (mut x0, mut x1) = (x as usize, x as usize);
        while x0 > 0 && row[x0 - 1] == old {
            x0 -= 1;
        }
        while x1 + 1 < w as usize && row[x1 + 1] == old {
            x1 += 1;
        }
        row[x0..=x1].fill(new_val);
        for ny in [y - 1, y + 1] {
            if ny < 0 || ny >= h {
                continue;
            }
            let next = &img[(ny * w) as usize..((ny + 1) * w) as usize];
            let mut inside = false;
            for (nx, &v) in next.iter().enumerate().take(x1 + 1).skip(x0) {
                if v == old && !inside {
                    stack.push((nx as i32, ny));
                }
                inside = v == old;
            }
        }
    }
}

/// The contour of a mask: set pixels with a 4-neighbour unset or off the canvas.
fn contour(mask: &[u8], w: i32, h: i32) -> Vec<bool> {
    let at = |x: i32, y: i32| mask[(y * w + x) as usize] != 0;
    let mut out = vec![false; (w * h) as usize];
    out.par_chunks_mut(w as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i32;
            for (x, o) in (0..w).zip(row.iter_mut()) {
                *o = at(x, y)
                    && (x == 0
                        || !at(x - 1, y)
                        || x == w - 1
                        || !at(x + 1, y)
                        || y == 0
                        || !at(x, y - 1)
                        || y == h - 1
                        || !at(x, y + 1));
            }
        });
    out
}

/// The stitching module's `normL2(Point3f(255, 255, 255), Point3f(0, 0, 0))`, which is the
/// squared norm: the cost of a step outside the component.
fn bad_region_cost() -> f32 {
    3.0 * 255.0 * 255.0
}

fn diff(a: &[f32], pa: usize, b: &[f32], pb: usize) -> f32 {
    let d = |c: usize| {
        let v = a[3 * pa + c] - b[3 * pb + c];
        v * v
    };
    d(0) + d(1) + d(2)
}

/// OpenCV's `cv::partition` with `ClosePoints(min_dist)`: classes numbered in the order of their
/// first point.
fn partition(points: &[Pt], min_dist: i32) -> Vec<usize> {
    let n = points.len();
    let mut label = vec![usize::MAX; n];
    let mut next = 0;
    for i in 0..n {
        if label[i] != usize::MAX {
            continue;
        }
        label[i] = next;
        let mut stack = vec![i];
        while let Some(a) = stack.pop() {
            for b in 0..n {
                let (dx, dy) = (points[a].0 - points[b].0, points[a].1 - points[b].1);
                if label[b] == usize::MAX && dx * dx + dy * dy < min_dist * min_dist {
                    label[b] = next;
                    stack.push(b);
                }
            }
        }
        next += 1;
    }
    label
}

/// `min_element` of OpenCV's (cost, control) steps that apply: the cheapest, then the lowest
/// control.
fn best_step(steps: &[(bool, f32, u8)]) -> Option<(f32, u8)> {
    steps
        .iter()
        .filter(|s| s.0)
        .map(|s| (s.1, s.2))
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
}

fn round_div(sum: i32, n: usize) -> f64 {
    (sum as f64 / n as f64).round_ties_even()
}

impl<'a> Pair<'a> {
    fn lab(&self, x: i32, y: i32) -> i32 {
        self.labels[(y * self.w + x) as usize]
    }

    fn on_contour(&self, x: i32, y: i32, l: i32) -> bool {
        let (w, h) = (self.w, self.h);
        x == 0
            || self.lab(x - 1, y) != l
            || x == w - 1
            || self.lab(x + 1, y) != l
            || y == 0
            || self.lab(x, y - 1) != l
            || y == h - 1
            || self.lab(x, y + 1) != l
    }

    fn touches(&self, x: i32, y: i32, l: i32) -> bool {
        (x > 0 && self.lab(x - 1, y) == l)
            || (y > 0 && self.lab(x, y - 1) == l)
            || (x < self.w - 1 && self.lab(x + 1, y) == l)
            || (y < self.h - 1 && self.lab(x, y + 1) == l)
    }

    fn find_components(&mut self, m1: &[u8], m2: &[u8]) {
        let (w, h) = (self.w, self.h);
        self.labels = m1
            .iter()
            .zip(m2)
            .map(|(&a, &b)| match (a != 0, b != 0) {
                (true, true) => i32::MAX,
                (true, false) => i32::MAX - 1,
                (false, true) => i32::MAX - 2,
                _ => 0,
            })
            .collect();
        let mut ncomps = 0;
        for y in 0..h {
            for x in 0..w {
                let v = self.lab(x, y);
                if v >= i32::MAX - 2 {
                    self.states.push(match v {
                        i32::MAX => INTERS,
                        v if v == i32::MAX - 1 => FIRST,
                        _ => SECOND,
                    });
                    ncomps += 1;
                    flood_fill(&mut self.labels, (w, h), (x, y), ncomps);
                }
            }
        }
        let n = ncomps as usize;
        (self.tls, self.brs) = (vec![(i32::MAX, i32::MAX); n], vec![(i32::MIN, i32::MIN); n]);
        self.contours = vec![Vec::new(); n];
        for (l, x, y) in self.contour_points(0..h, 0..w, None) {
            self.add_contour_point((l - 1) as usize, x, y);
        }
    }

    /// The contour pixels of every component (or of label `only`) in the box, in raster order;
    /// a component's box is the box of its contour.
    fn contour_points(
        &self,
        ys: std::ops::Range<i32>,
        xs: std::ops::Range<i32>,
        only: Option<i32>,
    ) -> Vec<(i32, i32, i32)> {
        let rows: Vec<Vec<(i32, i32, i32)>> = ys
            .into_par_iter()
            .map(|y| {
                xs.clone()
                    .filter_map(|x| {
                        let l = self.lab(x, y);
                        let wanted = match only {
                            Some(o) => l == o,
                            None => l != 0,
                        };
                        (wanted && self.on_contour(x, y, l)).then_some((l, x, y))
                    })
                    .collect()
            })
            .collect();
        rows.concat()
    }

    fn add_contour_point(&mut self, ci: usize, x: i32, y: i32) {
        self.tls[ci] = (self.tls[ci].0.min(x), self.tls[ci].1.min(y));
        self.brs[ci] = (self.brs[ci].0.max(x + 1), self.brs[ci].1.max(y + 1));
        self.contours[ci].push((x, y));
    }

    fn find_edges(&mut self) {
        for ci in 0..self.contours.len() {
            let l = ci as i32 + 1;
            for &(x, y) in &self.contours[ci] {
                let mut near = [0; 4];
                if x > 0 {
                    near[0] = self.lab(x - 1, y);
                }
                if y > 0 {
                    near[1] = self.lab(x, y - 1);
                }
                if x < self.w - 1 {
                    near[2] = self.lab(x + 1, y);
                }
                if y < self.h - 1 {
                    near[3] = self.lab(x, y + 1);
                }
                for n in near {
                    if n != 0 && n != l {
                        self.edges.insert((ci as i32, n - 1));
                        self.edges.insert((n - 1, ci as i32));
                    }
                }
            }
        }
    }

    fn has_only_one_neighbor(&self, comp: i32) -> bool {
        self.edges
            .range((comp, i32::MIN)..=(comp, i32::MAX))
            .count()
            == 1
    }

    fn close_to_contour(&self, x: i32, y: i32, mask: &[bool]) -> bool {
        (-2..=2).any(|dy| {
            let yy = y + dy;
            yy >= 0
                && yy < self.h
                && (-2..=2).any(|dx| {
                    let xx = x + dx;
                    xx >= 0 && xx < self.w && mask[(yy * self.w + xx) as usize]
                })
        })
    }

    fn seam_tips(&self, comp1: usize, comp2: usize) -> Option<(Pt, Pt)> {
        let l2 = comp2 as i32 + 1;
        let special: Vec<Pt> = self.contours[comp1]
            .iter()
            .copied()
            .filter(|&(x, y)| {
                self.close_to_contour(x, y, &self.contour1)
                    && self.close_to_contour(x, y, &self.contour2)
                    && self.touches(x, y, l2)
            })
            .collect();
        if special.len() < 2 {
            return None;
        }
        let labels = partition(&special, 10);
        let n = labels.iter().max()? + 1;
        if n < 2 {
            return None;
        }
        let mut sum = vec![(0i32, 0i32); n];
        let mut points: Vec<Vec<Pt>> = vec![Vec::new(); n];
        for (&p, &l) in special.iter().zip(&labels) {
            sum[l] = (sum[l].0 + p.0, sum[l].1 + p.1);
            points[l].push(p);
        }
        let centre = |i: usize| {
            (
                round_div(sum[i].0, points[i].len()),
                round_div(sum[i].1, points[i].len()),
            )
        };
        let (mut idx, mut max_dist) = ([0, 0], f64::MIN);
        for i in 0..n - 1 {
            for j in i + 1..n {
                let ((x1, y1), (x2, y2)) = (centre(i), centre(j));
                let d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                if d > max_dist {
                    max_dist = d;
                    idx = [i, j];
                }
            }
        }
        let tip = |i: usize| {
            let (cx, cy) = centre(i);
            let mut best = (f64::MAX, points[i][0]);
            for &p in &points[i] {
                let d =
                    (p.0 as f64 - cx) * (p.0 as f64 - cx) + (p.1 as f64 - cy) * (p.1 as f64 - cy);
                if d < best.0 {
                    best = (d, p);
                }
            }
            best.1
        };
        Some((tip(idx[0]), tip(idx[1])))
    }

    /// The vertical and horizontal step costs over the component's box: the mean squared color
    /// difference of the two parts across the step, or the bad-region cost.
    fn costs(&self, comp: usize) -> (Vec<f32>, Vec<f32>) {
        let l = comp as i32 + 1;
        let ((x0, y0), (x1, y1)) = (self.tls[comp], self.brs[comp]);
        let bad = bad_region_cost();
        let w = self.w;
        let px = |x: i32, y: i32| (y * w + x) as usize;
        let (img1, img2) = (self.img1, self.img2);
        let step = |x: i32, y: i32, (xb, yb): Pt| {
            (diff(img1, px(xb, yb), img2, px(x, y)) + diff(img1, px(x, y), img2, px(xb, yb))) / 2.0
        };
        let cost_v: Vec<f32> = (y0..y1)
            .into_par_iter()
            .flat_map_iter(|y| {
                (x0..x1 + 1).map(move |x| {
                    if x > 0 && x < w && self.lab(x, y) == l && self.lab(x - 1, y) == l {
                        step(x, y, (x - 1, y))
                    } else {
                        bad
                    }
                })
            })
            .collect();
        let cost_h: Vec<f32> = (y0..y1 + 1)
            .into_par_iter()
            .flat_map_iter(|y| {
                (x0..x1).map(move |x| {
                    if y > 0 && y < self.h && self.lab(x, y) == l && self.lab(x, y - 1) == l {
                        step(x, y, (x, y - 1))
                    } else {
                        bad
                    }
                })
            })
            .collect();
        (cost_v, cost_h)
    }

    /// The minimum-cost path between the tips through the component, or None when the far tip
    /// cannot be reached; and whether it runs horizontally.
    fn estimate_seam(&self, comp: usize, p1: Pt, p2: Pt) -> Option<(Vec<Pt>, bool)> {
        let (cost_v, cost_h) = self.costs(comp);
        let l = comp as i32 + 1;
        let (tl, br) = (self.tls[comp], self.brs[comp]);
        let (rw, rh) = (br.0 - tl.0, br.1 - tl.1);
        let (mut src, mut dst) = ((p1.0 - tl.0, p1.1 - tl.1), (p2.0 - tl.0, p2.1 - tl.1));
        let horizontal = (dst.0 - src.0).abs() > (dst.1 - src.1).abs();
        let swapped = if horizontal {
            src.0 > dst.0
        } else {
            src.1 > dst.1
        };
        if swapped {
            std::mem::swap(&mut src, &mut dst);
        }
        let n = (rw * rh) as usize;
        let at = |x: i32, y: i32| (y * rw + x) as usize;
        let cv = |x: i32, y: i32| cost_v[(y * (rw + 1) + x) as usize];
        let ch = |x: i32, y: i32| cost_h[(y * rw + x) as usize];
        let mut control = vec![0u8; n];
        let mut reachable = vec![false; n];
        let mut cost = vec![0f32; n];
        reachable[at(src.0, src.1)] = true;
        if horizontal {
            for x in src.0 + 1..=dst.0 {
                for y in 0..rh {
                    if self.lab(x + tl.0, y + tl.1) != l {
                        continue;
                    }
                    let (r, c) = (&reachable, &cost);
                    let steps = [
                        (r[at(x - 1, y)], c[at(x - 1, y)] + ch(x - 1, y), 1),
                        (
                            y > 0 && r[at(x - 1, (y - 1).max(0))],
                            c[at(x - 1, (y - 1).max(0))]
                                + ch(x - 1, (y - 1).max(0))
                                + cv(x, (y - 1).max(0)),
                            2,
                        ),
                        (
                            y < rh - 1 && r[at(x - 1, (y + 1).min(rh - 1))],
                            c[at(x - 1, (y + 1).min(rh - 1))]
                                + ch(x - 1, (y + 1).min(rh - 1))
                                + cv(x, y),
                            3,
                        ),
                    ];
                    if let Some((c, k)) = best_step(&steps) {
                        cost[at(x, y)] = c;
                        control[at(x, y)] = k;
                        reachable[at(x, y)] = true;
                    }
                }
            }
        } else {
            for y in src.1 + 1..=dst.1 {
                for x in 0..rw {
                    if self.lab(x + tl.0, y + tl.1) != l {
                        continue;
                    }
                    let (r, c) = (&reachable, &cost);
                    let steps = [
                        (r[at(x, y - 1)], c[at(x, y - 1)] + cv(x, y - 1), 1),
                        (
                            x > 0 && r[at((x - 1).max(0), y - 1)],
                            c[at((x - 1).max(0), y - 1)]
                                + cv((x - 1).max(0), y - 1)
                                + ch((x - 1).max(0), y),
                            2,
                        ),
                        (
                            x < rw - 1 && r[at((x + 1).min(rw - 1), y - 1)],
                            c[at((x + 1).min(rw - 1), y - 1)]
                                + cv((x + 1).min(rw - 1), y - 1)
                                + ch(x, y),
                            3,
                        ),
                    ];
                    if let Some((c, k)) = best_step(&steps) {
                        cost[at(x, y)] = c;
                        control[at(x, y)] = k;
                        reachable[at(x, y)] = true;
                    }
                }
            }
        }
        if !reachable[at(dst.0, dst.1)] {
            return None;
        }
        let mut p = dst;
        let mut seam = vec![(p.0 + tl.0, p.1 + tl.1)];
        while if horizontal {
            p.0 != src.0
        } else {
            p.1 != src.1
        } {
            let k = control[at(p.0, p.1)];
            if horizontal {
                p.1 += (k == 3) as i32 - (k == 2) as i32;
                p.0 -= 1;
            } else {
                p.0 += (k == 3) as i32 - (k == 2) as i32;
                p.1 -= 1;
            }
            seam.push((p.0 + tl.0, p.1 + tl.1));
        }
        if !swapped {
            seam.reverse();
        }
        Some((seam, horizontal))
    }

    fn update_labels_using_seam(
        &mut self,
        comp1: usize,
        comp2: usize,
        seam: &[Pt],
        horizontal: bool,
    ) {
        let (tl, br) = (self.tls[comp1], self.brs[comp1]);
        let (mw, mh) = (br.0 - tl.0, br.1 - tl.1);
        let mut mask = vec![0i32; (mw * mh) as usize];
        let at = |x: i32, y: i32| (y * mw + x) as usize;
        for &(x, y) in self.contours[comp1].iter().chain(seam) {
            mask[at(x - tl.0, y - tl.1)] = 255;
        }
        let (l1, l2) = (comp1 as i32 + 1, comp2 as i32 + 1);
        let mut ncomps = 0;
        for y in 0..mh {
            for x in 0..mw {
                if mask[at(x, y)] == 0 && self.lab(x + tl.0, y + tl.1) == l1 {
                    ncomps += 1;
                    flood_fill(&mut mask, (mw, mh), (x, y), ncomps);
                }
            }
        }
        const DX: [i32; 8] = [-1, 1, 0, 0, -1, 1, -1, 1];
        const DY: [i32; 8] = [0, 0, -1, 1, -1, -1, 1, 1];
        for &(cx, cy) in &self.contours[comp1] {
            let (x, y) = (cx - tl.0, cy - tl.1);
            let mut ok = false;
            for j in 0..8 {
                let (c, r) = (x + DX[j], y + DY[j]);
                if c >= 0
                    && c < mw
                    && r >= 0
                    && r < mh
                    && mask[at(c, r)] != 0
                    && mask[at(c, r)] != 255
                {
                    ok = true;
                    mask[at(x, y)] = mask[at(c, r)];
                }
            }
            if !ok {
                mask[at(x, y)] = 0;
            }
        }
        for &(sx, sy) in seam {
            let (x, y) = (sx - tl.0, sy - tl.1);
            let (nx, ny) = if horizontal { (x, y + 1) } else { (x + 1, y) };
            mask[at(x, y)] =
                if nx < mw && ny < mh && mask[at(nx, ny)] != 0 && mask[at(nx, ny)] != 255 {
                    mask[at(nx, ny)]
                } else {
                    0
                };
        }
        let mut connect2 = vec![0usize; ncomps as usize + 1];
        let mut connect_other = vec![0usize; ncomps as usize + 1];
        let (w, h) = (self.w, self.h);
        let other = |s: &Self, x: i32, y: i32| {
            let v = s.lab(x, y);
            v != l1 && v != l2
        };
        for &(x, y) in &self.contours[comp1] {
            // A mask value of 255 or 0 here counts nowhere, as in OpenCV's maps.
            let m = mask[at(x - tl.0, y - tl.1)];
            let slot = (1..=ncomps).contains(&m).then_some(m as usize);
            if self.touches(x, y, l2)
                && let Some(s) = slot
            {
                connect2[s] += 1;
            }
            let near_other = (x > 0 && other(self, x - 1, y))
                || (y > 0 && other(self, x, y - 1))
                || (x < w - 1 && other(self, x + 1, y))
                || (y < h - 1 && other(self, x, y + 1));
            if near_other && let Some(s) = slot {
                connect_other[s] += 1;
            }
        }
        let len = self.contours[comp1].len() as f64;
        let adjacent: Vec<bool> = (0..=ncomps as usize)
            .map(|i| {
                i > 0 && connect2[i] as f64 / len > 0.05 && (connect_other[i] as f64 / len) < 0.1
            })
            .collect();
        for y in 0..mh {
            for x in 0..mw {
                let m = mask[at(x, y)];
                if m > 0 && m <= ncomps && adjacent[m as usize] {
                    self.labels[((y + tl.1) * w + x + tl.0) as usize] = l2;
                }
            }
        }
    }

    fn refresh(&mut self, comp: usize) {
        let l = comp as i32 + 1;
        let ((x0, y0), (x1, y1)) = (self.tls[comp], self.brs[comp]);
        self.tls[comp] = (i32::MAX, i32::MAX);
        self.brs[comp] = (i32::MIN, i32::MIN);
        self.contours[comp].clear();
        for (_, x, y) in self.contour_points(y0..y1, x0..x1, Some(l)) {
            self.add_contour_point(comp, x, y);
        }
    }

    fn resolve_conflicts(&mut self) {
        loop {
            let conflict = self.edges.iter().copied().find(|&(c1, c2)| {
                let s1 = self.states[c1 as usize];
                s1 & INTERS != 0 && s1 & !INTERS != self.states[c2 as usize]
            });
            let Some((c1, c2)) = conflict else { break };
            let (u1, u2) = (c1 as usize, c2 as usize);
            if self.has_only_one_neighbor(c1) {
                let ((x0, y0), (x1, y1)) = (self.tls[u1], self.brs[u1]);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let p = (y * self.w + x) as usize;
                        if self.labels[p] == c1 + 1 {
                            self.labels[p] = c2 + 1;
                        }
                    }
                }
                self.states[u1] = if self.states[u2] == FIRST {
                    SECOND
                } else {
                    FIRST
                };
            } else {
                if let Some((p1, p2)) = self.seam_tips(u1, u2)
                    && let Some((seam, horizontal)) = self.estimate_seam(u1, p1, p2)
                {
                    self.update_labels_using_seam(u1, u2, &seam, horizontal);
                }
                self.states[u1] = if self.states[u2] == FIRST {
                    INTERS | SECOND
                } else {
                    INTERS | FIRST
                };
            }
            self.refresh(u1);
            self.refresh(u2);
            self.edges.remove(&(c1, c2));
            self.edges.remove(&(c2, c1));
        }
    }
}

/// One pair of `DpSeamFinder::process`: cut the overlap of `m1` and `m2` between the parts.
fn process(img1: &[f32], img2: &[f32], m1: &mut [u8], m2: &mut [u8], (h, w): (usize, usize)) {
    let (w, h) = (w as i32, h as i32);
    let mut pair = Pair {
        w,
        h,
        img1,
        img2,
        labels: Vec::new(),
        contour1: contour(m1, w, h),
        contour2: contour(m2, w, h),
        states: Vec::new(),
        tls: Vec::new(),
        brs: Vec::new(),
        contours: Vec::new(),
        edges: BTreeSet::new(),
    };
    pair.find_components(m1, m2);
    pair.find_edges();
    pair.resolve_conflicts();
    let won = |l: i32, side: u8| l > 0 && pair.states[(l - 1) as usize] & side != 0;
    m2.par_iter_mut()
        .zip(m1.par_iter())
        .zip(pair.labels.par_iter())
        .for_each(|((b, &a), &l)| {
            if won(l, FIRST) && a != 0 {
                *b = 0;
            }
        });
    m1.par_iter_mut()
        .zip(m2.par_iter())
        .zip(pair.labels.par_iter())
        .for_each(|((a, &b), &l)| {
            if won(l, SECOND) && b != 0 {
                *a = 0;
            }
        });
}

/// `DpSeamFinder("COLOR").find(images, [(0, 0)] * n, masks)` of HxWx3 float32 parts on one
/// canvas: each pair cuts its overlap in place, the pairs in reverse order of OpenCV's sort,
/// which leaves pairs of equal centre distance in order for up to 16 pairs.
pub fn dp_seams(images: &[&[f32]], masks: &mut [Vec<u8>], (h, w): (usize, usize)) {
    let n = images.len();
    let mut pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();
    pairs.reverse();
    for (i, j) in pairs {
        let (lo, hi) = masks.split_at_mut(j);
        process(images[i], images[j], &mut lo[i], &mut hi[0], (h, w));
    }
}
