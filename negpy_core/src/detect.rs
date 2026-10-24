//! `detect_film_bounds_with_confidence`: the outer film box by agreement across adaptive
//! thresholds, corroborated by structural masks (geometry/logic.py).

use rayon::prelude::*;

use crate::analysis::{Snap, crop, detection_plane, rgb, snap_edge_to_gradient};
use crate::contours::{contour_area, find_external_contours};
use crate::filter::Border;
use crate::gray;
use crate::morph::{self, Op};
use crate::shapes::{bounding_rect_f32, box_points, fill_convex_poly, min_area_rect};
use crate::stats::{np_median, np_percentiles, np_percentiles_f64, pairwise_sum_f32};

pub type Roi = (i64, i64, i64, i64);

const FINE_ROTATION_LIMIT: f64 = 45.0;
const ADAPTIVE_THRESHOLD_COUNT: f64 = 7.0;
// Film fills most of a scan under every known holder; on a positive whose bed clips to white, a
// dark subject can pass every other check on its own geometry.
const MIN_FILM_BOX_COVERAGE: f64 = 0.25;
pub const SIDE_NAMES: [&str; 4] = ["top", "right", "bottom", "left"];
pub const CORNER_SIDES: [(&str, [usize; 2]); 4] = [
    ("top_left", [0, 3]),
    ("top_right", [0, 1]),
    ("bottom_right", [2, 1]),
    ("bottom_left", [2, 3]),
];
pub const EVIDENCE_SOURCE_ORDER: [&str; 5] = [
    "adaptive-dark",
    "adaptive-bright",
    "inverse-threshold",
    "blackhat",
    "edges",
];

fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    v.clamp(lo, hi)
}

/// `_odd_kernel_size(value, lower, upper)`.
fn odd_kernel_size(value: f64, lower: i64, upper: i64) -> i64 {
    let mut size = py_round(value).clamp(lower, upper);
    if size % 2 == 0 {
        size = if size < upper { size + 1 } else { size - 1 };
    }
    size.max(1)
}

/// `_vertical_edge_profile(lum)`: per column, the normalized mean |Sobel x|, and its contrast.
fn vertical_edge_profile(lum: &[f32], h: usize, w: usize) -> (Vec<f32>, f64) {
    if lum.is_empty() {
        return (Vec::new(), 0.0);
    }
    let mut grad = vec![0f32; h * w];
    if crate::filter::sep_filter2d(
        lum,
        &mut grad,
        (h, w, 1),
        (&[-1.0, 0.0, 1.0], &[1.0, 2.0, 1.0]),
        Border::Reflect101,
    )
    .is_none()
    {
        return (vec![0.0; w], 0.0);
    }
    // np.mean(axis=0): summed down each column.
    let mut profile = vec![0f32; w];
    for r in grad.chunks_exact(w) {
        for (p, &g) in profile.iter_mut().zip(r) {
            *p += g.abs();
        }
    }
    profile.iter_mut().for_each(|p| *p /= h as f32);
    let window = odd_kernel_size(0.006 * w as f64, 3, 15) as usize;
    let mut blurred = vec![0f32; w];
    if crate::filter::gaussian_blur(
        &profile,
        &mut blurred,
        (1, w, 1),
        (window, 1),
        (0.0, 0.0),
        Border::Reflect101,
    )
    .is_none()
    {
        blurred = profile;
    }
    let p = np_percentiles(&blurred, &[50.0, 99.5]).unwrap_or(vec![0.0, 0.0]);
    let (baseline, normalizer) = (p[0] as f64, p[1] as f64);
    let contrast = (normalizer - baseline).max(0.0);
    if normalizer > 1e-8 {
        let n32 = normalizer as f32;
        blurred
            .iter_mut()
            .for_each(|v| *v = (*v / n32).clamp(0.0, 1.0));
    } else {
        blurred.fill(0.0);
    }
    (blurred, contrast)
}

/// `_outer_ring_values(lum)`: the frame's outer ring, top, bottom, left, right.
/// 2% wide: a tightly framed scan leaves 1-3% of bed, so a wider ring reads film and fails the
/// ring gate in `oriented_boundary_evidence` on a good box.
fn outer_ring_values(lum: &[f32], h: usize, w: usize) -> Vec<f32> {
    let m = h.min(w);
    let ring = (py_round(0.02 * m as f64).max(2) as usize).min((m / 2).max(1));
    let mut out = Vec::new();
    out.extend_from_slice(&lum[..ring.min(h) * w]);
    out.extend_from_slice(&lum[h.saturating_sub(ring) * w..]);
    if h > 2 * ring {
        for y in ring..h - ring {
            out.extend_from_slice(&lum[y * w..y * w + ring.min(w)]);
        }
        for y in ring..h - ring {
            out.extend_from_slice(&lum[y * w + w.saturating_sub(ring)..(y + 1) * w]);
        }
    }
    out
}

/// `_threshold_polarities`: which polarities the trials sample.
fn threshold_polarities(
    lum: &[f32],
    h: usize,
    w: usize,
    span: f64,
    ring_median: f64,
) -> Vec<&'static str> {
    let (y1, y2) = (
        py_round(0.2 * h as f64) as usize,
        py_round(0.8 * h as f64) as usize,
    );
    let (x1, x2) = (
        py_round(0.2 * w as f64) as usize,
        py_round(0.8 * w as f64) as usize,
    );
    let (y2, x2) = (y2.max(y1 + 1).min(h), x2.max(x1 + 1).min(w));
    let center_median = np_median(&mut crop(lum, w, (y1.min(h), y2, x1.min(w), x2))) as f64;
    let gate = 0.025f64.max(0.05 * span);
    if ring_median - center_median > gate {
        vec!["dark"]
    } else if center_median - ring_median > gate {
        vec!["bright"]
    } else {
        vec!["dark", "bright"]
    }
}

fn rect_close_open(
    mask: &[u8],
    h: usize,
    w: usize,
    close: (usize, usize, usize),
    open: (usize, usize, usize),
) -> Vec<u8> {
    let mut a = vec![0u8; h * w];
    let mut b = vec![0u8; h * w];
    let _ = morph::morphology(
        mask,
        &mut a,
        (h, w, 1),
        Op::Close,
        (close.0, close.1),
        close.2,
        Border::Constant,
    );
    let _ = morph::morphology(
        &a,
        &mut b,
        (h, w, 1),
        Op::Open,
        (open.0, open.1),
        open.2,
        Border::Constant,
    );
    b
}

/// `_adaptive_threshold_mask(blurred_lum, threshold, polarity)`.
fn adaptive_threshold_mask(
    blurred: &[f32],
    h: usize,
    w: usize,
    threshold: f64,
    dark: bool,
) -> Vec<u8> {
    let t = threshold as f32;
    let mask: Vec<u8> = blurred
        .iter()
        .map(|&v| {
            if (dark && v <= t) || (!dark && v >= t) {
                255
            } else {
                0
            }
        })
        .collect();
    let m = h.min(w) as f64;
    let c = odd_kernel_size(0.010 * m, 3, 21) as usize;
    let o = odd_kernel_size(0.003 * m, 3, 9) as usize;
    rect_close_open(&mask, h, w, (c, c, 1), (o, o, 1))
}

fn mask_from_inverse_threshold(gray_img: &[u8], h: usize, w: usize) -> Vec<u8> {
    let mut blurred = vec![0u8; h * w];
    let _ = gray::gaussian_blur_u8(
        gray_img,
        &mut blurred,
        (h, w, 1),
        (7, 7),
        (0.0, 0.0),
        Border::Reflect101,
    );
    let mut thresh = vec![0u8; h * w];
    let t = gray::otsu_threshold(&blurred);
    gray::threshold_binary(&blurred, &mut thresh, t, 255, true);
    let mut closed = vec![0u8; h * w];
    let _ = morph::morphology(
        &thresh,
        &mut closed,
        (h, w, 1),
        Op::Close,
        (17, 17),
        2,
        Border::Constant,
    );
    let mut out = vec![0u8; h * w];
    let _ = morph::morphology(
        &closed,
        &mut out,
        (h, w, 1),
        Op::Open,
        (5, 5),
        1,
        Border::Constant,
    );
    out
}

fn mask_from_blackhat(gray_img: &[u8], h: usize, w: usize) -> Vec<u8> {
    let mut bh = vec![0u8; h * w];
    let _ = morph::morphology(
        gray_img,
        &mut bh,
        (h, w, 1),
        Op::BlackHat,
        (31, 31),
        1,
        Border::Constant,
    );
    let mut blurred = vec![0u8; h * w];
    let _ = gray::gaussian_blur_u8(
        &bh,
        &mut blurred,
        (h, w, 1),
        (5, 5),
        (0.0, 0.0),
        Border::Reflect101,
    );
    let mut thresh = vec![0u8; h * w];
    let t = gray::otsu_threshold(&blurred);
    gray::threshold_binary(&blurred, &mut thresh, t, 255, false);
    let mut out = vec![0u8; h * w];
    let _ = morph::morphology(
        &thresh,
        &mut out,
        (h, w, 1),
        Op::Close,
        (21, 21),
        2,
        Border::Constant,
    );
    out
}

fn mask_from_edges(gray_img: &[u8], h: usize, w: usize) -> Vec<u8> {
    let mut blurred = vec![0u8; h * w];
    let _ = gray::gaussian_blur_u8(
        gray_img,
        &mut blurred,
        (h, w, 1),
        (5, 5),
        (0.0, 0.0),
        Border::Reflect101,
    );
    let mut edges = vec![0u8; h * w];
    let _ = gray::canny(&blurred, &mut edges, h, w, 40.0, 160.0);
    let mut dilated = vec![0u8; h * w];
    let _ = morph::morphology(
        &edges,
        &mut dilated,
        (h, w, 1),
        Op::Dilate,
        (7, 7),
        2,
        Border::Constant,
    );
    let mut out = vec![0u8; h * w];
    let _ = morph::morphology(
        &dilated,
        &mut out,
        (h, w, 1),
        Op::Close,
        (15, 15),
        2,
        Border::Constant,
    );
    out
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub roi: Roi,
    pub correction_angle: f64,
    pub source: &'static str,
    pub threshold_index: Option<usize>,
    pub polarity: &'static str,
    pub boundary_score: f64,
    pub geometry_score: f64,
    pub sides: [bool; 4],
    pub corners: [bool; 4],
}

/// np.linalg.norm of a float32 2-vector: sqrt((-0 + x*x) + y*y) in float32.
fn norm2(x: f32, y: f32) -> f32 {
    ((-0.0f32 + x * x) + y * y).sqrt()
}

/// `_order_quad`: top-left, top-right, bottom-right, bottom-left.
fn order_quad(q: &[(f32, f32); 4]) -> [(f32, f32); 4] {
    let sums: Vec<f32> = q.iter().map(|p| (-0.0f32 + p.0) + p.1).collect();
    let diffs: Vec<f32> = q.iter().map(|p| p.1 - p.0).collect();
    let argmin = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0, |b, (i, &x)| if x < v[b] { i } else { b })
    };
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
    };
    [
        q[argmin(&sums)],
        q[argmin(&diffs)],
        q[argmax(&sums)],
        q[argmax(&diffs)],
    ]
}

/// `_polygon_values(lum, polygon)`: the finite luma values under a filled convex polygon.
fn polygon_values(lum: &[f32], h: usize, w: usize, poly: &[(f32, f32); 4]) -> Vec<f32> {
    let minx = poly.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let maxx = poly.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
    let miny = poly.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let maxy = poly.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
    let left = (minx as f64).floor().max(0.0) as i64;
    let right = ((maxx as f64).ceil() as i64 + 1).min(w as i64);
    let top = (miny as f64).floor().max(0.0) as i64;
    let bottom = ((maxy as f64).ceil() as i64 + 1).min(h as i64);
    if right <= left || bottom <= top {
        return Vec::new();
    }
    let (mw, mh) = ((right - left) as usize, (bottom - top) as usize);
    let local: Vec<(i64, i64)> = poly
        .iter()
        .map(|p| {
            (
                (p.0 - left as f32).round_ties_even() as i64,
                (p.1 - top as f32).round_ties_even() as i64,
            )
        })
        .collect();
    let mut mask = vec![0u8; mw * mh];
    fill_convex_poly(&mut mask, mw, mh, &local, 1);
    let mut out = Vec::new();
    for y in 0..mh {
        for x in 0..mw {
            if mask[y * mw + x] != 0 {
                let v = lum[(top as usize + y) * w + left as usize + x];
                if v.is_finite() {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// `_oriented_boundary_evidence`: (boundary score, supported sides, supported corners).
fn oriented_boundary_evidence(
    lum: &[f32],
    h: usize,
    w: usize,
    quad: &[(f32, f32); 4],
    dark: bool,
    ring_median: f64,
    span: f64,
) -> (f64, [bool; 4], [bool; 4]) {
    let ordered = order_quad(quad);
    let min_edge = (0..4)
        .map(|i| {
            norm2(
                ordered[(i + 1) % 4].0 - ordered[i].0,
                ordered[(i + 1) % 4].1 - ordered[i].1,
            )
        })
        .fold(f32::INFINITY, f32::min);
    let band = py_round(0.012 * 1.0f64.max(min_edge as f64)).clamp(2, 24) as f64;
    let contrast_gate = 0.02f64.max(0.04 * span);
    let contrast_scale = 0.06f64.max(0.20 * span);
    let ring_tolerance = 0.06f64.max(0.16 * span);
    let mut supported = [false; 4];
    let mut scores = Vec::new();
    for i in 0..4 {
        let (start, end) = (ordered[i], ordered[(i + 1) % 4]);
        let vector = (end.0 - start.0, end.1 - start.1);
        let length = norm2(vector.0, vector.1) as f64;
        if length < 4.0 {
            continue;
        }
        // Skip the corners, where interpolation and sprocket holes mix the two populations.
        // The ordered points run clockwise in image coordinates, so (-dy, dx) points inward.
        let k = 0.08f32;
        let inner_start = (start.0 + k * vector.0, start.1 + k * vector.1);
        let inner_end = (end.0 - k * vector.0, end.1 - k * vector.1);
        let l32 = length as f32;
        let inward = (-vector.1 / l32, vector.0 / l32);
        let b32 = band as f32;
        let off = (inward.0 * b32, inward.1 * b32);
        let inside_poly = [
            inner_start,
            inner_end,
            (inner_end.0 + off.0, inner_end.1 + off.1),
            (inner_start.0 + off.0, inner_start.1 + off.1),
        ];
        let outside_poly = [
            inner_start,
            inner_end,
            (inner_end.0 - off.0, inner_end.1 - off.1),
            (inner_start.0 - off.0, inner_start.1 - off.1),
        ];
        let mut inside = polygon_values(lum, h, w, &inside_poly);
        let mut outside = polygon_values(lum, h, w, &outside_poly);
        let expected = 8.0f64.max(0.20 * 0.84 * length * band);
        if (inside.len() as f64) < expected || (outside.len() as f64) < expected {
            continue;
        }
        let (im, om) = (
            np_median(&mut inside) as f64,
            np_median(&mut outside) as f64,
        );
        let contrast = if dark { om - im } else { im - om };
        let ring_delta = (om - ring_median).abs();
        if contrast < contrast_gate
            || ring_delta > ring_tolerance
            || contrast.is_nan()
            || ring_delta.is_nan()
        {
            continue;
        }
        let cs = clip(contrast / contrast_scale, 0.0, 1.0);
        let rs = clip(1.0 - ring_delta / ring_tolerance, 0.0, 1.0);
        supported[i] = true;
        scores.push(0.75 * cs + 0.25 * rs);
    }
    let mut corners = [false; 4];
    for (c, (_, sides)) in CORNER_SIDES.iter().enumerate() {
        corners[c] = supported[sides[0]] && supported[sides[1]];
    }
    let boundary = scores.iter().fold(0.0f64, |s, &v| s + v) / 4.0;
    (boundary, supported, corners)
}

/// `_correction_angle_from_quad(quad, aspect_ratio)`: the residual deskew in degrees.
fn correction_angle_from_quad(quad: &[(f32, f32); 4], aspect: f64) -> f64 {
    // A near-square box has no stable orientation axis.
    if aspect < 1.12 {
        return 0.0;
    }
    let edges: Vec<(f32, f32)> = (0..4)
        .map(|i| {
            (
                quad[(i + 1) % 4].0 - quad[i].0,
                quad[(i + 1) % 4].1 - quad[i].1,
            )
        })
        .collect();
    let norms: Vec<f32> = edges.iter().map(|e| norm2(e.0, e.1)).collect();
    let k = norms
        .iter()
        .enumerate()
        .fold(0, |b, (i, &n)| if n > norms[b] { i } else { b });
    // Python's math.degrees multiplies by 180 / pi.
    let mut angle =
        libm::atan2(edges[k].1 as f64, edges[k].0 as f64) * (180.0 / std::f64::consts::PI);
    while angle >= 90.0 {
        angle -= 180.0;
    }
    while angle < -90.0 {
        angle += 180.0;
    }
    if angle > 45.0 {
        angle -= 90.0;
    } else if angle < -45.0 {
        angle += 90.0;
    }
    // OpenCV's positive fine rotation turns a horizontal edge to a negative image angle, so the
    // measured residual is already the additive fix.
    clip(angle, -FINE_ROTATION_LIMIT, FINE_ROTATION_LIMIT)
}

struct Scene<'a> {
    lum: &'a [f32],
    h: usize,
    w: usize,
    ring_median: f64,
    span: f64,
}

/// `_candidate_from_contour`.
fn candidate_from_contour(
    contour: &[(i32, i32)],
    s: &Scene,
    source: &'static str,
    threshold_index: Option<usize>,
    polarity: Option<&'static str>,
) -> Option<Candidate> {
    let (h, w) = (s.h, s.w);
    let image_area = (h * w) as f64;
    let rect = min_area_rect(contour);
    let (rw, rh) = (rect.size.0 as f64, rect.size.1 as f64);
    let rect_area = rw * rh;
    if rect_area <= 0.0 {
        return None;
    }
    let area = contour_area(contour);
    let area_ratio = rect_area / image_area.max(1.0);
    let (short, long) = (rw.min(rh), rw.max(rh));
    let aspect = long / short.max(1.0);
    let rectangularity = area / rect_area;
    if !(0.08..=1.02).contains(&area_ratio) {
        return None;
    }
    if short < 8.0f64.max(0.04 * h.min(w) as f64) || aspect > 8.0 || rectangularity < 0.30 {
        return None;
    }
    let quad = box_points(&rect);
    let (x, y, bw, bh) = bounding_rect_f32(&quad);
    let (x, y, bw, bh) = (x as i64, y as i64, bw as i64, bh as i64);
    let roi = (
        y.max(0),
        (y + bh).min(h as i64),
        x.max(0),
        (x + bw).min(w as i64),
    );
    if roi.1 <= roi.0 || roi.3 <= roi.2 {
        return None;
    }
    let polarities: Vec<&'static str> = match polarity {
        Some(p) => vec![p],
        None => vec!["dark", "bright"],
    };
    let mut best: Option<Candidate> = None;
    for p in polarities {
        let (boundary, sides, corners) =
            oriented_boundary_evidence(s.lum, h, w, &quad, p == "dark", s.ring_median, s.span);
        let n_sides = sides.iter().filter(|&&b| b).count();
        if n_sides < 2 {
            continue;
        }
        let rs = clip((rectangularity - 0.30) / 0.65, 0.0, 1.0);
        let geometry = 0.70 * rs + 0.30 * (n_sides as f64 / 4.0);
        let c = Candidate {
            roi,
            correction_angle: correction_angle_from_quad(&quad, aspect),
            source,
            threshold_index,
            polarity: p,
            boundary_score: boundary,
            geometry_score: clip(geometry, 0.0, 1.0),
            sides,
            corners,
        };
        let better = match &best {
            None => true,
            Some(b) => (c.boundary_score, c.geometry_score) > (b.boundary_score, b.geometry_score),
        };
        if better {
            best = Some(c);
        }
    }
    best
}

/// `_candidates_from_mask`: the twelve largest contours (stable by area, largest first) scored.
fn candidates_from_mask(
    mask: &[u8],
    s: &Scene,
    source: &'static str,
    threshold_index: Option<usize>,
    polarity: Option<&'static str>,
) -> Vec<Candidate> {
    let mut contours = find_external_contours(mask, s.h, s.w);
    let areas: Vec<f64> = contours.iter().map(|c| contour_area(c)).collect();
    let mut order: Vec<usize> = (0..contours.len()).collect();
    order.sort_by(|&a, &b| areas[b].total_cmp(&areas[a]));
    let picked: Vec<Vec<(i32, i32)>> = order
        .iter()
        .take(12)
        .map(|&i| std::mem::take(&mut contours[i]))
        .collect();
    picked
        .iter()
        .filter_map(|c| candidate_from_contour(c, s, source, threshold_index, polarity))
        .collect()
}

fn roi_iou(a: &Roi, b: &Roi) -> f64 {
    let (y1, y2, x1, x2) = (a.0.max(b.0), a.1.min(b.1), a.2.max(b.2), a.3.min(b.3));
    let inter = (y2 - y1).max(0) * (x2 - x1).max(0);
    let area = |r: &Roi| (r.1 - r.0).max(0) * (r.3 - r.2).max(0);
    inter as f64 / ((area(a) + area(b) - inter) as f64).max(1.0)
}

fn normalized_edge_distance(a: &Roi, b: &Roi, h: usize, w: usize) -> f64 {
    let scales = [
        h.max(1) as f64,
        h.max(1) as f64,
        w.max(1) as f64,
        w.max(1) as f64,
    ];
    let (av, bv) = ([a.0, a.1, a.2, a.3], [b.0, b.1, b.2, b.3]);
    (0..4)
        .map(|i| (av[i] - bv[i]).abs() as f64 / scales[i])
        .fold(f64::NEG_INFINITY, f64::max)
}

/// `_cluster_film_candidates`: union-find on overlapping boxes, groups in first-seen order.
fn cluster_candidates(c: &[Candidate], h: usize, w: usize) -> Vec<Vec<usize>> {
    let n = c.len();
    let mut parents: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for a in 0..n {
        for b in a + 1..n {
            if roi_iou(&c[a].roi, &c[b].roi) < 0.78 {
                continue;
            }
            if normalized_edge_distance(&c[a].roi, &c[b].roi, h, w) <= 0.035 {
                let (ra, rb) = (find(&mut parents, a), find(&mut parents, b));
                if ra != rb {
                    parents[rb] = ra;
                }
            }
        }
    }
    let mut roots: Vec<usize> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let r = find(&mut parents, i);
        match roots.iter().position(|&x| x == r) {
            Some(g) => groups[g].push(i),
            None => {
                roots.push(r);
                groups.push(vec![i]);
            }
        }
    }
    groups
}

/// The geometry module's `_weighted_median`: values sorted (ties by position), the first whose
/// cumulative weight reaches half the total.
fn weighted_median(values: &[f64], weights: &[f64]) -> f64 {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let ow: Vec<f64> = order.iter().map(|&i| weights[i]).collect();
    let total = if ow.len() < 8 {
        ow.iter().fold(-0.0f64, |s, &v| s + v)
    } else {
        pairwise_sum_f64(&ow)
    };
    let cutoff = 0.5 * total;
    let mut cum = 0.0f64;
    let mut index = ow.len();
    for (i, &v) in ow.iter().enumerate() {
        cum += v;
        if cum >= cutoff {
            index = i;
            break;
        }
    }
    values[order[index.min(order.len() - 1)]]
}

/// NumPy's float64 pairwise sum of a contiguous run.
fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(-0.0, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r = [0f64; 8];
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
    pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
}

#[derive(Clone)]
pub struct Consensus {
    pub roi: Roi,
    pub correction_angle: f64,
    pub confidence: f64,
    pub sides: Vec<&'static str>,
    pub corners: Vec<&'static str>,
    pub sources: Vec<&'static str>,
    pub geometry_score: f64,
}

/// `_select_consensus_cluster`.
pub fn select_consensus(cands: &[Candidate], h: usize, w: usize) -> Option<Consensus> {
    let mut accepted: Vec<Consensus> = Vec::new();
    for raw in cluster_candidates(cands, h, w) {
        // _dedupe_cluster: the best candidate per trial, in first-seen trial order.
        let mut keys: Vec<(&str, Option<usize>, &str)> = Vec::new();
        let mut cluster: Vec<&Candidate> = Vec::new();
        for &i in &raw {
            let c = &cands[i];
            let key = (c.source, c.threshold_index, c.polarity);
            match keys.iter().position(|k| *k == key) {
                Some(k) => {
                    if (c.boundary_score, c.geometry_score)
                        > (cluster[k].boundary_score, cluster[k].geometry_score)
                    {
                        cluster[k] = c;
                    }
                }
                None => {
                    keys.push(key);
                    cluster.push(c);
                }
            }
        }
        let mut adaptive: Vec<(&str, usize)> = Vec::new();
        let mut structural: Vec<&str> = Vec::new();
        for c in &cluster {
            if c.source.starts_with("adaptive-") {
                if let Some(t) = c.threshold_index
                    && !adaptive.contains(&(c.polarity, t))
                {
                    adaptive.push((c.polarity, t));
                }
            } else if !structural.contains(&c.source) {
                structural.push(c.source);
            }
        }
        if adaptive.len() < 3 && !(adaptive.len() >= 2 && structural.len() >= 2) {
            continue;
        }
        let weights: Vec<f64> = cluster
            .iter()
            .map(|c| 0.05f64.max(c.boundary_score * c.geometry_score))
            .collect();
        let edge = |e: usize, c: &Candidate| [c.roi.0, c.roi.1, c.roi.2, c.roi.3][e] as f64;
        let roi_values: Vec<Vec<f64>> = (0..4)
            .map(|e| cluster.iter().map(|c| edge(e, c)).collect())
            .collect();
        let r: Vec<i64> = roi_values
            .iter()
            .map(|v| py_round(weighted_median(v, &weights)))
            .collect();
        let roi = (r[0], r[1], r[2], r[3]);
        if roi.1 <= roi.0 || roi.3 <= roi.2 {
            continue;
        }
        let scales = [
            h.max(1) as f64,
            h.max(1) as f64,
            w.max(1) as f64,
            w.max(1) as f64,
        ];
        let mads: Vec<f64> = (0..4)
            .map(|e| {
                let center = weighted_median(&roi_values[e], &weights);
                let dev: Vec<f64> = roi_values[e]
                    .iter()
                    .map(|v| (v - center).abs() / scales[e])
                    .collect();
                weighted_median(&dev, &weights)
            })
            .collect();
        let dispersion = mads.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let stability = clip(1.0 - dispersion / 0.03, 0.0, 1.0);
        let angles: Vec<f64> = cluster.iter().map(|c| c.correction_angle).collect();
        let correction_angle = weighted_median(&angles, &weights);
        let angle_dev: Vec<f64> = angles
            .iter()
            .map(|a| (a - correction_angle).abs())
            .collect();
        let angle_stability = clip(1.0 - weighted_median(&angle_dev, &weights) / 3.0, 0.0, 1.0);
        let geo: Vec<f64> = cluster.iter().map(|c| c.geometry_score).collect();
        let geometry_score = clip(
            0.55 * weighted_median(&geo, &weights) + 0.25 * stability + 0.20 * angle_stability,
            0.0,
            1.0,
        );
        let bnd: Vec<f64> = cluster.iter().map(|c| c.boundary_score).collect();
        let boundary = weighted_median(&bnd, &weights);
        let support = 1.0f64.min(adaptive.len() as f64 / ADAPTIVE_THRESHOLD_COUNT);
        let corroboration = 1.0f64.min(structural.len() as f64 / 3.0);
        let confidence = clip(
            0.40 * support
                + 0.25 * stability
                + 0.20 * boundary
                + 0.10 * geometry_score
                + 0.05 * corroboration,
            0.0,
            1.0,
        );
        if confidence < 0.55 {
            continue;
        }
        // _supported_by_majority: values present in at least half the cluster.
        let needed = ((0.5 * cluster.len() as f64).ceil() as usize).max(1);
        let sides: Vec<&'static str> = (0..4)
            .filter(|&i| cluster.iter().filter(|c| c.sides[i]).count() >= needed)
            .map(|i| SIDE_NAMES[i])
            .collect();
        let corners: Vec<&'static str> = (0..4)
            .filter(|&i| cluster.iter().filter(|c| c.corners[i]).count() >= needed)
            .map(|i| CORNER_SIDES[i].0)
            .collect();
        let mut sources: Vec<&'static str> = EVIDENCE_SOURCE_ORDER
            .iter()
            .copied()
            .filter(|s| cluster.iter().any(|c| c.source == *s))
            .collect();
        let mut rest: Vec<&'static str> = cluster
            .iter()
            .map(|c| c.source)
            .filter(|s| !sources.contains(s))
            .collect();
        rest.sort();
        rest.dedup();
        sources.extend(rest);
        accepted.push(Consensus {
            roi,
            correction_angle,
            confidence,
            sides,
            corners,
            sources,
            geometry_score,
        });
    }
    // Confidence first; area breaks a tie between inner and outer clusters toward the film.
    let area = |c: &Consensus| (c.roi.1 - c.roi.0) * (c.roi.3 - c.roi.2);
    let mut best: Option<Consensus> = None;
    for c in accepted {
        let better = match &best {
            None => true,
            Some(b) => (c.confidence, area(&c)) > (b.confidence, area(b)),
        };
        if better {
            best = Some(c);
        }
    }
    best
}

pub fn film_box_covers_enough(h: usize, w: usize, roi: &Roi) -> bool {
    h > 0
        && w > 0
        && ((roi.1 - roi.0) * (roi.3 - roi.2)) as f64 >= MIN_FILM_BOX_COVERAGE * (h * w) as f64
}

/// `_film_surround_is_plausible`: a light bed or a dark holder around the box.
pub fn film_surround_is_plausible(lum: &[f32], h: usize, w: usize, roi: &Roi) -> bool {
    let clampy = |v: i64| v.clamp(0, h as i64) as usize;
    let clampx = |v: i64| v.clamp(0, w as i64) as usize;
    let (y1, y2, x1, x2) = (clampy(roi.0), clampy(roi.1), clampx(roi.2), clampx(roi.3));
    let mut out = Vec::with_capacity(h * w);
    for (y, row) in lum.chunks_exact(w).enumerate() {
        if y >= y1 && y < y2 && x2 > x1 {
            out.extend_from_slice(&row[..x1]);
            out.extend_from_slice(&row[x2..]);
        } else {
            out.extend_from_slice(row);
        }
    }
    if (out.len() as f64) < 0.005 * lum.len() as f64 {
        return true;
    }
    let out_med = np_median(&mut out) as f64;
    let out_high = np_percentiles(&out, &[90.0]).map_or(f64::NAN, |p| p[0] as f64);
    let box_med = np_median(&mut crop(lum, w, (y1, y2.max(y1), x1, x2.max(x1)))) as f64;
    // Near-clipping, not only bright: a thin negative's picture clears 0.85, but the bed is the
    // light source at the top of the histogram, where detection_luma anchors it.
    let bed_like = out_med >= 0.85 && out_high >= 0.98;
    let holder_like = out_med <= 0.30 && out_med <= box_med - 0.15;
    bed_like || holder_like
}

/// `_snap_film_bounds_to_bed_gradient(roi, lum)`.
fn snap_film_bounds(roi: Roi, lum: &[f32], h: usize, w: usize) -> Roi {
    let (y1, y2, x1, x2) = roi;
    let (yu1, yu2, xu1, xu2) = (
        y1.max(0) as usize,
        (y2.max(0) as usize).min(h),
        x1.max(0) as usize,
        (x2.max(0) as usize).min(w),
    );
    let mut col = vec![0f32; w];
    for y in yu1..yu2 {
        for (c, p) in col.iter_mut().enumerate() {
            *p += lum[y * w + c];
        }
    }
    col.iter_mut().for_each(|p| *p /= (yu2 - yu1) as f32);
    let row: Vec<f32> = (0..h)
        .map(|y| pairwise_sum_f32(&lum[y * w + xu1..y * w + xu2]) / (xu2 - xu1) as f32)
        .collect();
    // The 16 px floor: the mask morphology kernels are fixed-size, so their inflation does not
    // shrink with the image the way the 2% window does.
    let snap = Snap {
        window_out: 0.02,
        window_in: 0.02,
        min_dominance: 3.0,
        min_window_px: 16,
    };
    let nx1 = snap_edge_to_gradient(&col, x1, true, snap);
    let nx2 = snap_edge_to_gradient(&col, x2, false, snap);
    let ny1 = snap_edge_to_gradient(&row, y1, true, snap);
    let ny2 = snap_edge_to_gradient(&row, y2, false, snap);
    if ny2 - ny1 <= 0 || nx2 - nx1 <= 0 {
        return roi;
    }
    // Keep the bed-to-film rows inside the box: the refinement after this needs them.
    (
        (ny1 - 2).max(0),
        (ny2 + 2).min(h as i64),
        (nx1 - 2).max(0),
        (nx2 + 2).min(w as i64),
    )
}

/// The structural masks of an HxW or HxWx3 image, in trial order.
fn structural_masks(img: &[f32], h: usize, w: usize, cn: usize) -> Vec<(&'static str, Vec<u8>)> {
    let mut preview = vec![0u8; h * w * 3];
    crate::analysis::normalize_to_uint8(&rgb(img, cn), &mut preview);
    let mut gray_img = vec![0u8; h * w];
    gray::bgr_to_gray(&preview, &mut gray_img);
    ["inverse-threshold", "blackhat", "edges"]
        .par_iter()
        .map(|&source| {
            let mask = match source {
                "inverse-threshold" => mask_from_inverse_threshold(&gray_img, h, w),
                "blackhat" => mask_from_blackhat(&gray_img, h, w),
                _ => mask_from_edges(&gray_img, h, w),
            };
            (source, mask)
        })
        .collect()
}

/// One adaptive threshold trial.
struct Trial {
    source: &'static str,
    index: usize,
    polarity: &'static str,
    threshold: f64,
}

#[derive(Clone)]
pub struct Detection {
    pub consensus: Option<Consensus>,
    pub profile: Vec<f32>,
    pub contrast: f64,
}

/// `detect_film_bounds_with_confidence(img)` for an HxW or HxWx3 float32 image.
pub fn detect_film_bounds(img: &[f32], h: usize, w: usize, cn: usize) -> Detection {
    static LAST: crate::memo::Memo<Detection, 8> = crate::memo::Memo::new();
    let key = crate::memo::key(img, &[h, w, cn]);
    LAST.get_or(key, || detect(img, h, w, cn, key))
}

fn detect(img: &[f32], h: usize, w: usize, cn: usize, key: u128) -> Detection {
    let empty = |profile: Vec<f32>, contrast: f64| Detection {
        consensus: None,
        profile,
        contrast,
    };
    if h < 2 || w < 2 {
        return empty(Vec::new(), 0.0);
    }
    let Some(lum) = detection_plane(img, h, w, cn, key) else {
        return empty(Vec::new(), 0.0);
    };
    let lum = lum.as_slice();
    let threshold_inputs = || {
        let lohi = np_percentiles_f64(lum, &[2.0, 98.0]);
        let span = lohi[1] - lohi[0];
        if span < 0.04 || span.is_nan() {
            return None;
        }
        let ring_median = np_median(&mut outer_ring_values(lum, h, w)) as f64;
        let polarities = threshold_polarities(lum, h, w, span, ring_median);
        let blur = odd_kernel_size(0.004 * h.min(w) as f64, 3, 9) as usize;
        let mut blurred = vec![0f32; h * w];
        let _ = crate::filter::gaussian_blur(
            lum,
            &mut blurred,
            (h, w, 1),
            (blur, blur),
            (0.0, 0.0),
            Border::Reflect101,
        );
        Some((lohi, span, ring_median, polarities, blurred))
    };
    // The structural masks need none of the threshold inputs, so they build alongside them.
    let (((profile, contrast), inputs), masks) = rayon::join(
        || rayon::join(|| vertical_edge_profile(lum, h, w), threshold_inputs),
        || structural_masks(img, h, w, cn),
    );
    let Some((lohi, span, ring_median, polarities, blurred)) = inputs else {
        return empty(profile, contrast);
    };
    let scene = Scene {
        lum,
        h,
        w,
        ring_median,
        span,
    };

    // Trials in Python's order: each polarity's thresholds, then the three structural masks.
    let mut trials: Vec<Trial> = Vec::new();
    for &p in &polarities {
        // Biased toward the surround, so the outer film edge of a three-tier negative gets at
        // least as many trials as the inner picture edge.
        let fractions: [f64; 7] = if p == "dark" {
            [0.35, 0.50, 0.62, 0.72, 0.80, 0.87, 0.93]
        } else {
            [0.07, 0.13, 0.20, 0.28, 0.38, 0.50, 0.65]
        };
        let (lo, hi) = (lohi[0], lohi[1]);
        if hi - lo < 0.04 {
            continue;
        }
        let source = if p == "dark" {
            "adaptive-dark"
        } else {
            "adaptive-bright"
        };
        for (i, f) in fractions.iter().enumerate() {
            trials.push(Trial {
                source,
                index: i,
                polarity: p,
                threshold: lo + f * (hi - lo),
            });
        }
    }
    let (mut results, structural_results): (Vec<Vec<Candidate>>, Vec<Vec<Candidate>>) = rayon::join(
        || {
            trials
                .par_iter()
                .map(|t| {
                    let mask =
                        adaptive_threshold_mask(&blurred, h, w, t.threshold, t.polarity == "dark");
                    candidates_from_mask(&mask, &scene, t.source, Some(t.index), Some(t.polarity))
                })
                .collect()
        },
        || {
            masks
                .par_iter()
                .map(|(source, mask)| candidates_from_mask(mask, &scene, source, None, None))
                .collect()
        },
    );
    results.extend(structural_results);
    let candidates: Vec<Candidate> = results.into_iter().flatten().collect();

    let Some(consensus) = select_consensus(&candidates, h, w) else {
        return empty(profile, contrast);
    };
    if !film_box_covers_enough(h, w, &consensus.roi)
        || !film_surround_is_plausible(lum, h, w, &consensus.roi)
    {
        return empty(profile, contrast);
    }
    let roi = snap_film_bounds(consensus.roi, lum, h, w);
    Detection {
        consensus: Some(Consensus { roi, ..consensus }),
        profile,
        contrast,
    }
}
