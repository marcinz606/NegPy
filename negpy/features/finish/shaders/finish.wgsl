struct FinishUniforms {
    vignette_stops: f32,
    vignette_size: f32,
    vignette_roundness: f32,
    full_crop_w: f32,
    full_crop_h: f32,
    tile_off_x: f32,
    tile_off_y: f32,
    carrier_width_px: f32,   // 0 = off
    carrier_rough: f32,
    carrier_flare: f32,      // 0 = off
    carrier_bw: f32,         // 1 = neutral flare (B&W process)
    carrier_corner: f32,     // aperture corner roundness
    paper_r: f32,            // bare-paper colour, scene-linear (matches the mat)
    paper_g: f32,
    paper_b: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: FinishUniforms;
// (8, CARRIER_SAMPLES) from carrier_profiles(): rows 0-3 gate wobble, rows 4-7 filed edge.
@group(0) @binding(3) var<storage, read> carrier_prof: array<f32>;

// Every CARRIER_* below mirrors logic.py — keep in sync or preview drifts from export.
const CARRIER_SAMPLES: i32 = 2048;
const CARRIER_JITTER: f32 = 0.24;
const CARRIER_INNER_ROUGH: f32 = 0.2;
const CARRIER_OUTER_JITTER: f32 = 0.225;
const CARRIER_CORNER: f32 = 1.4;
const CARRIER_MARGIN: f32 = 0.7;
const CARRIER_SOFT: f32 = 0.22;
const CARRIER_FILED_SOFT: f32 = 0.06;
const CARRIER_FLARE_DEPTH: f32 = 0.35;
const CARRIER_FLARE_SPILL: f32 = 0.7;
const CARRIER_FLARE_GAIN: f32 = 0.55;
const CARRIER_FLARE_BASE: f32 = 0.35;
const CARRIER_FLARE_HUE: f32 = 1.0;
const CARRIER_NOISE_SEED: u32 = 0x51ED270Bu;
const CARRIER_NOISE_CELL: f32 = 1.5;
const CARRIER_NOISE_OCTAVES: i32 = 4;
const CARRIER_NOISE_OUTER: f32 = 0.275;
const CARRIER_NOISE_INNER: f32 = 0.07;

// u32 wrap-around only, so numpy lands on the same lattice values.
fn hash_lattice(ix: u32, iy: u32) -> f32 {
    var h = (ix * 0x27D4EB2Du) ^ (iy * 0x165667B1u) ^ CARRIER_NOISE_SEED;
    h ^= h >> 15u;
    h = h * 0x2C1B3C6Du;
    h ^= h >> 13u;
    h = h * 0x297A2D39u;
    h ^= h >> 16u;
    return f32(h >> 8u) * (2.0 / 16777216.0) - 1.0;
}

fn value_noise(x: f32, y: f32) -> f32 {
    let fx = floor(x);
    let fy = floor(y);
    var ux = x - fx;
    var uy = y - fy;
    ux = ux * ux * (3.0 - 2.0 * ux);
    uy = uy * uy * (3.0 - 2.0 * uy);
    let i0 = bitcast<u32>(i32(fx));
    let j0 = bitcast<u32>(i32(fy));
    let lo = mix(hash_lattice(i0, j0), hash_lattice(i0 + 1u, j0), ux);
    let hi = mix(hash_lattice(i0, j0 + 1u), hash_lattice(i0 + 1u, j0 + 1u), ux);
    return mix(lo, hi, uy);
}

fn carrier_noise(x: f32, y: f32) -> f32 {
    var total = 0.0;
    var amp = 1.0;
    var freq = 1.0;
    var norm = 0.0;
    for (var i = 0; i < CARRIER_NOISE_OCTAVES; i++) {
        total += amp * value_noise(x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return total / norm;
}

fn carrier_prof_at(row: i32, s: f32) -> f32 {
    let idx = min(i32(s * f32(CARRIER_SAMPLES)), CARRIER_SAMPLES - 1);
    return carrier_prof[row * CARRIER_SAMPLES + idx];
}

/// x = filed boundary, y = film-gate boundary, px from the print edge. `end` = distance to
/// the nearer end of this edge. Both boundaries take the same corner arc, so the band
/// keeps its width around a corner; the arc runs from the aperture corner, not the print
/// edge, or most of it is spent inside the paper margin.
fn carrier_bounds(edge: i32, s: f32, end: f32, n2: f32) -> vec2<f32> {
    let margin = params.carrier_width_px * CARRIER_MARGIN;
    let radius = params.carrier_width_px * CARRIER_CORNER * params.carrier_corner;
    var cut = 0.0;
    if (radius > 0.0) {
        let x = clamp(radius - (end - margin), 0.0, radius);
        cut = radius - sqrt(max(radius * radius - x * x, 0.0));
    }
    let jitter = CARRIER_OUTER_JITTER * carrier_prof_at(edge + 4, s) + CARRIER_NOISE_OUTER * n2;
    let outer = margin + params.carrier_width_px * params.carrier_rough * jitter + cut;
    let wobble = CARRIER_JITTER * CARRIER_INNER_ROUGH * carrier_prof_at(edge, s) + CARRIER_NOISE_INNER * n2;
    let inner = margin + params.carrier_width_px * (1.0 + wobble) + cut;
    return vec2<f32>(outer, inner);
}

/// x = flare weight, yzw = weight * tint.
fn carrier_flare(edge: i32, s: f32, d: f32, outer: f32, a_in: f32, n2: f32) -> vec4<f32> {
    let reach = max(1.0, params.carrier_width_px * CARRIER_FLARE_DEPTH);
    let off = d - outer;
    let t = clamp(1.0 - max(off, 0.0) / reach + min(off, 0.0) / (reach * CARRIER_FLARE_SPILL), 0.0, 1.0);
    // Floored |noise|, not a one-sided gate: that left whole edges with no flare.
    let n = CARRIER_FLARE_BASE + (1.0 - CARRIER_FLARE_BASE) * abs(n2);
    let amp = params.carrier_flare * CARRIER_FLARE_GAIN * t * t * n * (1.0 - a_in);
    var tint = vec3<f32>(1.0);
    if (params.carrier_bw == 0.0) {
        let theta = CARRIER_FLARE_HUE * carrier_prof_at(edge, s);
        tint = 0.5 + 0.5 * cos(vec3<f32>(theta) + vec3<f32>(0.0, 2.0943951, 4.1887902));
    }
    return vec4<f32>(amp, amp * tint);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;

    let full = vec2<f32>(params.full_crop_w, params.full_crop_h);
    let center = full * 0.5;
    let max_dist = length(center);
    let px = vec2<f32>(f32(coords.x) + params.tile_off_x, f32(coords.y) + params.tile_off_y);
    let off = abs(px - center);
    let d_radial = length(off) / max(max_dist, 1.0);
    let d_rect = max(off.x / max(center.x, 1.0), off.y / max(center.y, 1.0));
    let d = mix(d_radial, d_rect, params.vignette_roundness);

    // Remap: size=0 → vignette at edges, size=1 → covers entire image
    let midpoint = 1.0 - params.vignette_size;
    let t = clamp((d - midpoint) / max(1e-6, 1.0 - midpoint), 0.0, 1.0);

    // Smooth cosine falloff
    let factor = 0.5 * (1.0 - cos(t * 3.14159265));

    if (params.vignette_stops != 0.0) {
        color = color * exp2(-params.vignette_stops * factor);
    }

    // Filed-carrier rebate: multiply toward black inside the jittered frame,
    // mirroring apply_carrier() in logic.py.
    if (params.carrier_width_px > 0.0) {
        let soft = max(1.0, params.carrier_width_px * CARRIER_SOFT);
        let sx = (px.x + 0.5) / full.x;
        let sy = (px.y + 0.5) / full.y;
        let d_t = px.y;
        let d_b = full.y - 1.0 - px.y;
        let d_l = px.x;
        let d_r = full.x - 1.0 - px.x;
        let end_x = min(px.x, full.x - 1.0 - px.x);
        let end_y = min(px.y, full.y - 1.0 - px.y);
        let cell = max(1.0, params.carrier_width_px * CARRIER_NOISE_CELL);
        let n2 = carrier_noise(px.x / cell, px.y / cell);
        let b_t = carrier_bounds(0, sx, end_x, n2);
        let b_b = carrier_bounds(1, sx, end_x, n2);
        let b_l = carrier_bounds(2, sy, end_y, n2);
        let b_r = carrier_bounds(3, sy, end_y, n2);
        let in_t = clamp((d_t - b_t.y) / soft + 0.5, 0.0, 1.0);
        let in_b = clamp((d_b - b_b.y) / soft + 0.5, 0.0, 1.0);
        let in_l = clamp((d_l - b_l.y) / soft + 0.5, 0.0, 1.0);
        let in_r = clamp((d_r - b_r.y) / soft + 0.5, 0.0, 1.0);
        let soft_filed = max(1.0, params.carrier_width_px * CARRIER_FILED_SOFT);
        let out_t = clamp((d_t - b_t.x) / soft_filed + 0.5, 0.0, 1.0);
        let out_b = clamp((d_b - b_b.x) / soft_filed + 0.5, 0.0, 1.0);
        let out_l = clamp((d_l - b_l.x) / soft_filed + 0.5, 0.0, 1.0);
        let out_r = clamp((d_r - b_r.x) / soft_filed + 0.5, 0.0, 1.0);
        // Products here == the CPU's sequential per-edge slab mixes.
        let paper = vec3<f32>(params.paper_r, params.paper_g, params.paper_b);
        let a_out = out_t * out_b * out_l * out_r;
        color = color * (in_t * in_b * in_l * in_r) * a_out + paper * (1.0 - a_out);

        // Edge order must match apply_carrier()'s slabs — the lerp is order-dependent.
        if (params.carrier_flare > 0.0) {
            let f_t = carrier_flare(0, sx, d_t, b_t.x, in_t, n2);
            color = color * (1.0 - f_t.x) + f_t.yzw;
            let f_b = carrier_flare(1, sx, d_b, b_b.x, in_b, n2);
            color = color * (1.0 - f_b.x) + f_b.yzw;
            let f_l = carrier_flare(2, sy, d_l, b_l.x, in_l, n2);
            color = color * (1.0 - f_l.x) + f_l.yzw;
            let f_r = carrier_flare(3, sy, d_r, b_r.x, in_r, n2);
            color = color * (1.0 - f_r.x) + f_r.yzw;
        }
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
