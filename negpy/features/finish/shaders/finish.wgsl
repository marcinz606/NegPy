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
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: FinishUniforms;
// (4, 512) edge jitter profiles from carrier_profiles() — top, bottom, left, right.
@group(0) @binding(3) var<storage, read> carrier_prof: array<f32>;

const CARRIER_SAMPLES: i32 = 512;
const CARRIER_JITTER: f32 = 0.24;
// Penumbra fraction of the rebate width; mirrors CARRIER_SOFT in logic.py.
const CARRIER_SOFT: f32 = 0.35;

fn carrier_edge_width(edge: i32, s: f32) -> f32 {
    let idx = min(i32(s * f32(CARRIER_SAMPLES)), CARRIER_SAMPLES - 1);
    let jitter = carrier_prof[edge * CARRIER_SAMPLES + idx];
    return params.carrier_width_px * (1.0 + CARRIER_JITTER * params.carrier_rough * jitter);
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
        let a_t = clamp((px.y - carrier_edge_width(0, sx)) / soft + 0.5, 0.0, 1.0);
        let a_b = clamp(((full.y - 1.0 - px.y) - carrier_edge_width(1, sx)) / soft + 0.5, 0.0, 1.0);
        let a_l = clamp((px.x - carrier_edge_width(2, sy)) / soft + 0.5, 0.0, 1.0);
        let a_r = clamp(((full.x - 1.0 - px.x) - carrier_edge_width(3, sy)) / soft + 0.5, 0.0, 1.0);
        color = color * (a_t * a_b * a_l * a_r);
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
