struct FinishUniforms {
    vignette_strength: f32,
    vignette_size: f32,
    full_crop_w: f32,
    full_crop_h: f32,
    tile_off_x: f32,
    tile_off_y: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: FinishUniforms;

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
    let d = length(px - center) / max(max_dist, 1.0);

    // Remap: size=0 → vignette at edges, size=1 → covers entire image
    let midpoint = 1.0 - params.vignette_size;
    let t = clamp((d - midpoint) / max(1e-6, 1.0 - midpoint), 0.0, 1.0);

    // Smooth cosine falloff
    let factor = 0.5 * (1.0 - cos(t * 3.14159265));

    let strength_abs = abs(params.vignette_strength);
    if (params.vignette_strength < 0.0) {
        color = color * (1.0 - factor * strength_abs);
    } else if (params.vignette_strength > 0.0) {
        color = color + (1.0 - color) * factor * strength_abs;
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
