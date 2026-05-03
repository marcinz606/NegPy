struct FinishUniforms {
    vignette_strength: f32,
    vignette_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
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

    // Center and max distance
    let center = vec2<f32>(f32(dims.x) * 0.5, f32(dims.y) * 0.5);
    let max_dist = length(center);
    let px = vec2<f32>(f32(coords.x), f32(coords.y));
    let d = length(px - center) / max_dist;

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
