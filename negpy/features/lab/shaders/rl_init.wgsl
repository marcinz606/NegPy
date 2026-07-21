// Richardson-Lucy init: extract linear luminance Y from the (OETF-encoded) lab
// input and seed the deconvolution state. Channels carry (blur target x, obs y,
// est z); init sets all three to Y. Mirrors apply_rl_sharpening in logic.py.
// No uniform binding — this pass reads nothing from LabUniforms, and the auto
// layout prunes an unused binding (bind-group arity must match).
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;

// Working-space TRC (Adobe RGB 1998: pure 563/256 gamma) — mirrors lab.wgsl.
fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return pow(e, vec3<f32>(2.19921875));
}

// Linear Adobe RGB -> luminance Y (D65, Yn=1) — mirrors LUM_* in logic.py.
fn lum(rgb: vec3<f32>) -> f32 {
    return max(rgb.r, 0.0) * 0.2973769 + max(rgb.g, 0.0) * 0.6273491 + max(rgb.b, 0.0) * 0.0752741;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));

    let y = lum(oetf_decode(textureLoad(input_tex, coords, 0).rgb));
    textureStore(output_tex, coords, vec4<f32>(y, y, y, 0.0));
}
