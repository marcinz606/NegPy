// Dodge/burn local adjustments. The per-pixel multiplicative factor map is
// rasterised on the CPU (cv2 fillPoly + Gaussian feather) and uploaded as a
// texture, so the shader only multiplies and clamps — guaranteeing CPU parity.

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var factor_tex: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));

    let color = textureLoad(input_tex, coords, 0).rgb;
    let factor = textureLoad(factor_tex, coords, 0).r;
    let adjusted = clamp(color * factor, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output_tex, coords, vec4<f32>(adjusted, 1.0));
}
