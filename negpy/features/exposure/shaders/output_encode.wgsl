// Output transform: scene-linear -> display-encoded (ProPhoto ROMM: gamma 1.8 + linear toe).
// Final GPU step; mirrors the CPU working_oetf_encode.

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;

fn oetf_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(pow(x, vec3<f32>(0.55555556)), x * 16.0, x < vec3<f32>(0.001953125));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(input_tex, coords, 0).rgb;
    textureStore(output_tex, coords, vec4<f32>(oetf_encode(color), 1.0));
}
