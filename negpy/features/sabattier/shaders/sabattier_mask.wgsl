// Sabattier, pass 1 of 3: the developed-silver mask, sigmoid((D - R) / edge_width) per
// pixel. Mirrors apply_sabattier. Passes 2 and 3 blur it and fold the tones.
struct SabattierUniforms {
    re: f32,          // re-exposure density R = reexposure * d_max
    fold_width: f32,  // SABATTIER_CONSTANTS["fold_width"]
    edge_width: f32,  // SABATTIER_CONSTANTS["edge_width"]
    strength: f32,
    radius: f32,      // taps each side of the Mackie-line blur, len(gaussian_kernel_1d) // 2
    line: f32,        // 1 with lines, 0 without
    d_max: f32,
    pad: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: SabattierUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let rgb = clamp(textureLoad(input_tex, coords, 0).rgb, vec3<f32>(1e-6), vec3<f32>(1.0));
    let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let dens = -log(clamp(luma, 1e-6, 1.0)) / log(10.0);
    let developed = 1.0 / (1.0 + exp(-clamp((dens - params.re) / params.edge_width, -30.0, 30.0)));
    textureStore(output_tex, coords, vec4<f32>(developed, 0.0, 0.0, 1.0));
}
